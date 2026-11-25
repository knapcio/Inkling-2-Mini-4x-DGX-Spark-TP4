"""CPU tests of side-slot validation and the side-slot pressure guard."""

import os
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from inkling_kv12s import IncompatibleImage, MAX_SIDE_SLOTS, side_count
from inkling_kv12s import patch as hooks
from inkling_kv12s.pressure import SideGuard, outermost
from inkling_kv12s.storage import EVENT_LOST, EVENT_WORDS, PAGE


class EnvBool:
    """SGLang's EnvBool parse (sglang/srt/environ.py)."""

    def __init__(self, name, default):
        self.name, self.default = name, default

    def get(self):
        value = os.environ.get(self.name)
        if value is None:
            return self.default
        value = value.lower()
        if value in ("true", "1", "yes", "y"):
            return True
        if value in ("false", "0", "no", "n"):
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class CountTests(unittest.TestCase):
    def test_side_counts_refuse_negative_and_absurd_values(self):
        for raw in ("-1", "x", "1.5", str(MAX_SIDE_SLOTS + 1)):
            with patch.dict(os.environ, {"INKLING_KV12S_FALLBACK_PAGES": raw}):
                with self.assertRaises(IncompatibleImage):
                    side_count("INKLING_KV12S_FALLBACK_PAGES", "64", 0)
        with patch.dict(os.environ, {"INKLING_KV12S_FALLBACK_PAGES": "0"}):
            self.assertEqual(side_count("INKLING_KV12S_FALLBACK_PAGES", "64", 0), 0)
        with patch.dict(os.environ, {"INKLING_KV12S_OVERFLOW_HEADS": "0"}):
            with self.assertRaises(IncompatibleImage):
                hooks._overflow_count("INKLING_KV12S_OVERFLOW_HEADS", "2048")
        with patch.dict(os.environ, {"INKLING_KV12S_OVERFLOW_HEADS": "4"}):
            self.assertEqual(hooks._overflow_count("INKLING_KV12S_OVERFLOW_HEADS", "2048"), 4)

    def test_compact_verify_guard_reads_sglangs_boolean(self):
        name = "SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL"
        args = SimpleNamespace(enable_streaming_session=False, enable_multimodal=False)
        modules = {
            "sglang.srt.runtime_context": SimpleNamespace(get_server_args=lambda: args),
            "sglang.srt.environ": SimpleNamespace(envs=SimpleNamespace(**{name: EnvBool(name, False)})),
        }
        spec = SimpleNamespace(speculative_algorithm="DSPARK")
        with patch.dict(sys.modules, modules), patch.object(hooks, "_FA4_WORKSPACE", False):
            for value, accepted in (("1", True), ("true", True), ("y", True), ("on", False),
                                    ("0", False), ("", False), (None, False)):
                env = {} if value is None else {name: value}
                with patch.dict(os.environ, env):
                    if value is None:
                        os.environ.pop(name, None)
                    if accepted:
                        hooks._check_serving_args(spec)
                    else:
                        with self.assertRaises(IncompatibleImage, msg=value):
                            hooks._check_serving_args(spec)


def lanes(count=2, live=6, fallback=2, overflow=16, heads=2):
    locks = torch.zeros((count, live + fallback + overflow), dtype=torch.int32)
    events = torch.zeros((count, EVENT_WORDS), dtype=torch.int32)
    lane = [SimpleNamespace(live=live, fallback_pages=fallback, heads=heads,
                            overflow_locks=locks[i, live + fallback:]) for i in range(count)]
    return lane, locks, events


class Tree:
    def __init__(self, evictable, on_evict):
        self.evictable, self.on_evict, self.calls = evictable, on_evict, []

    def full_evictable_size(self):
        return self.evictable

    def evict(self, params):
        self.calls.append(params.num_tokens)
        self.evictable = max(0, self.evictable - params.num_tokens)
        self.on_evict()


class Batch:
    def __init__(self, reqs, prefill_only=False):
        self.reqs, self.is_prefill_only = reqs, prefill_only

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs


def request(rid):
    state = {"finished": False}
    req = SimpleNamespace(rid=rid, skip_radix_cache_insert=False, finished=lambda: state["finished"],
                          discarded=None)
    req.clear_requested_finish = lambda: None
    req.request_discard = lambda reason: setattr(req, "discarded", reason)
    return req


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.lanes, self.locks, self.events = lanes()
        # prefill 256 tokens -> 2 + 2 pages; 4 page-heads per page.
        self.guard = SideGuard(self.lanes, self.locks, self.events, prefill_tokens=256, page=PAGE)
        evict_params = SimpleNamespace(EvictParams=lambda num_tokens: SimpleNamespace(num_tokens=num_tokens))
        self.modules = {"sglang.srt.mem_cache.base_prefix_cache": evict_params}

    def test_usage_counts_the_worst_case_sealable_pages(self):
        self.guard.measure()
        self.assertEqual((self.guard.free_pages, self.guard.live_free), (2 + 16 // 4, 6))
        self.locks[1, 6] = 1          # one fallback slot
        self.locks[1, 8:8 + 9] = 1    # nine overflow heads
        self.locks[0, :3] = 1         # three tails
        self.guard.measure()
        self.assertEqual((self.guard.free_pages, self.guard.live_free), (1 + 7 // 4, 3))
        index = self.guard.record()
        self.assertEqual(self.guard.host[index].tolist(), [[3, 0, 0], [0, 1, 9]])
        self.events[:, EVENT_LOST] = 1
        self.assertEqual(self.guard._read(self.guard.record()), 2)

    def test_default_pools_leave_the_schedule_alone(self):
        self.guard.measure()
        running = Batch([object()] * 2)
        self.assertTrue(self.guard.free_pages >= self.guard._need(2))
        tree = Tree(10 * PAGE, lambda: None)
        with patch.dict(sys.modules, self.modules):
            self.guard.relieve(SimpleNamespace(tree_cache=tree, token_to_kv_pool_allocator=None), running)
        self.assertEqual(tree.calls, [])
        self.assertFalse(self.guard.defer_prefill(running))
        self.assertTrue(self.guard.decode_fits(2))

    def test_pressure_evicts_prefix_pages_then_defers_and_retracts(self):
        self.locks[:, 6:] = 1  # every fallback and overflow slot taken
        self.guard.measure()
        self.assertEqual(self.guard.free_pages, 0)

        def release_some():
            self.locks[:, 8:12] = 0  # an evicted prefix page returns four overflow heads

        tree = Tree(PAGE, release_some)
        scheduler = SimpleNamespace(tree_cache=tree, token_to_kv_pool_allocator=SimpleNamespace())
        running = Batch([object()] * 3)
        with patch.dict(sys.modules, self.modules):
            self.guard.relieve(scheduler, running)
        self.assertEqual(tree.calls, [PAGE])
        self.assertEqual(self.guard.free_pages, 1)
        self.assertTrue(self.guard.pressure)
        self.assertTrue(self.guard.defer_prefill(running))
        self.assertFalse(self.guard.defer_prefill(Batch([])))  # nothing decodes: admit
        # Decode retracts until the rest fits: a tail per request, plus a live
        # slot for each seal beyond the one sealable page.
        self.assertTrue(self.guard.decode_fits(3))    # 6 live >= 3 + 2
        self.assertFalse(self.guard.decode_fits(4))   # 6 live < 4 + 3
        self.locks[:, :4] = 1
        self.guard.measure()
        self.assertFalse(self.guard.decode_fits(2))   # 2 live < 2 + 1
        self.assertTrue(self.guard.decode_fits(1))    # never retract the last request
        self.locks[:, :4] = 0                          # retraction freed them: the loop measures again
        self.assertFalse(self.guard.decode_fits(3))
        self.assertTrue(self.guard.decode_fits(3, measure=True))

    def _scheduler(self, chunked=None, inflight=(), holding=None):
        self.retired = []
        retirer = SimpleNamespace(abort=lambda req, reason, finished_reason: self.retired.append(
            (req.rid, reason.message, finished_reason)))
        scheduler = SimpleNamespace(chunked_req=chunked, collect_inflight_reqs=lambda: list(inflight),
                                    request_retirer=retirer, _pending_chunked_abort_req=None,
                                    running_batch=Batch([]))
        if holding is not None:
            scheduler.still_in_batch = lambda req: any(req is h for h in holding)
        return scheduler

    def _modules(self):
        class FinishAbort:
            def __init__(self, message, status_code=None):
                self.message, self.status_code = message, status_code

            def to_json(self):
                return {"type": "abort", "message": self.message, "status_code": self.status_code}

        return {"sglang.srt.managers.req_types": SimpleNamespace(FINISH_ABORT=FinishAbort)}

    def test_a_loss_aborts_every_request_holding_rows_without_a_commit_or_insert(self):
        a, b, c, retracted = request("a"), request("b"), request("c"), request("r")
        # c was launched by the newest forward and is not yet in any scheduler batch list;
        # r was requeued (KV released) but a stale last_batch still names it.
        scheduler = self._scheduler(inflight=(a, retracted), holding=[a, b, c])
        self.guard.record(Batch([a, b]))
        self.guard.record(Batch([c, retracted]))
        self.events[0, EVENT_LOST] = 1
        with patch.dict(sys.modules, self._modules()):
            self.guard.on_result(scheduler, Batch([a, b]), SimpleNamespace(_inkling_kv12s_probe=self.guard.record()))
            self.assertEqual(sorted(rid for rid, *_ in self.retired), ["a", "b", "c"])
            self.assertTrue(all(req.skip_radix_cache_insert for req in (a, b, c)))
            self.assertFalse(retracted.skip_radix_cache_insert)  # it holds no rows: it re-prefills
            # Counted: the next result aborts nothing more; a new loss does again.
            self.guard.on_result(scheduler, Batch([a]), SimpleNamespace(_inkling_kv12s_probe=self.guard.record()))
            self.assertEqual(len(self.retired), 3)
            self.events[1, EVENT_LOST] = 1
            self.guard.on_result(scheduler, Batch([a]), SimpleNamespace(_inkling_kv12s_probe=self.guard.record()))
            self.assertEqual(len(self.retired), 6)

    def test_chunk_with_lost_rows_is_aborted_before_its_stash(self):
        chunk, other = request("chunk"), request("other")
        scheduler = self._scheduler(chunked=chunk, inflight=(other,), holding=[other])
        scheduler.tree_cache = Tree(0, lambda: None)
        self.guard.record(Batch([chunk]))          # the chunk's forward
        self.events[0, EVENT_LOST] = 1
        self.guard.record(Batch([chunk]))          # (the probe reads at its record point)
        with patch.dict(sys.modules, {**self._modules(), **self.modules}):
            self.guard.before_schedule(scheduler, Batch([other]), Batch([chunk]))
        self.assertIs(scheduler._pending_chunked_abort_req, chunk)  # aborted at the top of this pass
        self.assertIn("exhausted", chunk.discarded.message)
        self.assertTrue(chunk.skip_radix_cache_insert)
        self.assertEqual([rid for rid, *_ in self.retired], ["other"])

    def test_no_chunk_means_no_wait_before_scheduling(self):
        scheduler = self._scheduler()
        scheduler.tree_cache = Tree(0, lambda: None)
        self.guard.record(Batch([request("x")]))
        self.events[0, EVENT_LOST] = 1          # seen at the result, not by the scheduling pass
        with patch.dict(sys.modules, self.modules):
            self.guard.before_schedule(scheduler, Batch([]), None)
        self.assertEqual(self.retired, [])

    def test_probe_attaches_to_the_outermost_forward_only(self):
        with patch("inkling_kv12s.pressure.GUARD", [self.guard]):
            inner = SimpleNamespace()
            outer = outermost(lambda: (outermost(lambda: inner), SimpleNamespace())[1])
        self.assertFalse(hasattr(inner, "_inkling_kv12s_probe"))
        self.assertTrue(hasattr(outer, "_inkling_kv12s_probe"))


class RegistrationTests(unittest.TestCase):
    def test_scheduler_hooks_consult_the_guard(self):
        calls = []

        class Scheduler:
            def process_batch_result(self, batch, result):
                calls.append("process")

            def get_next_batch_to_run(self, running_batch, last_batch):
                calls.append("plan")

            def _should_defer_prefill(self, running_batch):
                return False

        class ScheduleBatch:
            reqs = [1, 2, 3]

            def check_decode_mem(self, selected_indices=None):
                return True

        side = SimpleNamespace(on_result=lambda *a: calls.append("probe"),
                               before_schedule=lambda *a: calls.append("relieve"),
                               defer_prefill=lambda batch: True,
                               decode_fits=lambda n: n <= 2)
        hooks._register_scheduler(SimpleNamespace(Scheduler=Scheduler))
        hooks._register_decode_capacity(SimpleNamespace(ScheduleBatch=ScheduleBatch))
        with patch("inkling_kv12s.pressure.GUARD", [side]):
            Scheduler().process_batch_result(None, None)
            Scheduler().get_next_batch_to_run(None, None)
            self.assertTrue(Scheduler()._should_defer_prefill(None))
            self.assertFalse(ScheduleBatch().check_decode_mem())
            self.assertTrue(ScheduleBatch().check_decode_mem([0, 1]))
        self.assertEqual(calls, ["probe", "process", "relieve", "plan"])
        with patch("inkling_kv12s.pressure.GUARD", []):
            self.assertFalse(Scheduler()._should_defer_prefill(None))
            self.assertTrue(ScheduleBatch().check_decode_mem())


if __name__ == "__main__":
    unittest.main()
