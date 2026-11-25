"""CPU tests of inkling_cert's DFLASH arming (fake worker classes)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from inkling_cert import dflash, patches  # noqa: E402


class FakeCert:
    def __init__(self):
        self.log = []

    def set_on(self, v):
        self.log.append(1 if v else 0)


class Arming(unittest.TestCase):
    def setUp(self):
        self.saved = dict(patches.STATE)
        patches.STATE.update(mode="1", disabled=None, cert=FakeCert(), cfg={"wide": False, "log_every": 0},
                             patched=[], calls=0)

    def tearDown(self):
        patches.STATE.clear()
        patches.STATE.update(self.saved)

    def build(self, skip):
        seen = []

        class TpModelWorker:
            def forward_batch_generation(self, *a, **k):
                seen.append(patches.STATE["cert"].log[-1] if patches.STATE["cert"].log else None)
                return "out"

        class DFlashWorkerV2:
            def __init__(self):
                self.target_worker = TpModelWorker()

            def _can_skip_verify_logits(self, batch, si):
                return skip

            block_size = 8

            def forward_batch_generation(self, batch, *a, **k):
                self.target_worker.forward_batch_generation(batch=None, forward_batch=None, is_verify=False)
                out = self.target_worker.forward_batch_generation(batch=None, forward_batch=None, is_verify=True)
                self.accepted = self._accept_block(bs=2, next_token_logits=None, target_predict=None)
                return out

            def _accept_block(self, **kw):
                return kw["target_predict"]

        dflash.patch_worker(types.SimpleNamespace(DFlashWorkerV2=DFlashWorkerV2))
        dflash.patch_tp_worker(types.SimpleNamespace(TpModelWorker=TpModelWorker))
        return DFlashWorkerV2, seen

    def batch(self, greedy=True, logprob=False):
        si = types.SimpleNamespace(is_all_greedy=greedy, has_watermark_candidates=False,
                                   sampling_mask_batch_indices=None)
        return types.SimpleNamespace(sampling_info=si, return_logprob=logprob, has_grammar=False)

    def test_greedy_armed_then_reset(self):
        W, seen = self.build(True)
        self.assertEqual(W().forward_batch_generation(self.batch()), "out")
        log = patches.STATE["cert"].log
        self.assertEqual(seen[-1], 1)      # the verify forward saw on = 1
        self.assertEqual(log[-1], 0)       # reset right after
        self.assertIsNone(seen[0])         # the non-verify forward was not armed

    def test_not_armed(self):
        for skip, batch in ((False, self.batch()), (True, self.batch(greedy=False)), (True, self.batch(logprob=True))):
            patches.STATE["cert"] = FakeCert()
            W, seen = self.build(skip)
            W().forward_batch_generation(batch)
            self.assertEqual(seen[-1], 0)

    def test_wide_ids_injected_only_when_armed(self):
        import torch

        for skip, expect in ((True, True), (False, False)):
            patches.STATE["cert"] = FakeCert()
            W, seen = self.build(skip)
            ids = torch.arange(24)
            orig_tp = W().target_worker.forward_batch_generation

            w = W()

            def verify_then_replay(*a, _o=w.target_worker.forward_batch_generation, **k):
                out = _o(*a, **k)
                if k.get("is_verify"):
                    dflash.REPLAY.update(ids=ids, wide=True)  # what the runner hook records on a wide replay
                return out

            w.target_worker.forward_batch_generation = verify_then_replay
            w.forward_batch_generation(self.batch())
            if expect:
                self.assertTrue(torch.equal(w.accepted, ids[:16]))
            else:
                self.assertIsNone(w.accepted)
            del orig_tp

    def test_ambiguous_wide_replay_is_fatal(self):
        W, seen = self.build(True)
        w = W()

        def verify(*a, _o=w.target_worker.forward_batch_generation, **k):
            out = _o(*a, **k)
            if k.get("is_verify"):
                dflash.REPLAY.update(ids=None, wide=True)
            return out

        w.target_worker.forward_batch_generation = verify
        with self.assertRaises(BaseException):
            w.forward_batch_generation(self.batch())

    def test_outside_dflash_untouched(self):
        W, seen = self.build(True)
        W().target_worker.forward_batch_generation(is_verify=True)  # no DFLASH step context
        self.assertEqual(patches.STATE["cert"].log, [])

    def _wide_worker(self, compact):
        patches.STATE["cfg"] = {"wide": True, "log_every": 0}
        rv = types.ModuleType("sglang.srt.speculative.ragged_verify")
        rv.ragged_verify_compact_enabled = lambda: compact
        names = ["sglang", "sglang.srt", "sglang.srt.speculative", "sglang.srt.speculative.ragged_verify"]
        saved = {n: sys.modules.get(n) for n in names}
        for n in names[:-1]:
            sys.modules.setdefault(n, types.ModuleType(n))
        sys.modules[names[-1]] = rv

        class DFlashWorkerV2:
            def forward_batch_generation(self, batch):
                pass

            def _accept_block(self, **kw):
                pass

            def _can_skip_verify_logits(self, b, s):
                return True

        try:
            dflash.patch_worker(types.SimpleNamespace(DFlashWorkerV2=DFlashWorkerV2))
        finally:
            for n, m in saved.items():
                if m is None:
                    sys.modules.pop(n, None)
                else:
                    sys.modules[n] = m
        return DFlashWorkerV2

    def test_wide_refused_under_compact(self):
        # the certified IDs are request-major: the wide head is refused with compact ragged verify
        with self.assertRaises(BaseException):
            self._wide_worker(compact=True)

    def test_wide_armed_static(self):
        W = self._wide_worker(compact=False)
        self.assertTrue(getattr(W.forward_batch_generation, "_inkling_cert", False))

@unittest.skipIf(__import__("importlib").util.find_spec("torch") is None, "needs torch")
class DeadRows(unittest.TestCase):
    def test_gather_selected_by_on(self):
        import torch

        vs = types.ModuleType("inkling_vcap.state")
        vs.STATE = types.SimpleNamespace(src_index=None, stride=8)
        sys.modules["inkling_vcap.state"] = vs
        try:
            dflash.ACTIVE["dflash"] = True
            x = torch.arange(16 * 4, dtype=torch.float32).view(16, 4)
            cert = types.SimpleNamespace(on=torch.ones(1, dtype=torch.int32))
            self.assertIs(dflash.dead_rows_to_anchor(x, cert), x)   # no src_index: untouched (CPU / not CUDA)
        finally:
            dflash.ACTIVE["dflash"] = False
            sys.modules.pop("inkling_vcap.state", None)
        self.assertIs(dflash.dead_rows_to_anchor(x, cert), x)       # not DFLASH: untouched


if __name__ == "__main__":
    unittest.main()
