"""CPU tests of (B), the draft-tail snapshots of inkling_dftail: slot parsing, the host index (match rules, isolation,
LRU), and the restore / capture path end to end on a stand-in paged ring with the stock locs / record_floors
arithmetic and CPU KV buffers (torch on CPU; no sglang import).

    python3 -m unittest discover -s tests/dftail
"""
import io
import os
import sys
import types
import unittest
from array import array
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/
import inkling_dftail as dt  # noqa: E402
from inkling_dftail import _hook, _sources, bpatch  # noqa: E402
from inkling_dftail import snapshot as sn  # noqa: E402

try:
    import torch
except ImportError:
    torch = None

PAGE = 128
WINDOW_LEFT = 2047          # draft window: 2048 positions
BLOCK = 8
ROWS = (-(-(WINDOW_LEFT + 1 + BLOCK) // PAGE) + 1) * PAGE   # stock paged rows_per_req: 2304
NREQ = 4
HEADS, DIM, LAYERS = 2, 4, 3


def quiet():
    return redirect_stderr(io.StringIO())


def reset_stats():
    import weakref
    sn.STATE.update(inert="", stores=weakref.WeakSet(), charged=1 << 40)
    for k in sn.STATE["stats"]:
        sn.STATE["stats"][k] = 0


class FakeRing:
    """The stock DFlashDraftKVRing arithmetic for a paged ring with DCP 1 (locs, record_floors)."""

    def __init__(self):
        self.rows_per_req, self.dcp_size, self.window_left = ROWS, 1, WINDOW_LEFT
        self.paged, self.sink_rows = True, PAGE
        self.span = ROWS
        self.min_pos = torch.zeros(NREQ, dtype=torch.int32)
        self._floor_generation = torch.full((NREQ,), -1, dtype=torch.int64)
        self.pool = types.SimpleNamespace(size=PAGE + NREQ * ROWS)

    def locs(self, req_pool_indices, positions):
        positions = positions.to(torch.int64)
        positions = positions - self.min_pos[req_pool_indices].to(torch.int64)
        local = self.sink_rows + req_pool_indices.to(torch.int64) * self.rows_per_req + torch.remainder(
            positions, self.rows_per_req)
        return local

    def record_floors(self, reqs, prefix_lens, req_generation):
        slots = torch.tensor([req.kv.req_pool_idx for req in reqs], dtype=torch.int64)
        generation = req_generation[slots]
        fresh = self._floor_generation[slots] != generation
        if not bool(fresh.any()):
            return
        slots = slots[fresh]
        self._floor_generation[slots] = generation[fresh]
        floors = torch.tensor([int(p) for p in prefix_lens], dtype=torch.int32)[fresh]
        self.min_pos[slots] = floors


def fval(pos, layer, kind):
    """The draft KV a position holds (a function of the position only, like a ring row of one prefix)."""
    base = pos.to(torch.float32)[:, None, None]
    return (base * 10 + layer + (0.5 if kind == "v" else 0.0)).expand(-1, HEADS, DIM).clone()


class Rig:
    """A ring, its KV buffers, a snapshot store, and a write helper standing in for the extend's append."""

    def __init__(self, slots=2):
        self.ring = FakeRing()
        rows = PAGE + NREQ * ROWS
        self.k = [torch.zeros(rows, HEADS, DIM) for _ in range(LAYERS)]
        self.v = [torch.zeros(rows, HEADS, DIM) for _ in range(LAYERS)]
        self.store = sn.Store(self.ring, self.k, self.v, slots)
        self.ring._inkling_snap = self.store
        self.gen = torch.zeros(NREQ, dtype=torch.int64)
        self.record = bpatch.make_record_floors(FakeRing.record_floors)

    def write(self, slot, start, end):
        """the extend appending draft KV for positions [start, end) (only the last span positions land)"""
        start = max(start, end - self.ring.span)
        pos = torch.arange(start, end)
        loc = self.ring.locs(torch.full_like(pos, slot), pos)
        for i in range(LAYERS):
            self.k[i][loc] = fval(pos, i, "k")
            self.v[i][loc] = fval(pos, i, "v")

    def read_ok(self, slot, start, end):
        pos = torch.arange(start, end)
        loc = self.ring.locs(torch.full_like(pos, slot), pos)
        return all(torch.equal(self.k[i][loc], fval(pos, i, "k")) and torch.equal(self.v[i][loc], fval(pos, i, "v"))
                   for i in range(LAYERS))

    def extend(self, req, prefix_len, end):
        """one prefill extend [prefix_len, end) of req: record_floors (wrapped), append, then the forward's drain"""
        req.extend_end = end
        self.record(self.ring, [req], [prefix_len], self.gen)
        self.write(req.kv.req_pool_idx, prefix_len, end)
        bpatch.drain(self.store, self.gen)

    def alloc(self, req, slot):
        self.gen[slot] += 1
        req.kv.req_pool_idx = slot


def make_req(tokens, salt=None, extra=None):
    return types.SimpleNamespace(origin_input_ids=array("q", tokens), full_untruncated_fill_ids=array("q", tokens),
                                 cache_salt=salt, extra_key=extra, extend_end=None,
                                 kv=types.SimpleNamespace(req_pool_idx=None))


def toks(n, seed=1):
    return [(i * 7919 + seed * 104729) % 50000 for i in range(n)]


class Slots(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(sn.slots_from_env({}), 0)
        self.assertEqual(sn.slots_from_env({sn.ENV_SLOTS: "0"}), 0)
        self.assertEqual(sn.slots_from_env({sn.ENV_SLOTS: " 8 "}), 8)
        for raw in ("x", "-1", "65", "1.5"):
            with self.assertRaises(RuntimeError, msg=raw):
                sn.slots_from_env({sn.ENV_SLOTS: raw})


class IndexRules(unittest.TestCase):
    def setUp(self):
        self.ix = sn.Index(2, 2048, PAGE)
        self.t = array("q", toks(6000))

    def put(self, end, lo, salt=None, extra=None, tokens=None):
        tokens = self.t if tokens is None else tokens
        return self.ix.place(sn.prefix_digest(tokens, end, salt, extra), end, lo, sn._tail(tokens, end), salt, extra)

    def test_hit_and_lo(self):
        self.put(5000, 2952)
        slot, e, lo = self.ix.find(self.t, 4992)
        self.assertEqual((slot, e.end), (0, 5000))
        self.assertEqual(lo, 3072)               # max(2952, 4992 - 2048) = 2952 -> page-aligned up
        self.assertEqual(lo % PAGE, 0)

    def test_end_below_hit_misses(self):
        self.put(5000, 2952)
        self.assertIsNone(self.ix.find(self.t, 5120))   # the ring would have a gap [5000, 5120)

    def test_end_beyond_request_misses(self):
        self.put(5000, 2952)
        self.assertIsNone(self.ix.find(self.t[:4999], 4096))

    def test_too_little_to_restore(self):
        self.put(5000, 4900)
        self.assertIsNone(self.ix.find(self.t, 4992))   # lo 4992: nothing below the hit

    def test_different_prefix_misses(self):
        self.put(5000, 2952)
        other = array("q", self.t)
        other[10] += 1                                    # same tail, different earlier token
        self.assertIsNone(self.ix.find(other, 4992))

    def test_salt_and_extra_isolate(self):
        self.put(5000, 2952, salt="a")
        self.assertIsNone(self.ix.find(self.t, 4992))
        self.assertIsNone(self.ix.find(self.t, 4992, salt="b"))
        self.assertIsNotNone(self.ix.find(self.t, 4992, salt="a"))
        self.put(5000, 2952, extra="lora1")
        self.assertIsNone(self.ix.find(self.t, 4992, extra="lora2"))

    def test_refresh_lru_and_versions(self):
        self.assertEqual(self.put(5000, 2952), 0)
        self.assertIsNone(self.put(5000, 2952))           # same snapshot: refreshed, no copy
        self.assertEqual(self.put(4000, 1952), 1)
        self.ix.find(self.t, 4992)                        # touches slot 0
        self.assertEqual(self.put(3000, 952), 1)          # LRU is slot 1
        self.assertEqual(self.ix.entries[1].version, 1)
        self.assertEqual(self.ix.entries[0].end, 5000)

    def test_window_after_the_extend(self):
        self.put(5000, 2952)
        slot, e, lo = self.ix.find(self.t, 4992, extend_end=5800)
        self.assertEqual(lo, 3840)                         # max(2952, 5800 - 2048) = 3752 -> 3840
        self.assertIsNone(self.ix.find(self.t, 4992, extend_end=7000))  # the extend alone fills the window

    def test_prefers_most_context(self):
        self.put(5000, 4000)
        self.put(5100, 3000)
        slot, e, lo = self.ix.find(self.t, 4992)
        self.assertEqual(e.end, 5100)


@unittest.skipIf(torch is None, "needs torch")
class RestoreCapture(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def test_turn_two_sees_turn_one_window(self):
        rig = Rig()
        t1 = toks(5000)
        a = make_req(t1)
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 5000)                        # turn 1: cold prefill, prompt complete -> captured
        self.assertEqual(sn.STATE["stats"]["captures"], 1)
        e = rig.store.index.entries[0]
        self.assertEqual((e.end, e.lo), (5000, 5000 - 2048))
        t2 = t1 + toks(300, seed=2)
        b = make_req(t2)
        rig.alloc(b, 1)
        with quiet():
            rig.extend(b, 4992, 5300)                     # turn 2: prefix hit at 4992
        lo = int(rig.ring.min_pos[1])
        self.assertEqual(lo, 3328)                        # floor lowered: max(2952, 5300 - 2048) page-aligned up
        self.assertTrue(rig.read_ok(1, lo, 5300))         # the whole window, as an unbroken session holds it
        self.assertEqual(sn.STATE["stats"]["restores"], 1)
        self.assertEqual(sn.STATE["stats"]["restored_positions"], 4992 - 3328)
        self.assertEqual(sn.STATE["stats"]["captures"], 2)  # turn 2's own prompt end

    def test_chain_across_turns(self):
        rig = Rig(slots=1)
        t = toks(9000)
        ends = [5000, 6000, 7000, 8000]
        prev = 0
        for turn, end in enumerate(ends):
            r = make_req(t[:end])
            rig.alloc(r, turn % NREQ)
            hit = 0 if turn == 0 else (prev // PAGE) * PAGE
            with quiet():
                rig.extend(r, hit, end)
            lo = int(rig.ring.min_pos[turn % NREQ])
            window = max(lo, end - 2048)
            self.assertTrue(rig.read_ok(turn % NREQ, window, end), msg=turn)
            if turn:
                self.assertLess(lo, hit)                   # restored below the hit
                self.assertLessEqual(lo, end - 2048 + PAGE)  # the whole window (page-aligned floor)
            prev = end

    def test_salted_request_does_not_restore(self):
        rig = Rig()
        t = toks(5000)
        a = make_req(t, salt="tenant-a")
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 5000)
        b = make_req(t + toks(100, 3), salt="tenant-b")
        rig.alloc(b, 1)
        with quiet():
            rig.extend(b, 4992, 5100)
        self.assertEqual(int(rig.ring.min_pos[1]), 4992)  # stock floor
        self.assertEqual(sn.STATE["stats"]["restores"], 0)

    def test_later_chunk_never_restores(self):
        rig = Rig()
        t = toks(5000)
        a = make_req(t)
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 5000)
        b = make_req(t + toks(5000, 4))
        rig.alloc(b, 1)
        with quiet():
            rig.extend(b, 4992, 6000)                     # first chunk: restores
            restores = sn.STATE["stats"]["restores"]
            self.assertEqual(restores, 1)
            floor = int(rig.ring.min_pos[1])
            self.assertLess(floor, 4992)
            rig.extend(b, 6000, 10000)                    # later chunk: slot not fresh
        self.assertEqual(sn.STATE["stats"]["restores"], restores)
        self.assertEqual(int(rig.ring.min_pos[1]), floor)

    def test_capture_needs_a_complete_prompt(self):
        rig = Rig()
        a = make_req(toks(9000))
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 8192)                        # chunk 1 of 2
        self.assertEqual(sn.STATE["stats"]["captures"], 0)
        with quiet():
            rig.extend(a, 8192, 9000)                     # completes the prompt
        self.assertEqual(sn.STATE["stats"]["captures"], 1)
        self.assertEqual(rig.store.index.entries[0].lo, 9000 - 2048)

    def test_embed_overrides_not_captured(self):
        rig = Rig()
        a = make_req(toks(5000))
        a.positional_embed_overrides = object()
        rig.alloc(a, 0)
        rig.extend(a, 0, 5000)
        self.assertEqual(sn.STATE["stats"]["captures"], 0)

    def test_short_prompt_not_captured(self):
        rig = Rig()
        a = make_req(toks(sn.MIN_POSITIONS - 1))
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, sn.MIN_POSITIONS - 1)
        self.assertEqual(sn.STATE["stats"]["captures"], 0)

    def test_capture_clamps_to_floor(self):
        """a starved request (stock floor at its hit) captures only the rows its ring holds"""
        rig = Rig()
        t = toks(6000)
        a = make_req(t)
        rig.alloc(a, 0)
        rig.extend(a, 4096, 6000)                         # hit at 4096, no snapshot: floor 4096
        e = rig.store.index.entries[0]
        self.assertEqual((e.lo, e.end), (4096, 6000))

    def test_capture_skipped_on_generation_change(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        a.extend_end = 5000
        rig.record(rig.ring, [a], [0], rig.gen)
        rig.write(0, 0, 5000)
        rig.gen[0] += 1                                   # the slot was reallocated before the drain
        bpatch.drain(rig.store, rig.gen)
        self.assertEqual(sn.STATE["stats"]["captures"], 0)

    def test_unknown_floor_not_captured(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        a.extend_end = 5000
        sn.plan_captures(rig.store, [a], [0], rig.gen)    # no record_floors for this allocation
        self.assertEqual(rig.store.pending, [])

    def test_extra_step_cannot_alias_the_snapshot(self):
        """overlap scheduling: a finished request may run one more step; its rows (<= block + 1 positions past E)
        never land on the rows of [E - window, E)"""
        ring = FakeRing()
        for end in (4096, 5000, 131071):
            for floor in (0, end - 2048, end - 3000):
                ring.min_pos[0] = max(0, floor)
                snap = ring.locs(torch.zeros(2048, dtype=torch.int64), torch.arange(end - 2048, end))
                extra = ring.locs(torch.zeros(BLOCK + 1, dtype=torch.int64), torch.arange(end, end + BLOCK + 1))
                self.assertEqual(len(set(snap.tolist()) & set(extra.tolist())), 0)

    def test_flush_clears(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        rig.extend(a, 0, 5000)
        worker = types.SimpleNamespace(draft_kv_ring=rig.ring)
        bpatch.make_clear(lambda self: None)(worker)
        self.assertEqual(rig.store.index.entries, [None, None])

    def test_failure_goes_inert(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        rig.store.index.find = None                       # any planning error
        b = make_req(toks(5000))
        rig.alloc(b, 1)
        err = io.StringIO()
        with redirect_stderr(err):
            rig.record(rig.ring, [b], [4992], rig.gen)
        self.assertIn("inert", err.getvalue())
        self.assertIsNone(getattr(rig.ring, "_inkling_snap", None))
        self.assertEqual(int(rig.ring.min_pos[1]), 4992)  # stock floor recorded

    def test_restore_locs_match_the_published_floor(self):
        rig = Rig()
        rig.ring.min_pos[2] = 3200
        for start, end in ((3200, 5000), (4000, 4100)):
            a = rig.store._locs(2, start, end, floor=3200)
            b = rig.ring.locs(torch.full((end - start,), 2, dtype=torch.int64), torch.arange(start, end))
            self.assertTrue(torch.equal(a, b))

    def test_restore_failure_keeps_stock_floors(self):
        rig = Rig()
        t = toks(5000)
        a = make_req(t)
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 5000)
        calls = {"n": 0}
        real = rig.store.restore

        def flaky(*args):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("injected after the first request's copies")
            return real(*args)

        rig.store.restore = flaky
        b, c = make_req(t + toks(100, 5)), make_req(t + toks(100, 6))
        rig.alloc(b, 1)
        rig.alloc(c, 2)
        b.extend_end = c.extend_end = 5100
        err = io.StringIO()
        with redirect_stderr(err):
            rig.record(rig.ring, [b, c], [4992, 4992], rig.gen)
        self.assertIn("restore failed", err.getvalue())
        self.assertEqual(int(rig.ring.min_pos[1]), 4992)   # both keep the stock floor
        self.assertEqual(int(rig.ring.min_pos[2]), 4992)
        self.assertEqual(sn.STATE["stats"]["restores"], 0)
        self.assertIsNone(getattr(rig.ring, "_inkling_snap", None))

    def test_mixed_batch_decode_rows_not_captured(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        with quiet():
            rig.extend(a, 0, 5000)
        caps = sn.STATE["stats"]["captures"]
        a.full_untruncated_fill_ids = array("q", list(a.origin_input_ids) + [1, 2, 3])
        a.extend_end = 5003                               # a running decode row in a mixed batch
        rig.record(rig.ring, [a], [5002], rig.gen)
        bpatch.drain(rig.store, rig.gen)
        self.assertEqual(sn.STATE["stats"]["captures"], caps)


@unittest.skipIf(torch is None, "needs torch")
class ForwardWrapper(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def test_pending_cleared_then_drained(self):
        rig = Rig()
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        rig.store.pending.append(("stale",))              # from a forward that raised: dropped at entry

        def orig(self, batch):
            a.extend_end = 5000
            rig.record(rig.ring, [a], [0], rig.gen)
            rig.write(0, 0, 5000)
            return "out"

        worker = types.SimpleNamespace(draft_kv_ring=rig.ring, model_runner=types.SimpleNamespace(
            req_to_token_pool=types.SimpleNamespace(req_generation=rig.gen)))
        self.assertEqual(bpatch.make_forward(orig)(worker, None), "out")
        self.assertEqual(sn.STATE["stats"]["captures"], 1)
        self.assertEqual(rig.store.pending, [])


@unittest.skipIf(torch is None, "needs torch")
class BuildStore(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def worker(self, rows=PAGE + NREQ * ROWS, cls="MHATokenToKVPool", **pool_kw):
        ring = FakeRing()
        pool_cls = type(cls, (), {})
        pool = pool_cls()
        pool.__dict__.update(k_buffer=[torch.zeros(rows, HEADS, DIM) for _ in range(2)],
                             v_buffer=[torch.zeros(rows, HEADS, DIM) for _ in range(2)],
                             start_layer=0, dtype=torch.float32, store_dtype=torch.float32, **pool_kw)
        ring.pool = pool
        layers = [types.SimpleNamespace(self_attn=types.SimpleNamespace(attn=types.SimpleNamespace(layer_id=i)))
                  for i in range(2)]
        return types.SimpleNamespace(draft_kv_ring=ring, draft_model=types.SimpleNamespace(layers=layers))

    def test_builds(self):
        w = self.worker()
        with quiet():
            store = bpatch.build_store(w, 3)
        self.assertIs(w.draft_kv_ring._inkling_snap, store)
        self.assertEqual(tuple(store.k_snap[0].shape), (3, WINDOW_LEFT + 1, HEADS, DIM))
        self.assertEqual(store.nbytes, 2 * 2 * 3 * 2048 * HEADS * DIM * 4)

    def test_refusals(self):
        for w, why in ((types.SimpleNamespace(draft_kv_ring=None), "no DFLASH draft KV ring"),
                       (self.worker(is_quantized_kv_cache=True), "quantized"),
                       (self.worker(cls="MHATokenToKVPoolMXFP8"), "not the token-major NHD pool"),
                       (self.worker(use_hnd=True), "not the token-major NHD pool"),
                       (self.worker(kv_cache_layout="vectorized_5d"), "not the token-major NHD pool"),
                       (self.worker(rows=100), "does not hold the ring")):
            reset_stats()
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertIsNone(bpatch.build_store(w, 3))
            self.assertIn(why, err.getvalue())
        w = self.worker()
        w.draft_kv_ring.dcp_size = 2
        reset_stats()
        with quiet():
            self.assertIsNone(bpatch.build_store(w, 3))
        self.assertIn("DCP", sn.STATE["inert"])


class ChargeAndUpdates(unittest.TestCase):
    def setUp(self):
        reset_stats()

    def test_charge_adds_the_pool_to_the_ring(self):
        mod = types.ModuleType("sglang.srt.speculative.dflash_utils")
        mod.get_dflash_attention_sliding_window_size = lambda cfg: 2047
        saved = sys.modules.get(mod.__name__)
        sys.modules[mod.__name__] = mod
        try:
            cfg = types.SimpleNamespace(dflash_draft_cell_size_per_token=20480, spec_aux_config=types.SimpleNamespace(
                draft_model_config=types.SimpleNamespace(hf_config=None)))
            sn.STATE["charged"] = None
            with quiet():
                self.assertEqual(bpatch.make_ring_bytes(lambda self: 1000, 8)(cfg), 1000 + 8 * 2048 * 20480)
            self.assertEqual(sn.STATE["charged"], 8 * 2048 * 20480)           # 320 MiB
            sn.STATE["charged"] = None
            self.assertEqual(bpatch.make_ring_bytes(lambda self: 0, 8)(cfg), 0)  # no ring (draft worker): no charge
            self.assertIsNone(sn.STATE["charged"])
        finally:
            if saved is None:
                sys.modules.pop(mod.__name__, None)
            else:
                sys.modules[mod.__name__] = saved

    @unittest.skipIf(torch is None, "needs torch")
    def test_uncharged_pool_is_refused(self):
        w = BuildStore.worker(self)
        sn.STATE["charged"] = None
        with quiet():
            self.assertIsNone(bpatch.build_store(w, 3))
        self.assertIn("not charged", sn.STATE["inert"])
        reset_stats()
        sn.STATE["charged"] = 2 * 2048 * HEADS * DIM * 4 * 4  # room for 2 slots only
        with quiet():
            self.assertIsNone(bpatch.build_store(w, 3))
        self.assertIn("not charged", sn.STATE["inert"])

    @unittest.skipIf(torch is None, "needs torch")
    def test_weight_update_clears_every_store(self):
        rig = Rig()
        sn.STATE["stores"].add(rig.store)
        a = make_req(toks(5000))
        rig.alloc(a, 0)
        rig.extend(a, 0, 5000)
        self.assertIsNotNone(rig.store.index.entries[0])
        self.assertEqual(bpatch.make_select_runners(lambda self, sel="all": ["r"])(None, "draft"), ["r"])
        self.assertEqual(rig.store.index.entries, [None, None])

    def test_patch_config_and_updater_once(self):
        cfg = type("KVCacheConfigurator", (), {"_dflash_draft_kv_ring_bytes": lambda self: 0})
        upd = type("SchedulerWeightUpdaterManager", (), {"_select_runners": lambda self, s="all": []})
        self.assertTrue(bpatch.patch_config(types.SimpleNamespace(KVCacheConfigurator=cfg), 8))
        self.assertFalse(bpatch.patch_config(types.SimpleNamespace(KVCacheConfigurator=cfg), 8))
        self.assertTrue(bpatch.patch_updater(types.SimpleNamespace(SchedulerWeightUpdaterManager=upd)))
        self.assertFalse(bpatch.patch_updater(types.SimpleNamespace(SchedulerWeightUpdaterManager=upd)))


class Patching(unittest.TestCase):
    def test_patch_ring_and_worker_once(self):
        ring_cls = type("DFlashDraftKVRing", (), {"record_floors": lambda *a: None, "locs": lambda *a: None})
        worker_cls = type("DFlashWorkerV2", (), {"_init_draft_kv_ring": lambda self: None,
                                                 "forward_batch_generation": lambda self, b: None,
                                                 "clear_cache_pool": lambda self: None})
        self.assertTrue(bpatch.patch_ring(types.SimpleNamespace(DFlashDraftKVRing=ring_cls)))
        self.assertFalse(bpatch.patch_ring(types.SimpleNamespace(DFlashDraftKVRing=ring_cls)))
        self.assertTrue(bpatch.patch_worker(types.SimpleNamespace(DFlashWorkerV2=worker_cls), 8))
        self.assertFalse(bpatch.patch_worker(types.SimpleNamespace(DFlashWorkerV2=worker_cls), 8))
        with self.assertRaises(RuntimeError) as cm:
            bpatch.patch_worker(types.SimpleNamespace(DFlashWorkerV2=type("W", (), {})), 8, dt.OFF_HINT)
        self.assertIn("DFLASH_TAIL_SNAPSHOTS=0", str(cm.exception))

    def test_install_arms_b_alone(self):
        dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
        saved = _sources.check
        calls = []
        _sources.check = lambda root=None, off_hint="", files=(): calls.append(files)
        try:
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(dt.install({sn.ENV_SLOTS: "8"}), 0)
            self.assertEqual(calls, [_sources.PART_B])
            self.assertIn("draft-tail snapshots, 8 slots", err.getvalue())
            for m in bpatch.MODULES:
                self.assertIn(m, _hook.pending())
            self.assertNotIn(dt.MODULE, _hook.pending())
            self.assertEqual(sn.STATE["slots"], 8)
        finally:
            _sources.check = saved
            for m in bpatch.MODULES:
                _hook._callbacks.pop(m, None)
            dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
            sn.STATE["slots"] = 0

    def test_install_both_checks_both_parts(self):
        dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
        saved = _sources.check
        calls = []
        _sources.check = lambda root=None, off_hint="", files=(): calls.append(files)
        try:
            with quiet():
                dt.install({dt.ENV: "2048", sn.ENV_SLOTS: "8"})
            self.assertEqual(calls, [_sources.PART_A + _sources.PART_B])
        finally:
            _sources.check = saved
            for m in (dt.MODULE,) + bpatch.MODULES:
                _hook._callbacks.pop(m, None)
            dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
            sn.STATE["slots"] = 0

    def test_bad_slots_refuse(self):
        dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
        with self.assertRaises(RuntimeError):
            dt.install({sn.ENV_SLOTS: "-3"})
        self.assertFalse(dt.STATE["installed"])


class PinsB(unittest.TestCase):
    def test_pinned_files(self):
        self.assertEqual(_sources.PART_B, (_sources.DRAFT_RING, _sources.DFLASH_WORKER, _sources.KV_CONFIG,
                                           _sources.WEIGHT_UPDATER))
        self.assertIn("KVCacheConfigurator._dflash_draft_kv_ring_bytes", _sources.EXPECTED[_sources.KV_CONFIG])
        self.assertIn("SchedulerWeightUpdaterManager._select_runners", _sources.EXPECTED[_sources.WEIGHT_UPDATER])

    def test_pinned_names(self):
        self.assertEqual(sorted(_sources.EXPECTED[_sources.DRAFT_RING]),
                         ["DFlashDraftKVRing.__init__", "DFlashDraftKVRing.locs", "DFlashDraftKVRing.record_floors"])
        self.assertEqual(sorted(_sources.EXPECTED[_sources.DFLASH_WORKER]),
                         ["DFlashWorkerV2._init_draft_kv_ring", "DFlashWorkerV2.clear_cache_pool",
                          "DFlashWorkerV2.forward_batch_generation"])


if __name__ == "__main__":
    unittest.main()
