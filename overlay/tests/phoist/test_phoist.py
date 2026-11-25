"""CPU tests of inkling_phoist (python3 -m unittest discover -s tests/phoist, inside the image for the torch cases)."""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import inkling_phoist as ph  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class Switch(unittest.TestCase):
    def test_switch(self):
        self.assertFalse(ph.enabled({}))
        self.assertFalse(ph.enabled({"INKLING_PFUSE_HOIST": "0"}))
        with self.assertRaises(RuntimeError):
            ph.enabled({"INKLING_PFUSE_HOIST": "1"})                       # no pfuse
        with self.assertRaises(RuntimeError):
            ph.enabled({"INKLING_PFUSE_HOIST": "1", "INKLING_PFUSE": "resnorm,sconvres"})
        self.assertTrue(ph.enabled({"INKLING_PFUSE_HOIST": "1", "INKLING_PFUSE": "all"}))
        self.assertTrue(ph.enabled({"INKLING_PFUSE_HOIST": "1", "INKLING_PFUSE": "resnorm, loopmix"}))
        with self.assertRaises(RuntimeError):
            ph.enabled({"INKLING_PFUSE_HOIST": "x", "INKLING_PFUSE": "all"})
        self.assertEqual(ph.enabled({"INKLING_PFUSE_HOIST": "1", "INKLING_PFUSE": "all"}), "a")
        self.assertEqual(ph.enabled({"INKLING_PFUSE_HOIST": "exact", "INKLING_PFUSE": "all"}), "exact")

    def test_sources(self):
        d = tempfile.mkdtemp()
        for rel, needles in ph._SOURCES:
            p = os.path.join(d, rel)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            open(p, "w").write("\n".join(needles))
        ph.check_sources(d)
        rel, needles = ph._SOURCES[0]
        open(os.path.join(d, rel), "w").write("\n".join(needles[:-1]))
        with self.assertRaises(RuntimeError):
            ph.check_sources(d)


@unittest.skipIf(torch is None, "needs torch")
class Mapping(unittest.TestCase):
    def setUp(self):
        ph.PACKS.clear()
        self.calls = []
        fake = types.ModuleType("inkling_phoist.phoist")

        def prepare(bank, sq, down_all, tokens, variant=None):
            self.calls.append(("prepare", down_all))
            return {"T": tokens, "id": len(self.calls)}

        def loop_mix_hoisted(prep, layer, side, *a, **kw):
            self.calls.append(("hoisted", prep["id"], layer, side))
            return "hoisted"

        fake.prepare, fake.loop_mix_hoisted = prepare, loop_mix_hoisted
        fake._down_tiles = lambda tokens, device: (tokens <= 4096, 64, 64, 128)
        sys.modules["inkling_phoist.phoist"] = fake
        ph.phoist = fake
        ph.STATE.update(checked={}, cfg_ok={(True, 64, 64, 128): 99}, trusted=True, failed=None, variant="exact")
        self._nocap = None
        if not torch.cuda.is_available():  # CPU-only torch (e.g. macOS) raises instead of answering "not capturing"
            import unittest.mock as um
            self._nocap = um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False)
            self._nocap.start()

    def tearDown(self):
        sys.modules.pop("inkling_phoist.phoist", None)
        if self._nocap is not None:
            self._nocap.stop()

    def model(self, val):
        cw = [(torch.full((4, 64, 4608), float(val), dtype=torch.bfloat16),) for _ in range(12)]
        layers = [types.SimpleNamespace() for _ in range(6)] + [types.SimpleNamespace(controller_weights=c)
                                                              for c in cw] + [types.SimpleNamespace()] * 6
        return types.SimpleNamespace(config=types.SimpleNamespace(loop_core_layers=range(6, 18), loop_config={}),
                                     layers=layers, loop_scaled_queries=torch.zeros(12, 4, ph.D))

    def bank(self, sq, rows=8):
        bl = ph.BankList([torch.zeros(rows, ph.D), torch.zeros(rows, ph.D)])
        bl.sq, bl.prep = sq, None
        return bl

    def call(self, fn, m, bank, layer, side, rows=8):
        r = slice(0, 3) if side == "attn" else slice(3, 4)
        down = m.layers[6 + layer].controller_weights[0][r]
        return fn([torch.zeros(rows, ph.D)], None, torch.zeros(1, ph.D), True, bank, bank.sq[layer, r], down, None,
                  None)

    def test_locate(self):
        sq = torch.zeros(12, 4, ph.D)
        self.assertEqual(ph.locate(sq, sq[0, 0:3], 3), (0, "attn"))
        self.assertEqual(ph.locate(sq, sq[7, 0:3], 3), (7, "attn"))
        self.assertEqual(ph.locate(sq, sq[11, 3:4], 1), (11, "mlp"))
        self.assertIsNone(ph.locate(sq, sq[5, 1:4], 3))
        self.assertIsNone(ph.locate(sq, sq[5, 0:1], 1))
        self.assertIsNone(ph.locate(sq, torch.zeros(3, ph.D), 3))
        self.assertIsNone(ph.locate(None, sq[0, 0:3], 3))

    def setUp_trusted(self):
        ph.STATE.update(checked={}, cfg_ok={(True, 64, 64, 128): 99}, trusted=True, failed=None)

    def test_self_check(self):
        import unittest.mock as um

        ph.STATE.update(checked={}, cfg_ok={}, trusted=False, failed=None, variant="exact")
        m = self.model(1)
        ph.build_pack(m)
        vals = {"pfuse": torch.zeros(2), "hoist": torch.zeros(2)}
        fake = sys.modules["inkling_phoist.phoist"]
        fake.loop_mix_hoisted = lambda *a, **kw: vals["hoist"]
        fn = ph.make_loop_mix(lambda *a, **kw: vals["pfuse"])
        with um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            b = self.bank(m.loop_scaled_queries)
            self.assertIs(self.call(fn, m, b, 0, "attn"), vals["pfuse"])  # capture before any check: pfuse
        with um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            b = self.bank(m.loop_scaled_queries)
            for li in range(12):
                for side in ("attn", "mlp"):
                    self.assertIs(self.call(fn, m, b, li, side), vals["pfuse"])  # checking pass returns pfuse's
            self.assertEqual(ph.STATE["checked"][8], 24)
            self.assertIs(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), vals["hoist"])
            vals["hoist"] = torch.ones(2)  # a mismatch at a new size disables the hoist
            r = self.call(fn, m, self.bank(m.loop_scaled_queries, rows=16), 0, "attn", rows=16)
            self.assertIs(r, vals["pfuse"])
            self.assertEqual(ph.STATE["failed"], (16, 0, "attn"))
            self.assertIs(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), vals["pfuse"])  # disabled

    def test_tolerance(self):
        base = torch.linspace(-3, 3, 192).reshape(3, 64)
        self.assertTrue(ph.tolerance_ok(base * (1 + 1e-6), base)[0])      # FP32-order-sized difference
        self.assertFalse(ph.tolerance_ok(base + 0.5, base)[0])            # gross error
        bad = base.clone()
        bad[1, 3] = float("nan")
        self.assertFalse(ph.tolerance_ok(bad, base)[0])                   # NaN
        inf = base.clone()
        inf[2, 0] = float("inf")
        self.assertFalse(ph.tolerance_ok(base, inf)[0])                   # inf in the reference
        loc = base.clone()
        loc[0] = loc[0] * 1.5                                             # one controller corrupted
        self.assertFalse(ph.tolerance_ok(loc, base)[0])
        self.assertTrue(ph.tolerance_ok(torch.zeros(3, 4), torch.zeros(3, 4))[0])
        self.assertFalse(ph.tolerance_ok(torch.zeros(3, 4), torch.zeros(3, 5))[0])

    def test_option_a_gating(self):
        import unittest.mock as um

        ph.STATE.update(checked={}, cfg_ok={}, trusted=False, failed=None, variant="a")
        m = self.model(1)
        pack = ph.build_pack(m)
        fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
        with um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            self.assertEqual(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), "pfuse")  # no self-test
            pack["startup_ok"] = True
            self.assertEqual(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), "pfuse")  # not armed
            pack["armed"] = {ph.config_key(8, None)}
            self.assertEqual(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), "hoisted")
        with um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            self.assertEqual(self.call(fn, m, self.bank(m.loop_scaled_queries), 1, "mlp"), "hoisted")
        self.assertEqual(ph.STATE["checked"], {})  # no per-call checks in Option A
        ph.STATE["failed"] = (8, 0, "attn")
        self.assertEqual(self.call(fn, m, self.bank(m.loop_scaled_queries), 0, "attn"), "pfuse")
        ph.STATE.update(failed=None, variant="exact")

    def test_prepare_once_per_bank_and_reuse(self):
        fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
        m = self.model(1)
        ph.build_pack(m)
        b = self.bank(m.loop_scaled_queries)
        for li in range(12):
            for side in ("attn", "mlp"):
                self.assertEqual(self.call(fn, m, b, li, side), "hoisted")
        self.assertEqual(sum(c[0] == "prepare" for c in self.calls), 1)
        b2 = self.bank(m.loop_scaled_queries)  # next pass: new bank, new prep
        self.call(fn, m, b2, 0, "attn")
        self.assertEqual(sum(c[0] == "prepare" for c in self.calls), 2)

    def test_two_models_own_packs(self):
        fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
        a, b = self.model(1), self.model(21)
        ph.build_pack(a)
        ph.build_pack(b)
        self.assertEqual(self.call(fn, a, self.bank(a.loop_scaled_queries), 3, "attn"), "hoisted")
        used = [c[1] for c in self.calls if c[0] == "prepare"][-1]
        self.assertEqual(float(used[0, 0, 0, 0]), 1.0)  # model A's pack, not B's
        # pfuse passing another model's down for A's bank -> falls through
        down_b = b.layers[6 + 3].controller_weights[0][0:3]
        bank_a = self.bank(a.loop_scaled_queries)
        r = fn([torch.zeros(8, ph.D)], None, torch.zeros(1, ph.D), True, bank_a, bank_a.sq[3, 0:3], down_b, None, None)
        self.assertEqual(r, "pfuse")

    def test_interleaved_banks(self):
        fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
        m = self.model(1)
        ph.build_pack(m)
        ba, bb = self.bank(m.loop_scaled_queries), self.bank(m.loop_scaled_queries)
        self.call(fn, m, ba, 0, "attn")
        self.call(fn, m, bb, 0, "attn")
        self.call(fn, m, ba, 1, "attn")
        hoisted = [c for c in self.calls if c[0] == "hoisted"]
        self.assertEqual(hoisted[0][1], hoisted[2][1])   # bank A keeps its own prep
        self.assertNotEqual(hoisted[0][1], hoisted[1][1])

    def test_fallthrough(self):
        fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
        m = self.model(1)
        ph.build_pack(m)
        plain = [torch.zeros(8, ph.D), torch.zeros(8, ph.D)]  # not a BankList
        down = m.layers[6].controller_weights[0][0:3]
        self.assertEqual(fn([torch.zeros(8, ph.D)], None, torch.zeros(1, ph.D), True, plain,
                            m.loop_scaled_queries[0, 0:3], down, None, None), "pfuse")
        b = self.bank(m.loop_scaled_queries)
        self.assertEqual(fn([torch.zeros(8, ph.D)], None, torch.zeros(1, ph.D), True, b, b.sq[0, 0:3], down, None,
                            None, norm_weights=[1, 2, 3], fuse_norm=False), "pfuse")
        self.assertFalse(any(c[0] == "prepare" for c in self.calls))

    def test_reload_in_place(self):
        m = self.model(1)
        p1 = ph.build_pack(m)
        buf = p1["down_all"]
        for l_ in m.layers[6:18]:
            l_.controller_weights[0].fill_(5.0)
        p2 = ph.build_pack(m)
        self.assertIs(p2["down_all"], buf)
        self.assertEqual(float(buf[11, 3, 0, 0]), 5.0)

    def test_bank_patch(self):
        class Bank:
            def __init__(self, blocks, scaled_queries):
                self.blocks = [x for x in blocks]
                self.scaled_queries = scaled_queries

        mod = types.SimpleNamespace(LoopMessageBank=Bank)
        ph._patch_bank(mod)
        sq = torch.zeros(12, 4, ph.D)
        b = mod.LoopMessageBank([torch.zeros(1), torch.zeros(1)], sq)
        self.assertIsInstance(b.blocks, ph.BankList)
        self.assertIs(b.blocks.sq, sq)
        self.assertIsNone(b.blocks.prep)
        self.assertEqual(len(b.blocks), 2)
        ph._patch_bank(mod)
        self.assertTrue(getattr(mod.LoopMessageBank.__init__, "_inkling_phoist"))


if __name__ == "__main__":
    unittest.main()
