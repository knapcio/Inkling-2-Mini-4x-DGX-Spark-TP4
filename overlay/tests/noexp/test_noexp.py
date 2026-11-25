"""CPU tests of inkling_noexp: switch / dependency checks, source check, activation condition, routing call shape."""
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import inkling_noexp as ne  # noqa: E402

try:
    import torch
except ImportError:
    torch = None


class Switch(unittest.TestCase):
    def test_off(self):
        self.assertFalse(ne.enabled({}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "0"}))

    def test_needs_vcap_remap(self):
        # without the verify cap or its dead-row remap NOEXP stays inactive (stock routing)
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1"}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x", "INKLING_VERIFY_CAP_REMAP": "0"}))
        self.assertTrue(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x"}))
        with self.assertRaises(RuntimeError):
            ne.enabled({"INKLING_NOEXP": "maybe", "INKLING_VERIFY_CAP": "policy:x"})
        for off in ("false", " OFF ", "none"):
            self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": off}))
        self.assertFalse(ne.enabled({"INKLING_NOEXP": "1", "INKLING_VERIFY_CAP": "policy:x", "INKLING_VERIFY_CAP_REMAP": " off "}))

    def test_sources(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "srt/models/inkling_common/inkling2_moe.py")
        os.makedirs(os.path.dirname(p))
        open(p, "w").write("\n".join(ne._NEEDLES))
        ne.check_sources(d)
        open(p, "w").write("\n".join(ne._NEEDLES[:-1]))
        with self.assertRaises(RuntimeError):
            ne.check_sources(d)


@unittest.skipIf(torch is None, "needs torch (run in the SGLang image)")
class Routing(unittest.TestCase):
    def _vcap(self, **kw):
        st = types.SimpleNamespace(enabled=True, in_verify_moe=True, capping=True,
                                   config=types.SimpleNamespace(remap=True), stride=8,
                                   src_index=torch.tensor([0, 1, 2, 0, 0, 0, 0, 0, 8, 9, 8, 8, 8, 8, 8, 8]))
        for k, v in kw.items():
            setattr(st, k, v)
        mod = types.ModuleType("inkling_vcap.state")
        mod.STATE = st
        sys.modules["inkling_vcap.state"] = mod
        return st

    def tearDown(self):
        sys.modules.pop("inkling_vcap.state", None)

    def test_live_mask(self):
        self._vcap()
        live = ne.live_rows(16)
        self.assertEqual(live.tolist(), [True] * 3 + [False] * 5 + [True] * 2 + [False] * 6)

    def test_inactive_cases(self):
        self.assertIsNone(ne.live_rows(16))                     # no vcap
        self._vcap(capping=False)
        self.assertIsNone(ne.live_rows(16))
        self._vcap(in_verify_moe=False)
        self.assertIsNone(ne.live_rows(16))
        self._vcap()
        self.assertIsNone(ne.live_rows(12))                     # not a multiple of the stride
        self.assertIsNone(ne.live_rows(24))                     # beyond src_index

    def test_route_call(self):
        self._vcap()
        seen = {}

        def route_metadata(flat, w, start, num_local, num_experts):
            seen.update(flat=flat.clone(), start=start, num_local=num_local, num_experts=num_experts)
            return "s2d", "d2s", "g", None, "off", None

        ids = torch.arange(16 * 8).reshape(16, 8) % 128
        out = ne.route_dead_free(route_metadata, ids, torch.ones(16, 8), ne.live_rows(16), 128)
        self.assertEqual(out, ("s2d", "d2s", "g", "off"))
        self.assertEqual((seen["start"], seen["num_local"], seen["num_experts"]), (0, 128, 129))
        flat = seen["flat"].reshape(16, 8)
        self.assertTrue(torch.equal(flat[0], ids[0].int()))
        self.assertTrue((flat[3] == 128).all() and (flat[15] == 128).all())
        self.assertEqual(flat.dtype, torch.int32)

    def test_patch_passthrough(self):
        calls = []
        mod = types.SimpleNamespace(route_metadata=None, ROUTE_MAX_IDS=4096,
                                    Inkling2Experts=type("E", (), {"_sm120_route": lambda self, i, w: calls.append(1)
                                                                   or "stock"}))
        self.assertTrue(ne.patch(mod))
        self.assertFalse(ne.patch(mod))
        e = mod.Inkling2Experts()
        e.num_global_experts = 128
        from unittest import mock

        with mock.patch.object(torch.cuda, "is_current_stream_capturing", lambda: False):  # CPU-only PyTorch
            self.assertEqual(e._sm120_route(torch.zeros(16, 8, dtype=torch.int32), torch.zeros(16, 8)), "stock")
        self.assertEqual(calls, [1])  # no vcap state / not capturing: stock




@unittest.skipIf(torch is None, "needs torch")
class FusedMask(unittest.TestCase):
    """The vcap-remap wrapper (one launch, masked ids as a side output) and its hand-off to _sm120_route."""

    def setUp(self):
        import unittest.mock as um

        self.um = um
        ne.STATE.update(applied=True, masked=None, fused_mask=0, calls=0, mask_ok=True)
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        fake = types.ModuleType("inkling_noexp._mask")

        def remap_mask(w, i, src, ph):
            m = i.shape[0]
            live = src[:m] == torch.arange(m)
            return w[src[:m]], i[src[:m]], torch.where(live[:, None], i[src[:m]], torch.full_like(i, ph))

        fake.remap_mask = remap_mask
        sys.modules["inkling_noexp._mask"] = fake
        model = types.ModuleType(ne.MODEL_MODULE)
        model.ROUTE_MAX_IDS = 4096
        self._saved_model = sys.modules.get(ne.MODEL_MODULE)
        sys.modules[ne.MODEL_MODULE] = model

    def tearDown(self):
        sys.modules.pop("inkling_noexp._mask", None)
        if self._saved_model is None:
            sys.modules.pop(ne.MODEL_MODULE, None)
        else:
            sys.modules[ne.MODEL_MODULE] = self._saved_model
        ne.STATE.update(masked=None, mask_ok=None)

    def obj(self, use_fused=True):
        return types.SimpleNamespace(stride=8, src_index=torch.tensor([0, 1, 1, 1, 1, 1, 1, 1]),
                                     use_fused=use_fused, remap_recorded=0)

    def test_paths(self):
        orig = lambda self, w, i: ("orig", w, i)  # noqa: E731
        remap = ne.make_remap(orig)
        w, i = torch.rand(8, 8), torch.arange(64).reshape(8, 8)
        cap = self.um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True)
        dev = self.um.patch.object(ne, "_on_device", return_value=True)
        with self.um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=False), dev:
            self.assertEqual(remap(self.obj(), w, i)[0], "orig")             # eager: vcap's remap
        with cap, dev:
            self.assertEqual(remap(self.obj(use_fused=False), w, i)[0], "orig")  # vcap fused off: vcap's remap
            ne.STATE["mask_ok"] = None
            self.assertEqual(remap(self.obj(), w, i)[0], "orig")             # mask self-check not passed
            ne.STATE["mask_ok"] = True
            o = self.obj()
            ow, oi = remap(o, w, i)
            self.assertEqual(o.remap_recorded, 1)
            self.assertTrue(torch.equal(oi, i[o.src_index]))                 # the gate sees vcap's tensor
            self.assertIs(ne.STATE["masked"][0], oi)
            masked = ne.STATE["masked"][1]
            self.assertEqual(masked[2:].unique().tolist(), [128])            # rows 2..7 dead (src 1)
            self.assertTrue(torch.equal(masked[:2], i[:2]))
        seen = {}

        def route_metadata(flat, wf, start, nl, ne_):
            seen.update(flat=flat.clone(), args=(start, nl, ne_))
            return ("s2d", "d2s", "g", None, "off")

        mod = types.SimpleNamespace(route_metadata=route_metadata, ROUTE_MAX_IDS=4096)
        fn = ne.make_route(mod, lambda self, i, w: "stock")
        host = types.SimpleNamespace(num_global_experts=128)
        with self.um.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            other = oi.clone()
            ne.STATE["masked"] = (oi, masked)
            self.assertNotEqual(fn(host, other, ow), ("s2d", "d2s", "g", "off"))  # another tensor: not served
            ne.STATE["masked"] = (oi, masked)
            self.assertEqual(fn(host, oi, ow), ("s2d", "d2s", "g", "off"))
            self.assertEqual(seen["args"], (0, 128, 129))
            self.assertTrue(torch.equal(seen["flat"], masked.flatten()))
            self.assertIsNone(ne.STATE["masked"])


if __name__ == "__main__":
    unittest.main()
