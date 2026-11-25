"""CPU tests of inkling_padx: switch parsing, runner wiring (live count per replay, capture scope) and the route
wrapper's mask (patched route forced to act by faking capture; route_metadata replaced by a recorder)."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import inkling_padx as px  # noqa: E402


class Switch(unittest.TestCase):
    def test_enabled(self):
        self.assertFalse(px.enabled({}))
        self.assertFalse(px.enabled({"INKLING_PADX": "0"}))
        self.assertTrue(px.enabled({"INKLING_PADX": "1"}))
        with self.assertRaises(RuntimeError):
            px.enabled({"INKLING_PADX": "maybe"})


@unittest.skipIf(torch is None, "needs torch")
class Wiring(unittest.TestCase):
    def setUp(self):
        from inkling_padx import patches

        self.p = patches
        px.STATE.update(runner=None, calls=0, passthrough=0, fills=0)

    def test_runner(self):
        seen = []

        class Runner:
            model_runner = types.SimpleNamespace(device=torch.device("cpu"))

            def _set_dp_real_num_tokens(self, raw_num_token):
                seen.append(("set", raw_num_token))

            def capture(self):
                seen.append(("capture", px.STATE["runner"] is self, int(self._inkling_padx_real[0])))

        mod = types.SimpleNamespace(DecodeCudaGraphRunner=Runner)
        self.assertTrue(self.p.patch_runner(mod))
        self.assertFalse(self.p.patch_runner(mod))
        r = Runner()
        r._set_dp_real_num_tokens(40)  # before capture: no buffer, stock only
        self.assertFalse(hasattr(r, "_inkling_padx_real"))
        r.capture()
        self.assertIsNone(px.STATE["runner"])
        r._set_dp_real_num_tokens(72)
        self.assertEqual(int(r._inkling_padx_real[0]), 72)
        self.assertEqual(seen, [("set", 40), ("capture", True, px.BIG), ("set", 72)])

    def test_route_mask(self):
        calls = []

        def route_metadata(flat, weights, start, local, total):
            calls.append((flat.clone(), local, total))
            return "s2d", "d2s", "g", None, "off"

        def stock(self, ids, weights):
            return "stock"

        mod = types.SimpleNamespace(route_metadata=route_metadata, ROUTE_MAX_IDS=4096,
                                    Inkling2Experts=type("E", (), {"_sm120_route": stock}))
        self.assertTrue(self.p.patch_model(mod))
        route = mod.Inkling2Experts._sm120_route
        host = types.SimpleNamespace(num_global_experts=128)
        ids = torch.arange(320 * 8).view(320, 8) % 128
        w = torch.ones(320, 8)
        self.assertEqual(route(host, ids, w), "stock")  # no capturing runner
        runner = types.SimpleNamespace(_inkling_padx_real=torch.tensor([72], dtype=torch.int32),
                                       capture_bs=list(range(1, 9)) + list(range(10, 33, 2)) + [40, 44, 48],
                                       captured_req_width=8)
        px.STATE["runner"] = runner
        orig = torch.cuda.is_current_stream_capturing
        orig_cuda = torch.Tensor.is_cuda
        try:
            self.assertEqual(route(host, ids, w), "stock")  # not capturing
            torch.cuda.is_current_stream_capturing = lambda: True
            torch.Tensor.is_cuda = property(lambda self: True)
            out = route(host, ids[:80], w[:80])
            self.assertEqual(out, ("s2d", "d2s", "g", "off"))
            flat, local, total = calls[-1]
            m = flat.view(80, 8)
            self.assertTrue(torch.equal(m[:72], ids[:72].to(torch.int32)))
            self.assertTrue(bool((m[72:] == 128).all()))
            self.assertEqual((local, total), (128, 129))
            self.assertEqual(route(types.SimpleNamespace(num_global_experts=64), ids[:80], w[:80]), "stock")
            # graphs that are never replayed with padding keep the stock route (c1..c8 verify, c10's 9 is padded)
            for rows, want in ((8, "stock"), (64, "stock"), (16, "stock"), (80, "mask"), (128, "mask"),
                               (320, "mask"), (24, "stock")):
                n0 = len(calls)
                self.assertEqual(ids[:rows].shape[0], rows)
                out = route(host, ids[:rows], w[:rows])
                self.assertEqual("stock" if out == "stock" else "mask", want, rows)
                self.assertEqual(len(calls) - n0, 0 if want == "stock" else 1)
            self.assertTrue(self.p.can_pad(types.SimpleNamespace(), 80))  # unknown layout: mask (exact)
        finally:
            torch.cuda.is_current_stream_capturing = orig
            torch.Tensor.is_cuda = orig_cuda
            px.STATE["runner"] = None


if __name__ == "__main__":
    unittest.main()
