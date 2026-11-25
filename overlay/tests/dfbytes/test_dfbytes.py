"""CPU tests of inkling_dfbytes: option parsing and the patch wiring on fake modules."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

import inkling_dfbytes as dfb  # noqa: E402


class Parse(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(dfb.parse({}), ())
        self.assertEqual(dfb.parse({dfb.ENV: "0"}), ())
        self.assertEqual(dfb.parse({dfb.ENV: "mx8, hot8"}), ("mx8", "hot8"))
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "mx8,bogus"})
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "hot8,hot16"})
        self.assertEqual(dfb.parse({dfb.ENV: "mx8,mx8w,hot8"}), ("mx8", "mx8w", "hot8"))
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "mx8,mx8w", "INKLING_DFLASH_MX8W_MAX_M": "x"})
        with self.assertRaises(RuntimeError):
            dfb.parse({dfb.ENV: "mx8w,hot8"})  # mx8w reuses mx8's bytes


@unittest.skipIf(torch is None, "needs torch")
class Wiring(unittest.TestCase):
    def setUp(self):
        from inkling_dfbytes import patches

        self.p = patches
        dfb.STATE.update(opts=("mx8", "hot8"), hot_file="x", hot_n=4)
        self.calls = []

    def test_patch_dflash(self):
        calls = self.calls

        class Base:
            def load_weights(self, weights):
                calls.append("load")
                return "loaded"

        class Sel(Base):
            lm_head = None
            merge_candidates_fused = False

            def compute_candidates(self, hidden):
                calls.append("stock")
                return "stock"

            def _transform_unary_logits(self, v):
                return v

        mod = types.SimpleNamespace(DFlashDraftModel=Base, DFlash2DraftModel=Sel, radix_topk=None)
        prepared = []
        orig_prep = self.p.prepare_layers
        self.p.prepare_layers = lambda m: prepared.append(m)
        try:
            self.p.patch_dflash(mod)
            self.p.patch_dflash(mod)  # idempotent
            s = Sel()
            self.assertEqual(s.load_weights([]), "loaded")
            self.assertEqual(prepared, [s])
            self.assertEqual(s.compute_candidates(torch.zeros(2, 4)), "stock")  # no lm_head yet: stock
        finally:
            self.p.prepare_layers = orig_prep

    def test_worker_builds_hot_before_capture(self):
        built = []
        orig = self.p._ensure_hot
        self.p._ensure_hot = lambda model, head: built.append((model, head))
        try:
            class W:
                def _maybe_build_draft_sampler(self):
                    return "built"

            mod = types.SimpleNamespace(DFlashWorkerV2=W, unwrap_lora_layer=lambda m: m)
            self.p.patch_worker(mod)
            w = W()
            head = object()
            w.draft_model = types.SimpleNamespace(candidate_selector=object())
            w._target_worker = types.SimpleNamespace(model_runner=types.SimpleNamespace(
                model=types.SimpleNamespace(lm_head=head)))
            self.assertEqual(w._maybe_build_draft_sampler(), "built")
            self.assertEqual(built, [(w.draft_model, head)])
        finally:
            self.p._ensure_hot = orig


@unittest.skipIf(torch is None, "needs torch")
class Mx8w(unittest.TestCase):
    def test_dispatch_rows(self):
        from inkling_dfbytes import patches as p

        calls = []

        class Base:
            def apply(self, layer, x, bias=None):
                calls.append(("inner", x.shape[0]))
                return "inner"

        Sub = p._mx8w_method_class(Base)
        self.assertIs(p._mx8w_method_class(Base), Sub)  # cached
        dfb.STATE["db_opts"] = types.SimpleNamespace(w8_max_m=16)
        dfb.STATE["mx8w_max_m"] = 256
        layer = types.SimpleNamespace(_idf_b12x_sf="sf", _idb_w8=types.SimpleNamespace(q="q", n=5, k=4))
        orig = p.b12x_linear
        p.b12x_linear = lambda x2, q, sf: (calls.append(("b12x", x2.shape[0], q, sf)), torch.zeros(x2.shape[0], 5))[1]
        orig_cuda = torch.Tensor.is_cuda
        try:
            torch.Tensor.is_cuda = property(lambda self: True)
            m = Sub()
            for rows in (8, 16, 17, 64, 256, 257):
                m.apply(layer, torch.zeros(rows, 4, dtype=torch.bfloat16))
            m.apply(layer, torch.zeros(64, 4, dtype=torch.bfloat16), bias=torch.zeros(5))  # bias: inner
            m.apply(layer, torch.zeros(64, 4, dtype=torch.float32))  # not BF16: inner
            y = m.apply(layer, torch.zeros(2, 32, 4, dtype=torch.bfloat16))  # 3-D input keeps its leading dims
            self.assertEqual(tuple(y.shape), (2, 32, 5))
            m.apply(layer, torch.zeros(64, 8, dtype=torch.bfloat16)[:, :4])  # strided view: inner
            m.apply(layer, torch.zeros(64, 6, dtype=torch.bfloat16))  # K mismatch: inner
        finally:
            torch.Tensor.is_cuda = orig_cuda
            p.b12x_linear = orig
        kinds = [c[0] + str(c[1]) for c in calls]
        self.assertEqual(kinds, ["b12x8", "b12x16", "b12x17", "b12x64", "b12x256", "inner257", "inner64", "inner64",
                                 "b12x64", "inner64", "inner64"])
        self.assertTrue(all(c[2:] == ("q", "sf") for c in calls if c[0] == "b12x"))  # operands forwarded


if __name__ == "__main__":
    unittest.main()
