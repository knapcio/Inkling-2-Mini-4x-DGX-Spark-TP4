"""CPU unit tests (torch CPU, no Triton, no GPU).

  python -m unittest discover -s tests -v      (from overlay/; or -s tests/draftbytes)

The Triton kernels are not run here (gpu_check.py covers them against the
same reference math). These tests cover: option parsing, the no-op contract,
the FP8 reference numerics vs BF16, the patch plumbing end to end against
CPU stand-ins of the patched sglang modules (tests/stubs), the loud failure
modes, and the exactness argument for a restricted (hot-vocab) proposal.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

from inkling_draftbytes import _opts  # noqa: E402
from inkling_draftbytes.quant import (  # noqa: E402
    bf16_pack,
    dequantize,
    linear_ref,
    quantize,
)

HOT_FILE = os.path.join(tempfile.gettempdir(), f"idb_hot_{os.getpid()}.json")


def setUpModule():
    import random

    ids = list(range(4000))
    random.Random(0).shuffle(ids)
    with open(HOT_FILE, "w") as f:
        json.dump(ids, f)


def tearDownModule():
    try:
        os.remove(HOT_FILE)
    except OSError:
        pass


def run_scenario(name, options, extra_env=None, expect_ok=True):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "stubs"), ROOT])
    env["INKLING_DRAFT_HOTVOCAB_FILE"] = HOT_FILE
    if options is None:
        env.pop("INKLING_DRAFT_BYTES", None)
    else:
        env["INKLING_DRAFT_BYTES"] = options
    env.update(extra_env or {})
    proc = subprocess.run(
        [sys.executable, os.path.join(HERE, "_scenarios.py"), name],
        env=env, capture_output=True, text=True, timeout=300,
    )
    if not expect_ok:
        return proc
    if proc.returncode != 0:
        raise AssertionError(f"{name} [{options}] failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


class DwinTests(unittest.TestCase):
    def test_options(self):
        o = _opts.parse("dwin:16k", environ={})
        self.assertTrue(o.active)
        self.assertEqual((o.dwin, o.dwin_sink), (16384, 128))
        o = _opts.parse("dwin:8192:0, gemv", environ={})
        self.assertEqual((o.dwin, o.dwin_sink, o.gemv), (8192, 0, True))
        self.assertIn("dwin:8192:0", o.describe())
        self.assertEqual(_opts.parse("gemv", environ={}).dwin, 0)
        for bad in ("dwin", "dwin:0", "dwin:abc", "dwin:16k:x"):
            with self.assertRaises(_opts.OptionError, msg=bad):
                _opts.parse(bad, environ={})

    def test_window_table_matches_reference(self):
        from inkling_draftbytes.dwin import DwinCfg, reference_keys, window_page_metadata

        g = torch.Generator().manual_seed(0)
        for P, S, W in ((4, 4, 12), (4, 0, 8), (4, 8, 17), (128, 128, 16384), (4, 4, 1), (4, 0, 4), (128, 0, 300)):
            sp, wp = DwinCfg(W, S).pages(P)
            for width in (3, sp + wp, sp + wp + 1, 20, 400):
                table = torch.stack([torch.randperm(1000, generator=g)[:width] for _ in range(3)]).to(torch.int32)
                step = max(1, P // 4) if P <= 8 else 211
                for n0 in range(0, min(width * P, 3 * (sp + wp) * P + 5), step):
                    lens = torch.tensor([n0, max(0, n0 - 3), min(width * P, n0 + P + 1)], dtype=torch.int32)
                    built = window_page_metadata(table, lens, P, sp, W)
                    if width <= sp + wp:
                        self.assertIsNone(built)
                        continue
                    t, l, cu = built
                    self.assertEqual(tuple(t.shape), (3, sp + wp))
                    self.assertEqual(t.dtype, torch.int32)
                    self.assertEqual(cu.tolist(), [0] + torch.cumsum(l, 0).tolist())
                    for r in range(3):
                        n = int(lens[r])
                        keys = reference_keys(n, P, sp, W)  # independent: pages of the sink + of the last W tokens
                        self.assertEqual(int(l[r]), len(keys), (P, S, W, width, n))
                        phys = [int(t[r, i // P]) * P + i % P for i in range(len(keys))]
                        want = [int(table[r, j // P]) * P + j % P for j in keys]
                        self.assertEqual(phys, want, (P, S, W, width, n))
                        self.assertTrue(set(range(max(0, n - W), n)) <= set(keys))
                        self.assertTrue(set(range(min(n, sp * P))) <= set(keys))
                        if n < W + (sp + 1) * P:
                            self.assertEqual(keys, list(range(n)))

    def test_end_to_end_stubs(self):
        r = run_scenario("dwin_paths", "dwin:24:4")
        self.assertTrue(r["patched"])
        self.assertEqual(r["marks"], [False, True])
        self.assertEqual((r["sp"], r["wp"]), (1, 7))
        for c in r["cases"]:
            self.assertTrue(c["restored"], c)
            self.assertLess(c["err_vs_ref"], 1e-5, c)
            self.assertEqual(c["max_k"], [(1 + 7) * 4, None], c)  # windowed call gets the static bound; stock not
            self.assertTrue(c["no_leftover_attr"], c)
            if all(c["identity"]):
                self.assertTrue(c["equal_stock"], c)
            else:
                self.assertFalse(c["equal_stock"], c)
        self.assertEqual(r["passthrough_same_meta"], [True, True, True])
        self.assertTrue(r["raised_and_restored"])
        self.assertGreaterEqual(r["stats"]["passthrough_marked"], 2)

    def test_off_leaves_backend_stock(self):
        r = run_scenario("import_fa_only", "gemv")
        self.assertFalse(r["patched"])

    def test_stacks_with_tau(self):
        # merge check: tau (dspark_draft temperatures) and dwin (FA backend, drafter global layer) are independent
        o = _opts.parse("tau:0.7,dwin:16k:0", environ={})
        self.assertEqual((o.tau, o.dwin, o.dwin_sink), (0.7, 16384, 0))
        self.assertEqual(o.describe(), "tau:0.7,dwin:16384:0")
        r = run_scenario("dwin_paths", "dwin:24:4,tau:0.7")
        self.assertTrue(r["patched"])
        self.assertEqual((r["sp"], r["wp"]), (1, 7))
        for c in r["cases"]:
            self.assertTrue(c["restored"], c)
            self.assertLess(c["err_vs_ref"], 1e-5, c)
            self.assertEqual(c["equal_stock"], all(c["identity"]), c)


class OptionTests(unittest.TestCase):
    def test_empty_is_off(self):
        self.assertFalse(_opts.parse("", environ={}).active)
        self.assertFalse(_opts.parse("off", environ={}).active)

    def test_full_list(self):
        o = _opts.parse(
            "gemv, fold8:g64, head8:mx, draftmx8, hotvocab:32768",
            environ={"INKLING_DRAFT_HOTVOCAB_FILE": "/x.json"},
        )
        self.assertTrue(o.gemv)
        self.assertEqual((o.fold8.kind, o.fold8.group), ("group", 64))
        self.assertEqual(o.head8.kind, "mx")
        self.assertEqual(o.draftmx8.kind, "mx")
        self.assertEqual(o.hotvocab, 32768)
        self.assertEqual(o.describe(), "gemv,fold8:g64,head8:mx,draftmx8:mx,hotvocab:32768")

    def test_errors(self):
        for bad in ("gemm", "hotvocab", "hotvocab:0", "fold8:g24", "fold8:int4"):
            with self.assertRaises(_opts.OptionError, msg=bad):
                _opts.parse(bad, environ={"INKLING_DRAFT_HOTVOCAB_FILE": "/x"})
        with self.assertRaises(_opts.OptionError):
            _opts.parse("hotvocab:1024", environ={})


class Fp8NumericsTests(unittest.TestCase):
    """FP8 fold vs the BF16 fold, on fold-shaped data.

    Data: fold = head[N, 2304] @ P[2304, 256] (both Gaussian), N = 8192 rows,
    latent = tanh(Gaussian) like the RNN output. Error = ||y8 - y16|| / ||y16||
    over a [8, N] bias block.

    Documented tolerances (e4m3 keeps 3 mantissa bits; the per-element rounding
    error is uniform within +-2^-4 relative, about 2.6 % RMS once averaged over
    a 256-term dot product of mixed-magnitude terms):
      row  (fp32 scale per row)        rel <= 0.035
      g32  (fp32 scale per 32)         rel <= 0.035
      mx   (e8m0 scale per 32, MXFP8)  rel <= 0.040
    The math of the GPU kernel equals linear_ref up to FP32 summation order
    (FP8 -> BF16 is exact, e8m0 scales are powers of two); gpu_check.py
    asserts kernel vs reference within 1 BF16 ulp on >= 99.9 % of outputs.
    """

    @classmethod
    def setUpClass(cls):
        g = torch.Generator().manual_seed(0)
        head = torch.randn(8192, 2304, generator=g) * 0.02
        proj = torch.randn(2304, 256, generator=g) / 2304 ** 0.5
        cls.fold = (head @ proj).to(torch.bfloat16)
        cls.fold_bias = (torch.randn(8192, generator=g) * 0.1).to(torch.bfloat16)
        cls.latent = torch.tanh(torch.randn(8, 256, generator=g)).to(torch.bfloat16)
        cls.ref = linear_ref(cls.latent, bf16_pack(cls.fold), cls.fold_bias).float()

    def _rel(self, kind, group=0):
        pack = quantize(self.fold, kind, group)
        y = linear_ref(self.latent, pack, self.fold_bias).float()
        return ((y - self.ref).norm() / self.ref.norm()).item(), pack

    def test_row(self):
        rel, pack = self._rel("row")
        self.assertLessEqual(rel, 0.035)
        self.assertEqual(pack.scale.shape, (8192, 1))
        self.assertEqual(pack.nbytes(), 8192 * 256 + 8192 * 4)

    def test_group32(self):
        rel, _ = self._rel("group", 32)
        self.assertLessEqual(rel, 0.035)

    def test_mx(self):
        rel, pack = self._rel("mx")
        self.assertLessEqual(rel, 0.040)
        self.assertEqual(pack.scale.dtype, torch.uint8)
        # e8m0 dequant never clips: |q| <= 448 and no saturated block amax
        self.assertLessEqual(pack.q.float().abs().max().item(), 448.0)

    def test_dequant_roundtrip_bound(self):
        pack = quantize(self.fold, "row")
        w = self.fold.float()
        err = (dequantize(pack) - w).abs()
        # Per element: half an e4m3 ulp, i.e. <= 2^-4 relative for normal values,
        # plus the subnormal floor (2^-9 * scale) near zero.
        scale = pack.scale.float()
        bound = w.abs() * 2 ** -4 + scale * 2 ** -10 + 1e-12
        self.assertTrue(bool((err <= bound).all()))

    def test_bf16_pack_matches_addmm(self):
        # Same math as the stock addmm; only the FP32 summation order differs,
        # so outputs agree within one BF16 ulp.
        y = linear_ref(self.latent, bf16_pack(self.fold), self.fold_bias).float()
        ref = torch.addmm(self.fold_bias.float(), self.latent.float(), self.fold.float().T)
        ref16 = ref.to(torch.bfloat16).float()
        self.assertTrue(bool(((y - ref16).abs() <= ref16.abs() * 2 ** -7 + 1e-30).all()))
        self.assertGreater((y == ref16).float().mean().item(), 0.99)

    def test_zero_rows(self):
        w = torch.zeros(64, 256, dtype=torch.bfloat16)
        for kind in ("row", "mx"):
            pack = quantize(w, kind, 32)
            self.assertTrue(torch.equal(dequantize(pack), w.float()))


class PatchTests(unittest.TestCase):
    def test_noop_when_unset(self):
        r = run_scenario("noop", None)
        self.assertEqual(
            r, {"installed": False, "meta_path_grew": False,
                "step_bias_stock": True, "project_stock": True},
        )

    def test_foldk_equals_stock(self):
        r = run_scenario("fold_compare", "foldk")
        self.assertLess(r["rel_err"], 1e-5)  # CPU BLAS summation order differs by platform
        self.assertIn("Inkling2MTPHead.step_bias", r["patched"])

    def test_fold8_close_to_stock(self):
        for opt in ("fold8", "fold8:mx", "fold8:g32"):
            r = run_scenario("fold_compare", opt)
            self.assertTrue(r["shape_ok"], opt)
            self.assertEqual(r["dtype"], "torch.bfloat16")
            self.assertLess(r["rel_err"], 0.04, opt)

    def test_hotvocab_full_set_is_bit_identical(self):
        # Hot set = every id under the vocab limit: proposals and kept logits
        # must equal the stock path bit for bit.
        r = run_scenario("hot_compare", "hotvocab:4000")
        self.assertGreater(r["hot_exact_frac"], 0.9999)  # CPU BLAS summation order differs by platform
        self.assertEqual(r["agree_when_ref_hot"], 1.0)
        self.assertEqual(r["compared_steps"], 14)

    def test_hotvocab_subset(self):
        r = run_scenario("hot_compare", "hotvocab:2500")
        self.assertTrue(r["off_all_neginf"])
        self.assertTrue(r["tokens_all_hot"])
        self.assertTrue(r["hot_neginf_match"])
        self.assertGreater(r["hot_exact_frac"], 0.9999)  # CPU BLAS summation order differs by platform

    def test_hotvocab_stacked(self):
        r = run_scenario("hot_compare", "gemv,fold8,head8,hotvocab:2500")
        self.assertTrue(r["off_all_neginf"])
        self.assertTrue(r["tokens_all_hot"])
        for name in ("dspark_draft._draft_step", "Inkling2MTPHead.step_bias",
                     "inkling2_mtp.project_through_lm_head"):
            self.assertIn(name, r["patched"])

    def test_hotvocab_sampled_rows(self):
        self.assertTrue(run_scenario("hot_sampled", "hotvocab:2000")["ok"])

    def test_rnnhoist(self):
        for opt in ("rnnhoist", "rnnhoist,fold8,hotvocab:2500"):
            r = run_scenario("rnnhoist_compare", opt)
            self.assertIn("Inkling2MTPHead._rnn_step", r["patched"], opt)
            self.assertIn("Inkling2MTPHead.init_state", r["patched"], opt)
            self.assertEqual(r["hoisted_calls"], 7, opt)  # every step of the walk took the hoisted path
            self.assertEqual(r["step_index"], list(range(7)), opt)
            self.assertEqual(r["outside_walk"], [-1, -1, -1, -1], opt)
            self.assertTrue(r["too_many_rows"], opt)
            self.assertLessEqual(r["proj_worst_ulp"], 1.0, opt)  # FP32 order only: at most one BF16 ulp
            self.assertTrue(r["neginf_match"], opt)
            self.assertGreater(r["logits_within_1ulp"], 0.999, opt)

    def test_gemv_cpu_falls_back_to_stock(self):
        r = run_scenario("head_compare", "gemv")
        self.assertLess(r["rel_err"], 1e-5)  # CPU BLAS summation order differs by platform

    def test_head8(self):
        r = run_scenario("head_compare", "head8")
        self.assertLess(r["rel_err"], 0.04)

    def test_draftmx8(self):
        r = run_scenario("draftmx8", "draftmx8")
        self.assertTrue(r["isinstance_kept"])
        self.assertTrue(r["cpu_fallback_exact"])
        self.assertEqual(r["n_wrapped"], 4)
        self.assertEqual(r["pack_label"], "mx")
        self.assertLess(r["w8_rel_err"], 0.04)

    def test_missing_target_fails_loudly(self):
        proc = run_scenario(
            "import_only", "fold8",
            extra_env={"STUB_DROP": "Inkling2MTPHead.step_bias"}, expect_ok=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("DraftBytesPatchError", proc.stderr)
        self.assertIn("step_bias", proc.stderr)

    def test_sitecustomize_exits_on_bad_option(self):
        env = dict(os.environ, PYTHONPATH=ROOT, INKLING_DRAFT_BYTES="nonsense")
        proc = subprocess.run([sys.executable, "-c", "print('ran')"], env=env,
                              capture_output=True, text=True, timeout=120)
        self.assertEqual(proc.returncode, 78)
        self.assertIn("FATAL", proc.stderr)
        self.assertNotIn("ran", proc.stdout)

    def test_sitecustomize_noop(self):
        env = dict(os.environ, PYTHONPATH=ROOT)
        env.pop("INKLING_DRAFT_BYTES", None)
        proc = subprocess.run(
            [sys.executable, "-c", "import sys; print('inkling_draftbytes' in sys.modules)"],
            env=env, capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(proc.stdout.strip(), "False")


class RestrictedProposalExactness(unittest.TestCase):
    """Speculative sampling with a draft q that is zero off the hot set still
    returns exactly the target p: P(out = x) = q(x) min(1, p/q) + P(reject)
    * max(0, p - q)(x) / Z. Checked in closed form for random p, q."""

    def test_output_is_target(self):
        g = torch.Generator().manual_seed(1)
        for _ in range(50):
            v = 64
            p = torch.softmax(torch.randn(v, generator=g, dtype=torch.float64) * 2, 0)
            hot = torch.rand(v, generator=g) < 0.4
            logits = torch.randn(v, generator=g, dtype=torch.float64) * 2
            logits[~hot] = float("-inf")
            q = torch.softmax(logits, 0)
            accept = torch.where(q > 0, torch.minimum(torch.ones_like(p), p / q.clamp_min(1e-300)), 0.0)
            p_accept = (q * accept).sum()
            residual = (p - q).clamp_min(0)
            out = q * accept + (1 - p_accept) * residual / residual.sum()
            self.assertTrue(torch.allclose(out, p, atol=1e-12))


if __name__ == "__main__":
    unittest.main()
