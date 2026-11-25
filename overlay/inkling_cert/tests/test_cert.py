"""CPU tests of inkling_cert (numpy + torch, no GPU, no sglang). From overlay/:
    python3 -m unittest discover -s inkling_cert/tests -v
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inkling_cert import cert_math as cm  # noqa: E402
from inkling_cert import quant  # noqa: E402

K = cm.K


def head(V, seed=0, scale_rows=True):
    """A head shaped like Inkling's: rows of norm ~1, a few heavy-tailed entries."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((V, K)) * (1 + 0.5 * rng.standard_normal((V, 1)) ** 2)
    W[rng.random((V, K)) < 0.002] *= 6
    if scale_rows:
        W /= np.linalg.norm(W, axis=1, keepdims=True)
    return cm.bf16_round(W.astype(np.float32))


def rows_like(W, M, seed=0, peak=0.25):
    """Hidden rows with ||x|| ~ 220 and a peaked logit distribution (cos ~ peak with one or two rows)."""
    rng = np.random.default_rng(seed)
    V = W.shape[0]
    X = rng.standard_normal((M, K))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    for m in range(M):
        tgt = rng.integers(0, V, size=2)
        X[m] = X[m] * math.sqrt(1 - 2 * peak ** 2) + peak * W[tgt[0]] + peak * rng.random() * W[tgt[1]]
    X *= 220.0 / np.linalg.norm(X, axis=1, keepdims=True)
    return cm.bf16_round(X.astype(np.float32))


def exact_logits(x, W):
    return (x.astype(np.float64) @ W.astype(np.float64).T).astype(np.float32)


class PackTest(unittest.TestCase):
    def test_roundtrip(self):
        g = torch.Generator().manual_seed(0)
        q = torch.randint(-31, 32, (48, K), generator=g, dtype=torch.int32).to(torch.int8)
        p = quant.pack6(q)
        self.assertEqual(p.numel(), 48 // 8 * quant.STEPS * 384)
        self.assertTrue(torch.equal(quant.unpack6(p, 48), q))

    def test_kernel_lane_mapping(self):
        """The k index each lane / mma uses for B (decoded by the kernel's rule) equals the one it uses for A."""
        g = torch.Generator().manual_seed(1)
        q = torch.randint(-31, 32, (16, K), generator=g, dtype=torch.int32).to(torch.int8)
        b = quant.pack6(q).view(2, quant.STEPS, 384).long()
        seen = np.zeros((16, K), int)
        for i in range(2):                      # the CTA's two 8-row tiles
            for s in range(quant.STEPS):
                for lane in range(32):
                    gg, t = lane >> 2, lane & 3
                    nib = [int(sum(int(b[i, s, 8 * lane + 4 * h + j]) << (8 * j) for j in range(4))) for h in range(2)]
                    hi = int(sum(int(b[i, s, 256 + 4 * lane + j]) << (8 * j) for j in range(4)))
                    for j4 in range(4):
                        for half, p in ((0, 2 * j4), (1, 2 * j4 + 1)):
                            w = nib[0] if j4 < 2 else nib[1]
                            v = ((w >> (4 * (p & 3))) & 0x000F000F) | (((hi >> (2 * p)) & 0x00030003) << 4)
                            # A operand of this register: x row, values 16t + 4 j4 + 2 half + {0, 1} of step s
                            ks = [64 * s + 16 * t + 4 * j4 + 2 * half, 64 * s + 16 * t + 4 * j4 + 2 * half + 1]
                            vals = [(v & 0xFFFF) - 32, (v >> 16) - 32]
                            for k, val in zip(ks, vals):
                                self.assertEqual(val, int(q[8 * i + gg, k]))
                                seen[8 * i + gg, k] += 1
        self.assertTrue((seen == 1).all())

    def test_six_pair_exact_in_bf16(self):
        u = np.arange(64, dtype=np.uint32)
        bits = (0x4300 | u).astype(np.uint32) << 16
        vals = bits.view(np.float32) - np.float32(160.0)
        self.assertTrue(np.array_equal(vals, u.astype(np.float32) - 32))


class TablesTest(unittest.TestCase):
    def test_torch_matches_numpy(self):
        W = head(64, seed=3)
        t = quant.build_tables(torch.from_numpy(W).to(torch.bfloat16), chunk=32)
        q, sc, D = cm.quantize(W)
        rt, nv = cm.tables(W, D)
        self.assertTrue(torch.equal(quant.unpack6(t["q"], 64), torch.from_numpy(q)))
        self.assertTrue(np.array_equal(t["sc"].numpy().T, sc))
        self.assertTrue(np.array_equal(t["rt"].numpy().T, rt))
        self.assertTrue(np.array_equal(t["nv"].numpy(), nv))
        self.assertLessEqual(np.abs(q).max(), 31)

    def test_bound_covers_exact(self):
        W = head(4096, seed=4)
        q, sc, D = cm.quantize(W)
        rt, nv = cm.tables(W, D)
        x = rows_like(W, 8, seed=5)
        lo, hi = cm.screen(x, D, rt, nv, 4096)
        l = x.astype(np.float64) @ W.astype(np.float64).T
        self.assertTrue((lo <= l).all() and (l <= hi).all())


class CertifyTest(unittest.TestCase):
    V = 8192
    V_REAL = 8000

    def setUp(self):
        self.W = head(self.V, seed=7)
        self.q, self.sc, self.D = cm.quantize(self.W)
        self.rt, self.nv = cm.tables(self.W, self.D)

    def _check(self, x, l=None, s_err=None):
        lo, hi = cm.screen(x, self.D, self.rt, self.nv, self.V_REAL, s_err=s_err)
        mask, certified = cm.select(lo, hi)
        l = exact_logits(x, self.W) if l is None else l
        self.assertTrue(np.array_equal(cm.certified_token(l, mask, self.V_REAL), cm.stock_token(l, self.V_REAL)))
        return mask, certified

    def test_peaked_rows(self):
        x = rows_like(self.W, 16, seed=11)
        mask, certified = self._check(x)
        self.assertTrue(certified)
        self.assertLess(mask.sum(), self.V // 16 // 4)

    def test_adversarial_screen_error(self):
        """Screen errors pushed to the bound's accumulation term in the worst direction for each id."""
        x = rows_like(self.W, 8, seed=12)
        l = exact_logits(x, self.W)
        w = cm.stock_token(l, self.V_REAL)
        xn = np.linalg.norm(x.astype(np.float64), axis=1)
        err = cm.C_ACC * self.nv[None, :].astype(np.float64) * xn[:, None] * 0.999
        sign = -np.ones_like(err)
        sign[np.arange(8), w] = -1.0           # the winner looks lower, every other id higher
        sign[:, :] = np.where(np.arange(self.V)[None, :] == w[:, None], -1.0, 1.0)
        self._check(x, s_err=sign * err)

    def test_exact_ties_lowest_id(self):
        W = self.W.copy()
        W[5000] = W[300]                       # identical rows: identical logits
        W[6000] = W[300]
        self.W = W
        self.q, self.sc, self.D = cm.quantize(W)
        self.rt, self.nv = cm.tables(W, self.D)
        x = np.repeat(cm.bf16_round((W[300] * 220.0).astype(np.float32))[None], 4, 0)
        mask, certified = self._check(x)
        self.assertEqual(int(cm.certified_token(exact_logits(x, W), mask, self.V_REAL)[0]), 300)

    def test_softcap_ties(self):
        """Distinct raw logits that the softcap maps to the same FP32 value: the lowest id must still win."""
        x = rows_like(self.W, 1, seed=13)
        l = exact_logits(x, self.W)
        w = int(cm.stock_token(l, self.V_REAL)[0])
        lw = l[0, w]
        # put a lower-id token just below w in raw logit but equal after softcap (large logits saturate)
        tested = 0
        for big in (45.0, 60.0, 90.0, 120.0):
            l2 = l.copy()
            l2[0, w] = big
            other = 7 if w != 7 else 9
            cand = np.float32(big)
            while cm.softcap(np.array([cand]))[0] == cm.softcap(np.array([np.float32(big)]))[0]:
                cand = np.nextafter(cand, np.float32(-np.inf))
            cand = np.nextafter(cand, np.float32(np.inf))
            if cand == np.float32(big):
                continue                       # no softcap tie below this value in FP32
            tested += 1
            l2[0, other] = cand
            self.assertLess(l2[0, other], l2[0, w])
            self.assertEqual(int(cm.stock_token(l2, self.V_REAL)[0]), min(other, w))
            # bounds as tight as possible around the true values: the screen interval from l2 itself
            lo = l2 - 1e-3
            hi = l2 + 1e-3
            lo[:, self.V_REAL:] = -np.inf
            hi[:, self.V_REAL:] = -np.inf
            mask, certified = cm.select(lo.astype(np.float32), hi.astype(np.float32))
            self.assertTrue(certified)
            self.assertEqual(int(cm.certified_token(l2, mask, self.V_REAL)[0]), min(other, w))
        self.assertGreater(tested, 0)
        del lw

    def test_padded_ids_never_raise_threshold(self):
        x = rows_like(self.W, 2, seed=14)
        l = exact_logits(x, self.W)
        l[:, self.V_REAL:] = 1e4      # padded ids with huge logits must not matter
        s_err = np.zeros_like(l, dtype=np.float64)
        s_err[:, self.V_REAL:] = 1e4
        self._check(x, l=l, s_err=s_err)

    def test_out_of_range_falls_back(self):
        x = rows_like(self.W, 2, seed=15) * np.float32(40.0)   # logits in the thousands: not certifiable
        lo, hi = cm.screen(x, self.D, self.rt, self.nv, self.V_REAL)
        mask, certified = cm.select(lo, hi)
        self.assertFalse(certified)
        self.assertTrue(mask.all())

    def test_not_on_is_full(self):
        x = rows_like(self.W, 2, seed=16)
        lo, hi = cm.screen(x, self.D, self.rt, self.nv, self.V_REAL)
        mask, certified = cm.select(lo, hi, on=False)
        self.assertFalse(certified)
        self.assertTrue(mask.all())

    def test_fill_below_any_winner(self):
        """f(-inf) = -30 stays strictly below f(T) for every T the range test admits."""
        for T in (-150.0, -100.0, -30.0, 0.0):
            lo = np.full((1, 32), -np.inf, np.float32)
            hi = np.full((1, 32), -np.inf, np.float32)
            lo[0, 0], hi[0, 0] = T, T + 1
            mask, certified = cm.select(lo, hi)
            if certified:
                self.assertGreater(cm.softcap(np.array([T], np.float32))[0], cm.softcap(np.array([-np.inf], np.float32))[0])


class ModeTest(unittest.TestCase):
    def test_parse(self):
        import inkling_cert as ic

        self.assertEqual(ic.parse_mode(""), "0")
        self.assertEqual(ic.parse_mode("1"), "1")
        self.assertEqual(ic.parse_mode("check"), "check")
        with self.assertRaises(RuntimeError):
            ic.parse_mode("maybe")


if __name__ == "__main__":
    unittest.main()
