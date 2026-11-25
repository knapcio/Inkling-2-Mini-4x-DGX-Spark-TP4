"""Block verification (Sun et al., ICLR 2025, Algorithm 2) under the verify cap.

  python -m unittest discover -s tests/vcap -p 'test_block_verify.py' -v   (from overlay/)

The engine's block-verification kernel (speculative_sampling_classic_kernel with
BLOCK_VERIFICATION=True) is mirrored here in float64 (``block_verify_ref``), line by
line: the prefix probabilities p_i = min(p_{i-1} p(X_i)/q(X_i), 1) (0 when q(X_i) is
not in (0, 1]), the backward walk with h_i = r_i / (r_i + 1 - p_i), r_i =
sum max(p_i P_i - Q_i, 0) (h = p at the last level, or when p_i is 0 or 1), the final
sample from max(p_tau P_tau - Q_tau, 0) (P_gamma alone when every draft is accepted).

The cap gives each request a live length L from the draft's max probabilities (a
stopping rule: whether d_j is verified depends on d_<j only). logic.mask_dead_draft_rows
turns the dead drafts into a stopped draft (README, "Block verification under the cap");
the stock kernel then runs unchanged. Dead target rows get a deliberately wrong
distribution. The Monte Carlo checks the joint law of the first three emitted tokens
against the exact target law (300k sequences), asserts that the boundary cases occur,
and runs two negative controls: the stock kernel over all rows with the cap applied
afterwards (what vcap refused before), and the mask driven by a look-ahead rule (the
proposed token's own probability)."""

import os
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

from inkling_vcap import logic  # noqa: E402


def block_verify_ref(drafts, ps, qs, u, u_final):
    """Mirror of the stock Triton block-verification branch + final sampling.
    drafts [n, g] (X_1..X_g), ps [n, g+1, V] target rows, qs [n, g, V] draft rows,
    u [n, g] level coins, u_final [n]. Returns (num_accept [n], final_token [n])."""
    n, g = drafts.shape
    V = ps.shape[-1]
    rows = torch.arange(n)
    prefix = torch.ones(n, dtype=ps.dtype)
    prefix_probs = []
    for step in range(1, g + 1):
        tok = drafts[:, step - 1]
        p = ps[rows, step - 1, tok]
        q = qs[rows, step - 1, tok]
        q_ok = (q > 0.0) & (q <= 1.0)
        ratio = prefix * p / torch.where(q_ok, q, torch.ones_like(q))
        prefix = torch.where(q_ok, torch.clamp(ratio, max=1.0), torch.zeros_like(q))
        prefix_probs.append(prefix)
    num_accept = torch.zeros(n, dtype=torch.long)
    residual_scale = torch.ones(n, dtype=ps.dtype)
    norm_sum = torch.full((n,), -1.0, dtype=ps.dtype)
    for step in range(g, 0, -1):
        pp = prefix_probs[step - 1]
        coin = u[:, step - 1]
        cand = (num_accept == 0) & (coin < pp)
        accept_prob = pp.clone()
        residual_mass = torch.full((n,), -1.0, dtype=ps.dtype)
        if step < g:
            need = (pp > 0.0) & (pp < 1.0)
            rm = (pp[:, None] * ps[:, step] - qs[:, step]).clamp(min=0.0).sum(-1)
            accept_prob = torch.where(need, rm / (rm + 1.0 - pp), pp)
            residual_mass = torch.where(need, rm, residual_mass)
        acc = cand & (coin < accept_prob)
        num_accept = torch.where(acc, torch.full_like(num_accept, step), num_accept)
        residual_scale = torch.where(acc, pp, residual_scale)
        norm_sum = torch.where(acc, residual_mass, norm_sum)
    all_acc = num_accept == g
    p_row = ps[rows, num_accept]
    q_row = qs[rows, num_accept.clamp(max=g - 1)]
    q_row = torch.where((q_row >= 0.0) & (q_row <= 1.0), q_row, torch.zeros_like(q_row))
    val = torch.where(all_acc[:, None], p_row,
                      (residual_scale[:, None] * p_row - q_row).clamp(min=0.0))
    ns = torch.where(norm_sum < 0.0, val.sum(-1), norm_sum)
    target_u = u_final * ns
    hit = val.cumsum(-1) > target_u[:, None]
    final = torch.where(hit.any(-1), hit.to(torch.int8).argmax(-1),
                        torch.full((n,), V - 1, dtype=torch.long))
    return num_accept, final


class BlockVerifyLawTest(unittest.TestCase):
    P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
                     dtype=torch.float64)  # target: order-1 Markov over 3 tokens
    Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]],
                     dtype=torch.float64)  # draft
    G = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64)  # dead-row garbage

    def _simulate(self, n, arm, threshold=0.45, gamma=3, out_len=3, seed=0):
        gen = torch.Generator().manual_seed(seed)
        P, Q, G = self.P, self.Q, self.G
        out = torch.full((n, out_len), -1, dtype=torch.long)
        filled = torch.zeros(n, dtype=torch.long)
        anchor = torch.zeros(n, dtype=torch.long)
        st = {"rows": 0, "boundary": 0, "inside": 0, "l1": 0, "full": 0, "trim": 0,
              "accepted": 0}
        while bool((filled < out_len).any()):
            prev, drafts, qs = anchor, [], []
            for _ in range(gamma):
                q = Q[prev]
                d = torch.multinomial(q, 1, generator=gen).squeeze(1)
                drafts.append(d)
                qs.append(q)
                prev = d
            drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1).contiguous()
            if arm == "lookahead":  # negative control: cut on the proposed token's prob
                m = qs.gather(2, drafts[..., None]).squeeze(2)
            else:
                m = qs.max(-1).values
            surv = torch.cumprod(m, 1)
            L = 1 + (surv >= threshold).sum(1)
            if arm == "full":
                L = torch.full_like(L, gamma + 1)
            row_tok = torch.cat([anchor[:, None], drafts], 1)
            ps = P[row_tok]
            dead = torch.arange(gamma + 1)[None, :] >= L[:, None]
            ps = torch.where(dead[..., None], G.expand_as(ps), ps).contiguous()
            u = torch.rand((n, gamma), generator=gen, dtype=torch.float64)
            u_final = torch.rand(n, generator=gen, dtype=torch.float64)
            if arm in ("masked", "full", "lookahead"):
                logic.mask_dead_draft_rows(qs, drafts, L)
                tau, final = block_verify_ref(drafts, ps, qs, u, u_final)
                self.assertTrue(bool((tau <= L - 1).all()), "accepted past the cut")
                correct, bonus = tau, final
            elif arm in ("cap_after", "leak"):
                # cap_after: the stock kernel over every row (dead ones included), then
                # the engine's CapCorrectLen + bonus gather: when the cap trims, the bonus
                # is predicts[accept_index[L-1]] = X_L. Exact as well (README proof).
                # leak (negative control): on a trim, take the final sample drawn at the
                # dead row tau instead.
                tau, final = block_verify_ref(drafts, ps, qs, u, u_final)
                correct = torch.minimum(tau, L - 1)
                at_cap = drafts.gather(1, correct.clamp(max=gamma - 1)[:, None]).squeeze(1)
                bonus = torch.where(correct < tau, final if arm == "leak" else at_cap, final)
            else:
                raise ValueError(arm)
            emitted = torch.cat([drafts, bonus[:, None]], 1)
            n_emit = correct + 1
            active = filled < out_len
            st["rows"] += int(active.sum())
            st["accepted"] += int(correct[active].sum())
            st["boundary"] += int((active & (correct == L - 1) & (L >= 2) & (L <= gamma)).sum())
            st["inside"] += int((active & (correct < L - 1) & (L <= gamma)).sum())
            st["l1"] += int((active & (L == 1)).sum())
            st["full"] += int((active & (L == gamma + 1)).sum())
            if arm in ("cap_after", "leak"):
                st["trim"] += int((active & (correct < tau)).sum())
            for k in range(gamma + 1):
                take = active & (k < n_emit) & (filled + k < out_len)
                tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
                idx = (filled + k).clamp(max=out_len - 1)
                out[take, idx[take]] = tok[take]
            filled = torch.where(active, filled + n_emit, filled)
            anchor = torch.where(active, bonus, anchor)
        return out, st

    def _error(self, arm, n=300_000, seed=0, **kw):
        out, st = self._simulate(n, arm, seed=seed, **kw)
        emp = torch.zeros(3, 3, 3, dtype=torch.float64)
        emp.index_put_((out[:, 0], out[:, 1], out[:, 2]),
                       torch.ones(n, dtype=torch.float64), accumulate=True)
        emp /= n
        P = self.P
        exact = torch.einsum("a,ab,bc->abc", P[0], P, P)
        return float((emp - exact).abs().max()), st

    def test_reference_is_exact_uncapped(self):
        err, st = self._error("full")
        print(f"\nfull width: err {err:.2e} accepted/step {st['accepted'] / st['rows']:.3f}",
              file=sys.stderr)
        self.assertLess(err, 3.5e-3, (err, st))

    def test_masked_block_verify_under_cap_is_exact(self):
        for gamma, threshold, seed in ((3, 0.505, 0), (5, 0.3, 1)):
            err, st = self._error("masked", gamma=gamma, threshold=threshold, seed=seed)
            print(f"\nmasked gamma {gamma} thr {threshold}: err {err:.2e} {st}", file=sys.stderr)
            self.assertLess(err, 3.5e-3, (err, st))
            r = st["rows"]
            # every kind of step occurs often: accepted up to the cut, stopped inside,
            # full width; L = 1 with gamma 3
            self.assertGreater(st["boundary"], r // 20, st)
            self.assertGreater(st["inside"], r // 20, st)
            if gamma == 3:
                self.assertGreater(st["l1"], r // 50, st)
                self.assertGreater(st["full"], r // 50, st)

    def test_stock_kernel_with_cap_after_is_exact_too(self):
        """P(tau >= i | X_1..X_i) = p_i holds for ANY target rows past i (README), so
        the stock kernel over garbage dead rows + the engine's cap is exact as well."""
        for gamma, threshold, seed in ((3, 0.505, 2), (5, 0.3, 3)):
            err, st = self._error("cap_after", gamma=gamma, threshold=threshold, seed=seed)
            print(f"\ncap after stock kernel gamma {gamma}: err {err:.2e} {st}", file=sys.stderr)
            self.assertGreater(st["trim"], st["rows"] // 20, st)
            self.assertLess(err, 3.5e-3, (err, st))

    def test_same_acceptance_masked_and_cap_after(self):
        """E[accepted] = sum_k E[p_k] over the live levels in both."""
        _, a = self._error("masked", seed=4)
        _, b = self._error("cap_after", seed=5)
        ra, rb = a["accepted"] / a["rows"], b["accepted"] / b["rows"]
        print(f"\naccepted/step masked {ra:.4f} cap_after {rb:.4f}", file=sys.stderr)
        self.assertLess(abs(ra - rb), 0.01)

    def test_negative_control_dead_row_leak(self):
        err, st = self._error("leak")
        print(f"\ndead-row leak: err {err:.2e} {st}", file=sys.stderr)
        self.assertGreater(st["trim"], st["rows"] // 20, st)
        self.assertGreater(err, 1e-2, (err, st))

    def test_negative_control_lookahead_cut(self):
        err, st = self._error("lookahead", threshold=0.3)
        print(f"\nlook-ahead cut: err {err:.2e} {st}", file=sys.stderr)
        self.assertGreater(err, 1e-2, (err, st))


class MaskTest(unittest.TestCase):
    def test_mask_pattern(self):
        torch.manual_seed(0)
        bs, G, V = 4, 7, 11
        probs = torch.softmax(torch.randn(bs, G, V), -1).contiguous()
        ref = probs.clone()
        toks = torch.randint(0, V, (bs, G))
        lens = torch.tensor([1, 3, 8, 7, 0, 0])  # 0 = padding beyond bs (ignored)
        logic.mask_dead_draft_rows(probs, toks, lens)
        for b, L in enumerate([1, 3, 8, 7]):
            g = L - 1
            for j in range(G):
                row = probs[b, j]
                if j < g:
                    self.assertTrue(torch.equal(row, ref[b, j]))
                elif j == g:
                    self.assertTrue(bool((row == 0).all()))
                else:
                    self.assertEqual(float(row[toks[b, j]]), 0.0)
                    other = torch.ones(V, dtype=torch.bool)
                    other[toks[b, j]] = False
                    self.assertTrue(torch.equal(row[other], ref[b, j][other]))

    def test_padding_request_l0(self):
        probs = torch.full((1, 3, 4), 0.25)
        logic.mask_dead_draft_rows(probs, torch.tensor([[1, 2, 3]]), torch.tensor([0]))
        self.assertTrue(bool((probs[0, 0] == 0).all()))
        self.assertEqual([float(probs[0, j, t]) for j, t in ((1, 2), (2, 3))], [0.0, 0.0])


class WrapperTest(unittest.TestCase):
    """The accept wrapper arms the mask only for block verification with a cutoff, and
    the _draft_probs wrapper applies it with the step's live lengths."""

    def setUp(self):
        from inkling_vcap import patches
        from inkling_vcap.config import parse_config
        from inkling_vcap.state import STATE

        self.patches, self.STATE = patches, STATE
        self.saved = dict(STATE.__dict__)
        STATE.config = parse_config({"INKLING_VERIFY_CAP": "fixed:3"})
        STATE.allocate(stride=4, capacity=8, device="cpu")
        STATE.enabled = True
        STATE.capping = True
        STATE.live_lens[:2] = torch.tensor([3, 1])
        STATE._src_key = ("dyn", 0, 2)

    def tearDown(self):
        self.STATE.__dict__.clear()
        self.STATE.__dict__.update(self.saved)

    def _build(self):
        mod = types.SimpleNamespace()
        seen = {}

        def _draft_probs(draft_block, vocab, device):
            return torch.full((*draft_block.draft_tokens.shape, vocab), 1.0 / vocab)

        def accept_draft_tokens(*, candidates, target_logits=None, draft_block=None,
                                cutoff_layout=None, block_verification=False):
            seen["lens"] = None if cutoff_layout is None else cutoff_layout.verify_lens.tolist()
            seen["probs"] = mod._draft_probs(draft_block, 5, "cpu")
            return seen

        mod._draft_probs = self.patches._wrap_draft_probs(_draft_probs)
        mod.accept_draft_tokens = self.patches._wrap_accept_draft_tokens(accept_draft_tokens)
        return mod

    def test_masks_with_block_verification(self):
        mod = self._build()
        block = types.SimpleNamespace(draft_tokens=torch.tensor([[1, 2, 3], [4, 0, 1]]))
        seen = mod.accept_draft_tokens(candidates=torch.zeros(2, 4), draft_block=block,
                                       block_verification=True)
        self.assertEqual(seen["lens"], [3, 1])
        p = seen["probs"]
        self.assertTrue(torch.equal(p[0, :2], torch.full((2, 5), 0.2)))
        self.assertTrue(bool((p[0, 2] == 0).all()) and bool((p[1, 0] == 0).all()))
        self.assertEqual(float(p[1, 1, 0]), 0.0)
        self.assertIsNone(self.STATE.bv_lens)  # cleared after the call
        self.assertEqual(self.STATE.bv_masked, 1)

    def test_no_mask_without_block_verification(self):
        mod = self._build()
        block = types.SimpleNamespace(draft_tokens=torch.tensor([[1, 2, 3], [4, 0, 1]]))
        seen = mod.accept_draft_tokens(candidates=torch.zeros(2, 4), draft_block=block)
        self.assertTrue(bool((seen["probs"] == 0.2).all()))

    def test_full_width_step_skips_mask(self):
        mod = self._build()
        self.STATE.live_lens[:2] = 4
        self.STATE._src_key = ("const", 4, 2)
        block = types.SimpleNamespace(draft_tokens=torch.tensor([[1, 2, 3], [4, 0, 1]]))
        seen = mod.accept_draft_tokens(candidates=torch.zeros(2, 4), draft_block=block,
                                       block_verification=True)
        self.assertTrue(bool((seen["probs"] == 0.2).all()))
        self.assertEqual(self.STATE.bv_skipped, 1)


if __name__ == "__main__":
    unittest.main()
