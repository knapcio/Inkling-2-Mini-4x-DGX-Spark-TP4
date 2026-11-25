"""inkling_vcap.fused_accept: the accept computed from logits (chain and block
verification, with and without the cap), checked through its dense torch primitives
(TorchPrims; the Triton kernels implement the same three primitives and are compared
with these on the GPU, tools/gpu_check_fused_accept.py).

Monte Carlo as in test_block_verify.py: 3-token Markov target/draft at temperatures
(target T = 0.8, draft T_d = 0.56), garbage dead target rows, L from the max-probability
rule, first-three-token joint law vs exact (300k sequences). Negative control: at the
stop (tau == g < G) the bonus subtracts the dead draft row (no stop) -> biased."""
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inkling_vcap import fused_accept as fa  # noqa: E402

P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]], dtype=torch.float64)
Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]], dtype=torch.float64)
G_ROW = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64)
T, TD = 0.8, 0.56


class NoStopPrims(fa.TorchPrims):
    def sample(self, tl2, dl2, st, t_temps, d_temps, tau, scale, use_q):
        return super().sample(tl2, dl2, st, t_temps, d_temps, tau, scale, torch.ones_like(use_q))


def simulate(n, *, bv, capped, prims, gamma=3, threshold=0.505, seed=0, out_len=3):
    gen = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    out = torch.full((n, out_len), -1, dtype=torch.long)
    filled = torch.zeros(n, dtype=torch.long)
    anchor = torch.zeros(n, dtype=torch.long)
    st = {"rows": 0, "acc": 0, "stop": 0}
    while bool((filled < out_len).any()):
        prev, drafts, qs = anchor, [], []
        for _ in range(gamma):
            q = Q[prev]
            d = torch.multinomial(q, 1, generator=gen).squeeze(1)
            drafts.append(d)
            qs.append(q)
            prev = d
        drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1)
        L = 1 + (torch.cumprod(qs.max(-1).values, 1) >= threshold).sum(1)
        row_tok = torch.cat([anchor[:, None], drafts], 1)
        ps = P[row_tok]
        dead = torch.arange(gamma + 1)[None, :] >= L[:, None]
        if capped:
            ps = torch.where(dead[..., None], G_ROW.expand_as(ps), ps)
        tl_ = (T * torch.log(ps)).float().reshape(n * (gamma + 1), 3)
        dl_ = (TD * torch.log(qs)).float()
        coins = torch.rand((n, gamma), generator=gen)
        tau, bonus, _ = fa.accept_core(
            prims, candidates=row_tok, target_logits=tl_, draft_logits=dl_,
            t_temps=torch.full((n,), T), d_temps=torch.full((n,), TD),
            live_lens=L if capped else None, block_verification=bv, coins=coins)
        tau = tau.long()
        if capped:
            assert bool((tau <= L - 1).all())
        emitted = torch.cat([drafts, bonus[:, None]], 1)
        n_emit = tau + 1
        active = filled < out_len
        st["rows"] += int(active.sum())
        st["acc"] += int(tau[active].sum())
        st["stop"] += int((active & capped & (tau == L - 1) & (L <= gamma)).sum()) if capped else 0
        for k in range(gamma + 1):
            take = active & (k < n_emit) & (filled + k < out_len)
            tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
            idx = (filled + k).clamp(max=out_len - 1)
            out[take, idx[take]] = tok[take]
        filled = torch.where(active, filled + n_emit, filled)
        anchor = torch.where(active, bonus, anchor)
    emp = torch.zeros(3, 3, 3, dtype=torch.float64)
    emp.index_put_((out[:, 0], out[:, 1], out[:, 2]), torch.ones(n, dtype=torch.float64),
                   accumulate=True)
    emp /= n
    exact = torch.einsum("a,ab,bc->abc", P[0], P, P)
    return float((emp - exact).abs().max()), st


class FusedAcceptLawTest(unittest.TestCase):
    N = 300_000

    def test_exact_arms(self):
        for i, (bv, capped) in enumerate(((False, False), (True, False), (False, True), (True, True))):
            err, st = simulate(self.N, bv=bv, capped=capped,
                               prims=fa.TorchPrims(torch.Generator().manual_seed(100 + i)), seed=i)
            print(f"\nfused bv={bv} capped={capped}: err {err:.2e} accepted/step "
                  f"{st['acc'] / st['rows']:.4f} stops {st['stop']}", file=sys.stderr)
            self.assertLess(err, 3.5e-3, (bv, capped, err, st))
            if capped:
                self.assertGreater(st["stop"], st["rows"] // 20, st)

    def test_negative_control_no_stop(self):
        for i, bv in enumerate((False, True)):
            err, st = simulate(self.N, bv=bv, capped=True,
                               prims=NoStopPrims(torch.Generator().manual_seed(7 + i)), seed=10 + i)
            print(f"\nno-stop control bv={bv}: err {err:.2e}", file=sys.stderr)
            self.assertGreater(err, 1e-2, (bv, err))


if __name__ == "__main__":
    unittest.main()
