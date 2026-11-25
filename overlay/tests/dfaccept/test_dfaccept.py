"""CPU tests: the sparse-q core equals inkling_vcap.fused_accept's dense core fed the same q as log-probabilities."""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402

from inkling_dfaccept import core  # noqa: E402
from inkling_vcap import fused_accept as fa  # noqa: E402


def case(seed, bs=5, G=7, V=300, K=16):
    g = torch.Generator().manual_seed(seed)
    S = G + 1
    tlog = torch.randn(bs * S, V, generator=g) * 2
    tlog[:, V - 20:] = float("-inf")  # padded vocabulary rows
    cand = torch.stack([torch.stack([torch.randperm(V - 20, generator=g)[:K] for _ in range(G)]) for _ in range(bs)])
    q = torch.softmax(torch.randn(bs, G, K, generator=g) * 2, -1)
    pick = torch.multinomial(q.view(-1, K), 1, generator=g).view(bs, G)
    drafts = cand.gather(2, pick[..., None]).squeeze(2)
    candidates = torch.cat([torch.zeros(bs, 1, dtype=torch.int64), drafts], 1)
    temps = torch.rand(bs, generator=g) + 0.3
    coins = torch.rand(bs, G, generator=g)
    Qd = torch.zeros(bs, G, V).scatter_(2, cand, q)
    return dict(candidates=candidates, tlog=tlog, cand=cand, q=q, temps=temps, coins=coins, Qd=Qd)


class SparseVsDense(unittest.TestCase):
    def test_tau_and_bonus_law(self):
        for bv in (False, True):
            for live in (None, torch.tensor([8, 3, 1, 5, 8])):
                for seed in range(6):
                    c = case(seed)
                    tau_s, _, _ = core.accept_core_sparse(
                        core.TorchPrims(torch.Generator().manual_seed(1)), candidates=c["candidates"],
                        target_logits=c["tlog"], cand_ids=c["cand"], q_rows=c["q"], t_temps=c["temps"],
                        live_lens=live, block_verification=bv, coins=c["coins"])
                    tau_d, _, _ = fa.accept_core(
                        fa.TorchPrims(torch.Generator().manual_seed(1)), candidates=c["candidates"],
                        target_logits=c["tlog"], draft_logits=torch.log(c["Qd"]), t_temps=c["temps"],
                        d_temps=torch.ones(5), live_lens=live, block_verification=bv, coins=c["coins"])
                    self.assertTrue(torch.equal(tau_s, tau_d), (bv, live, seed))

    def test_masses_equal_dense(self):
        c = case(3)
        P = core.TorchPrims()
        bs, G = 5, 7
        S = G + 1
        st = P.stats(c["tlog"], c["temps"], S, torch.ones(bs * S, dtype=torch.bool))
        st = (st[0].view(bs, S), st[1].view(bs, S))
        pp = torch.rand(bs, G)
        need = torch.ones(bs, G, dtype=torch.bool)
        ms = P.masses(c["tlog"], st, c["temps"], c["cand"], c["q"], pp, need)
        Pd, Qd = P.dense(c["tlog"], st, c["temps"], c["cand"], c["q"])
        for j in range(G - 1):
            ref = (pp[:, j, None] * Pd[:, j + 1] - Qd[:, j + 1]).clamp(min=0).sum(-1)
            self.assertTrue(torch.allclose(ms[:, j], ref))


if __name__ == "__main__":
    unittest.main()
