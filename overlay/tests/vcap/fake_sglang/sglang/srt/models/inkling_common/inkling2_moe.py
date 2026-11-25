"""CPU stand-in with the patched classes' names and signatures: a toy float64 MoE.

EP (set by tests/vcap/fake_tp.py for the TP4 stand-in): the group of the expert-parallel ranks. Each rank then runs
only its contiguous slice of the experts on every token (as the image's Marlin _forward_source does at EP4) and the
partial sums are all-reduced; ``touched`` records the LOCAL experts that received at least one row."""

import torch
from torch import nn

EP = None


class Inkling2Gate(nn.Module):
    def __init__(self, hidden, experts, topk, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.topk = topk
        self.weight = nn.Parameter(
            torch.randn(experts, hidden, generator=g, dtype=torch.float64),
            requires_grad=False,
        )

    def forward(self, x):
        logits = x @ self.weight.T
        top, ids = torch.topk(logits, self.topk, dim=-1)
        return torch.softmax(top, dim=-1), ids.to(torch.int32)


class _ToyExperts(nn.Module):
    def __init__(self, hidden, experts, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed + 1)
        self.w = nn.Parameter(
            torch.randn(experts, hidden, hidden, generator=g, dtype=torch.float64)
            / hidden**0.5,
            requires_grad=False,
        )
        self.touched = []  # experts read per call (the "union")

    def forward(self, x, weights, ids, record=False):
        if EP is not None:
            return self._forward_ep(x, weights, ids, record)
        if record:
            self.touched.append(sorted(set(ids.reshape(-1).tolist())))
        out = torch.zeros_like(x)
        for k in range(ids.shape[1]):
            w = self.w[ids[:, k].long()]  # [M, H, H]
            out += weights[:, k : k + 1] * torch.tanh(torch.einsum("mh,mhj->mj", x, w))
        return out

    def _forward_ep(self, x, weights, ids, record):
        n = self.w.shape[0] // EP.world_size
        lo = EP.rank_in_group * n
        local = (ids >= lo) & (ids < lo + n)
        if record:
            self.touched.append(sorted(set(ids[local].reshape(-1).tolist())))
        out = torch.zeros_like(x)
        for k in range(ids.shape[1]):
            rows = local[:, k]
            if bool(rows.any()):
                w = self.w[ids[rows, k].long()]
                out[rows] += weights[rows, k : k + 1] * torch.tanh(torch.einsum("mh,mhj->mj", x[rows], w))
        return EP.all_reduce(out)


class Inkling2MoE(nn.Module):
    def __init__(self, hidden=16, experts=16, topk=2, seed=0):
        super().__init__()
        self.gate = Inkling2Gate(hidden, experts, topk, seed)
        self.experts = _ToyExperts(hidden, experts, seed)

    def forward(self, x, forward_batch=None, combine=True):
        weights, ids = self.gate(x)
        mode = getattr(forward_batch, "forward_mode", None)
        out = self.experts(x, weights, ids, record=bool(mode and mode.is_target_verify()))
        return out
