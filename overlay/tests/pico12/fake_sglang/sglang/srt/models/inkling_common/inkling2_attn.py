"""Stand-in of sglang/srt/models/inkling_common/inkling2_attn.py: _stacked and _project_sources are the stock
code; the linears are minimal modules with an UnquantizedLinearMethod."""
from typing import NamedTuple

import torch
from torch import nn

from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

INKLING_SPLITS = [3072, 1024, 1024, 384, 3072, 24]  # Inkling-2-Mini layer 0 at TP1 (sum 8600)


class AttentionSources(NamedTuple):
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor


def _stacked(sources):
    """torch.cat(sources) without a copy when the sources are adjacent in memory."""
    q = sources.q
    storage = q.untyped_storage().data_ptr()
    adjacent = all(
        x.is_contiguous()
        and x.shape == q.shape
        and x.dtype == q.dtype
        and x.untyped_storage().data_ptr() == storage
        and x.data_ptr() == q.data_ptr() + i * q.nbytes
        for i, x in enumerate(sources)
    )
    if adjacent and q.ndim == 2:
        return q.as_strided((3 * q.shape[0], q.shape[1]), q.stride())
    return torch.cat(sources)


class _Linear(nn.Module):
    def __init__(self, n, k, bias=False, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        w = (torch.randn((n, k), generator=g) * 0.05).to(torch.bfloat16)
        self.weight = nn.Parameter(w, requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(n, dtype=torch.bfloat16), requires_grad=False) if bias else None
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x, output_tensor=None):
        if output_tensor is not None:
            return self.quant_method.apply_into(self, x, output_tensor, bias=self.bias), None
        return self.quant_method.apply(self, x, self.bias), None


MergedColumnParallelLinear = RowParallelLinear = _Linear


class Inkling2Attention(nn.Module):
    def __init__(self, hidden=64, heads_dim=96, splits=None, q_bias=False, seed=0):
        super().__init__()
        self.split_sizes = [*(splits or INKLING_SPLITS)]
        self.qkvr = MergedColumnParallelLinear(sum(self.split_sizes), hidden, bias=q_bias, seed=seed)
        self.wo_ud = RowParallelLinear(hidden, heads_dim, seed=seed + 1)

    def _project_qkvr(self, x):
        projected, _ = self.qkvr(x)
        return projected.split(self.split_sizes, dim=-1)[:6]

    def _project_sources(self, sources):
        if len({id(sources.q), id(sources.k), id(sources.v)}) == 3:
            q_proj, k_proj, v_proj = (
                projected.split(self.split_sizes, dim=-1)[:6]
                for projected in self.qkvr(_stacked(sources))[0].chunk(3)
            )
            q, _, _, r, gate, sinks = q_proj
            return q, k_proj[1], v_proj[2], r, gate, sinks
        q_proj = self._project_qkvr(sources.q)
        k_proj = q_proj if sources.k is sources.q else self._project_qkvr(sources.k)
        v_proj = (
            q_proj
            if sources.v is sources.q
            else k_proj
            if sources.v is sources.k
            else self._project_qkvr(sources.v)
        )
        q, _, _, r, gate, sinks = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sinks
