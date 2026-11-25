"""CPU stand-in for sglang.srt.models.dflash: the candidate plane path the capture hooks (candidate_topk calls the
module-level _project_candidate_logits; compute_candidates applies the unary transform)."""

import torch
from torch import nn


def _project_candidate_logits(hidden, lm_head, *, num_org, use_quant_head=False):
    return (hidden.to(lm_head.weight.dtype) @ lm_head.weight[:num_org].T)


def candidate_topk(hidden, lm_head, k, *, with_partition=False, merged_unary_transform=None):
    num_org = int(lm_head.weight.shape[0])
    logits = _project_candidate_logits(
        hidden, lm_head, num_org=num_org, use_quant_head=False
    )
    vals, ids = torch.topk(logits, k, dim=-1)
    return ids.long(), vals, None


class DFlash2DraftModel(nn.Module):
    def __init__(self, vocab, hidden, k, seed=3, multiplier=0.5, softcap=30.0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.randn(vocab, hidden, generator=g) * 0.7)
        self.k = k
        self.multiplier, self.softcap = multiplier, softcap

    def _transform_unary_logits(self, logits):
        logits = logits.float()
        logits.mul_(self.multiplier)
        logits.div_(self.softcap).tanh_().mul_(self.softcap)
        return logits

    def compute_candidates(self, hidden):
        ids, vals, _ = candidate_topk(
            hidden, self.lm_head, self.k
        )
        return ids, self._transform_unary_logits(vals)
