"""Stand-in of sglang/srt/models/inkling2.py: a model with Inkling2Attention layers and a BF16 LM head."""
import torch
from torch import nn

from sglang.srt.models.inkling_common.inkling2_attn import Inkling2Attention


class _Layer(nn.Module):
    def __init__(self, i, **kw):
        super().__init__()
        self.attn = Inkling2Attention(seed=10 * i, **kw)


class _LMHead(nn.Module):
    def __init__(self, vocab, hidden):
        super().__init__()
        g = torch.Generator().manual_seed(99)
        self.weight = nn.Parameter((torch.randn((vocab, hidden), generator=g) * 0.05).to(torch.bfloat16),
                                   requires_grad=False)


class _LLM(nn.Module):
    def __init__(self, layers, vocab, hidden, **kw):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(i, hidden=hidden, **kw) for i in range(layers)])
        self.lm_head = _LMHead(vocab, hidden)


class Inkling2ForConditionalGeneration(nn.Module):
    def __init__(self, layers=2, vocab=256, hidden=64, **kw):
        super().__init__()
        self.llm = _LLM(layers, vocab, hidden, **kw)
        self.post_loads = 0

    @property
    def lm_head(self):
        return self.llm.lm_head

    def load_weights(self, weights=()):
        for name, value in weights:
            dict(self.named_parameters())[name].data.copy_(value)
        self.post_load_weights()
        return set()

    def post_load_weights(self) -> None:
        self.post_loads += 1


class Inkling2LoopForConditionalGeneration(Inkling2ForConditionalGeneration):
    pass
