"""Stand-in of sglang/srt/models/inkling2_mtp.py: a draft with a few BF16 linears that borrows the target head."""
from torch import nn

from sglang.srt.models.dspark import project_through_lm_head
from sglang.srt.models.inkling_common.inkling2_attn import _Linear


class _DraftLayer(nn.Module):
    def __init__(self, hidden, i):
        super().__init__()
        self.qkv_proj = _Linear(3 * hidden, hidden, seed=100 + i)
        self.o_proj = _Linear(hidden, hidden, seed=200 + i)
        self.gate_up_proj = _Linear(4 * hidden, hidden, seed=300 + i)
        self.down_proj = _Linear(hidden, 2 * hidden, seed=400 + i)
        self.odd = _Linear(12, hidden, seed=500 + i)  # N not a multiple of 8: pico must skip it


class Inkling2ForConditionalGenerationMTP(nn.Module):
    def __init__(self, hidden=64, layers=1):
        super().__init__()
        self.layers = nn.ModuleList([_DraftLayer(hidden, i) for i in range(layers)])
        self.lm_head = None

    def load_weights(self, weights=()):
        return set()

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        self.lm_head = lm_head

    def compute_base_logits(self, hidden):
        logits = project_through_lm_head(hidden, self.lm_head)
        return logits, None
