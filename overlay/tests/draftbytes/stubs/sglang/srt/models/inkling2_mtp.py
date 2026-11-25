"""CPU stand-in with the interface the overlay patches (written for the tests;
same call signatures and dtype behaviour as the image's module, small math)."""

import os
import sys

import torch
from torch import nn


def should_apply_lm_head_quant_method(lm_head, quant_method):
    return False


def project_through_lm_head(hidden, lm_head):
    weight = lm_head.weight
    return torch.matmul(hidden.to(weight.dtype), weight.T)


class UnquantizedLinearMethod:
    def apply(self, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)


class StubLinear(nn.Module):
    def __init__(self, n, k):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n, k, dtype=torch.bfloat16) * 0.02, requires_grad=False)
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, x):
        return self.quant_method.apply(self, x, None), None


class StubSelfAttn(nn.Module):
    def __init__(self, layer_id, sliding_window_size):
        super().__init__()
        from sglang.srt.layers.attention.flashattention_backend import RadixAttention

        self.attn = RadixAttention(layer_id, sliding_window_size, window_anchor_at_start=sliding_window_size >= 0)


class StubLayer(nn.Module):
    def __init__(self, hidden, layer_id=0, sliding_window_size=-1):
        super().__init__()
        self.qkv_proj = StubLinear(3 * hidden, hidden)
        self.o_proj = StubLinear(hidden, hidden)
        self.self_attn = StubSelfAttn(layer_id, sliding_window_size)


class LMHead(nn.Module):
    def __init__(self, padded, org, hidden):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(padded, hidden, dtype=torch.bfloat16) * 0.05, requires_grad=False)
        self.org_vocab_size = org
        self.tp_size = 1
        self.quant_method = None


def rnn_gates(proj, state):
    """(state', latent) from the BF16 rnn_proj output (stand-in for the image's Triton rnn_gates)."""
    gate, cand, out = proj.float().chunk(3, -1)
    g = torch.sigmoid(gate)
    new_state = (g * state.float() + (1 - g) * torch.tanh(cand)).to(state.dtype)
    return new_state, torch.tanh(out).to(state.dtype)


class Inkling2MTPHead(nn.Module):
    vocab_shard = None
    step_softcap = None
    step_vocab_limit = None

    def __init__(self, vocab_size, markov_rank, hidden_size):
        super().__init__()
        self.vocab_size, self.markov_rank, self.hidden_size = vocab_size, markov_rank, hidden_size
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank).to(torch.bfloat16)
        self.rnn_proj = nn.Linear(2 * markov_rank + 2 * hidden_size, 3 * markov_rank).to(torch.bfloat16)
        self.hidden_out = nn.Linear(markov_rank, hidden_size).to(torch.bfloat16)
        self._fold_weight = self._fold_bias = self._gather_logits = None

    @torch.no_grad()
    def attach_fold(self, weight, multiplier):
        self._fold_weight = weight.new_empty((weight.shape[0], self.markov_rank))
        self._fold_bias = weight.new_empty((weight.shape[0],))
        self.refresh_fold(weight, multiplier)

    @torch.no_grad()
    def refresh_fold(self, weight, multiplier):
        w = weight.double() @ self.hidden_out.weight.double()
        b = weight.double() @ self.hidden_out.bias.double()
        if multiplier:
            w, b = w / multiplier, b / multiplier
        self._fold_weight.copy_(w)
        self._fold_bias.copy_(b)

    @torch.no_grad()
    def _rnn_step(self, state, prev_tokens, hidden, f_rnn):
        inputs = torch.cat([state, self.markov_w1(prev_tokens.long()), hidden, f_rnn], -1)
        return rnn_gates(self.rnn_proj(inputs), state)

    def init_state(self, hidden_states):
        bs = hidden_states.shape[0]
        return hidden_states.new_zeros(bs, self.markov_rank), self._f_rnn[:bs]

    @torch.no_grad()
    def step_bias(self, prev_tokens, *, hidden_states, state):
        state, f_rnn = state
        state, latent = self._rnn_step(state, prev_tokens, hidden_states, f_rnn)
        bias = torch.addmm(self._fold_bias, latent.to(self._fold_weight.dtype), self._fold_weight.T)
        if self.vocab_shard is None:
            bias = self._gather_logits(bias)
        return bias, (state, f_rnn)

    def step_logits_dtype(self, base_dtype):
        return torch.promote_types(base_dtype, self._fold_weight.dtype)


class Inkling2ForConditionalGenerationMTP(nn.Module):
    def __init__(self, padded=4096, org=4000, limit=3990, rank=64, hidden=128, softcap=30.0):
        super().__init__()
        self.logits_mup_width_multiplier = None
        self.markov_head = Inkling2MTPHead(padded, rank, hidden)
        self.markov_head.step_softcap = softcap
        self.markov_head.step_vocab_limit = limit
        # layer 0 local (window), layer 1 global, like the drafter's local_layer_ids [0, 2] pattern
        self.layers = nn.ModuleList([StubLayer(hidden, 0, 127), StubLayer(hidden, 1, -1)])
        self.lm_head = None
        self._org = org

    def attach_shared_modules(self, *, embed_tokens, lm_head):
        self.lm_head = lm_head
        self.markov_head.attach_fold(lm_head.weight, self.logits_mup_width_multiplier)
        self.markov_head._gather_logits = self._gather_logits

    def _gather_logits(self, logits):
        return logits[..., : int(self.lm_head.org_vocab_size)]

    def compute_base_logits(self, hidden):
        if self.logits_mup_width_multiplier:
            hidden = hidden / self.logits_mup_width_multiplier
        logits = project_through_lm_head(hidden, self.lm_head)
        return self._gather_logits(logits), None


for _name in filter(None, os.environ.get("STUB_DROP", "").split(",")):
    owner, _, attr = _name.rpartition(".")
    delattr(globals()[owner] if owner else sys.modules[__name__], attr)

EntryClass = Inkling2ForConditionalGenerationMTP
