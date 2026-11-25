"""CPU stand-in for the draft step loop (same signatures and rounding order as
the image: base + bias in FP32, round to the step dtype, softcap, round, vocab
limit; argmax at temperature 0, Gumbel-max otherwise)."""

from typing import NamedTuple

import torch


class DraftNoise(NamedTuple):
    key: int
    offset: torch.Tensor


class MarkovSampleStep:
    @classmethod
    def execute(cls, *, base_logits, bias, temperatures, key, offset, step, step_dtype,
                logits_out=None, has_sampled_rows=None, softcap=None, vocab_limit=None, **_):
        assert bias.shape[0] == base_logits.shape[0] and bias.stride(1) == 1
        width = bias.shape[1]
        value = base_logits.float().clone()
        value[:, :width] += bias.float()
        value = value.to(step_dtype).float()
        if softcap is not None:
            value = (torch.tanh(value / softcap) * softcap).to(step_dtype).float()
        if vocab_limit is not None:
            cols = torch.arange(value.shape[1])
            value[:, cols >= vocab_limit] = float("-inf")
        if logits_out is not None:
            logits_out.copy_(value)
        keys = value.clone()
        if has_sampled_rows is not False:
            gen = torch.Generator().manual_seed(int(key) + 7919 * int(step) + int(offset.item()))
            u = torch.rand(value.shape, generator=gen).clamp_(1e-12, 1 - 1e-7)
            gumbel = -torch.log(-torch.log(u))
            t = temperatures.view(-1, 1)
            sampled = t != 0
            keys = torch.where(sampled, value / torch.where(sampled, t, torch.ones_like(t)) + gumbel, value)
        keys[:, width:] = float("-inf")
        return keys.argmax(dim=1)


def _draft_step(markov_head, prev_tokens, base_logits, *, hidden_states, state,
                temperatures, noise, step, logits_out, sharded_accept):
    bias, state = markov_head.step_bias(prev_tokens, hidden_states=hidden_states, state=state)
    tokens = MarkovSampleStep.execute(
        base_logits=base_logits, bias=bias, temperatures=temperatures, key=noise.key,
        offset=noise.offset, step=step, step_dtype=markov_head.step_logits_dtype(base_logits.dtype),
        logits_out=logits_out, has_sampled_rows=None if logits_out is not None else False,
        softcap=markov_head.step_softcap, vocab_limit=markov_head.step_vocab_limit,
    )
    return tokens, state


def draft_markov_block(*, markov_head, base_logits, anchor_tokens, draft_hidden,
                       step_temperatures, noise, logits_out=None, greedy_draft=False):
    bs, gamma = base_logits.shape[:2]
    if not greedy_draft and logits_out is None:
        logits_out = torch.empty((bs, gamma, base_logits.shape[-1]),
                                 dtype=markov_head.step_logits_dtype(base_logits.dtype))
    state = markov_head.init_state(draft_hidden)
    prev = anchor_tokens.long()
    out = []
    for step in range(gamma):
        prev, state = _draft_step(
            markov_head, prev, base_logits[:, step], hidden_states=draft_hidden[:, step],
            state=state, temperatures=step_temperatures, noise=noise, step=step,
            logits_out=None if greedy_draft else logits_out[:, step], sharded_accept=False,
        )
        out.append(prev)
    return torch.stack(out, 1), (None if greedy_draft else logits_out)


class DraftParams(NamedTuple):
    temperatures: torch.Tensor
    greedy_mask: torch.Tensor
    step_temperatures: torch.Tensor


def resolve_draft_params(*, sampling_info, bs, device=None, out=None):
    """Same arithmetic as the image's resolve_draft_params."""
    if out is None:
        out = DraftParams(
            temperatures=torch.empty(bs, dtype=torch.float32),
            greedy_mask=torch.empty(bs, dtype=torch.bool),
            step_temperatures=torch.empty(bs, dtype=torch.float32),
        )
    temperatures, greedy_mask, step_temperatures = (buf[:bs] for buf in out)
    if sampling_info is None:
        temperatures.fill_(1.0)
        greedy_mask.fill_(True)
        step_temperatures.zero_()
    else:
        torch.clamp(sampling_info.temperatures.view(-1)[:bs].to(torch.float32), min=1e-5,
                    out=temperatures)
        torch.le(sampling_info.top_ks.view(-1)[:bs], 1, out=greedy_mask)
        torch.where(greedy_mask, torch.zeros(()), temperatures, out=step_temperatures)
    return DraftParams(temperatures, greedy_mask, step_temperatures)
