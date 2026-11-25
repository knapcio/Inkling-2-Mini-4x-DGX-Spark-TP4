"""CPU stand-in: the patched names/signatures with reference greedy semantics."""

import torch

from inkling_vcap import logic


class _Accept:
    def __init__(self, correct_len, bonus, cap_trim_lens):
        self.correct_len = correct_len
        self.bonus = bonus
        self.cap_trim_lens = cap_trim_lens
        self.commit_lens = (correct_len + 1).to(torch.int32)


def accept_draft_tokens(
    *, candidates, target_logits, draft_block=None, sampling_info=None,
    draft_input=None, gamma=0, verify_num_draft_tokens=0, cutoff_layout=None,
    block_verification=False, rejection_noise=None,
):
    bs, stride = candidates.shape
    predict = target_logits.argmax(-1).view(bs, stride)
    lens = None if cutoff_layout is None else cutoff_layout.verify_lens
    correct, bonus = logic.greedy_accept(candidates, predict, None)
    trimmed = torch.zeros_like(correct)
    if lens is not None:
        correct, trimmed = logic.cap_correct_len(correct, lens)
        bonus = predict[torch.arange(bs), correct]
    return correct, bonus, trimmed


class StaticIO:
    def __init__(self, candidates, logits, seq_lens):
        self.candidates = candidates
        self.logits = logits
        self.seq_lens = seq_lens


class DsparkVerifyEpilogue:
    def __init__(self, max_bs, stride):
        self.max_bs = max_bs
        self.stride = stride
        self.simulate_fold = False
        self.static_io = {}
        self.verify_lens_buf = torch.zeros(max_bs, dtype=torch.int64)
        self._state = None
        self.last = None

    def begin_static_step(self, bs, armed):
        if self._state == (bs, armed):
            return
        self.verify_lens_buf[:bs].fill_(self.stride)
        self.verify_lens_buf[bs:].zero_()
        self._state = (bs, armed)

    def _static_epilogue(self, io):
        self.static_io[io.candidates.shape[0]] = io
        return self.run_static(io, accept=self._accept_static_greedy, gate=1)

    def run_static(self, io, *, accept, gate, kv_projected=False):
        bs = io.candidates.shape[0]
        verify_lens = self.verify_lens_buf[:bs]
        gated = accept(io, verify_lens=verify_lens, gate=gate, gated=True)
        return gated

    def _accept_static_greedy(self, io, *, verify_lens, gate, gated):
        # Uncapped greedy accept (the engine's static path never caps).
        commit = self._accept(candidates=io.candidates, logits=io.logits,
                              draft_tokens=io.candidates[:, 1:], seq_lens=io.seq_lens)
        return torch.minimum(commit, verify_lens.to(torch.int32)) * gate if gated else None

    def _accept(self, *, candidates, logits, draft_tokens, seq_lens,
                cutoff_verify_lens=None):
        bs, stride = candidates.shape
        predict = logits.argmax(-1).view(bs, stride)
        correct, bonus = logic.greedy_accept(candidates, predict, cutoff_verify_lens)
        self.last = _Accept(correct, bonus, torch.zeros_like(correct))
        return self.last.commit_lens


class TargetVerifyExecutor:
    def __init__(self, forward_fn):
        self.forward_fn = forward_fn

    def run_non_compact(self, *, batch, draft_input, verify_ids_2d, verify_window,
                        sampling_info):
        return self.forward_fn(verify_ids_2d)
