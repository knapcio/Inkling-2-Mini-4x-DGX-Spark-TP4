"""CPU stand-in for the DSpark proposer: a toy draft with kept step logits."""

from types import SimpleNamespace

import torch


class DraftBlockProposer:
    def __init__(self, worker, gamma, vocab, keep_prob=0.65):
        self.worker = worker
        self.gamma = gamma
        self.vocab = vocab
        self.keep_prob = keep_prob

    def propose(self, *, batch, draft_input, bs, sampling_info, verify_window=None,
                device=None, target_model=None, anchor_prepared=False):
        tokens = torch.empty((bs, self.gamma), dtype=torch.int64)
        logits = torch.empty((bs, self.gamma, self.vocab), dtype=torch.float32)
        for b in range(bs):
            truth = self.worker.greedy_continuation(b, self.gamma)
            pos = len(self.worker.seqs[b])
            g = torch.Generator().manual_seed(1000 * b + pos)
            for j in range(self.gamma):
                keep = torch.rand((), generator=g).item() < self.keep_prob
                tok = truth[j] if keep else int(torch.randint(self.vocab, (), generator=g))
                tokens[b, j] = tok
                row = torch.randn(self.vocab, generator=g)
                peak = 6.0 * torch.rand((), generator=g).item()
                # The proposed token is the draft argmax, with a random margin.
                row[tok] = max(row[tok].item() + peak, row.max().item() + 1e-3)
                logits[b, j] = row
        block = SimpleNamespace(
            draft_tokens=tokens,
            corrected_logits=logits,
            temperatures=torch.ones(bs),
            greedy_mask=torch.ones(bs, dtype=torch.bool),
        )
        return SimpleNamespace(draft_block=block)
