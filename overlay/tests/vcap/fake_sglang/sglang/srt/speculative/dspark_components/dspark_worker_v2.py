"""CPU stand-in for the DSpark worker: a toy causal target with Inkling2MoE layers,
per-position caches (the "KV cache"), the static verify step, and greedy accept.

Every per-position value is computed from that position's prefix only, and cached
values are summed with cumsum over the full sequence, so a correct verify cap must
reproduce the uncapped greedy sequence bit for bit."""

from types import SimpleNamespace

import torch
from torch import nn

from sglang.srt.models.inkling_common import inkling2_moe
from sglang.srt.speculative.dspark_components import dspark_verify
from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockProposer

V, H = 40, 16


class _Mode:
    def __init__(self, verify):
        self._verify = verify

    def is_target_verify(self):
        return self._verify


class ToyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.emb = torch.randn(V, H, generator=g, dtype=torch.float64)
        self.unemb = torch.randn(V, H, generator=g, dtype=torch.float64)
        self.layers = nn.ModuleList(inkling2_moe.Inkling2MoE(H, 16, 2, s) for s in (1, 2))
        # layer 1 is applied twice (a "looped" layer), like Inkling's loop core.
        self.schedule = [0, 1, 1]


class DSparkWorkerV2:
    def __init__(self, *, bs, prompts, stride=8, path="eager", gen_tokens=40):
        self._hosts_draft = True
        self.verify_num_draft_tokens = stride
        self.gamma = stride - 1
        self._verify_planner = SimpleNamespace(mode_value="static")
        self._simulate_acc_len = -1.0
        self._greedy_draft = False
        self._sharded_accept = False
        self._draft_sampler = None
        self.device = torch.device("cpu")
        self.model = ToyTarget()
        self.target_worker = SimpleNamespace(model_runner=SimpleNamespace(model=self.model))
        self._verify_epilogue = (
            dspark_verify.DsparkVerifyEpilogue(max_bs=4, stride=stride)
            if path == "epilogue" else None
        )
        self._verify_executor = dspark_verify.TargetVerifyExecutor(self._target_forward)
        self._proposer = DraftBlockProposer(self, self.gamma, V)
        self.seqs = [list(p) for p in prompts]
        self.bs = bs
        self.gen_tokens = gen_tokens
        n_slots = len(self.model.schedule) + 1
        # caches[b][slot] -> [n, H] per-position values of committed positions
        self.caches = [[torch.zeros(0, H, dtype=torch.float64) for _ in range(n_slots)]
                       for _ in range(bs)]
        for b in range(bs):  # "prefill": commit the prompt except its last token
            self._commit(b, torch.tensor(self.seqs[b][:-1]), None, None)
        self.anchors = [s[-1] for s in self.seqs]
        self.verify_rows = 0

    # ---------------------------------------------------------- toy target
    def _forward_rows(self, token_rows, caches, verify):
        """token_rows: [bs, T]; returns logits [bs*T, V] and per-slot values."""
        bs, t = token_rows.shape
        fb = SimpleNamespace(forward_mode=_Mode(verify))
        m = self.model
        xs, idxs, ns = [], [], []
        per_slot = [[m.emb[token_rows[b]]] for b in range(bs)]
        for b in range(bs):
            e = m.emb[token_rows[b]]
            n = caches[b][0].shape[0]
            idx = torch.arange(n, n + t, dtype=torch.float64)[:, None] + 1
            xs.append(torch.tanh(e + torch.cumsum(torch.cat([caches[b][0], e]), 0)[n:] / idx))
            idxs.append(idx)
            ns.append(n)
        x = torch.cat(xs)  # [bs*t, H], request-major
        for slot, layer_id in enumerate(m.schedule, start=1):
            out = m.layers[layer_id](x, fb)
            new = []
            for b in range(bs):
                o = out[b * t:(b + 1) * t]
                per_slot[b].append(o)
                mixed = torch.cumsum(torch.cat([caches[b][slot], o]), 0)[ns[b]:] / idxs[b]
                new.append(x[b * t:(b + 1) * t] + mixed)
            x = torch.cat(new)
        return x @ m.unemb.T, per_slot

    def _target_forward(self, verify_ids_2d):
        self.verify_rows += verify_ids_2d.numel()
        logits, self._pending = self._forward_rows(verify_ids_2d, self.caches, True)
        return logits

    def _commit(self, b, tokens, per_slot, n_rows):
        if per_slot is None:  # prefill path
            if len(tokens) == 0:
                return
            _, per_slot = self._forward_rows(tokens[None], [self.caches[b]], False)
            per_slot = per_slot[0]
            n_rows = len(tokens)
        for slot, vals in enumerate(per_slot):
            self.caches[b][slot] = torch.cat([self.caches[b][slot], vals[:n_rows]])

    def greedy_continuation(self, b, n):
        cache = [c.clone() for c in self.caches[b]]
        tok, out = self.anchors[b], []
        for _ in range(n):
            logits, slots = self._forward_rows(torch.tensor([[tok]]), [cache], False)
            for slot, vals in enumerate(slots[0]):
                cache[slot] = torch.cat([cache[slot], vals])
            tok = int(logits[-1].argmax())
            out.append(tok)
        return out

    # ---------------------------------------------------------- decode step
    def accept_and_finalize(self, verify_ids_2d, logits):
        correct, bonus, trimmed = dspark_verify.accept_draft_tokens(
            candidates=verify_ids_2d, target_logits=logits, cutoff_layout=None)
        return SimpleNamespace(correct_len=correct, bonus=bonus, cap_trim_lens=trimmed,
                               commit_lens=(correct + 1).to(torch.int32))

    def _forward_decode(self, batch=None):
        bs = self.bs
        proposal = self._proposer.propose(batch=None, draft_input=None, bs=bs,
                                          sampling_info=None)
        drafts = proposal.draft_block.draft_tokens
        verify_ids_2d = torch.cat([torch.tensor(self.anchors)[:, None], drafts], 1)
        if self._verify_epilogue is not None:
            self._verify_epilogue.begin_static_step(bs, True)
        logits = self._verify_executor.run_non_compact(
            batch=None, draft_input=None, verify_ids_2d=verify_ids_2d,
            verify_window=None, sampling_info=None)
        if self._verify_epilogue is not None:
            io = dspark_verify.StaticIO(verify_ids_2d, logits, torch.zeros(bs))
            self._verify_epilogue._static_epilogue(io)
            last = self._verify_epilogue.last
            accept = SimpleNamespace(correct_len=last.correct_len, bonus=last.bonus,
                                     cap_trim_lens=last.cap_trim_lens,
                                     commit_lens=last.commit_lens)
        else:
            accept = self.accept_and_finalize(verify_ids_2d, logits)
        for b in range(bs):
            c = int(accept.correct_len[b])
            committed = verify_ids_2d[b, : c + 1].tolist()
            self._commit(b, None, self._pending[b], c + 1)
            self.seqs[b].extend(committed[1:] + [int(accept.bonus[b])])
            self.anchors[b] = int(accept.bonus[b])
        return SimpleNamespace(accept_lens=accept.commit_lens,
                               block_accept_lens=accept.commit_lens + accept.cap_trim_lens)

    def run(self):
        steps = 0
        while min(len(s) for s in self.seqs) < self.gen_tokens:
            self._forward_decode()
            steps += 1
        return [s[: self.gen_tokens] for s in self.seqs], steps
