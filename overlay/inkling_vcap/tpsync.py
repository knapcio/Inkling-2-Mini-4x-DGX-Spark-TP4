"""Rank-invariant live lengths for the DFLASH verify cap at TP > 1 (INKLING_VERIFY_CAP_TP_SYNC).

Under tensor / expert parallelism every rank runs the same decode step on the same batch, and every rank plans its
own verify cap from its own copy of the walked-row features (dflash.DflashFeatures). The live lengths decide the
accept (accept lengths, commit lengths, the KV / sequence bookkeeping and the next step's inputs), so ranks that
disagree on one length run different steps from then on: the TP collectives of the following forwards would pair up
different work, which deadlocks or corrupts all four machines. Every decision that feeds the accept therefore has to
be identical on every rank.

Why the features are identical on every rank already [argued from the image code, not measured]: the selector reads
candidate ids / unary logits that candidate_topk all-gathers from the vocab shards (bit copies, then the same
torch.topk on the same tensor), the replicated predecessor / successor tables, and the draft hidden states; the
draft's TP all-reduces return the same bits on every rank (NCCL ring all-reduce; RoCEnante's one-shot kernel sums in a
fixed rank order, README B3). The walk, the features and the policy lookup are deterministic functions of those.
The stock engine relies on the same property for the drafted tokens themselves (SGLANG_SPEC_TP_SYNC defaults to the
startup memory probes only). A rank-local event can still break it: a feature write that fails on one rank only
(DflashFeatures.failed), a kernel with a rank-dependent reduction order. So by default the lengths are synchronised:

  gather (default)  all-gather the [bs] int64 live lengths over the TP group and let every rank take rank 0's row
                    (one tiny collective per capped step, eager, between the draft graph and the target verify graph;
                    never inside a CUDA-graph capture). Identical by construction.
  check             gather, then compare the rows on the host (one device sync per capped step) and count / log
                    disagreements; rank 0's row is used, as with gather. For validation runs only.
There is no mode without the collective: an adaptive cap at TP > 1 always synchronises (validated at attach; the
variable is ignored at TP1 and for fixed:K, which never gathers).

Whether a step issues the collective is decided on the host from rank-invariant inputs only (configuration, batch
size, graph capacity, the policy's band price, the batch's custom-logit-processor flag): see dflash.plan_cap. Those
inputs are the same on every rank only if the ranks run the same cap settings: check_settings() all-gathers a digest
of the effective settings once at attach and refuses to start, on every rank alike, when they differ. A rank whose own
proposal fails (missing features, an exception while computing it) still takes part, proposing full width, so the
collective count never differs (dfcap.plan).
"""

from __future__ import annotations

import hashlib
import json

import torch

MODES = ("gather", "check")


def settings_digest(settings: dict) -> torch.Tensor:
    """[4] int64 of the sha256 of a canonical JSON of ``settings`` (str / int / float / bool / None values)."""
    raw = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).digest()
    return torch.tensor([int.from_bytes(raw[i:i + 8], "little", signed=True) for i in range(0, 32, 8)],
                        dtype=torch.int64)


def check_settings(group, settings: dict, device) -> None:
    """All-gather the settings digest over the TP group; raise on every rank alike when any rank differs (every rank
    sees the same gathered rows, so either all ranks start or all refuse)."""
    world = int(group.world_size)
    mine = settings_digest(settings).to(device)
    out = torch.empty(world * 4, dtype=torch.int64, device=device)
    group.all_gather_into_tensor(out, mine)
    rows = out.view(world, 4).cpu()
    bad = [r for r in range(world) if not torch.equal(rows[r], rows[0])]
    if bad:
        raise RuntimeError(f"verify-cap settings differ across TP ranks (ranks {bad} differ from rank 0): refusing to "
                           f"start; this rank's settings: {json.dumps(settings, sort_keys=True)}")


class LiveLenSync:
    """Callable: rewrites a contiguous int64 [bs] view of the live lengths in place with TP rank 0's values."""

    def __init__(self, group, mode: str, capacity: int, device, log=None) -> None:
        if mode not in MODES:
            raise ValueError(f"tp sync mode must be one of {MODES}, got {mode!r}")
        self.group = group
        self.world = int(group.world_size)
        self.rank = int(getattr(group, "rank_in_group", 0))
        self.mode = mode
        self.capacity = int(capacity)
        self.buf = torch.empty(self.world * self.capacity, dtype=torch.int64, device=device)
        self.calls = 0
        self.mismatch_steps = 0
        self.mismatch_rows = 0
        self._log = log

    def __call__(self, lens: torch.Tensor) -> None:
        self.calls += 1
        if self.world <= 1:
            return
        if lens.dtype != torch.int64 or lens.dim() != 1 or not lens.is_contiguous():
            raise RuntimeError(f"live-length sync needs a contiguous int64 [bs] tensor, got {lens.dtype} "
                               f"{tuple(lens.shape)}")
        bs = int(lens.numel())
        if bs > self.capacity:
            raise RuntimeError(f"live-length sync: bs {bs} above the capacity {self.capacity}")
        if lens.is_cuda and torch.cuda.is_current_stream_capturing():
            # The cap is planned eagerly between the draft and the verify graphs; a capture here would freeze one
            # step's lengths into a graph.
            raise RuntimeError("live-length sync inside a CUDA-graph capture")
        out = self.buf[: self.world * bs]
        self.group.all_gather_into_tensor(out, lens)
        rows = out.view(self.world, bs)
        if self.mode == "check":
            bad = (rows != rows[:1]).any(dim=0)
            n = int(bad.sum().item())  # host sync (check mode only)
            if n:
                self.mismatch_steps += 1
                self.mismatch_rows += n
                if self._log is not None and self.mismatch_steps <= 20:
                    self._log(f"tp sync: live lengths differ across ranks on {n}/{bs} requests (step "
                              f"{self.calls}, mismatching steps so far {self.mismatch_steps}): "
                              f"{rows.tolist()} -> rank 0's row used")
        lens.copy_(rows[0])
