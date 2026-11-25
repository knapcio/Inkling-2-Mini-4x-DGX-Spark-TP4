"""TP4 stand-in for the CPU tests: one process per rank, gloo collectives, the image's TP hooks the overlay reads.

install(rank, world, init_method) (called by run_fake_dflash.py when FAKE_TP > 1, before the worker exists):
- torch.distributed gloo group of ``world`` processes;
- ``sglang.srt.runtime_context`` with get_parallel() -> tp_size / pp_size / attn_dp_enabled / tp_group / tp_rank, as
  the image's ParallelContext (inkling_vcap reads it through patches._runtime_context);
- the stand-in target's experts split over the ranks (fake inkling2_moe.EP: local experts + all-reduce, as Marlin
  _forward_source at EP4) and its verify logits vocab-sharded (fake dflash_worker_v2.TP: deferred gather).
With FAKE_NO_DIST=1 no process group is created (for refusals that must fire before any collective).
"""

import sys
import types
from types import SimpleNamespace

import torch
import torch.distributed as dist


class FakeGroup:
    """The GroupCoordinator methods the overlay and the stand-in call."""

    def __init__(self, rank, world, backed=True):
        self.rank_in_group = int(rank)
        self.world_size = int(world)
        self.backed = backed
        self.counts = {"all_gather_into_tensor": 0, "all_reduce": 0, "all_gather": 0}

    def _need(self):
        if not self.backed:
            raise RuntimeError("fake TP group without a process group (FAKE_NO_DIST=1)")

    def all_gather_into_tensor(self, output, input):
        self._need()
        self.counts["all_gather_into_tensor"] += 1
        parts = list(output.view(self.world_size, -1).unbind(0))
        flat = input.reshape(-1).contiguous()
        bufs = [torch.empty_like(flat) for _ in range(self.world_size)]
        dist.all_gather(bufs, flat)
        for dst, src in zip(parts, bufs):
            dst.copy_(src)

    def all_reduce(self, t):
        self._need()
        self.counts["all_reduce"] += 1
        dist.all_reduce(t)
        return t

    def all_gather(self, t, dim=-1):
        self._need()
        self.counts["all_gather"] += 1
        t = t.contiguous()
        bufs = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(bufs, t)
        return torch.cat(bufs, dim=dim)


def install(rank, world, init_method, *, backed=True, attn_dp=False):
    if backed:
        dist.init_process_group("gloo", init_method=init_method, rank=rank, world_size=world)
    group = FakeGroup(rank, world, backed=backed)
    rc = types.ModuleType("sglang.srt.runtime_context")
    par = SimpleNamespace(tp_size=world, pp_size=1, attn_dp_enabled=attn_dp, tp_group=group, tp_rank=rank)
    rc.get_parallel = lambda: par
    sys.modules["sglang.srt.runtime_context"] = rc
    from sglang.srt.models.inkling_common import inkling2_moe
    from sglang.srt.speculative import dflash_worker_v2

    inkling2_moe.EP = group
    dflash_worker_v2.TP = group
    return group
