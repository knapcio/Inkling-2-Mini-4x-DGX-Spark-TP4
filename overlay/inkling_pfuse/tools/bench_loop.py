#!/usr/bin/env python3
"""Per-kernel time of the loopmix / resnorm pipelines at T rows (torch profiler, cold-ish: 128 MiB read between reps).

  python3 tools/bench_loop.py [--tokens 8192] [--reps 10]
"""
import argparse
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = 2304


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=10)
    a = ap.parse_args()
    from inkling_pfuse import mix

    T = a.tokens
    g = torch.Generator(device="cuda").manual_seed(0)

    def bf(*s):
        return torch.randn(*s, device="cuda", generator=g).to(torch.bfloat16)

    src = [bf(T, D) for _ in range(3)]
    add = bf(T, D)
    bank = [bf(T, D) for _ in range(2)]
    q1 = (torch.randn(1, D, device="cuda") * D ** -0.5).contiguous()
    q3 = (torch.randn(3, D, device="cuda") * D ** -0.5).contiguous()
    w = [(torch.rand(D, device="cuda") + 0.5).to(torch.bfloat16) for _ in range(3)]
    down = (torch.randn(3, 64, 2 * D, device="cuda") * 0.02).to(torch.bfloat16)
    gain = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)
    forget = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)
    flush = torch.empty(32 << 20, device="cuda")
    cases = {
        "loop attn local": lambda: mix.loop_mix(src, None, q1, True, bank, q3, down, gain, forget, norm_weights=w),
        "loop attn local onepass": lambda: mix.loop_mix(src, None, q1, True, bank, q3, down, gain, forget,
                                                         norm_weights=w, fuse_norm=False),
        "loop mlp": lambda: mix.loop_mix(src, add, q1, True, bank, q1, down[:1], gain[:1], forget[:1]),
        "resnorm local": lambda: mix.resnorm(src, None, q1, w[:1], 1e-6, True),
        "resnorm mlp+add": lambda: mix.resnorm(src, add, q1, w[:1], 1e-6, True),
    }
    with torch.no_grad():
        for name, fn in cases.items():
            fn()
            torch.cuda.synchronize()
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.reps):
                    flush.sum()
                    fn()
                torch.cuda.synchronize()
            agg = collections.defaultdict(list)
            for e in prof.events():
                if e.device_type == torch.autograd.DeviceType.CUDA:
                    agg[e.name[:60]].append(e.device_time_total if hasattr(e, "device_time_total") else e.cuda_time_total)
            print(f"{name}:")
            for k, v in agg.items():
                if "reduce" in k.lower() or "sum" in k.lower() and "kernel" in k.lower():
                    continue
                v = sorted(v)
                print(f"   {v[len(v)//2]:9.1f} us  x{len(v)/a.reps:.0f}  {k}")


if __name__ == "__main__":
    main()
