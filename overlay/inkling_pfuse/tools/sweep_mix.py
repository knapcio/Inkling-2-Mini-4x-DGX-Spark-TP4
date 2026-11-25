#!/usr/bin/env python3
"""Sweep the _loop_mix_kernel launch (BM, BN, warps, stages) at T rows; outputs must stay bit-equal to the default."""
import collections
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
D = 2304


def main():
    from inkling_pfuse import mix

    T = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    g = torch.Generator(device="cuda").manual_seed(0)
    bf = lambda *s: torch.randn(*s, device="cuda", generator=g).to(torch.bfloat16)  # noqa: E731
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
        "attn norm": lambda: mix.loop_mix(src, None, q1, True, bank, q3, down, gain, forget, norm_weights=w),
        "attn fp32": lambda: mix.loop_mix(src, None, q1, True, bank, q3, down, gain, forget),
        "mlp fp32": lambda: mix.loop_mix(src, add, q1, True, bank, q1, down[:1], gain[:1], forget[:1]),
    }
    configs = [(32, 64, 8, 2), (32, 64, 4, 2), (32, 64, 4, 3), (32, 64, 8, 3), (16, 64, 4, 2), (64, 64, 8, 2),
               (32, 128, 8, 2), (16, 128, 4, 2), (64, 32, 4, 2)]
    with torch.no_grad():
        for name, fn in cases.items():
            mix.MIX.update(per_ctrl=False, split_down=False)
            ref = fn()
            mix.MIX.update(per_ctrl=True, split_down=True)
            for bm, bn, wp, st in configs:
                mix.MIX.update(bm1=bm, bn1=bn, warps1=wp, stages1=st)
                try:
                    out = fn()
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    print(f"{name:10s} bm={bm} bn={bn} w={wp} s={st}: {type(exc).__name__}", flush=True)
                    continue
                same = torch.equal(out, ref)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    for _ in range(8):
                        flush.sum()
                        fn()
                    torch.cuda.synchronize()
                ts = sorted(e.device_time_total for e in prof.events()
                            if e.device_type == torch.autograd.DeviceType.CUDA and "_loop_mix" in e.name)
                tall = sum(e.device_time_total for e in prof.events()
                           if e.device_type == torch.autograd.DeviceType.CUDA and "flush" not in e.name
                           and "reduce" not in e.name.lower()) / 8
                print(f"{name:10s} bm={bm:3d} bn={bn:3d} w={wp} s={st}: mix {ts[len(ts)//2]:8.1f} us  all {tall:8.1f} us  "
                      f"{'bit-equal' if same else 'DIFFERS'}", flush=True)


if __name__ == "__main__":
    main()
