#!/usr/bin/env python3
"""Tile sweep of inkling_draftbytes' W8 (MXFP8) GEMM for the DFlash2 drafter's 4 linear shapes at 32 / 64 / 192 rows.

  cd overlay && TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/sweep_dfw8.py --draft <DFlash2 drafter dir>

Shapes (layer 0 real weights): qkv 6144x2304, o 2304x4096, gate_up 13824x2304, down 2304x6912. For every launch
config (BN x warps x stages [x ROWS mode above 64 rows]) of the existing _w8_mm_kernel: median us, cold L2 (256 MiB
read before each rep), and whether the output is bit-identical to the package's default launch (the MX kernel keeps
one MMA chain per output whatever the tile, so it should be). Baseline: the serving BF16 linear
(UnquantizedLinearMethod.apply). Prints the best config per (shape, rows) and the 20-linear (5-layer) totals.
"""
import argparse
import itertools
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--rows", default="32,64,192")
    ap.add_argument("--reps", type=int, default=15)
    a = ap.parse_args()
    from safetensors import safe_open
    from inkling_draftbytes import _opts, ops, quant
    from inkling_draftbytes import _triton as T
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    opts = _opts.parse("draftmx8,wide8", environ={})
    ops.configure(opts)
    dev = torch.device("cuda")
    p = "layers.0."
    names = {p + n for n in ("self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                             "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight",
                             "mlp.down_proj.weight")}
    t = {}
    with safe_open(os.path.join(a.draft, "model.safetensors"), "pt") as s:
        for k in names:
            t[k] = s.get_tensor(k)
    shapes = {
        "qkv": torch.cat([t[p + "self_attn.q_proj.weight"], t[p + "self_attn.k_proj.weight"],
                          t[p + "self_attn.v_proj.weight"]]),
        "o": t[p + "self_attn.o_proj.weight"],
        "gate_up": torch.cat([t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]]),
        "down": t[p + "mlp.down_proj.weight"],
    }
    flush = torch.ones(64 << 20, device=dev)
    fo = torch.empty((), device=dev)
    method = UnquantizedLinearMethod.__new__(UnquantizedLinearMethod)

    def cold(fn):
        ts = []
        for _ in range(a.reps):
            torch.sum(flush, 0, out=fo)
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record(); fn(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(ts[2:])

    g = torch.Generator(device=dev).manual_seed(2)
    best_tot = {}
    base_tot = {}
    def_tot = {}
    for name, w in shapes.items():
        w = w.to(dev).contiguous()
        lay = torch.nn.Module()
        lay.weight = torch.nn.Parameter(w, requires_grad=False)
        pack = quant.quantize_mode(w, opts.draftmx8)
        n, k = w.shape
        for m in (int(r) for r in a.rows.split(",")):
            x = (torch.randn((m, k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
            # reference: one launch per 64-row block (ROWS 0, BN 32); wide8's default ROWS 1 launch at > 64 rows needs
            # more shared memory than GB10 has for MX packs
            ref = ops.w8_linear(x, pack, None, torch.bfloat16, bn=32, num_warps=4)
            base = cold(lambda: UnquantizedLinearMethod.apply(method, lay, x))
            dflt = cold(lambda: ops.w8_linear(x, pack, None, torch.bfloat16, bn=32, num_warps=4))
            results = []
            rows_modes = (0, 1) if m > 64 else (0,)
            bk = T._bk(pack, 256 if k <= 2304 else 128)
            for bn, warps, stages, rmode in itertools.product((32, 64, 128), (4, 8), (None, 3), rows_modes):
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                mp = T._mp(m)
                rm, nrb, gy = T._rows(m, mp, rmode)
                extra = {} if stages is None else {"num_stages": stages}

                wt, w_fp8 = T._w_arg(pack)
                sc = T._scale_arg(pack, wt)

                def run():
                    T._w8_mm_kernel[(triton_cdiv(n, bn), gy)](
                        x, wt, sc, x, out, m, n, x.stride(0), out.stride(0), K=k, G=pack.group, BK=bk, MP=mp, BN=bn,
                        SCALE=pack.scale_kind, HAS_BIAS=False, W_FP8=w_fp8, ROWS=rm, NRB=nrb, num_warps=warps,
                        **extra)
                try:
                    run(); torch.cuda.synchronize()
                except Exception as exc:  # out of resources etc.
                    results.append((float("inf"), bn, warps, stages, rmode, f"skip {type(exc).__name__}"))
                    continue
                same = torch.equal(out.view(torch.int16), ref.view(torch.int16))
                results.append((cold(run), bn, warps, stages, rmode, "bit-identical" if same else "DIFFERS"))
            results.sort(key=lambda r: r[0])
            best = results[0]
            log(f"{name:8s} {n}x{k} rows {m:3}: BF16 {base:7.1f} us, w8 default {dflt:7.1f} us, best {best[0]:7.1f} us "
                f"(BN {best[1]}, warps {best[2]}, stages {best[3]}, ROWS {best[4]}, {best[5]}); next: "
                + "; ".join(f"{r[0]:.0f} us BN{r[1]} w{r[2]} s{r[3]} R{r[4]} {r[5]}" for r in results[1:4]))
            best_tot[m] = best_tot.get(m, 0) + best[0]
            base_tot[m] = base_tot.get(m, 0) + base
            def_tot[m] = def_tot.get(m, 0) + dflt
    for m in sorted(best_tot):
        log(f"5 layers x 4 linears, rows {m:3}: BF16 {5 * base_tot[m] / 1000:.3f} ms, w8 default "
            f"{5 * def_tot[m] / 1000:.3f} ms, w8 best {5 * best_tot[m] / 1000:.3f} ms")
    return 0


def triton_cdiv(a, b):
    return (a + b - 1) // b


if __name__ == "__main__":
    sys.exit(main())
