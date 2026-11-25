#!/usr/bin/env python3
"""Microbenchmark of the draft's vocab and dense reads, stock vs inkling_draftbytes.
Run inside the SGLang container with the server STOPPED (~3 GB GPU, nothing pinned):

  docker run --rm --gpus all -v <repo>/overlay:/opt/inkling-draft:ro \
      -e PYTHONPATH=/opt/inkling-draft lmsysorg/sglang:inkling-2-mini \
      python3 /opt/inkling-draft/bench_draft.py [--quick] [--json out.json]

Every number is a CUDA-graph replay with the L2 flushed before each call
(the flush's own time is subtracted): DRAM-cold, launch overhead excluded.
GB/s counts the bytes the op must move (weights + scales + activations + output).

Sections
  A  LM head (draft base logits), V x 2304, rows M (M = 7 x bs in the draft):
     stock torch.matmul (cuBLAS) | sm12x Triton GEMV (M <= 16) | own unscaled
     kernel (any M) | head8 row / mx (own FP8 kernel) | FlashInfer mm_mxfp8
     (the --speculative-dspark-lm-head-dtype mxfp8 path; reported if it runs)
  B  Markov fold, V x 256, rows M = bs = 1..8: stock torch.addmm | foldk |
     fold8 row / g32 / mx, plus a BN sweep for fold8 row
  C  hot vocab per draft step at N hot ids: base logits on N rows + scatter,
     and one fused hot step (fold + combine + scatter), x7, vs stock
  D  draft layers' linears at M = 8, 16: stock (the image's linear dispatch)
     vs draftmx8 mx / row, with a BN x BK sweep
"""

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from inkling_draftbytes import ops  # noqa: E402
from inkling_draftbytes.quant import bf16_pack, quantize  # noqa: E402
from inkling_draftbytes.timing import gbps, time_ms  # noqa: E402

ROWS = []


def row(section, name, m, ms, nbytes, **extra):
    gb = gbps(nbytes, ms)
    ROWS.append(dict(section=section, name=name, m=m, ms=ms, gbps=gb, **extra))
    print(f"  {section} {name:34s} M={m:<3d} {ms * 1000:9.1f} us  {gb:7.1f} GB/s")


def try_time(fn, **kw):
    try:
        return time_ms(fn, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"    (skipped: {type(e).__name__}: {str(e)[:120]})")
        return None


def section_a(v, h, ms_list):
    print(f"\n== A: LM head {v} x {h} BF16 ({v * h * 2 / 1e6:.0f} MB)")
    w = (torch.randn(v, h, device="cuda") * 0.02).to(torch.bfloat16)
    try:
        from sglang.kernels.ops.gemm import sm12x_bf16_gemv as gm
    except Exception:  # noqa: BLE001
        gm = None
    p_row = quantize(w, "row", 0)
    p_mx = quantize(w, "mx", 0)
    mxfp8 = None
    try:
        from flashinfer import block_scale_interleave, mm_mxfp8, mxfp8_quantize

        wq, wsf = mxfp8_quantize(w, is_sf_swizzled_layout=False)
        wsf = block_scale_interleave(wsf.view(torch.uint8).reshape(v, h // 32).contiguous()).contiguous()
        mxfp8 = (mm_mxfp8, mxfp8_quantize, wq, wsf)
    except Exception as e:  # noqa: BLE001
        print(f"  (flashinfer mm_mxfp8 unavailable: {type(e).__name__}: {str(e)[:100]})")
    for m in ms_list:
        x = torch.randn(m, h, device="cuda").to(torch.bfloat16)
        io = m * h * 2 + m * v * 2
        t = try_time(lambda: torch.matmul(x, w.T))
        if t:
            row("A", "stock torch.matmul", m, t, v * h * 2 + io)
        if gm is not None and m <= 16:
            t = try_time(lambda: gm.sm12x_bf16_gemv(x, w))
            if t:
                row("A", "gemv sm12x (image)", m, t, v * h * 2 + io)
        t = try_time(lambda: ops.w8_linear(x, bf16_pack(w)))
        if t:
            row("A", "own kernel bf16", m, t, v * h * 2 + io)
        for p in (p_row, p_mx):
            t = try_time(lambda: ops.w8_linear(x, p))
            if t:
                row("A", f"head8:{p.label}", m, t, p.nbytes() + io)
        if mxfp8 is not None:
            mm, quant, wq, wsf = mxfp8

            def run():
                hq, hsf = quant(x, is_sf_swizzled_layout=True)
                return mm(hq, wq.t(), hsf, wsf, backend="cute-dsl")

            t = try_time(run)
            if t:
                row("A", "flashinfer mm_mxfp8 (flag path)", m, t, wq.numel() + wsf.numel() + io)
    del w, p_row, p_mx, mxfp8
    torch.cuda.empty_cache()


def section_b(v, r, ms_list, quick):
    print(f"\n== B: Markov fold {v} x {r} ({v * r * 2 / 1e6:.0f} MB BF16)")
    fold = (torch.randn(v, r, device="cuda") * 0.05).to(torch.bfloat16)
    fb = (torch.randn(v, device="cuda") * 0.1).to(torch.bfloat16)
    packs = [quantize(fold, "row", 0), quantize(fold, "group", 32), quantize(fold, "mx", 0)]
    for m in ms_list:
        lat = torch.tanh(torch.randn(m, r, device="cuda")).to(torch.bfloat16)
        io = m * r * 2 + v * 2 + m * v * 2
        t = try_time(lambda: torch.addmm(fb, lat, fold.T))
        if t:
            row("B", "stock torch.addmm", m, t, v * r * 2 + io)
        t = try_time(lambda: ops.w8_linear(lat, bf16_pack(fold), fb, narrow_k=True))
        if t:
            row("B", "foldk (own, BF16)", m, t, v * r * 2 + io)
        for p in packs:
            t = try_time(lambda: ops.w8_linear(lat, p, fb, narrow_k=True))
            if t:
                row("B", f"fold8:{p.label}", m, t, p.nbytes() + io)
        if not quick and m in (1, 4):
            for bn in (32, 64, 128):
                for bk in (64, 128, 256):
                    for warps in (4, 8):
                        t = try_time(lambda: ops.w8_linear(lat, packs[0], fb, narrow_k=True,
                                                           bn=bn, bk=bk, num_warps=warps))
                        if t:
                            row("B", f"fold8:row BN={bn} BK={bk} w={warps}", m, t,
                                packs[0].nbytes() + io)
    del fold, packs
    torch.cuda.empty_cache()


def section_c(v, h, r, limit, softcap, hot_list, bs_list):
    print("\n== C: per draft step, hot vocab vs stock (7 steps of fold, 1 base GEMM)")
    w = (torch.randn(v, h, device="cuda") * 0.02).to(torch.bfloat16)
    fold = (torch.randn(v, r, device="cuda") * 0.05).to(torch.bfloat16)
    fb = (torch.randn(v, device="cuda") * 0.1).to(torch.bfloat16)
    try:
        from sglang.kernels.ops.gemm import sm12x_bf16_gemv as gm
    except Exception:  # noqa: BLE001
        gm = None
    for bs in bs_list:
        m = 7 * bs
        x = torch.randn(m, h, device="cuda").to(torch.bfloat16)
        lat = torch.tanh(torch.randn(bs, r, device="cuda")).to(torch.bfloat16)

        def stock():
            torch.matmul(x, w.T)
            for _ in range(7):
                torch.addmm(fb, lat, fold.T)

        t = try_time(stock)
        if t:
            row("C", "stock base + 7 x addmm", bs, t, v * h * 2 + 7 * v * r * 2)
        for n in hot_list:
            ids = torch.randperm(limit, device="cuda")[:n].sort().values
            ids32 = ids.to(torch.int32)
            wh = w.index_select(0, ids)
            for fold8 in (False, True):
                sub = fold.index_select(0, ids)
                pack = quantize(sub, "row", 0) if fold8 else bf16_pack(sub.contiguous())
                fbh = fb.index_select(0, ids)
                out = torch.full((bs, v), float("-inf"), device="cuda", dtype=torch.bfloat16)

                def hot():
                    compact = ops.bf16_gemv(x, wh, 64, gm)
                    full = torch.full((m, v), float("-inf"), device="cuda", dtype=torch.bfloat16)
                    full.index_copy_(1, ids, compact)
                    for _ in range(7):
                        ops.hot_step(lat, pack, fbh, full[:bs], ids32, out,
                                     softcap=softcap, vocab_limit=limit)

                t = try_time(hot)
                if t:
                    row("C", f"hot {n} fold {'fp8' if fold8 else 'bf16'}", bs, t,
                        wh.numel() * 2 + 7 * pack.nbytes(), note="fold may be L2-resident")
                t1 = try_time(lambda: ops.hot_step(lat, pack, fbh, out, ids32, out,
                                                   softcap=softcap, vocab_limit=limit), cold=False)
                if t1:
                    row("C", f"  one hot step {n} {'fp8' if fold8 else 'bf16'} (L2 warm)", bs, t1,
                        pack.nbytes())
    del w, fold
    torch.cuda.empty_cache()


def section_d(h, quick):
    print("\n== D: draft layer linears (per layer; 3 layers in the draft)")
    print("   stock = what UnquantizedLinearMethod picks by default on sm12x: the image's")
    print("   sm12x GEMV at <= 16 rows (cuBLAS F.linear shown for reference)")
    try:
        from sglang.kernels.ops.gemm import sm12x_bf16_gemv as gm
    except Exception:  # noqa: BLE001
        gm = None
    shapes = {"qkv_proj": (5120, h), "o_proj": (h, 3072), "gate_up_proj": (18432, h),
              "down_proj": (h, 9216), "kv_proj_ctx": (2048, h)}
    for name, (n, k) in shapes.items():
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        packs = [quantize(w, "mx", 0), quantize(w, "row", 0)]
        for m in (8, 16):
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            io = m * k * 2 + m * n * 2
            t = try_time(lambda: torch.nn.functional.linear(x, w))
            if t:
                row("D", f"{name} cuBLAS F.linear", m, t, n * k * 2 + io)
            if gm is not None and gm.use_sm12x_bf16_gemv(x, w):
                t = try_time(lambda: gm.sm12x_bf16_gemv(x, w))
                if t:
                    row("D", f"{name} stock sm12x GEMV", m, t, n * k * 2 + io)
            for p in packs:
                t = try_time(lambda: ops.w8_linear(x, p))
                if t:
                    row("D", f"{name} draftmx8:{p.label}", m, t, p.nbytes() + io)
            if not quick and m == 8:
                for bn in (16, 32, 64):
                    for bk in (128, 256, 512):
                        t = try_time(lambda: ops.w8_linear(x, packs[0], bn=bn, bk=bk))
                        if t:
                            row("D", f"{name} mx BN={bn} BK={bk}", m, t, packs[0].nbytes() + io)
        del w, packs
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", type=int, default=201024)
    ap.add_argument("--limit", type=int, default=200058)
    ap.add_argument("--hidden", type=int, default=2304)
    ap.add_argument("--softcap", type=float, default=30.0)
    ap.add_argument("--hot", type=int, nargs="+", default=[16384, 32768, 65536])
    ap.add_argument("--quick", action="store_true", help="skip the tile sweeps")
    ap.add_argument("--sections", default="ABCD")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    print(f"torch {torch.__version__}, {torch.cuda.get_device_name()}, "
          f"cc {torch.cuda.get_device_capability()}")
    h, r = args.hidden, 256
    if "A" in args.sections:
        section_a(args.vocab, h, [1, 2, 4, 7, 8, 14, 16, 21, 28, 56])
    if "B" in args.sections:
        section_b(args.vocab, r, list(range(1, 9)), args.quick)
    if "C" in args.sections:
        section_c(args.vocab, h, r, args.limit, args.softcap, args.hot, [1, 2, 4])
    if "D" in args.sections:
        section_d(h, args.quick)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(ROWS, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
