#!/usr/bin/env python3
"""GPU check of lever A (inkling_kernels.qkvr_gemv) against the stock QKVR path.

Runs inside the SGLang container on the Spark. Inkling2 qkvr at TP1: weight
[8600, 2304] BF16 (39.6 MB), split sizes q 3072, k 1024, v 1024, r 384, gate 3072,
sinks 24. For each M it builds three distinct [M, 2304] sources laid out like the
model (adjacent rows of one [3, M, 2304] buffer) and compares:

  stock   : the stacked [3M, 2304] projection as UnquantizedLinearMethod.apply runs
            it on sm12x (sm12x_bf16_gemv when 3M <= 16, else cuBLAS F.linear)
  lever A : inkling_kernels.qkvr_gemv.split_gemv (column split, M rows per source)
  per-src : sm12x_bf16_gemv on each source separately (M <= 16 only); lever A
            must be bit-identical to this with the default tile
  fp64    : exact reference

on the columns each source keeps. Also checks CUDA-graph capture/replay, the
patched Inkling2Attention._project_sources against the stock method on a stub
module (same values where the stock path is the GEMV, same view shapes/strides),
and times each path with an L2 flush before every call (weights cold).

  python3 gpu_check_qkvr.py [--ms 1,2,4,8,16,32,64] [--iters 200]
"""

import argparse
import os
import statistics
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

H, SPLITS = 2304, [3072, 1024, 1024, 384, 3072, 24]
N = sum(SPLITS)


def log(*a):
    print(*a, flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ms", default="1,2,4,8,16,32,64")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--flush-mb", type=int, default=96)
    p.add_argument("--tiles", default="32,256,4;64,256,4;32,128,4;64,128,8;16,256,4",
                   help="lever A tile variants to time (BN,BK,WARPS;...)")
    args = p.parse_args()

    import torch
    import torch.nn.functional as F

    from inkling_kernels import qkvr_gemv
    from sglang.kernels.ops.gemm.sm12x_bf16_gemv import (
        MAX_M,
        sm12x_bf16_gemv,
        use_sm12x_bf16_gemv,
    )

    dev = torch.device("cuda", torch.cuda.current_device())
    prop = torch.cuda.get_device_properties(dev)
    log(f"device: {prop.name} sm_{prop.major}{prop.minor} SMs={prop.multi_processor_count}"
        f"  torch {torch.__version__}  stock GEMV MAX_M={MAX_M}")
    g = torch.Generator(device=dev).manual_seed(0)
    weight = (torch.randn((N, H), device=dev, generator=g) * 0.02).to(torch.bfloat16)
    wbytes = weight.numel() * 2
    k_range, v_range = qkvr_gemv.regions(SPLITS, 32)
    default_cfg = {"bn": 32, "bk": 256, "warps": 4}
    flush = torch.ones(args.flush_mb * (1 << 20) // 4, device=dev)
    sink = torch.empty((), device=dev)

    def stock_linear(x):
        return sm12x_bf16_gemv(x, weight) if use_sm12x_bf16_gemv(x, weight) else F.linear(x, weight)

    def kept(out3, m):
        """(q-chunk cols outside k/v, k-chunk k cols, v-chunk v cols) as one flat vector."""
        q = out3[:m]
        return torch.cat([q[:, : k_range[0]].flatten(), q[:, v_range[1]:].flatten(),
                          out3[m:2 * m, k_range[0]:k_range[1]].flatten(),
                          out3[2 * m:, v_range[0]:v_range[1]].flatten()])

    def err(a, ref):
        d = (a.double() - ref).abs()
        scale = ref.abs()
        big = scale > 1e-2 * ref.pow(2).mean().sqrt()
        return dict(max_abs=float(d.max()),
                    max_rel=float((d[big] / scale[big]).max()) if big.any() else 0.0,
                    norm_rel=float(d.max() / scale.max()))

    def bits_equal(a, b):
        return torch.equal(a.view(torch.int16), b.view(torch.int16))

    def timeit(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        for i in range(args.iters):
            sink.copy_(flush.sum(dtype=torch.float32))
            st[i].record()
            fn()
            en[i].record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) * 1000 for s, e in zip(st, en))

    failures = []
    ms = [int(v) for v in args.ms.split(",")]
    log(f"\n{'M':>3} {'3M':>4} {'stock path':>10} | {'A vs stock':>32} | {'A vs per-src':>12} | "
        f"{'A vs fp64':>10} {'stock vs fp64':>13} | graph")
    for m in ms:
        buf = torch.randn((3, m, H), device=dev, generator=g).to(torch.bfloat16)
        q, k, v = buf.unbind(0)
        stacked = buf.view(3 * m, H)
        stock_out = stock_linear(stacked)
        stock_kind = "gemv" if use_sm12x_bf16_gemv(stacked, weight) else "cuBLAS"
        a_out = qkvr_gemv.split_gemv(stacked, weight, None, m, k_range, v_range, default_cfg)
        ref = (stacked.double() @ weight.double().t())
        torch.cuda.synchronize()
        A, S, R = kept(a_out, m), kept(stock_out, m), kept(ref, m)
        e_as = err(A, S.double())
        n_diff = int((A.view(torch.int16) != S.view(torch.int16)).sum())
        e_a, e_s = err(A, R), err(S, R)
        if m <= MAX_M:
            per = torch.cat([sm12x_bf16_gemv(x, weight) for x in (q, k, v)])
            same = bits_equal(A, kept(per, m))
            per_txt = "bit-equal" if same else "DIFFERS"
            if not same:
                failures.append(f"M={m}: lever A differs from per-source stock GEMV")
        else:
            per_txt = "n/a"
        # CUDA graph capture + replay must reproduce the eager result.
        static_out = torch.empty_like(a_out)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            qkvr_gemv.split_gemv(stacked, weight, None, m, k_range, v_range, default_cfg,
                                 out=static_out)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            qkvr_gemv.split_gemv(stacked, weight, None, m, k_range, v_range, default_cfg,
                                 out=static_out)
        static_out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        g_ok = bits_equal(kept(static_out, m), A)
        if not g_ok:
            failures.append(f"M={m}: CUDA graph replay differs from eager")
        if e_a["max_abs"] > 2 * max(e_s["max_abs"], 1e-6) + 1e-3:
            failures.append(f"M={m}: lever A error {e_a['max_abs']:.3g} >> stock {e_s['max_abs']:.3g}")
        log(f"{m:3d} {3 * m:4d} {stock_kind:>10} | max_abs {e_as['max_abs']:.2e} "
            f"rel {e_as['max_rel']:.2e} diff {n_diff:5d} | {per_txt:>12} | "
            f"{e_a['max_abs']:.2e}   {e_s['max_abs']:.2e}      | {'ok' if g_ok else 'FAIL'}")

    # Patched method vs stock method on a stub attention module.
    log("\nInkling2Attention._project_sources: patched vs stock (stub module)")
    try:
        from sglang.srt.models.inkling_common import inkling2_attn as attn_mod
    except Exception as exc:  # noqa: BLE001
        attn_mod = None
        log(f"  skipped: cannot import inkling2_attn outside the server ({type(exc).__name__}: {exc})")
    if attn_mod is not None:
        stock_method = attn_mod.Inkling2Attention._project_sources
        stock_method = getattr(stock_method, "__wrapped__", stock_method)
        fake_unquant = SimpleNamespace(
            _sm12x_bf16_gemv_eligible=lambda x, w, b=None: use_sm12x_bf16_gemv(x, w, b))
        for max_m in (16,):
            patched = qkvr_gemv.make_project_sources(
                stock_method, attn_mod, fake_unquant, dict(default_cfg, max_m=max_m))
            for m in ms:
                buf = torch.randn((3, m, H), device=dev, generator=g).to(torch.bfloat16)
                for layout in ("adjacent", "separate"):
                    if layout == "adjacent":
                        srcs = attn_mod.AttentionSources(*buf.unbind(0))
                    else:
                        srcs = attn_mod.AttentionSources(*(t.clone() for t in buf.unbind(0)))
                    me = SimpleNamespace(split_sizes=list(SPLITS),
                                         qkvr=_FakeLinear(weight, stock_linear))
                    me._project_qkvr = lambda x, me=me: me.qkvr(x)[0].split(me.split_sizes, -1)[:6]
                    o_stock = stock_method(me, srcs)
                    o_new = patched(me, srcs)
                    torch.cuda.synchronize()
                    layout_ok = all(a.shape == b.shape and a.stride() == b.stride()
                                    for a, b in zip(o_new, o_stock))
                    diffs = [int((a.view(torch.int16) != b.view(torch.int16)).sum())
                             for a, b in zip(o_new, o_stock)]
                    mx = max(float((a.float() - b.float()).abs().max()) for a, b in zip(o_new, o_stock))
                    took = "lever A" if 0 < m <= max_m else "stock"
                    log(f"  M={m:3d} {layout:9s} path={took:8s} views {'same' if layout_ok else 'DIFFER'}"
                        f"  mismatched elems per output {diffs}  max_abs {mx:.2e}")
                    if not layout_ok:
                        failures.append(f"M={m} {layout}: output views differ in shape/stride")
                    if took == "stock" and any(diffs):
                        failures.append(f"M={m} {layout}: fallback path not identical")

    # Timings.
    log(f"\nTimings (median us, L2 flushed; weight {wbytes / 1e6:.1f} MB; GB/s = weight bytes / time)")
    tiles = [tuple(int(x) for x in t.split(",")) for t in args.tiles.split(";") if t]
    hdr = f"{'M':>3} {'stock(3M)':>14} {'3x per-src':>14} " + " ".join(
        f"{'A ' + '/'.join(map(str, t)):>16}" for t in tiles)
    log(hdr)
    for m in ms:
        buf = torch.randn((3, m, H), device=dev, generator=g).to(torch.bfloat16)
        stacked = buf.view(3 * m, H)
        q, k, v = buf.unbind(0)
        t_stock = timeit(lambda: stock_linear(stacked))
        t_per = timeit(lambda: [stock_linear(x) for x in (q, k, v)])
        cells = [f"{t_stock:7.1f} {wbytes / t_stock / 1e3:5.0f}G",
                 f"{t_per:7.1f} {3 * wbytes / t_per / 1e3:5.0f}G"]
        for bn, bk, wp in tiles:
            cfg = {"bn": bn, "bk": bk, "warps": wp}
            rg = qkvr_gemv.regions(SPLITS, bn)
            if rg is None:
                cells.append(f"{'unaligned':>16}")
                continue
            try:
                t_a = timeit(lambda: qkvr_gemv.split_gemv(stacked, weight, None, m, rg[0], rg[1], cfg))
                cells.append(f"{t_a:9.1f} {wbytes / t_a / 1e3:5.0f}G")
            except Exception as exc:  # noqa: BLE001
                cells.append(f"{'err ' + type(exc).__name__:>16}")
        log(f"{m:3d} " + " ".join(f"{c:>14}" for c in cells[:2]) + " " +
            " ".join(f"{c:>16}" for c in cells[2:]))
    log("\nNote: 'stock(3M)' is what serving runs today for 3 distinct sources; lever A runs"
        " at M <= INKLING_QKVR_MAX_M (default 16). A non-default BK changes summation order.")
    if failures:
        log("\nFAILURES:")
        for f in failures:
            log("  " + f)
        sys.exit(1)
    log("\nALL CHECKS PASSED")


class _FakeLinear:
    """Stands in for MergedColumnParallelLinear: (output, bias) like the real layer."""

    def __init__(self, weight, fn):
        self.weight, self.bias, self._fn = weight, None, fn

    def __call__(self, x):
        return self._fn(x), None


if __name__ == "__main__":
    main()
