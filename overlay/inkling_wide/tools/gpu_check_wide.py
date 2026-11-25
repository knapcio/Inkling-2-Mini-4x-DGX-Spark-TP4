#!/usr/bin/env python3
"""GPU check + microbenchmark of inkling_wide (weight-streaming BF16 GEMM, 17..256 rows) on a Spark. Run inside the
SGLang container with the server STOPPED (peak ~5 GB of GPU memory, for the head):

  docker run --rm --gpus all --ipc=host -v <repo>/overlay:/opt/inkling-overlay:ro -v <model dir>:/model:ro \
      -v <cache dir>:/root/.cache -e PYTHONPATH=/opt/inkling-overlay --entrypoint python3 \
      lmsysorg/sglang:inkling-2-mini /opt/inkling-overlay/inkling_wide/tools/gpu_check_wide.py

  options: --model /model --layer 0 --check-ms ... --ms ... --iters 20 --no-real --no-time --sweep --quick

What it does:
  1. builds (or reuses) inkling_wide's and inkling_pico12's extensions; prints registers / local memory per kernel;
  2. for qkvr [8600 x 2304], wo [2304 x 3072] and the head [201024 x 2304] (real layer-L / unembed weights when
     /model is mounted, else random), codes the weight with inkling_pico12 (pico's bf16x) and [gated]:
       - the wide reader's decode of the coded buffer == the BF16 weight bit for bit;
       - at every --check-ms M: wide on the coded weight == wide on the raw weight bit for bit; PDL on == off;
         CUDA-graph replay == eager; split-K 1 and the default split both within the FP32-sum bound of an FP64
         reference; wide's max error vs FP64 no worse than cuBLAS's (x1.25 slack, BF16 ulps for BF16 outputs);
     [reported]: max abs / max rel error vs FP64 of wide and cuBLAS, max |wide - cuBLAS|;
  3. the stacked qkvr case (three distinct sources) [gated]: every kept element == the per-source plain call bit
     for bit, the views have the stock layout; [reported] vs cuBLAS on the 3M stacked rows;
  4. times (median of --iters, L2 evicted before each call by READING a 256 MiB buffer) cuBLAS, pico (16-row calls of
     its packed GEMV, i.e. ceil(M/16) weight reads), wide packed and wide raw, and the stacked call; GB/s of the
     bytes one pass reads; summed dense time at M=64 (kill line: cuBLAS / wide >= 1.15);
  5. --sweep: split-K sweep per shape and M.
Last line: `wide gpu_check result PASS|FAIL (...)`; exit 0 on PASS, 1 on FAIL, 3 when refused by the memory guard.
"""

import argparse
import glob
import os
import statistics
import sys
import time

OVERLAY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if OVERLAY not in sys.path:
    sys.path.insert(0, OVERLAY)

FAILS = []
SPLITS = [3072, 1024, 1024, 384, 3072, 24]
PROJ = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")


def log(*a):
    print(*a, flush=True)


def gate(name, ok, detail=""):
    log(f"{name} {detail} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)


def mem_available_gib():
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1048576
    except OSError:
        pass
    return float("nan")


def load_real(model_dir, layer, dev, torch):
    from safetensors import safe_open

    where = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as h:
            for key in h.keys():
                where.setdefault(key, path)

    def get(name):
        if name not in where:
            raise SystemExit(f"wide real weights: {name} not in {model_dir}")
        with safe_open(where[name], framework="pt", device="cpu") as h:
            t = h.get_tensor(name)
        if t.dtype != torch.bfloat16 or t.dim() != 2:
            raise SystemExit(f"wide real weights: {name} is {t.dtype} {tuple(t.shape)}")
        return t

    base = f"model.llm.layers.{layer}.attn."
    parts = [get(f"{base}{p}.weight") for p in PROJ]
    qkvr = torch.cat(parts).to(dev)
    splits = [t.shape[0] for t in parts]
    wo = get(f"{base}wo_ud.weight").to(dev)
    head = get("model.llm.unembed.weight").to(dev)
    log(f"wide real weights layer={layer} qkvr={tuple(qkvr.shape)} splits={splits} wo={tuple(wo.shape)} "
        f"head={tuple(head.shape)}")
    return [("qkvr", qkvr, splits), ("wo", wo, None), ("head", head, None)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--check-ms", default="17,24,40,64,65,80,96,100,128,160,192,256")
    p.add_argument("--head-check-ms", default="17,64,256")
    p.add_argument("--ms", default="9,16,17,24,32,48,64,96,128,192,256")
    p.add_argument("--stacked-ms", default="17,24,32,64,85")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--flush-mib", type=int, default=256)
    p.add_argument("--min-avail-gib", type=float, default=12.0)
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--quick", action="store_true", help="first run: few M, no head timing sweep")
    p.add_argument("--shapes", default="qkvr,wo,head")
    args = p.parse_args()
    if args.quick:
        args.check_ms, args.head_check_ms, args.ms, args.stacked_ms = "17,64,256", "17,256", "24,64,256", "24,64"

    avail = mem_available_gib()
    log(f"wide memguard MemAvailable_GiB={avail:.1f} need_GiB={args.min_avail_gib:.1f}")
    if avail == avail and avail < args.min_avail_gib:
        log("wide gpu_check REFUSED: not enough MemAvailable (stop the server first)")
        sys.exit(3)

    import torch
    import torch.nn.functional as F

    import inkling_pico12
    import inkling_wide
    from inkling_pico12 import build as pbuild
    from inkling_pico12 import ops as pops
    from inkling_wide import build, ops

    dev = torch.device("cuda", torch.cuda.current_device())
    prop = torch.cuda.get_device_properties(dev)
    log(f"wide device={prop.name} cc={prop.major}.{prop.minor} sms={prop.multi_processor_count} "
        f"torch={torch.__version__} cuda={torch.version.cuda}")
    wcfg = inkling_wide.config_from_env()
    t0 = time.monotonic()
    ext = build.load(wcfg["cache_dir"], verbose=wcfg["verbose"])
    ext.prepare(dev.index)
    log(f"wide ext {'reused' if build.INFO['cached'] else 'built'} in {time.monotonic() - t0:.1f} s at "
        f"{build.INFO['build_dir']}")
    for fmt in (0, 2):
        for m in (16, 32, 48, 64, 96, 128, 192, 256):
            for mt, mg in (((1, 1), (2, 1)) if m > 128 else ((0, 1), (0, 2)) if m in (96, 128) else ((0, 1),)):
                regs, local, thr = ext.kernel_info(fmt, m, mt, mg)
                log(f"wide kernel fmt={'bf16x' if fmt else 'raw'} M<={m} mt={mt or 2} mg={mg} regs={regs} "
                    f"local_bytes={local} threads={thr}")
                if local > 64:  # a few bytes of header spill (bf16x, M <= 64) are tolerated and reported
                    FAILS.append(f"spill fmt={fmt} M={m} mt={mt} mg={mg}")
    pcfg = inkling_pico12.config_from_env()
    pops.EXT = pbuild.load(pcfg["cache_dir"], pcfg["std"], verbose=False)
    pops.SET.wk, pops.SET.pdl = pcfg["wk"], pcfg["pdl"]
    ops.EXT = ext
    ops.SET.sms = prop.multi_processor_count
    ops.SET.min_m, ops.SET.max_m = 1, 256

    flush = torch.empty(args.flush_mib * (1 << 20) // 4, device=dev, dtype=torch.float32).fill_(1.0)
    sink = torch.empty((), device=dev)

    def timeit(fn, iters=None):
        iters = iters or args.iters
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            sink.copy_(flush.sum())  # reads 256 MiB: L2 (24 MiB) holds no weight afterwards
            st[i].record()
            fn()
            en[i].record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in zip(st, en))

    def gauss(n, k, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        sigma = 0.01 + 0.04 * (torch.arange(n, device=dev, dtype=torch.float32) * 0.618034 % 1.0)
        return (torch.randn((n, k), device=dev, generator=g) * sigma[:, None]).to(torch.bfloat16)

    def acts(m, k, seed):
        g = torch.Generator(device=dev).manual_seed(seed)
        return (torch.rand((m, k), device=dev, generator=g) * 4 - 2).to(torch.bfloat16)

    def bits_equal(a, b):
        view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
        return a.shape == b.shape and torch.equal(a.contiguous().view(view), b.contiguous().view(view))

    def ref64(x, w, chunk=16384):
        xd, xa = x.double(), x.double().abs()
        outs, mags = [], []
        for r0 in range(0, w.shape[0], chunk):
            wd = w[r0 : r0 + chunk].double()
            outs.append(xd @ wd.T)
            mags.append(xa @ wd.abs().T)
        return torch.cat(outs, 1), torch.cat(mags, 1)

    def bound_worst(y, ref, mag, k):
        tol = mag * 2.0**-23 * (k / 16 + 32) + 1e-30
        if y.dtype == torch.bfloat16:
            tol = tol + ref.abs() * 2.0**-8
        return float(((y.double() - ref).abs() / tol).max())

    def errs(y, ref):
        e = (y.double() - ref).abs()
        return float(e.max()), float(e.max() / ref.abs().max().clamp_min(1e-30)), float(e.mean())

    def cublas(x, w, f32):
        return torch.mm(x, w.T, out_dtype=torch.float32) if f32 else F.linear(x, w)

    def pico_multi(x, pack, out):
        for r0 in range(0, x.shape[0], 16):
            r1 = min(x.shape[0], r0 + 16)
            pops.gemv(x[r0:r1], pack, out[r0:r1])
        return out

    if not args.no_real and os.path.isdir(args.model):
        weights = load_real(args.model, args.layer, dev, torch)
    else:
        log("wide note no real weights: random Gaussian weights")
        weights = [("qkvr", gauss(sum(SPLITS), 2304, 1), list(SPLITS)), ("wo", gauss(2304, 3072, 2), None),
                   ("head", gauss(201024, 2304, 3), None)]
    shapes = set(args.shapes.split(","))
    weights = [wt for wt in weights if wt[0] in shapes]

    rows = []  # timing table
    dense64 = {}
    while weights:
        name, w, splits = weights.pop(0)
        n, k = w.shape
        f32 = name == "head"
        out_dtype = torch.float32 if f32 else torch.bfloat16
        kind = {"qkvr": "qkvr", "wo": "wo", "head": "head"}[name]
        pack = pops.encode(w, name)
        dec = ext.decode(pack.buf, n, k)
        same = bits_equal(dec, w)
        del dec
        gate(f"wide decode {name}", same,
             f"[{n}x{k}] raw_MB={n * k * 2 / 1e6:.1f} packed_MB={pack.nbytes / 1e6:.1f} "
             f"ratio={pack.nbytes / (n * k * 2):.4f} raw_tiles={pack.raw_tiles}/{pack.tiles} decoded==weight")
        e_raw = ops.make(w, name, kind, None)
        ops.SET.fmt = "auto"
        e_x = ops.make(w, name, kind, pack)
        alt_max = 2 if f32 else 12  # forced splits below (check: 1 and alt; sweep up to 12, head up to 2)
        ops.ensure_workspace([e_raw, e_x], 1, 256, dev, grow=True, floats=alt_max * 256 * n)
        cms = [int(v) for v in (args.head_check_ms if f32 else args.check_ms).split(",") if v]
        for m in cms:
            x = acts(m, k, 100 + m)
            s_def = ops.choose_split(n, k, m, kind)
            y_x = torch.full((m, n), float("nan"), device=dev, dtype=out_dtype)
            ops.gemm(x, e_x, y_x, m)
            y_r = torch.full_like(y_x, float("nan"))
            ops.gemm(x, e_raw, y_r, m)
            ops.SET.pdl = False
            y_np = torch.full_like(y_x, float("nan"))
            ops.gemm(x, e_x, y_np, m)
            ops.SET.pdl = True
            ops.SET.splitk = {kind: 1}
            y_s1 = torch.full_like(y_x, float("nan"))
            ops.gemm(x, e_x, y_s1, m)
            s_alt = (2 if f32 else 3) if s_def == 1 else 1
            ops.SET.splitk = {kind: s_alt}
            y_s2 = torch.full_like(y_x, float("nan"))
            ops.gemm(x, e_x, y_s2, m)
            ops.SET.splitk = {}
            y_mt = y_x
            if m > 64:  # the bucket's other kernel (MT 2 above 128 rows, two warp groups at 65..128): same bits
                ops.SET.mt, ops.SET.mg = (2, 0) if m > 128 else (0, 2)
                y_mt = torch.full_like(y_x, float("nan"))
                ops.gemm(x, e_x, y_mt, m)
                ops.SET.mt = ops.SET.mg = 0
            pfd0 = ops.SET.pfd
            ops.SET.pfd = 0
            y_pf = torch.full_like(y_x, float("nan"))
            ops.gemm(x, e_x, y_pf, m)
            ops.SET.pfd = pfd0
            yg = torch.full_like(y_x, float("nan"))
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                ops.gemm(x, e_x, yg, m)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                ops.gemm(x, e_x, yg, m)
            yg.fill_(float("nan"))
            g.replay()
            g.replay()  # twice: the split-K counters must be back at zero
            y_c = cublas(x, w, f32)
            torch.cuda.synchronize()
            ref, mag = ref64(x, w)
            wb = max(bound_worst(y_x, ref, mag, k), bound_worst(y_s1, ref, mag, k), bound_worst(y_s2, ref, mag, k))
            ea, er, em = errs(y_x, ref)
            ca, cr, cm = errs(y_c, ref)
            nan = bool(torch.isnan(y_x).any() or torch.isnan(y_s1).any() or torch.isnan(y_s2).any())
            no_worse = ea <= 1.25 * ca + (0 if f32 else 0)
            gate(f"wide bits {name} M={m}",
                 bits_equal(y_x, y_r) and bits_equal(y_x, y_np) and bits_equal(y_x, yg) and bits_equal(y_x, y_mt)
                 and bits_equal(y_x, y_pf)
                 and wb <= 1.0 and not nan,
                 f"split={s_def}/alt{s_alt} packed_vs_raw={'IDENTICAL' if bits_equal(y_x, y_r) else 'DIFFERENT'} "
                 f"pdl_on_vs_off={'IDENTICAL' if bits_equal(y_x, y_np) else 'DIFFERENT'} "
                 f"graph_vs_eager={'IDENTICAL' if bits_equal(y_x, yg) else 'DIFFERENT'} "
                 f"variant={'IDENTICAL' if bits_equal(y_x, y_mt) else 'DIFFERENT'} "
                 f"prefetch_off={'IDENTICAL' if bits_equal(y_x, y_pf) else 'DIFFERENT'} fp64_bound_worst={wb:.3f}")
            gate(f"wide accuracy {name} M={m}", no_worse,
                 f"err_vs_fp64 wide max_abs={ea:.3e} max_rel={er:.3e} mean={em:.3e} | cublas max_abs={ca:.3e} "
                 f"max_rel={cr:.3e} mean={cm:.3e} | max|wide-cublas|={float((y_x.double() - y_c.double()).abs().max()):.3e}")
            del ref, mag, y_x, y_r, y_np, y_s1, y_s2, yg, y_c, g, y_mt, y_pf
            torch.cuda.empty_cache()

        if name == "qkvr" and splits is not None:
            regions = ops.regions_of(splits)
            gate("wide stacked regions", regions is not None, f"{regions}")
            for m in [int(v) for v in args.stacked_ms.split(",") if v]:
                q, kk, v = acts(m, k, 7), acts(m, k, 8), acts(m, k, 9)
                out = ops.stacked_project(q, kk, v, e_x, regions)
                views = ops.split_like_stock(out, splits)
                plain = [ops.linear(src, e_x, torch.bfloat16) for src in (q, kk, v)]
                st_out = cublas(torch.cat([q, kk, v]), w, False)
                st_views = ops.split_like_stock(st_out, splits)
                pq = plain[0].split(splits, dim=-1)
                exp = (pq[0], plain[1].split(splits, dim=-1)[1], plain[2].split(splits, dim=-1)[2], pq[3], pq[4], pq[5])
                same = all(bits_equal(a, b) for a, b in zip(views, exp))
                layout = all((a.shape, a.stride(), a.storage_offset()) == (b.shape, b.stride(), b.storage_offset())
                             for a, b in zip(views, st_views))
                dmax = max(float((a.double() - b.double()).abs().max()) for a, b in zip(views, st_views))
                gate(f"wide stacked M={m}", same and layout,
                     f"kept==per-source {'IDENTICAL' if same else 'DIFFERENT'} layout={'stock' if layout else 'OTHER'} "
                     f"max|wide-cublas_stacked|={dmax:.3e}")
                if not args.no_time:
                    t_c = timeit(lambda: cublas(torch.cat([q, kk, v]), w, False))
                    t_x = timeit(lambda: ops.stacked_project(q, kk, v, e_x, regions))
                    t_r = timeit(lambda: ops.stacked_project(q, kk, v, e_raw, regions))
                    log(f"wide time stacked M={m}x3 cublas(cat+3M)_us={t_c:.1f} wide_x_us={t_x:.1f} "
                        f"({pack.nbytes / t_x / 1e3:.0f} GB/s) wide_raw_us={t_r:.1f} ({n * k * 2 / t_r / 1e3:.0f} GB/s) "
                        f"speedup_x={t_c / t_x:.2f}")
                    rows.append(("qkvr-stacked", m, t_c, float("nan"), t_x, t_r, pack.nbytes, n * k * 2))
                    if m == 64:
                        dense64["stacked"] = (t_c, t_x)

        if not args.no_time:
            for m in [int(v) for v in args.ms.split(",") if v]:
                x = acts(m, k, 200 + m)
                y = torch.empty((m, n), device=dev, dtype=out_dtype)
                yp = torch.empty((m, n), device=dev, dtype=out_dtype)
                it = max(5, args.iters // 4) if f32 else args.iters
                t_c = timeit(lambda: cublas(x, w, f32), it)
                t_p = timeit(lambda: pico_multi(x, pack, yp), it)
                t_x = timeit(lambda: ops.gemm(x, e_x, y, m), it)
                t_r = timeit(lambda: ops.gemm(x, e_raw, y, m), it)
                rows.append((name, m, t_c, t_p, t_x, t_r, pack.nbytes, n * k * 2))
                log(f"wide time {name} M={m} split={ops.choose_split(n, k, m, kind)} cublas_us={t_c:.1f} "
                    f"({n * k * 2 / t_c / 1e3:.0f} GB/s) pico16x{-(-m // 16)}_us={t_p:.1f} wide_x_us={t_x:.1f} "
                    f"({pack.nbytes / t_x / 1e3:.0f} GB/s packed, {n * k * 2 / t_x / 1e3:.0f} eff) wide_raw_us={t_r:.1f} "
                    f"({n * k * 2 / t_r / 1e3:.0f} GB/s) speedup_x={t_c / t_x:.2f}")
                if m == 64 and name in ("qkvr", "wo"):
                    dense64[name] = (t_c, t_x)
            if args.sweep:
                for m in ((24, 64, 96, 128, 256) if name != "head" else (64, 256)):
                    x = acts(m, k, 300 + m)
                    y = torch.empty((m, n), device=dev, dtype=out_dtype)
                    res = []
                    for s in (1, 2, 3, 4, 6, 8, 12):
                        if s > k // 64 or (name == "head" and s > 2):
                            continue
                        ops.SET.splitk = {kind: s}
                        res.append((timeit(lambda: ops.gemm(x, e_x, y, m), max(5, args.iters // 2)), s))
                    ops.SET.splitk = {}
                    log(f"wide sweep {name} M={m} " + " ".join(f"S{s}={t:.1f}" for t, s in res) +
                        f" best=S{min(res)[1]} default=S{ops.choose_split(n, k, m, kind)}")
                    pf = []
                    for d in (0, 1, 2, 4, 8, 16):
                        ops.SET.pfd = d
                        pf.append((timeit(lambda: ops.gemm(x, e_x, y, m), max(5, args.iters // 2)), d))
                        if name != "head":
                            ops.SET.fmt = "raw"
                            t_raw = timeit(lambda: ops.gemm(x, e_raw, y, m), max(5, args.iters // 2))
                            pf[-1] = pf[-1] + (t_raw,)
                    ops.SET.pfd = 4
                    log(f"wide sweep {name} M={m} pfd " + " ".join(
                        f"P{v[1]}={v[0]:.1f}" + (f"/raw{v[2]:.1f}" if len(v) > 2 else "") for v in pf))
                    if m > 64:
                        for mt, mg in ((((1, 0), (2, 0)) if m > 128 else ((0, 1), (0, 2)))):
                            ops.SET.mt, ops.SET.mg = mt, mg
                            t_mt = timeit(lambda: ops.gemm(x, e_x, y, m), max(5, args.iters // 2))
                            log(f"wide sweep {name} M={m} mt={mt} mg={mg} us={t_mt:.1f} "
                                f"split={ops.choose_split(n, k, m, kind)}")
                        ops.SET.mt = ops.SET.mg = 0
        del w, pack, e_raw, e_x
        torch.cuda.empty_cache()

    if rows:
        log("wide summary (us, L2 cold; GB/s of one pass's bytes: packed for wide_x, raw BF16 for the others)")
        log("shape          M   cublas  pico16x   wide_x   wide_raw   GB/s:cublas wide_x wide_raw   wide_x/cublas")
        for name, m, t_c, t_p, t_x, t_r, pb, rb in rows:
            log(f"{name:13s} {m:4d} {t_c:8.1f} {t_p:8.1f} {t_x:8.1f} {t_r:9.1f}   {rb / t_c / 1e3:10.0f} "
                f"{pb / t_x / 1e3:6.0f} {rb / t_r / 1e3:8.0f}   {t_c / t_x:12.2f}")
    if "qkvr" in dense64 and "wo" in dense64:
        c = dense64["qkvr"][0] + dense64["wo"][0]
        x = dense64["qkvr"][1] + dense64["wo"][1]
        log(f"wide dense M=64 (qkvr + wo) cublas_us={c:.1f} wide_x_us={x:.1f} ratio={c / x:.2f} "
            f"kill_line(>=1.15)={'MET' if c / x >= 1.15 else 'MISSED'}")
    log(f"wide gpu_check result {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failed: {', '.join(FAILS[:8])})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
