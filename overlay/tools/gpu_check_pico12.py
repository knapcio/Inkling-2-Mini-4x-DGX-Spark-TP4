#!/usr/bin/env python3
"""GPU check of inkling_pico12 (pico's packed 12-bit BF16 GEMV) on the Spark. Run inside the SGLang container
with the server STOPPED (peak ~3.5 GB of GPU memory, for the head):

  docker run --rm --gpus all --ipc=host -v <repo>/overlay:/opt/inkling-overlay:ro \
      -v <model dir>:/model:ro -v <cache dir>:/root/.cache -e PYTHONPATH=/opt/inkling-overlay \
      --entrypoint python3 lmsysorg/sglang:inkling-2-mini /opt/inkling-overlay/tools/gpu_check_pico12.py

  options: --model /model (default; skipped with a note if absent) --layer 0 --ms 1,8,16 --iters 50 --no-real

What it does:
  1. builds (or reuses) the extension in INKLING_PICO12_CACHE (default /root/.cache/inkling-pico12);
  2. for qkvr [8600 x 2304], wo [2304 x 3072] and the head [201024 x 2304], random (Gaussian, sigma 0.01-0.05
     per row, as an earlier standalone probe) and, if /model is mounted, the real layer-L / unembed weights:
       - codes the weight (encode verifies the decode bit for bit) and reports the coded size;
       - at M = 1, 8, 16 [gated]: packed == pico's raw-BF16 GEMV bit for bit (pico's claim); packed within the
         FP32-sum + rounding bound of an FP64 reference; PDL on == off; CUDA-graph replay == eager;
         [reported]: max |packed - stock| and the share of differing outputs, where stock is what SGLang runs
         at that M (sm12x_bf16_gemv Triton for qkvr / wo, sm12x_fp32_lm_head's kernel for the head), and both
         paths' max error against FP64;
  3. the stacked qkvr case (three distinct [M, 2304] sources, M = 4, 8, 16) through inkling_pico12's stacked
     path [gated]: every kept element equals the per-source packed GEMV bit for bit, the views returned by the
     patched Inkling2Attention._project_sources (real sglang code on a stand-in module) have the stock
     layout, and a CUDA graph of the stacked path replays bit for bit; [reported]: vs stock (cuBLAS on 3M rows)
     and vs inkling_kernels.qkvr_gemv's Triton split GEMV;
  4. times every path with an L2-evicting read sweep (256 MiB) before each call: median us and GB/s of the
     BF16 bytes (the "effective" rate for the packed kernel).
Last line: `pico12 gpu_check result PASS|FAIL (...)`; exit code 0 on PASS, 1 on FAIL, 3 when refused by the
MemAvailable guard.
"""

import argparse
import glob
import os
import statistics
import sys
import time
from types import SimpleNamespace

OVERLAY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if OVERLAY not in sys.path:
    sys.path.insert(0, OVERLAY)

FAILS = []
SPLITS = [3072, 1024, 1024, 384, 3072, 24]  # q, k, v, r, gate, sinks at TP1 (layer 0)
PROJ = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")  # SGLang's stacking order


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--ms", default="1,8,16")
    p.add_argument("--stacked-ms", default="4,8,16")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--flush-mib", type=int, default=256)
    p.add_argument("--min-avail-gib", type=float, default=12.0)
    p.add_argument("--no-real", action="store_true")
    p.add_argument("--no-time", action="store_true")
    args = p.parse_args()

    avail = mem_available_gib()
    log(f"pico12 memguard MemAvailable_GiB={avail:.1f} need_GiB={args.min_avail_gib:.1f}")
    if avail == avail and avail < args.min_avail_gib:
        log("pico12 gpu_check REFUSED: not enough MemAvailable (stop the server first)")
        sys.exit(3)

    import torch
    import torch.nn.functional as F

    import inkling_pico12
    from inkling_pico12 import build, ops

    cfg = inkling_pico12.config_from_env()
    dev = torch.device("cuda", torch.cuda.current_device())
    prop = torch.cuda.get_device_properties(dev)
    log(f"pico12 device={prop.name} cc={prop.major}.{prop.minor} sms={prop.multi_processor_count} "
        f"torch={torch.__version__} cuda={torch.version.cuda}")
    t0 = time.monotonic()
    ext = build.load(cfg["cache_dir"], cfg["std"], verbose=cfg["verbose"])
    log(f"pico12 ext {'reused' if build.INFO['cached'] else 'built'} in {time.monotonic() - t0:.1f} s "
        f"at {build.INFO['build_dir']} max_rows={ext.max_rows}")
    ops.EXT = ext
    ops.SET.wk, ops.SET.pdl = cfg["wk"], cfg["pdl"]
    ops.SET.max_m = ops.SET.stacked_max_m = 16

    from sglang.kernels.ops.gemm import sm12x_bf16_gemv as smg

    try:
        from inkling_kernels import qkvr_gemv as ikq
    except Exception as exc:  # noqa: BLE001
        ikq = None
        log(f"pico12 note inkling_kernels.qkvr_gemv unavailable ({type(exc).__name__}: {exc})")

    flush = torch.empty(args.flush_mib * (1 << 20) // 4, device=dev, dtype=torch.float32).fill_(1.0)
    sink = torch.empty((), device=dev)

    def timeit(fn):
        for _ in range(args.warmup):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
        for i in range(args.iters):
            sink.copy_(flush.sum())  # evicts L2 (24 MB on GB10): weights come from DRAM
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
        """FP64 x @ w^T and sum |x_k w_k| per output, over row chunks of w (the head would need 3.7 GB at once)."""
        xd, xa = x.double(), x.double().abs()
        outs, mags = [], []
        for r0 in range(0, w.shape[0], chunk):
            wd = w[r0 : r0 + chunk].double()
            outs.append(xd @ wd.T)
            mags.append(xa @ wd.abs().T)
        return torch.cat(outs, 1), torch.cat(mags, 1)

    def within_bound(y, ref, mag, k):
        """|y - ref| <= FP32 sum bound (2^-23 sum|terms| (K/16 + 32), as the probe) + half a BF16 ulp if BF16."""
        tol = mag * 2.0**-23 * (k / 16 + 32) + 1e-30
        if y.dtype == torch.bfloat16:
            tol = tol + ref.abs() * 2.0**-8
        err = (y.double() - ref).abs()
        return bool((err <= tol).all()), float((err / tol).max())

    def stock_gemv(x, w, out_dtype):
        """What SGLang runs at M <= 16 on sm12x with the default (auto) BF16 backend."""
        if out_dtype == torch.float32 and x.shape[0] == 1:  # sm12x_fp32_lm_head's one-row branch
            out = torch.empty((1, w.shape[0]), device=dev, dtype=torch.float32)
            return smg.b12x_bf16_gemv(x=x, weight=w, out=out)
        return smg.sm12x_bf16_gemv(x, w, out_dtype=out_dtype)

    # ---- weights -----------------------------------------------------------------------------------------
    def random_weights():
        return {
            "qkvr": (gauss(sum(SPLITS), 2304, 1), list(SPLITS)),
            "wo": (gauss(2304, 3072, 2), None),
            "head": (gauss(201024, 2304, 3), None),
        }

    sets = [("random", random_weights)]  # loaded one set at a time (peak memory)
    real_dir = args.model if (not args.no_real and os.path.isdir(args.model)) else None
    if real_dir:
        sets.append((f"real(layer {args.layer})", lambda: load_real(real_dir, args.layer, dev, torch)))
    else:
        log(f"pico12 note no real weights ({args.model} not mounted or --no-real): random weights only")

    ms = [int(v) for v in args.ms.split(",") if v]
    times = []
    for set_name, make in sets:
        weights = make()
        for name in list(weights):
            w, splits = weights.pop(name)
            n, k = w.shape
            out_dtype = torch.float32 if name == "head" else torch.bfloat16
            pack = ops.encode(w, f"{set_name}.{name}")
            tiled = ext.tile(w)
            undec = ext.check(pack.buf, tiled, n, k)
            gate(f"pico12 shape {set_name} {name}",
                 undec == 0,
                 f"[{n}x{k}] raw_MB={n * k * 2 / 1e6:.2f} packed_MB={pack.nbytes / 1e6:.2f} "
                 f"packed/raw={pack.nbytes / (n * k * 2):.4f} raw_tiles={pack.raw_tiles}/{pack.tiles} mt={pack.mt} "
                 f"undecodable_pieces={undec}")
            for m in ms:
                x = acts(m, k, 100 + m)
                y = torch.full((m, n), float("nan"), device=dev, dtype=out_dtype)
                ops.gemv(x, pack, y)
                yr = torch.full_like(y, float("nan"))
                ext.gemv_raw(x, tiled, n, k, yr, pack.mt, ops.SET.wk, ops.SET.pdl)
                yn = torch.full_like(y, float("nan"))
                ext.gemv(x, pack.buf, n, k, yn, pack.mt, ops.SET.wk, not ops.SET.pdl)
                ys = stock_gemv(x, w, out_dtype)
                torch.cuda.synchronize()
                ref, mag = ref64(x, w)
                ok_b, worst = within_bound(y, ref, mag, k)
                same_raw, same_pdl = bits_equal(y, yr), bits_equal(y, yn)
                g = torch.cuda.CUDAGraph()
                yg = torch.full_like(y, float("nan"))
                side = torch.cuda.Stream()
                side.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(side):
                    ops.gemv(x, pack, yg)
                torch.cuda.current_stream().wait_stream(side)
                with torch.cuda.graph(g):
                    ops.gemv(x, pack, yg)
                yg.fill_(float("nan"))
                g.replay()
                torch.cuda.synchronize()
                same_graph = bits_equal(y, yg)
                e_p = float((y.double() - ref).abs().max())
                e_s = float((ys.double() - ref).abs().max())
                d_ps = float((y.double() - ys.double()).abs().max())
                n_diff = int((y != ys).sum())
                gate(f"pico12 bits {set_name} {name} M={m}",
                     same_raw and same_pdl and same_graph and ok_b and not torch.isnan(y).any(),
                     f"packed_vs_raw={'IDENTICAL' if same_raw else 'DIFFERENT'} "
                     f"pdl_on_vs_off={'IDENTICAL' if same_pdl else 'DIFFERENT'} "
                     f"graph_vs_eager={'IDENTICAL' if same_graph else 'DIFFERENT'} fp64_bound_worst={worst:.3f}")
                log(f"pico12 vs_stock {set_name} {name} M={m} max_abs_packed_vs_stock={d_ps:.3e} "
                    f"differing={n_diff}/{y.numel()} err_vs_fp64 packed={e_p:.3e} stock={e_s:.3e}")
                if not args.no_time and set_name == "random":
                    bytes_ = n * k * 2
                    t_p = timeit(lambda: ops.gemv(x, pack, y))
                    t_r = timeit(lambda: ext.gemv_raw(x, tiled, n, k, yr, pack.mt, ops.SET.wk, ops.SET.pdl))
                    t_s = timeit(lambda: stock_gemv(x, w, out_dtype))
                    t_c = timeit(lambda: (torch.mm(x, w.T, out_dtype=torch.float32) if name == "head"
                                          else F.linear(x, w)))
                    times.append((name, m, t_p, t_r, t_s, t_c, bytes_))
                    log(f"pico12 time {name} M={m} packed_us={t_p:.1f} ({bytes_ / t_p / 1e3:.0f} GB/s eff) "
                        f"pico_raw_us={t_r:.1f} stock_us={t_s:.1f} cublas_us={t_c:.1f} "
                        f"packed/stock={t_p / t_s:.3f}")
            tiled = None
            if name == "qkvr":
                stacked_checks(set_name, w, splits, pack, args, ops, ikq, F, torch, dev, acts, bits_equal,
                               ref64, timeit)
            pack = w = None
            torch.cuda.empty_cache()

    if times:
        log("pico12 summary (random weights, us, L2 cold): name M packed pico_raw stock cublas  packed/stock")
        for name, m, t_p, t_r, t_s, t_c, _ in times:
            log(f"pico12 summary {name:5s} M={m:2d} {t_p:8.1f} {t_r:8.1f} {t_s:8.1f} {t_c:8.1f}  {t_p / t_s:.3f}")
    ok = not FAILS
    log(f"pico12 gpu_check result {'PASS' if ok else 'FAIL'} "
        f"({'all gated checks passed' if ok else 'failed: ' + '; '.join(FAILS[:5])})")
    sys.exit(0 if ok else 1)


def stacked_checks(set_name, w, splits, pack, args, ops, ikq, F, torch, dev, acts, bits_equal, ref64, timeit):
    n, k = w.shape
    lo, hi = ops.kv_bounds(splits)
    kv = ops.encode(w[lo:hi], f"{set_name}.qkvr[kv]")
    plan = ops.QkvrPlan(pack, kv, lo, hi)
    log(f"pico12 stacked {set_name} kv_pack [{lo}:{hi}] packed_MB={kv.nbytes / 1e6:.2f} (extra memory per layer)")
    try:
        from sglang.srt.models.inkling_common import inkling2_attn as attn_mod
    except Exception as exc:  # noqa: BLE001
        attn_mod = None
        log(f"pico12 note cannot import inkling2_attn here ({type(exc).__name__}: {exc}); "
            "the _project_sources layout check uses a copy of the stock code")
    stacked_fn = attn_mod._stacked if attn_mod is not None else _stacked_copy
    stock_method = (getattr(attn_mod.Inkling2Attention._project_sources, "__wrapped__", None)
                    or attn_mod.Inkling2Attention._project_sources) if attn_mod is not None else _project_sources_copy
    from inkling_pico12 import patches

    wrapped = patches.make_project_sources(stock_method, stacked_fn)
    cols_q = [c for c in range(n) if not lo <= c < hi]

    def kept(out3, m):
        return torch.cat([out3[:m, cols_q].flatten(), out3[m : 2 * m, lo : lo + splits[1]].flatten(),
                          out3[2 * m :, lo + splits[1] : hi].flatten()])

    class _Lin:
        def __init__(self):
            self.weight, self.bias = w, None

        def __call__(self, x):  # the stock projection: SGLang's apply at these rows (Triton GEMV <= 16, else cuBLAS)
            return (ops_stock(x), None)

    def ops_stock(x):
        from sglang.kernels.ops.gemm import sm12x_bf16_gemv as smg

        return smg.sm12x_bf16_gemv(x, w) if smg.use_sm12x_bf16_gemv(x, w) else F.linear(x, w)

    for m in [int(v) for v in args.stacked_ms.split(",") if v]:
        buf = acts(3 * m, k, 500 + m).view(3, m, k)
        srcs = SimpleNamespace(q=buf[0], k=buf[1], v=buf[2])
        stacked = buf.view(3 * m, k)
        out = ops.stacked_project(stacked, m, plan)
        per = torch.empty((3 * m, n), device=dev, dtype=torch.bfloat16)
        for s in range(3):
            ops.gemv(buf[s], pack, per[s * m : (s + 1) * m])
        stock = F.linear(stacked, w) if 3 * m > 16 else ops_stock(stacked)
        torch.cuda.synchronize()
        same_per = bits_equal(kept(out, m), kept(per, m))
        ref, _ = ref64(stacked, w)
        e_p = float((kept(out, m).double() - kept(ref, m)).abs().max())
        e_s = float((kept(stock, m).double() - kept(ref, m)).abs().max())
        d_s = float((kept(out, m).double() - kept(stock, m).double()).abs().max())
        extra = ""
        if ikq is not None:
            rg = ikq.regions(splits, 32)
            if rg is not None:
                tri = ikq.split_gemv(stacked, w, None, m, rg[0], rg[1], {"bn": 32, "bk": 256, "warps": 4})
                torch.cuda.synchronize()
                extra = (f" max_abs_vs_qkvr_gemv={float((kept(out, m).double() - kept(tri, m).double()).abs().max()):.3e}"
                         f" qkvr_gemv_err={float((kept(tri, m).double() - kept(ref, m)).abs().max()):.3e}")
        # the patched _project_sources on a stand-in module vs the stock method: layout and values
        me = SimpleNamespace(split_sizes=list(splits), qkvr=_Lin())
        me._project_qkvr = lambda x, me=me: me.qkvr(x)[0].split(me.split_sizes, -1)[:6]
        me.__dict__["_pico12_qkvr"] = plan
        sources = attn_mod.AttentionSources(buf[0], buf[1], buf[2]) if attn_mod is not None else srcs
        o_new, o_stock = wrapped(me, sources), stock_method(me, sources)
        torch.cuda.synchronize()
        layout = all((a.shape, a.stride(), a.storage_offset()) == (b.shape, b.stride(), b.storage_offset())
                     for a, b in zip(o_new, o_stock))
        want = ops.split_like_stock(per, list(splits))
        vals = all(bits_equal(a, b) for a, b in zip(o_new, want))
        # CUDA graph of the stacked path
        g = torch.cuda.CUDAGraph()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            ops.stacked_project(stacked, m, plan)
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(g):
            og = ops.stacked_project(stacked, m, plan)
        og.fill_(float("nan"))
        g.replay()
        torch.cuda.synchronize()
        same_graph = bits_equal(kept(og, m), kept(out, m))
        gate(f"pico12 stacked {set_name} 3x{m}", same_per and layout and vals and same_graph,
             f"kept_vs_per_source={'IDENTICAL' if same_per else 'DIFFERENT'} "
             f"project_sources_views={'STOCK' if layout else 'DIFFER'} project_sources_values="
             f"{'IDENTICAL' if vals else 'DIFFERENT'} graph_vs_eager={'IDENTICAL' if same_graph else 'DIFFERENT'}")
        log(f"pico12 stacked_vs_stock {set_name} 3x{m} stock={'cuBLAS' if 3 * m > 16 else 'gemv'} "
            f"max_abs_vs_stock={d_s:.3e} err_vs_fp64 pico={e_p:.3e} stock={e_s:.3e}{extra}")
        if not args.no_time and set_name == "random":
            t_p = timeit(lambda: ops.stacked_project(stacked, m, plan))
            t_s = timeit(lambda: F.linear(stacked, w) if 3 * m > 16 else ops_stock(stacked))
            t_t = (timeit(lambda: ikq.split_gemv(stacked, w, None, m, rg[0], rg[1],
                                                 {"bn": 32, "bk": 256, "warps": 4}))
                   if ikq is not None and rg is not None else float("nan"))
            log(f"pico12 time stacked 3x{m} pico_us={t_p:.1f} stock_us={t_s:.1f} qkvr_gemv_us={t_t:.1f} "
                f"pico/qkvr_gemv={t_p / t_t:.3f} pico/stock={t_p / t_s:.3f}")
    del kv


def _stacked_copy(sources):
    import torch

    return torch.cat([sources.q, sources.k, sources.v])


def _project_sources_copy(self, sources):
    q_proj, k_proj, v_proj = (
        projected.split(self.split_sizes, dim=-1)[:6] for projected in self.qkvr(_stacked_copy(sources))[0].chunk(3)
    )
    q, _, _, r, gate_, sinks = q_proj
    return q, k_proj[1], v_proj[2], r, gate_, sinks


def load_real(model_dir, layer, dev, torch):
    """Layer `layer`'s stacked qkvr and wo_ud and the unembedding from the checkpoint (exact names and dtypes)."""
    from safetensors import safe_open

    where = {}
    for path in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as h:
            for key in h.keys():
                where.setdefault(key, path)

    def get(name):
        if name not in where:
            raise SystemExit(f"pico12 real weights: {name} not in {model_dir}")
        with safe_open(where[name], framework="pt", device="cpu") as h:
            t = h.get_tensor(name)
        if t.dtype != torch.bfloat16 or t.dim() != 2:
            raise SystemExit(f"pico12 real weights: {name} is {t.dtype} {tuple(t.shape)}, expected a BF16 matrix")
        return t

    base = f"model.llm.layers.{layer}.attn."
    parts = [get(f"{base}{p}.weight") for p in PROJ]
    qkvr = torch.cat(parts).to(dev)
    splits = [t.shape[0] for t in parts]
    wo = get(f"{base}wo_ud.weight").to(dev)
    head = get("model.llm.unembed.weight").to(dev)
    print(f"pico12 real weights layer={layer} qkvr={tuple(qkvr.shape)} splits={splits} wo={tuple(wo.shape)} "
          f"head={tuple(head.shape)}", flush=True)
    return {"qkvr": (qkvr, splits), "wo": (wo, None), "head": (head, None)}


if __name__ == "__main__":
    main()
