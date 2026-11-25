#!/usr/bin/env python3
"""Microbench: FlashInfer's stock b12x MXFP8 GEMM (mm_mxfp8, backend "b12x", SM12x) for the DFlash2 drafter's linears.

    cd overlay && python3 tools/bench_mx8w.py --draft /work/df2 [--hidden /work/hid] [--rows 8,16,24,32,64,128,192]

Per drafter step the 5 decoder layers run qkv [6144,2304], o [2304,4096], gate_up [13824,2304], down [2304,6912] and
two conv kernel projections [576,2304] (real weights, all 5 layers, so the cold-L2 totals stream every byte). Eager
per step: the context fc [2304,9216] and the stacked context K/V projection [10240,2304]. Candidate plane: 64K hot
rows x 2304 (random weights; the serving hot8 path is inkling_dfbytes' FP8 row-scaled Triton W8 kernel).

For every row count: one CUDA graph per variant over the 30 layer linears, cold L2 (256 MiB read before each replay):
  bf16      F.linear (cuBLAS; what the serving BF16 linear calls)
  mx8       the current mx8 path (inkling_draftbytes W8: CUDA GEMV <= 16 rows, Triton above), if importable
  b12x      per linear: flashinfer.mxfp8_quantize(x) + mm_mxfp8(..., backend="b12x") with the default tile plan
  b12x*     as b12x with the best tactic per (shape, rows) from a per-linear sweep (GEMM only, quant timed apart)
Accuracy: relative Frobenius error vs BF16 per linear (max over linears), b12x (W8A8) vs mx8 (W8A16), on random
N(0, 0.5) rows and, with --hidden, on captured target post-norm rows (a proxy for normed drafter activations).
Kill rule (dflev plan): 20+10-linear total speedup b12x* (incl. quant) vs bf16 < 1.3x at 32..192 rows.
"""
import argparse
import glob
import os
import statistics
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def log(*a):
    print(*a, flush=True)


def graph(fn):
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        fn()
        fn()
    torch.cuda.current_stream().wait_stream(s_)
    torch.cuda.synchronize()
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    return gr, out


def cold(gr, flush, fo, reps=15):
    ts = []
    for _ in range(reps):
        torch.sum(flush, 0, out=fo)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts[2:])


def relerr(a, b):
    return float((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-30))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", required=True)
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--rows", default="8,16,24,32,64,128,192")
    ap.add_argument("--no-sweep", action="store_true")
    a = ap.parse_args()
    from safetensors import safe_open
    import flashinfer
    from flashinfer import mm_mxfp8, mxfp8_quantize
    from flashinfer.autotuner import autotune
    from flashinfer.gemm.gemm_mm_mxfp8_cute_dsl import _b12x_gemm_mxfp8_runner

    dev = torch.device("cuda")
    cap = torch.cuda.get_device_capability(dev)
    log(f"flashinfer {flashinfer.__version__}, torch {torch.__version__}, cuda {torch.version.cuda}, sm {cap}, "
        f"SMs {torch.cuda.get_device_properties(dev).multi_processor_count}")
    flush = torch.ones(64 << 20, device=dev)
    fo = torch.empty((), device=dev)
    g = torch.Generator(device=dev).manual_seed(3)

    # ---- weights
    names = {}
    for i in range(5):
        p = f"layers.{i}."
        names[f"L{i}.qkv"] = [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight"]
        names[f"L{i}.o"] = [p + "self_attn.o_proj.weight"]
        names[f"L{i}.gate_up"] = [p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight"]
        names[f"L{i}.down"] = [p + "mlp.down_proj.weight"]
        names[f"L{i}.aconv"] = [p + "attention_conv.kernel_projection.weight"]
        names[f"L{i}.mconv"] = [p + "mlp_conv.kernel_projection.weight"]
    names["ctx.fc"] = ["fc.weight"]
    names["ctx.kv"] = [f"layers.{i}.self_attn.{t}_proj.weight" for i in range(5) for t in ("k", "v")]
    want = {n for v in names.values() for n in v}
    raw = {}
    with safe_open(os.path.join(a.draft, "model.safetensors"), "pt") as s:
        for k in s.keys():
            if k in want:
                raw[k] = s.get_tensor(k)
    W = {key: torch.cat([raw[n] for n in parts]).to(dev).contiguous() for key, parts in names.items()}
    del raw
    W["hot"] = (torch.randn((65536, 2304), device=dev, generator=g) * 0.02).to(torch.bfloat16)
    layer_keys = [k for k in W if k.startswith("L")]
    log(f"layer linears {len(layer_keys)}: BF16 {sum(W[k].numel() * 2 for k in layer_keys) / 1e6:.0f} MB; "
        f"ctx fc+kv {sum(W[k].numel() * 2 for k in ('ctx.fc', 'ctx.kv')) / 1e6:.0f} MB")

    # MXFP8 weights (1D 128x4-swizzled E8M0 scales), as mm_mxfp8's docstring prescribes
    Q = {}
    for k, w in W.items():
        wq, ws = mxfp8_quantize(w, is_sf_swizzled_layout=True)
        Q[k] = (wq, ws)
    log(f"MXFP8 weights: {sum(q[0].numel() + q[1].numel() for k, q in Q.items() if k in layer_keys) / 1e6:.0f} MB "
        f"(layers)")

    # current mx8 path (optional)
    mx8 = None
    try:
        from inkling_draftbytes import _opts, ops, patches as dbp, quant, w8gemv
        from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

        opts = _opts.parse("draftmx8,wide8,w8tile,w8gemv", environ={})
        import dataclasses
        opts = dataclasses.replace(opts, w8_max_m=16)
        dbp.STATE.opts = opts
        ops.configure(opts)
        w8gemv.load()
        Wm = dbp._w8_method_class(UnquantizedLinearMethod)
        method = Wm.__new__(Wm)
        mods = {}
        for k in layer_keys:
            lay = torch.nn.Module()
            lay.weight = torch.nn.Parameter(W[k], requires_grad=False)
            lay._idb_w8 = quant.quantize_mode(W[k], opts.draftmx8)
            lay._idb_w8g = w8gemv.choose_cfg(lay._idb_w8.n, lay._idb_w8.k, 0) if w8gemv.eligible_pack(lay._idb_w8) else 0
            mods[k] = lay
        mx8 = (method, mods)
        log("mx8 path: importable (W8 GEMV <= 16 rows, BF16 above as served)")
    except Exception as exc:  # noqa: BLE001
        log(f"mx8 path: not importable ({type(exc).__name__}: {exc}); skipped")

    runner = _b12x_gemm_mxfp8_runner(cap[0], cap[1], True, torch.bfloat16)

    def b12x_default(x, k):
        xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
        wq, ws = Q[k]
        with autotune(False):
            return mm_mxfp8(xq, wq.t(), xs, ws, out_dtype=torch.bfloat16, backend="b12x")

    hid = None
    if a.hidden:
        import numpy as np
        caps = []
        for path in sorted(glob.glob(os.path.join(a.hidden, "*.npz"))):
            with np.load(path) as c:
                r = c["hidden"]
                caps.append(torch.from_numpy(r.astype(np.int16)).view(torch.bfloat16) if r.dtype.kind in "iu"
                            else torch.from_numpy(r).to(torch.bfloat16))
        hid = torch.cat(caps)[:192].to(dev)
        log(f"captured rows for accuracy: {hid.shape[0]}")

    rows_list = [int(r) for r in a.rows.split(",")]
    summary = []
    best_tac = {}
    with torch.no_grad():
        # ---- accuracy (192 rows; every linear)
        try:
            for label, src in (("random", None), ("captured", hid)):
                if label == "captured" and src is None:
                    continue
                eb = em = 0.0
                for k in layer_keys + ["ctx.fc", "ctx.kv"]:
                    kk = W[k].shape[1]
                    if src is not None and kk == 2304:
                        x = src
                    else:
                        x = (torch.randn((192, kk), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                    ref = F.linear(x, W[k])
                    eb = max(eb, relerr(b12x_default(x, k), ref))
                    if mx8 is not None and k in mx8[1]:
                        # the served mx8 path is the CUDA GEMV at <= 16 rows (the Triton W8 default launch needs more
                        # shared memory than GB10 has at 192 rows): compare on the first 16 rows
                        xs16 = x[:16].contiguous()
                        em = max(em, relerr(mx8[0].apply(mx8[1][k], xs16), F.linear(xs16, W[k])))
                log(f"accuracy ({label} rows): max rel err vs BF16  b12x W8A8 {eb:.3e}   mx8 W8A16 {em:.3e}")
        except Exception:  # noqa: BLE001 - keep the timing sections
            log('accuracy section failed:\n' + traceback.format_exc())

        # ---- serving layout check: reuse the mx8 pack (FP8 [N,K] + E8M0 [N,K/32] linear) with the scales swizzled by
        # flashinfer.block_scale_interleave, instead of a second FP8 copy
        if mx8 is not None:
            try:
                from flashinfer import block_scale_interleave
                worst = worst_vs_fi = 0.0
                for k in ("L0.qkv", "L0.o", "L0.gate_up", "L0.down", "L0.aconv"):
                    pack = mx8[1][k]._idb_w8
                    ws2 = block_scale_interleave(pack.scale.contiguous())
                    x = (torch.randn((64, pack.k), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                    xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                    with autotune(False):
                        y = mm_mxfp8(xq, pack.q.t(), xs, ws2, out_dtype=torch.bfloat16, backend="b12x")
                    ref = F.linear(x, W[k])
                    worst = max(worst, relerr(y, ref))
                    worst_vs_fi = max(worst_vs_fi, relerr(y, b12x_default(x, k)))
                log(f"pack reuse (mx8 FP8 bytes + block_scale_interleave scales): max rel err vs BF16 {worst:.3e}, "
                    f"vs flashinfer-quantized weight {worst_vs_fi:.3e}, scale bytes "
                    f"{sum(block_scale_interleave(mx8[1][k]._idb_w8.scale.contiguous()).numel() for k in layer_keys) / 1e6:.1f} MB")
            except Exception:  # noqa: BLE001
                log("pack reuse check failed:\n" + traceback.format_exc())

        # ---- per-linear tactic sweep (GEMM only) + quant cost
        uniq = {}
        for k in layer_keys + ["ctx.fc", "ctx.kv", "hot"]:
            uniq.setdefault(tuple(W[k].shape), k)
        for m in rows_list:
            for shape, k in uniq.items():
                n, kk = shape
                x = (torch.randn((m, kk), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
                wq, ws = Q[k]
                out = torch.empty((m, n), dtype=torch.bfloat16, device=dev)
                ws_buf = torch.empty(32 << 20, dtype=torch.uint8, device=dev)
                inputs = [xq, wq.t(), xs, ws, torch.bfloat16, out, ws_buf]
                gb, _ = graph(lambda: F.linear(x, W[k]))
                tb = cold(gb, flush, fo)
                del gb
                gq, _ = graph(lambda: mxfp8_quantize(x, is_sf_swizzled_layout=True))
                tq = cold(gq, flush, fo)
                del gq
                res = []
                tacs = [None]
                if not a.no_sweep:
                    try:
                        tacs += list(runner.get_valid_tactics(inputs, None))
                    except Exception as exc:  # noqa: BLE001
                        log(f"  get_valid_tactics failed for {shape}: {exc}")
                ref = F.linear(x, W[k])
                for tac in tacs:
                    try:
                        t0 = time.time()
                        runner.forward(inputs, tactic=tac)
                        torch.cuda.synchronize()
                        comp = time.time() - t0
                        e = relerr(out, ref)
                        gt, _ = graph(lambda: runner.forward(inputs, tactic=tac))
                        tt = cold(gt, flush, fo)
                        del gt
                        res.append((tt, tac, e, comp))
                    except Exception as exc:  # noqa: BLE001
                        res.append((float("inf"), tac, float("nan"), 0.0))
                        log(f"  tactic {tac} on {shape} m={m}: {type(exc).__name__}: {str(exc)[:160]}")
                res.sort(key=lambda r: r[0])
                dflt = next((r for r in res if r[1] is None), None)
                best = res[0]
                best_tac[(shape, m)] = best[1]
                gbw = W[k].numel() * 2 / tb / 1e6
                log(f"m {m:3} {k:8s} {n:5}x{kk:<5} bf16 {tb * 1e3:7.1f} us ({gbw:5.0f} GB/s) | quant {tq * 1e3:5.1f} us | "
                    f"b12x default {dflt[0] * 1e3 if dflt else float('nan'):7.1f} us | best {best[0] * 1e3:7.1f} us "
                    f"{best[1]} err {best[2]:.2e} | speedup(best+quant) {tb / (best[0] + tq):4.2f}x")
                del inputs, out, ws_buf

        # ---- in-graph totals over the 30 layer linears (5 layers, distinct weights)
        def b12x_best(x, k, m):
            n = W[k].shape[0]
            xq, xs = mxfp8_quantize(x, is_sf_swizzled_layout=True)
            wq, ws = Q[k]
            out = torch.empty((x.shape[0], n), dtype=torch.bfloat16, device=dev)
            return runner.forward([xq, wq.t(), xs, ws, torch.bfloat16, out, WSB], tactic=best_tac.get(
                (tuple(W[k].shape), m)))

        WSB = torch.empty(32 << 20, dtype=torch.uint8, device=dev)
        for m in rows_list:
            xs_ = {kk: (torch.randn((m, kk), device=dev, generator=g) * 0.5).to(torch.bfloat16)
                   for kk in {W[k].shape[1] for k in layer_keys}}
            row = {"m": m}
            try:
                gb, _ = graph(lambda: [F.linear(xs_[W[k].shape[1]], W[k]) for k in layer_keys])
                row["bf16"] = cold(gb, flush, fo); del gb
                gd, _ = graph(lambda: [b12x_default(xs_[W[k].shape[1]], k) for k in layer_keys])
                row["b12x"] = cold(gd, flush, fo); del gd
                gs, _ = graph(lambda: [b12x_best(xs_[W[k].shape[1]], k, m) for k in layer_keys])
                row["b12x*"] = cold(gs, flush, fo); del gs
                if mx8 is not None:
                    method, mods = mx8
                    gm, _ = graph(lambda: [method.apply(mods[k], xs_[W[k].shape[1]]) for k in layer_keys])
                    row["mx8"] = cold(gm, flush, fo); del gm
            except Exception:  # noqa: BLE001
                log(traceback.format_exc())
            summary.append(row)
            log("TOTAL 30 layer linears, rows %3d: %s" % (m, "  ".join(
                f"{k} {v:6.3f} ms" for k, v in row.items() if k != "m")) + (
                f"  | speedup b12x* vs bf16 {row['bf16'] / row['b12x*']:.2f}x" if "b12x*" in row else ""))
    ok = [r for r in summary if 32 <= r["m"] <= 192 and "b12x*" in r]
    verdict = "PASS" if ok and min(r["bf16"] / r["b12x*"] for r in ok) >= 1.3 else "KILL"
    log(f"VERDICT (min speedup >= 1.3x at 32..192 rows): {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
