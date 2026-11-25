#!/usr/bin/env python3
"""Tile sweep for Inkling2's SM120 NVFP4 (and MXFP8) expert grouped GEMMs on GB10.

Runs inside the SGLang container on the Spark. Builds the expert-sorted NVFP4
activations with SGLang's own routing and quantization kernels (route_metadata,
nvfp4_routed_quantize), random NVFP4 expert weights for all 128 experts with
swizzled E4M3 block scales, and calls FlashInfer's group_gemm_nvfp4_nt_groupwise
exactly as sglang/kernels/ops/gemm/inkling2_grouped_sm120.py does, once per
(swap_ab, tile_n, tile_k) candidate. Each candidate's output is compared bitwise
with the stock tile choice (_tiles). Timing: CUDA events around each call, an L2
flush (reading a buffer larger than L2) before every call so the expert weights are
cold as in serving, median of many iterations.

Shapes (sglang/srt/models/inkling_common/inkling2_moe.py, model config.json):
hidden 2304, expert intermediate 1152, 128 experts, top-8.
  w13: N = 2 * 1152 = 2304, K = 2304   (weight [128, 2304, 1152] uint8)
  w2 : N = 2304,            K = 1152   (weight [128, 2304, 576] uint8)
Layers 0-19 NVFP4, 20-23 MXFP8 (moe_gemm_mxfp8_nt_groupwise, --mxfp8).

Scenarios "T:U": T MoE tokens (ids.shape[0]; 8 = c1 verify, 64 = c8 verify) with U
distinct experts over the T * 8 routing picks (token t takes pool[(8t + j) % U]).

Examples:
  python3 bench_moe_tiles.py                         # default grid, NVFP4
  python3 bench_moe_tiles.py --mxfp8                 # also the MXFP8 GEMM
  python3 bench_moe_tiles.py --isolate               # one subprocess per tile config
  python3 bench_moe_tiles.py --full --json /tmp/tiles.json
"""

import argparse
import inspect
import json
import os
import statistics
import subprocess
import sys
import time

E, TOPK, HIDDEN, INTER = 128, 8, 2304, 1152
GEMMS = {"w13": (2 * INTER, HIDDEN), "w2": (HIDDEN, INTER)}  # name -> (N, K)
CHILD_MARK = "@@INKLING_TILE_RESULT@@"


def log(*a):
    print(*a, flush=True)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--scenarios", default="1:8,8:20,8:27,8:40,64:100,64:128")
    p.add_argument("--gemms", default="w13,w2")
    p.add_argument("--swap-ab", default="1,0")
    p.add_argument("--tile-n", default=None, help="default 32,64,128,256")
    p.add_argument("--tile-k", default=None, help="default 128,256")
    p.add_argument("--tile-m", default="", help="only if the API has tile_m")
    p.add_argument("--full", action="store_true", help="add tile_n 192, tile_k 64")
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--flush-mb", type=int, default=96)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mxfp8", action="store_true", help="also bench the MXFP8 GEMM")
    p.add_argument("--isolate", action="store_true",
                   help="run each tile config in a subprocess (survives faults)")
    p.add_argument("--json", default="", help="write all results here")
    p.add_argument("--only", default="", help=argparse.SUPPRESS)  # child mode
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    a.tile_n = a.tile_n or ("32,64,128,192,256" if a.full else "32,64,128,256")
    a.tile_k = a.tile_k or ("64,128,256" if a.full else "128,256")
    return a


def ints(s):
    return [int(v) for v in s.split(",") if v.strip()]


def scenarios(s):
    out = []
    for item in s.split(","):
        t, u = item.split(":")
        t, u = int(t), int(u)
        if not (1 <= t and 1 <= u <= E and u >= min(TOPK, E)):
            raise SystemExit(f"bad scenario {item}")
        out.append((t, min(u, t * TOPK)))
    return out


# ----------------------------------------------------------------- environment


def introspect(fn, label):
    log(f"--- {label}")
    try:
        sig = inspect.signature(fn)
        log(f"signature: {fn.__module__}.{fn.__name__}{sig}")
    except (TypeError, ValueError) as exc:
        sig = None
        log(f"signature: unavailable ({exc})")
    try:
        src = inspect.getsource(fn)
        hits = [ln.strip() for ln in src.splitlines()
                if any(t in ln for t in ("tile_", "swap_ab", "mma_sm", "raise", "assert"))]
        for ln in hits[:30]:
            log(f"  src| {ln}")
    except (OSError, TypeError):
        log("  (source unavailable)")
    return sig


def swizzle_blockscale(scale):
    """[B, M, K] block scales -> CUTLASS 128x4 atom layout (M % 128 == 0, K % 4 == 0)."""
    b, m, k = scale.shape
    assert m % 128 == 0 and k % 4 == 0, (m, k)
    s = scale.reshape(b, m // 128, 4, 32, k // 4, 4).permute(0, 1, 4, 3, 2, 5)
    return s.contiguous().reshape(b, m, k)


def route(torch, t, u, seed, device):
    from sglang.kernels.ops.moe.inkling2_route import route_metadata

    g = torch.Generator().manual_seed(seed * 1000 + t * 7 + u)
    pool = torch.randperm(E, generator=g)[:u]
    idx = (torch.arange(t)[:, None] * TOPK + torch.arange(TOPK)[None, :]) % u
    ids = pool[idx].to(torch.int32).to(device)
    weights = torch.rand((t, TOPK), generator=g).to(device)
    src2dst, dst2src, gammas, counts, offsets, *_ = route_metadata(
        ids.flatten(), weights.flatten(), 0, E, E
    )
    distinct = int((counts > 0).sum().item())
    return dst2src, offsets, distinct


# ----------------------------------------------------------------- NVFP4


class Nvfp4Case:
    def __init__(self, torch, gemm, n, k, device, seed):
        g = torch.Generator(device=device).manual_seed(seed + n + k)
        self.name, self.n, self.k = gemm, n, k
        self.weight = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8,
                                    device=device, generator=g)
        # E4M3 block scales in a narrow positive range (no NaN/Inf in outputs).
        raw = (torch.rand((E, n, k // 16), device=device, generator=g) * 1.5 + 0.25)
        raw = raw.to(torch.float8_e4m3fn)
        self.wscale = swizzle_blockscale(raw).flatten(1)  # as the model stores it
        self.wscale_arg = self.wscale.view(torch.uint8).view(E, n, k // 16)
        self.w_bytes_per_expert = n * (k // 2) + n * (k // 16)

    def inputs(self, torch, t, u, seed, device):
        from sglang.kernels.ops.moe.inkling2_experts_sm120 import nvfp4_routed_quantize

        dst2src, offsets, distinct = route(torch, t, u, seed, device)
        rows = dst2src.numel()
        g = torch.Generator(device=device).manual_seed(seed + 17 * t + u)
        if self.name == "w13":
            x = torch.randn((t, self.k), device=device, generator=g).to(torch.bfloat16)
            values, scales, _ = nvfp4_routed_quantize(x=x, dst2src=dst2src,
                                                      top_k=TOPK, offsets=offsets)
        else:
            # w2 consumes one quantized row per routed row, in expert order.
            x = torch.randn((rows, self.k), device=device, generator=g).to(torch.bfloat16)
            ident = torch.arange(rows, dtype=torch.int32, device=device)
            values, scales, _ = nvfp4_routed_quantize(x=x, dst2src=ident,
                                                      top_k=1, offsets=offsets)
        io_bytes = values.numel() + rows * (self.k // 16) + rows * self.n * 2
        return dict(values=values, scales=scales, offsets=offsets, rows=rows,
                    distinct=distinct,
                    bytes=distinct * self.w_bytes_per_expert + io_bytes,
                    w_bytes=distinct * self.w_bytes_per_expert)

    def call(self, fn, inp, out, cfg):
        kw = dict(tile_n=cfg["tile_n"], tile_k=cfg["tile_k"], swap_ab=cfg["swap_ab"], out=out)
        if "tile_m" in cfg:
            kw["tile_m"] = cfg["tile_m"]
        fn(inp["values"], self.weight, inp["scales"], self.wscale_arg, inp["offsets"], **kw)


# ----------------------------------------------------------------- MXFP8


class Mxfp8Case:
    def __init__(self, torch, gemm, n, k, device, seed):
        from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import (
            mxfp8_weight_scales,
        )

        g = torch.Generator(device=device).manual_seed(seed + 3 * n + k)
        self.name, self.n, self.k = gemm, n, k
        self.weight = (torch.randn((E, n, k), device=device, generator=g) * 0.5).to(
            torch.float8_e4m3fn)
        ue8m0 = torch.randint(122, 130, (E, n, k // 32), dtype=torch.uint8,
                              device=device, generator=g)
        self.wscale = mxfp8_weight_scales(ue8m0)
        self.w_bytes_per_expert = n * k + n * (k // 32)

    def inputs(self, torch, t, u, seed, device):
        from sglang.kernels.ops.moe.inkling2_experts_sm120 import mxfp8_routed_quantize

        dst2src, offsets, distinct = route(torch, t, u, seed, device)
        rows = dst2src.numel()
        g = torch.Generator(device=device).manual_seed(seed + 19 * t + u)
        if self.name == "w13":
            x = torch.randn((t, self.k), device=device, generator=g).to(torch.bfloat16)
            values, scales = mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=TOPK,
                                                   offsets=offsets)
        else:
            x = torch.randn((rows, self.k), device=device, generator=g).to(torch.bfloat16)
            ident = torch.arange(rows, dtype=torch.int32, device=device)
            values, scales = mxfp8_routed_quantize(x=x, dst2src=ident, top_k=1,
                                                   offsets=offsets)
        io_bytes = values.numel() + rows * (self.k // 32) + rows * self.n * 2
        return dict(values=values, scales=scales, offsets=offsets, rows=rows,
                    distinct=distinct,
                    bytes=distinct * self.w_bytes_per_expert + io_bytes,
                    w_bytes=distinct * self.w_bytes_per_expert)

    def call(self, fn, inp, out, cfg):
        kw = {k: v for k, v in cfg.items()}
        fn(inp["values"], self.weight, inp["scales"], self.wscale, inp["offsets"],
           scale_granularity_mnk=(1, 1, 32), out=out, **kw)


# ----------------------------------------------------------------- timing


def bitwise_diff(torch, a, b):
    if torch.equal(a.view(torch.int16), b.view(torch.int16)):
        return None
    d = (a.float() - b.float()).abs()
    finite = torch.isfinite(d)
    return dict(mismatched=int((a.view(torch.int16) != b.view(torch.int16)).sum()),
                max_abs=float(d[finite].max()) if finite.any() else float("nan"),
                ref_max=float(b.float().abs().max()))


def time_call(torch, run, flush, sink, iters, warmup):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        sink.copy_(flush.sum(dtype=torch.float32))  # read > L2: weights cold
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    us = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return dict(median_us=statistics.median(us), min_us=us[0],
                p90_us=us[int(0.9 * (len(us) - 1))], mean_us=statistics.fmean(us))


def context_dead(torch, device):
    """True if the CUDA context is poisoned (sticky error after a bad launch)."""
    try:
        torch.cuda.synchronize()
        return torch.ones(1, device=device).add_(1).item() != 2.0
    except Exception:  # noqa: BLE001
        return True


def sweep(torch, fn, cases, scen, configs, default_cfg, args, device, kind):
    """results: list of dicts, one per (config, gemm, scenario)."""
    flush = torch.ones(args.flush_mb * (1 << 20) // 4, device=device)
    sink = torch.empty((), device=device)
    results = []
    prepared = {}
    for case in cases:
        for t, u in scen:
            inp = case.inputs(torch, t, u, args.seed, device)
            out_ref = torch.empty((inp["rows"], case.n), dtype=torch.bfloat16, device=device)
            dcfg = default_cfg(t, case.k)
            case.call(fn, inp, out_ref, dcfg)
            out_again = torch.empty_like(out_ref)
            case.call(fn, inp, out_again, dcfg)
            torch.cuda.synchronize()
            det = bitwise_diff(torch, out_again, out_ref)
            if det is not None:
                log(f"WARNING {kind} {case.name} T={t}: stock config is not deterministic "
                    f"run to run ({det}); bitwise checks below are not meaningful")
            if not torch.isfinite(out_ref.float()).all():
                log(f"WARNING {kind} {case.name} T={t}: non-finite reference output")
            prepared[(case.name, t, u)] = (inp, out_ref, dcfg)
    for cfg in configs:
        tag = cfg_tag(cfg)
        t0 = time.time()
        first = True
        failed = False
        for case in cases:
            if failed:
                break
            for t, u in scen:
                inp, out_ref, dcfg = prepared[(case.name, t, u)]
                rec = dict(kind=kind, gemm=case.name, n=case.n, k=case.k, tokens=t,
                           distinct=inp["distinct"], rows=inp["rows"], cfg=cfg,
                           tag=tag, is_default=(cfg == dcfg))
                out = torch.empty_like(out_ref)
                try:
                    case.call(fn, inp, out, cfg)
                    torch.cuda.synchronize()
                except Exception as exc:  # noqa: BLE001
                    rec["error"] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"
                    results.append(rec)
                    if context_dead(torch, device):
                        log(f"FATAL CUDA error on {tag}: {rec['error']}\n"
                            "The CUDA context is unusable; rerun with --isolate.")
                        return results, True
                    failed = True  # rejected config: skip its other shapes
                    break
                if first:
                    rec["first_call_s"] = round(time.time() - t0, 1)
                    first = False
                rec["bitwise"] = bitwise_diff(torch, out, out_ref)
                tm = time_call(torch, lambda: case.call(fn, inp, out, cfg), flush, sink,
                               args.iters, args.warmup)
                rec.update(tm)
                rec["gbps"] = inp["bytes"] / (tm["median_us"] * 1e3)
                rec["w_gbps"] = inp["w_bytes"] / (tm["median_us"] * 1e3)
                results.append(rec)
        log(f"  {kind} {tag}: {'REJECTED' if failed else 'done'} in {time.time() - t0:.1f}s")
    return results, False


def cfg_tag(cfg):
    return ",".join(f"{k}={int(v) if isinstance(v, bool) else v}"
                    for k, v in cfg.items()) or "default"


# ----------------------------------------------------------------- reporting


def report(results, kind):
    ok = [r for r in results if "error" not in r]
    bad = {}
    for r in results:
        if "error" in r:
            bad.setdefault(r["tag"], r["error"])
    if bad:
        log(f"\n{kind}: configs rejected / failed:")
        for tag, err in bad.items():
            log(f"  {tag}: {err}")
    keys = sorted({(r["gemm"], r["tokens"], r["distinct"]) for r in ok},
                  key=lambda x: (x[0] != "w13", x[1], x[2]))
    for gemm, t, u in keys:
        rows = sorted((r for r in ok if (r["gemm"], r["tokens"], r["distinct"]) == (gemm, t, u)),
                      key=lambda r: r["median_us"])
        base = next((r for r in rows if r["is_default"]), None)
        log(f"\n{kind} {gemm} N={rows[0]['n']} K={rows[0]['k']} tokens={t} "
            f"distinct={u} rows={rows[0]['rows']}")
        log(f"  {'config':<38} {'med us':>8} {'min us':>8} {'p90 us':>8} "
            f"{'GB/s':>7} {'wGB/s':>7} {'vs stock':>8}  exact")
        for r in rows:
            sp = base["median_us"] / r["median_us"] if base else float("nan")
            ex = "yes" if r["bitwise"] is None else (
                f"NO ({r['bitwise']['mismatched']} el, max {r['bitwise']['max_abs']:.3g})")
            mark = " *stock" if r["is_default"] else ""
            log(f"  {r['tag']:<38} {r['median_us']:8.1f} {r['min_us']:8.1f} "
                f"{r['p90_us']:8.1f} {r['gbps']:7.1f} {r['w_gbps']:7.1f} {sp:8.3f}  {ex}{mark}")


def suggest(results):
    """Best bit-exact config per (tokens, width), summed over scenarios with that T."""
    ok = [r for r in results if "error" not in r and r["bitwise"] is None]
    rules, lines = [], []
    for t in sorted({r["tokens"] for r in ok}):
        for gemm in ("w13", "w2"):
            sel = [r for r in ok if r["tokens"] == t and r["gemm"] == gemm]
            if not sel:
                continue
            tot = {}
            for r in sel:
                tot.setdefault(r["tag"], [0.0, r["cfg"], r["is_default"], r["k"]])
                tot[r["tag"]][0] += r["median_us"]
            nscen = len({r["distinct"] for r in sel})
            complete = {k: v for k, v in tot.items()
                        if sum(1 for r in sel if r["tag"] == k) == nscen}
            if not complete:
                continue
            best_tag, (best_us, best_cfg, _, width) = min(complete.items(),
                                                           key=lambda kv: kv[1][0])
            stock = next((v for v in complete.values() if v[2]), None)
            gain = (stock[0] - best_us) if stock else float("nan")
            lines.append(f"  tokens<={t} {gemm} (K={width}): best {best_tag} "
                         f"sum {best_us:.1f} us vs stock {stock[0] if stock else float('nan'):.1f} us "
                         f"(saves {gain:.1f} us summed over {nscen} scenario(s))")
            kv = ",".join(f"{k}={int(v)}" for k, v in best_cfg.items() if k in
                          ("swap_ab", "tile_n", "tile_k"))
            rules.append((f"{t}@{width}:{kv}", stock is not None and best_tag == next(
                k for k, v in complete.items() if v[2])))
    log("\nBest bit-exact NVFP4 config per token bucket:")
    for ln in lines:
        log(ln)
    if rules and not all(is_stock for _, is_stock in rules):
        # Every bucket gets a rule (stock values where stock wins), because a
        # rule's bound covers all smaller token counts.
        log("\nSuggested override (A/B end to end):")
        log(f"  INKLING_KERNELS=moe_tiles INKLING_MOE_TILES='{';'.join(r for r, _ in rules)}'")
    else:
        log("\nThe stock tiles are already the fastest bit-exact choice; no override.")


# ----------------------------------------------------------------- main


def build_configs(args, sig, extra_tile_m):
    configs = []
    for sw in ints(args.swap_ab):
        for tn in ints(args.tile_n):
            for tk in ints(args.tile_k):
                for tm in (extra_tile_m or [None]):
                    cfg = dict(swap_ab=bool(sw), tile_n=tn, tile_k=tk)
                    if tm is not None:
                        cfg["tile_m"] = tm
                    configs.append(cfg)
    return configs


def main():
    args = parse_args()
    import torch

    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(device)
    prop = torch.cuda.get_device_properties(device)
    if not args.child:
        import flashinfer

        log(f"device: {prop.name} sm_{prop.major}{prop.minor} SMs={prop.multi_processor_count} "
            f"L2={getattr(prop, 'L2_cache_size', 0) / 2**20:.0f} MiB  torch {torch.__version__}  "
            f"flashinfer {getattr(flashinfer, '__version__', '?')}")
    from flashinfer.gemm import group_gemm_nvfp4_nt_groupwise as nvfp4_fn
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped

    stock_tiles = getattr(grouped._tiles, "__wrapped__", grouped._tiles)

    def default_nvfp4(t, k):
        sw, tn, tk = stock_tiles(tokens=t, width=k)
        cfg = dict(swap_ab=bool(sw), tile_n=tn, tile_k=tk)
        if tile_m:  # stock passes no tile_m: the API default
            cfg["tile_m"] = sig.parameters["tile_m"].default
        return cfg

    sig = introspect(nvfp4_fn, "group_gemm_nvfp4_nt_groupwise") if not args.child else \
        inspect.signature(nvfp4_fn)
    tile_m = ints(args.tile_m) if args.tile_m and sig and "tile_m" in sig.parameters else []
    if args.tile_m and not tile_m:
        log("note: --tile-m ignored, the API has no tile_m")
    configs = build_configs(args, sig, tile_m)
    if args.only:
        sw, tn, tk = ints(args.only)[:3]
        configs = [c for c in configs if (int(c["swap_ab"]), c["tile_n"], c["tile_k"]) == (sw, tn, tk)]
    scen = scenarios(args.scenarios)
    gemms = [g for g in args.gemms.split(",") if g]
    all_results = []

    if args.isolate and not args.child:
        log(f"isolated sweep: {len(configs)} configs x {len(gemms)} GEMMs x {len(scen)} scenarios")
        for cfg in configs:
            only = f"{int(cfg['swap_ab'])},{cfg['tile_n']},{cfg['tile_k']}"
            cmd = [sys.executable, os.path.abspath(__file__), "--child", "--only", only,
                   "--scenarios", args.scenarios, "--gemms", args.gemms,
                   "--swap-ab", args.swap_ab, "--tile-n", args.tile_n,
                   "--tile-k", args.tile_k, "--tile-m", args.tile_m,
                   "--iters", str(args.iters), "--warmup", str(args.warmup),
                   "--flush-mb", str(args.flush_mb), "--seed", str(args.seed)]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            got = [ln for ln in proc.stdout.splitlines() if ln.startswith(CHILD_MARK)]
            if proc.returncode != 0 or not got:
                tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or ["?"]
                log(f"  {cfg_tag(cfg)}: child exit {proc.returncode}: {tail[0][:200]}")
                for g in gemms:
                    for t, u in scen:
                        all_results.append(dict(kind="nvfp4", gemm=g, tokens=t, distinct=u,
                                                cfg=cfg, tag=cfg_tag(cfg),
                                                error=f"child exit {proc.returncode}: {tail[0][:200]}"))
                continue
            res = json.loads(got[-1][len(CHILD_MARK):])
            all_results.extend(res)
            log(f"  {cfg_tag(cfg)}: ok")
    else:
        cases = [Nvfp4Case(torch, g, *GEMMS[g], device, args.seed) for g in gemms]
        res, fatal = sweep(torch, nvfp4_fn, cases, scen, configs, default_nvfp4, args,
                           device, "nvfp4")
        all_results.extend(res)
        if args.child:
            for r in res:
                r["cfg"] = {k: (int(v) if isinstance(v, bool) else v) for k, v in r["cfg"].items()}
            print(CHILD_MARK + json.dumps(res), flush=True)
            sys.exit(3 if fatal else 0)
        del cases
        torch.cuda.empty_cache()

    for r in all_results:  # child JSON turns bools into ints
        r["cfg"]["swap_ab"] = bool(r["cfg"]["swap_ab"])
    report(all_results, "NVFP4")
    suggest(all_results)

    if args.mxfp8:
        from flashinfer.grouped_mm import moe_gemm_mxfp8_nt_groupwise as mx_fn

        msig = introspect(mx_fn, "moe_gemm_mxfp8_nt_groupwise")
        fixed = {"a", "b", "a_scale", "b_scale", "m_indptr", "out", "out_dtype",
                 "scale_granularity_mnk", "scale_major_mode"}
        tunable = [p for p in (msig.parameters if msig else {}) if p not in fixed]
        log(f"MXFP8 non-data parameters: {tunable or 'none'}")
        grid = {"swap_ab": [bool(v) for v in ints(args.swap_ab)],
                "tile_n": ints(args.tile_n), "tile_k": ints(args.tile_k),
                "tile_m": ints(args.tile_m) if args.tile_m else []}
        mx_cfgs = [dict()]
        for name in ("swap_ab", "tile_m", "tile_n", "tile_k"):
            if name in tunable and grid[name]:
                mx_cfgs = [dict(c, **{name: v}) for c in mx_cfgs for v in grid[name]]
        if len(mx_cfgs) == 1:
            log("MXFP8: no tile tunables in this FlashInfer; timing the default only")
        else:
            mx_cfgs.insert(0, {})  # the API default = what SGLang runs (reference)
        mcases = [Mxfp8Case(torch, g, *GEMMS[g], device, args.seed) for g in gemms]
        mres, _ = sweep(torch, mx_fn, mcases, scen, mx_cfgs, lambda t, k: {}, args,
                        device, "mxfp8")
        report(mres, "MXFP8")
        all_results.extend(mres)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(all_results, f, indent=1, default=str)
        log(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
