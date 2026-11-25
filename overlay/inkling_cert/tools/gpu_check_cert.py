#!/usr/bin/env python3
"""GPU check + microbench of inkling_cert on the real Inkling-2-Mini head (run in the SGLang image, server stopped).

    python3 inkling_cert/tools/gpu_check_cert.py --model /model --hidden DIR [--quick]

DIR holds *.npz with `hidden` (uint16 BF16 bits [T, 2304]): real LM-head input rows (inkling_cert/tools or the
CPU-oracle dump). Gates (any failure -> FAIL):
  1  packing: the 6-bit copy decodes (by the kernel's rule) to the quantised head; tables finite
  2  screen accuracy: |s - <D, x>| and |pico - <W, x>| within the accumulation model; lo <= pico <= hi everywhere
  3  exactness: certified token == full-head token (stock softcap + padding mask + stock argmax) on every real row,
     at M = 8 and M = 16, plus near-tie rows (two ids within 1e-3, softcap saturation, exact duplicates if any);
     computed tiles bit-identical to pico's full head, the rest -inf; on = 0 gives pico's full head bit for bit
  4  CUDA graph: replay with changing rows and on flag == eager
Bench (L2 flushed by reading 256 MiB before every rep, CUDA events, median of N): pico full head, cert path
(screen + exact), screen alone, at M = 8 and 16 with real candidate masks. Last line: `cert gpu_check result PASS|FAIL`.
"""

import argparse
import glob
import json
import os
import struct
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OVERLAY)

from inkling_cert import build as cbuild, cert_math as cm, ops as cops, quant  # noqa: E402
from inkling_pico12 import build as pbuild, ops as pops  # noqa: E402

FAIL = []


def gate(ok, msg):
    print(("  ok   " if ok else "  FAIL ") + msg, flush=True)
    if not ok:
        FAIL.append(msg)


def load_head(model_dir):
    path = os.path.join(model_dir, "model-00001-of-00001.safetensors")
    if not os.path.exists(path):
        idx = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
        path = os.path.join(model_dir, idx["model.llm.unembed.weight"])
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
        meta = hdr["model.llm.unembed.weight"]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(meta["shape"])
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).cuda()


def load_hidden(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
        z = np.load(f)
        rows.append(torch.from_numpy(z["hidden"].astype(np.int16)).view(torch.bfloat16))
    return torch.cat(rows).cuda()


def stock_fns():
    from sglang.kernels.ops.activation.softcap import softcap_inplace_logits
    from sglang.kernels.ops.sampling import argmax as sampling_argmax

    return softcap_inplace_logits, sampling_argmax


def tokens(logits, v_real, sc, am):
    y = logits.clone()
    sc(y, cm.CAP)
    y[:, v_real:] = float("-inf")
    return am(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--hidden", default="/work/hid")
    ap.add_argument("--v-real", type=int, default=200058)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name()} torch {torch.__version__}", flush=True)
    t0 = time.time()
    pops.EXT = pbuild.load(os.environ.get("INKLING_PICO12_CACHE") or None)
    ext = cbuild.load(os.environ.get("INKLING_CERT_HEAD_CACHE") or None)
    print(f"extensions ready {time.time() - t0:.1f}s ({cbuild.INFO.get('build_dir')})", flush=True)
    W = load_head(a.model)
    V = W.shape[0]
    pack = pops.encode(W, "lm_head")
    t0 = time.time()
    tables = quant.build_tables(W)
    torch.cuda.synchronize()
    print(f"head [{V}x{W.shape[1]}] pico {pack.nbytes / 1e6:.1f} MB; tables {time.time() - t0:.1f}s eps "
          f"{tables['eps'] * 100:.3f}%", flush=True)
    cert = cops.Cert(ext, pack, tables, pops.SET.wk, dev)
    print(f"screen bytes {cert.screen_bytes / 1e6:.1f} MB", flush=True)
    sc, am = stock_fns()

    print("1 packing", flush=True)
    for r0 in (0, 100000, V - 4096):
        q, s_, d = quant.quantize_rows(W[r0:r0 + 4096])
        off = r0 // 8 * quant.STEPS * 384
        got = quant.unpack6(cert.q[off: off + 4096 // 8 * quant.STEPS * 384], 4096)
        gate(torch.equal(got, q) and torch.equal(cert.sc[:, r0:r0 + 4096].t(), s_), f"rows {r0}..: decode == quantised")
    gate(bool(torch.isfinite(cert.rt.float()).all() and torch.isfinite(cert.nv).all()), "tables finite")

    X = load_hidden(a.hidden)
    if a.quick:
        X = X[:512]
    N = X.shape[0]
    print(f"{N} real hidden rows", flush=True)

    print("2 screen accuracy (M=8 batches)", flush=True)
    nchk = min(N, 512 if not a.quick else 64) // 8 * 8
    SD = torch.empty((nchk, V), dtype=torch.float32, device=dev)
    BD = torch.empty_like(SD)
    REF = torch.empty_like(SD)
    for b0 in range(0, nchk, 8):
        x = X[b0:b0 + 8].contiguous()
        sd = torch.empty((8, V), dtype=torch.float32, device=dev)
        bd = torch.empty_like(sd)
        cert.screen(x, cert.force, a.v_real, cm.CAP, sd, bd)
        SD[b0:b0 + 8], BD[b0:b0 + 8] = sd, bd
        pops.gemv(x, pack, REF[b0:b0 + 8])
    worst_s = worst_p = worst_cov = 0.0
    bad_cov = 0
    xd = X[:nchk].double()
    xn = xd.norm(dim=1)
    for c0 in range(0, V, 8192):
        c1 = min(V, c0 + 8192)
        wd = W[c0:c1].double()
        q = quant.unpack6(cert.q[c0 // 8 * quant.STEPS * 384: c1 // 8 * quant.STEPS * 384], c1 - c0).double()
        dd = (q.view(c1 - c0, quant.NGROUPS, quant.GROUP) * cert.sc[:, c0:c1].double().t()[..., None]).view(c1 - c0, -1)
        exact = xd @ wd.t()
        sdx = xd @ dd.t()
        acc_w = cm.C_ACC * xn[:, None] * wd.norm(dim=1)[None, :]
        acc_d = cm.C_ACC * xn[:, None] * dd.norm(dim=1)[None, :]
        s_, b_, r_ = SD[:, c0:c1].double(), BD[:, c0:c1].double(), REF[:, c0:c1].double()
        worst_s = max(worst_s, float(((s_ - sdx).abs() / acc_d.clamp_min(1e-30)).max()))
        worst_p = max(worst_p, float(((r_ - exact).abs() / acc_w.clamp_min(1e-30)).max()))
        bad_cov += int(((r_ < s_ - b_) | (r_ > s_ + b_)).sum())
        worst_cov = max(worst_cov, float(((r_ - s_).abs() / b_).max()))
    del SD, BD, REF
    gate(worst_s <= 1.0, f"screen accumulation error / model <= 1 (max {worst_s:.4f})")
    gate(worst_p <= 1.0, f"pico accumulation error / model <= 1 (max {worst_p:.4f})")
    gate(bad_cov == 0, f"s - B <= pico <= s + B for every logit ({bad_cov} outside; max |pico - s| / B = {worst_cov:.4f})")

    print("3 exactness", flush=True)
    for M in (8, 16):
        mism = cert_steps = tiles = 0
        tile_list = []
        nbits = 0
        for b0 in range(0, N - M + 1, M):
            x = X[b0:b0 + M].contiguous()
            ref = torch.empty((M, V), dtype=torch.float32, device=dev)
            pops.gemv(x, pack, ref)
            out = torch.empty_like(ref)
            cert.run(x, out, cert.force, a.v_real, cm.CAP)
            certified = int(cert.flags[2])
            t_ref = tokens(ref, a.v_real, sc, am)
            t_out = tokens(out, a.v_real, sc, am)
            mism += int((t_ref != t_out).sum())
            msk = cert.mask.bool().repeat_interleave(16)
            if not torch.equal(out[:, msk].view(torch.int32), ref[:, msk].view(torch.int32)):
                nbits += 1
            if bool(torch.isfinite(out[:, ~msk]).any()):
                nbits += 1
            cert_steps += certified
            nt = int(cert.mask.sum())
            if certified:
                tiles += nt
                tile_list.append(nt)
        tl = np.array(tile_list) if tile_list else np.zeros(1)
        gate(mism == 0, f"M={M}: certified token == full-head token on {N // M * M} rows ({mism} mismatches)")
        gate(nbits == 0, f"M={M}: computed tiles bit-identical to pico, others -inf ({nbits} bad steps)")
        print(f"     M={M}: certified {cert_steps}/{N // M} steps; tiles/step mean {tl.mean():.1f} p50 "
              f"{np.median(tl):.0f} p90 {np.percentile(tl, 90):.0f} p99 {np.percentile(tl, 99):.0f} max {tl.max():.0f}",
              flush=True)
    # off -> full head bit for bit
    x = X[:8].contiguous()
    ref = torch.empty((8, V), dtype=torch.float32, device=dev)
    pops.gemv(x, pack, ref)
    out = torch.empty_like(ref)
    cert.run(x, out, cert.off, a.v_real, cm.CAP)
    gate(torch.equal(out.view(torch.int32), ref.view(torch.int32)) and int(cert.flags[2]) == 0,
         "on = 0: pico's full head bit for bit")
    # near ties: two ids with nearly equal logits; saturated logits; random pairs
    g = torch.Generator(device=dev).manual_seed(7)
    nt_mism = nt_n = 0
    for scale in (150.0, 230.0, 400.0, 800.0):
        for _ in range(16 if a.quick else 64):
            ids = torch.randint(0, a.v_real, (8, 2), generator=g, device=dev)
            w = W[ids[:, 0]].float() + W[ids[:, 1]].float()
            noise = torch.randn((8, W.shape[1]), generator=g, device=dev) * 0.02
            x = ((w / w.norm(dim=1, keepdim=True) + noise) * scale).to(torch.bfloat16).contiguous()
            ref = torch.empty((8, V), dtype=torch.float32, device=dev)
            pops.gemv(x, pack, ref)
            out = torch.empty_like(ref)
            cert.run(x, out, cert.force, a.v_real, cm.CAP)
            nt_mism += int((tokens(ref, a.v_real, sc, am) != tokens(out, a.v_real, sc, am)).sum())
            nt_n += 8
    gate(nt_mism == 0, f"near-tie rows (two ids, |x| 150-800): {nt_n} rows, {nt_mism} mismatches")
    # exact duplicate rows in the head, if any (hash rows)
    h = (W.view(torch.int16).long() * torch.arange(1, W.shape[1] + 1, device=dev)).sum(1)
    hs, order = torch.sort(h[: a.v_real])
    dup = order[1:][hs[1:] == hs[:-1]]
    if dup.numel():
        d0 = order[:-1][hs[1:] == hs[:-1]][:8]
        x = (W[d0].float() * 230.0).to(torch.bfloat16).contiguous()
        m = x.shape[0]
        ref = torch.empty((m, V), dtype=torch.float32, device=dev)
        pops.gemv(x, pack, ref)
        out = torch.empty_like(ref)
        cert.run(x, out, cert.force, a.v_real, cm.CAP)
        gate(torch.equal(tokens(ref, a.v_real, sc, am), tokens(out, a.v_real, sc, am)),
             f"duplicate head rows ({dup.numel()} pairs): exact ties resolve like the full head")
    else:
        print("     no duplicate head rows", flush=True)

    print("4 CUDA graph", flush=True)
    xs = torch.empty((8, W.shape[1]), dtype=torch.bfloat16, device=dev)
    og = torch.empty((8, V), dtype=torch.float32, device=dev)
    xs.copy_(X[:8])
    cert.set_on(1)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        cert.run(xs, og, cert.on, a.v_real, cm.CAP)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cert.run(xs, og, cert.on, a.v_real, cm.CAP)
    ok = True
    for i, b0 in enumerate(range(8, min(N, 8 + 8 * 40), 8)):
        xs.copy_(X[b0:b0 + 8])
        cert.set_on(i % 3 != 0)
        graph.replay()
        ref = torch.empty((8, V), dtype=torch.float32, device=dev)
        e = torch.empty_like(ref)
        cert.run(xs, e, cert.on, a.v_real, cm.CAP)
        ok &= torch.equal(og.view(torch.int32), e.view(torch.int32))
        if i % 3 == 0:
            pops.gemv(xs, pack, ref)
            ok &= torch.equal(og.view(torch.int32), ref.view(torch.int32))
    gate(ok, "graph replay == eager (40 steps, on toggled)")
    cert.set_on(0)

    print("5 bench (L2 flushed, CUDA events)", flush=True)
    flush = torch.empty((256 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.zeros((1,), dtype=torch.int64, device=dev)

    def timeit(fn, reps):
        ts = []
        for _ in range(reps):
            sink.add_(flush.view(torch.int64).sum())  # read 256 MiB
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        return float(np.median(ts)), float(np.percentile(ts, 10))

    res = {}
    for M in (8, 16):
        x = X[:M].contiguous()
        out = torch.empty((M, V), dtype=torch.float32, device=dev)
        # pick a batch with the median candidate count for the cert timing
        counts = []
        for b0 in range(0, min(N, 64 * M), M):
            cert.run(X[b0:b0 + M].contiguous(), out, cert.force, a.v_real, cm.CAP)
            counts.append((int(cert.mask.sum()), b0))
        counts.sort()
        med_b0 = counts[len(counts) // 2][1]
        p90_b0 = counts[int(len(counts) * 0.9)][1]
        xm = X[med_b0:med_b0 + M].contiguous()
        x9 = X[p90_b0:p90_b0 + M].contiguous()
        t_pico = timeit(lambda: pops.gemv(x, pack, out), a.reps)
        t_cert = timeit(lambda: cert.run(xm, out, cert.force, a.v_real, cm.CAP), a.reps)
        t_cert9 = timeit(lambda: cert.run(x9, out, cert.force, a.v_real, cm.CAP), a.reps)
        t_scr = timeit(lambda: cert.screen(xm, cert.force, a.v_real, cm.CAP), a.reps)
        t_full = timeit(lambda: cert.run(xm, out, cert.off, a.v_real, cm.CAP), a.reps)
        gbs = cert.screen_bytes / (t_scr[0] * 1e-3) / 1e9
        res[M] = dict(pico_ms=t_pico, cert_ms_median_batch=t_cert, cert_ms_p90_batch=t_cert9, screen_ms=t_scr,
                      fallback_ms=t_full, screen_GBps=gbs, tiles_median=counts[len(counts) // 2][0],
                      tiles_p90=counts[int(len(counts) * 0.9)][0])
        print(f"     M={M}: pico full head {t_pico[0]:.3f} ms ({pack.nbytes / t_pico[0] / 1e6:.0f} GB/s) | cert "
              f"{t_cert[0]:.3f} ms at {res[M]['tiles_median']} tiles, {t_cert9[0]:.3f} ms at {res[M]['tiles_p90']} "
              f"(p90 batch) | screen alone {t_scr[0]:.3f} ms = {gbs:.0f} GB/s of {cert.screen_bytes / 1e6:.0f} MB | "
              f"not-certified step (screen + full) {t_full[0]:.3f} ms", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    print(f"cert gpu_check result {'FAIL' if FAIL else 'PASS'}" + (f" ({len(FAIL)} failed)" if FAIL else ""), flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
