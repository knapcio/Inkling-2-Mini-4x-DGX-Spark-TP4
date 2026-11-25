#!/usr/bin/env python3
"""GPU check of inkling_prologue's fused K / V prologue kernel (SGLang image, server stopped).

Stock: short_convolution (prefill kernel) -> qk_l2_clip_fwd_cutedsl (K, threshold from the config, no scale) or
rms_norm without weight (V), over [T, 8, 128]. Fused: inkling_prologue.conv_post. Real layer-0 k / v conv weights
when /model is mounted. Two sequences (a cached prefix in slot 1 + a fresh one), T in 2048 / 2049 / 3001 / 8192.
Sweeps the row-sum orders of _triton._row_sumsq (PER x XOR, and -1 = the old tl.sum) and prints which give 0 differing
elements. The default (inkling_prologue.SUM_ORDER) must be bit-identical to stock: 0 ulp at every size; also the
error of both vs an FP64 post-op on the same BF16 conv output; then cold-L2 timing at 8192 rows."""
import glob
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FAILED = []
SWEEP = {}
D, H = 1024, 128


def main():
    import torch

    import sglang.srt.models.inkling2  # noqa: F401  (serving assembler for Triton, kernel imports)
    from inkling_prologue import conv_post
    from sglang.kernels.ops.layernorm.inkling2_qk_l2_clip import qk_l2_clip_fwd_cutedsl
    from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm
    from sglang.kernels.ops.mamba.inkling2_sconv import short_convolution

    thr, eps_clip, eps_norm = 5.0, 1e-6, 1e-6
    weights = {}
    try:
        cfg = json.load(open("/model/config.json"))
        tc = cfg.get("text_config", cfg)
        thr = float(tc.get("qk_l2_clip_threshold", thr))
        eps_norm = float(tc.get("rms_norm_eps", eps_norm))
        from safetensors import safe_open

        for f in sorted(glob.glob("/model/model*.safetensors")):
            h = safe_open(f, "pt", device="cpu")
            for name in ("k_sconv", "v_sconv"):
                key = f"model.llm.layers.0.attn.{name}.weight"
                if key in h.keys():
                    weights[name] = h.get_tensor(key).cuda().to(torch.bfloat16).reshape(D, -1).contiguous()
    except Exception as exc:  # noqa: BLE001
        print(f"config / weights: {exc!r}")
    print(f"threshold {thr}, eps clip {eps_clip}, eps norm {eps_norm}, real weights {sorted(weights)}", flush=True)
    g = torch.Generator(device="cuda").manual_seed(3)
    flush = (torch.empty((), device="cuda"), torch.ones(128 * (1 << 20) // 4, device="cuda"))

    def bench(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize()
        ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
        for a, b in ev:
            flush[0].copy_(flush[1].sum())
            a.record()
            fn()
            b.record()
        torch.cuda.synchronize()
        return statistics.median(a.elapsed_time(b) * 1e3 for a, b in ev)

    def cute_sumsq(y, post):
        """The CuTe row_sum tree on the BF16 conv output y [T, D] (FP32, eager torch: every op rounds once):
        per lane a sequential fold over its vectors in register order, then a butterfly with descending offsets."""
        t32 = y.float().view(y.shape[0], D // H, 2, 8, 8) if post == 1 else y.float().view(y.shape[0], D // H, 2, 16, 4)
        lanes = t32.shape[3]
        acc = torch.zeros(t32.shape[:2] + (lanes,), dtype=torch.float32, device=y.device)
        for j in range(2):
            for e in range(t32.shape[4]):
                v = t32[:, :, j, :, e]
                acc = acc + v * v
        off = lanes // 2
        while off:
            idx = torch.arange(lanes, device=y.device) ^ off
            acc = acc + acc[:, :, idx]
            off //= 2
        return acc[:, :, 0]

    def ordered(t):
        b = t.contiguous().view(torch.int16).int()
        return torch.where(b < 0, -(b & 0x7FFF), b)

    for T in (2048, 2049, 3001, 8192):
        cut = T // 3
        meta = {
            "safe_idx": torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
            "cache_mask": torch.tensor([True, False], device="cuda"),
            "cu": torch.tensor([0, cut, T], dtype=torch.int64, device="cuda"),
            "si": torch.cat([torch.zeros(cut, dtype=torch.int32), torch.ones(T - cut, dtype=torch.int32)]).cuda(),
        }
        cache = (torch.randn(3, 3, D, device="cuda", generator=g)).to(torch.bfloat16)
        for name, post in (("k_sconv", 1), ("v_sconv", 2)):
            w = weights.get(name)
            if w is None:
                w = (torch.randn(D, 4, device="cuda", generator=g) * 0.3).to(torch.bfloat16)
            # heads with very different scales: zero heads, heads around / far above the clip threshold
            hs = torch.tensor([0.0, 0.05, 0.4, 1.0, 1.3, 3.0, 8.0, 30.0], device="cuda").repeat_interleave(H)
            xa = (torch.randn(T, D + 64, device="cuda", generator=g) * hs.new_ones(D + 64)).to(torch.bfloat16)
            x = xa[:, :D]  # row stride D + 64 like a view of a wider projection
            x.mul_(hs.to(torch.bfloat16))
            # order-sensitive rows (one big value + tiny ones that only survive if added together first); the conv
            # mixes neighbouring rows, so these feed sums of many magnitudes
            for r0 in range(5, T, 97):
                x[r0].zero_()
                x[r0, ::H] = 1.0
                x[r0, 4::H] = 2.0 ** -12
                x[r0, 5::H] = 2.0 ** -12
                x[r0, 64::H] = 2.0 ** -12

            def stock():
                y = short_convolution(x, w, cache, meta).view(T, D // H, H)
                if post == 1:
                    return qk_l2_clip_fwd_cutedsl(y, thr, eps_clip, scale=None, output_scale=None).view(T, D)
                return rms_norm(y, None, eps_norm).view(T, D)

            def fused():
                return conv_post(x, w, cache, meta, post, thr if post == 1 else 0.0,
                                 eps_clip if post == 1 else eps_norm)

            ys = stock()
            exact = []
            for order in [(p_, x_) for p_ in range(5) for x_ in range(2)] + [(-1, 0)]:
                yo = conv_post(x, w, cache, meta, post, thr if post == 1 else 0.0,
                               eps_clip if post == 1 else eps_norm, order=order)
                n_diff = int((ys.view(torch.int16) != yo.view(torch.int16)).sum().item())
                SWEEP.setdefault((post, order), []).append(n_diff)
                if n_diff == 0:
                    exact.append(order)
            print(f"T={T} {name}: orders with 0 differing elements: {exact or 'none'}", flush=True)
            yf = fused()
            ssf = torch.empty(T, D // H, dtype=torch.float32, device="cuda")
            conv_post(x, w, cache, meta, post, thr if post == 1 else 0.0, eps_clip if post == 1 else eps_norm, ss=ssf)
            ssr = cute_sumsq(short_convolution(x, w, cache, meta), post)
            ss_ok = torch.equal(ssf.view(torch.int32), ssr.view(torch.int32))
            raw_ok = torch.equal(ys.view(torch.int16), yf.view(torch.int16))
            print(f"T={T} {name}: FP32 row sum vs the CuTe tree {'BIT-IDENTICAL' if ss_ok else 'DIFFERS'}; "
                  f"output raw bits {'IDENTICAL' if raw_ok else 'DIFFER'}", flush=True)
            if not (ss_ok and raw_ok):
                FAILED.append(f"T={T} {name} bits")
            y = short_convolution(x, w, cache, meta).double().view(T, D // H, H)
            ss = (y * y).sum(-1, keepdim=True)
            if post == 1:
                ref = y * torch.clamp(thr / (ss.sqrt() + eps_clip), max=1.0)
            else:
                ref = y * torch.rsqrt(ss / H + eps_norm)
            ref = ref.view(T, D)
            es, ef = (ys.double() - ref).abs(), (yf.double() - ref).abs()
            bit = (ys.view(torch.int16) == yf.view(torch.int16)).float().mean().item()
            ulp = (ordered(ys) - ordered(yf)).abs().max().item()
            # elementwise bound: one BF16 ulp of the reference value (FP32 reduction order may flip one rounding)
            tol = ref.abs() * 2.0 ** -7 + 1e-30
            within = bool(((yf.double() - ref).abs() <= tol + es).all())
            ok = (bool(torch.isfinite(yf).all()) and ef.max().item() <= 1.5 * es.max().item() + 1e-7
                  and ef.mean().item() <= 1.5 * es.mean().item() + 1e-9 and within and ulp == 0)  # 0 ulp required
            print(f"T={T} {name}: bit-equal {100 * bit:.3f} %, max {ulp} ulp, err max stock {es.max().item():.2e} "
                  f"fused {ef.max().item():.2e}, mean stock {es.mean().item():.2e} fused {ef.mean().item():.2e} "
                  f"{'ok' if ok else 'FAIL'}", flush=True)
            if not ok:
                FAILED.append(f"T={T} {name}")
            if T == 8192:
                ts, tf = bench(stock), bench(fused)
                print(f"   time stock {ts:.0f} us, fused {tf:.0f} us, saved {ts - tf:+.0f} us per call "
                      f"({40 * (ts - tf) / 1e3:+.1f} ms per 8K chunk over 40 SWA visits)", flush=True)
    for post in (1, 2):
        rows = sorted((sum(v), k[1]) for k, v in SWEEP.items() if k[0] == post)
        print(f"post {post} ({'clip' if post == 1 else 'rmsnorm'}): differing elements over all sizes by (PER, XOR): "
              + ", ".join(f"{o}={n}" for n, o in rows), flush=True)
    print("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
