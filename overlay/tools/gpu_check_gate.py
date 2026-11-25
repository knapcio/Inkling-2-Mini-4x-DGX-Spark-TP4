#!/usr/bin/env python3
"""GPU check + timing of inkling_attn's gated epilogue (INKLING_ATTN_GATE): gate_sink inside the SWA prefill kernel.

  python3 tools/gpu_check_gate.py            (SGLang image, model server stopped; /model for the real proj)

1. bit identity: y of the gated launch == SGLang's gate_sink(out, gate, lse, sinks) on the ungated launch's FP32 out
   and LSE, on every written row (chunk 8192 prefix 0, chunk 2048 prefix 8192, 3 requests + padding rows), LSE
   bit-identical too; gate / sinks as strided views of a qkvr-shaped buffer, like serving;
2. the patched gate_sink after a gated launch (the state patch.make_varlen leaves) returns that y; a mismatched call
   raises; _gate_ok refuses unsupported layouts (fallback to the stock path); outside a gated visit it is stock;
3. CUDA graph: a captured gated launch replays equal to eager;
4. registers / spills of the gated and ungated SWA instantiations (cuobjdump -res-usage on the built extension);
5. cold-L2 timing at 8192 rows prefix 0: ungated attention + stock gate_sink (back to back, one event pair) vs gated
   attention, 4 and 8 warps.
"""
import os
import re
import statistics
import subprocess
import sys

os.environ.setdefault("INKLING_ATTN_GATE", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
HEADS, KVH, HD, DREL, WINDOW, PAGE, QKVR = 24, 8, 128, 16, 1024, 128, 8600
SCALE = HD ** -0.5
FAILED = []


def log(*a):
    print(*a, flush=True)


def load_proj(torch, dev):
    try:
        import glob

        from safetensors import safe_open

        for path in sorted(glob.glob("/model/model*.safetensors")):
            with safe_open(path, framework="pt") as f:
                k = "model.llm.layers.0.attn.rel_logits_proj.proj"
                if k in f.keys():
                    return f.get_tensor(k).to(dev, torch.bfloat16).contiguous()
    except Exception as exc:  # noqa: BLE001
        log(f"proj: random ({exc!r})")
    g = torch.Generator(device=dev).manual_seed(0)
    return torch.randn((DREL, WINDOW), device=dev, generator=g).to(torch.bfloat16)


def case(torch, dev, reqs, pad, seed):
    """Paged KV + q / r / gate / sinks for requests [(sq, prefix)] plus `pad` padding rows."""
    g = torch.Generator(device=dev).manual_seed(seed)
    sks = [sq + p for sq, p in reqs]
    pages = [-(-sk // PAGE) for sk in sks]
    tot = sum(pages) + 4
    perm = torch.randperm(tot, generator=g, device=dev)
    pt = torch.zeros((len(reqs), max(pages)), dtype=torch.int32, device=dev)
    c = 0
    for i, n in enumerate(pages):
        pt[i, :n] = perm[c:c + n].to(torch.int32)
        c += n
    T = sum(sq for sq, _ in reqs) + pad
    kc = (torch.randn((tot, PAGE, KVH, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
    vc = torch.randn((tot, PAGE, KVH, HD), device=dev, generator=g).to(torch.bfloat16)
    q = (torch.randn((T, HEADS, HD), device=dev, generator=g) * 1.5).to(torch.bfloat16)
    r = (torch.randn((T, HEADS, DREL), device=dev, generator=g) * 2.0).to(torch.bfloat16)
    qkvr = (torch.randn((T, QKVR), device=dev, generator=g) * 2.0).to(torch.bfloat16)
    gate = qkvr[:, 5504:5504 + HEADS * HD]       # [q 3072 | k 1024 | v 1024 | rel 384 | gate 3072 | sinks 24]
    sinks = qkvr[:, 5504 + HEADS * HD:]
    cu = [0]
    for sq, _ in reqs:
        cu.append(cu[-1] + sq)
    return dict(q=q, k_cache=kc, v_cache=vc, page_table=pt, cu_seqlens_q=torch.tensor(cu, dtype=torch.int32, device=dev),
                seqused_k=torch.tensor(sks, dtype=torch.int32, device=dev), r=r), gate, sinks, cu[-1], qkvr


def bits(t):
    import torch

    return t.contiguous().view(torch.int16 if t.element_size() == 2 else torch.int32)


def main():
    import torch

    import inkling_attn
    from inkling_attn import ops, patch

    inkling_attn.install(("prefill_swa",))
    import sglang.srt.models.inkling_common.inkling2_attn as AM  # patched gate_sink (INKLING_ATTN_GATE=1)
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink as stock_gate_sink

    dev = torch.device("cuda")
    proj = load_proj(torch, dev)
    log("gate_sink patched:", getattr(AM.gate_sink, patch.MARK, False), "GATE:", patch.GATE)
    if not getattr(AM.gate_sink, patch.MARK, False):
        FAILED.append("gate_sink not patched")
    cases = [("8192 prefix 0", [(8192, 0)], 0), ("2048 prefix 8192", [(2048, 8192)], 0),
             ("3 requests + 77 padding rows", [(700, 0), (1500, 3000), (33, 900)], 77)]
    for name, reqs, pad in cases:
        for warps in (4, 8):
            kw, gate, sinks, rows, _ = case(torch, dev, reqs, pad, seed=len(name) + warps)
            T = kw["q"].shape[0]
            common = dict(kw, proj=proj, softmax_scale=SCALE, window_left=WINDOW - 1, warps=warps)
            out = torch.empty((T, HEADS, HD), dtype=torch.float32, device=dev)
            out, lse = ops.prefill_attention(out=out, **common)
            ref = stock_gate_sink(out.reshape(T, -1), gate, lse.T, sinks, output_dtype=torch.bfloat16)
            y = torch.empty((T, HEADS * HD), dtype=torch.bfloat16, device=dev)
            lse2 = torch.empty_like(lse)
            y_, lse2 = ops.prefill_attention(out=torch.empty_like(out), lse=lse2, gate=gate, sinks=sinks, y=y,
                                             **common)
            torch.cuda.synchronize()
            ok_y = y_ is y and torch.equal(bits(y[:rows]), bits(ref[:rows]))
            ok_l = torch.equal(bits(lse[:, :rows]), bits(lse2[:, :rows]))
            nd = int((bits(y[:rows]) != bits(ref[:rows])).sum())
            log(f"{name}, {warps} warps: y {'BIT-IDENTICAL' if ok_y else f'DIFFERS ({nd} elements)'}, "
                f"lse {'BIT-IDENTICAL' if ok_l else 'DIFFERS'}")
            if not (ok_y and ok_l):
                FAILED.append(f"bits {name} w{warps}")
            if name.startswith("3 requests") and warps == 4:
                # patched gate_sink after a gated launch; mismatch raises; layout refusal; stock outside
                call = dict(common, out=torch.empty_like(out))
                call.pop("warps")
                ok = True
                with patch.use_visit(kw["r"], proj, WINDOW - 1) as v:
                    v.gate, v.sinks, v.tokens = gate, sinks, T
                    ok = ok and patch._gate_ok(v, call)
                    yv = torch.empty_like(y)
                    _, lv = ops.prefill_attention(**call, gate=gate, sinks=sinks, y=yv)
                    v.gated = (call["out"], yv)
                    yy = AM.gate_sink(call["out"].reshape(T, -1), gate, lv.T, sinks, output_dtype=torch.bfloat16)
                    ok = ok and yy is yv and v.gated is None and torch.equal(bits(yy[:rows]), bits(ref[:rows]))
                    v.gated = (call["out"], yv)
                    try:
                        AM.gate_sink(torch.empty_like(out).reshape(T, -1), gate, lv.T, sinks,
                                     output_dtype=torch.bfloat16)
                        ok = False
                    except RuntimeError:
                        pass
                    v.gated = None
                    g_odd = torch.empty((T, HEADS * HD + 1), dtype=torch.bfloat16, device=dev)[:, 1:]
                    v.gate = g_odd
                    ok = ok and not patch._gate_ok(v, call)
                    v.gate = gate
                ys = AM.gate_sink(out.reshape(T, -1), gate, lse.T, sinks, output_dtype=torch.bfloat16)  # no visit
                torch.cuda.synchronize()
                ok = ok and torch.equal(bits(ys), bits(ref))
                log(f"patched gate_sink (gated visit, mismatch, layout refusal, stock outside): {'OK' if ok else 'FAIL'} "
                    f"{dict(patch.STATS)}")
                if not ok:
                    FAILED.append("patched gate_sink")
                # CUDA graph replay of the gated launch with new inputs
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                yg = torch.empty_like(y)
                with torch.cuda.stream(st):
                    ops.prefill_attention(out=torch.empty_like(out), gate=gate, sinks=sinks, y=yg, **common)
                torch.cuda.current_stream().wait_stream(st)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    ops.prefill_attention(out=torch.empty_like(out), gate=gate, sinks=sinks, y=yg, **common)
                kw["q"].mul_(0.7)
                gate.mul_(-1.0)
                gr.replay()
                o3, l3 = ops.prefill_attention(out=torch.empty_like(out), **common)
                r3 = stock_gate_sink(o3.reshape(T, -1), gate, l3.T, sinks, output_dtype=torch.bfloat16)
                torch.cuda.synchronize()
                ok = torch.equal(bits(yg[:rows]), bits(r3[:rows]))
                log(f"graph replay after input change: {'BIT-IDENTICAL' if ok else 'DIFFERS'}")
                if not ok:
                    FAILED.append("graph")
                # a smaller replay in the same captured bucket: the last request (33 rows) shrinks to 10 rows; every
                # row past the new cu_seqlens_q[-1] must be zero in y (no stale rows from the larger replay)
                cu, sk = common["cu_seqlens_q"], common["seqused_k"]
                shrink = int(cu[-1] - cu[-2]) - 10
                cu[-1] -= shrink
                sk[-1] -= shrink
                gr.replay()
                o4, l4 = ops.prefill_attention(out=torch.empty_like(out), **common)
                r4 = stock_gate_sink(o4.reshape(T, -1), gate, l4.T, sinks, output_dtype=torch.bfloat16)
                torch.cuda.synchronize()
                end = int(cu[-1])
                live = torch.equal(bits(yg[:end]), bits(r4[:end]))
                tail = bool((yg[end:] == 0).all())
                log(f"smaller replay ({end} of {T} rows): live rows {'BIT-IDENTICAL' if live else 'DIFFER'}, "
                    f"tail rows {'ZERO' if tail else 'STALE'}")
                if not (live and tail):
                    FAILED.append("smaller replay")
                cu[-1] += shrink
                sk[-1] += shrink
    # registers / spills
    from inkling_attn import build

    so = [os.path.join(build.INFO["build_dir"], f) for f in os.listdir(build.INFO["build_dir"]) if f.endswith(".so")]
    try:
        pr = subprocess.run(["cuobjdump", "-res-usage", so[0]], capture_output=True, text=True, timeout=120)
        if pr.returncode != 0 or "REG:" not in pr.stdout:
            log(f"cuobjdump failed (rc {pr.returncode}): {pr.stderr[:200]}")
        res = pr.stdout
        cur = None
        for line in res.splitlines():
            m = re.search(r"Function (\S*prefill_fwd\S*)", line)
            if m:
                cur = m.group(1)
            elif cur and "REG:" in line:
                tmpl = re.sub(r"_ZN12inkling_attn11prefill_fwdIL", "", cur)[:60]
                log(f"  {tmpl}: {line.strip()}")
                cur = None
    except Exception as exc:  # noqa: BLE001
        log(f"cuobjdump unavailable: {exc!r}")
    # timing (cold L2)
    flush = (torch.empty((), device=dev), torch.ones(128 * (1 << 20) // 4, device=dev))

    def bench(fn, iters=15):
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

    for warps in (4, 8):
        kw, gate, sinks, rows, _ = case(torch, dev, [(8192, 0)], 0, seed=11)
        T = kw["q"].shape[0]
        common = dict(kw, proj=proj, softmax_scale=SCALE, window_left=WINDOW - 1, warps=warps)
        out = torch.empty((T, HEADS, HD), dtype=torch.float32, device=dev)
        y = torch.empty((T, HEADS * HD), dtype=torch.bfloat16, device=dev)
        t_attn = bench(lambda: ops.prefill_attention(out=out, **common))

        def ref_pair():
            o, l_ = ops.prefill_attention(out=out, **common)
            return stock_gate_sink(o.reshape(T, -1), gate, l_.T, sinks, output_dtype=torch.bfloat16)

        t_ref = bench(ref_pair)
        t_gated = bench(lambda: ops.prefill_attention(out=out, gate=gate, sinks=sinks, y=y, **common))
        log(f"8192 rows prefix 0, {warps} warps: attention alone {t_attn:.0f} us; attention + gate_sink {t_ref:.0f} us; "
            f"gated {t_gated:.0f} us -> saved {t_ref - t_gated:+.0f} us per SWA visit "
            f"({40 * (t_ref - t_gated) / 1e3:+.1f} ms per 8K chunk over 40 visits); mainloop+epilogue vs attention "
            f"alone {t_gated - t_attn:+.0f} us")
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
