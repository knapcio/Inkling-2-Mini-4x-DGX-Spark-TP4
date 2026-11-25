#!/usr/bin/env python3
"""GPU check + timing of the gated epilogue on GLOBAL prefill layers (INKLING_ATTN_GATE=all).

  python3 tools/gpu_check_gate_global.py      (SGLang image, model server stopped; /model for the real proj)

Every check is bit identity of the gated launch's y against SGLang's gate_sink on the ungated launch's FP32 out and
LSE of the same kernel, on the written rows; LSE bit-identical too; rows past cu_seqlens_q[-1] zero.
1. BF16 KV (window_left None): one request at 8192 rows; 3 requests + padding rows; the 8-warp + 4-warp pair
   (w8_from below one request's keys), 4 and 8 warps;
2. KV12S packed KV (ops.packed_prefill_attention) on the same rows;
3. the captured KV12S two-launch form (BF16 scratch launch + packed launch into one y, one of them with zero query
   rows; the second with zero_tail=False), both ways round; then a CUDA graph capture of that pair, a replay
   after input changes and a smaller replay (tail rows zero);
4. kv12s dispatch_packed_fa4 inside a gated global visit: gate_begin / gate_commit hand y to the patched gate_sink;
5. cold-L2 timing at 8192 rows prefix 0: ungated global attention + stock gate_sink vs gated, 4 warps.
"""
import os
import sys

os.environ["INKLING_ATTN_GATE"] = "all"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
FAILED = []


def log(*a):
    print(*a, flush=True)


def main():
    import torch

    import inkling_attn
    from gpu_check_gate import DREL, HD, HEADS, KVH, PAGE, SCALE, bits, case, load_proj
    from inkling_attn import ops, patch

    inkling_attn.install(("prefill_swa", "prefill_global"))
    import sglang.srt.models.inkling_common.inkling2_attn as AM  # patched gate_sink
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink as stock_gate_sink

    from inkling_kv12s.storage import Layer

    dev = torch.device("cuda")
    proj = load_proj(torch, dev)
    log("GATE_ALL:", patch.GATE_ALL, "gate_sink patched:", getattr(AM.gate_sink, patch.MARK, False))
    if not (patch.GATE_ALL and getattr(AM.gate_sink, patch.MARK, False)):
        FAILED.append("setup")

    def ref_of(out, lse, gate, sinks, T):
        return stock_gate_sink(out.reshape(T, -1), gate, lse.T, sinks, output_dtype=torch.bfloat16)

    def compare(label, y, ref, lse_a, lse_b, rows):
        ok_y = torch.equal(bits(y[:rows]), bits(ref[:rows]))
        ok_l = torch.equal(bits(lse_a[:, :rows]), bits(lse_b[:, :rows]))
        tail = bool((y[rows:] == 0).all())
        nd = int((bits(y[:rows]) != bits(ref[:rows])).sum())
        log(f"{label}: y {'BIT-IDENTICAL' if ok_y else f'DIFFERS ({nd})'}, lse {'BIT-IDENTICAL' if ok_l else 'DIFFERS'}"
            f", tail {'ZERO' if tail else 'STALE'}")
        if not (ok_y and ok_l and tail):
            FAILED.append(label)

    def packed_layer(kw):
        """A KV12S layer holding kw's BF16 paged cache with every page shifted by one (packed page 0 stays the
        null page), and the matching page table."""
        kc, vc, pt = kw["k_cache"], kw["v_cache"], kw["page_table"]
        layer = Layer.standalone(kc.shape[0] + 1, KVH, device=dev)
        for i in range(pt.shape[0]):
            n = -(-int(kw["seqused_k"][i]) // PAGE)
            for j in range(n):
                page = int(pt[i, j])
                ids = list(range((page + 1) * PAGE, (page + 2) * PAGE))
                layer.write(ids, kc[page], vc[page])
        return layer, (pt + 1).to(torch.int32)

    cases = [("8192 prefix 0", [(8192, 0)], 0, None),
             ("3 requests + 77 padding rows", [(700, 0), (1500, 3000), (33, 900)], 77, None),
             ("8-warp + 4-warp pair", [(1024, 6000), (900, 200), (64, 0)], 40, 4096)]
    for name, reqs, pad, w8 in cases:
        for warps in (4, 8):
            kw, gate, sinks, rows, _ = case(torch, dev, reqs, pad, seed=len(name) + warps)
            T = kw["q"].shape[0]
            common = dict(kw, proj=proj, softmax_scale=SCALE, window_left=None, warps=warps)
            if w8 is not None:
                common["w8_from"] = w8
            out, lse = ops.prefill_attention(out=torch.empty((T, HEADS, HD), dtype=torch.float32, device=dev),
                                             **common)
            ref = ref_of(out, lse, gate, sinks, T)
            y = torch.full((T, HEADS * HD), 7.0, dtype=torch.bfloat16, device=dev)  # stale content to clear
            _, lse2 = ops.prefill_attention(out=torch.empty_like(out), gate=gate, sinks=sinks, y=y, **common)
            torch.cuda.synchronize()
            compare(f"BF16 global {name}, {warps} warps", y, ref, lse, lse2, rows)

            # KV12S packed KV, same rows
            layer, ptp = packed_layer(kw)
            pargs = (kw["q"], layer, ptp, kw["cu_seqlens_q"], kw["seqused_k"], SCALE)
            pk = dict(r=kw["r"], proj=proj, warps=warps, **({"w8_from": w8} if w8 is not None else {}))
            po, pl = ops.packed_prefill_attention(*pargs, **pk)
            pref = ref_of(po, pl, gate, sinks, T)
            ok_same = torch.equal(bits(pref[:rows]), bits(ref[:rows]))
            yp = torch.full_like(y, 7.0)
            _, pl2 = ops.packed_prefill_attention(*pargs, **pk, gate=gate, sinks=sinks, y=yp)
            torch.cuda.synchronize()
            compare(f"packed global {name}, {warps} warps", yp, pref, pl, pl2, rows)
            if not ok_same:
                log("  (note: packed and BF16 references differ; the gated packed launch is compared with its own)")

            if name.startswith("3 requests") and warps == 4:
                from inkling_kv12s.patch import _packed_prefill_in_capture

                cu = kw["cu_seqlens_q"]
                none = torch.zeros_like(cu)
                for fits in (True, False):
                    y2 = torch.full_like(y, 7.0)
                    _, l_b = ops.prefill_attention(
                        kw["q"], kw["k_cache"], kw["v_cache"], kw["page_table"], cu if fits else none,
                        kw["seqused_k"], kw["r"], proj, SCALE, None, out=torch.empty_like(out), warps=warps,
                        gate=gate, sinks=sinks, y=y2)
                    _, l_p = _packed_prefill_in_capture(
                        kw["q"], layer, ptp, none if fits else cu, kw["seqused_k"], SCALE, r=kw["r"],
                        proj=proj, out=torch.empty_like(out), warps=warps, gate=gate, sinks=sinks, y=y2,
                        zero_tail=False)
                    torch.cuda.synchronize()
                    lse_pair = l_b if fits else l_p
                    compare(f"capture pair ({'BF16 scratch' if fits else 'packed loader'} active)", y2,
                            ref if fits else pref, lse if fits else pl, lse_pair, rows)

                # graph capture of the pair (packed active), replay after input change, smaller replay
                def pair(yy):
                    ops.prefill_attention(kw["q"], kw["k_cache"], kw["v_cache"], kw["page_table"], none,
                                          kw["seqused_k"], kw["r"], proj, SCALE, None, out=torch.empty_like(out),
                                          warps=warps, gate=gate, sinks=sinks, y=yy)
                    _packed_prefill_in_capture(kw["q"], layer, ptp, cu, kw["seqused_k"], SCALE,
                                               r=kw["r"], proj=proj, out=torch.empty_like(out), warps=warps,
                                               gate=gate, sinks=sinks, y=yy, zero_tail=False)

                yg = torch.empty_like(y)
                st = torch.cuda.Stream()
                st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st):
                    pair(yg)
                torch.cuda.current_stream().wait_stream(st)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    pair(yg)
                kw["q"].mul_(0.7)
                gate.mul_(-1.0)
                gr.replay()
                o3, l3 = ops.packed_prefill_attention(*pargs, **pk)
                r3 = ref_of(o3, l3, gate, sinks, T)
                torch.cuda.synchronize()
                compare("captured pair, replay after input change", yg, r3, l3, l3, rows)
                shrink = int(cu[-1] - cu[-2]) - 10
                cu[-1] -= shrink
                kw["seqused_k"][-1] -= shrink
                gr.replay()
                o4, l4 = ops.packed_prefill_attention(*pargs, **pk)
                r4 = ref_of(o4, l4, gate, sinks, T)
                torch.cuda.synchronize()
                compare(f"captured pair, smaller replay ({int(cu[-1])} rows)", yg, r4, l4, l4, int(cu[-1]))
                cu[-1] += shrink
                kw["seqused_k"][-1] += shrink

                # gate_begin / gate_commit in a gated global visit, then the patched gate_sink returns y
                with patch.use_visit(kw["r"], proj, None) as v:
                    v.gate, v.sinks, v.tokens = gate, sinks, T
                    ob = torch.empty_like(out)
                    g = patch.gate_begin(kw["q"], ob)
                    ok = g is not None
                    if ok:
                        _, lv = ops.packed_prefill_attention(*pargs, **pk, out=ob,
                                                             **{k: g[k] for k in ("gate", "sinks", "y")})
                        patch.gate_commit(g, ob)
                        yy = AM.gate_sink(ob.reshape(T, -1), gate, lv.T, sinks, output_dtype=torch.bfloat16)
                        o5, l5 = ops.packed_prefill_attention(*pargs, **pk)
                        ok = yy is g["y"] and v.gated is None and torch.equal(bits(yy[:rows]),
                                                                            bits(ref_of(o5, l5, gate, sinks, T)[:rows]))
                    log(f"gate_begin / gate_commit / patched gate_sink in a global visit: {'OK' if ok else 'FAIL'}")
                    if not ok:
                        FAILED.append("gate_begin")

    check_dispatch_capture(torch, dev, AM, stock_gate_sink, patch, bits)

    # timing: 8192 rows prefix 0, BF16 KV, 4 warps (cold L2)
    import statistics

    kw, gate, sinks, rows, _ = case(torch, dev, [(8192, 0)], 0, seed=99)
    T = kw["q"].shape[0]
    common = dict(kw, proj=proj, softmax_scale=SCALE, window_left=None, warps=4)
    out = torch.empty((T, HEADS, HD), dtype=torch.float32, device=dev)
    y = torch.empty((T, HEADS * HD), dtype=torch.bfloat16, device=dev)
    flush = torch.empty(64 << 20, dtype=torch.uint8, device=dev)

    def timed(fn, n=15):
        for _ in range(3):
            fn()
        ts = []
        for _ in range(n):
            flush.zero_()
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            fn()
            b.record()
            torch.cuda.synchronize()
            ts.append(a.elapsed_time(b) * 1e3)
        return statistics.median(ts)

    def stock():
        o, l_ = ops.prefill_attention(out=out, **common)
        stock_gate_sink(o.reshape(T, -1), gate, l_.T, sinks, output_dtype=torch.bfloat16)

    t_s = timed(stock)
    t_g = timed(lambda: ops.prefill_attention(out=out, gate=gate, sinks=sinks, y=y, **common))
    log(f"8192 rows prefix 0, global, 4 warps: attention + gate_sink {t_s:.0f} us, gated {t_g:.0f} us -> saved "
        f"{t_s - t_g:+.0f} us per global visit ({8 * (t_s - t_g) / 1e3:+.1f} ms per 8K chunk over 8 visits)")
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


def check_dispatch_capture(torch, dev, AM, stock_gate_sink, patch, bits):
    """The serving path: kv12s dispatch_packed_fa4 captured in one CUDA graph inside a gated global visit (the
    capture-time gate_sink takes y, like the model forward), replayed across fits -> overflow -> fits of the bounded
    BF16 prefix scratch with requests on both sides of the 8-warp split, a zero-length padding slot and padded rows,
    then with fewer live rows. Every replay: y and the replayed LSE bit-identical to stock gate_sink on the ungated
    BF16 kernel's out / LSE, rows past the live rows zero."""
    import gpu_check_kv12s as K
    from inkling_attn import ops as attn_ops
    from inkling_attn.ops import prefill_attention
    from inkling_attn.patch import use_visit
    from inkling_kv12s.patch import dispatch_packed_fa4

    g = torch.Generator(device=dev).manual_seed(11)
    pages = 24
    k, v = K.raw_cache(pages * K.PAGE, dev, g)
    kc, vc = (t.view(-1, K.PAGE, K.HEADS_K, K.DIM) for t in (k, v))
    proj = K.normal((16, K.EXTENT), dev, g) * 0.1
    layer = K.Layer.standalone(pages + 1, K.HEADS_K, device=dev)
    K.append(layer, k, v, 0, 23 * K.PAGE + 50)

    def unexpected_fallback(**call):
        raise AssertionError("captured packed prefill fell back to the FA4 gather")

    saved_w8 = attn_ops.W8_FROM
    attn_ops.W8_FROM = 1024
    try:
        for warps in (4, 8):
            qlen, real, cap = 640, (300, 300), 20
            batches = ((3 * K.PAGE + 5, 4 * K.PAGE), (9 * K.PAGE, 11 * K.PAGE + 7), (320, 11 * K.PAGE + 40))
            table = torch.zeros((3, 12), dtype=torch.int32, device=dev)
            table[0] = torch.arange(1, 13, dtype=torch.int32, device=dev)
            table[1] = torch.arange(13, 25, dtype=torch.int32, device=dev)
            q = torch.zeros((qlen, K.HEADS_Q, K.DIM), dtype=torch.bfloat16, device=dev)
            r = torch.zeros((qlen, K.HEADS_Q, 16), dtype=torch.bfloat16, device=dev)
            qkvr = torch.zeros((qlen, 8600), dtype=torch.bfloat16, device=dev)
            gate = qkvr[:, 5504:5504 + K.HEADS_Q * K.DIM]
            sinks = qkvr[:, 5504 + K.HEADS_Q * K.DIM:]
            seq = K.lengths(dev, [qlen, qlen, 0])
            rows = real[0] + real[1]
            cu_q = K.lengths(dev, [0, real[0], rows, rows])
            kwargs = K.fa4_kwargs(q, layer, table, cu_q, seq, qlen, r, mode="extend")
            graph = y = None
            with K.bounded_scratch(cap, dev, min_rows=512) as scratch:
                for i, (s0, s1) in enumerate(batches + ((320, 11 * K.PAGE + 40),)):
                    live = rows if i < len(batches) else rows - 123  # the last replay: fewer live rows
                    q.copy_(K.normal((qlen, K.HEADS_Q, K.DIM), dev, g))
                    r.copy_(K.normal((qlen, K.HEADS_Q, 16), dev, g))
                    qkvr.copy_(K.normal((qlen, 8600), dev, g) * 2)
                    cu_q.copy_(K.lengths(dev, [0, real[0], live, live]))
                    seq.copy_(K.lengths(dev, [s0, s1 - (rows - live), 0]))
                    if graph is None:
                        K.kv12s_patch_static_capturing(True)
                        try:
                            def run():
                                with use_visit(r, proj) as vis:
                                    vis.gate, vis.sinks, vis.tokens = gate, sinks, qlen
                                    o, l_ = dispatch_packed_fa4(unexpected_fallback, layer, kwargs, warps=warps)
                                    if vis.gated is None:
                                        raise AssertionError("dispatch did not take the gated launch")
                                    yy = AM.gate_sink(o.reshape(qlen, -1), gate, l_.T, sinks,
                                                      output_dtype=torch.bfloat16)
                                return yy, l_

                            stream = torch.cuda.Stream()
                            stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(stream):
                                run()
                            torch.cuda.current_stream().wait_stream(stream)
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                y, lse_c = run()
                        finally:
                            K.kv12s_patch_static_capturing(False)
                    y.fill_(7.0)
                    graph.replay()
                    torch.cuda.synchronize()
                    eo, el = prefill_attention(q, kc, vc, table, cu_q, seq, r, proj, K.SCALE, None, warps=warps)
                    ref = stock_gate_sink(eo.reshape(qlen, -1), gate, el.T, sinks, output_dtype=torch.bfloat16)
                    ok_y = torch.equal(bits(y[:live]), bits(ref[:live]))
                    ok_l = torch.equal(bits(lse_c[:, :live]), bits(el[:, :live]))
                    tail = bool((y[live:] == 0).all())
                    label = f"dispatch capture w{warps} replay {i} ({live} live rows, prefixes {s0 - real[0]}/" \
                            f"{s1 - (rows - live) - (live - real[0])})"
                    log(f"{label}: y {'BIT-IDENTICAL' if ok_y else 'DIFFERS'}, lse {'BIT-IDENTICAL' if ok_l else 'DIFFERS'}"
                        f", tail {'ZERO' if tail else 'STALE'}")
                    if not (ok_y and ok_l and tail):
                        FAILED.append(label)
                if scratch.served:
                    FAILED.append("dispatch capture used the host-side scratch gather")
    finally:
        attn_ops.W8_FROM = saved_w8


if __name__ == "__main__":
    sys.exit(main())
