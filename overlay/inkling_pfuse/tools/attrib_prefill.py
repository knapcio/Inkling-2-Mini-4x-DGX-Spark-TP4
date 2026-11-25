#!/usr/bin/env python3
"""Per-kernel attribution of the prefill elementwise remainder (stock SGLang kernels).

The prefill torch-profiler traces (R2PP-*) hold no prefill GPU work, so this rebuilds the remainder from the
model's control flow: it walks the 48 layer visits of one T-token chunk exactly like Inkling2CausalLLM._run_loop
(attn-res state, loop message bank, block boundaries), records every elementwise / glue kernel call with its
shape, then times each distinct call once on the GPU with the stock kernel (L2 flushed by reading a 128 MiB
buffer before each rep, CUDA events, median of --reps) and multiplies by its count.

Excluded (other budget lines): qkvr / wo / router / dyn-conv GEMMs, rel-bias, FA4, MoE grouped GEMMs and MoE glue.

  python3 tools/attrib_prefill.py [--tokens 8192] [--reps 20] [--json out.json]
"""
import argparse
import collections
import json
import statistics
import sys

import torch

D, HEADS, KVH, HD, DREL, RANK = 2304, 24, 8, 128, 16, 64
LOCAL = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22}
CORE = range(6, 18)
PASSES = 3
BLOCK = 6
QKVR_N = 8600


def walk():
    """Kernel calls of one prefill chunk: list of (kind, params, tag)."""
    ops = []

    def op(kind, tag, **p):
        ops.append((kind, tuple(sorted(p.items())), tag))

    state = {"blocks": 1, "has_partial": True}  # blocks counted (embed is the partial at start)
    # embed_norm (bf16 -> bf16) of the embeddings
    op("rms_norm", "embed", rows=1, src="bf16")

    def layer(lid, messages, st):
        local = lid in LOCAL
        boundary = lid % BLOCK == 0
        nq = 1 if local else 3
        # attention input
        if st["blocks"] == 0:
            src_fp32 = False
        else:
            op("attn_res", "attn_in", n=st["blocks"] + 1, q=nq, addend=False)
            src_fp32 = True
        if messages:
            op("attn_res", "msg_attn", n=2, q=3, addend=False)
            op("ctrl", "msg_attn", num=3)
            if local:
                op("rms_norm", "attn_norm", rows=3, src="fp32")
            else:
                for _ in range(3):
                    op("rms_norm", "qkv_norm", rows=1, src="fp32")
        elif local:
            op("rms_norm", "attn_norm", rows=1, src="fp32" if src_fp32 else "bf16")
        else:
            for _ in range(3):
                op("rms_norm", "qkv_norm", rows=1, src="fp32" if src_fp32 else "bf16")
        # prologue
        for name in ("k", "v"):
            op("sconv", name + "_sconv", width=KVH * HD, dynamic=not local)
            op("sconv_update", name + "_sconv", width=KVH * HD)
        op("clip", "q_clip", heads=HEADS, dim=HD, scaled=True, tau=not local, strided=True)
        op("clip", "k_clip", heads=KVH, dim=HD, scaled=False, tau=False, strided=False)
        op("vnorm", "v_norm")
        op("clip", "rel_clip", heads=HEADS, dim=DREL, scaled=True, tau=not local, strided=True)
        op("kv_store", "kv_store")
        op("gate_sink", "gate_sink")
        op("sconv", "attn_sconv", width=D, dynamic=False)
        op("sconv_update", "attn_sconv", width=D)
        # mlp input
        if boundary:
            op("attn_res", "mlp_in", n=st["blocks"] + 2, q=1, addend=False)
        else:
            op("attn_res", "mlp_in", n=st["blocks"] + 1, q=1, addend=True)
        if messages:
            op("attn_res", "msg_mlp", n=2, q=1, addend=False)
            op("ctrl", "msg_mlp", num=1)
        op("rms_norm", "mlp_norm", rows=1, src="fp32")
        op("sconv", "mlp_sconv", width=D, dynamic=False)
        op("sconv_update", "mlp_sconv", width=D)
        if boundary:
            st["blocks"] += 1
            op("add2", "finalize")
        else:
            op("residual_add", "finalize")

    # Layer 0 sees no blocks: the embedding is the partial
    st = {"blocks": 0}
    for lid in range(0, CORE.start):
        layer(lid, False, st)
    prelude_blocks = st["blocks"]
    for p in range(PASSES):
        st = {"blocks": prelude_blocks}
        for lid in CORE:
            layer(lid, p > 0, st)
    for lid in range(CORE.stop, 24):
        layer(lid, False, st)
    op("attn_res", "output", n=st["blocks"] + 1, q=1, addend=False)
    op("rms_norm", "final_norm", rows=1, src="fp32")
    op("aux_copy", "dspark_aux", n=4)
    return ops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--json")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    ops = walk()
    calls = collections.Counter((k, p) for k, p, _ in ops)
    tags = collections.defaultdict(collections.Counter)
    for k, p, t in ops:
        tags[(k, p)][t] += 1
    if a.dry:
        for (k, p), c in sorted(calls.items()):
            print(c, k, dict(p), dict(tags[(k, p)]))
        return
    T = a.tokens
    dev = torch.device("cuda")
    torch.manual_seed(0)
    from sglang.kernels.ops.attention.inkling2_attn import gate_sink
    from sglang.kernels.ops.attention.inkling2_attn_res import attn_res
    from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers
    from sglang.kernels.ops.elementwise.inkling2_residual_add import residual_add
    from sglang.kernels.ops.layernorm.inkling2_qk_l2_clip import qk_l2_clip_fwd_cutedsl
    from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm
    from sglang.kernels.ops.mamba.inkling2_sconv import short_convolution

    flush = torch.empty(32 << 20, dtype=torch.float32, device=dev).fill_(1.0)

    def bf(*s):
        return (torch.randn(*s, device=dev) * 0.5).to(torch.bfloat16)

    pool = [bf(T, D) for _ in range(6)]
    w = torch.rand(D, device=dev) + 0.5
    qkvr = bf(T, QKVR_N)

    def timeit(fn, nbytes):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(a.reps):
            flush.sum()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        us = statistics.median(ts)
        return us, nbytes / us / 1e3

    meta = {}

    def sconv_meta():
        if "m" not in meta:
            meta["m"] = {
                "safe_idx": torch.ones(1, dtype=torch.int64, device=dev),
                "cache_mask": torch.zeros(1, dtype=torch.bool, device=dev),
                "cu": torch.tensor([0, T], dtype=torch.int64, device=dev),
                "si": torch.zeros(T, dtype=torch.int32, device=dev),
            }
        return meta["m"]

    def build(kind, p):
        p = dict(p)
        B, F = T * D * 2, T * D * 4
        if kind == "attn_res":
            n, q = p["n"], p["q"]
            blocks = pool[: n - 1] + [pool[5]]
            add = pool[4] if p["addend"] else None
            if add is not None:
                blocks = pool[:n]
            qs = torch.randn(q, D, device=dev) * D ** -0.5
            nb = n * B + (B if add is not None else 0) + q * F
            return (lambda: attn_res(blocks, add, qs, normalize_values=True)), nb
        if kind == "ctrl":
            num = p["num"]
            dst = torch.randn(T, D, device=dev)
            dests = (dst,) * num
            msgs = torch.randn(num, T, D, device=dev).unbind(0)
            down = bf(num, RANK, 2 * D) * 0.02
            gain = bf(num, D, RANK) * 0.1
            forget = bf(num, D, RANK) * 0.1
            # norm: 2 reads + bf16 write; down: read normed; mix: 2 reads + fp32 write
            nb = num * (2 * F + 2 * B) + num * 2 * B + num * 3 * F
            return (lambda: message_controllers(dests, msgs, down, gain, forget)), nb
        if kind == "rms_norm":
            rows = p["rows"]
            x = torch.randn(rows, T, D, device=dev) if p["src"] == "fp32" else bf(rows, T, D)
            x = x if rows > 1 else x[0]
            nb = x.numel() * x.element_size() + x.numel() * 2
            return (lambda: rms_norm(x, w, 1e-6, output_dtype=torch.bfloat16)), nb
        if kind == "sconv":
            width, dyn = p["width"], p["dynamic"]
            x = bf(T, width)
            m = sconv_meta()
            cache = torch.zeros(2, 3, width, dtype=torch.bfloat16, device=dev)
            if dyn:
                weight = bf(4, width)
                dw = bf(T, KVH, 4)
            else:
                weight = bf(width, 4)
                dw = None
            nb = 2 * T * width * 2
            return (lambda: short_convolution(x, weight, cache, m, dynamic_weights=dw)), nb
        if kind == "sconv_update":
            return None, 0
        if kind == "clip":
            h, d = p["heads"], p["dim"]
            if p["strided"]:
                x = qkvr[:, : h * d].reshape(T, h, d)
            else:
                x = bf(T, h, d)
            scale = (torch.rand(h, d, device=dev) + 0.5) if p["scaled"] else None
            tau = torch.rand(T, device=dev) + 1.0 if p["tau"] else None
            nb = 2 * T * h * d * 2
            return (lambda: qk_l2_clip_fwd_cutedsl(x, 22.6, 1e-6, scale=scale, output_scale=tau)), nb
        if kind == "vnorm":
            x = bf(T, KVH, HD)
            nb = 2 * T * KVH * HD * 2
            return (lambda: rms_norm(x, None, 1e-6)), nb
        if kind == "gate_sink":
            out = torch.randn(T, HEADS * HD, device=dev)
            lse = torch.randn(HEADS, T, device=dev).t()
            gate = qkvr[:, 4096 : 4096 + HEADS * HD]
            sinks = qkvr[:, 8000 : 8000 + HEADS]
            nb = T * HEADS * HD * (4 + 2 + 2)
            return (lambda: gate_sink(out, gate, lse, sinks, output_dtype=torch.bfloat16)), nb
        if kind == "kv_store":
            k = bf(T, KVH, HD)
            v = bf(T, KVH, HD)
            kp = torch.empty(T + 256, KVH, HD, dtype=torch.bfloat16, device=dev)
            vp = torch.empty_like(kp)
            loc = torch.arange(T, device=dev) + 128

            def f():
                kp[loc] = k
                vp[loc] = v

            return f, 4 * T * KVH * HD * 2
        if kind == "residual_add":
            return (lambda: residual_add(pool[0], pool[1], pool[2])), 4 * B
        if kind == "add2":
            return (lambda: pool[1] + pool[2]), 3 * B
        if kind == "aux_copy":
            return (lambda: torch.cat([pool[0], pool[1], pool[2], pool[3]], dim=-1)), 8 * B
        raise KeyError(kind)

    rows = []
    total = 0.0
    for (k, p), c in sorted(calls.items(), key=lambda x: str(x)):
        fn, nb = build(k, p)
        if fn is None:
            continue
        try:
            us, gbs = timeit(fn, nb)
        except Exception as exc:  # noqa: BLE001
            print(f"{k} {dict(p)} FAILED: {exc!r}", flush=True)
            continue
        total += us * c
        rows.append(dict(kind=k, params=dict(p), calls=c, tags=dict(tags[(k, p)]), us=us, mb=nb / 1e6,
                         gbs=gbs, ms_chunk=us * c / 1e3, model_ms=c * nb / 224e9 * 1e3))
        print(f"{k:12s} {str(dict(p)):60s} x{c:3d} {us:8.1f} us {nb/1e6:8.1f} MB {gbs:6.1f} GB/s "
              f"-> {us*c/1e3:7.2f} ms/chunk (model {c*nb/224e9*1e3:6.2f})", flush=True)
    by_kind = collections.defaultdict(lambda: [0.0, 0.0, 0])
    for r in rows:
        by_kind[r["kind"]][0] += r["ms_chunk"]
        by_kind[r["kind"]][1] += r["model_ms"]
        by_kind[r["kind"]][2] += r["calls"]
    print(f"\nT={T}: total {total/1e3:.1f} ms per chunk")
    for k, (ms, mod, c) in sorted(by_kind.items(), key=lambda x: -x[1][0]):
        print(f"  {k:12s} {c:4d} calls {ms:8.1f} ms  (bytes at 224 GB/s: {mod:7.1f} ms)")
    if a.json:
        json.dump(dict(tokens=T, rows=rows, total_ms=total / 1e3), open(a.json, "w"), indent=1)


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
