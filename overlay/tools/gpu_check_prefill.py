#!/usr/bin/env python3
"""GPU check of the inkling_prefill levers against the stock SGLang kernels.

Runs inside the SGLang container on the Spark with the model server STOPPED. It reads
real layer weights from the checkpoint (qkvr of layers 5 and 6, rel-bias tables of
layers 0 and 5, NVFP4 w13 / w2 scales of layer 5), never changes clocks, and uses
about 3 GB of unified memory.

  python3 /opt/inkling-overlay/tools/gpu_check_prefill.py [--model /model]
          [--sections qkvr,partial,quant,bias,graphs,install] [--iters 20] [--json out.json]
          [--tp 4 [--rank r]]   (qkvr only: one attention-TP rank's qkvr shard, see below)

  scripts/launch.sh down
  docker run --rm --gpus all --ipc=host \
    -v ~/inkling/repo/overlay:/opt/inkling-overlay:ro \
    -v ~/models/Inkling-2-Mini:/model:ro \
    -v ~/.cache/inkling-2-mini-sglang:/root/.cache \
    --entrypoint python3 lmsysorg/sglang:inkling-2-mini \
    /opt/inkling-overlay/tools/gpu_check_prefill.py --json /root/.cache/gpu_check_prefill.json

Per lever it reports BITWISE equality with the stock path (every check must say
'bit-equal'; the process exits 1 otherwise) and the time of stock vs lever:

qkvr     Inkling2Attention._project_sources stock (3M stacked rows x 8600 columns,
         F.linear) vs qkvr_split for every prefill graph
         bucket M >= 128 plus odd sizes (rows per source); all six returned tensors.
         --tp N: the same on rank --rank's shard ([2152, 2304] at TP4: each projection
         narrowed to its rank rows, plus SGLang's zero pad rows to a multiple of 8), with
         the split's TP floor disabled; a non-bit-equal M below the first bucket that
         stays bit-equal up to 8192 is what INKLING_PREFILL_QKVR_TP_MIN_M must exceed.
partial  routed_partial(...).to(bf16) vs partial_bf16.routed_partial_bf16, NVFP4-scaled
         (real w2 amax / denominator), T = 8 ... 8192 tokens.
quant    nvfp4_routed_quantize vs quant_once at T = 300 / 1024 / 8192, uniform and skewed
         routing: values, amax, every scale byte the stock kernel writes, and the
         grouped GEMM w13 output on the real layer-5 weights.
bias     FA4 SM120 prefill attention with the stock rel-bias (einsum + internal shear)
         vs bias_layout (sheared kernel + rel_bias_presheared via the patched SM120
         wrapper): the sheared buffer
         against the stock shear's workspace and against a slow reference, and the
         attention output and LSE; SWA (layer 0) and global (layer 5), chunk 1024 /
         8192 at prefix 0 / 24576, and a 3-request batch.
graphs   each lever captured in a CUDA graph and replayed (bitwise vs eager).
install  INKLING_PREFILL=all in a subprocess, importing the real patch targets:
         every lever must report applied.
"""

import argparse
import importlib
import itertools
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(HERE)
sys.path.insert(0, OVERLAY)

H, INTER, E, TOPK = 2304, 1152, 128, 8
HEADS, KVH, HD = 24, 8, 128
SPLITS = [3072, 1024, 1024, 384, 3072, 24]
PROJ_NAMES = ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")

RESULTS = []

# every prefill CUDA-graph bucket >= 128 tokens (generate_prefill_cuda_graph_batch_sizes(8192))
# verify row counts, and odd eager sizes; the lever never runs below 128 rows (it differs at 64, K5)
QKVR_MS = sorted(set(
    [m for m in (list(range(128, 257, 16)) + list(range(288, 513, 32)) + list(range(576, 1025, 64))
                 + list(range(1280, 4097, 256)) + list(range(4608, 8193, 512)))]
    + list(range(128, 521, 8))  # target verify: 8 rows x 16..65 requests (fused prologue path)
    + [129, 333, 1000, 3001, 7777]))


def log(*a):
    print(*a, flush=True)


def record(section, name, check, stock_us=None, lever_us=None, **extra):
    ok = check == "bit-equal" or check.startswith("bit-equal")
    r = dict(section=section, name=name, check=check, ok=ok, stock_us=stock_us, lever_us=lever_us, **extra)
    RESULTS.append(r)
    t = ""
    if stock_us is not None and lever_us is not None:
        t = f"  stock {stock_us:9.1f} us  lever {lever_us:9.1f} us  ({(1 - lever_us / stock_us) * 100:+5.1f} % time saved)"
    log(f"  [{'PASS' if ok else 'FAIL'}] {section:<8} {name:<52} {check}{t}")


def bitwise(torch, a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return f"MISMATCH shape/dtype {tuple(a.shape)} {a.dtype} vs {tuple(b.shape)} {b.dtype}"
    ia = a.contiguous().view(torch.uint8)
    ib = b.contiguous().view(torch.uint8)
    if torch.equal(ia, ib):
        return "bit-equal"
    d = (a.float() - b.float()).abs()
    d = d[torch.isfinite(d)]
    n = int((ia != ib).sum())
    return f"DIFF {n} bytes, max_abs {float(d.max()) if d.numel() else float('nan'):.3g}"


def bench(torch, fn, iters, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) * 1e3 for a, b in zip(st, en))


class Ckpt:
    def __init__(self, path):
        from safetensors import safe_open

        self.path = path
        with open(os.path.join(path, "model.safetensors.index.json")) as f:
            self.map = json.load(f)["weight_map"]
        self._open = safe_open
        self._files = {}

    def get(self, name, dev):
        fn = self.map[name]
        if fn not in self._files:
            self._files[fn] = self._open(os.path.join(self.path, fn), framework="pt", device="cpu")
        return self._files[fn].get_tensor(name).to(dev)


# ===================================================================== qkvr_split


def sec_qkvr(torch, dev, ck, a):
    import types

    import torch.nn.functional as F

    from inkling_prefill import qkvr_split

    try:
        from sglang.srt.models.inkling_common import inkling2_attn as attn_mod

        stock = attn_mod.Inkling2Attention._project_sources
        stacked_fn = attn_mod._stacked
        log("  qkvr: stock = sglang Inkling2Attention._project_sources")
    except Exception as exc:  # noqa: BLE001
        attn_mod = None
        log(f"  qkvr: sglang inkling2_attn not importable ({exc!r}); using a replica")

        def stacked_fn(s):
            return torch.cat([s.q, s.k, s.v])

        def stock(self, sources):
            q_proj, k_proj, v_proj = (
                p.split(self.split_sizes, dim=-1)[:6] for p in self.qkvr(stacked_fn(sources))[0].chunk(3)
            )
            q, _, _, r, gate, sinks = q_proj
            return q, k_proj[1], v_proj[2], r, gate, sinks

    class Method:
        pass

    Method.__name__ = "UnquantizedLinearMethod"

    class Layer:
        def __init__(self, w):
            self.weight, self.bias, self.quant_method = w, None, Method()

        def __call__(self, x):  # UnquantizedLinearMethod.apply on sm12x for M > 16
            return F.linear(x, self.weight), None

    Sources = getattr(attn_mod, "AttentionSources", None) or (
        lambda q, k, v: types.SimpleNamespace(q=q, k=k, v=v))
    g = torch.Generator(device=dev).manual_seed(0)
    tp, rank = a.tp, a.rank
    assert tp >= 1 and 0 <= rank < tp and all(s % tp == 0 for s in SPLITS), (tp, rank)
    splits = [s // tp for s in SPLITS]
    pad = -sum(splits) % 8  # Inkling2Attention: qkvr_pad_size appended as a 7th split
    if pad:
        splits.append(pad)
    for layer in (5, 6):
        parts = [ck.get(f"model.llm.layers.{layer}.attn.{p}.weight", dev) for p in PROJ_NAMES]
        parts = [x.narrow(0, rank * (x.shape[0] // tp), x.shape[0] // tp) for x in parts]
        if pad:
            parts.append(torch.zeros((pad, H), dtype=parts[0].dtype, device=dev))
        w = torch.cat(parts).contiguous()
        del parts
        assert tuple(w.shape) == (sum(splits), H), w.shape
        # tp_size 1 on the stand-in: this check measures the split itself (no TP floor)
        me = types.SimpleNamespace(split_sizes=list(splits), qkvr=Layer(w), tp_size=1)
        tag = f"tp{tp}r{rank} " if tp > 1 else ""
        for mode in ("strided",):
            fn = qkvr_split.make_project_sources(stock, None, {"min_m": 17, "out": mode})
            for m in QKVR_MS:
                buf = (torch.randn((3, m, H), device=dev, generator=g) * 1.0).to(torch.bfloat16)
                src = Sources(buf[0], buf[1], buf[2])
                ref = stock(me, src)
                got = fn(me, src)
                checks = [bitwise(torch, x, y) for x, y in zip(got, ref)]
                bad = [f"{n}:{c}" for n, c in zip(("q", "k", "v", "r", "gate", "sinks"), checks) if c != "bit-equal"]
                same_shape = all(x.shape == y.shape and x.stride() == y.stride() for x, y in zip(got, ref))
                stride = got[0].stride(0)  # attention_prologue: q, k, v, rel share (stride, 1)
                same_shape = same_shape and all(t.stride() == (stride, 1) for t in got[:4])
                timed = m in (1024, 4096, 8192)
                t_s = bench(torch, lambda: stock(me, src), a.iters) if timed else None
                t_l = bench(torch, lambda: fn(me, src), a.iters) if timed else None
                record("qkvr", f"{tag}layer {layer} {mode:<8} M={m}", "bit-equal" if not bad and same_shape else
                       ("; ".join(bad) or "shape/stride mismatch vs stock"), t_s, t_l)
        del w


# ===================================================================== partial_bf16


def make_routing(torch, dev, t, kind, seed):
    g = torch.Generator(device=dev).manual_seed(seed * 7919 + t)
    if kind == "skew":
        logits = torch.randn((t, E), device=dev, generator=g) + torch.randn((E,), device=dev, generator=g) * 0.5
    else:
        logits = torch.rand((t, E), device=dev, generator=g)
    w, ids = torch.topk(logits, TOPK, dim=1)
    return ids.to(torch.int32), torch.softmax(w.float(), dim=1)


# route_metadata's CUDA path (512 < ids <= 4096) needs > 48 KB of dynamic shared
# memory above ~2370 ids and failed to launch on GB10 in K5 (route.cuh:223,
# cudaErrorInvalidValue). The levers do not depend on which routine sorted the ids
# (both produce the stable expert sort), so above 2048 ids this script uses the
# sort path, which serving uses above 4096 ids.
ROUTE_METADATA_MAX_IDS_HERE = 2048


def sort_route(torch, ids, weights):
    """Inkling2Experts._sm120_route's stable expert sort (route_metadata for small T)."""
    from sglang.kernels.ops.moe.inkling2_route import route_metadata
    from sglang.kernels.ops.moe.inkling_moe import compute_grouped_gemm_metadata, get_src2dst

    flat = ids.flatten()
    if flat.numel() <= ROUTE_METADATA_MAX_IDS_HERE:
        src2dst, dst2src, gammas, _, offsets, *_ = route_metadata(flat, weights.flatten(), 0, E, E)
        return src2dst, dst2src, gammas, offsets
    sorted_ids, dst2src = torch.sort(flat.to(torch.int16), stable=True)
    dst2src = dst2src.int()
    _, offsets, _, _ = compute_grouped_gemm_metadata(sorted_ids, E, block_size_m=128)
    return get_src2dst(dst2src), dst2src, weights.flatten().float()[dst2src], offsets


def sec_partial(torch, dev, ck, a):
    from inkling_prefill import partial_bf16
    from sglang.kernels.ops.moe.inkling2_moe import routed_partial

    w2_amax = ck.get("model.llm.layers.5.mlp.experts.w2_weight_amax", dev).float()
    w2_den = ck.get("model.llm.layers.5.mlp.experts.w2_weight_outer_scale_denominator", dev).float()
    g = torch.Generator(device=dev).manual_seed(1)
    for t in (8, 64, 1024, 8192):
        ids, wts = make_routing(torch, dev, t, "uniform", 1)
        src2dst, dst2src, gammas, offsets = sort_route(torch, ids, wts)
        rows = dst2src.numel()
        down = (torch.randn((rows, H), device=dev, generator=g) * 4).to(torch.bfloat16)
        x_amax = torch.rand((rows,), device=dev, generator=g) * 20 + 0.5
        for scaled in (True, False):
            sc = (x_amax, w2_amax, w2_den) if scaled else None
            ref = routed_partial(down, offsets, src2dst, TOPK, nvfp4_scales=sc).to(torch.bfloat16)
            got = partial_bf16.routed_partial_bf16(down, offsets, src2dst, TOPK, nvfp4_scales=sc)
            ts = bench(torch, lambda: routed_partial(down, offsets, src2dst, TOPK, nvfp4_scales=sc).to(torch.bfloat16), a.iters)
            tl_ = bench(torch, lambda: partial_bf16.routed_partial_bf16(down, offsets, src2dst, TOPK, nvfp4_scales=sc), a.iters)
            record("partial", f"T={t} {'nvfp4-scaled' if scaled else 'unscaled'}", bitwise(torch, got, ref), ts, tl_)


# ===================================================================== quant_once


def written_scale_positions(torch, offsets_cpu, rows, width, dev):
    """Byte positions the stock NVFP4 quantizer writes (valid rows + zeroed tails)."""
    from inkling_prefill import _layout as L

    sk = L.scale_k_of(width)
    off = [int(v) for v in offsets_cpu]
    cols = torch.arange(sk, device=dev)
    pos = []
    for e in range(len(off) - 1):
        count = off[e + 1] - off[e]
        if count == 0:
            continue
        base = L.scale_base(off, e, sk)
        local = torch.arange(-(-count // 128) * 128, device=dev)[:, None]
        p = ((local // 128) * (sk // 4) + cols[None, :] // 4) * 512 + (local % 32) * 16 + ((local // 32) % 4) * 4 + cols[None, :] % 4
        pos.append((base + p).flatten())
    return torch.cat(pos)


def sec_quant(torch, dev, ck, a):
    from inkling_prefill import quant_once
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    stock = ex.nvfp4_routed_quantize
    w13 = ck.get("model.llm.layers.5.mlp.experts.w13_weight", dev)
    w13_scale = swizzle_blockscale(ck.get("model.llm.layers.5.mlp.experts.w13_weight_scale", dev)).flatten(1)
    g = torch.Generator(device=dev).manual_seed(2)
    for t in (300, 1024, 8192):
        for kind in ("uniform", "skew"):
            ids, wts = make_routing(torch, dev, t, kind, 2)
            _, dst2src, _, offsets = sort_route(torch, ids, wts)
            x = (torch.randn((t, H), device=dev, generator=g) * 3).to(torch.bfloat16)
            v0, s0, a0 = stock(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets)
            v1, s1, a1 = quant_once.quant_once(stock, ex._nvfp4_outputs, x, dst2src, TOPK, offsets)
            pos = written_scale_positions(torch, offsets.cpu().tolist(), dst2src.numel(), H, dev)
            checks = {
                "values": bitwise(torch, v1, v0),
                "amax": bitwise(torch, a1, a0),
                "scales": bitwise(torch, s1.flatten()[pos], s0.flatten()[pos]),
            }
            g0 = grouped.nvfp4_expert_matmul(v0, s0, w13, w13_scale, offsets, t)
            g1 = grouped.nvfp4_expert_matmul(v1, s1, w13, w13_scale, offsets, t)
            checks["gemm_w13"] = bitwise(torch, g1, g0)
            bad = [f"{k}:{c}" for k, c in checks.items() if c != "bit-equal"]
            ts = bench(torch, lambda: stock(x=x, dst2src=dst2src, top_k=TOPK, offsets=offsets), a.iters)
            tl_ = bench(torch, lambda: quant_once.quant_once(stock, ex._nvfp4_outputs, x, dst2src, TOPK, offsets), a.iters)
            record("quant", f"T={t} {kind} (values, amax, {pos.numel()} scale bytes, GEMM)",
                   "bit-equal" if not bad else "; ".join(bad), ts, tl_)


# ===================================================================== bias_layout


def sec_bias(torch, dev, ck, a):
    from inkling_prefill import bias_layout
    from sglang.kernels.ops.attention import flash_attention_v4_sm120 as sm120

    interface = importlib.import_module(bias_layout.INTERFACE_MODULE)
    bias_layout.patch_sm120(sm120)  # forwards rel_bias_presheared for tagged tensors
    fa = sm120.flash_attn_with_kvcache
    page = 128
    g = torch.Generator(device=dev).manual_seed(3)
    layers = {"swa": 0, "global": 5}
    cases = [("1 req", [(1024, 0)]), ("1 req", [(8192, 0)]), ("1 req", [(8192, 24576)]), ("1 req", [(1024, 24576)]),
             ("3 reqs", [(700, 0), (2000, 5000), (333, 24576)])]
    max_tokens = max(sum(c + p for c, p in reqs) for _, reqs in cases)
    pages_total = -(-max_tokens // page) + 8 * len(cases)
    kc = (torch.randn((pages_total, page, KVH, HD), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    vc = (torch.randn((pages_total, page, KVH, HD), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    recorded = []
    orig_empty = interface._shear_bias_empty

    def recording_empty(*args, **kw):
        buf = orig_empty(*args, **kw)
        recorded.append(buf)
        return buf

    for kind, layer in layers.items():
        proj = ck.get(f"model.llm.layers.{layer}.attn.rel_logits_proj.proj", dev).to(torch.bfloat16).contiguous()
        local = kind == "swa"
        ws = (1023, 0) if local else (-1, -1)
        for label, reqs in cases:
            lens = [c for c, _ in reqs]
            seqk = [c + p for c, p in reqs]
            total = sum(lens)
            npages = [-(-s // page) for s in seqk]
            ptab = torch.zeros((len(reqs), max(npages)), dtype=torch.int32, device=dev)
            start = 0
            for i, n in enumerate(npages):
                ptab[i, :n] = torch.arange(start, start + n, dtype=torch.int32, device=dev)
                start += n
            cu = torch.tensor([0] + list(itertools.accumulate(lens)), dtype=torch.int32, device=dev)
            sk = torch.tensor(seqk, dtype=torch.int32, device=dev)
            q = torch.randn((total, HEADS, HD), device=dev, generator=g).to(torch.bfloat16)
            r = (torch.randn((total, HEADS, 16), device=dev, generator=g) * 2).to(torch.bfloat16)

            def run(bias, out, **kw):
                return fa(q=q, k_cache=kc, v_cache=vc, page_table=ptab, cache_seqlens=sk, cu_seqlens_q=cu,
                          max_seqlen_q=max(lens), softmax_scale=HD ** -0.5, causal=True, window_size=ws,
                          softcap=0.0, return_softmax_lse=True, num_splits=1, pack_gqa=False, out=out,
                          rel_bias=bias, rel_bias_prep_cache={}, **kw)

            o0 = torch.empty((total, HEADS, HD), device=dev, dtype=torch.float32)
            o1 = torch.empty_like(o0)
            pre = torch.einsum("thd,de->the", r, proj)
            recorded.clear()
            interface._shear_bias_empty = recording_empty
            try:
                res0 = run(pre, o0)
            finally:
                interface._shear_bias_empty = orig_empty
            torch.cuda.synchronize()
            stock_sheared = recorded[-1].clone() if recorded else None
            mine = bias_layout.sheared_rel_bias(r, proj, cu, sk, local, 1023)  # torch.empty, no aliasing
            with bias_layout.use_presheared(mine):
                res1 = run(mine, o1)
            torch.cuda.synchronize()
            lse0, lse1 = res0[1], res1[1]
            ref = bias_layout.sheared_reference(r, proj, cu, sk, local, 1023)
            checks = {
                "vs_reference": bitwise(torch, mine[:total], ref[:total]),
                "vs_stock_shear": bitwise(torch, mine[:total], stock_sheared[:total]) if stock_sheared is not None
                else "no stock buffer recorded",
                "attn_out": bitwise(torch, o1, o0),
                "lse": bitwise(torch, lse1, lse0),
            }
            bad = [f"{k}:{c}" for k, c in checks.items() if c != "bit-equal"]

            def stock_path():
                run(torch.einsum("thd,de->the", r, proj), o0)

            def lever_path():
                bias = bias_layout.sheared_rel_bias(r, proj, cu, sk, local, 1023, empty=orig_empty)
                with bias_layout.use_presheared(bias):
                    run(bias, o1)

            ts = bench(torch, stock_path, max(5, a.iters // 2))
            tl_ = bench(torch, lever_path, max(5, a.iters // 2))
            desc = ",".join(f"{c}@{p}" for c, p in reqs)
            record("bias", f"{kind:<6} {label} [{desc}]", "bit-equal" if not bad else "; ".join(bad), ts, tl_)
            del pre, mine, ref, stock_sheared


# ===================================================================== CUDA graphs


def graph_replay(torch, fn):
    """(eager outputs, replayed outputs) of fn() captured in a CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    eager = fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return eager, out


def sec_graphs(torch, dev, ck, a):
    import types

    import torch.nn.functional as F

    from inkling_prefill import bias_layout, partial_bf16, qkvr_split, quant_once
    from sglang.kernels.ops.attention import flash_attention_v4_sm120 as sm120
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as ex

    g = torch.Generator(device=dev).manual_seed(4)

    def flat(xs):
        return [x for x in (xs if isinstance(xs, (tuple, list)) else (xs,)) if torch.is_tensor(x)]

    def check(name, fn):
        try:
            eager, out = graph_replay(torch, fn)
            res = [bitwise(torch, o, e) for o, e in zip(flat(out), flat(eager))]
            bad = [c for c in res if c != "bit-equal"]
            record("graphs", name, "bit-equal" if not bad else "; ".join(bad))
        except Exception as exc:  # noqa: BLE001
            record("graphs", name, f"ERROR {type(exc).__name__}: {str(exc)[:200]}")

    w = torch.cat([ck.get(f"model.llm.layers.5.attn.{p}.weight", dev) for p in PROJ_NAMES]).contiguous()

    class Method:
        pass

    Method.__name__ = "UnquantizedLinearMethod"
    me = types.SimpleNamespace(split_sizes=list(SPLITS), qkvr=types.SimpleNamespace(weight=w, bias=None, quant_method=Method()))
    buf = torch.randn((3, 1024, H), device=dev, generator=g).to(torch.bfloat16)
    src = types.SimpleNamespace(q=buf[0], k=buf[1], v=buf[2])
    fn = qkvr_split.make_project_sources(lambda s, x: None, None, {"min_m": 17, "out": "strided"})
    check("qkvr_split M=1024", lambda: fn(me, src))

    ids, wts = make_routing(torch, dev, 1024, "uniform", 4)
    src2dst, dst2src, _, offsets = sort_route(torch, ids, wts)
    down = torch.randn((dst2src.numel(), H), device=dev, generator=g).to(torch.bfloat16)
    check("partial_bf16 T=1024", lambda: partial_bf16.routed_partial_bf16(down, offsets, src2dst, TOPK))
    x = torch.randn((1024, H), device=dev, generator=g).to(torch.bfloat16)
    check("quant_once T=1024",
          lambda: quant_once.quant_once(ex.nvfp4_routed_quantize, ex._nvfp4_outputs, x, dst2src, TOPK, offsets)[::2])

    bias_layout.patch_sm120(sm120)
    proj = ck.get("model.llm.layers.0.attn.rel_logits_proj.proj", dev).to(torch.bfloat16).contiguous()
    t = 1024
    page = 128
    kc = (torch.randn((16, page, KVH, HD), device=dev, generator=g) * 0.5).to(torch.bfloat16)
    vc = torch.randn_like(kc)
    ptab = torch.arange(16, dtype=torch.int32, device=dev)[None]
    cu = torch.tensor([0, t], dtype=torch.int32, device=dev)
    sk = torch.tensor([t + 512], dtype=torch.int32, device=dev)
    q = torch.randn((t, HEADS, HD), device=dev, generator=g).to(torch.bfloat16)
    r = torch.randn((t, HEADS, 16), device=dev, generator=g).to(torch.bfloat16)
    o = torch.empty((t, HEADS, HD), device=dev, dtype=torch.float32)

    def attn():
        bias = bias_layout.sheared_rel_bias(r, proj, cu, sk, True, 1023)
        with bias_layout.use_presheared(bias):
            return sm120.flash_attn_with_kvcache(
                q=q, k_cache=kc, v_cache=vc, page_table=ptab, cache_seqlens=sk, cu_seqlens_q=cu, max_seqlen_q=t,
                softmax_scale=HD ** -0.5, causal=True, window_size=(1023, 0), softcap=0.0, return_softmax_lse=True,
                num_splits=1, pack_gqa=False, out=o, rel_bias=bias, rel_bias_prep_cache={})

    check("bias_layout + FA4 presheared T=1024 prefix 512", attn)
    del w, F


# ===================================================================== install


INSTALL_SNIPPET = """
import os, sys
sys.path.insert(0, {overlay!r})
import inkling_prefill as P
P.install()
import importlib
for name in ("sglang.kernels.ops.attention.flash_attention_v4_sm120",
             "sglang.kernels.ops.moe.inkling2_experts_sm120",
             "sglang.srt.models.inkling_common.attn",
             "sglang.srt.models.inkling_common.inkling2_attn",
             "sglang.srt.models.inkling_common.inkling2_moe"):
    importlib.import_module(name)
print("APPLIED", sorted(P.STATE["applied"]))
"""


def sec_install(torch, dev, ck, a):
    env = dict(os.environ, INKLING_PREFILL="all", INKLING_PREFILL_VERBOSE="1")
    p = subprocess.run([sys.executable, "-c", INSTALL_SNIPPET.format(overlay=OVERLAY)],
                       capture_output=True, text=True, env=env)
    line = next((ln for ln in p.stdout.splitlines() if ln.startswith("APPLIED")), "")
    want = ["bias_layout@sglang.kernels.ops.attention.flash_attention_v4_sm120",
            "bias_layout@sglang.srt.models.inkling_common.attn",
            "bias_layout@sglang.srt.models.inkling_common.inkling2_attn",
            "partial_bf16@sglang.srt.models.inkling_common.inkling2_moe",
            "qkvr_split@sglang.srt.models.inkling_common.inkling2_attn",
            "quant_once@sglang.kernels.ops.moe.inkling2_experts_sm120"]
    missing = [w for w in want if w not in line]
    log("  " + "\n  ".join(p.stderr.strip().splitlines()[-8:]))
    record("install", "INKLING_PREFILL=all on the real sglang",
           "bit-equal (all 6 patches applied)" if p.returncode == 0 and not missing
           else f"FAILED rc={p.returncode} missing={missing}")


SECTIONS = dict(qkvr=sec_qkvr, partial=sec_partial, quant=sec_quant, bias=sec_bias, graphs=sec_graphs,
                install=sec_install)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="/model")
    ap.add_argument("--sections", default=",".join(SECTIONS))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--json", default="")
    ap.add_argument("--tp", type=int, default=1, help="qkvr: check one attention-TP rank's shard shapes")
    ap.add_argument("--rank", type=int, default=0)
    a = ap.parse_args()
    import torch

    dev = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.set_device(dev)
    p = torch.cuda.get_device_properties(dev)
    log(f"device {p.name} sm_{p.major}{p.minor} SMs={p.multi_processor_count}  torch {torch.__version__}")
    ck = Ckpt(a.model)
    t0 = time.time()
    for s in [x for x in a.sections.split(",") if x]:
        log(f"== {time.strftime('%H:%M:%S')} {s}")
        try:
            SECTIONS[s](torch, dev, ck, a)
        except Exception as exc:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            record(s, "section", f"ERROR {type(exc).__name__}: {str(exc)[:300]}")
            try:  # a failed launch leaves cudaGetLastError set; do not blame the next section
                torch.cuda.synchronize()
                torch.cuda.cudart().cudaGetLastError()
            except Exception:  # noqa: BLE001
                pass
        torch.cuda.empty_cache()
    fails = [r for r in RESULTS if not r["ok"]]
    log(f"\n{len(RESULTS) - len(fails)} / {len(RESULTS)} checks bit-equal, {time.time() - t0:.0f} s")
    for r in fails:
        log(f"  FAIL {r['section']} {r['name']}: {r['check']}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(RESULTS, f, indent=1)
        log(f"wrote {a.json}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
