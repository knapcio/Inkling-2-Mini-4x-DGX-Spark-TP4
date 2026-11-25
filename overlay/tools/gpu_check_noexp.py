#!/usr/bin/env python3
"""GPU check of the installed inkling_noexp patch (inside the SGLang image, server stopped).

  cd overlay && INKLING_NOEXP=1 INKLING_VERIFY_CAP=policy:x python3 tools/gpu_check_noexp.py [--model /model]

(The v2.7.2 in-route mask on real layers. tools/gpu_check_noexp_mask.py checks v2.7.3's fused mask against it.)

Installs the overlay, imports the model module (patched?), stands in for inkling_vcap's state (enabled, capping,
in a verify MoE call, src_index of the given live widths with anchor remap), then runs the real NVFP4 MoE chain of
layer 7 (router, quant, w13, SwiGLU, w2, BF16 combine; stock functions) twice on the same rows:
  shipped: dead rows remapped to the anchor's ids (vcap), stock _sm120_route
  noexp:   the patched _sm120_route
Live rows' combined output must be bit-identical; dead rows' must be 0. Widths: c8 [4,5,5,4,5,4,5,5], c16 (10x3,
6x4), c4 [8,1,3,8], all-live and all-dead-but-anchor; plus a CUDA-graph replay with a changed src_index.
"""
import argparse
import glob
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(HERE)
sys.path.insert(0, OVERLAY)
FAILED = []


class Ckpt:
    """Tensors of the Inkling-2-Mini checkpoint by name (model*.safetensors under --model)."""

    def __init__(self, model):
        from safetensors import safe_open

        self.h = {}
        for f in sorted(glob.glob(os.path.join(model, "model*.safetensors"))):
            h = safe_open(f, "pt", device="cpu")
            for k in h.keys():
                self.h[k] = h

    def __call__(self, name):
        k = "model.llm." + name
        return self.h[k].get_tensor(k).cuda()


def nvfp4_experts(ck, layer):
    import torch
    from sglang.srt.layers.quantization.utils import swizzle_blockscale

    p = f"layers.{layer}.mlp.experts."
    e = types.SimpleNamespace()
    e.w13, e.w2 = ck(p + "w13_weight"), ck(p + "w2_weight")
    assert e.w13.dtype == torch.uint8, f"layer {layer} experts are not NVFP4"
    e.w13_scale = swizzle_blockscale(ck(p + "w13_weight_scale")).flatten(1)
    e.w2_scale = swizzle_blockscale(ck(p + "w2_weight_scale")).flatten(1)
    e.w13_amax, e.w2_amax = ck(p + "w13_weight_amax").float(), ck(p + "w2_weight_amax").float()
    e.w13_den = ck(p + "w13_weight_outer_scale_denominator").float()
    e.w2_den = ck(p + "w2_weight_outer_scale_denominator").float()
    return e


def router(ck, layer):
    from sglang.srt.models.inkling_common.inkling2_moe import Inkling2Gate

    cfg = types.SimpleNamespace(num_experts_per_tok=8, route_scale=8.0, n_routed_experts=128, hidden_size=2304,
                                use_gate_bias=True)
    g = Inkling2Gate(cfg).cuda()
    g.weight.data = ck(f"layers.{layer}.mlp.gate.weight")
    g.bias.data = ck(f"layers.{layer}.mlp.gate.bias").float()
    return g


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    a = ap.parse_args()
    os.environ.setdefault("INKLING_NOEXP", "1")
    os.environ.setdefault("INKLING_VERIFY_CAP", "policy:x")
    import inkling_noexp

    inkling_noexp.install()
    import torch
    import sglang.srt.models.inkling_common.inkling2_moe as M
    from inkling_prefill.partial_bf16 import routed_partial_bf16
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as exs

    patched = M.Inkling2Experts._sm120_route
    ok = getattr(patched, inkling_noexp.MARK, False)
    log("patched:", ok)
    if not ok:
        FAILED.append("not patched")
    stock_route = patched.__wrapped__
    ck = Ckpt(a.model)
    e = nvfp4_experts(ck, 7)
    gate = router(ck, 7)
    host = types.SimpleNamespace(num_global_experts=128)
    st = types.SimpleNamespace(enabled=True, in_verify_moe=True, capping=True, config=types.SimpleNamespace(remap=True),
                               stride=8, src_index=None)
    vmod = types.ModuleType("inkling_vcap.state")
    vmod.STATE = st
    sys.modules["inkling_vcap.state"] = vmod

    def forced(h, ids, weights):  # the patched path's routing, eager (the patch itself acts only under capture)
        live = inkling_noexp.live_rows(ids.shape[0])
        return inkling_noexp.route_dead_free(M.route_metadata, ids, weights, live, 128)

    def chain(x, route_fn, weights, ids, ex=None):
        src2dst, dst2src, gammas, offsets = route_fn(host, ids, weights)
        T = x.shape[0]
        if ex is not None:  # MXFP8 layer (20-23): the stock _sm120_mxfp8 sequence
            v, sc = exs.mxfp8_routed_quantize(x=x, dst2src=dst2src, top_k=8, offsets=offsets)
            gu = grouped.mxfp8_expert_matmul(values=v, scales=sc, weight=ex.w13, weight_scales=ex.w13_cute,
                                             offsets=offsets)
            v, sc = exs.mxfp8_routed_swiglu_quantize(acc=gu, gammas=gammas, offsets=offsets, gate_cap=4.0, up_cap=25.0)
            down = grouped.mxfp8_expert_matmul(values=v, scales=sc, weight=ex.w2, weight_scales=ex.w2_cute,
                                               offsets=offsets)
            return routed_partial_bf16(down, offsets, src2dst, 8)
        v, sc, amax = exs.nvfp4_routed_quantize(x=x, dst2src=dst2src, top_k=8, offsets=offsets)
        gu = grouped.nvfp4_expert_matmul(values=v, scales=sc, weight=e.w13, weight_scales=e.w13_scale, offsets=offsets,
                                         tokens=T)
        v, sc, amax2 = exs.nvfp4_routed_swiglu_quantize(gu, amax, e.w13_amax, e.w13_den, gammas, offsets, 4.0, 25.0)
        down = grouped.nvfp4_expert_matmul(values=v, scales=sc, weight=e.w2, weight_scales=e.w2_scale, offsets=offsets,
                                           tokens=T)
        return routed_partial_bf16(down, offsets, src2dst, 8, nvfp4_scales=(amax2, e.w2_amax, e.w2_den))

    g = torch.Generator(device="cuda").manual_seed(21)
    cases = {"c8": [4, 5, 5, 4, 5, 4, 5, 5], "c16": [3] * 10 + [4] * 6, "c4": [8, 1, 3, 8], "full": [8] * 8,
             "anchors": [1] * 8}
    with torch.no_grad():
        for name, lens in cases.items():
            B = len(lens)
            T = 8 * B
            src = torch.tensor([(r // 8) * 8 if r % 8 >= lens[r // 8] else r for r in range(T)], device="cuda")
            st.src_index = src
            req = torch.randn(B, 1, 2304, device="cuda", generator=g)
            x = (req + 0.5 * torch.randn(B, 8, 2304, device="cuda", generator=g)).reshape(T, 2304).to(torch.bfloat16)
            weights, ids = gate(x)
            live = src == torch.arange(T, device="cuda")
            ref = chain(x, stock_route, weights[src], ids[src])        # shipped: vcap anchor remap
            out = chain(x, forced, weights[src], ids[src])              # noexp on the remapped gate output
            eager = chain(x, patched, weights[src], ids[src])           # patched, eager: must be stock
            torch.cuda.synchronize()
            if not torch.equal(eager.view(torch.int16), ref.view(torch.int16)):
                FAILED.append(f"{name} eager passthrough")
                log(f"  {name}: eager patched call is not the stock path")
            same = torch.equal(ref[live].view(torch.int16), out[live].view(torch.int16))
            zero = bool((out[~live] == 0).all()) if (~live).any() else True
            log(f"  {name:8s} live {int(live.sum()):4d}/{T}: live rows {'EXACT' if same else 'DIFFER'}, dead rows "
                f"{'0' if zero else 'NONZERO'}")
            if not (same and zero):
                FAILED.append(name)
        # MXFP8 layer 21 (CuTe grouped GEMM path), c8 and c16 widths
        from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales

        p = "layers.21.mlp.experts."
        mx = types.SimpleNamespace(w13=ck(p + "w13_weight"), w2=ck(p + "w2_weight"))
        mx.w13_cute = mxfp8_weight_scales(ck(p + "w13_weight_scale"))
        mx.w2_cute = mxfp8_weight_scales(ck(p + "w2_weight_scale"))
        gate21 = router(ck, 21)
        for name in ("c8", "c16"):
            lens = cases[name]
            B, T = len(lens), 8 * len(lens)
            src = torch.tensor([(r // 8) * 8 if r % 8 >= lens[r // 8] else r for r in range(T)], device="cuda")
            st.src_index = src
            x = torch.randn(T, 2304, device="cuda", generator=g).to(torch.bfloat16)
            weights, ids = gate21(x)
            live = src == torch.arange(T, device="cuda")
            ref = chain(x, stock_route, weights[src], ids[src], ex=mx)
            out = chain(x, forced, weights[src], ids[src], ex=mx)
            torch.cuda.synchronize()
            same = torch.equal(ref[live].view(torch.int16), out[live].view(torch.int16))
            zero = bool((out[~live] == 0).all())
            log(f"  MXFP8 L21 {name}: live rows {'EXACT' if same else 'DIFFER'}, dead rows {'0' if zero else 'NONZERO'}")
            if not (same and zero):
                FAILED.append(f"mxfp8 {name}")
        # graph replay: the live mask is read from src_index at replay (patched path, under real capture)
        lens_a, lens_b = [4, 5, 5, 4, 5, 4, 5, 5], [8, 2, 6, 1, 7, 3, 8, 5]
        T = 64
        mk = lambda lens: torch.tensor([(r // 8) * 8 if r % 8 >= lens[r // 8] else r for r in range(T)],  # noqa: E731
                                       device="cuda")
        st.src_index = mk(lens_a)
        x = torch.randn(T, 2304, device="cuda", generator=g).to(torch.bfloat16)
        weights, ids = gate(x)

        def run():
            s = st.src_index
            return chain(x, patched, weights[s], ids[s])

        s_ = torch.cuda.Stream()
        s_.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s_):
            run()
        torch.cuda.current_stream().wait_stream(s_)
        gr = torch.cuda.CUDAGraph()
        with torch.cuda.graph(gr):
            out = run()
        st.src_index.copy_(mk(lens_b))
        gr.replay()
        torch.cuda.synchronize()
        src = st.src_index
        live = src == torch.arange(T, device="cuda")
        ref = chain(x, stock_route, weights[src], ids[src])
        same = torch.equal(ref[live].view(torch.int16), out[live].view(torch.int16)) and bool((out[~live] == 0).all())
        log(f"  graph replay with new widths: {'EXACT' if same else 'DIFFER'}")
        if not same:
            FAILED.append("graph")
    log("state:", inkling_noexp.STATE)
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
