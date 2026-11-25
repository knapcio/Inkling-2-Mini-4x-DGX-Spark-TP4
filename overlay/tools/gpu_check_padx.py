#!/usr/bin/env python3
"""GPU check of the installed inkling_padx patch (inside the SGLang image, server stopped).

  cd overlay && INKLING_PADX=1 python3 tools/gpu_check_padx.py [--model /model] [--noexp]

Installs the overlay, imports the model module, stands in for the decode graph runner (a fake runner holding the
device live-row count, set as the capturing runner), then runs the real MoE chain (router, quant, grouped GEMMs,
SwiGLU, BF16 combine; stock functions) of NVFP4 layer 7 and MXFP8 layer 21 on the same rows:
  stock:  stock _sm120_route over every row (today: padding rows keep stale ids and route)
  padx:   the patched _sm120_route under CUDA-graph capture, replayed with the live count written by the patched
          DecodeCudaGraphRunner._set_dp_real_num_tokens
Live rows' combined output must be bit-identical; padding rows' must be 0. Shapes: c9 in the c10 graph (72 of 80
rows), c15 in c16, c33 in c40, a full bucket, one live request; the same graph replayed at two live counts; with
--noexp also dead verify rows (inkling_noexp under a fake capped vcap state) composed with padding.
Eager calls must be the stock path.
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
    ap.add_argument("--noexp", action="store_true")
    a = ap.parse_args()
    os.environ["INKLING_PADX"] = "1"
    if a.noexp:
        os.environ["INKLING_NOEXP"] = "1"
        os.environ.setdefault("INKLING_VERIFY_CAP", "policy:x")
        import inkling_noexp
        inkling_noexp.install()
    import inkling_padx
    inkling_padx.install()
    import torch
    import sglang.srt.models.inkling_common.inkling2_moe as M
    import sglang.srt.model_executor.runner.decode_cuda_graph_runner as R
    from inkling_padx import patches as P
    from inkling_prefill.partial_bf16 import routed_partial_bf16
    from sglang.kernels.ops.gemm import inkling2_grouped_sm120 as grouped
    from sglang.kernels.ops.moe import inkling2_experts_sm120 as exs

    patched = M.Inkling2Experts._sm120_route
    ok = getattr(patched, inkling_padx.MARK, False) and getattr(R.DecodeCudaGraphRunner._set_dp_real_num_tokens,
                                                                inkling_padx.MARK, False)
    log("patched:", ok)
    if not ok:
        FAILED.append("not patched")
    stock_route = patched.__wrapped__
    while hasattr(stock_route, "__wrapped__"):
        stock_route = stock_route.__wrapped__
    ck = Ckpt(a.model)
    e = nvfp4_experts(ck, 7)
    gate = router(ck, 7)
    host = types.SimpleNamespace(num_global_experts=128)
    fake = types.SimpleNamespace(model_runner=types.SimpleNamespace(device=torch.device("cuda")),
                                 buffers=types.SimpleNamespace(num_token_non_padded=None),
                                 capture_bs=list(range(1, 9)) + list(range(10, 33, 2)) + [40, 44, 48],
                                 captured_req_width=8)  # the speculative decode bucket list of the served stack
    import inspect
    src_lb = inspect.getsource(R.DecodeCudaGraphRunner.load_batch)
    if src_lb.count("self._set_dp_real_num_tokens(") < 2:  # both replay paths must refresh the live count
        FAILED.append("load_batch does not call _set_dp_real_num_tokens on both paths")
    st = None
    if a.noexp:
        st = types.SimpleNamespace(enabled=True, in_verify_moe=True, capping=True,
                                   config=types.SimpleNamespace(remap=True), stride=8, src_index=None)
        vmod = types.ModuleType("inkling_vcap.state")
        vmod.STATE = st
        sys.modules["inkling_vcap.state"] = vmod

    def chain(x, route_fn, weights, ids, ex=None):
        src2dst, dst2src, gammas, offsets = route_fn(host, ids, weights)
        T = x.shape[0]
        if ex is not None:
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

    def captured(x, weights, ids, ex=None, pending_src=None):
        """chain() captured in a CUDA graph through the PRODUCTION DecodeCudaGraphRunner.capture wrapper (it sets the
        capture scope and resets the live count to BIG); the stock capture body is replaced by this test's via the
        fake runner's model_context / _capture. pending_src: emulate inkling_noexp's fused remap (its remap_mask
        kernel writes the masked twin into STATE["masked"], consumed by identity in the route)."""
        box = {}

        def route_fn(h, ids_, w_):
            if pending_src is not None:
                from inkling_noexp._mask import remap_mask
                import inkling_noexp
                w2, i2, masked = remap_mask(w_, ids_, pending_src, 128)
                inkling_noexp.STATE["masked"] = (i2, masked)
                return patched(h, i2, w2)
            return patched(h, ids_, w_)

        def body():
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                chain(x, route_fn, weights, ids, ex)
            torch.cuda.current_stream().wait_stream(s_)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                box["out"] = chain(x, route_fn, weights, ids, ex)
            box["gr"] = gr

        import contextlib
        fake.model_context = contextlib.nullcontext
        fake._capture = body
        R.DecodeCudaGraphRunner.capture(fake)
        if inkling_padx.STATE["runner"] is not None:
            FAILED.append("capture scope not restored")
        if not hasattr(fake, "_inkling_padx_real"):
            FAILED.append("capture wrapper did not create the live-count buffer")
        return box["gr"], box["out"]

    def set_real(n):  # what the patched runner does on every replay
        R.DecodeCudaGraphRunner._set_dp_real_num_tokens(fake, n)

    g = torch.Generator(device="cuda").manual_seed(31)
    # (name, graph rows, live rows, expect the padding mask) - the expectation is written out, independent of can_pad:
    # with the served bucket list, graphs of 10 / 16 / 40 requests can be replayed with padding, graphs of 2 / 4 cannot
    cases = [("c9in10", 80, 72, True), ("c15in16", 128, 120, True), ("c33in40", 320, 264, True),
             ("full16", 128, 128, True), ("one", 80, 8, True), ("c1in2", 16, 8, False), ("c3in4", 32, 24, False)]
    with torch.no_grad():
        for layer_name, gate_l, ex in (("NVFP4 L7", gate, None), ("MXFP8 L21", None, "mx")):
            if ex == "mx":
                from sglang.kernels.ops.quantization.inkling2_mxfp8_scales_sm120 import mxfp8_weight_scales
                p = "layers.21.mlp.experts."
                ex = types.SimpleNamespace(w13=ck(p + "w13_weight"), w2=ck(p + "w2_weight"))
                ex.w13_cute = mxfp8_weight_scales(ck(p + "w13_weight_scale"))
                ex.w2_cute = mxfp8_weight_scales(ck(p + "w2_weight_scale"))
                gate_l = router(ck, 21)
            for name, T, live_n, expect_mask in cases:
                B = T // 8
                req = torch.randn(B, 1, 2304, device="cuda", generator=g)
                x = (req + 0.5 * torch.randn(B, 8, 2304, device="cuda", generator=g)).reshape(T, 2304).to(torch.bfloat16)
                weights, ids = gate_l(x)
                lens = None
                if st is not None:  # dead verify rows too: widths 1..8 per request (anchor remap, as vcap)
                    lens = [1 + (b * 5) % 8 for b in range(B)]
                    src = torch.tensor([(r // 8) * 8 if r % 8 >= lens[r // 8] else r for r in range(T)], device="cuda")
                    st.src_index = src
                    weights, ids = weights[src], ids[src]
                    dead_live = src == torch.arange(T, device="cuda")
                else:
                    dead_live = torch.ones(T, dtype=torch.bool, device="cuda")
                eager = chain(x, patched, weights, ids, ex)
                ref = chain(x, stock_route, weights, ids, ex)
                use_pending = st is not None and (len(name) % 2 == 1)
                if use_pending:  # fused noexp branch: the gate output stays un-remapped; remap_mask does it in-graph
                    weights_in, ids_in = gate_l(x)
                else:
                    weights_in, ids_in = weights, ids
                gr, out = captured(x, weights_in, ids_in, ex, pending_src=st.src_index if use_pending else None)
                set_real(live_n)
                gr.replay()
                torch.cuda.synchronize()
                gated = expect_mask  # graphs that never pad keep the stock route: every row as stock
                if P.can_pad(fake, T) != expect_mask:
                    FAILED.append(f"can_pad({T} rows) != expected {expect_mask}")
                pad_live = torch.arange(T, device="cuda") < (live_n if gated else T)
                live = pad_live & dead_live
                if not torch.equal(eager.view(torch.int16), ref.view(torch.int16)) and st is None:
                    FAILED.append(f"{layer_name} {name} eager passthrough")
                same = torch.equal(ref[live].view(torch.int16), out[live].view(torch.int16))
                zero = bool((out[~live] == 0).all()) if (~live).any() else True
                # second replay of the same graph at another live count
                set_real(max(8, live_n - 16))
                gr.replay()
                torch.cuda.synchronize()
                live2 = (torch.arange(T, device="cuda") < (max(8, live_n - 16) if gated else T)) & dead_live
                same2 = torch.equal(ref[live2].view(torch.int16), out[live2].view(torch.int16))
                zero2 = bool((out[~live2] == 0).all()) if (~live2).any() else True
                log(f"  {layer_name} {name:8s}{' fused-noexp' if use_pending else ''}{'' if gated else ' (no-pad graph: stock route)'} live {int(live.sum()):4d}/{T}: live rows {'EXACT' if same else 'DIFFER'}, "
                    f"masked rows {'0' if zero else 'NONZERO'}; replay at {max(8, live_n - 16) // 8} req: "
                    f"{'EXACT' if same2 and zero2 else 'DIFFER'}")
                if not (same and zero and same2 and zero2):
                    FAILED.append(f"{layer_name} {name}")
                del gr
    log("state:", {k: v for k, v in inkling_padx.STATE.items() if k != "runner"})
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
