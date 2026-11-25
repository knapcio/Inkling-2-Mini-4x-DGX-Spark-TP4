#!/usr/bin/env python3
"""Integration check of inkling_wide with the REAL sglang modules of the image (patch targets, hook order with
inkling_pico12, dispatch), on stand-in layer objects that carry real UnquantizedLinearMethod instances and real
layer-0 / unembed weights. GPU, server stopped, ~3 GB.

  PYTHONPATH=<overlay> python3 inkling_wide/tools/gpu_integration_wide.py [--model /model] [--no-pico]

Gated: install() of inkling_pico12 then inkling_wide applies every patch to the real modules; after both packages'
prepare_target, qkvr / wo_ud / the head are served by pico at <= 16 rows and by wide at 17..256 (on pico's coded
buffers), by stock above; results match cuBLAS within the BF16 / FP32 summation tolerance; the real patched
Inkling2Attention._project_sources returns the stock view layout for three sources at M = 24 and 64; captured in a
CUDA graph, the stacked call replays bit for bit. Last line `wide integration result PASS|FAIL`.
"""

import argparse
import os
import sys
import types

OVERLAY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if OVERLAY not in sys.path:
    sys.path.insert(0, OVERLAY)

FAILS = []


def log(*a):
    print(*a, flush=True)


def gate(name, ok, detail=""):
    log(f"{name} {detail} {'PASS' if ok else 'FAIL'}")
    if not ok:
        FAILS.append(name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/model")
    p.add_argument("--no-pico", action="store_true")
    args = p.parse_args()
    os.environ["INKLING_WIDE"] = "dense,lmhead"
    if not args.no_pico:
        os.environ["INKLING_PICO12"] = "dense,lmhead"

    import inkling_pico12
    import inkling_wide

    if not args.no_pico:
        gate("wide integ pico install", inkling_pico12.install() == ("dense", "lmhead"))
    gate("wide integ install", inkling_wide.install() == ("dense", "lmhead"))

    import torch
    import torch.nn.functional as F
    from torch import nn

    from sglang.srt.layers.quantization import unquant as uq
    from sglang.srt.models.inkling_common import inkling2_attn as am
    import sglang.srt.models.inkling2 as mm

    from inkling_pico12 import patches as ppatches
    from inkling_wide import ops, patches

    gate("wide integ patched lm head", getattr(uq.sm12x_fp32_lm_head, "_wide", False)
         and getattr(uq.sm12x_fp32_lm_head_applies, "_wide", False))
    gate("wide integ patched _project_sources", getattr(am.Inkling2Attention._project_sources, "inkling_wide", False)
         and (args.no_pico or getattr(am.Inkling2Attention._project_sources, "inkling_pico12", False)))
    gate("wide integ patched post_load_weights + forward (scope)",
         getattr(mm.Inkling2ForConditionalGeneration.post_load_weights, "_wide", False)
         and getattr(mm.Inkling2ForConditionalGeneration.forward, "_wide", False))
    # the scope marker on a real ForwardMode
    from sglang.srt.model_executor.forward_mode import ForwardMode

    sc = [patches.is_in_scope(types.SimpleNamespace(forward_mode=fm)) for fm in
          (ForwardMode.DECODE, ForwardMode.TARGET_VERIFY, ForwardMode.EXTEND)]
    gate("wide integ scope modes", sc == [True, True, False], f"decode/verify/extend={sc}")
    ops.SET.scope = "all"  # the direct layer calls below run outside a model forward
    log(f"wide integ patched: {patches.STATE['patched']}")

    dev = torch.device("cuda", torch.cuda.current_device())
    from safetensors import safe_open
    import glob

    where = {}
    for path in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
        with safe_open(path, framework="pt", device="cpu") as h:
            for key in h.keys():
                where.setdefault(key, path)

    def get(name):
        with safe_open(where[name], framework="pt", device="cpu") as h:
            return h.get_tensor(name).to(dev)

    base = "model.llm.layers.0.attn."
    parts = [get(base + n + ".weight") for n in ("wq_du", "wk_dv", "wv_dv", "wr_du", "gate_attn", "attn_sink_proj")]
    splits = [t.shape[0] for t in parts]
    w_qkvr = torch.cat(parts)
    del parts
    w_wo = get(base + "wo_ud.weight")
    w_head = get("model.llm.unembed.weight")

    class Lin(nn.Module):
        def __init__(self, w):
            super().__init__()
            self.weight = nn.Parameter(w, requires_grad=False)
            self.bias = None
            self.quant_method = uq.UnquantizedLinearMethod()

        def forward(self, x):
            return self.quant_method.apply(self, x, None), None

    attn = am.Inkling2Attention.__new__(am.Inkling2Attention)
    nn.Module.__init__(attn)
    attn.qkvr, attn.wo_ud, attn.split_sizes = Lin(w_qkvr), Lin(w_wo), list(splits)
    model = nn.Module()
    model.llm = nn.Module()
    model.llm.layers = nn.ModuleList([nn.Module()])
    model.llm.layers[0].attn = attn
    model.llm.lm_head = nn.Module()
    model.llm.lm_head.weight = nn.Parameter(w_head, requires_grad=False)
    if not args.no_pico:
        ppatches.prepare_target(model)
    patches.prepare_target(model)  # the order the stacked post_load_weights wrappers give
    log(f"wide integ qkvr method {type(attn.qkvr.quant_method).__name__}, fmt "
        f"{'bf16x' if attn.qkvr.__dict__['_wide'].fmt == ops.BF16X else 'raw'}")
    gate("wide integ formats", all(e.fmt == (ops.RAW if args.no_pico else ops.BF16X) for e in
                                   (attn.qkvr.__dict__["_wide"], attn.wo_ud.__dict__["_wide"],
                                    next(iter(patches.HEADS.values())))))

    calls = {"wide": 0}
    real_ext = ops.EXT

    class Counting:
        def __getattr__(self, name):
            return getattr(real_ext, name)

        def gemm(self, *a):
            calls["wide"] += 1
            return real_ext.gemm(*a)

    ops.EXT = Counting()

    def close(y, ref, bf):
        tol = (2.0**-7 if bf else 2.0**-12) * float(ref.abs().max()) + 1e-6
        d = float((y.float() - ref.float()).abs().max())
        if d > tol:
            log(f"wide integ mismatch max_abs={d:.3e} tol={tol:.3e} ref_max={float(ref.abs().max()):.3e}")
        return d <= tol

    g = torch.Generator(device=dev).manual_seed(1)
    ops.SET.max_m_kind = {}  # every kind to 256 here (the dispatch, not the per-kind caps, is under test)
    for m, who in ((8, "pico"), (16, "pico"), (17, "wide"), (64, "wide"), (256, "wide"), (300, "stock")):
        if args.no_pico and who == "pico":
            who = "stock"
        x = (torch.rand((m, 2304), device=dev, generator=g) * 2 - 1).to(torch.bfloat16)
        xo = (torch.rand((m, 3072), device=dev, generator=g) * 2 - 1).to(torch.bfloat16)
        calls["wide"] = 0
        y, _ = attn.qkvr(x)
        yo, _ = attn.wo_ud(xo)
        yh = uq.sm12x_fp32_lm_head(x, model.llm.lm_head.weight, None) if uq.sm12x_fp32_lm_head_applies(
            x, model.llm.lm_head.weight, None) else torch.mm(x, w_head.T, out_dtype=torch.float32)
        torch.cuda.synchronize()
        used = calls["wide"]
        # FP32 references (cuBLAS's own BF16 wo at 40..64 rows errs 2x more than wide vs FP64: not a reference)
        ok = (close(y, x.float() @ w_qkvr.float().T, True) and close(yo, xo.float() @ w_wo.float().T, True)
              and close(yh, x.float() @ w_head.float().T, False))
        gate(f"wide integ dispatch M={m}", ok and (used == 3) == (who == "wide"), f"expected={who} wide_calls={used}")

    for m in (24, 64):
        q, k, v = ((torch.rand((m, 2304), device=dev, generator=g) * 2 - 1).to(torch.bfloat16) for _ in range(3))
        src = am.AttentionSources(q, k, v)
        calls["wide"] = 0
        got = am.Inkling2Attention._project_sources(attn, src)
        used = calls["wide"]
        st = F.linear(torch.cat([q, k, v]), w_qkvr)
        q_p, k_p, v_p = (c.split(splits, dim=-1)[:6] for c in st.chunk(3))
        exp = (q_p[0], k_p[1], v_p[2], q_p[3], q_p[4], q_p[5])
        layout = all((a.shape, a.stride(), a.storage_offset()) == (b.shape, b.stride(), b.storage_offset())
                     for a, b in zip(got, exp))
        vals = all(close(a, b, True) for a, b in zip(got, exp))
        gate(f"wide integ stacked M={m}", layout and vals and used == 1, f"layout={'stock' if layout else 'OTHER'} "
             f"close={vals} wide_calls={used}")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            am.Inkling2Attention._project_sources(attn, src)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gout = am.Inkling2Attention._project_sources(attn, src)
        graph.replay()
        torch.cuda.synchronize()
        same = all(torch.equal(a.contiguous().view(torch.int16), b.contiguous().view(torch.int16))
                   for a, b in zip(gout, got))
        gate(f"wide integ stacked graph M={m}", same)
    log(f"wide integration result {'PASS' if not FAILS else 'FAIL'} ({', '.join(FAILS)})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
