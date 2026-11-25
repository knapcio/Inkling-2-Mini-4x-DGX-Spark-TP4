#!/usr/bin/env python3
"""Microbench of inkling_dfbytes' draft-side byte cuts for the DFlash2 drafter (inside the image, no server).

  cd overlay && TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas \\
    python3 tools/bench_dfbytes.py --model /model --draft <DFlash2 drafter dir> --hidden <dir of captured *.npz> --hot FILE

1. Layers: the drafter's 5 x 4 linears (qkv 6144x2304, o 2304x4096, gate_up 13824x2304, down 2304x6912; real
   weights) at 8 / 56 / 192 rows in one CUDA graph, cold L2: stock F.linear (BF16) vs the mx8 path (MXFP8 copy,
   inkling_draftbytes' W8 method: CUDA GEMV <= 16 rows, Triton above). Relative output error of mx8 vs BF16.
   (The 10 grouped-conv kernel projections, 576 x 2304 each, are also packed in serving but not timed here.)
2. Candidate projection: hidden [N, 2304] -> top-16 ids/values over the target head (200,058 valid rows):
   stock (BF16 head GEMM, radix top-k) vs hot8 (FP8 row copy of the first N hot ids) vs hot16 (BF16 hot rows),
   at N = 7 / 56 / 168 rows, CUDA graph, cold L2. With --hidden (captured target post-norm rows, a proxy for the
   drafter's hidden): recall of the stock top-16 / top-1 inside the hot top-16, over every captured row.
"""
import argparse
import glob
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FAILED = []


def log(*a):
    print(*a, flush=True)


def tensors(path, names):
    from safetensors import safe_open

    out = {}
    files = sorted(glob.glob(os.path.join(path, "*.safetensors")))
    for f in files:
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k in names:
                    out[k] = s.get_tensor(k)
    missing = [n for n in names if n not in out]
    if missing:
        raise SystemExit(f"missing tensors in {path}: {missing[:4]}")
    return out


def graph(fn):
    s_ = torch.cuda.Stream()
    s_.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s_):
        fn()
    torch.cuda.current_stream().wait_stream(s_)
    gr = torch.cuda.CUDAGraph()
    with torch.cuda.graph(gr):
        out = fn()
    return gr, out


def cold(gr, flush, fo, reps=25):
    ts = []
    for _ in range(reps):
        torch.sum(flush, 0, out=fo)
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
        ts.append(e0.elapsed_time(e1))
    return statistics.median(ts[3:])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--draft", required=True)
    ap.add_argument("--hidden", default=None)
    ap.add_argument("--hot", required=True, help="ranked hot-vocabulary JSON")
    ap.add_argument("--hot-n", default="32768,65536")
    a = ap.parse_args()
    import numpy as np
    from inkling_draftbytes import _opts, ops, patches as dbp, quant, w8gemv
    from sglang.kernels.ops.sampling.topk import radix_topk
    from inkling_dfbytes.patches import hot_logits
    from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod

    opts = _opts.parse("draftmx8,wide8,w8tile,w8gemv", environ={})
    dbp.STATE.opts = opts
    ops.configure(opts)
    w8gemv.load()
    dev = torch.device("cuda")
    flush = torch.ones(64 << 20, device=dev)
    fo = torch.empty((), device=dev)
    log("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"))

    # ---- 1. layers
    names = []
    for i in range(5):
        p = f"layers.{i}."
        names += [p + "self_attn.q_proj.weight", p + "self_attn.k_proj.weight", p + "self_attn.v_proj.weight",
                  p + "self_attn.o_proj.weight", p + "mlp.gate_proj.weight", p + "mlp.up_proj.weight",
                  p + "mlp.down_proj.weight"]
    t = tensors(a.draft, set(names))
    W8 = dbp._w8_method_class(UnquantizedLinearMethod)
    method = W8.__new__(W8)
    layers = []
    for i in range(5):
        p = f"layers.{i}."
        mats = [torch.cat([t[p + "self_attn.q_proj.weight"], t[p + "self_attn.k_proj.weight"],
                           t[p + "self_attn.v_proj.weight"]]), t[p + "self_attn.o_proj.weight"],
                torch.cat([t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]]), t[p + "mlp.down_proj.weight"]]
        for w in mats:
            w = w.to(dev).contiguous()
            lay = torch.nn.Module()
            lay.weight = torch.nn.Parameter(w, requires_grad=False)
            lay._idb_w8 = quant.quantize_mode(w, opts.draftmx8)
            pack = lay._idb_w8
            lay._idb_w8g = w8gemv.choose_cfg(pack.n, pack.k, 0) if w8gemv.eligible_pack(pack) else 0
            layers.append(lay)
    del t
    bf16_mb = sum(l.weight.numel() * 2 for l in layers) / 1e6
    mx8_mb = sum(l._idb_w8.nbytes() for l in layers) / 1e6
    log(f"layers: {len(layers)} linears, BF16 {bf16_mb:.0f} MB, MXFP8 {mx8_mb:.0f} MB")
    g = torch.Generator(device=dev).manual_seed(1)
    with torch.no_grad():
        for rows in (8, 56, 192):
            xs = {l.weight.shape[1]: (torch.randn((rows, l.weight.shape[1]), device=dev, generator=g) * 0.5
                                      ).to(torch.bfloat16) for l in layers}
            # stock: the serving BF16 linear (UnquantizedLinearMethod.apply, incl. its small-row dispatch)
            gb, ob = graph(lambda: [UnquantizedLinearMethod.apply(method, l, xs[l.weight.shape[1]]) for l in layers])
            gm, om = graph(lambda: [method.apply(l, xs[l.weight.shape[1]]) for l in layers])
            gb.replay(); gm.replay(); torch.cuda.synchronize()
            err = max(float(((m.float() - b.float()).norm() / b.float().norm())) for m, b in zip(om, ob))
            tb, tm = cold(gb, flush, fo), cold(gm, flush, fo)
            served = "w8" if rows <= opts.w8_max_m else "stock (rows > w8_max_m)"
            log(f"  rows {rows:3}: 20 linears BF16 {tb:6.3f} ms, mx8 {tm:6.3f} ms ({served}); max rel err {err:.2e}")
            del gb, gm, ob, om

    # ---- 2. candidate projection
    head_key = "model.llm.unembed.weight"
    head = tensors(a.model, {head_key})[head_key].to(dev)
    valid = 200058
    wv = head[:valid]
    from sglang.srt.models.dflash import _project_candidate_logits as project

    fake_head = torch.nn.Module()
    fake_head.weight = torch.nn.Parameter(head, requires_grad=False)
    hidden_rows = None
    if a.hidden:
        caps = []
        for path in sorted(glob.glob(os.path.join(a.hidden, "*.npz"))):
            with np.load(path) as c:
                r = c["hidden"]
                caps.append(torch.from_numpy(r.astype(np.int16)).view(torch.bfloat16) if r.dtype.kind in "iu"
                            else torch.from_numpy(r).to(torch.bfloat16))
        hidden_rows = torch.cat(caps).to(dev)
        log(f"captured hidden rows: {hidden_rows.shape[0]}")
    for hot_n in (int(x) for x in a.hot_n.split(",")):
        ids = dbp.load_hot_ids(a.hot, hot_n, valid)
        ids64 = torch.tensor(ids, dtype=torch.int64, device=dev)
        rows_w = head.index_select(0, ids64)
        p8 = quant.quantize_mode(rows_w, _opts.parse_mode("row", "row"))
        p16 = quant.bf16_pack(rows_w.contiguous())
        log(f"hot {hot_n}: FP8 {p8.nbytes() / 1e6:.0f} MB, BF16 {p16.nbytes() / 1e6:.0f} MB "
            f"(full BF16 head {wv.numel() * 2 / 1e6:.0f} MB)")

        def stock(h):  # the stock DFLASH projection (sglang.srt.models.dflash._project_candidate_logits)
            vals, idx = radix_topk(project(h, fake_head, num_org=valid, use_quant_head=False), 16)
            return idx.long(), vals

        def hot(h, pack):
            vals, idx = radix_topk(hot_logits(h, pack), 16)
            return ids64.index_select(0, idx.reshape(-1).long()).view(idx.shape), vals

        with torch.no_grad():
            if hidden_rows is not None:
                for label, pack in (("hot8", p8), ("hot16", p16)):
                    rec = top1 = vdiff = 0.0
                    n = 0
                    for c in range(0, hidden_rows.shape[0], 512):
                        h = hidden_rows[c:c + 512]
                        si, sv = stock(h)
                        hi, hv = hot(h, pack)
                        inter = (si[:, :, None] == hi[:, None, :])
                        rec += float(inter.any(2).float().mean(1).sum())
                        best = si.gather(1, sv.float().argmax(1, keepdim=True))
                        top1 += float((best == hi).any(1).float().sum())
                        n += h.shape[0]
                    log(f"  {label} (hot {hot_n}): stock top-16 recall {rec / n:.4f}, stock top-1 in hot top-16 "
                        f"{top1 / n:.4f} over {n} captured rows")
            for rows in (7, 56, 168):
                h = (torch.randn((rows, 2304), device=dev, generator=g) * 4).to(torch.bfloat16)
                gs, _ = graph(lambda: stock(h))
                g8, _ = graph(lambda: hot(h, p8))
                g16, _ = graph(lambda: hot(h, p16))
                ts, t8, t16 = cold(gs, flush, fo), cold(g8, flush, fo), cold(g16, flush, fo)
                log(f"  rows {rows:3}: stock {ts:6.3f} ms, hot8 {t8:6.3f} ms, hot16 {t16:6.3f} ms")
                del gs, g8, g16
        del p8, p16, rows_w
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
