#!/usr/bin/env python3
"""GPU check + microbenchmark of inkling_pfuse against the stock SGLang kernels.

Runs inside the SGLang image with the model server stopped; reads a few small real tensors from the checkpoint
(attn-res queries, loop message queries, controller weights, norm weights of layers 7, 11 and 12), never changes
clocks, peak device memory ~3 GB.

  python3 tools/gpu_check_pfuse.py [--model /model] [--sections resnorm,loop,graphs,install] [--reps 20]
                                   [--tokens 8192,2048,1000] [--json out.json]

Per case it reports:
  bit   fraction of BF16 outputs bit-equal to the stock sequence and the max difference in BF16 ulps
  err   max |out - ref| / max|ref| and mean |out - ref| of the stock and of the fused output against an FP64
        reference of the same math (same BF16 rounding points: controller normed input, controller hidden)
  time  stock sequence vs fused at --tokens[0] rows, L2 flushed by reading a 128 MiB buffer before each rep
A case FAILS when the fused error exceeds 1.5x the stock error (max and mean) or when a graph replay differs from
eager; exit code 1 then.
"""

import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OVERLAY)

D = 2304
RESULTS = []
FAILED = []


def log(*a):
    print(*a, flush=True)


class Weights:
    def __init__(self, model):
        from safetensors import safe_open

        self.files = sorted(glob.glob(os.path.join(model, "model*.safetensors")))
        self.handles = [safe_open(f, "pt", device="cpu") for f in self.files]
        self.names = {}
        for h in self.handles:
            for k in h.keys():
                self.names[k] = h

    def get(self, name):
        return self.names["model.llm." + name].get_tensor("model.llm." + name).cuda()


def stock_ops():
    from sglang.kernels.ops.attention.inkling2_attn_res import attn_res
    from sglang.kernels.ops.elementwise.inkling2_loop import message_controllers
    from sglang.kernels.ops.layernorm.inkling2_rms_norm import rms_norm

    return attn_res, message_controllers, rms_norm


# ---------------------------------------------------------------- FP64 references
def ref_attn_res(sources, addend, queries, normalize=True):
    xs = [s.double() for s in sources]
    if addend is not None:
        xs[-1] = xs[-1] + addend.double()
    x = torch.stack(xs)  # N, T, D
    rstd = torch.rsqrt(x.pow(2).mean(-1) + 1e-6)  # N, T
    logits = torch.einsum("ntd,qd->qnt", x, queries.double()) * rstd[None]
    p = torch.softmax(logits, dim=1)  # q, n, t
    if normalize:
        p = p * rstd[None]
    return torch.einsum("qnt,ntd->qtd", p, x)


def ref_norm(x, w, eps=1e-6):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w.double()


def ref_loop(dest, msgs, down, gain, forget, shared):
    """FP64 message_controllers with the stock BF16 rounding of the normed input and of the hidden."""
    outs = []
    for i in range(msgs.shape[0]):
        d = dest[0] if shared else dest[i]
        m = msgs[i]
        nd = (d * torch.rsqrt(d.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).double()
        nm = (m * torch.rsqrt(m.pow(2).mean(-1, keepdim=True) + 1e-6)).to(torch.bfloat16).double()
        h = (torch.cat([nd, nm], -1) @ down[i].double().t()).to(torch.bfloat16).double()
        g = h @ gain[i].double().t()
        f = h @ forget[i].double().t()
        outs.append((1 - torch.tanh(f)) * d + torch.tanh(g) * m)
    return torch.stack(outs)


def _ordered(x):
    """BF16 bits -> integers ordered like the values (+0 and -0 both map to 0)."""
    b = x.contiguous().view(torch.int16).int()
    return torch.where(b < 0, -(b & 0x7FFF), b)


def compare(name, stock, fused, ref, rows):
    """bit statistics over ALL rows; errors vs the FP64 reference on the first `rows` rows."""
    s = stock[..., :rows, :].float()
    f = fused[..., :rows, :].float()
    r = ref.float()
    if stock.dtype == torch.bfloat16:
        si, fi = _ordered(stock), _ordered(fused)
        bit = (si == fi).float().mean().item()
        ulps = (si - fi).abs().max().item()
    else:
        bit = (stock == fused).float().mean().item()
        ulps = float("nan")
    scale = r.abs().max().item()
    es, ef = (s - r).abs(), (f - r).abs()
    rec = dict(name=name, bit_equal=bit, max_ulps=ulps, stock_max=es.max().item() / scale, fused_max=ef.max().item() / scale,
               stock_mean=es.mean().item(), fused_mean=ef.mean().item())
    ok = rec["fused_max"] <= 1.5 * rec["stock_max"] + 1e-7 and rec["fused_mean"] <= 1.5 * rec["stock_mean"] + 1e-9
    rec["ok"] = ok
    log(f"  {name:44s} bit-equal {bit*100:7.3f} %  max {ulps} ulp | err/max stock {rec['stock_max']:.2e} fused "
        f"{rec['fused_max']:.2e} | mean stock {rec['stock_mean']:.2e} fused {rec['fused_mean']:.2e}  "
        f"{'ok' if ok else 'FAIL'}")
    if not ok:
        FAILED.append(name)
    RESULTS.append(rec)


def make_timer(reps):
    flush = torch.empty(32 << 20, dtype=torch.float32, device="cuda").fill_(1.0)

    def timeit(fn):
        fn()
        torch.cuda.synchronize()
        ts = []
        for _ in range(reps):
            flush.sum()
            s, e = torch.cuda.Event(True), torch.cuda.Event(True)
            s.record()
            fn()
            e.record()
            torch.cuda.synchronize()
            ts.append(s.elapsed_time(e) * 1e3)
        return statistics.median(ts)

    return timeit


def sources(T, n, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    scales = [0.6, 2.0, 1.0, 3.5, 0.8, 1.5]
    return [(torch.randn(T, D, device="cuda", generator=g) * scales[i % 6]).to(torch.bfloat16) for i in range(n)]


# ---------------------------------------------------------------- resnorm
def section_resnorm(W, tokens, timeit, ref_rows):
    from inkling_pfuse import mix

    attn_res, _, rms_norm = stock_ops()
    cases = [
        # name, n sources, addend, query tensor, norm weights
        ("attn local N=3", 3, False, ["attn_res.attn_queries.6"], ["layers.7.attn_norm.weight"]),
        ("attn global N=3 Q=3", 3, False, ["attn_res.attn_queries.10"],
         ["layers.11.q_input_norm.weight", "layers.11.k_input_norm.weight", "layers.11.v_input_norm.weight"]),
        ("mlp N=4 +addend", 4, True, ["attn_res.mlp_queries.7"], ["layers.7.mlp_norm.weight"]),
        ("mlp boundary N=5", 5, False, ["attn_res.mlp_queries.18"], ["layers.18.mlp_norm.weight"]),
    ]
    for T in tokens:
        log(f"resnorm T={T}")
        for name, n, add, qn, wn in cases:
            src = sources(T, n + (1 if add else 0), seed=T + n)
            addend = src.pop() if add else None
            q = W.get(qn[0]).float()
            q = (q.unsqueeze(0) if q.ndim == 1 else q) * D ** -0.5
            q = q.contiguous()
            ws = [W.get(w) for w in wn]

            def stock():
                outs = attn_res(src, addend, q, normalize_values=True)
                y = torch.empty((len(ws), T, D), dtype=torch.bfloat16, device="cuda")
                for o, w, yy in zip(outs, ws, y):
                    rms_norm(o, w, 1e-6, output_dtype=torch.bfloat16, out=yy)
                return y

            def fused():
                return mix.resnorm(src, addend, q, ws, 1e-6, True)

            ys, yf = stock(), fused()
            r = ref_attn_res([s[:ref_rows] for s in src], None if addend is None else addend[:ref_rows], q)
            ref = torch.stack([ref_norm(r[i], ws[i]) for i in range(len(ws))])
            compare(f"resnorm {name} T={T}", ys, yf, ref, ref_rows)
            if T == tokens[0]:
                ts, tf = timeit(stock), timeit(fused)
                log(f"    time stock {ts:8.1f} us  fused {tf:8.1f} us  saved {ts - tf:7.1f} us")
                RESULTS.append(dict(name=f"time resnorm {name}", tokens=T, stock_us=ts, fused_us=tf))


# ---------------------------------------------------------------- loop mix
def section_loop(W, tokens, timeit, ref_rows):
    from inkling_pfuse import mix

    attn_res, message_controllers, rms_norm = stock_ops()
    from sglang.srt.models.inkling_common.inkling2_loop import scaled_loop_queries

    mq = scaled_loop_queries(W.get("middle_cycle_cross_pass_queries"), W.get("middle_cycle_cross_pass_kv_queries"))
    mods = ("attn", "k", "v", "mlp")

    def ctrl(layer):
        return [torch.stack([W.get(f"layers.{layer}.controllers.{m}.{w}") for m in mods])
                for w in ("down_weight", "gain_weight", "forget_weight")]

    cases = [
        # name, layer, side, n state sources, addend, state query, norms
        ("attn local L7 N=3", 7, "attn", 3, False, "attn_res.attn_queries.6", ["layers.7.attn_norm.weight"] * 3),
        ("attn global L11 N=3", 11, "attn", 3, False, "attn_res.attn_queries.10",
         ["layers.11.q_input_norm.weight", "layers.11.k_input_norm.weight", "layers.11.v_input_norm.weight"]),
        ("mlp L7 N=3 +addend", 7, "mlp", 3, True, "attn_res.mlp_queries.7", None),
        ("mlp L12 boundary N=4", 12, "mlp", 4, False, "attn_res.mlp_queries.12", None),
    ]
    for T in tokens:
        log(f"loopmix T={T}")
        for name, layer, side, n, add, qn, wn in cases:
            src = sources(T, n + (1 if add else 0), seed=3 * T + layer)
            addend = src.pop() if add else None
            bank = sources(T, 2, seed=7 * T + layer)
            q = W.get(qn).float()
            q = ((q.unsqueeze(0) if q.ndim == 1 else q) * D ** -0.5).contiguous()
            rows = slice(0, 3) if side == "attn" else slice(3, 4)
            bq = mq[layer - 6, rows].contiguous()
            down, gain, forget = (w[rows].contiguous() for w in ctrl(layer))
            ws = [W.get(w) for w in wn] if wn else None
            mlp_w = W.get(f"layers.{layer}.mlp_norm.weight")
            shared = q.shape[0] == 1 and bq.shape[0] == 3

            def stock():
                dest = attn_res(src, addend, q, normalize_values=True)
                msgs = attn_res(bank, None, bq, normalize_values=True)
                dests = dest * 3 if shared else dest
                mixed = message_controllers(tuple(d.float() for d in dests), msgs, down, gain, forget)
                if ws is None:
                    return rms_norm(mixed[0], mlp_w, 1e-6, output_dtype=torch.bfloat16)[None]
                if shared:
                    return rms_norm(mixed, ws[0], 1e-6, output_dtype=torch.bfloat16)
                y = torch.empty((3, T, D), dtype=torch.bfloat16, device="cuda")
                for o, w, yy in zip(mixed.unbind(0), ws, y):
                    rms_norm(o, w, 1e-6, output_dtype=torch.bfloat16, out=yy)
                return y

            def fused(fuse_norm=True):
                out = mix.loop_mix(src, addend, q, True, bank, bq, down, gain, forget, norm_weights=ws, eps_norm=1e-6,
                                   fuse_norm=fuse_norm)
                if ws is None:
                    return rms_norm(out[0], mlp_w, 1e-6, output_dtype=torch.bfloat16)[None]
                return out

            ys, yf = stock(), fused()
            r = ref_attn_res([s[:ref_rows] for s in src], None if addend is None else addend[:ref_rows], q)
            rm = ref_attn_res([b[:ref_rows] for b in bank], None, bq)
            mixed = ref_loop(r, rm, down, gain, forget, shared)
            nw = ws or [mlp_w]
            ref = torch.stack([ref_norm(mixed[i], nw[i]) for i in range(mixed.shape[0])])
            compare(f"loopmix {name} T={T}", ys, yf, ref, ref_rows)
            if ws is None:  # also the FP32 mix output itself
                dest = attn_res(src, addend, q, normalize_values=True)
                msgs = attn_res(bank, None, bq, normalize_values=True)
                ms = message_controllers(tuple(d.float() for d in dest), msgs, down, gain, forget)
                mf = mix.loop_mix(src, addend, q, True, bank, bq, down, gain, forget, norm_weights=None)
                compare(f"loopmix {name} FP32 mix T={T}", ms, mf, mixed, ref_rows)
            if ws is not None:
                compare(f"loopmix {name} one-pass+stock norm T={T}", ys, fused(False), ref, ref_rows)
            if T == tokens[0]:
                ts, tf = timeit(stock), timeit(fused)
                extra = {}
                msg = ""
                if ws is not None:
                    extra["fused_onepass_us"] = t1 = timeit(lambda: fused(False))
                    msg = f"  one-pass+stock norm {t1:8.1f} us"
                log(f"    time stock {ts:8.1f} us  fused {tf:8.1f} us  saved {ts - tf:7.1f} us{msg}")
                RESULTS.append(dict(name=f"time loopmix {name}", tokens=T, stock_us=ts, fused_us=tf, **extra))


# ---------------------------------------------------------------- split normed + split down vs stock down
def section_split(tokens):
    """loop_mix with the stock _controller_down on [NUM, T, 2D] vs _controller_down_split on [ND + NUM, T, D]:
    the controller output (FP32) must be bit-identical."""
    from inkling_pfuse import mix

    for T in tokens:
        src = sources(T, 3, 21)
        add = sources(T, 1, 22)[0]
        bank = sources(T, 2, 23)
        q1 = (torch.randn(1, D, device="cuda") * D ** -0.5).contiguous()
        q3 = (torch.randn(3, D, device="cuda") * D ** -0.5).contiguous()
        down = (torch.randn(3, 64, 2 * D, device="cuda") * 0.02).to(torch.bfloat16)
        gain = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)
        forget = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)
        for name, args in (("local", (src, None, q1, True, bank, q3, down, gain, forget)),
                           ("global", (src, None, q3, True, bank, q3, down, gain, forget)),
                           ("mlp", (src, add, q1, True, bank, q1, down[:1], gain[:1], forget[:1]))):
            mix.MIX["split_down"] = False
            a = mix.loop_mix(*args)
            mix.MIX["split_down"] = True
            b = mix.loop_mix(*args)
            same = torch.equal(a, b)
            log(f"  split down {name} T={T}: {'bit-equal' if same else 'DIFFERS'}")
            RESULTS.append(dict(name=f"split down {name} T={T}", ok=same))
            if not same:
                FAILED.append(f"split down {name} T={T}")


# ---------------------------------------------------------------- sconv + residual
def section_sconvres(W, tokens, timeit):
    from inkling_pfuse import mix
    from sglang.kernels.ops.elementwise.inkling2_residual_add import residual_add
    from sglang.kernels.ops.mamba.inkling2_sconv import short_convolution

    w = W.get("layers.7.mlp_sconv.weight") if W is not None else None
    for T in [t for t in tokens if t >= mix.sconv_min_rows()] + [2048, 2049, 3001]:
        x, attn, part = sources(T, 3, seed=11 * T)
        weight = w.reshape(D, -1) if w is not None else (torch.randn(D, 4, device="cuda") * 0.3).to(torch.bfloat16)
        # two sequences: the first continues a cached prefix (slot 1), the second starts fresh
        cut = T // 3
        cache = (torch.randn(3, 3, D, device="cuda")).to(torch.bfloat16)
        meta = {
            "safe_idx": torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
            "cache_mask": torch.tensor([True, False], device="cuda"),
            "cu": torch.tensor([0, cut, T], dtype=torch.int64, device="cuda"),
            "si": torch.cat([torch.zeros(cut, dtype=torch.int32), torch.ones(T - cut, dtype=torch.int32)]).cuda(),
        }
        for has_partial in (True, False):
            def stock():
                y = short_convolution(x, weight, cache, meta)
                return residual_add(part, attn, y) if has_partial else attn + y

            def fused():
                return mix.sconv_residual(x, weight, cache, meta, attn, part if has_partial else None)

            ys, yf = stock(), fused()
            same = torch.equal(ys, yf)
            log(f"  sconvres T={T} partial={has_partial}: {'bit-equal' if same else 'DIFFERS'}")
            RESULTS.append(dict(name=f"sconvres T={T} partial={has_partial}", ok=same))
            if not same:
                FAILED.append(f"sconvres T={T} partial={has_partial}")
            if T == tokens[0]:
                ts, tf = timeit(stock), timeit(fused)
                log(f"    time stock {ts:8.1f} us  fused {tf:8.1f} us  saved {ts - tf:7.1f} us")
                RESULTS.append(dict(name=f"time sconvres partial={has_partial}", tokens=T, stock_us=ts, fused_us=tf))


# ---------------------------------------------------------------- CUDA graphs
def section_graphs(W):
    from inkling_pfuse import mix

    T = 2304
    src = sources(T, 3, 1)
    xs = sources(T, 1, 9)[0]
    sw = (torch.randn(D, 4, device="cuda") * 0.3).to(torch.bfloat16)
    cache = torch.randn(3, 3, D, device="cuda").to(torch.bfloat16)
    meta = {
        "safe_idx": torch.tensor([1, 2], dtype=torch.int64, device="cuda"),
        "cache_mask": torch.tensor([True, False], device="cuda"),
        "cu": torch.tensor([0, 700, T], dtype=torch.int64, device="cuda"),
        "si": torch.cat([torch.zeros(700, dtype=torch.int32), torch.ones(T - 700, dtype=torch.int32)]).cuda(),
    }
    add = sources(T, 1, 2)[0]
    bank = sources(T, 2, 3)
    q1 = (torch.randn(1, D, device="cuda") * D ** -0.5).contiguous()
    q3 = (torch.randn(3, D, device="cuda") * D ** -0.5).contiguous()
    w = [torch.rand(D, device="cuda").to(torch.bfloat16) + 0.5 for _ in range(3)]
    down = (torch.randn(3, 64, 2 * D, device="cuda") * 0.02).to(torch.bfloat16)
    gain = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)
    forget = (torch.randn(3, D, 64, device="cuda") * 0.1).to(torch.bfloat16)

    def run():
        a = mix.resnorm(src, add, q3, w, 1e-6, True)
        b = mix.loop_mix(src, None, q1, True, bank, q3, down, gain, forget, norm_weights=w)
        c = mix.loop_mix(src, add, q1, True, bank, q1, down[:1], gain[:1], forget[:1], norm_weights=None)
        d = mix.loop_mix(src, None, q3, True, bank, q3, down, gain, forget, norm_weights=w)
        e = mix.sconv_residual(xs, sw, cache, meta, src[0], src[1])
        f = mix.sconv_residual(xs, sw, cache, meta, src[0], None)
        return a, b, c, d, e, f

    eager = run()
    torch.cuda.synchronize()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = run()
    for i in range(3):
        for x in (*src, add, *bank):
            x.copy_((x.float() * 1.0).to(torch.bfloat16))  # same values, new writes
        g.replay()
        torch.cuda.synchronize()
        same = all(torch.equal(a, b) for a, b in zip(eager, outs))
        log(f"graphs replay {i}: {'bit-equal to eager' if same else 'DIFFERS'}")
        if not same:
            FAILED.append("graphs")
    # inputs changed -> replay must follow
    for x in (*src, *bank, xs):
        x.add_(0.25)  # not a scale: normalized attn-res mixes are scale-invariant
    g.replay()
    torch.cuda.synchronize()
    fresh = run()
    same = all(torch.equal(a, b) for a, b in zip(fresh, outs)) and not any(torch.equal(a, b) for a, b in zip(fresh, eager))
    log(f"graphs replay after input change: {'bit-equal to eager' if same else 'DIFFERS'}")
    if not same:
        FAILED.append("graphs-change")


def section_install():
    code = (
        "import os, sys; sys.path.insert(0, %r); os.environ['INKLING_PFUSE']='all';"
        "import inkling_pfuse; inkling_pfuse.install();"
        "import sglang.srt.models.inkling2 as m; import inkling_pfuse.layer as L;"
        "f = m.Inkling2DecoderLayer._forward; print('patched' if getattr(f, L.MARK, False) else 'NOT patched',"
        " inkling_pfuse.STATE['applied'])" % OVERLAY
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    log("install:", out.stdout.strip(), out.stderr.strip()[-600:])
    if "patched" not in out.stdout or "NOT" in out.stdout:
        FAILED.append("install")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--sections", default="split,resnorm,loop,sconvres,graphs,install")
    ap.add_argument("--tokens", default="8192,4097,4096,1025,513")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--ref-rows", type=int, default=512)
    ap.add_argument("--json")
    a = ap.parse_args()
    tokens = [int(t) for t in a.tokens.split(",")]
    sections = a.sections.split(",")
    torch.manual_seed(0)
    W = Weights(a.model) if {"resnorm", "loop", "sconvres"} & set(sections) else None
    timeit = make_timer(a.reps)
    with torch.no_grad():
        if "resnorm" in sections:
            section_resnorm(W, tokens, timeit, a.ref_rows)
        if "loop" in sections:
            section_loop(W, tokens, timeit, a.ref_rows)
        if "split" in sections:
            section_split(tokens)
        if "sconvres" in sections:
            section_sconvres(W, tokens, timeit)
        if "graphs" in sections:
            section_graphs(W)
    if "install" in sections:
        section_install()
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    if a.json:
        json.dump(dict(results=RESULTS, failed=FAILED), open(a.json, "w"), indent=1)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
