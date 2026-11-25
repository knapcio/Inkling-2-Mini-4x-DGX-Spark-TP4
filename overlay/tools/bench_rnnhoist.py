#!/usr/bin/env python3
"""rnnhoist above 16 rows: the 7-step Markov walk's rnn_proj, stock (cuBLAS on the 5120-wide input each step) vs
hoisted (inkling_draftbytes.rnnhoist), CUDA graph, cold L2, at the serving draftbytes options (wide8, w8tile).

  python3 tools/bench_rnnhoist.py [--model /model] [--rows 8,16,24,32]

With --model the real rnn_proj weight of the MTP head is used (safetensors key *markov_head.rnn_proj.weight / .bias);
otherwise a random BF16 weight of the same shape. Reports walk time and, against an FP64 reference of every step's
projection, the fraction of BF16 outputs rounded to the nearest value (stock and hoisted), as in docs/experiments.md.
"""
import argparse
import glob
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

R, H, GAMMA = 256, 2304, 7


def load_rnn(model):
    """rnn_proj weight / bias of the MTP head from any *.safetensors under model (the MTP shard, mtp.safetensors,
    is not in the target's index, so every file's header is read)."""
    if model is None:
        return None
    from safetensors import safe_open

    found = {}
    for f in sorted(glob.glob(os.path.join(model, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if "rnn_proj" in k:
                    found[k] = s.get_tensor(k)
    w = next((v for k, v in found.items() if k.endswith("rnn_proj.weight")), None)
    b = next((v for k, v in found.items() if k.endswith("rnn_proj.bias")), None)
    if w is None:
        raise SystemExit(f"no *rnn_proj.weight in {model}/*.safetensors (omit --model for random weights)")
    print("rnn_proj from checkpoint:", sorted(found), flush=True)
    return w, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--rows", default="8,16,24,32")
    ap.add_argument("--reps", type=int, default=40)
    a = ap.parse_args()
    from inkling_draftbytes import _opts, ops, rnnhoist

    opts = _opts.parse("draftmx8,wide8,w8tile,rnnhoist", environ={})
    ops.configure(opts)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(3)
    real = load_rnn(a.model)
    lin = torch.nn.Linear(2 * R + 2 * H, 3 * R, bias=True, device=dev, dtype=torch.bfloat16)
    with torch.no_grad():
        if real is not None and real[0] is not None:
            lin.weight.copy_(real[0].to(dev))
            if real[1] is not None:
                lin.bias.copy_(real[1].to(dev))
        else:
            lin.weight.copy_(torch.randn(lin.weight.shape, device=dev, generator=g) * 0.02)
            lin.bias.copy_(torch.randn(lin.bias.shape, device=dev, generator=g) * 0.02)
    packs = rnnhoist.build_packs(lin, R, H)
    w1 = (torch.randn((201024, R), device=dev, generator=g) * 0.05).to(torch.bfloat16)
    flush = torch.ones(64 << 20, device=dev)
    fo = torch.empty((), device=dev)
    print("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"), flush=True)
    for bs in (int(x) for x in a.rows.split(",")):
        hidden = (torch.randn((bs, GAMMA, H), device=dev, generator=g)).to(torch.bfloat16)
        f_rnn = (torch.randn((bs, H), device=dev, generator=g)).to(torch.bfloat16)
        state0 = (torch.randn((bs, R), device=dev, generator=g) * 0.5).to(torch.bfloat16)
        prev = torch.randint(0, 200000, (GAMMA, bs), device=dev, generator=g)

        def stock():
            outs = []
            st = state0
            for s in range(GAMMA):
                x = torch.cat([st, torch.nn.functional.embedding(prev[s], w1), hidden[:, s], f_rnn], -1)
                p = lin(x)
                outs.append(p)
                st = p[:, :R]  # stand-in for the gates (the same for both arms)
            return outs

        def hoisted():
            outs = []
            walk = rnnhoist.begin_walk(packs, hidden, f_rnn, 64)
            st = state0
            for s in range(GAMMA):
                assert rnnhoist.step_index(walk, hidden[:, s], f_rnn) == s
                p = rnnhoist.hoisted_proj(packs, walk, s, st, prev[s], w1)
                outs.append(p)
                st = p[:, :R]
            return outs

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

        with torch.no_grad():
            gs, os_ = graph(stock)
            gh, oh = graph(hoisted)
            gs.replay(); gh.replay(); torch.cuda.synchronize()
            # FP64 reference of each step's projection on the stock arm's inputs
            W = lin.weight.double()
            B = lin.bias.double()
            near_s = near_h = total = 0
            st = state0
            for s in range(GAMMA):
                x = torch.cat([st, torch.nn.functional.embedding(prev[s], w1), hidden[:, s], f_rnn], -1).double()
                ref = x @ W.T + B
                st = os_[s][:, :R]
                if s == 0:  # identical inputs only at step 0 (later states diverge by the arms' roundings)
                    for out, tag in ((os_[0], "s"), (oh[0], "h")):
                        err = (out.double() - ref).abs()
                        bf = out
                        nb = torch.stack([(bf.view(torch.int16) + d).view(torch.bfloat16).double() for d in (-1, 1)])
                        ok = (err[None] <= (nb - ref[None]).abs()).all(0)
                        if tag == "s":
                            near_s += int(ok.sum())
                        else:
                            near_h += int(ok.sum())
                    total += ref.numel()
            ts = {"stock": [], "hoisted": []}
            for _ in range(a.reps):
                for k, gr in (("stock", gs), ("hoisted", gh)):
                    torch.sum(flush, 0, out=fo)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                    ts[k].append(e0.elapsed_time(e1) * 1e3)
            s_us, h_us = statistics.median(ts["stock"][3:]), statistics.median(ts["hoisted"][3:])
            print(f"rows {bs:3}: walk stock {s_us:7.1f} us, rnnhoist {h_us:7.1f} us ({s_us - h_us:+.1f} us); "
                  f"step-0 nearest-rounded: stock {near_s / total:.3%}, hoisted {near_h / total:.3%}", flush=True)
        del gs, gh
    return 0


if __name__ == "__main__":
    sys.exit(main())
