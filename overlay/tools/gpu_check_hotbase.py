#!/usr/bin/env python3
"""GPU check + microbench of draftbytes `hotbase` (compact draft base logits). No server.

  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/gpu_check_hotbase.py

For draft walks of bs x 7 rows (bs 1, 8, 24) with the serving packs (fold8:row, head8:row, wide8, w8tile) and a
65536-id hot set over a 201024 vocabulary (vocab limit 200058, softcap 30):
  v2.7.2: base = full(-inf, [rows, V]); base.index_copy_(1, ids, head(hidden)); per step hot_step(base[:, s])
  hotbase: base = head(hidden) [rows, N]; per step hot_step(base[:, s], base_compact=True)
The 7 step-logit buffers must be bit-identical; timing is the base-logits part (head + fill + scatter vs head) in a
CUDA graph with cold L2.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

V, LIMIT, NHOT, HID, LAT, GAMMA, CAP = 201024, 200058, 65536, 2304, 256, 7, 30.0
FAILED = []


def main():
    from inkling_draftbytes import _opts, ops
    from inkling_draftbytes.quant import quantize_mode

    opts = _opts.parse("fold8,head8,draftmx8,wide8,w8tile", environ={})
    ops.configure(opts)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(11)
    print("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"), flush=True)
    ids = torch.randperm(LIMIT + 500, device=dev, generator=g)[:NHOT].sort().values  # a few ids past the limit
    ids64, ids32 = ids.to(torch.int64), ids.to(torch.int32)
    head = quantize_mode((torch.randn((NHOT, HID), device=dev, generator=g) * 0.02).to(torch.bfloat16),
                         opts.head8)
    fold = quantize_mode((torch.randn((NHOT, LAT), device=dev, generator=g) * 0.05).to(torch.bfloat16),
                         opts.fold8)
    fold_bias = (torch.randn(NHOT, device=dev, generator=g) * 0.1).to(torch.bfloat16)
    flush = torch.ones(64 << 20, device=dev)
    fo = torch.empty((), device=dev)
    for bs in (1, 8, 24):
        rows = bs * GAMMA
        hidden = torch.randn((rows, HID), device=dev, generator=g).to(torch.bfloat16)
        lat = [torch.randn((bs, LAT), device=dev, generator=g).to(torch.bfloat16) for _ in range(GAMMA)]
        outs = {k: [torch.empty((bs, V), dtype=torch.bfloat16, device=dev) for _ in range(GAMMA)] for k in "ab"}

        def base_full():
            compact = ops.w8_linear(hidden, head, None, torch.bfloat16)
            full = torch.full((rows, V), float("-inf"), dtype=compact.dtype, device=dev)
            full.index_copy_(1, ids64, compact)
            return full

        def base_compact():
            return ops.w8_linear(hidden, head, None, torch.bfloat16)

        def walk(base, key, compact):
            b3 = base.view(bs, GAMMA, -1)
            for s in range(GAMMA):
                out = outs[key][s]
                out.fill_(float("-inf"))
                ops.hot_step(lat[s], fold, fold_bias, b3[:, s], ids32, out, softcap=CAP, vocab_limit=LIMIT,
                             base_compact=compact)

        def graph(fn):
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                fn()
            torch.cuda.current_stream().wait_stream(s_)
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr):
                fn()
            return gr

        with torch.no_grad():
            ga = graph(lambda: walk(base_full(), "a", False))
            gb = graph(lambda: walk(base_compact(), "b", True))
            ok = True
            for _ in range(2):
                hidden.copy_(torch.randn((rows, HID), device=dev, generator=g).to(torch.bfloat16))
                for t in lat:
                    t.copy_(torch.randn(t.shape, device=dev, generator=g).to(torch.bfloat16))
                ga.replay(); gb.replay(); torch.cuda.synchronize()
                for x, y in zip(outs["a"], outs["b"]):
                    ok &= torch.equal(x.view(torch.int16), y.view(torch.int16))
            if not ok:
                FAILED.append(f"bs={bs}")
            pa, pb = graph(base_full), graph(base_compact)
            ts = {"full": [], "compact": []}
            for _ in range(30):
                for k, gr in (("full", pa), ("compact", pb)):
                    torch.sum(flush, 0, out=fo)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                    ts[k].append(e0.elapsed_time(e1) * 1e3)
            f, c = statistics.median(ts["full"][3:]), statistics.median(ts["compact"][3:])
            print(f"bs {bs:2} ({rows:3} rows): step logits {'bit-identical' if ok else 'DIFFER'} (7 steps, 2 replays); "
                  f"base logits head+fill+scatter {f:7.1f} us vs head {c:7.1f} us -> {(f - c) / 1000:.3f} ms/step",
                  flush=True)
        del ga, gb, pa, pb
    print("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK", flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
