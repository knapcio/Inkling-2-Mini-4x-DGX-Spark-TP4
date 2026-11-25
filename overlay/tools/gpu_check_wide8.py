#!/usr/bin/env python3
"""GPU check + microbenchmark of inkling_draftbytes `wide8` (the FP8 / BF16 draft kernels reading the weight once per
call for any row count). Server stopped; ~3 GB GPU.

  PYTHONPATH=<overlay> python3 tools/gpu_check_wide8.py --ref <copy of the pre-wide8 inkling_draftbytes/_triton.py>

For the draft LM head (201024 x 2304 FP8, head8:row and head8:mx, real unembed weights when /model is mounted) and
the Markov fold shape (201024 x 256 FP8 row, + bias), at M = 8, 64, 128, 168, 192, 256, 384:
  [gated] the new default launch (ROWS 0) == the pre-wide8 kernel bit for bit; wide8 mode 1 and mode 2 == the
          pre-wide8 kernel bit for bit (same per-element math); CUDA-graph replay == eager; the fused hot step
          (fold + combine + scatter) the same;
  [reported] median us (L2 evicted by READING a 256 MiB buffer before each call), GB/s of the weight bytes, and the
          pre-wide8 kernel's time; a num_stages sweep for mode 2 at 192 rows.
Last line `wide8 gpu_check result PASS|FAIL`.
"""

import argparse
import glob
import importlib.util
import os
import statistics
import sys

OVERLAY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
    p.add_argument("--ref", required=True)
    p.add_argument("--ms", default="8,64,128,168,192,256,384")
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--no-time", action="store_true")
    p.add_argument("--tile-sweep", action="store_true", help="only: mode-1 BN x warps x stages sweep with bit checks")
    p.add_argument("--ops", action="store_true", help="only: the package's dispatch (ops) with wide8 off vs on")
    args = p.parse_args()

    import torch

    import inkling_draftbytes  # noqa: F401  (parent package of the reference module)
    from inkling_draftbytes import _triton as new
    from inkling_draftbytes.quant import quantize

    spec = importlib.util.spec_from_file_location("inkling_draftbytes._triton_ref", args.ref)
    ref = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ref
    spec.loader.exec_module(ref)

    dev = torch.device("cuda")
    flush = torch.ones(64 << 20, dtype=torch.float32, device=dev)
    sink = torch.empty((), device=dev)

    def timeit(fn, iters=None):
        iters = iters or args.iters
        for _ in range(2):
            fn()
        torch.cuda.synchronize()
        st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
        for i in range(iters):
            sink.copy_(flush.sum())
            st[i].record()
            fn()
            en[i].record()
        torch.cuda.synchronize()
        return statistics.median(s.elapsed_time(e) * 1000.0 for s, e in zip(st, en))

    def bits(a, b):
        return a.shape == b.shape and torch.equal(a.view(torch.int16), b.view(torch.int16))

    head = None
    if os.path.isdir(args.model):
        from safetensors import safe_open

        for path in sorted(glob.glob(os.path.join(args.model, "*.safetensors"))):
            with safe_open(path, framework="pt", device="cpu") as h:
                if "model.llm.unembed.weight" in h.keys():
                    head = h.get_tensor("model.llm.unembed.weight").to(dev)
                    break
    if head is None:
        log("wide8 note: random head weights")
        head = (torch.randn(201024, 2304, device=dev) * 0.02).to(torch.bfloat16)
    g = torch.Generator(device=dev).manual_seed(5)
    fold = (torch.randn(head.shape[0], 256, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    fold_b = (torch.randn(head.shape[0], device=dev, generator=g) * 0.1).to(torch.bfloat16)

    from inkling_draftbytes.quant import bf16_pack

    cases = [
        ("head8:row", quantize(head, "row", 0), None, dict(bn=32, bk=256, num_warps=4)),
        ("head8:mx", quantize(head, "mx", 0), None, dict(bn=32, bk=256, num_warps=4)),
        ("fold8:row", quantize(fold, "row", 0), fold_b, dict(bn=64, bk=128, num_warps=4)),
        ("head bf16 (gemv / own kernel)", bf16_pack(head), None, dict(bn=32, bk=256, num_warps=4)),
    ]
    torch.cuda.empty_cache()
    if args.ops:
        from inkling_draftbytes import _opts, ops

        off, on = _opts.parse("head8,fold8", environ={}), _opts.parse("head8,fold8,wide8", environ={})
        log(f"wide8 ops options on: {on.describe()} mode={on.wide8_mode} bn={on.wide8_bn} warps={on.wide8_warps} "
            f"stages={on.wide8_stages}")
        for label, pack, bias, cfg in (cases[0], cases[2]):
            narrow = label.startswith("fold")
            for m in (8, 64, 65, 128, 168, 192, 256, 384):
                x = (torch.randn(m, pack.k, device=dev, generator=g)).to(torch.bfloat16)
                ops.configure(off)
                y0 = ops.w8_linear(x, pack, bias, torch.bfloat16, narrow_k=narrow)
                t0 = timeit(lambda: ops.w8_linear(x, pack, bias, torch.bfloat16, narrow_k=narrow))
                ops.configure(on)
                y1 = ops.w8_linear(x, pack, bias, torch.bfloat16, narrow_k=narrow)
                t1 = timeit(lambda: ops.w8_linear(x, pack, bias, torch.bfloat16, narrow_k=narrow))
                torch.cuda.synchronize()
                gate(f"wide8 ops {label} M={m}", bits(y0, y1),
                     f"off_us={t0:.1f} wide8_us={t1:.1f} speedup={t0 / t1:.2f} "
                     f"({pack.nbytes() / t1 / 1e3:.0f} GB/s) bits={'IDENTICAL' if bits(y0, y1) else 'DIFFERENT'}")
        ops.configure(off)
        log(f"wide8 gpu_check result {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failed: {', '.join(FAILS[:6])})")
        sys.exit(0 if not FAILS else 1)
    if args.tile_sweep:
        for label, pack, bias, cfg in cases[:1] + cases[2:3]:
            for m in (128, 192, 384):
                x = (torch.randn(m, pack.k, device=dev, generator=g)).to(torch.bfloat16)
                r = torch.empty((m, pack.n), dtype=torch.bfloat16, device=dev)
                ref.w8_mm(x, pack, bias, r, **cfg)
                t_ref = timeit(lambda: ref.w8_mm(x, pack, bias, r, **cfg))
                res = []
                for bn in (32, 64, 128):
                    for warps in (4, 8):
                        for st in (1, 2, 3):
                            c2 = dict(cfg, bn=bn, num_warps=warps)
                            o = torch.full_like(r, float("nan"))
                            try:
                                new.w8_mm(x, pack, bias, o, **c2, rows=1, num_stages=st)
                                torch.cuda.synchronize()
                                same = bits(o, r)
                                t = timeit(lambda: new.w8_mm(x, pack, bias, o, **c2, rows=1, num_stages=st))
                                kk = new.LAST["w8_mm"]
                                res.append((t, bn, warps, st, same, kk.n_regs, kk.n_spills))
                            except Exception as exc:  # noqa: BLE001
                                log(f"wide8 tile {label} M={m} bn={bn} w={warps} st={st} failed: {exc!r}"[:160])
                for t, bn, warps, st, same, regs, sp in sorted(res)[:6]:
                    log(f"wide8 tile {label} M={m} bn={bn} warps={warps} stages={st} us={t:.1f} "
                        f"bits_vs_ref={'IDENTICAL' if same else 'DIFFERENT'} regs={regs} spills={sp} "
                        f"ref_us={t_ref:.1f} speedup={t_ref / t:.2f}")
        log(f"wide8 tile sweep done")
        return
    ms = [int(v) for v in args.ms.split(",") if v]
    ms_bf16 = [64, 130, 192]
    table = []
    for label, pack, bias, cfg in cases:
        k = pack.k
        wbytes = pack.nbytes()
        for m in (ms_bf16 if label.startswith("head bf16") else ms):
            x = (torch.randn(m, k, device=dev, generator=g)).to(torch.bfloat16)
            outs = {}
            for name, fn in (
                ("ref", lambda o: ref.w8_mm(x, pack, bias, o, **cfg)),
                ("rows0", lambda o: new.w8_mm(x, pack, bias, o, **cfg, rows=0)),
                ("rows1", lambda o: new.w8_mm(x, pack, bias, o, **cfg, rows=1)),
                ("rows2", lambda o: new.w8_mm(x, pack, bias, o, **cfg, rows=2)),
                ("rows2s2", lambda o: new.w8_mm(x, pack, bias, o, **cfg, rows=2, num_stages=2)),
            ):
                o = torch.full((m, pack.n), float("nan"), dtype=torch.bfloat16, device=dev)
                try:
                    fn(o)
                except Exception as exc:  # noqa: BLE001  (e.g. out of shared memory at 2 stages)
                    log(f"wide8 note {label} M={m} {name} failed: {exc!r}"[:240])
                    if name == "rows2s2":
                        continue
                    raise
                k_last = new.LAST.get("w8_mm") if name != "ref" else None
                if k_last is not None and m in (192, 384):
                    try:
                        log(f"wide8 kernel {label} M={m} {name} regs={k_last.n_regs} spills={k_last.n_spills} "
                            f"shared={k_last.metadata.shared}")
                    except Exception:  # noqa: BLE001
                        pass
                outs[name] = (o, fn)
            torch.cuda.synchronize()
            # graph replay of mode 2
            og = torch.full((m, pack.n), float("nan"), dtype=torch.bfloat16, device=dev)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                new.w8_mm(x, pack, bias, og, **cfg, rows=2)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                new.w8_mm(x, pack, bias, og, **cfg, rows=2)
            og.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            r = outs["ref"][0]
            same = {n: bits(outs[n][0], r) for n in ("rows0", "rows1", "rows2", "rows2s2") if n in outs}
            dmax = max(float((outs[n][0].float() - r.float()).abs().max()) for n in ("rows1", "rows2"))
            gate(f"wide8 bits {label} M={m}", all(same.values()) and bits(og, r) and not torch.isnan(r).any(),
                 " ".join(f"{n}={'IDENTICAL' if v else 'DIFFERENT'}" for n, v in same.items())
                 + f" graph={'IDENTICAL' if bits(og, r) else 'DIFFERENT'} max|wide-ref|={dmax:.3e}")
            if not args.no_time:
                t = {n: timeit(lambda: outs[n][1](outs[n][0])) for n in ("ref", "rows1", "rows2")}
                table.append((label, m, t["ref"], t["rows1"], t["rows2"], wbytes))
                log(f"wide8 time {label} M={m} ref_us={t['ref']:.1f} ({wbytes / t['ref'] / 1e3:.0f} GB/s) "
                    f"mode1_us={t['rows1']:.1f} mode2_us={t['rows2']:.1f} ({wbytes / t['rows2'] / 1e3:.0f} GB/s) "
                    f"speedup={t['ref'] / min(t['rows1'], t['rows2']):.2f}")
                if m == 192 and label == "head8:row":
                    for st in (1, 2, 3, 4):
                        for mode in (1, 2):
                            try:
                                o = outs["rows2"][0]
                                tt = timeit(lambda: new.w8_mm(x, pack, bias, o, **cfg, rows=mode, num_stages=st))
                                log(f"wide8 sweep {label} M={m} mode={mode} stages={st} us={tt:.1f}")
                            except Exception as exc:  # noqa: BLE001
                                log(f"wide8 sweep {label} M={m} mode={mode} stages={st} failed: {exc!r}"[:200])
            del outs, og, graph
            torch.cuda.empty_cache()
        # the fused hot step on the fold shape (hot ids = a strided subset)
        if label == "fold8:row":
            # sparse hot ids (every 3rd id + an offset), a count that is not a multiple of BN
            ids32 = torch.arange(1, pack.n, 3, device=dev, dtype=torch.int32)[: 50001]
            from inkling_draftbytes.quant import quantize as _q

            pack = _q(fold[ids32.long()], "row", 0)
            bias = fold_b[ids32.long()].contiguous()
            for m in (8, 130, 384):
                lat = torch.tanh(torch.randn(m, k, device=dev, generator=g)).to(torch.bfloat16)
                width = fold.shape[0]
                base = torch.randn(m, width, device=dev, generator=g).to(torch.bfloat16)
                o_ref = torch.full((m, width), float("-inf"), dtype=torch.bfloat16, device=dev)
                ref.hot_step(lat, pack, bias, base, ids32, o_ref, softcap=30.0, vocab_limit=width - 7, **cfg)
                res = []
                for mode in (0, 1, 2):
                    o = torch.full_like(o_ref, float("-inf"))
                    new.hot_step(lat, pack, bias, base, ids32, o, softcap=30.0, vocab_limit=width - 7, **cfg,
                                 rows=mode)
                    torch.cuda.synchronize()
                    res.append(bits(o, o_ref))
                gate(f"wide8 hot_step M={m}", all(res), f"modes 0/1/2 identical={res}")
        del pack
        torch.cuda.empty_cache()
    del head

    if table:
        log("wide8 summary (us, L2 cold; GB/s of weight bytes)")
        log("case          M     ref(64-row blocks)   wide8 mode1   wide8 mode2   GB/s(best)  speedup")
        for label, m, a, b, c, wb in table:
            best = min(b, c)
            log(f"{label:12s} {m:4d} {a:12.1f} {b:14.1f} {c:13.1f} {wb / best / 1e3:10.0f} {a / best:8.2f}")
    log(f"wide8 gpu_check result {'PASS' if not FAILS else 'FAIL'} ({len(FAILS)} failed: {', '.join(FAILS[:6])})")
    sys.exit(0 if not FAILS else 1)


if __name__ == "__main__":
    main()
