#!/usr/bin/env python3
"""GPU check + microbench of the TP certified head (certtp.py) on ONE GPU: the real head split into --world vocab
shards, each with its own pico coding, 6-bit tables and ops.Cert, and the (T, H) all-gather replaced by copies
between them (run in the SGLang image, server stopped; no collective, no second GPU).

    python3 inkling_cert/tools/gpu_check_certtp.py --model /model --hidden DIR [--world 4] [--quick]

Reference: the stock TP pipeline: pico's GEMV of every shard, the stock softcap per shard, the shards concatenated
(the gather is pure data movement), ids >= v_real set to -inf, the stock argmax. Gates (any failure -> FAIL):
  1  per shard, on = 0: pico's shard bit for bit (fallback), and a peer's +inf (T, H): pico's shard bit for bit
  2  real rows at M = 8 and 16: certified token == stock token by both resolutions, the stock gather of the certified
     shards and the stock deferred argmax (image softcap_row_argmax over the shard's REAL ids + the rank merge of
     LogitsProcessor.deferred_argmax); the certified flag is the same on every shard; computed tiles bit-identical to
     pico, the others -inf
  3  near-tie rows across shards (x along W_a + W_b, a and b on different shards, norms 60-800)
  4  exact ties across shards: a head copy with row b (later shard) = row a; the token is a
  5  one shard with on = 0: every shard serves its full shard
  6  CUDA graph: the four shards' screen / exchange / select / exact captured in one graph, replayed with changing
     rows and on toggled == eager
Bench (L2 flushed): pico's shard GEMV vs the certified shard path (screen_tp + select_tp + exact; the collective is
not in it: add the measured [M, 2] all-gather, ~10-15 us on RoCE), at the median and p90 batch of the SLOWEST shard,
plus the image's softcap_row_argmax on a shard. Last line: `certtp gpu_check result PASS|FAIL`.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OVERLAY)
sys.path.insert(0, HERE)

from gpu_check_cert import gate, FAIL, load_head, load_hidden, stock_fns  # noqa: E402
from inkling_cert import build as cbuild, cert_math as cm, ops as cops, quant  # noqa: E402
from inkling_pico12 import build as pbuild, ops as pops  # noqa: E402


class Shards:
    def __init__(self, W, world, v_real, ext):
        self.world, self.v_real = world, v_real
        self.n = W.shape[0] // world
        assert self.n * world == W.shape[0] and self.n % 16 == 0
        self.W, self.packs, self.certs = W, [], []
        for r in range(world):
            w = W[r * self.n:(r + 1) * self.n].contiguous()
            pack = pops.encode(w, f"lm_head shard {r}")
            cert = cops.Cert(ext, pack, quant.build_tables(w), pops.SET.wk, W.device)
            cert.enable_tp(world, r, r * self.n, None)
            self.packs.append(pack)
            self.certs.append(cert)
        self.v = [max(0, min(self.n, v_real - r * self.n)) for r in range(world)]

    def full(self, x):
        outs = []
        for p in self.packs:
            o = torch.empty((x.shape[0], self.n), dtype=torch.float32, device=x.device)
            pops.gemv(x, p, o)
            outs.append(o)
        return outs

    def run(self, x, ons):
        """Certified shards with the exchange done by copies (the all-gather's result on every rank)."""
        m = x.shape[0]
        outs = [torch.empty((m, self.n), dtype=torch.float32, device=x.device) for _ in range(self.world)]
        for c, on, v in zip(self.certs, ons, self.v):
            c.screen_tp(x, on, v)
        recv = torch.cat([c.send[:m] for c in self.certs], 0)
        for c, on, o in zip(self.certs, ons, outs):
            r = c.recv[: self.world * m]
            r.copy_(recv)
            c.ext.select_tp(r, self.world, m, c.ht, c.V, on, c.flags, c.mask, c.stats, cm.CAP, cm.TWO_E, cm.DMAX)
            c.ext.exact(x, c.pack.buf, c.V, cm.K, c.mask, o, c.wk, c.flags)
        return outs


def gathered_token(shards, v_real, sc, am):
    y = torch.cat([s.clone() for s in shards], 1)
    sc(y, cm.CAP)
    y[:, v_real:] = float("-inf")
    return am(y)


def deferred_token(shards, sh, sc, row_argmax):
    loc = []
    for r, s in enumerate(shards):
        y = s.clone()
        sc(y, cm.CAP)
        loc.append(row_argmax(y, None, sh.v[r], r * sh.n))
    g = torch.stack(loc)                                              # [world, M, 2], the deferred all-gather
    best = g[..., 0].argmax(dim=0, keepdim=True)                      # LogitsProcessor.deferred_argmax (CUDA)
    return g[..., 1].gather(0, best).squeeze(0).long()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--hidden", default="/work/hid")
    ap.add_argument("--v-real", type=int, default=200058)
    ap.add_argument("--world", type=int, default=4)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=30)
    a = ap.parse_args()
    torch.manual_seed(0)
    dev = torch.device("cuda")
    print(f"device {torch.cuda.get_device_name()} torch {torch.__version__}", flush=True)
    pops.EXT = pbuild.load(os.environ.get("INKLING_PICO12_CACHE") or None)
    ext = cbuild.load(os.environ.get("INKLING_CERT_HEAD_CACHE") or None)
    from sglang.kernels.ops.sampling.softcap_row_argmax import softcap_row_argmax

    sc, am = stock_fns()
    W = load_head(a.model)
    t0 = time.time()
    sh = Shards(W, a.world, a.v_real, ext)
    torch.cuda.synchronize()
    print(f"{a.world} shards of [{sh.n}x{W.shape[1]}], real ids per shard {sh.v}; tables + pico {time.time() - t0:.1f}s",
          flush=True)
    X = load_hidden(a.hidden)
    if a.quick:
        X = X[:512]
    N = X.shape[0]
    print(f"{N} real hidden rows", flush=True)
    on1 = [c.force for c in sh.certs]

    def step(x, ons=None):
        ref = sh.full(x)
        outs = sh.run(x, on1 if ons is None else ons)
        t_ref = gathered_token(ref, a.v_real, sc, am)
        ok_g = torch.equal(gathered_token(outs, a.v_real, sc, am), t_ref)
        ok_d = torch.equal(deferred_token(outs, sh, sc, softcap_row_argmax), t_ref)
        ok_d0 = torch.equal(deferred_token(ref, sh, sc, softcap_row_argmax), t_ref)  # the stock deferral (limited)
        flags = [int(c.flags[2]) for c in sh.certs]
        bits = True
        for r, c in enumerate(sh.certs):
            msk = c.mask.bool().repeat_interleave(16)
            bits &= torch.equal(outs[r][:, msk].view(torch.int32), ref[r][:, msk].view(torch.int32))
            bits &= not bool(torch.isfinite(outs[r][:, ~msk]).any())
        return ok_g and ok_d and ok_d0, len(set(flags)) == 1, flags[0], bits, [int(c.mask.sum()) for c in sh.certs], \
            outs, ref

    print("1 fallbacks", flush=True)
    x = X[:8].contiguous()
    ref = sh.full(x)
    ok = True
    for r, c in enumerate(sh.certs):
        o = torch.empty_like(ref[r])
        c.run_tp(x, o, c.off, sh.v[r], cm.CAP, exchange="local")
        ok &= torch.equal(o.view(torch.int32), ref[r].view(torch.int32))
        o2 = torch.empty_like(ref[r])
        c.run_tp(x, o2, c.force, sh.v[r], cm.CAP, exchange="poison")
        ok &= torch.equal(o2.view(torch.int32), ref[r].view(torch.int32))
    gate(ok, "on = 0 and a peer's +inf: pico's shard bit for bit on every shard")

    print("2 exactness (real rows)", flush=True)
    for M in (8, 16):
        bad_tok = bad_flag = bad_bits = cert_steps = 0
        tiles_max = []
        for b0 in range(0, N - M + 1, M):
            okt, same, certified, bits, tiles, _, _ = step(X[b0:b0 + M].contiguous())
            bad_tok += not okt
            bad_flag += not same
            bad_bits += not bits
            cert_steps += certified
            if certified:
                tiles_max.append(max(tiles))
        tl = np.array(tiles_max) if tiles_max else np.zeros(1)
        gate(bad_tok == 0, f"M={M}: certified token == stock token (gather and deferred) on {N // M} steps "
                           f"({bad_tok} bad steps)")
        gate(bad_flag == 0, f"M={M}: the certified flag is the same on every shard ({bad_flag} bad steps)")
        gate(bad_bits == 0, f"M={M}: computed tiles bit-identical to pico, others -inf ({bad_bits} bad steps)")
        print(f"     M={M}: certified {cert_steps}/{N // M}; slowest shard tiles/step mean {tl.mean():.1f} p50 "
              f"{np.median(tl):.0f} p90 {np.percentile(tl, 90):.0f} p99 {np.percentile(tl, 99):.0f} max {tl.max():.0f}",
              flush=True)

    print("3 near ties across shards", flush=True)
    g = torch.Generator(device=dev).manual_seed(7)
    bad = rows = 0
    for scale in (60.0, 150.0, 230.0, 400.0, 800.0):
        for _ in range(8 if a.quick else 32):
            ra = torch.randint(0, a.world, (8,), generator=g, device=dev)
            rb = (ra + torch.randint(1, a.world, (8,), generator=g, device=dev)) % a.world
            ia = ra * sh.n + torch.randint(0, sh.n, (8,), generator=g, device=dev)
            ib = rb * sh.n + torch.randint(0, sh.n, (8,), generator=g, device=dev)
            ia, ib = ia.clamp_max(a.v_real - 1), ib.clamp_max(a.v_real - 1)
            w = W[ia].float() + W[ib].float()
            noise = torch.randn((8, W.shape[1]), generator=g, device=dev) * 0.02
            x = ((w / w.norm(dim=1, keepdim=True) + noise) * scale).to(torch.bfloat16).contiguous()
            okt, same, _, bits, _, _, _ = step(x)
            bad += not (okt and same and bits)
            rows += 8
    gate(bad == 0, f"near-tie rows across shards: {rows} rows, {bad} bad steps")

    print("4 exact ties across shards", flush=True)
    a_id, b_id = 12345, 2 * sh.n + 777
    W2 = W.clone()
    W2[b_id] = W2[a_id]
    sh2 = Shards(W2, a.world, a.v_real, ext)
    x = (W2[a_id].float()[None].repeat(8, 1) * torch.linspace(40.0, 70.0, 8, device=dev)[:, None]).to(torch.bfloat16)
    ref2 = sh2.full(x)
    outs2 = sh2.run(x, [c.force for c in sh2.certs])
    t_ref = gathered_token(ref2, a.v_real, sc, am)
    gate(bool((t_ref == a_id).all()) and torch.equal(gathered_token(outs2, a.v_real, sc, am), t_ref)
         and torch.equal(deferred_token(outs2, sh2, sc, softcap_row_argmax), t_ref),
         f"duplicate rows {a_id} (shard 0) = {b_id} (shard 2): token {a_id} by both resolutions")
    del sh2, W2

    print("5 one shard off", flush=True)
    x = X[8:16].contiguous()
    ons = list(on1)
    ons[2] = sh.certs[2].off
    okt, same, certified, _, _, outs, ref = step(x, ons)
    full = all(torch.equal(o.view(torch.int32), r_.view(torch.int32)) for o, r_ in zip(outs, ref))
    gate(okt and same and not certified and full, "rank 2 with on = 0: every shard serves pico's full shard")

    print("6 CUDA graph", flush=True)
    xs = torch.empty((8, W.shape[1]), dtype=torch.bfloat16, device=dev)
    xs.copy_(X[:8])
    for c in sh.certs:
        c.set_on(1)
    ons = [c.on for c in sh.certs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        sh.run(xs, ons)
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        og = sh.run(xs, ons)
    ok = True
    for i, b0 in enumerate(range(8, min(N, 8 + 8 * 40), 8)):
        xs.copy_(X[b0:b0 + 8])
        for c in sh.certs:
            c.set_on(i % 3 != 0)
        graph.replay()
        e = sh.run(xs, ons)
        ok &= all(torch.equal(p.view(torch.int32), q.view(torch.int32)) for p, q in zip(og, e))
    gate(ok, "graph replay == eager (40 steps, on toggled)")
    for c in sh.certs:
        c.set_on(0)

    print("7 bench (L2 flushed, CUDA events)", flush=True)
    flush = torch.empty((256 << 20,), dtype=torch.uint8, device=dev)
    sink = torch.zeros((1,), dtype=torch.int64, device=dev)

    def timeit(fn, reps):
        ts = []
        for _ in range(reps):
            sink.add_(flush.view(torch.int64).sum())
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            fn()
            e1.record()
            e1.synchronize()
            ts.append(e0.elapsed_time(e1))
        return float(np.median(ts))

    res = {}
    for M in (8, 16):
        counts = []
        for b0 in range(0, min(N, 64 * M), M):
            sh.run(X[b0:b0 + M].contiguous(), on1)
            tiles = [int(c.mask.sum()) for c in sh.certs]
            counts.append((max(tiles), int(np.argmax(tiles)), b0))
        counts.sort()
        res[M] = {}
        for name, (nt, r, b0) in (("median", counts[len(counts) // 2]), ("p90", counts[int(len(counts) * 0.9)])):
            x = X[b0:b0 + M].contiguous()
            c, p = sh.certs[r], sh.packs[r]
            out = torch.empty((M, sh.n), dtype=torch.float32, device=dev)
            t_pico = timeit(lambda: pops.gemv(x, p, out), a.reps)
            sh.run(x, on1)  # recv of this batch in place: select / exact below see the real candidate set

            def cert_path():
                c.screen_tp(x, c.force, sh.v[r])
                c.ext.select_tp(c.recv[: sh.world * M], sh.world, M, c.ht, c.V, c.force, c.flags, c.mask, c.stats,
                                cm.CAP, cm.TWO_E, cm.DMAX)
                c.ext.exact(x, c.pack.buf, c.V, cm.K, c.mask, out, c.wk, c.flags)

            t_cert = timeit(cert_path, a.reps)
            y = torch.randn((M, sh.n), device=dev)
            t_am = timeit(lambda: softcap_row_argmax(y, None, sh.v[r], r * sh.n), a.reps)
            res[M][name] = dict(tiles_slowest=nt, pico_shard_ms=t_pico, cert_shard_ms=t_cert, row_argmax_ms=t_am)
            print(f"     M={M} {name} batch (slowest shard {r}: {nt} tiles): pico shard {t_pico:.3f} ms | certified "
                  f"shard {t_cert:.3f} ms (+ the [M, 2] all-gather) | softcap_row_argmax {t_am * 1e3:.0f} us", flush=True)
    print("RESULT " + json.dumps(res), flush=True)
    print(f"certtp gpu_check result {'FAIL' if FAIL else 'PASS'}" + (f" ({len(FAIL)} failed)" if FAIL else ""),
          flush=True)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
