# Usage (inside the SGLang image, from the repository root):
#   cd overlay && PYTHONPATH=. python3 tools/gpu_check_fused_rsfix.py [--quick]   (in the image, server stopped)
"""GB10 check of the v2.7.3 race fix in inkling_vcap.fused_accept (VCAP_FUSED_ACCEPT):
Philox words folded before the FP32 conversion (SGLang PR #51681), a finite key floor
for lanes with positive mass, and the last positive-target lane when a row has no
positive residual. tools/fused_accept_v272.py (the v2.7.2 file, unchanged) is the stock
side. No hooks or server: TritonPrims is driven directly.

1. Philox endpoint: find (seed, flat offset) whose first word r0 >= 0xFFFFFF80 (FP32
   rounds it to 2^32), put a one-hot residual at that lane of that request at V=201,024;
   v2.7.2 emits 201,023 (a padded row), v2.7.3 must emit the lane.
2. No positive residual (P == Q exactly, bf16-exact logits) at V=201,024: v2.7.3 must
   emit the last positive-target lane (200,057); v2.7.2 emits 201,023.
3. Law at the real width (201,024 lanes, 200,058 valid, padded logits -inf): residual
   support of 7 lanes across split / chunk boundaries and the last valid id; >= 1e7
   draws for (s = 1, Q), plus (s = 0.6, Q) and (P only). Gate: 0 tokens off the support,
   chi-square Wilson-Hilferty z < 5 against the exact law. v2.7.2 off-support count
   reported on a smaller run.
Peak device memory < 1 GB (stride-0 batch views of one row).
"""
import argparse
import importlib.util
import math
import os
import sys
import time

import torch
import triton
import triton.language as tl

from inkling_vcap import fused_accept as fa

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("fused_accept_v272", os.path.join(HERE, "fused_accept_v272.py"))
fa_old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fa_old)

DEV = "cuda"
V, VALID = 201024, 200058
FAILS = []


def line(ok, name, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


@triton.jit
def _first_word_kernel(seed, out_ptr, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    r0, _r1, _r2, _r3 = tl.randint4x(seed, offs)  # the race kernel's call, counter = b * V + lane
    hit = (r0.to(tl.uint32) >= 0xFFFFFF80) & (offs < n)
    tl.store(out_ptr + offs, hit.to(tl.int8), mask=offs < n)


@triton.jit
def _words_kernel(seed, offset, out_ptr):
    o = offset + tl.zeros([1], tl.int32)
    r0, r1, _r2, _r3 = tl.randint4x(seed, o)
    tl.store(out_ptr + tl.arange(0, 1), r0.to(tl.int32, bitcast=True))
    tl.store(out_ptr + 1 + tl.arange(0, 1), r1.to(tl.int32, bitcast=True))


def words_at(seed, offset):
    out = torch.zeros(2, dtype=torch.int32, device=DEV)
    _words_kernel[(1,)](seed, offset, out)
    return [int(x) & 0xFFFFFFFF for x in out.tolist()]


def exact_exp_noise(r0, r1):
    """E = -log(1 - V), V = (r0 + (r1 + 0.5) 2^-32) 2^-32, in FP64 (1 - V formed exactly)."""
    one_minus = ((0xFFFFFFFF - r0) + (0xFFFFFFFF - r1 + 0.5) * 2.0 ** -32) * 2.0 ** -32
    return -math.log(one_minus)


def race_keys(mod, tgt, drf, bs, *, scale, use_q, seed):
    """Launch the race kernel directly; returns the per-split keys [bs, splits]."""
    tl2, dl2, st, tt = prims_inputs(tgt, drf, bs)
    mt, lt, md, ld = st
    num_splits, chunk = mod._splits(bs, V)
    keys = torch.empty(bs, num_splits, dtype=torch.float32, device=DEV)
    idx = torch.empty(bs, num_splits, dtype=torch.int32, device=DEV)
    last_p = torch.empty(bs, num_splits, dtype=torch.int32, device=DEV)
    mod._race_kernel[(bs, num_splits)](
        tl2, dl2, mt, lt, md, ld, tt, tt, torch.zeros(bs, dtype=torch.int32, device=DEV),
        torch.full((bs,), scale, device=DEV), torch.full((bs,), int(use_q), dtype=torch.int32, device=DEV),
        seed, keys, idx, last_p, V, chunk, 1, 1, tl2.stride(0), dl2.stride(0), num_splits, BLOCK=mod.BLOCK_V)
    return keys


def find_endpoint(max_rows=64, seeds=256):
    n = max_rows * V
    out = torch.empty(n, dtype=torch.int8, device=DEV)
    for seed in range(1, seeds + 1):
        _first_word_kernel[(triton.cdiv(n, 4096),)](seed, out, n, BLOCK=4096)
        hits = torch.nonzero(out).squeeze(1).tolist()
        for o in hits:
            if o % V < VALID:
                return seed, o // V, o % V
    return None


def one_row(entries, dtype, logits=False):
    """[1, V] logits: log(p) of the given probabilities (or the given logits), -inf elsewhere."""
    r = torch.full((1, V), -math.inf, dtype=torch.float32, device=DEV)
    for k, p in entries.items():
        r[0, k] = p if logits else math.log(p)
    return r.to(dtype)


def prims_inputs(tgt, drf, bs):
    """Stats of one target / draft row, broadcast (stride 0) to bs requests, S = G = 1."""
    ones = torch.ones(1, device=DEV)
    rp = fa.TorchPrims()
    mt, lt = rp.stats(tgt, ones, 1, None)
    md, ld = rp.stats(drf, ones, 1, None)
    st = tuple(x.view(1, 1).expand(bs, 1).contiguous() for x in (mt, lt, md, ld))
    return tgt.expand(bs, V), drf.expand(bs, V), st, torch.ones(bs, device=DEV)


def sample(mod, tgt, drf, bs, *, scale, use_q, seed=None):
    tl2, dl2, st, tt = prims_inputs(tgt, drf, bs)
    tp = mod.TritonPrims()
    if seed is not None:
        tp._seed = lambda: seed
    return tp.sample(tl2, dl2, st, tt, tt, torch.zeros(bs, dtype=torch.long, device=DEV),
                     torch.full((bs,), scale, device=DEV), torch.full((bs,), use_q, dtype=torch.bool, device=DEV))


def check_endpoint():
    found = find_endpoint()
    if found is None:
        line(False, "1.endpoint", "no endpoint word found in 256 seeds x 64 rows")
        return
    seed, b, lane = found
    r0, r1 = words_at(seed, b * V + lane)
    other = 7 if lane != 7 else 9
    # (i) one-hot residual at the endpoint lane: v2.7.2 must emit V - 1 (negative control).
    tgt = one_row({lane: 0.6, other: 0.4}, torch.float32)
    drf = one_row({other: 1.0}, torch.bfloat16)  # residual = 0.6 at `lane` only
    new = int(sample(fa, tgt, drf, b + 1, scale=1.0, use_q=True, seed=seed)[b])
    old = int(sample(fa_old, tgt, drf, b + 1, scale=1.0, use_q=True, seed=seed)[b])
    line(new == lane and old == V - 1, "1.endpoint.one-hot",
         f"seed {seed}, request {b}, lane {lane}, r0 {r0:#010x}: v2.7.3 {new} (want {lane}), "
         f"v2.7.2 {old} (want {V - 1})")
    # (ii) the kernel's key at that lane equals log(w) - log(E) of the two words (catches a
    # transform that is wrong but rescued by the key floor or the fallback).
    w_l = float(torch.softmax(tgt.double(), -1)[0, lane] - torch.softmax(drf.double(), -1)[0, lane])
    want = math.log(w_l) - math.log(exact_exp_noise(r0, r1))
    got = float(race_keys(fa, tgt, drf, b + 1, scale=1.0, use_q=True, seed=seed)[b].max())
    line(abs(got - want) < 1e-5, "1.endpoint.key",
         f"kernel key {got:.5f}, log(w) - log(E) from the words {want:.5f} (E {exact_exp_noise(r0, r1):.4f})")
    # (iii) a second, tiny positive lane: a transform that sends the endpoint lane to the
    # floor (or to -inf) lets it win; the exact transform keeps the endpoint lane ahead.
    tiny = 3 if lane > 3 else 11
    tgt2 = one_row({lane: 0.6, tiny: 1e-30, other: 0.4 - 1e-30}, torch.float32)
    new2 = int(sample(fa, tgt2, drf, b + 1, scale=1.0, use_q=True, seed=seed)[b])
    old2 = int(sample(fa_old, tgt2, drf, b + 1, scale=1.0, use_q=True, seed=seed)[b])
    line(new2 == lane and old2 == tiny, "1.endpoint.two-lane",
         f"v2.7.3 {new2} (want {lane}), v2.7.2 {old2} (want the tiny lane {tiny})")


def check_no_residual():
    ent = {1000: 0.0, 150000: -1.0, 200057: -0.5}  # bf16-exact logits, so P == Q bit for bit
    tgt = one_row(ent, torch.float32, logits=True)
    drf = one_row(ent, torch.bfloat16, logits=True)
    bs = 256
    new = sample(fa, tgt, drf, bs, scale=1.0, use_q=True)
    old = sample(fa_old, tgt, drf, bs, scale=1.0, use_q=True)
    line(bool((new == 200057).all()), "2.no-positive-residual",
         f"v2.7.3 picks {sorted(set(new.tolist()))} (want 200057), v2.7.2 {sorted(set(old.tolist()))}")


def chi2_z(obs, exp):
    big = exp >= 5
    o = torch.cat([obs[big], obs[~big].sum()[None]])
    e = torch.cat([exp[big], exp[~big].sum()[None]])
    keep = e > 0
    o, e = o[keep], e[keep]
    df = max(1, o.numel() - 1)
    chi2 = float(((o - e) ** 2 / e).sum())
    return ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))


def check_law(draws, old_draws):
    lanes = [5, 2047, 2048, 70001, 140000, 199999, 200057]
    pt = [0.22, 0.18, 0.15, 0.15, 0.12, 0.10, 0.08]
    qd = [0.10, 0.30, 0.05, 0.20, 0.15, 0.05, 0.15]
    tgt = one_row(dict(zip(lanes, pt)), torch.float32)
    drf = one_row(dict(zip(lanes, qd)), torch.bfloat16)
    bs = (2**31 - 1) // V  # the kernel's bs * V < 2^31 bound
    Pd = torch.softmax(tgt.double(), -1)[0]  # FP64 oracle on the stored FP32 / BF16 logits
    Qd = torch.softmax(drf.double(), -1)[0]
    arms = (("s1-q", 1.0, True, draws, bs), ("s0.6-q", 0.6, True, max(1, draws // 5), bs),
            ("p-only", 1.0, False, max(1, draws // 5), bs),
            # 128 requests -> 8 vocabulary splits: competing keys combined across splits
            ("s1-q-multisplit", 1.0, True, max(1, draws // 50), 128))
    for name, scale, use_q, n, bs in arms:
        w = (scale * Pd - (Qd if use_q else 0.0)).clamp(min=0)
        law = w / w.sum()
        support = w > 0
        for mod, tag, total in ((fa, "v2.7.3", n), (fa_old, "v2.7.2", old_draws if name == "s1-q" else 0)):
            if total <= 0:
                continue
            counts = torch.zeros(V + 1, dtype=torch.int64, device=DEV)
            done, t0 = 0, time.time()
            while done < total:
                m = min(bs, total - done)
                pick = sample(mod, tgt, drf, m, scale=scale, use_q=use_q)
                counts += torch.bincount(pick.clamp(0, V), minlength=V + 1)
                done += m
            off = int(counts[:V][~support].sum() + counts[V])
            if mod is fa:
                exp = law[support] * done
                z = chi2_z(counts[:V][support].double(), exp)
                enough = float(exp.min()) >= 50  # every support cell is its own chi-square cell
                line(off == 0 and z < 5.0 and enough, f"3.law.{name}",
                     f"{done} draws ({time.time() - t0:.0f} s): off-support {off}, chi-square z {z:.2f}, "
                     f"min expected count {float(exp.min()):.0f} (need >= 50), "
                     f"support {torch.nonzero(support).squeeze(1).tolist()}")
            else:
                print(f"INFO 3.law.{name} {tag}: {done} draws, off-support {off}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--draws", type=float, default=1e7)
    ap.add_argument("--old-draws", type=float, default=2e6)
    a = ap.parse_args()
    draws, old = (int(5e5), int(1e5)) if a.quick else (int(a.draws), int(a.old_draws))
    torch.manual_seed(0)
    print(f"INFO {torch.cuda.get_device_name()} torch {torch.__version__} triton {triton.__version__}", flush=True)
    check_endpoint()
    check_no_residual()
    check_law(draws, old)
    print(f"INFO peak device memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
    print(f"{'FAIL' if FAILS else 'PASS'} gpu_check_fused_rsfix: {len(FAILS)} failed {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
