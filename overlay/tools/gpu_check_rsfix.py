#!/usr/bin/env python3
"""GPU check for inkling_rsfix. Runs inside the pinned SGLang image on one GPU, no server:

    cd overlay && PYTHONPATH=. python3 tools/gpu_check_rsfix.py [--quick]   (in the image, server stopped)

Each section drives the real stock entry points with the stock and the fixed objects
switched on the imported modules (inkling_rsfix.apply / use), on identical inputs and
coins, and prints PASS / FAIL lines; the exit code is 1 when any FAIL was printed.

(a) crafted, 8 slots (gamma 7): chain_speculative_sampling_triton at the real vocab
    (201,024 lanes, 200,058 valid, padded p = 0) on the classic chain, split chain and
    classic block-verification paths and through the stock gate at bs = SMs - 1 (split)
    and SMs (classic), plus V = 24 (lane 23 = the stock fallback has p = 0). Dyadic rows
    with coin_final = 1.0 make target_u equal the exact scan end, i.e. the rounding case:
    the stock token is VOCAB - 1 (a zero-mass row); the fixed token must be the last
    positive-residual lane (all accepted / n = 1 / n = 0) or the last positive-target
    lane (degenerate residual); accept lengths must equal stock. Then non-dyadic rows and
    coin_final just below 1 (rounding for real), counting invalid tokens.
(b) Monte Carlo law at the real vocab: >= 1e7 residual / bonus draws per path (classic
    chain, classic block verification; the split chain with --split-draws), rows shared
    by every request of a launch (stride-0 batch views, so device memory stays small),
    8 distinct row sets with "dust" mass across the vocabulary. Fixed: 0 invalid tokens
    and the (accept length, token) joint law matches the exact law (chi-square, Wilson-
    Hilferty z < 5). Stock: invalid count reported.
(c) V = 24 Markov chains (target / draft transition matrices with zeros), gamma 7, chain
    and block verification: invalid bonus tokens (>= 24 or zero residual mass), stock vs
    fixed.
(d) CUDA graphs: the fixed chain call captured on a warmed stream (split kernels) and on a
    cold stream (classic kernel), replayed with new coins and a poisoned scratch, equals
    eager calls; crafted no-match rows give the known token on every replay.
(e) PR regressions: #51681 (Markov draft sampler, Philox endpoint; CUDA and CPU) and
    #51692 (blockwise rejection: equal laws near a coin of 1; shift invariance).
(f) timing of stock vs fixed final-token kernels (INFO only).

Support checks (support()) take the residual's sign with and without FMA contraction;
only block-verification lanes where the two disagree are "ambiguous".

Cost: peak device memory about 2.3 GB (8-slot real-vocab rows built in FP64 at bs 64);
run time is dominated by (b): ~2 x 153 launches of 65,536 requests per classic path and
~2 x 42,600 launches of 47 requests for the default 2e6 split draws (several minutes).
"""

import argparse
import contextlib
import importlib
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import inkling_rsfix as rf  # noqa: E402

DEV = torch.device("cuda")
F32 = torch.float32
VOCAB = 201024
VALID = 200058
FAILS = []


def line(ok, name, detail):
    print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)
    if not ok:
        FAILS.append(name)


def info(msg):
    print(f"INFO {msg}", flush=True)


rs = importlib.import_module("sglang.kernels.ops.speculative.reject_sampling")
T_REJ = rf.apply("reject", rs)


@contextlib.contextmanager
def variant(mod, table, which):
    rf.use(mod, table[which])
    try:
        yield
    finally:
        rf.use(mod, table["fixed"])


@contextlib.contextmanager
def split_gate(force):
    """force None: the stock gate; True / False: force the split / classic path."""
    stock = rs._should_split_final_sampling
    if force is not None:
        rs._should_split_final_sampling = lambda *a, **k: force
    try:
        yield
    finally:
        rs._should_split_final_sampling = stock


def accept(P, Q, cand, coins, coins_final, *, bv):
    """chain_speculative_sampling_triton; returns (n [bs] long, bonus [bs] long, predicts,
    accept_index)."""
    bs, S = cand.shape
    predicts = torch.full((bs * S,), -7, dtype=torch.int32, device=DEV)
    acc_idx = torch.full((bs, S), -1, dtype=torch.int32, device=DEV)
    acc_num = torch.full((bs,), -1, dtype=torch.int32, device=DEV)
    ridx = torch.arange(bs * S, dtype=torch.int64, device=DEV).view(bs, S)
    rs.chain_speculative_sampling_triton(
        predicts, acc_idx, acc_num, cand, ridx, ridx, ridx, coins, coins_final, P, Q,
        1.0, 1.0, True, block_verification=bv,
    )
    n = acc_num.long()
    bonus = predicts[acc_idx.gather(1, n[:, None]).squeeze(1).long()].long()
    return n, bonus, predicts, acc_idx


def run(P, Q, cand, coins, coins_final, *, bv, path, which):
    force = {"classic": False, "split": True, "gate": None}[path]
    with split_gate(force), variant(rs, T_REJ, which):
        return accept(P, Q, cand, coins, coins_final, bv=bv)


def prefix_probs(P, Q, cand):
    """[bs, S] prefix probabilities pp_0 = 1, pp_j = min(pp_{j-1} p_j / q_j, 1) (0 when q is
    not a probability), FP32 in the kernel's order. P / Q may be stride-0 views."""
    bs, S = cand.shape
    rows = torch.arange(bs, device=DEV)
    pp = [torch.ones(bs, dtype=F32, device=DEV)]
    for j in range(1, S):
        d = cand[:, j].long()
        p = P[rows, j - 1, d]
        q = Q[rows, j - 1, d]
        ok = (q > 0) & (q <= 1)
        pp.append(torch.where(ok, torch.minimum(pp[-1] * p / q, torch.ones_like(p)), torch.zeros_like(p)))
    return torch.stack(pp, 1)


def support(P, Q, cand, n, bonus, *, bv, valid_vocab):
    """(invalid [bs], ambiguous [bs]) for the emitted bonus tokens.

    The residual lane value is ``scale * p - q`` (scale 1 on the chain path, the FP32
    prefix probability under block verification), and the kernel may or may not contract
    it to an FMA. Its sign is taken both ways: ``fl32(scale * p) - q`` (an FP32 subtraction
    has the sign of the exact difference) and ``scale * p - q`` in FP64 (exact product).
    A lane is strictly positive when both are > 0, possibly positive when either is.
    ``invalid``: out of range, zero target mass, or (when some lane is strictly positive)
    not even possibly positive. ``ambiguous``: possibly but not strictly positive (only
    under block verification; excluded from stock-vs-fixed comparisons).
    The all-accepted row is p itself. P / Q may be stride-0 batch views."""
    bs, S = cand.shape
    rows = torch.arange(bs, device=DEV)
    P32 = P[rows, n]
    last = (n == S - 1)[:, None]
    Q32 = Q[rows, n.clamp(max=S - 2)]
    Q32 = torch.where((Q32 >= 0) & (Q32 <= 1), Q32, torch.zeros_like(Q32))
    scale = torch.ones(bs, dtype=F32, device=DEV)
    if bv:
        scale = torch.where(n > 0, prefix_probs(P, Q, cand)[rows, n], scale)
    nofma = (scale[:, None] * P32).double() - Q32.double() > 0
    fma = scale.double()[:, None] * P32.double() - Q32.double() > 0
    strict = torch.where(last, P32 > 0, nofma & fma)
    loose = torch.where(last, P32 > 0, nofma | fma)
    has_strict = strict.any(dim=1)
    has_loose = loose.any(dim=1)
    lanes = torch.arange(P32.shape[1], device=DEV)
    last_p = torch.where(P32 > 0, lanes, torch.full_like(lanes, -1)).amax(dim=1)
    inr = (bonus >= 0) & (bonus < valid_vocab)
    b = bonus.clamp(0, P32.shape[1] - 1)
    p_ok = P32[rows, b] > 0
    # No strictly positive lane: in some arithmetic mode the residual may be empty, and the
    # kernel then emits the last positive-target lane; possibly positive lanes stay legal.
    res_ok = torch.where(has_strict, loose[rows, b], loose[rows, b] | (bonus == last_p))
    ok = inr & p_ok & res_ok
    ambiguous = inr & ((loose[rows, b] & ~strict[rows, b]) | (~has_strict & has_loose))
    return ~ok, ambiguous


def compare(res_stock, res_fixed, fallback_token):
    """Count of rows where the accept length differs, or the fixed token differs from a
    stock token that was legal, unambiguous and not the stock no-match token."""
    (n_s, b_s, inv_s, amb_s), (n_f, b_f, _, amb_f) = res_stock, res_fixed
    differ = (n_s != n_f).sum()
    differ += ((b_s != b_f) & ~inv_s & ~amb_s & ~amb_f & (b_s != fallback_token)).sum()
    return differ


# --------------------------------------------------------------------------- (a) crafted
def _row(entries, V):
    r = torch.zeros(V, dtype=F32, device=DEV)
    for k, v in entries.items():
        r[k] = v
    return r


def crafted(V, lanes, S, name):
    """Dyadic rows (every sum exact) for one case; ``lanes`` = (a, b, c, d, e) ids.
    Rows 0..S-2: P = {a, b, c, d: 1/4}; Q = {a: 1/2, b: 1/4, d: 1/4}, so the chain residual
    of a rejected row is lane c only; the last target row is {a: 1/2, e: 1/2}. Drafts: a,
    then b. coin_final 1.0 sets target_u to the exact scan end (no lane matches)."""
    a, b, c, d, e = lanes
    P0 = {a: 0.25, b: 0.25, c: 0.25, d: 0.25}
    Q0 = {a: 0.5, b: 0.25, d: 0.25}
    PL = {a: 0.5, e: 0.5}
    # name: (coin of step 1, coin of later steps, Q of row 0, n, token at coin_final 1 / 0)
    case = {
        "all-accepted": (0.0, 0.0, Q0, S - 1, e, a),
        "n=1": (0.0, 1.0, Q0, 1, c, c),
        "n=0": (1.0, 1.0, Q0, 0, c, c),
        # q == p on row 0: zero residual, the last positive-target lane at any coin
        "degenerate": (1.0, 1.0, P0, 0, d, d),
    }[name]
    c1, c2, q0, n_exp, tok1, tok0 = case
    P = torch.stack([_row(P0, V)] * (S - 1) + [_row(PL, V)])
    Q = torch.stack([_row(q0, V)] + [_row(Q0, V)] * (S - 2))
    cand = torch.tensor([0, a] + [b] * (S - 2), dtype=torch.int64, device=DEV)
    coins = torch.tensor([c1] + [c2] * (S - 2), dtype=F32, device=DEV)
    return P, Q, cand, coins, n_exp, tok1, tok0


CASES = ("all-accepted", "n=1", "n=0", "degenerate")


def section_a(args):
    info("(a) crafted inputs, 8 slots: V=201024 (200058 valid) and V=24")
    sm = torch.cuda.get_device_properties(DEV).multi_processor_count
    S = 8
    tiles = -(-VOCAB // 4096)
    real = (5, 4097, 100000, 200057, 150000)
    shapes = [
        # (tag, V, valid, lanes, path, bv, bs, split kernels expected: set independently of
        # the gate under test: split below one request per SM, chain only, >= 2 tiles)
        ("v201024.classic-chain", VOCAB, VALID, real, "classic", False, 4, False),
        ("v201024.split-chain", VOCAB, VALID, real, "split", False, 4, True),
        ("v201024.classic-bv", VOCAB, VALID, real, "classic", True, 4, False),
        (f"v201024.gate-chain-bs{sm - 1}", VOCAB, VALID, real, "gate", False, sm - 1, True),
        (f"v201024.gate-chain-bs{sm}", VOCAB, VALID, real, "gate", False, sm, False),
        (f"v201024.gate-bv-bs{sm - 1}", VOCAB, VALID, real, "gate", True, sm - 1, False),
        # V = 24 (one tile: never split): lane 23 (the stock fallback V - 1) has p = 0.
        ("v24.chain", 24, 24, (0, 3, 10, 22, 15), "gate", False, 4, False),
        ("v24.bv", 24, 24, (0, 3, 10, 22, 15), "gate", True, 4, False),
    ]
    info(f"stock gate: split at bs={sm - 1}: {rs._should_split_final_sampling(sm - 1, tiles, DEV)}, "
         f"at bs={sm}: {rs._should_split_final_sampling(sm, tiles, DEV)}")
    for tag, V, valid, lanes, path, bv, bs, want_split in shapes:
        for name in CASES:
            P1, Q1, c1, coins1, n_exp, tok1, tok0 = crafted(V, lanes, S, name)
            P, Q = P1[None].expand(bs, -1, -1), Q1[None].expand(bs, -1, -1)
            cand = c1[None].expand(bs, -1).contiguous()
            coins = coins1[None].expand(bs, -1).contiguous()
            out = {}
            tiles_v = -(-V // 4096)
            split_ran = None
            for which in ("stock", "fixed"):
                for cf in (1.0, 0.0):
                    cfin = torch.full((bs,), cf, dtype=F32, device=DEV)
                    probe = which == "fixed" and cf == 1.0
                    if probe:
                        key = (torch.cuda.current_device(), torch.cuda.current_stream().cuda_stream, tiles_v)
                        sc = rf_reject().get(key)
                        if sc is not None:
                            sc[1].fill_(-12345)
                    n, b, _, _ = run(P, Q, cand, coins, cfin, bv=bv, path=path, which=which)
                    if probe:
                        sc_after = rf_reject().get(key)
                        split_ran = sc_after is not None and (sc is None or int(sc_after[1][0]) != -12345)
                    inv, _ = support(P, Q, cand, n, b, bv=bv, valid_vocab=valid)
                    out[which, cf] = (n.tolist(), b.tolist(), int(inv.sum()))
            n_f, b_f, inv_f = out["fixed", 1.0]
            ok = (n_f == [n_exp] * bs and b_f == [tok1] * bs and out["fixed", 0.0][1] == [tok0] * bs
                  and inv_f == 0 and out["fixed", 0.0][2] == 0)
            if name != "degenerate":  # a coin of 0 picks the first positive lane in both rules
                ok &= out["stock", 0.0][1] == [tok0] * bs
            ok &= out["stock", 1.0][0] == n_f  # accept lengths unchanged
            s_tok, s_inv = out["stock", 1.0][1][0], out["stock", 1.0][2]
            ok &= out["stock", 1.0][1] == [V - 1] * bs and s_inv == bs  # negative control: stock bug
            ok &= split_ran == want_split  # the launcher took the expected kernel
            line(ok, f"a.{tag}.{name}",
                 f"fixed n={n_f[0]} tok={b_f[0]} (want n={n_exp} tok={tok1}; {tok0} at coin_final 0); "
                 f"stock tok={s_tok} at coin_final 1.0 ({'invalid' if s_inv else 'valid'}, want {V - 1} invalid); "
                 f"split kernels ran={split_ran} (want {want_split})")

    # Rounding for real: non-dyadic rows with mass on ~3000 lanes, coin_final just below 1.
    g = torch.Generator(device=DEV).manual_seed(1)
    batches = 3 if args.quick else 12
    for path, bv, bs in (("classic", False, 64), ("split", False, min(40, sm - 1)), ("classic", True, 64)):
        tag = f"{path}-{'bv' if bv else 'chain'}"
        bad = {"stock": 0, "fixed": 0}
        differ = torch.zeros((), dtype=torch.int64, device=DEV)
        amb = 0
        rows = 0
        for _ in range(batches):
            P, Q, cand = random_rows(bs, S, g)
            coins = torch.rand((bs, S - 1), generator=g, device=DEV)
            ks = torch.randint(1, 5, (bs,), generator=g, device=DEV).float()
            cfin = (1.0 - ks * 2.0 ** -24).to(F32)
            res = {}
            for which in ("stock", "fixed"):
                n, b, _, _ = run(P, Q, cand, coins, cfin, bv=bv, path=path, which=which)
                inv, am = support(P, Q, cand, n, b, bv=bv, valid_vocab=VALID)
                bad[which] += int(inv.sum())
                amb += int(am.sum())
                res[which] = (n, b, inv, am)
            differ += compare(res["stock"], res["fixed"], VOCAB - 1)
            rows += bs
            del P, Q
        line(bad["fixed"] == 0 and int(differ) == 0, f"a.{tag}.rounding",
             f"{rows} rows x {S} slots, coin_final = 1 - k*2^-24 (k=1..4): invalid stock {bad['stock']} "
             f"fixed {bad['fixed']}; fixed != stock where stock was legal: {int(differ)} "
             f"(FMA-ambiguous picks {amb})")


def random_rows(bs, S, g, V=VOCAB, valid=VALID, dust=3000, main=12):
    """Per-request P [bs, S, V], Q [bs, S-1, V] with `main` large lanes (incl. tile
    boundaries and the last valid id) and `dust` small lanes, non-dyadic, normalised in
    FP64 then cast; padded ids valid..V-1 are 0. Draft tokens are main lanes."""
    fixed_lanes = torch.tensor([4095, 4096, 196607, 196608, valid - 1], device=DEV)
    P = torch.zeros((bs, S, V), dtype=torch.float64, device=DEV)
    Q = torch.zeros((bs, S - 1, V), dtype=torch.float64, device=DEV)
    lanes = torch.cat([fixed_lanes.expand(bs, -1),
                       torch.randint(0, valid, (bs, main - fixed_lanes.numel()), generator=g, device=DEV)], 1)
    for t, R in ((S, P), (S - 1, Q)):
        for j in range(t):
            w = torch.rand((bs, main), generator=g, device=DEV, dtype=torch.float64) + 0.05
            R[:, j].scatter_(1, lanes, w)
            dl = torch.randint(0, valid, (bs, dust), generator=g, device=DEV)
            dv = torch.rand((bs, dust), generator=g, device=DEV, dtype=torch.float64) * 1e-5
            R[:, j].scatter_add_(1, dl, dv)
            R[:, j] /= R[:, j].sum(1, keepdim=True)
    pick = torch.randint(0, main, (bs, S - 1), generator=g, device=DEV)
    cand = torch.cat([torch.zeros((bs, 1), dtype=torch.int64, device=DEV), lanes.gather(1, pick)], 1)
    return P.to(F32), Q.to(F32), cand


# ------------------------------------------------------------------------------ (b) law
def exact_law(P, Q, cand, *, bv):
    """{n: (prob of n, residual distribution over V as float64)} for one row set (P [S, V],
    Q [S-1, V], cand [S]) under the kernel's chain / block-verification rules."""
    S = cand.numel()
    Pd, Qd = P.double(), Q.double()
    Qg = torch.where((Qd >= 0) & (Qd <= 1), Qd, torch.zeros_like(Qd))
    d = cand.long()
    ratio = []
    for j in range(1, S):
        p, q = Pd[j - 1, d[j]], Qd[j - 1, d[j]]
        ratio.append(float(p / q) if 0 < float(q) <= 1 else None)
    law = {}
    if not bv:
        a = [min(1.0, r) if r is not None else 0.0 for r in ratio]
        reach = 1.0
        for n in range(S):
            stop = (1.0 - a[n]) if n < S - 1 else 1.0
            law[n] = reach * stop
            reach *= a[n] if n < S - 1 else 0.0
        scales = [1.0] * S
    else:
        pp = [1.0]
        for r in ratio:
            pp.append(min(pp[-1] * r, 1.0) if r is not None else 0.0)
        h = [None] * S
        for i in range(1, S):
            if i < S - 1 and 0.0 < pp[i] < 1.0:
                r_i = float((pp[i] * Pd[i] - Qg[i]).clamp(min=0).sum())
                h[i] = r_i / (r_i + 1.0 - pp[i])
            else:
                h[i] = pp[i]
        # backward walk: step S-1 first; each step accepts with its own coin
        rest = 1.0
        for i in range(S - 1, 0, -1):
            law[i] = rest * h[i]
            rest *= 1.0 - h[i]
        law[0] = rest
        scales = pp
    # Lanes the kernel can emit: scale * p - q > 0 with the kernel's FP32 scale, taking
    # the sign both with and without FMA contraction (see support()).
    if bv:
        pp32 = prefix_probs(P[None], Q[None], cand[None])[0]
    else:
        pp32 = torch.ones(S, dtype=F32, device=P.device)
    out = {}
    for n in range(S):
        if n == S - 1:
            w = Pd[n].clone()
            legal = w > 0
            strict = legal
        else:
            w = (scales[n] * Pd[n] - Qg[n]).clamp(min=0)
            q32 = Qg[n]
            pos_p = torch.nonzero(Pd[n] > 0).squeeze(1)
            fallback = torch.zeros_like(Pd[n], dtype=torch.bool)
            if pos_p.numel():
                fallback[pos_p[-1]] = True
            # Outcomes per arithmetic mode (FP32 product / FMA): its positive lanes, or its
            # fallback lane when its residual is empty. Legal = union, strict = intersection.
            modes = []
            for pos in ((pp32[n] * P[n]).double() - q32 > 0, pp32[n].double() * Pd[n] - q32 > 0):
                modes.append(pos if bool(pos.any()) else fallback)
            legal = modes[0] | modes[1]
            strict = modes[0] & modes[1]
        if float(w.sum()) > 0:
            out[n] = (law[n], w / w.sum(), legal, strict)
        else:  # degenerate residual: the fixed rule emits the last positive-target lane
            pos = torch.nonzero(Pd[n] > 0).squeeze(1)
            w = torch.zeros_like(w)
            if pos.numel():
                w[pos[-1]] = 1.0
            out[n] = (law[n], w, legal, strict)
    return out


def chi2_z(obs, exp):
    """(chi2, df, Wilson-Hilferty z, observations in zero-expectation cells). Cells with
    expected >= 5 are their own; cells with 0 < expected < 5 are pooled; cells with
    expected 0 are not pooled (their observations are returned separately)."""
    zero = exp <= 0
    impossible = int(obs[zero].sum())
    big = exp >= 5
    small = ~big & ~zero
    o = torch.cat([obs[big], obs[small].sum()[None]])
    e = torch.cat([exp[big], exp[small].sum()[None]])
    keep = e > 0
    o, e = o[keep], e[keep]
    df = max(1, o.numel() - 1)
    chi2 = float(((o - e) ** 2 / e).sum())
    z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return chi2, df, z, impossible


def section_b(args):
    info("(b) Monte Carlo law at V=201024, 200058 valid (padded p = 0)")
    sm = torch.cuda.get_device_properties(DEV).multi_processor_count
    S = 3
    sets = 8
    g = torch.Generator(device=DEV).manual_seed(2)
    rowsets = []
    for _ in range(sets):
        P, Q, cand = random_rows(1, S, g, dust=2000)
        rowsets.append((P, Q, cand))
    plans = [("classic", False, 65536, args.draws), ("classic", True, 65536, args.draws),
             ("split", False, sm - 1, args.split_draws)]
    for path, bv, bs, draws in plans:
        tag = f"{path}-{'bv' if bv else 'chain'}"
        per_set = -(-draws // sets)
        tot = {"stock": 0, "fixed": 0}
        zmax = {"stock": -1e9, "fixed": -1e9}
        imp = {"stock": 0, "fixed": 0}
        differ = torch.zeros((), dtype=torch.int64, device=DEV)
        t0 = time.time()
        for si, (P1, Q1, c1) in enumerate(rowsets):
            law = exact_law(P1[0], Q1[0], c1[0], bv=bv)
            # cells: per accept length n, every lane with positive law mass is its own cell
            # (padded / zero-mass / out-of-range lanes go to one invalid cell per n).
            pos = [torch.nonzero(law[n][2]).squeeze(1) for n in range(S)]
            C = max(int(x.numel()) for x in pos) + 1
            table = torch.full((S, VOCAB + 1), C - 1, dtype=torch.int64, device=DEV)
            strict_t = torch.zeros((S, VOCAB + 1), dtype=torch.bool, device=DEV)
            for n in range(S):
                strict_t[n, :VOCAB] = law[n][3]
            expv = torch.zeros(S * C, dtype=torch.float64, device=DEV)
            for n in range(S):
                table[n, pos[n]] = torch.arange(pos[n].numel(), device=DEV)
                expv[n * C: n * C + pos[n].numel()] = law[n][0] * law[n][1][pos[n]]
            P = P1.expand(bs, -1, -1)
            Q = Q1.expand(bs, -1, -1)
            cand = c1.expand(bs, -1).contiguous()
            counts = {w: torch.zeros(S * C, dtype=torch.int64, device=DEV) for w in ("stock", "fixed")}
            done = 0
            while done < per_set:
                m = min(bs, per_set - done)
                coins = torch.rand((m, S - 1), generator=g, device=DEV)
                cfin = torch.rand((m,), generator=g, device=DEV)
                res = {}
                for which in ("stock", "fixed"):
                    n, b, _, _ = run(P[:m], Q[:m], cand[:m], coins, cfin, bv=bv, path=path, which=which)
                    lane_cell = torch.where((b < 0) | (b >= VOCAB), C - 1, table[n, b.clamp(0, VOCAB)])
                    counts[which] += torch.bincount(n * C + lane_cell, minlength=S * C)
                    amb = ~strict_t[n, b.clamp(0, VOCAB)] & (lane_cell != C - 1)
                    res[which] = (n, b, lane_cell == C - 1, amb)
                differ += ((res["stock"][1] != res["fixed"][1]) & ~res["stock"][2] & ~res["stock"][3]
                           & ~res["fixed"][3]).sum()
                differ += (res["stock"][0] != res["fixed"][0]).sum()
                done += m
            for which in ("stock", "fixed"):
                cnt = counts[which].double()
                invalid = int(sum(cnt[n * C + C - 1] for n in range(S)))
                tot[which] += invalid
                _, _, z, im = chi2_z(cnt, expv * per_set)
                zmax[which] = max(zmax[which], z)
                imp[which] += im - invalid
        dt = time.time() - t0
        differ = int(differ)
        line(tot["fixed"] == 0 and zmax["fixed"] < 5.0 and differ == 0, f"b.{tag}.law",
             f"{per_set * sets} draws/variant ({sets} row sets, {dt:.0f} s): invalid stock {tot['stock']} "
             f"fixed {tot['fixed']}; max chi-square z stock {zmax['stock']:.2f} fixed {zmax['fixed']:.2f}; "
             f"draws in legal zero-expectation cells stock {imp['stock']} fixed {imp['fixed']}; "
             f"fixed != stock where stock was legal (or accept differs): {differ}")


# --------------------------------------------------------------------- (c) V = 24 Markov
def section_c(args):
    info("(c) V=24 Markov chains, gamma 7")
    V, S = 24, 8
    g = torch.Generator(device=DEV).manual_seed(3)
    rows_total = 400_000 if args.quick else args.markov_rows
    bs = 200_000

    def chain_matrix(temp):
        logits = torch.randn((V, V), generator=g, device=DEV, dtype=torch.float64) * 2.0
        keep = torch.rand((V, V), generator=g, device=DEV) > 0.3
        keep[torch.arange(V), torch.randint(0, V, (V,), generator=g, device=DEV)] = True
        w = torch.where(keep, torch.exp(logits / temp), torch.zeros_like(logits))
        return w / w.sum(1, keepdim=True)

    for bv in (False, True):
        tag = "bv" if bv else "chain"
        bad = {"stock": 0, "fixed": 0}
        oor = {"stock": 0, "fixed": 0}
        differ = torch.zeros((), dtype=torch.int64, device=DEV)
        done = 0
        while done < rows_total:
            Tm, Dm = chain_matrix(0.8), chain_matrix(1.0)
            x = torch.randint(0, V, (bs,), generator=g, device=DEV)
            xs, ds = [x], []
            for _ in range(S - 1):
                d = torch.multinomial(Dm[xs[-1]], 1, generator=g).squeeze(1)
                ds.append(d)
                xs.append(d)
            P = torch.stack([Tm[xi] for xi in xs], 1).to(F32).contiguous()
            Q = torch.stack([Dm[xi] for xi in xs[:-1]], 1).to(F32).contiguous()
            cand = torch.stack([x] + ds, 1)
            coins = torch.rand((bs, S - 1), generator=g, device=DEV)
            cfin = torch.rand((bs,), generator=g, device=DEV)
            res = {}
            for which in ("stock", "fixed"):
                n, b, _, _ = run(P, Q, cand, coins, cfin, bv=bv, path="gate", which=which)
                inv, am = support(P, Q, cand, n, b, bv=bv, valid_vocab=V)
                bad[which] += int(inv.sum())
                oor[which] += int(((b < 0) | (b >= V)).sum())
                res[which] = (n, b, inv, am)
            differ += compare(res["stock"], res["fixed"], V - 1)
            done += bs
        line(bad["fixed"] == 0 and differ == 0, f"c.markov24.{tag}",
             f"{done} rows: invalid stock {bad['stock']} (out of range {oor['stock']}) fixed {bad['fixed']} "
             f"(out of range {oor['fixed']}); fixed != stock where stock was legal: {int(differ)}")


# --------------------------------------------------------------------------- (d) graphs
def _graph_case(P, Q, cand, g, *, warm, reps, coins_fixed=None, cfin_fixed=None):
    ref_path = "split" if warm else "classic"
    """Capture chain_speculative_sampling_triton (fixed, chain) on a side stream, replay it
    ``reps`` times with new coins and compare each replay with an eager call. ``warm``: an
    eager call on the capture stream first (its split scratch then exists and the split
    kernels are captured); cold: no scratch for that stream, so the classic kernel is
    captured. Before every replay the scratch is poisoned (Result = -12345, Fallback =
    2 * VOCAB - 1), so a missing reset shows up as a wrong token."""
    bs, S = cand.shape
    tiles = -(-VOCAB // 4096)
    coins = torch.rand((bs, S - 1), generator=g, device=DEV)
    cfin = torch.rand((bs,), generator=g, device=DEV)
    predicts = torch.zeros((bs * S,), dtype=torch.int32, device=DEV)
    acc_idx = torch.zeros((bs, S), dtype=torch.int32, device=DEV)
    acc_num = torch.zeros((bs,), dtype=torch.int32, device=DEV)
    ridx = torch.arange(bs * S, dtype=torch.int64, device=DEV).view(bs, S)
    call_args = (predicts, acc_idx, acc_num, cand, ridx, ridx, ridx, coins, cfin, P, Q, 1.0, 1.0, True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        if warm:
            rs.chain_speculative_sampling_triton(*call_args)  # makes this stream's split scratch
        else:
            with split_gate(False):  # compiles the classic kernel, makes no scratch
                rs.chain_speculative_sampling_triton(*call_args)
    torch.cuda.current_stream().wait_stream(s)
    if not warm:
        assert rf_reject().get((torch.cuda.current_device(), s.cuda_stream, tiles)) is None
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=s):
        rs.chain_speculative_sampling_triton(*call_args)
    scratch = rf_reject().get((torch.cuda.current_device(), s.cuda_stream, tiles))
    split_runs = mism = 0
    tokens, lens = [], []
    for i in range(reps):
        if coins_fixed is None:
            coins.copy_(torch.rand((bs, S - 1), generator=g, device=DEV))
            cfin.copy_(torch.rand((bs,), generator=g, device=DEV))
            if i % 7 == 0:
                coins.fill_(1.0)  # rejections: the residual path
            if i % 5 == 0:
                cfin.fill_(1.0)  # no lane matches: the fallback
        else:
            coins.copy_(coins_fixed)
            cfin.copy_(cfin_fixed)
        if scratch is not None:
            scratch[1].fill_(-12345)
            scratch[3].fill_(2 * VOCAB - 1)
        graph.replay()
        torch.cuda.synchronize()
        if scratch is not None and int(scratch[1][0]) != -12345:
            split_runs += 1
        n_g = acc_num.long()
        b_g = predicts[acc_idx.gather(1, n_g[:, None]).squeeze(1).long()].long()
        n_r, b_r, _, _ = run(P, Q, cand, coins.clone(), cfin.clone(), bv=False, path=ref_path, which="fixed")
        mism += int((n_g != n_r).sum()) + int((b_g != b_r).sum())
        tokens.append(b_g)
        lens.append(n_g)
    arrived_zero = scratch is None or int(scratch[2].abs().sum()) == 0
    return split_runs, mism, arrived_zero, torch.stack(tokens), torch.stack(lens)


def section_d(args):
    info("(d) CUDA graph capture / replay (fixed): warm split capture, cold capture")
    sm = torch.cuda.get_device_properties(DEV).multi_processor_count
    bs, S = 4, 8
    reps = 50
    g = torch.Generator(device=DEV).manual_seed(4)
    gate = rs._should_split_final_sampling(bs, -(-VOCAB // 4096), DEV)
    P, Q, cand = random_rows(bs, S, g)
    split_runs, mism, reset, _, _ = _graph_case(P, Q, cand, g, warm=True, reps=reps)
    line(gate and split_runs == reps and mism == 0 and reset, "d.graph.split.warm",
         f"gate split={gate} (SMs {sm}); split kernels ran in {split_runs}/{reps} replays; "
         f"{mism} mismatches vs eager; arrival counters reset={reset}")
    split_runs, mism, reset, _, _ = _graph_case(P, Q, cand, g, warm=False, reps=10)
    line(split_runs == 0 and mism == 0, "d.graph.cold",
         f"no scratch for the capture stream -> classic kernel captured (split runs {split_runs}); "
         f"{mism} mismatches vs eager classic")
    del P, Q
    lanes = (5, 4097, 100000, 200057, 150000)
    for name in ("n=0", "degenerate", "all-accepted"):
        P1, Q1, c1, coins1, n_exp, tok1, _ = crafted(VOCAB, lanes, S, name)
        P, Q = P1[None].expand(bs, -1, -1), Q1[None].expand(bs, -1, -1)
        cand = c1[None].expand(bs, -1).contiguous()
        split_runs, mism, reset, toks, ns = _graph_case(
            P, Q, cand, g, warm=True, reps=10, coins_fixed=coins1[None].expand(bs, -1),
            cfin_fixed=torch.ones(bs, device=DEV))
        line(split_runs == 10 and mism == 0 and reset and bool((toks == tok1).all()) and bool((ns == n_exp).all()),
             f"d.graph.split.nomatch.{name}",
             f"coin_final 1.0, poisoned scratch: tokens {sorted(set(toks.flatten().tolist()))} (want {tok1}); "
             f"accept lengths {sorted(set(ns.flatten().tolist()))} (want {n_exp}); split runs {split_runs}/10; "
             f"{mism} mismatches")


def rf_reject():
    return importlib.import_module("inkling_rsfix.reject")._SPLIT_SCRATCH


# ------------------------------------------------------------------- (e) PR regressions
def section_e(args):
    info("(e) PR #51681 / #51692 regressions")
    ms = importlib.import_module("sglang.kernels.ops.speculative.dspark.markov_sample")
    T_MK = rf.apply("markov", ms)
    key = 0x2D2A9C317E44B1F3
    for device in ("cuda", "cpu"):
        out = {}
        for which in ("stock", "fixed"):
            with variant(ms, T_MK, which):
                logits = torch.tensor([[-torch.inf, 0.0]], device=device)
                out[which] = int(ms.MarkovSampleStep.execute(
                    base_logits=logits, bias=torch.zeros_like(logits),
                    temperatures=torch.ones(1, device=device), key=key,
                    offset=torch.tensor([4_757_896], device=device), step=0,
                    step_dtype=torch.float32))
        line(out["fixed"] == 1, f"e.51681.markov.{device}",
             f"logits [-inf, 0] at the PR's counter: stock token {out['stock']}, fixed {out['fixed']} (want 1)")
    # Fixed Triton vs fixed torch reference on random rows (both the PR's code).
    g = torch.Generator(device=DEV).manual_seed(5)
    rows, padded, width = 64, 9000, 8990
    base = torch.randn((rows, padded), generator=g, device=DEV)
    bias = torch.randn((rows, width), generator=g, device=DEV)
    temps = torch.rand((rows,), generator=g, device=DEV) + 0.5
    temps[::5] = 0.0
    kw = dict(base_logits=base, bias=bias, temperatures=temps, key=key,
              offset=torch.tensor([7], device=DEV), step=2, step_dtype=torch.float32)
    tri = ms.MarkovSampleStep.triton(**kw)
    tor = ms.MarkovSampleStep.torch(**kw)
    agree = float((tri == tor).float().mean())
    line(agree >= 0.999, "e.51681.markov.triton-vs-torch", f"{rows} rows x {width}: agreement {agree:.4f}")

    bw = importlib.import_module("sglang.kernels.ops.speculative.dspark.blockwise_rejection")
    T_BW = rf.apply("blockwise", bw)
    for logit, shift in ((0.0, 16.0), (-3.0, 0.0)):
        res = {}
        for which in ("stock", "fixed"):
            with variant(bw, T_BW, which):
                draft = torch.tensor([[-torch.inf, logit, 0.0, -torch.inf]], device=DEV)
                target = torch.cat([draft + shift, draft])
                temps1 = torch.ones(1, device=DEV)
                noise = bw.make_rejection_noise(0, "cuda")
                noise.counter.fill_(620191)
                length, bonus, _ = bw.BlockwiseRejection.execute(
                    candidates=torch.tensor([[0, 1]], device=DEV), target_logits=target,
                    draft_logits=draft.unsqueeze(0), target_temperatures=temps1,
                    draft_temperatures=temps1, noise=noise)
                res[which] = (int(length), int(bonus))
        ok = res["fixed"][0] == 1 and res["fixed"][1] in (1, 2)
        line(ok, f"e.51692.equal-laws.{logit:+.0f}.{shift:.0f}",
             f"coin one FP32 step below 1: stock (len, bonus) {res['stock']}, fixed {res['fixed']} (want len 1)")
    for mode in ("bonus", "one_hot", "residual"):
        res = {}
        for which in ("stock", "fixed"):
            with variant(bw, T_BW, which):
                res[which] = _shift_case(bw, mode)
        line(res["fixed"], f"e.51692.shift-invariance.{mode}", f"stock {res['stock']}, fixed {res['fixed']}")


def _shift_case(bw, mode):
    rows, vocab = 128, 8195
    temp = 2.0 ** -16
    scores = torch.full((vocab,), -torch.inf, device=DEV)
    scores[2], scores[-1] = -temp, 0.0
    proposal = torch.full_like(scores, -torch.inf)
    proposal[1] = 0.0
    if mode == "residual":
        proposal[2], proposal[-1] = -temp, -temp
    draft = proposal.repeat(rows, 1, 1) if mode != "one_hot" else None
    temps = torch.full((rows,), temp, device=DEV)
    candidates = torch.tensor([[0, 1]], device=DEV).expand(rows, -1)

    def once(shift):
        selected = scores + shift
        target = torch.stack([proposal if mode == "bonus" else selected, selected]).repeat(rows, 1)
        length, bonus, _ = bw.BlockwiseRejection.execute(
            candidates=candidates, target_logits=target, draft_logits=draft,
            target_temperatures=temps, draft_temperatures=temps,
            noise=bw.make_rejection_noise(0, "cuda"))
        ok = bool((length == int(mode == "bonus")).all()) and bool(((bonus == 2) | (bonus == vocab - 1)).all())
        return ok, bonus

    ok0, b0 = once(0.0)
    ok1, b1 = once(128.0)
    return ok0 and ok1 and torch.equal(b0, b1)


# ------------------------------------------------------------------------- (f) timing
def section_f(args):
    info("(f) timing, informational (gamma 7, V=201024, rows read from DRAM)")
    g = torch.Generator(device=DEV).manual_seed(6)
    S = 8
    for path, bv, bs in (("classic", True, 1), ("classic", True, 8), ("split", False, 1),
                         ("split", False, 8), ("classic", False, 64)):
        P, Q, cand = random_rows(bs, S, g, dust=2000)
        coins = torch.rand((bs, S - 1), generator=g, device=DEV)
        cfin = torch.rand((bs,), generator=g, device=DEV)
        predicts = torch.zeros((bs * S,), dtype=torch.int32, device=DEV)
        acc_idx = torch.zeros((bs, S), dtype=torch.int32, device=DEV)
        acc_num = torch.zeros((bs,), dtype=torch.int32, device=DEV)
        ridx = torch.arange(bs * S, dtype=torch.int64, device=DEV).view(bs, S)
        med = {}
        for which in ("stock", "fixed"):
            with split_gate(path == "split"), variant(rs, T_REJ, which):
                def call():
                    rs.chain_speculative_sampling_triton(
                        predicts, acc_idx, acc_num, cand, ridx, ridx, ridx, coins, cfin, P, Q,
                        1.0, 1.0, True, block_verification=bv)
                for _ in range(3):
                    call()
                times = []
                for _ in range(50):
                    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    a.record()
                    call()
                    b.record()
                    b.synchronize()
                    times.append(a.elapsed_time(b) * 1000.0)
                med[which] = sorted(times)[len(times) // 2]
        info(f"f.{path}-{'bv' if bv else 'chain'} bs={bs}: median us stock {med['stock']:.1f} fixed {med['fixed']:.1f}")
        del P, Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller (a)/(b)/(c) for a smoke run")
    ap.add_argument("--draws", type=float, default=1e7, help="(b) draws per classic path")
    ap.add_argument("--split-draws", type=float, default=2e6, help="(b) draws on the split path")
    ap.add_argument("--markov-rows", type=int, default=2_000_000, help="(c) rows per mode")
    ap.add_argument("--only", default="abcdef")
    args = ap.parse_args()
    if args.quick:
        args.draws, args.split_draws = 5e5, 1e5
    args.draws, args.split_draws = int(args.draws), int(args.split_draws)
    torch.manual_seed(0)
    info(f"device {torch.cuda.get_device_name(DEV)}, SMs "
         f"{torch.cuda.get_device_properties(DEV).multi_processor_count}, torch {torch.__version__}")
    t0 = time.time()
    for sec in args.only:
        {"a": section_a, "b": section_b, "c": section_c, "d": section_d, "e": section_e,
         "f": section_f}[sec](args)
    info(f"peak device memory {torch.cuda.max_memory_allocated(DEV) / 2**30:.2f} GiB, {time.time() - t0:.0f} s")
    print(f"{'FAIL' if FAILS else 'PASS'} gpu_check_rsfix: {len(FAILS)} failed {FAILS}", flush=True)
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
