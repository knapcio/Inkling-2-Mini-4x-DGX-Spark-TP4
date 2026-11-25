#!/usr/bin/env python3
"""Batch-level replay of DSpark verify-width policies on stock full-width steplogs.

Input: engine/m0/steplog .npz chunks logged at full width (live == 8) with the prefix
union counter (union_k). One row per (decode step, request); the rows of one engine
step form one batch of size bs. Every policy is replayed on the logged trajectory:

    request r with live length L_r commits min(A_r, L_r - 1) + 1 tokens
    step time  T = F(bs) + b(bs) * U(L_1..L_bs) + overhead(policy, bs)

U is the per-MoE-visit mean expert union (what union_k measures). Union model:

    one request with L rows covers u1(L) experts      (c1 prefix-union curve)
    U(L_1..L_bs) = E * (1 - prod_r (1 - u1(L_r) / E))  (requests overlap like independent
                                                        draws from an effective pool E)

E is fitted per step from the measured U(8) of that step (so fixed:8 reproduces the
log); for bs = 1 the step's own measured prefix union is used directly. The marginal
union of request r's next row is P_{-r} * (u1(L+1) - u1(L)), P_{-r} = prod_{s != r}
(1 - u1(L_s) / E): its new experts count only when no other request covers them.

Policies (per step, vectorised per batch size):

  fixed:k     every request k rows
  conf:T      shipped rule: surv_j = prod_{i<=j} m_i (raw draft max-probabilities),
              L = 1 + #{j : surv_j >= T}; the shipped table picks T by bs
  ltrc:theta  calibrated survival S_j = prod a_i, a_i = calib[mode][i][bin(m_i)];
              include d_j while S_j >= theta * c_j, c_j = u1(j+2) - u1(j+1) (the c1
              marginal union of the row that verifies d_j)
  ltrb:theta  batch-aware: include d_j while S_j >= theta * P_{-r} * c_j, P_{-r} from the
              other requests' current widths with a per-bs pool E_bs; K fixed-point
              rounds starting from L = 1 (monotone, so each request's width is a
              left-to-right stopping rule: see README)
  oracle      L = A + 1

All tunables (T, theta, k) are fitted per batch size on the training fold and scored
on the held-out fold (2-fold cross-fit by cluster; the calibration table is fitted
on the training fold too). Clusters: the request id at bs = 1, blocks of
--block consecutive steps at bs > 1. CIs: paired cluster bootstrap of the held-out
rate ratio vs the shipped rule.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

STRIDE = 8
GAMMA = 7
EDGES = (0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995)
# c1 prefix union of the L1 steplog (stock, greedy+sampled mixed), used when the data
# has no bs = 1 rows.
U1_DEFAULT = (8.0, 12.4, 15.9, 18.9, 21.4, 23.7, 25.6, 27.4)
SHIPPED = ((1, 0.30), (2, 0.25), (4, 0.15), (8, 0.0))


# ------------------------------------------------------------------ data
@dataclass
class Group:
    """All complete full-width steps of one batch size: arrays [n, bs, ...]."""

    bs: int
    m: np.ndarray  # [n, bs, 7] float
    acc: np.ndarray  # [n, bs] int accepted drafts at full width
    greedy: np.ndarray  # [n, bs] bool
    pf: np.ndarray  # [n, bs] int prev_full (full-width trajectory)
    union: np.ndarray  # [n, 8] measured prefix union (nan if absent)
    cluster: np.ndarray  # [n] str
    rid: np.ndarray  # [n, bs] str
    run: np.ndarray  # [n] int

    def __len__(self):
        return len(self.acc)

    def take(self, idx):
        return Group(self.bs, self.m[idx], self.acc[idx], self.greedy[idx], self.pf[idx],
                     self.union[idx], self.cluster[idx], self.rid[idx], self.run[idx])


def _files(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            out += sorted(glob.glob(os.path.join(p, "**", "steplog-*.npz"), recursive=True))
        else:
            out += sorted(glob.glob(p))
    return out


def load(paths, block=32, drop_rid=None):
    files = _files(paths)
    if not files:
        raise SystemExit(f"no steplog files under {paths}")
    cols = defaultdict(list)
    runs = {}
    for f in files:
        prefix = os.path.join(os.path.dirname(f),
                              re.sub(r"-\d{6}\.npz$", "", os.path.basename(f)))
        run = runs.setdefault(prefix, len(runs))
        z = np.load(f, allow_pickle=False)
        n = len(z["step"])
        cols["run"].append(np.full(n, run))
        for k in ("step", "row", "bs", "live", "greedy", "argmax_ok", "temp"):
            cols[k].append(z[k])
        cols["rid"].append(np.asarray([f"{run}:{r}" for r in z["rid"]]))
        cols["m"].append(z["m"].astype(np.float64))
        cols["acc"].append((z["uncapped"] if "uncapped" in z.files else z["accept"]).astype(np.int64))
        cols["union"].append(z["union_k"].astype(np.float64) if "union_k" in z.files
                             else np.full((n, STRIDE), np.nan))
    d = {k: np.concatenate(v) for k, v in cols.items()}
    notes = {"files": len(files), "runs": len(runs), "rows": len(d["step"])}
    order = np.lexsort((d["row"], d["step"], d["run"]))
    d = {k: v[order] for k, v in d.items()}
    # prev_full on the full-width trajectory (first step of a request: 1)
    pf = np.ones(len(d["acc"]), np.int64)
    last = {}
    for i, r in enumerate(d["rid"]):
        j = last.get(r)
        if j is not None:
            pf[i] = int(d["acc"][j] >= GAMMA)
        last[r] = i
    d["pf"] = pf
    bad_row = (d["live"] < STRIDE) | (d["greedy"] & ~d["argmax_ok"])
    if drop_rid:
        bad_row |= np.asarray([bool(re.search(drop_rid, r)) for r in d["rid"]])
    key = d["run"] * 10**9 + d["step"]
    uniq, start, count = np.unique(key, return_index=True, return_counts=True)
    groups = defaultdict(list)
    dropped = 0
    for s, c in zip(start, count):
        sl = slice(s, s + c)
        if c != d["bs"][s] or bad_row[sl].any() or (d["row"][sl] != np.arange(c)).any():
            dropped += 1
            continue
        groups[int(c)].append(s)
    notes["steps_dropped"] = dropped
    out = {}
    for bs, starts in sorted(groups.items()):
        idx = np.asarray(starts)[:, None] + np.arange(bs)[None, :]
        run = d["run"][idx[:, 0]]
        step = d["step"][idx[:, 0]]
        if bs == 1:
            cl = d["rid"][idx[:, 0]]
        else:
            cl = np.asarray([f"{r}:b{bs}:{s // block}" for r, s in zip(run, step)])
        out[bs] = Group(bs, d["m"][idx], d["acc"][idx], d["greedy"][idx].astype(bool),
                        d["pf"][idx], d["union"][idx[:, 0]], cl, d["rid"][idx], run)
    notes["steps"] = {bs: len(g) for bs, g in out.items()}
    return out, notes


# ------------------------------------------------------------------ union model
def u1_curve(groups):
    g = groups.get(1)
    if g is not None and np.isfinite(g.union).all() and len(g) > 50:
        return g.union.mean(axis=0), "measured c1 (bs=1 steps of this data)"
    return np.asarray(U1_DEFAULT), "L1 steplog c1 default"


def fit_pool(u1, bs, u_full):
    """E such that E(1 - (1 - u1[7]/E)^bs) = u_full (per step). inf if no overlap."""
    target = np.asarray(u_full, np.float64)
    out = np.empty_like(target)
    top = bs * u1[-1]
    for i, t in enumerate(target):
        if not np.isfinite(t) or t >= top - 1e-6:
            out[i] = np.inf
            continue
        t = max(t, u1[-1] + 1e-6)
        lo, hi = u1[-1], 1e7
        for _ in range(80):
            mid = math.sqrt(lo * hi)
            val = mid * (1 - (1 - u1[-1] / mid) ** bs)
            lo, hi = (mid, hi) if val < t else (lo, mid)
        out[i] = math.sqrt(lo * hi)
    return out


def union_of(u1, L, E):
    """[n] union for widths L [n, bs] with pools E [n] (inf -> sum)."""
    u = u1[L - 1]  # [n, bs]
    E = np.asarray(E, np.float64)[:, None] * np.ones_like(u)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = np.where(np.isfinite(E), u / E, 0.0)
        prod = np.prod(1 - frac, axis=1)
        val = E[:, 0] * (1 - prod)
    return np.where(np.isfinite(E[:, 0]), val, u.sum(axis=1))


# ------------------------------------------------------------------ cost model
class Cost:
    """T(bs, U) = F(bs) + b(bs) * U; overhead of a dynamic (draft-logit) policy
    ovh0 + ovh1 * bs ms. JSON: {"fixed_ms": {bs: F}, "b_ms_per_expert": {bs: b} or b,
    "ovh0_ms": x, "ovh1_ms": y}; missing bs interpolate linearly in bs."""

    def __init__(self, spec=None):
        spec = spec or {}
        self.F = {int(k): float(v) for k, v in spec.get("fixed_ms", {}).items()}
        b = spec.get("b_ms_per_expert", 0.97)
        self.b = ({int(k): float(v) for k, v in b.items()} if isinstance(b, dict)
                  else {1: float(b)})
        self.ovh0 = float(spec.get("ovh0_ms", 0.05))
        self.ovh1 = float(spec.get("ovh1_ms", 0.10))
        self.src = spec.get("source", "default model")

    @staticmethod
    def _interp(table, bs):
        ks = sorted(table)
        if bs in table:
            return table[bs]
        if len(ks) == 1:
            return table[ks[0]]
        return float(np.interp(bs, ks, [table[k] for k in ks]))

    def b_at(self, bs):
        return self._interp(self.b, bs)

    def F_at(self, bs, u1):
        if self.F:
            return self._interp(self.F, bs)
        # default: c1 step 60.4 ms at U8 = 27.4 -> F(1) = 33.8; +2.5 ms per extra
        # request (attention/dense/head/draft rows): a placeholder until profiled
        return 60.4 - 0.97 * 27.4 + 2.5 * (bs - 1)

    def ovh(self, bs):
        return self.ovh0 + self.ovh1 * bs


# ------------------------------------------------------------------ calibration
class Calib:
    def __init__(self, edges=EDGES, use_pf=True):
        self.edges = np.asarray(edges, np.float64)
        nb = len(edges) + 1
        self.use_pf = use_pf
        self.succ = np.zeros((2, GAMMA, nb, 2))
        self.trials = np.zeros((2, GAMMA, nb, 2))

    def bins(self, m):
        return np.searchsorted(self.edges.astype(np.float32), m.astype(np.float32),
                               side="right")

    def fit(self, groups_list):
        for g in groups_list:
            m = g.m.reshape(-1, GAMMA)
            a = g.acc.reshape(-1)
            mode = (~g.greedy.reshape(-1)).astype(int)  # 0 greedy, 1 sampled
            pf = g.pf.reshape(-1) if self.use_pf else np.zeros_like(a)
            b = self.bins(m)
            for j in range(GAMMA):
                trial = a >= j
                np.add.at(self.trials, (mode[trial], j, b[trial, j], pf[trial]), 1)
                ok = a >= j + 1
                np.add.at(self.succ, (mode[ok], j, b[ok, j], pf[ok]), 1)
        return self

    def table(self):
        return (self.succ + 1.0) / (self.trials + 2.0)

    def surv(self, g):
        t = self.table()
        mode = (~g.greedy).astype(int)[..., None]
        j = np.arange(GAMMA)[None, None, :]
        pf = (g.pf if self.use_pf else np.zeros_like(g.pf))[..., None]
        a = t[mode, j, self.bins(g.m), pf]
        return np.cumprod(a, axis=-1)


# ------------------------------------------------------------------ policies
def width_from(ok):
    """leading run of True over the gamma positions -> L in 1..8."""
    return 1 + np.cumprod(ok, axis=-1).sum(axis=-1)


def pol_fixed(g, k):
    return np.full(g.acc.shape, k, np.int64)


def pol_conf(g, T):
    if T <= 0:
        return pol_fixed(g, STRIDE)
    return width_from(np.cumprod(g.m, axis=-1) >= T)


def pol_ltrc(S, c, theta):
    return width_from(S >= theta * c[None, None, :])


def pol_ltrb(S, c, theta, u1, E_bs, rounds=3):
    n, bs, _ = S.shape
    L = np.ones((n, bs), np.int64)
    if not np.isfinite(E_bs) or bs == 1:
        return pol_ltrc(S, c, theta)
    for _ in range(rounds):
        f = 1 - u1[L - 1] / E_bs  # [n, bs]
        tot = np.prod(f, axis=1, keepdims=True)
        p_other = tot / f
        L = width_from(S >= theta * p_other[..., None] * c[None, None, :])
    return L


def pol_oracle(g):
    return g.acc + 1


# ------------------------------------------------------------------ scoring
UNION_MODEL = ["interp"]


def union_interp(u1, L, meas):
    """Union of a mixed-width batch from the step's own measured fixed-width curve:
    map each request's width to its c1 union u1(L_r), average, and read the step's
    measured U(k) curve at that mean (piecewise linear in u1). Exact for uniform
    widths (and at bs = 1); concave within a request like the c1 curve."""
    ubar = u1[L - 1].mean(axis=1)
    out = np.empty(len(L))
    for i in range(len(L)):
        out[i] = np.interp(ubar[i], u1, meas[i])
    return out


def score(g, L, cost, u1, E, dynamic):
    tok = (np.minimum(g.acc, L - 1) + 1).sum(axis=1).astype(np.float64)
    if g.bs == 1 and np.isfinite(g.union).all():
        U = g.union[np.arange(len(g)), L[:, 0] - 1]
    elif UNION_MODEL[0] == "interp" and np.isfinite(g.union).all():
        U = union_interp(u1, L, g.union)
    elif UNION_MODEL[0] == "rows" and np.isfinite(g.union).all():
        # pessimistic-side sensitivity: the batch curve read at the mean row count
        kbar = L.mean(axis=1)
        U = np.asarray([np.interp(kbar[i], np.arange(1, STRIDE + 1), g.union[i])
                        for i in range(len(L))])
    else:
        U = union_of(u1, L, E)
    ms = cost.F_at(g.bs, u1) + cost.b_at(g.bs) * U + (cost.ovh(g.bs) if dynamic else 0.0)
    return tok, ms, L.mean(axis=1), U


def rate(tok, ms):
    return 1000.0 * tok.sum() / ms.sum()


def fold_of(clusters, seed):
    return np.asarray([int(hashlib.sha256(f"{seed}:{c}".encode()).hexdigest(), 16) % 2
                       for c in clusters])


THETAS = np.exp(np.linspace(np.log(0.002), np.log(0.5), 60))
TS = np.concatenate([[0.0], np.exp(np.linspace(np.log(0.01), np.log(0.8), 50))])


def shipped_T(bs, table=SHIPPED):
    for k, v in table:
        if bs <= k:
            return v
    return table[-1][1]


def evaluate(groups, cost, args):
    u1, u1_src = u1_curve(groups)
    c = np.diff(u1)  # c_j = u1(j+2) - u1(j+1), j = 0..6
    results = {"u1": u1.tolist(), "u1_source": u1_src, "per_bs": {}}
    pools = {bs: fit_pool(u1, bs, g.union[:, -1]) if bs > 1 else np.full(len(g), np.inf)
             for bs, g in groups.items()}
    # union-model check: predicted fixed:k unions vs measured
    fitchk = {}
    for bs, g in groups.items():
        if bs == 1 or not np.isfinite(g.union).all():
            continue
        errs = []
        for k in range(1, STRIDE):
            pred = union_of(u1, np.full(g.acc.shape, k), pools[bs])
            errs.append(float(np.mean(pred - g.union[:, k - 1])))
        fitchk[bs] = {"mean_U8": float(g.union[:, -1].mean()),
                      "median_E": float(np.median(pools[bs][np.isfinite(pools[bs])]))
                      if np.isfinite(pools[bs]).any() else float("inf"),
                      "pred_minus_meas_k1..7": [round(e, 2) for e in errs],
                      "meas_mean_k1..8": [round(float(x), 2) for x in g.union.mean(axis=0)]}
    results["union_fit"] = fitchk
    folds = {bs: fold_of(g.cluster, args.seed) for bs, g in groups.items()}
    per_cluster = {}
    # one theta for every batch size (ltrc_global*): per fold, maximise the mean over
    # batch sizes (>= 30 training steps) of the training-fold rate vs shipped
    theta_global = {}
    for fold in (0, 1):
        cal = Calib(use_pf=False).fit([gg.take(folds[b] != fold) for b, gg in groups.items()])
        score_sum = np.zeros(len(THETAS))
        for bs, g in groups.items():
            gtr = g.take(folds[bs] != fold)
            if len(gtr) < 30:
                continue
            Etr = pools[bs][folds[bs] != fold]
            Tsh = shipped_T(bs)
            t0, m0, _, _ = score(gtr, pol_conf(gtr, Tsh), cost, u1, Etr, Tsh > 0)
            S = cal.surv(gtr)
            for i, th in enumerate(THETAS):
                t, m_, _, _ = score(gtr, pol_ltrc(S, c, th), cost, u1, Etr, True)
                score_sum[i] += rate(t, m_) / rate(t0, m0)
        theta_global[fold] = (args.fix_theta if args.fix_theta is not None
                              else float(THETAS[int(np.argmax(score_sum))]))
    results["theta_global"] = theta_global
    for bs, g in groups.items():
        if len(g) < args.min_steps:
            continue
        names = None
        tok_all, ms_all, width_all = {}, {}, {}
        chosen = defaultdict(list)
        for fold in (0, 1):
            tr_mask = folds[bs] != fold
            te_mask = folds[bs] == fold
            if tr_mask.sum() == 0 or te_mask.sum() == 0:
                continue
            calib = Calib(use_pf=not args.no_pf).fit(
                [gg.take(folds[b] != fold) for b, gg in groups.items()])
            calib_nopf = Calib(use_pf=False).fit(
                [gg.take(folds[b] != fold) for b, gg in groups.items()])
            gtr, gte = g.take(tr_mask), g.take(te_mask)
            Etr, Ete = pools[bs][tr_mask], pools[bs][te_mask]
            E_bs = float(np.median(Etr)) if bs > 1 else float("inf")
            Str, Ste = calib.surv(gtr), calib.surv(gte)
            Str0, Ste0 = calib_nopf.surv(gtr), calib_nopf.surv(gte)

            def best(fn_tr, grid, dynamic_of):
                vals = []
                for p in grid:
                    t, m_, _, _ = score(gtr, fn_tr(p), cost, u1, Etr, dynamic_of(p))
                    vals.append(rate(t, m_))
                i = int(np.argmax(vals))
                return grid[i], vals[i]

            pols = {}
            for k in range(1, STRIDE + 1):
                pols[f"fixed:{k}"] = (pol_fixed(gte, k), False)
            Tsh = shipped_T(bs)
            pols["shipped"] = (pol_conf(gte, Tsh), Tsh > 0)
            kbest, _ = best(lambda k: pol_fixed(gtr, k), list(range(1, STRIDE + 1)),
                            lambda k: False)
            pols["best_fixed*"] = (pol_fixed(gte, kbest), False)
            chosen["best_fixed*"].append(kbest)
            Tb, _ = best(lambda T: pol_conf(gtr, T), list(TS), lambda T: T > 0)
            pols["conf_tuned*"] = (pol_conf(gte, Tb), Tb > 0)
            chosen["conf_tuned*"].append(round(float(Tb), 4))
            th, _ = best(lambda t: pol_ltrc(Str0, c, t), list(THETAS), lambda t: True)
            pols["ltrc_nopf*"] = (pol_ltrc(Ste0, c, th), True)
            chosen["ltrc_nopf*"].append(round(float(th), 5))
            th, _ = best(lambda t: pol_ltrc(Str, c, t), list(THETAS), lambda t: True)
            pols["ltrc*"] = (pol_ltrc(Ste, c, th), True)
            pols["ltrc_global*"] = (pol_ltrc(Ste0, c, theta_global[fold]), True)
            for tab in args.alt_table or []:
                Ta = shipped_T(bs, tab)
                pols["cost:" + ",".join(f"{k}={v:g}" for k, v in tab)] = (pol_conf(gte, Ta), Ta > 0)
            chosen["ltrc*"].append(round(float(th), 5))
            if bs > 1:
                for rounds in (1, 3):
                    th, _ = best(lambda t: pol_ltrb(Str0, c, t, u1, E_bs, rounds),
                                 list(THETAS), lambda t: True)
                    nm = f"ltrb{rounds}_nopf*"
                    pols[nm] = (pol_ltrb(Ste0, c, th, u1, E_bs, rounds), True)
                    chosen[nm].append(round(float(th), 5))
            pols["oracle"] = (pol_oracle(gte), False)
            for name, (L, dyn) in pols.items():
                t, m_, w, _ = score(gte, L, cost, u1, Ete, dyn)
                tok_all.setdefault(name, []).append((gte.cluster, t, m_, w))
            names = list(pols)
        if names is None:
            continue
        # combine folds
        res = {}
        clus = None
        stacks = {}
        for name in names:
            cl = np.concatenate([x[0] for x in tok_all[name]])
            t = np.concatenate([x[1] for x in tok_all[name]])
            m_ = np.concatenate([x[2] for x in tok_all[name]])
            w = np.concatenate([x[3] for x in tok_all[name]])
            uc, inv = np.unique(cl, return_inverse=True)
            stacks[name] = (np.bincount(inv, t, len(uc)), np.bincount(inv, m_, len(uc)))
            res[name] = {"tok_s": rate(t, m_), "width": float(w.mean()),
                         "tok_per_req_step": float(t.mean() / bs), "ms_step": float(m_.mean())}
            clus = uc
        base = "shipped"
        rng = np.random.default_rng(args.seed)
        nc = len(clus)
        boots = {name: [] for name in names}
        for _ in range(args.bootstrap):
            idx = rng.integers(0, nc, nc)
            b0 = stacks[base][0][idx].sum() / stacks[base][1][idx].sum()
            for name in names:
                tt, mm = stacks[name]
                boots[name].append((tt[idx].sum() / mm[idx].sum()) / b0 - 1)
        for name in names:
            res[name]["vs_shipped"] = res[name]["tok_s"] / res[base]["tok_s"] - 1
            if args.bootstrap:
                res[name]["ci"] = [float(x) for x in np.percentile(boots[name], [2.5, 97.5])]
        mode_share = float(g.greedy.mean())
        results["per_bs"][bs] = {"steps": len(g), "clusters": int(nc),
                                 "greedy_share": mode_share,
                                 "mean_accept": float(g.acc.mean()),
                                 "chosen": dict(chosen), "policies": res}
        per_cluster[bs] = stacks
    return results


def export_policy(groups, cost, args, path):
    """Fit on ALL data: calibration (no prev_full: the device rule is stateless), per-bs
    theta (0 when full width is best), per-bs pool E; format inkling-vcap-policy/1."""
    u1, _ = u1_curve(groups)
    c = np.diff(u1)
    calib = Calib(use_pf=False).fit(list(groups.values()))
    tab = calib.table()[:, :, :, 0]  # [2, gamma, nb]
    theta, pool = [], []
    rounds = args.export_rounds
    for bs, g in sorted(groups.items()):
        if len(g) < args.min_steps:
            continue
        E = fit_pool(u1, bs, g.union[:, -1]) if bs > 1 else np.full(len(g), np.inf)
        E_bs = float(np.median(E)) if bs > 1 else float("inf")
        S = calib.surv(g)
        full_t, full_m, _, _ = score(g, pol_fixed(g, STRIDE), cost, u1, E, False)
        best_t, best_v = 0.0, rate(full_t, full_m)
        for t in THETAS:
            L = pol_ltrb(S, c, t, u1, E_bs, rounds) if rounds else pol_ltrc(S, c, t)
            tk, ms, _, _ = score(g, L, cost, u1, E, True)
            if rate(tk, ms) > best_v:
                best_t, best_v = float(t), rate(tk, ms)
        theta.append([bs, best_t if args.fix_theta is None else args.fix_theta])
        if bs > 1 and np.isfinite(E_bs) and rounds:
            pool.append([bs, max(E_bs, float(u1[-1]) + 1e-3)])
    if args.fix_theta is not None and not rounds:
        theta = [[1, args.fix_theta]]  # one key: every batch size
    obj = {"format": "inkling-vcap-policy/1", "stride": STRIDE,
           "bin_edges": list(EDGES),
           "calib": {"greedy": tab[0].tolist(), "sampled": tab[1].tolist()},
           "u1": [float(x) for x in u1], "theta": theta, "rounds": rounds if pool else 0}
    if pool and rounds:
        obj["pool"] = pool
    obj["source"] = {"inputs": args.inputs, "cost": cost.src,
                     "steps": {bs: len(g) for bs, g in groups.items()}}
    with open(path, "w") as f:
        json.dump(obj, f, indent=1)
    return obj


def fmt(results, cost, notes):
    out = [f"notes: {notes}", f"cost: {cost.src}",
           f"global theta (fold0, fold1): {results.get('theta_global')}",
           "u1 [" + results["u1_source"] + "]: " + " ".join(f"{x:.1f}" for x in results["u1"])]
    for bs, f in sorted(results["union_fit"].items()):
        out.append(f"union bs={bs}: measured U(1..8) {f['meas_mean_k1..8']}  E~{f['median_E']:.1f} "
                   f"model-minus-measured k1..7 {f['pred_minus_meas_k1..7']}")
    for bs, r in sorted(results["per_bs"].items()):
        out.append("")
        out.append(f"== bs={bs}: {r['steps']} steps, {r['clusters']} clusters, greedy share "
                   f"{r['greedy_share']:.2f}, mean accept {r['mean_accept']:.2f}; "
                   f"tuned (fold0, fold1): {r['chosen']}")
        out.append(f"{'policy':16s} {'tok/s':>8s} {'width':>6s} {'tok/req/st':>10s} {'ms/step':>8s} "
                   f"{'vs shipped':>10s} {'95% CI':>18s}")
        for name, p in r["policies"].items():
            ci = p.get("ci")
            out.append(f"{name:16s} {p['tok_s']:8.2f} {p['width']:6.2f} {p['tok_per_req_step']:10.3f} "
                       f"{p['ms_step']:8.2f} {100 * p['vs_shipped']:+9.2f}% "
                       + (f"[{100 * ci[0]:+6.2f},{100 * ci[1]:+6.2f}]%" if ci else ""))
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--cost", help="cost-model JSON (see Cost)")
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--min-steps", type=int, default=60)
    ap.add_argument("--no-pf", action="store_true")
    ap.add_argument("--drop-rid")
    ap.add_argument("--greedy-only", action="store_true")
    ap.add_argument("--sampled-only", action="store_true")
    ap.add_argument("--json")
    ap.add_argument("--alt-table", action="append",
                    help="extra cost: table to replay, e.g. 1=0.30,24=0.25 (repeatable)")
    ap.add_argument("--fix-theta", type=float, help="use this theta for ltrc_global*")
    ap.add_argument("--union-model", choices=["interp", "rows", "pool"], default="interp")
    ap.add_argument("--export-policy", help="write an inkling-vcap-policy/1 JSON")
    ap.add_argument("--export-rounds", type=int, default=0)
    args = ap.parse_args(argv)
    UNION_MODEL[0] = args.union_model
    if args.alt_table:
        args.alt_table = [tuple(sorted((int(k), float(v)) for k, v in
                                       (e.split("=") for e in t.split(","))))
                          for t in args.alt_table]
    groups, notes = load(args.inputs, block=args.block, drop_rid=args.drop_rid)
    if args.greedy_only or args.sampled_only:
        want = args.greedy_only
        groups = {bs: g.take(np.where(g.greedy.all(1) if want else (~g.greedy).all(1))[0])
                  for bs, g in groups.items()}
        groups = {bs: g for bs, g in groups.items() if len(g)}
    cost = Cost(json.load(open(args.cost)) if args.cost else None)
    res = evaluate(groups, cost, args)
    print(fmt(res, cost, notes))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1, default=float)
    if args.export_policy:
        obj = export_policy(groups, cost, args, args.export_policy)
        print(f"exported {args.export_policy}: theta {obj['theta']} pool {obj.get('pool')}")


if __name__ == "__main__":
    main()
