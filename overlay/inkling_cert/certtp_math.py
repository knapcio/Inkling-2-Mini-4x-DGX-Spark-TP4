"""NumPy model of the certified head under tensor parallelism (what screen_tp, the (T, H) all-gather, select_tp and
exact compute on each rank), of the stock TP pipeline it must reproduce, and of the stock deferred argmax. For the CPU
tests and as the statement of the argument. No torch, no sglang.

Layout: world ranks, shard r = ids [r n, (r + 1) n), v_real real ids in total (the last shard holds the padding);
shard r's real ids are the first v_r = clamp(v_real - r n, 0, n).

Stock (the reference), per verify row m: l = pico's FP32 GEMV of every shard, y = f(l) per shard (the stock softcap,
before the gather), the gathered row is y with ids >= v_real set to -inf, token = its lowest maximiser (stock argmax).
The deferred variant (SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1): each rank sends (max y over its first v_r ids, the
lowest global id attaining it) and the token is the id of the lowest rank holding the largest maximum; with the
real-vocabulary limit (certtp.patch_deferral) this is the same token.

Certified, per rank r: lo, hi = the TP1 screen of shard r over its v_r real ids; send (T_r, H_r) = (max lo, max hi),
or +inf when the rank is not certifying (on = 0) or its screen saw a non-finite value; all-gather; T = max_r T_r and
H = max_r H_r are the same bits on every rank; with delta(T, H) and the range test of cert_math.select (on T and H),
tile t of shard r is computed iff some row has a member with hi >= T - delta; any failed test (any rank's +inf
included) computes every tile on every rank. Computed ids hold pico's l, the others -inf; then the stock path.

Why the token is the same. T = lo_u for some real id u (on some rank), and the TP1 argument (cert_math) uses only
lo_u = T, l_u <= hi_u <= H and the range test on (T, H): the stock winner w has hi_w >= T - delta, so its tile is
computed on the rank that holds it; every uncomputed id gives f(-inf) = -c < y_w; computed ids hold the stock values.
Hence the maximum over computed real ids is y_w and its lowest maximiser is w, both in the gathered row and in the
deferred (per-rank lowest maximiser, lowest rank on equal maxima) form, because ranks hold increasing id ranges.
A rank's fallback (all tiles) only adds stock values, so the argument holds for any mix of computed sets.
"""

import math

import numpy as np

from . import cert_math as cm


def shard_real(v_real, n, r):
    return max(0, min(n, v_real - r * n))


def tile_max(a):
    """[M, V] -> [M, V / 16] max per 16-id tile (lt / ht of k_screen)."""
    M, V = a.shape
    return a.reshape(M, V // cm.TILE, cm.TILE).max(-1)


def send_th(lt, ht, on=True, bad=False):
    """k_rowmax_tp: [M, 2] of (T_r, H_r), or +inf."""
    M = lt.shape[0]
    if not on or bad:
        return np.full((M, 2), np.inf, np.float32)
    return np.stack([lt.max(1), ht.max(1)], 1).astype(np.float32)


def merge_th(recv):
    """recv [world, M, 2] -> (T, H, nan) as k_select_tp reads them (fmaxf over ranks)."""
    nan = np.isnan(recv).any(axis=(0, 2))
    with np.errstate(invalid="ignore"):
        T = np.nanmax(np.where(np.isnan(recv[..., 0]), -np.inf, recv[..., 0]), axis=0)
        H = np.nanmax(np.where(np.isnan(recv[..., 1]), -np.inf, recv[..., 1]), axis=0)
    return T.astype(np.float32), H.astype(np.float32), nan


def row_threshold(T, H):
    """(ok, threshold) of one row, the k_select / k_select_tp arithmetic (cert_math.select)."""
    L = max(abs(float(T)) + cm.DMAX, abs(float(H)))
    ex = math.exp(-2.0 * L / cm.CAP) if math.isfinite(L) else 0.0
    delta = 2.0 * cm.TWO_E / max(ex, 1e-30)
    ok = math.isfinite(T) and math.isfinite(H) and L < 1e4 and delta <= cm.DMAX and cm.CAP * ex > 4 * cm.TWO_E
    return ok, float(T) - (delta if ok else 0.0)


def select_tp(ht, recv, on=True, bad=False):
    """k_select_tp: (tile mask of this shard, certified, thresholds)."""
    M, nt = ht.shape
    T, H, nan = merge_th(recv)
    full = (not on) or bad
    thr = np.zeros(M, np.float64)
    for m in range(M):
        if full:
            break
        ok, thr[m] = row_threshold(T[m], H[m])
        if nan[m] or not ok:
            full = True
    if full:
        return np.ones(nt, bool), False, None
    return (ht >= thr[:, None]).any(0), True, thr


def local_argmax(y, valid, offset):
    """softcap_row_argmax without a softcap: per row (max over the first `valid` columns, lowest such column +
    offset); (-inf, offset) when valid == 0."""
    M = y.shape[0]
    out = np.empty((M, 2), np.float64)
    if valid <= 0:
        out[:, 0], out[:, 1] = -np.inf, offset
        return out
    z = y[:, :valid]
    out[:, 0] = z.max(1)
    out[:, 1] = z.argmax(1) + offset
    return out


def merge_argmax(g):
    """Stock deferred_argmax merge: g [world, M, 2]; the first rank holding the largest maximum (torch.argmax)."""
    best = g[..., 0].argmax(0)
    return g[best, np.arange(g.shape[1]), 1].astype(np.int64)


def stock_gathered_token(l, v_real, f=cm.softcap):
    return cm.stock_token(l, v_real, f)


def stock_deferred_token(l, v_real, world, clamp=True, f=cm.softcap):
    """The stock deferral on pico's full shards: clamp=False is the image's padded-vocabulary width."""
    n = l.shape[1] // world
    y = f(l)
    g = np.stack([local_argmax(y[:, r * n:(r + 1) * n], shard_real(v_real, n, r) if clamp else n, r * n)
                  for r in range(world)])
    return merge_argmax(g)


class Shard:
    """One rank's screen tables (cert_math.quantize / tables of its rows)."""

    def __init__(self, W, r, n, v_real):
        self.r, self.n, self.offset = r, n, r * n
        self.W = W[r * n:(r + 1) * n]
        self.q, self.sc, self.D = cm.quantize(self.W)
        self.rt, self.nv = cm.tables(self.W, self.D)
        self.v = shard_real(v_real, n, r)

    def screen(self, x, s_err=None):
        err = None if s_err is None else s_err[:, self.offset:self.offset + self.n]
        return cm.screen(x, self.D, self.rt, self.nv, self.v, s_err=err)


def tp_step(x, shards, l, v_real, on=None, bad=None, s_err=None, f=cm.softcap, bounds=None):
    """One certified TP step. Returns dict(gathered=token via the stock gather of the certified shards,
    deferred=token via the stock deferral with the real-vocabulary limit, certified=[per rank], masks=[per rank],
    thresholds=[per rank], bufs=[per rank certified shard before the softcap]). bounds: per rank (lo, hi) in place
    of the screen (tests: intervals as tight as the argument allows)."""
    world = len(shards)
    on = [True] * world if on is None else on
    bad = [False] * world if bad is None else bad
    lts, hts, sends = [], [], []
    for r, sh in enumerate(shards):
        lo, hi = sh.screen(x, s_err) if bounds is None else bounds[r]
        lts.append(tile_max(lo))
        hts.append(tile_max(hi))
    for r in range(world):
        sends.append(send_th(lts[r], hts[r], on[r], bad[r]))
    recv = np.stack(sends)                       # the all-gather: every rank receives the same [world, M, 2]
    res = {"certified": [], "masks": [], "thresholds": [], "bufs": []}
    ys, locs = [], []
    for r, sh in enumerate(shards):
        mask, certified, thr = select_tp(hts[r], recv, on[r], bad[r])
        buf = cm.certified_buffer(l[:, sh.offset:sh.offset + sh.n], mask)
        y = f(buf)
        res["certified"].append(certified)
        res["masks"].append(mask)
        res["thresholds"].append(thr)
        res["bufs"].append(buf)
        ys.append(y)
        locs.append(local_argmax(y, sh.v, sh.offset))
    full = np.concatenate(ys, 1)
    full[:, v_real:] = -np.inf
    res["gathered"] = full.argmax(1)
    res["deferred"] = merge_argmax(np.stack(locs))
    return res
