#!/usr/bin/env python3
"""CPU emulation (numpy, float64) of attn_kernel.cuh's control flow and lane-level index math.

Emulates, per CTA / warp / lane: the work decomposition (linear id -> head, request, query block), the key-tile
range and per-warp skip, the bias tile computed as T = r x proj-slice with the mma C-fragment layout, its staging
in the per-warp shared buffer (odd rows shifted), the sheared 32-bit reads back into the S C-fragment layout,
masking and the online softmax. Compares against a direct dense reference. MMA arithmetic itself is plain matmul
here (fragment layouts of ldmatrix / mma for QK and PV are standard and not emulated).
"""
import numpy as np

BM, BN, TS = 64, 64, 88


def bf16(x):
    x = np.asarray(x, dtype=np.float32)
    u = x.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16) << 16
    return u.astype(np.uint32).view(np.float32)


def kernel(q, kfull, vfull, r, proj, reqs, heads, group, local, wl, scale):
    """q [T,H,D], kfull/vfull per request [sk, Hkv, D], r [T,H,16], proj [16,E]; reqs [(sq, sk)]."""
    T = q.shape[0]
    E = proj.shape[1]
    out = np.full((T, heads, q.shape[2]), np.nan)
    lse = np.full((heads, T), np.nan)
    cu = np.concatenate([[0], np.cumsum([s for s, _ in reqs])])
    n_req = len(reqs)
    tiles = -(-T // BM) + n_req
    grid = tiles * heads
    touched_pages = set()
    for bid in range(grid):
        lin = grid - 1 - bid
        head = lin % heads
        tile = lin // heads
        b = -1
        for i in range(n_req):
            ln = max(cu[i + 1] - cu[i], 0)
            nb = -(-ln // BM)
            if tile < nb:
                b, start, sq = i, cu[i], ln
                break
            tile -= nb
        if b < 0:
            continue
        sk = reqs[b][1]
        off = sk - sq
        m0 = tile * BM
        kvh = head // group
        lo_key = max(0, m0 + off - wl) if local else 0
        hi_key = min(sk, m0 + BM + off)
        nb_lo = lo_key // BN
        nb_hi = -(-hi_key // BN) if hi_key > lo_key else nb_lo
        K = kfull[b][:, kvh]
        V = vfull[b][:, kvh]
        for warp in range(4):
            wrow0 = m0 + warp * 16
            warp_rows = wrow0 < sq
            pw0 = wrow0 + off
            rows = np.arange(16) + wrow0
            valid = rows < sq
            tq = start + np.where(valid, rows, 0)
            Q = np.where(valid[:, None], q[tq, head], 0.0)
            R = np.where(valid[:, None], r[tq, head], 0.0)
            o = np.zeros((16, q.shape[2]))
            m = np.full(16, -np.inf)
            lsum = np.zeros(16)
            for nb in range(nb_hi - 1, nb_lo - 1, -1):
                n0 = nb * BN
                touched_pages.add((b, n0 // 128))
                # zero-filled tile
                js = n0 + np.arange(BN)
                ok = (js >= lo_key) & (js < sk)
                Kt = np.where(ok[:, None], K[np.clip(js, 0, sk - 1)], 0.0)
                Vt = np.where(ok[:, None], V[np.clip(js, 0, sk - 1)], 0.0)
                active = warp_rows and n0 <= pw0 + 15 and ((not local) or n0 + BN - 1 >= pw0 - wl)
                if not active:
                    continue
                s = Q @ Kt.T  # [16, 64]
                dbase = pw0 - n0 - (BN - 1)
                if dbase < E:
                    # T via the C fragment layout: lane (g, qd), ct -> T[g / g+8][8ct+2qd+e]
                    tw = np.full((16, TS), np.nan)
                    for lane in range(32):
                        g, qd = lane >> 2, lane & 3
                        par = g & 1
                        for ct in range(10):
                            d = dbase + 8 * ct + g  # this lane's B column (n = g)
                            # the mma output column c = 8ct + 2qd + e comes from the B column n = 2qd+e of the
                            # same ct block, i.e. d_c = dbase + 8ct + 2qd + e (lane g only feeds n = g)
                            for e in range(2):
                                c = 8 * ct + 2 * qd + e
                                dc = dbase + c
                                pc = proj[:, dc] if 0 <= dc < E else np.zeros(16)
                                for hh in range(2):
                                    a = g + 8 * hh
                                    tw[a, c + par] = bf16(np.float32(R[a] @ pc))
                    sb = np.full((16, BN), np.nan)
                    for lane in range(32):
                        g, qd = lane >> 2, lane & 3
                        par = g & 1
                        for nt in range(8):
                            for hh in range(2):
                                a = g + 8 * hh
                                s1 = a + (BN - 2) - 8 * nt - 2 * qd + par
                                assert s1 % 2 == 0 and 0 <= s1 and s1 + 1 < TS
                                lo, hi = tw[a, s1], tw[a, s1 + 1]
                                sb[a, 8 * nt + 2 * qd] = hi
                                sb[a, 8 * nt + 2 * qd + 1] = lo
                    assert not np.isnan(sb).any()
                    s = s * scale + sb
                else:
                    s = s * scale
                need_mask = (n0 + BN - 1 > pw0) or (local and n0 < pw0 + 15 - wl)
                pr = pw0 + np.arange(16)[:, None]
                bad = (js[None, :] > pr) | ((js[None, :] < pr - wl) if local else False)
                if need_mask:
                    s = np.where(bad, -np.inf, s)
                else:
                    assert not bad[valid].any(), "unmasked tile has invisible keys"
                mx = s.max(axis=1)
                mnew = np.maximum(m, mx)
                muse = np.where(mnew == -np.inf, 0.0, mnew)
                alpha = np.exp2((m - muse) * np.log2(np.e))
                p = np.exp2((s - muse[:, None]) * np.log2(np.e))
                m = mnew
                lsum = lsum * alpha + p.sum(axis=1)
                o = o * alpha[:, None] + bf16(p).astype(np.float64) @ Vt
            for a in range(16):
                if rows[a] < sq:
                    t = start + rows[a]
                    l = lsum[a]
                    out[t, head] = o[a] / l if l > 0 else 0.0
                    lse[head, t] = m[a] + np.log(l) if l > 0 else -np.inf
    return out, lse, touched_pages


def reference(q, kfull, vfull, r, proj, reqs, heads, group, local, wl, scale):
    T = q.shape[0]
    E = proj.shape[1]
    out = np.zeros((T, heads, q.shape[2]))
    lse = np.zeros((heads, T))
    t0 = 0
    for b, (sq, sk) in enumerate(reqs):
        off = sk - sq
        for h in range(heads):
            K, V = kfull[b][:, h // group], vfull[b][:, h // group]
            for i in range(sq):
                p = i + off
                j = np.arange(sk)
                d = p - j
                vis = (d >= 0) & ((d <= wl) if local else True)
                tb = bf16((r[t0 + i, h] @ proj).astype(np.float32)).astype(np.float64)
                bias = np.where((d >= 0) & (d < E), tb[np.clip(d, 0, E - 1)], 0.0)
                s = np.where(vis, q[t0 + i, h] @ K.T * scale + bias, -np.inf)
                mm = s.max()
                e = np.exp(s - mm)
                out[t0 + i, h] = e @ V / e.sum()
                lse[h, t0 + i] = mm + np.log(e.sum())
        t0 += sq
    return out, lse


def run(reqs, local, E, wl, heads=3, group=3, seed=0):
    rng = np.random.default_rng(seed)
    D = 128
    T = sum(s for s, _ in reqs)
    q = bf16(rng.normal(0, 1.5, (T, heads, D))).astype(np.float64)
    r = bf16(rng.normal(0, 2, (T, heads, 16))).astype(np.float64)
    proj = bf16(rng.normal(0, 1, (16, E))).astype(np.float64)
    kf = [bf16(rng.normal(0, 1.5, (sk, heads // group, D))).astype(np.float64) for _, sk in reqs]
    vf = [bf16(rng.normal(0, 1, (sk, heads // group, D))).astype(np.float64) for _, sk in reqs]
    o1, l1, pages = kernel(q, kf, vf, r, proj, reqs, heads, group, local, wl, D ** -0.5)
    o2, l2 = reference(q, kf, vf, r, proj, reqs, heads, group, local, wl, D ** -0.5)
    eo = np.abs(o1 - o2).max()
    el = np.abs(l1 - l2).max()
    for b, (sq, sk) in enumerate(reqs):
        for (bb, pg) in pages:
            if bb == b:
                assert pg * 128 < sk, f"page {pg} of request {b} holds no key < sk={sk}"
    print(f"reqs {reqs} local {local} E {E} wl {wl}: max|out| err {eo:.2e} lse err {el:.2e}")
    assert not np.isnan(o1).any() and eo < 2e-2 and el < 1e-3, "mismatch"


if __name__ == "__main__":
    # small window / extent so that SWA edges and the global "beyond extent" region are exercised cheaply
    run([(200, 200)], True, 128, 127)
    run([(150, 700), (0, 0), (70, 70), (3, 300)], True, 128, 127)
    run([(200, 200)], False, 128, -1)
    run([(150, 700), (0, 0), (70, 70), (3, 300)], False, 128, -1)
    run([(130, 1100)], True, 1024, 1023)
    print("emulation OK")
