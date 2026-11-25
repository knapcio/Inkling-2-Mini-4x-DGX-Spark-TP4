#!/usr/bin/env python3
"""GPU check + microbench of inkling_dfaccept (sparse-q fused sampled accept for DFLASH). No server.

  cd overlay && TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/gpu_check_dfaccept.py

1. Primitives vs the dense torch reference (Q densified from the sparse rows): row statistics, residual masses
   (block verification) within FP32 tolerance; accept lengths equal for the same coins (chain and block
   verification, with and without a verify cap).
2. Bonus law (Monte Carlo): at V = 512 with padded -inf lanes, 2e5 draws of the race against the exact residual
   distribution max(s P - Q, 0) / sum, for use_q on / off: max |freq - prob| and a chi-square p-value; the rsfix
   fallback (no positive residual -> the last positive-target lane).
3. Timing at the real vocabulary (V = 201024, K = 16, 8 rows a request), bs 8 and 24: the stock path (dense q scatter +
   accept_sampling from sglang, as DFlashWorkerV2._selector_sampling_accept runs it) vs the sparse core, CUDA events,
   median of 20 eager calls.
"""
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import torch  # noqa: E402

FAILED = []


def log(*a):
    print(*a, flush=True)


def make(bs, G, V, K, dev, g, pad=20):
    S = G + 1
    tlog = (torch.randn(bs * S, V, device=dev, generator=g) * 3).to(torch.bfloat16).float()
    tlog[:, V - pad:] = float("-inf")
    cand = torch.stack([torch.randperm(V - pad, device=dev, generator=g)[:K] for _ in range(bs * G)]).view(bs, G, K)
    q = torch.softmax(torch.randn(bs, G, K, device=dev, generator=g) * 2, -1)
    pick = torch.multinomial(q.view(-1, K), 1, generator=g).view(bs, G)
    drafts = cand.gather(2, pick[..., None]).squeeze(2)
    candidates = torch.cat([torch.zeros(bs, 1, dtype=torch.int64, device=dev), drafts], 1)
    temps = torch.rand(bs, device=dev, generator=g) * 0.8 + 0.4
    return candidates, tlog, cand, q, temps


def main():
    from inkling_dfaccept import core

    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(7)
    log("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"))
    tp, rp = core.TritonPrims(), core.TorchPrims()
    # ---- 1. primitives and accept lengths
    bs, G, V, K = 12, 7, 4096, 16
    S = G + 1
    candidates, tlog, cand, q, temps = make(bs, G, V, K, dev, g)
    need = torch.ones(bs * S, dtype=torch.bool, device=dev)
    mt, lt = tp.stats(tlog, temps, S, need)
    rmt, rlt = rp.stats(tlog, temps, S, need)
    ok = torch.allclose(mt, rmt) and torch.allclose(lt, rlt, atol=1e-5)
    pp = torch.rand(bs, G, device=dev, generator=g)
    nd = torch.ones(bs, G, dtype=torch.bool, device=dev)
    nd[:, G - 1] = False  # as production: level G (no slot G) never needs a mass (level < g <= G)
    st = (mt.view(bs, S), lt.view(bs, S))
    m1 = tp.masses(tlog, st, temps, cand, q, pp, nd)
    m2 = rp.masses(tlog, st, temps, cand, q, pp, nd)
    ok &= torch.allclose(m1, m2, atol=1e-5, rtol=1e-4)
    log(f"stats / masses vs dense reference: {'ok' if ok else 'DIFFER'} (max mass diff {(m1 - m2).abs().max():.2e})")
    if not ok:
        FAILED.append("prims")
    for bv in (False, True):
        for live in (None, torch.randint(1, 9, (bs,), device=dev, generator=g)):
            coins = torch.rand(bs, G, device=dev, generator=g)
            a = core.accept_core_sparse(tp, candidates=candidates, target_logits=tlog, cand_ids=cand, q_rows=q,
                                        t_temps=temps, live_lens=live, block_verification=bv, coins=coins)
            b = core.accept_core_sparse(rp, candidates=candidates, target_logits=tlog, cand_ids=cand, q_rows=q,
                                        t_temps=temps, live_lens=live, block_verification=bv, coins=coins)
            same = torch.equal(a[0], b[0])
            log(f"accept lengths (bv={bv}, cap={'on' if live is not None else 'off'}): {'equal' if same else 'DIFFER'}")
            if not same:
                FAILED.append(f"tau bv={bv}")
    # ---- 2. bonus law
    V2, N = 512, 200000
    _, tl2, cand2, q2, t2 = make(1, G, V2, K, dev, g, pad=40)
    st2 = rp.stats(tl2, t2, S, torch.ones(S, dtype=torch.bool, device=dev))
    st2 = (st2[0].view(1, S), st2[1].view(1, S))
    P, Q = rp.dense(tl2, st2, t2, cand2, q2)
    for use_q, scale, row in ((True, 1.0, 2), (False, 1.0, 7), (True, 0.6, 3)):
        w = (scale * P[0, row] - (Q[0, min(row, G - 1)] if use_q else 0)).clamp(min=0)
        prob = w / w.sum()
        reps = 2000
        counts = torch.zeros(V2, device=dev)
        b = N // reps
        tl_rep = tl2.view(1, S, V2).expand(b, S, V2).reshape(b * S, V2).contiguous()
        st_rep = (st2[0].expand(b, S).contiguous(), st2[1].expand(b, S).contiguous())
        for _ in range(reps):
            x = tp.sample(tl_rep, st_rep, t2.expand(b).contiguous(), cand2.expand(b, G, K).contiguous(),
                          q2.expand(b, G, K).contiguous(), torch.full((b,), row, device=dev),
                          torch.full((b,), scale, device=dev), torch.full((b,), use_q, device=dev))
            counts += torch.bincount(x, minlength=V2).float()
        freq = counts / counts.sum()
        dev_max = float((freq - prob).abs().max())
        exp_ = prob * counts.sum()
        m = exp_ > 5
        chi = float((((counts - exp_) ** 2) / exp_.clamp(min=1e-9))[m].sum())
        dof = int(m.sum()) - 1
        z = (chi - dof) / (2 * dof) ** 0.5
        good = dev_max < 5e-3 and abs(z) < 4 and float(counts[prob == 0].sum()) == 0
        log(f"bonus law row {row} use_q={use_q} s={scale}: max|freq-prob| {dev_max:.2e}, chi2 z {z:+.2f}, "
            f"zero-prob hits {int(counts[prob == 0].sum())}: {'ok' if good else 'FAIL'}")
        if not good:
            FAILED.append(f"law row {row}")
    # rsfix fallback: a target row whose residual has no positive lane
    tl3 = torch.full((S, V2), float("-inf"), device=dev)
    tl3[:, 10] = 0.0
    tl3[:, 300] = 0.0
    cand3 = torch.zeros(1, G, K, dtype=torch.int64, device=dev)
    cand3[..., 0], cand3[..., 1] = 10, 300
    q3 = torch.zeros(1, G, K, device=dev)
    q3[..., 0] = q3[..., 1] = 0.5
    st3 = rp.stats(tl3, torch.ones(1, device=dev), S, torch.ones(S, dtype=torch.bool, device=dev))
    st3 = (st3[0].view(1, S), st3[1].view(1, S))
    x = tp.sample(tl3, st3, torch.ones(1, device=dev), cand3, q3, torch.tensor([2], device=dev),
                  torch.ones(1, device=dev), torch.tensor([True], device=dev))
    log(f"no positive residual -> last positive-target lane: {int(x)} ({'ok' if int(x) == 300 else 'FAIL'})")
    if int(x) != 300:
        FAILED.append("fallback")
    # ---- 3. timing vs the stock dense path
    try:
        from sglang.kernels.ops.speculative.dspark.dspark_accept import accept_sampling
        from types import SimpleNamespace
    except Exception as exc:  # noqa: BLE001
        log(f"stock accept_sampling unavailable: {exc!r}")
        accept_sampling = None
    for bs in (8, 24):
        V = 201024
        candidates, tlog, cand, q, temps = make(bs, G, V, K, dev, g, pad=966)
        tlog = tlog.contiguous()
        coins = torch.rand(bs, G, device=dev, generator=g)
        ts = {"stock": [], "sparse": []}
        dense = torch.zeros((bs, G, V), dtype=torch.float32, device=dev)
        si = SimpleNamespace(need_top_k_sampling=False, need_top_p_sampling=False, need_min_p_sampling=False,
                             temperatures=temps.view(-1, 1), is_all_greedy=False)
        di = SimpleNamespace(max_top_k=None, uniform_top_k_value=None)
        for _ in range(22):
            if accept_sampling is not None:
                e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                e0.record()
                dense.scatter_(-1, cand, q)
                accept_sampling(candidates=candidates, target_logits=tlog, draft_probs=dense, sampling_info=si,
                                draft_input=di, gamma=G, verify_num_draft_tokens=G + 1)
                dense.scatter_(-1, cand, 0.0)
                e1.record(); torch.cuda.synchronize()
                ts["stock"].append(e0.elapsed_time(e1))
            e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
            e0.record()
            core.accept_core_sparse(tp, candidates=candidates, target_logits=tlog, cand_ids=cand, q_rows=q,
                                    t_temps=temps, live_lens=None, block_verification=False, coins=coins)
            e1.record(); torch.cuda.synchronize()
            ts["sparse"].append(e0.elapsed_time(e1))
        st_ = statistics.median(ts["stock"][2:]) if ts["stock"] else float("nan")
        log(f"bs {bs:2}: stock dense accept {st_:.3f} ms, sparse fused {statistics.median(ts['sparse'][2:]):.3f} ms "
            f"(eager, FP32 target logits [bs*8, 201024])")
    # ---- 4. integration law: stock accept_sampling (dense q) vs the fused core, joint law of (accept length, bonus)
    if accept_sampling is not None:
        Vs, B, reps = 64, 4000, 25
        c1, t1, k1, q1, tp1 = make(1, G, Vs, 8, dev, g, pad=8)
        for bv in (False, True):
            hist = {"stock": torch.zeros(G + 1, Vs, device=dev), "fused": torch.zeros(G + 1, Vs, device=dev)}
            cand_b = c1.expand(B, G + 1).contiguous()
            tl_b = t1.view(1, G + 1, Vs).expand(B, G + 1, Vs).reshape(B * (G + 1), Vs).contiguous()
            k_b, q_b = k1.expand(B, G, 8).contiguous(), q1.expand(B, G, 8).contiguous()
            tmp = tp1.expand(B).contiguous()
            si = SimpleNamespace(need_top_k_sampling=False, need_top_p_sampling=False, need_min_p_sampling=False,
                                 temperatures=tmp.view(-1, 1), is_all_greedy=False)
            di = SimpleNamespace(max_top_k=None, uniform_top_k_value=None)
            dense = torch.zeros((B, G, Vs), dtype=torch.float32, device=dev).scatter_(-1, k_b, q_b)
            for _ in range(reps):
                n1, b1, _c = accept_sampling(candidates=cand_b, target_logits=tl_b, draft_probs=dense, sampling_info=si,
                                             draft_input=di, gamma=G, verify_num_draft_tokens=G + 1,
                                             block_verification=bv)
                hist["stock"].index_put_((n1.long(), b1.long()), torch.ones(B, device=dev), accumulate=True)
                coins = torch.rand(B, G, device=dev)
                n2, b2, _c = core.accept_core_sparse(tp, candidates=cand_b, target_logits=tl_b, cand_ids=k_b,
                                                     q_rows=q_b, t_temps=tmp, live_lens=None, block_verification=bv,
                                                     coins=coins)
                hist["fused"].index_put_((n2.long(), b2.long()), torch.ones(B, device=dev), accumulate=True)
            ps, pf = hist["stock"] / hist["stock"].sum(), hist["fused"] / hist["fused"].sum()
            tv = 0.5 * float((ps - pf).abs().sum())
            n = B * reps
            # two-sample: TV of two n-draw empirical laws of the same law is ~ sqrt(cells / n) / 2 at most
            cells = int(((ps + pf) > 0).sum())
            bound = 2.0 * (cells / n) ** 0.5
            ok = tv < bound
            log(f"joint law (accept length, bonus) stock vs fused, bv={bv}: TV {tv:.4f} over {cells} cells, "
                f"{n} draws each (bound {bound:.4f}): {'ok' if ok else 'FAIL'}")
            if not ok:
                FAILED.append(f"joint law bv={bv}")
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
