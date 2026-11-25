# Usage (inside the SGLang image, overlay on PYTHONPATH): INKLING_VERIFY_CAP=fixed:8 python3 tools/gpu_check_fused_accept.py
"""GB10 check of inkling_vcap.fused_accept (INKLING_VERIFY_CAP_FUSED_ACCEPT). Run in the image with
PYTHONPATH=/work/overlay INKLING_VERIFY_CAP=fixed:8 (arms the hooks).

1. startup self-check (Triton primitives vs dense torch primitives);
2. race-kernel law at V = 201024: residual max(P - Q, 0) with mass spread over chunk/split boundaries;
3. law Monte Carlo through the real accept_draft_tokens (V = 3; chain / BV, capped / uncapped);
4. cold-L2 timing of the real accept_draft_tokens, stock vs fused, at bs 1 / 8 / 24, chain and BV."""
import statistics
import sys
import types

import torch

from sglang.srt.speculative.dspark_components import dspark_verify as dv
from inkling_vcap import fused_accept as fa
from inkling_vcap.config import parse_config
from inkling_vcap.patches import fused_accept_selfcheck
from inkling_vcap.state import STATE

dev = "cuda"
ok = True
err = fused_accept_selfcheck(dev)
print("1. selfcheck:", err, flush=True)
ok &= err is None
TP = STATE.fused_accept

# ---------------------------------------------------------------- 2. race law at full vocab
V = 201024
ids = torch.tensor([5, 2047, 2048, 70001, 140000, 201023], device=dev)
pt = torch.tensor([0.30, 0.25, 0.20, 0.15, 0.06, 0.04], device=dev)
qd = torch.tensor([0.10, 0.40, 0.05, 0.15, 0.20, 0.10], device=dev)
w = (pt - qd).clamp(min=0)
exact = w / w.sum()
bs, S, G = 64, 8, 7
tl2 = torch.full((bs * S, V), -1e4, device=dev)
tl2[:, ids] = torch.log(pt)
dl2 = torch.full((bs * G, V), float("-inf"), dtype=torch.bfloat16, device=dev)
dl2[:, ids] = torch.log(qd).to(torch.bfloat16)
qd_bf = torch.softmax(dl2[0, ids].float(), 0)  # the bf16-rounded draft law
exact = (pt - qd_bf).clamp(min=0)
exact = exact / exact.sum()
tt = torch.ones(bs, device=dev)
td = torch.ones(bs, device=dev)
mt, lt = TP.stats(tl2, tt, S, torch.ones(bs * S, dtype=torch.bool, device=dev))
md, ld = TP.stats(dl2, td, G, torch.ones(bs * G, dtype=torch.bool, device=dev))
st = (mt.view(bs, S), lt.view(bs, S), md.view(bs, G), ld.view(bs, G))
counts = torch.zeros(V, device=dev)
tau = torch.zeros(bs, dtype=torch.long, device=dev)
for _ in range(500):
    pick = TP.sample(tl2, dl2, st, tt, td, tau, torch.ones(bs, device=dev),
                     torch.ones(bs, dtype=torch.bool, device=dev))
    counts.index_add_(0, pick, torch.ones(bs, device=dev))
n = counts.sum()
emp = counts[ids] / n
race_err = float((emp - exact).abs().max())
stray = float(n - counts[ids].sum())
print(f"2. race law V=201024: {int(n)} samples, max abs err {race_err:.4f} (exact {exact.tolist()}), "
      f"off-support picks {stray:.0f}", flush=True)
ok &= race_err < 0.012 and stray == 0

# 2b. poisoned (all -inf) draft rows past the cut with a valid target: L = 1 must give P_0 samples
dl2.fill_(float("-inf"))
L1 = torch.ones(bs, dtype=torch.long, device=dev)
cand = torch.zeros(bs, S, dtype=torch.long, device=dev)
cnt = torch.zeros(V, device=dev)
for _ in range(200):
    for bvv in (False, True):
        c_, b_, _t = fa.accept_core(TP, candidates=cand, target_logits=tl2, draft_logits=dl2.view(bs, G, V),
                                    t_temps=tt, d_temps=td, live_lens=L1, block_verification=bvv,
                                    coins=torch.rand(bs, G, device=dev))
        assert int(c_.max()) == 0
        cnt.index_add_(0, b_, torch.ones(bs, device=dev))
emp0 = cnt[ids] / cnt.sum()
e0 = float((emp0 - pt / pt.sum()).abs().max())
stray0 = float(cnt.sum() - cnt[ids].sum())
print(f"2b. L=1 with all -inf draft rows: max abs err vs P_0 {e0:.4f}, off-support {stray0:.0f}", flush=True)
ok &= e0 < 0.012 and stray0 == 0
del tl2, dl2

# ---------------------------------------------------------------- 3. law MC through accept_draft_tokens
P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]], dtype=torch.float64, device=dev)
Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]], dtype=torch.float64, device=dev)
GR = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64, device=dev)
N, GAM, T, TD = 300_000, 3, 0.8, 0.56


def setup(n, cap="fixed:4"):
    STATE.config = parse_config({"INKLING_VERIFY_CAP": cap})
    STATE.allocate(stride=GAM + 1 if n == N else 8, capacity=n, device=dev)
    STATE.enabled = True
    STATE.capping = True
    STATE._src_key = ("dyn", 0, n)


def mc(bv, capped, fused, seed, fastprobs=False):
    setup(N)
    STATE.fused_accept = TP if fused else None
    STATE.use_fastprobs = fastprobs
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = torch.full((N, 3), -1, dtype=torch.long, device=dev)
    filled = torch.zeros(N, dtype=torch.long, device=dev)
    anchor = torch.zeros(N, dtype=torch.long, device=dev)
    si = types.SimpleNamespace(temperatures=torch.full((N, 1), T, device=dev), is_all_greedy=False,
                               is_any_greedy=False, need_top_k_sampling=False, need_top_p_sampling=False)
    acc, steps = 0, 0
    while bool((filled < 3).any()):
        prev, drafts, qs = anchor, [], []
        for _ in range(GAM):
            q = Q[prev]
            d = torch.multinomial(q, 1, generator=gen).squeeze(1)
            drafts.append(d)
            qs.append(q)
            prev = d
        drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1)
        L = 1 + (torch.cumprod(qs.max(-1).values, 1) >= 0.505).sum(1)
        if not capped:
            L = torch.full_like(L, GAM + 1)
        row_tok = torch.cat([anchor[:, None], drafts], 1)
        ps = P[row_tok]
        dead = torch.arange(GAM + 1, device=dev)[None, :] >= L[:, None]
        ps = torch.where(dead[..., None], GR.expand_as(ps), ps)
        STATE.live_lens[:N].copy_(L)
        block = types.SimpleNamespace(
            greedy_draft=False, corrected_logits=(TD * torch.log(qs)).float().contiguous(),
            temperatures=torch.full((N,), TD, device=dev), draft_tokens=drafts.contiguous(),
            greedy_mask=torch.zeros(N, dtype=torch.bool, device=dev), vocab_shard=None)
        correct, bonus, _ = dv.accept_draft_tokens(
            candidates=row_tok.contiguous(),
            target_logits=(T * torch.log(ps)).float().reshape(N * (GAM + 1), 3),
            draft_block=block, sampling_info=si, draft_input=None, gamma=GAM,
            verify_num_draft_tokens=GAM + 1, block_verification=bv)
        correct, bonus = correct.long(), bonus.long()
        assert bool((correct <= L - 1).all())
        emitted = torch.cat([drafts, bonus[:, None]], 1)
        n_emit = correct + 1
        active = filled < 3
        acc += int(correct[active].sum())
        steps += int(active.sum())
        for k in range(GAM + 1):
            take = active & (k < n_emit) & (filled + k < 3)
            tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
            idx = (filled + k).clamp(max=2)
            out[take, idx[take]] = tok[take]
        filled = torch.where(active, filled + n_emit, filled)
        anchor = torch.where(active, bonus, anchor)
    emp = torch.zeros(3, 3, 3, dtype=torch.float64, device=dev)
    emp.index_put_((out[:, 0], out[:, 1], out[:, 2]), torch.ones(N, dtype=torch.float64, device=dev),
                   accumulate=True)
    emp /= N
    ex = torch.einsum("a,ab,bc->abc", P[0], P, P)
    return float((emp - ex).abs().max()), acc / steps


for i, (bv, capped) in enumerate(((False, False), (True, False), (False, True), (True, True))):
    calls0 = STATE.fused_accept_calls
    e, a = mc(bv, capped, True, 10 + i)
    used = STATE.fused_accept_calls - calls0
    good = e < 3.5e-3 and used > 0
    ok &= good
    print(f"3. fused bv={bv} capped={capped}: err {e:.2e} accepted/step {a:.4f} fused calls {used} "
          f"{'PASS' if good else 'FAIL'}", flush=True)

# combined stacks (draft at tau x T is already the setup: TD = 0.7 x T):
# FASTPROBS q + BV dead-draft mask (the bs < MIN_BS path), and FASTPROBS + fused + BV
for name, kw in (("fastprobs+bv-mask", dict(bv=True, capped=True, fused=False, fastprobs=True)),
                 ("fastprobs+fused+bv", dict(bv=True, capped=True, fused=True, fastprobs=True))):
    m0, f0 = STATE.bv_masked, STATE.fastprobs_calls
    e, a = mc(seed=40 + len(name), **kw)
    good = e < 3.5e-3
    ok &= good
    print(f"3c. {name}: err {e:.2e} accepted/step {a:.4f} masked {STATE.bv_masked - m0} "
          f"fastprobs {STATE.fastprobs_calls - f0} {'PASS' if good else 'FAIL'}", flush=True)
STATE.use_fastprobs = False

# ---------------------------------------------------------------- 4. timing
flush_buf = torch.empty(96 * 1024 * 1024 // 4, dtype=torch.float32, device=dev)
hot = torch.randperm(V, generator=torch.Generator().manual_seed(1))[:65536].to(dev)


def timeit(fn, reps=25):
    ts = []
    for _ in range(reps):
        flush_buf.sum()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    return statistics.median(ts)


print("4. bs | mode | stock us | fused us")
for bsz in (1, 8, 24):
    setup(bsz, cap="fixed:8")
    STATE.live_lens[:bsz] = 8
    g = torch.Generator(device=dev).manual_seed(bsz)
    tlog = torch.randn((bsz * 8, V), generator=g, device=dev) * 4
    dlog = torch.full((bsz, 7, V), float("-inf"), dtype=torch.bfloat16, device=dev)
    dlog[:, :, hot] = (tlog.view(bsz, 8, V)[:, :7, hot] + torch.randn((bsz, 7, 65536), generator=g,
                                                                        device=dev)).to(torch.bfloat16)
    drafts = torch.argmax(dlog.float(), -1)
    cand = torch.cat([torch.zeros(bsz, 1, dtype=torch.long, device=dev), drafts], 1)
    si = types.SimpleNamespace(temperatures=torch.full((bsz, 1), 0.8, device=dev), is_all_greedy=False,
                               is_any_greedy=False, need_top_k_sampling=False, need_top_p_sampling=False)
    block = types.SimpleNamespace(greedy_draft=False, corrected_logits=dlog,
                                  temperatures=torch.full((bsz,), 0.56, device=dev), draft_tokens=drafts,
                                  greedy_mask=torch.zeros(bsz, dtype=torch.bool, device=dev), vocab_shard=None)
    for bv in (False, True):
        def run():
            dv.accept_draft_tokens(candidates=cand, target_logits=tlog, draft_block=block, sampling_info=si,
                                   draft_input=None, gamma=7, verify_num_draft_tokens=8, block_verification=bv)
        STATE.fused_accept = None
        t_stock = timeit(run)
        STATE.fused_accept = TP
        t_fused = timeit(run)
        print(f"   {bsz} | {'BV' if bv else 'chain'} | {t_stock:.0f} | {t_fused:.0f}", flush=True)
    del tlog, dlog
    torch.cuda.empty_cache()

print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
