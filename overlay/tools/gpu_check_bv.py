# Usage (inside the SGLang image, overlay on PYTHONPATH): INKLING_VERIFY_CAP=fixed:4 python3 tools/gpu_check_bv.py
"""GPU law test of block verification under the verify cap, through the REAL engine path:
dspark_verify.accept_draft_tokens (vcap-wrapped) -> _draft_probs (vcap-wrapped mask) ->
AcceptSampling -> the stock Triton chain_speculative_sampling kernel (BLOCK_VERIFICATION).

Run in the image with PYTHONPATH=/work/overlay INKLING_VERIFY_CAP=fixed:4 (arms the hooks).
Same Markov setup as tests/vcap/test_block_verify.py: 3-token target P, draft Q, garbage dead
rows, L from the draft max-probability rule. Arms: bv-mask, bv-stock (stock kernel + cap),
chain (+cap), bv-full (no cap), leak (negative control: bonus from the dead-row final sample).
Prints max abs error of the first-3-token joint law and accepted drafts per step."""
import os
import sys
import types

import torch

from sglang.srt.speculative.dspark_components import dspark_verify as dv
from inkling_vcap.config import parse_config
from inkling_vcap.state import STATE

dev = "cuda"
P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]], dtype=torch.float64, device=dev)
Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]], dtype=torch.float64, device=dev)
G = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64, device=dev)
GAMMA, V = 3, 3
N = int(os.environ.get("BV_N", "300000"))

assert getattr(dv.accept_draft_tokens, "__inkling_vcap__", False), "vcap accept wrapper missing"
assert getattr(dv._draft_probs, "__inkling_vcap__", False), "vcap _draft_probs wrapper missing"


def setup(bv_mode):
    STATE.config = parse_config({"INKLING_VERIFY_CAP": "fixed:4", "INKLING_VERIFY_CAP_BV": bv_mode})
    STATE.allocate(stride=GAMMA + 1, capacity=N, device=dev)
    STATE.enabled = True
    STATE.capping = True
    STATE._src_key = ("dyn", 0, N)


def simulate(arm, threshold=0.505, seed=0, out_len=3):
    setup("stock" if arm in ("bv-stock", "leak") else "mask")
    torch.manual_seed(seed)
    gen = torch.Generator(device=dev).manual_seed(seed)
    out = torch.full((N, out_len), -1, dtype=torch.long, device=dev)
    filled = torch.zeros(N, dtype=torch.long, device=dev)
    anchor = torch.zeros(N, dtype=torch.long, device=dev)
    rows = torch.arange(N, device=dev)
    acc_sum, steps = 0, 0
    si = types.SimpleNamespace(
        temperatures=torch.ones(N, 1, device=dev), is_all_greedy=False, is_any_greedy=False,
        need_top_k_sampling=False, need_top_p_sampling=False)
    while bool((filled < out_len).any()):
        prev, drafts, qs = anchor, [], []
        for _ in range(GAMMA):
            q = Q[prev]
            d = torch.multinomial(q, 1, generator=gen).squeeze(1)
            drafts.append(d)
            qs.append(q)
            prev = d
        drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1)
        L = 1 + (torch.cumprod(qs.max(-1).values, 1) >= threshold).sum(1)
        if arm == "bv-full":
            L = torch.full_like(L, GAMMA + 1)
        row_tok = torch.cat([anchor[:, None], drafts], 1)
        ps = P[row_tok]
        dead = torch.arange(GAMMA + 1, device=dev)[None, :] >= L[:, None]
        ps = torch.where(dead[..., None], G.expand_as(ps), ps)
        STATE.live_lens[:N].copy_(L)
        candidates = row_tok.contiguous()
        block = types.SimpleNamespace(
            greedy_draft=False, corrected_logits=torch.log(qs).float().contiguous(),
            temperatures=torch.ones(N, device=dev), draft_tokens=drafts.contiguous(),
            greedy_mask=torch.zeros(N, dtype=torch.bool, device=dev), vocab_shard=None)
        correct, bonus, trim = dv.accept_draft_tokens(
            candidates=candidates, target_logits=torch.log(ps).float().reshape(N * (GAMMA + 1), V),
            draft_block=block, sampling_info=si, draft_input=None, gamma=GAMMA,
            verify_num_draft_tokens=GAMMA + 1,
            block_verification=arm != "chain")
        correct = correct.long()
        bonus = bonus.long()
        assert bool((correct <= L - 1).all())
        if arm == "leak":
            # negative control: on a trim, emit a fresh draw from the first dead row instead
            fresh = torch.multinomial(ps[rows, L.clamp(max=GAMMA)], 1, generator=gen).squeeze(1)
            bonus = torch.where(trim > 0, fresh, bonus)
        emitted = torch.cat([drafts, bonus[:, None]], 1)
        n_emit = correct + 1
        active = filled < out_len
        acc_sum += int(correct[active].sum())
        steps += int(active.sum())
        for k in range(GAMMA + 1):
            take = active & (k < n_emit) & (filled + k < out_len)
            tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
            idx = (filled + k).clamp(max=out_len - 1)
            out[take, idx[take]] = tok[take]
        filled = torch.where(active, filled + n_emit, filled)
        anchor = torch.where(active, bonus, anchor)
    emp = torch.zeros(V, V, V, dtype=torch.float64, device=dev)
    emp.index_put_((out[:, 0], out[:, 1], out[:, 2]),
                   torch.ones(N, dtype=torch.float64, device=dev), accumulate=True)
    emp /= N
    exact = torch.einsum("a,ab,bc->abc", P[0], P, P)
    return float((emp - exact).abs().max()), acc_sum / steps


ok = True
for arm, seed, limit in (("bv-full", 0, ("lt", 3.5e-3)), ("bv-mask", 1, ("lt", 3.5e-3)),
                         ("bv-mask", 2, ("lt", 3.5e-3)), ("bv-stock", 3, ("lt", 3.5e-3)),
                         ("chain", 4, ("lt", 3.5e-3)), ("leak", 5, ("gt", 1e-2))):
    err, acc = simulate(arm, seed=seed)
    good = err < limit[1] if limit[0] == "lt" else err > limit[1]
    ok &= good
    print(f"{arm:9s} seed {seed}: max abs err {err:.2e} accepted/step {acc:.4f} "
          f"masked_calls {STATE.bv_masked} {'PASS' if good else 'FAIL'}", flush=True)
print("ALL PASS" if ok else "SOME FAILED")
sys.exit(0 if ok else 1)
