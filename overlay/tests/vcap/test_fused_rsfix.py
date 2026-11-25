"""inkling_vcap.fused_accept, v2.7.3 race fix (SGLang PR #51681 pattern), CPU side.

The Triton race kernel cannot run here. Its noise transform and key rule are mirrored in
FP32 torch below, line for line, and compared with the v2.7.2 transform; the dense
TorchPrims fallback is tested directly. tools/gpu_check_fused_rsfix.py runs the kernel.

    python -m pytest -q tests/vcap/test_fused_rsfix.py   (from overlay/)
"""
import math
import os
import sys

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inkling_vcap import fused_accept as fa  # noqa: E402

F32 = torch.float32
S32 = 2.3283064365386963e-10  # 2^-32
S64 = 5.421010862427522e-20  # 2^-64
PR_WORD = 0xFFFFFF9F  # the word of SGLang PR #51681's reproducer (rounds to 2^32 in FP32)


def noise_v272(r0, r1):
    """v2.7.2 _race_kernel transform, FP32."""
    hi = r0.to(F32) * S32
    lo = (r1.to(F32) + 0.5) * S64
    vv = torch.where(hi > 0, hi, lo)
    return torch.where(vv < 1e-3, vv + 0.5 * vv * vv, -torch.log(1.0 - vv))


def noise_v273(r0, r1):
    """v2.7.3 _race_kernel transform, FP32 (words as int64 holding uint32 values)."""
    upper = r0 >= 0x80000000
    f0 = torch.where(upper, 0xFFFFFFFF - r0, r0)
    f1 = torch.where(upper, 0xFFFFFFFF - r1, r1)
    small = f0.to(F32) * S32 + (f1.to(F32) + 0.5) * S64
    return torch.where(upper, -torch.log(small), -torch.log1p(-small))


def key_v272(w, e):
    return torch.where(w > 0, torch.log(w) - torch.log(e), torch.tensor(-math.inf))


def key_v273(w, e):
    return torch.where(w > 0, torch.clamp(torch.log(w) - torch.log(e), min=-3.0e38),
                       torch.tensor(-math.inf))


def race(keys, pv, V):
    """One split covering the row: the kernel's selection plus the v2.7.3 fallback."""
    k = keys.max()
    pick = int(torch.nonzero(keys == k)[0]) if bool(k > -math.inf) else V - 1
    pos = torch.nonzero(pv > 0)
    last = int(pos[-1]) if pos.numel() else -1
    return pick, last


def test_endpoint_words_give_finite_noise():
    top = torch.arange(0xFFFFFF80, 0x100000000, dtype=torch.int64)  # every word that rounds up
    r1 = torch.randint(0, 2**32, top.shape, generator=torch.Generator().manual_seed(0), dtype=torch.int64)
    assert bool(torch.isinf(noise_v272(top, r1)).all())  # the v2.7.2 bug
    e = noise_v273(top, r1)
    assert bool(torch.isfinite(e).all()) and bool((e > 0).all())
    # PR word: E = -log(1 - V) with 1 - V = (0x60 + (~r1 + 0.5) 2^-32) 2^-32 ~ 2.2e-8 -> E ~ 17.6
    e_pr = noise_v273(torch.tensor([PR_WORD]), torch.tensor([0]))
    assert abs(float(e_pr) - (-math.log((0x60 + (0xFFFFFFFF + 0.5) * 2**-32) * 2**-32))) < 1e-5


def test_noise_matches_exact_exponential_transform():
    g = torch.Generator().manual_seed(1)
    r0 = torch.randint(0, 2**32, (200_000,), generator=g, dtype=torch.int64)
    r1 = torch.randint(0, 2**32, (200_000,), generator=g, dtype=torch.int64)
    r0[:8] = torch.tensor([0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFF7F, 0xFFFFFF80, PR_WORD, 0xFFFFFFFF])
    u = (r0.double() + (r1.double() + 0.5) * 2.0 ** -32) * 2.0 ** -32
    exact = -torch.log1p(-u)
    upper = r0 >= 0x80000000
    exact[upper] = -torch.log(((0xFFFFFFFF - r0[upper]).double()
                               + (0xFFFFFFFF - r1[upper] + 0.5).double() * 2.0 ** -32) * 2.0 ** -32)
    rel = ((noise_v273(r0, r1).double() - exact).abs() / exact).max()
    assert float(rel) < 1e-6


def test_one_hot_residual_at_an_endpoint_word():
    V, lane = 201024, 123456
    w = torch.zeros(V, dtype=F32)
    w[lane] = 0.3
    pv = torch.zeros(V, dtype=F32)
    pv[[lane, 7]] = torch.tensor([0.6, 0.4])
    r0 = torch.randint(0, 0xFFFFFF80, (V,), generator=torch.Generator().manual_seed(2), dtype=torch.int64)
    r1 = torch.zeros(V, dtype=torch.int64)
    r0[lane] = PR_WORD
    pick_old, _ = race(key_v272(w, noise_v272(r0, r1)), pv, V)
    pick_new, last = race(key_v273(w, noise_v273(r0, r1)), pv, V)
    assert pick_old == V - 1  # v2.7.2 emits the padded id 201,023
    assert pick_new == lane and last == lane


def test_key_floor_keeps_positive_mass_ahead():
    # A raw -inf key on a positive lane (infinite noise, or log(w) flushed on the GPU):
    # the v2.7.2 rule loses the lane to the zero-mass lanes, the floor keeps it ahead.
    w = torch.tensor([0.0, 0.5, 0.0])
    e = torch.tensor([1.0, math.inf, 1.0])
    old, new = key_v272(w, e), key_v273(w, e)
    assert bool(torch.isinf(old).all())  # every key -inf: v2.7.2 falls back to V - 1
    assert bool(torch.isfinite(new[1])) and int(new.argmax()) == 1


def _prims_case(target_rows, draft_rows):
    tl2 = torch.tensor(target_rows, dtype=F32)
    dl2 = torch.tensor(draft_rows, dtype=F32)
    bs = tl2.shape[0]
    tt = torch.ones(bs)
    rp = fa.TorchPrims(torch.Generator().manual_seed(3))
    mt, lt = rp.stats(tl2, tt, 1, None)
    md, ld = rp.stats(dl2, tt, 1, None)
    st = (mt.view(bs, 1), lt.view(bs, 1), md.view(bs, 1), ld.view(bs, 1))
    return rp, tl2, dl2, st, tt


def test_torch_reference_no_positive_residual_emits_last_target_lane():
    inf = -math.inf
    row = [inf, 0.0, inf, inf, -1.0, inf, inf, -0.5, inf, inf]  # positive at 1, 4, 7; 9 = "padded"
    rp, tl2, dl2, st, tt = _prims_case([row, row], [row, row])  # P == Q: zero residual
    tau = torch.zeros(2, dtype=torch.long)
    for _ in range(20):
        pick = rp.sample(tl2, dl2.view(2, 1, -1).reshape(2, -1), st, tt, tt, tau,
                         torch.ones(2), torch.ones(2, dtype=torch.bool))
        assert pick.tolist() == [7, 7]  # v2.7.2: 9 (V - 1, a zero-mass lane)


def test_torch_reference_row_without_positive_target_keeps_v_minus_1():
    inf = -math.inf
    row = [inf] * 6  # not a distribution: the only case left at V - 1
    rp, tl2, dl2, st, tt = _prims_case([row], [row])
    pick = rp.sample(tl2, dl2, st, tt, tt, torch.zeros(1, dtype=torch.long), torch.ones(1),
                     torch.ones(1, dtype=torch.bool))
    assert pick.tolist() == [5]


def test_torch_reference_law_unchanged_with_positive_residual():
    inf = -math.inf
    t = [inf, 0.0, 0.0, inf, 0.0, inf]
    d = [inf, 0.0, inf, inf, inf, inf]
    rp, tl2, dl2, st, tt = _prims_case([t] * 4000, [d] * 4000)
    pick = rp.sample(tl2, dl2, st, tt, tt, torch.zeros(4000, dtype=torch.long),
                     torch.ones(4000), torch.ones(4000, dtype=torch.bool))
    assert set(pick.tolist()) <= {2, 4}  # residual = P - Q is 1/3 on lanes 2 and 4
    assert abs(float((pick == 2).float().mean()) - 0.5) < 0.04


@pytest.mark.skipif(fa.triton is not None, reason="source needles only matter without a GPU run")
def test_kernel_source_has_the_fix():
    src = open(fa.__file__, encoding="utf-8").read()
    for needle in ("upper = w0 >= 0x80000000", "f0 = tl.where(upper, ~w0, w0)",
                   "-libdevice.log1p(-small)", "tl.maximum(tl.log(w) - tl.log(e), -3.0e38)",
                   "tl.store(lastp_ptr + b * num_splits + split, tl.max(last_p, axis=0))",
                   "dead = (keys.amax(dim=1) == float(\"-inf\")) & (fallback >= 0)"):
        assert needle in src, needle
    assert "w[dead, V - 1] = 1.0" not in src
