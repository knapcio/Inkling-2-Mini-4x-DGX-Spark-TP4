"""CPU unit tests: python -m unittest discover -s tests -v  (from overlay/; or -s tests/vcap).

Pure logic (config, live lengths, remap index, accept cap), a Monte Carlo check that
the cap is a valid stopping rule for sampled speculative decoding, the import-hook
patching (signature/source guards), and an end-to-end toy DSpark loop through the
real sitecustomize in subprocesses (greedy byte-identity: off vs fixed vs conf, remap
on vs off, eager vs in-graph-style epilogue accept)."""

import json
import os
import subprocess
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

from inkling_vcap import importhook, logic  # noqa: E402
from inkling_vcap.config import VcapConfigError, parse_config  # noqa: E402


class ConfigTest(unittest.TestCase):
    def test_off(self):
        for v in (None, "", "off", "OFF", "0", " none "):
            env = {} if v is None else {"INKLING_VERIFY_CAP": v}
            self.assertIsNone(parse_config(env))

    def test_modes(self):
        c = parse_config({"INKLING_VERIFY_CAP": "fixed:4"})
        self.assertEqual((c.kind, c.fixed_len, c.remap, c.log_every), ("fixed", 4, True, 500))
        c = parse_config({"INKLING_VERIFY_CAP": "conf:0.3, min=2,max=6",
                          "INKLING_VERIFY_CAP_REMAP": "0",
                          "INKLING_VERIFY_CAP_LOG_EVERY": "10",
                          "INKLING_VERIFY_CAP_UNION": "1"})
        self.assertEqual((c.kind, c.threshold, c.min_len, c.max_len), ("conf", 0.3, 2, 6))
        self.assertEqual((c.remap, c.log_every, c.union), (False, 10, True))
        self.assertEqual(parse_config({"INKLING_VERIFY_CAP": "confmax:0.5"}).kind, "conf")

    def test_cost_and_bsmax(self):
        c = parse_config({"INKLING_VERIFY_CAP": "cost:1=0.30,2=0.25,4=0.15,8=0.0,min=2"})
        self.assertEqual(c.kind, "cost")
        self.assertEqual(c.cost_table, ((1, 0.30), (2, 0.25), (4, 0.15), (8, 0.0)))
        self.assertEqual([c.threshold_for(b) for b in (1, 2, 3, 4, 5, 8, 64)],
                         [0.30, 0.25, 0.15, 0.15, 0.0, 0.0, 0.0])
        self.assertTrue(c.uses_draft_probs)
        self.assertEqual(c.policy(1, 8).threshold, 0.30)
        self.assertEqual(c.policy(6, 8).const_len, 8)  # T=0: every row, no logit work
        self.assertFalse(parse_config({"INKLING_VERIFY_CAP": "cost:4=0"}).uses_draft_probs)
        f = parse_config({"INKLING_VERIFY_CAP": "fixed:4,bsmax=2"})
        self.assertEqual([f.policy(b, 8).const_len for b in (1, 2, 3, 16)], [4, 4, 8, 8])
        k = parse_config({"INKLING_VERIFY_CAP": "conf:0.3,bsmax=1,max=6"})
        self.assertEqual(k.policy(1, 8).threshold, 0.3)
        self.assertEqual(k.policy(2, 8).const_len, 8)  # above bsmax: full width

    def test_bad(self):
        for v in ("fixed", "fixed:0", "fixed:x", "conf:1.5", "foo:1", "conf:0.2,min=0",
                  "conf:0.2,min=5,max=3", "conf:0.2,bogus=1", "cost:", "cost:x=0.1",
                  "cost:1=2", "cost:1=0.1,1=0.2", "conf:0.2,4=0.1", "fixed:3,bsmax=0"):
            with self.assertRaises(VcapConfigError, msg=v):
                parse_config({"INKLING_VERIFY_CAP": v})
        with self.assertRaises(VcapConfigError):
            parse_config({"INKLING_VERIFY_CAP": "fixed:9"}).validate_for_stride(8)


class LiveLensTest(unittest.TestCase):
    def test_threshold_leading_run(self):
        p = torch.tensor([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                          [0.2, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                          [0.9, 0.5, 0.99, 0.99, 0.99, 0.99, 0.99],
                          [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        L = logic.live_lens_from_probs(p, threshold=0.4, min_len=1, max_len=8)
        # surv row0: .9 .81 .729 .656 .59 .531 .478 -> all 7 >= .4 -> L=8
        # row1: .2 -> none -> L=1 ; row2: .9 .45 .4455 .441 .4366 .4322 .4279 -> 7 -> 8
        self.assertEqual(L.tolist(), [8, 1, 8, 8])
        L = logic.live_lens_from_probs(p, threshold=0.5, min_len=2, max_len=6)
        self.assertEqual(L.tolist(), [6, 2, 2, 6])
        self.assertEqual(L.dtype, torch.int64)

    def test_draft_token_probs(self):
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(3, 7, 50, generator=g)
        tokens = torch.randint(50, (3, 7), generator=g)
        temps = torch.tensor([1.0, 0.5, 2.0])
        greedy = torch.tensor([True, False, False])
        p = logic.draft_token_probs(logits, tokens, temperatures=temps, greedy_mask=greedy,
                                    chunk_rows=5)
        eff = torch.tensor([1.0, 0.5, 2.0])[:, None, None]
        ref = torch.softmax(logits / eff, -1).gather(-1, tokens[..., None]).squeeze(-1)
        torch.testing.assert_close(p, ref)
        pm = logic.draft_token_probs(logits.to(torch.bfloat16), tokens, use_max=True)
        refm = torch.softmax(logits.to(torch.bfloat16).float(), -1).amax(-1)
        torch.testing.assert_close(pm, refm)
        # out-of-range ids are clamped, NaN rows give probability 0
        bad = logits.clone()
        bad[0, 0, :] = float("nan")
        p2 = logic.draft_token_probs(bad, tokens + 100)
        self.assertEqual(float(p2[0, 0]), 0.0)
        self.assertTrue(bool(((p2 >= 0) & (p2 <= 1)).all()))


class RemapTest(unittest.TestCase):
    def test_src_index(self):
        lay = logic.RowLayout(capacity=4, stride=4, device="cpu")
        lens = torch.tensor([4, 2, 1, 0])
        src = logic.build_src_index(lens, lay)
        self.assertEqual(src.tolist(), [0, 1, 2, 3, 4, 5, 4, 4, 8, 8, 8, 8, 0, 0, 0, 0])
        # identity when every request is fully live
        self.assertEqual(logic.build_src_index(torch.full((4,), 4), lay).tolist(),
                         list(range(16)))
        # partial rebuild touches only the first num_reqs requests
        out = src.clone()
        logic.build_src_index(torch.tensor([1, 3, 1, 0]), lay, num_reqs=2, out=out)
        self.assertEqual(out.tolist(), [0, 0, 0, 0, 4, 5, 6, 4, 8, 8, 8, 8, 0, 0, 0, 0])

    def test_remap_shrinks_union(self):
        g = torch.Generator().manual_seed(1)
        ids = torch.stack([torch.randperm(128, generator=g)[:8] for _ in range(8)]).int()
        w = torch.rand(8, 8, generator=g)
        lay = logic.RowLayout(1, 8, "cpu")
        for L in range(1, 9):
            src = logic.build_src_index(torch.tensor([L]), lay)
            w2, ids2 = logic.remap_rows(w, ids, src)
            self.assertTrue(torch.equal(ids2[:L], ids[:L]))
            self.assertTrue(torch.equal(w2[:L], w[:L]))
            self.assertTrue(bool((ids2[L:] == ids[0]).all()))
            live_union = set(ids[:L].reshape(-1).tolist())
            self.assertEqual(set(ids2.reshape(-1).tolist()), live_union)
            self.assertEqual(int(logic.distinct_experts(ids2, 128)), len(live_union))

    def test_greedy_cap(self):
        cand = torch.tensor([[9, 1, 2, 3, 4], [9, 1, 7, 3, 4], [9, 5, 2, 3, 4]])
        pred = torch.tensor([[1, 2, 3, 4, 6], [1, 2, 3, 4, 6], [1, 2, 3, 4, 6]])
        c, b = logic.greedy_accept(cand, pred)
        self.assertEqual((c.tolist(), b.tolist()), ([4, 1, 0], [6, 2, 1]))
        c, b = logic.greedy_accept(cand, pred, torch.tensor([3, 3, 3]))
        # capped at L-1 = 2 drafts; bonus is the argmax of row 2
        self.assertEqual((c.tolist(), b.tolist()), ([2, 1, 0], [3, 2, 1]))
        capped, trim = logic.cap_correct_len(torch.tensor([4, 1, 0, 3]),
                                             torch.tensor([3, 3, 3, 0]))
        self.assertEqual((capped.tolist(), trim.tolist()), ([2, 1, 0, 0], [2, 0, 0, 3]))


class PlanStepTest(unittest.TestCase):
    def _state(self, spec):
        from inkling_vcap.state import VcapState

        st = VcapState()
        st.config = parse_config({"INKLING_VERIFY_CAP": spec})
        st.epilogue = types.SimpleNamespace(_static_step_state=None)
        st.allocate(stride=8, capacity=4, device="cpu")
        return st

    def test_const_steps_launch_nothing_when_unchanged(self):
        st = self._state("fixed:3,bsmax=2")
        st.plan_step(bs=2, proposal=None)
        self.assertEqual(st.live_lens.tolist(), [3, 3, 0, 0])
        self.assertEqual(st.src_index[:16].tolist(),
                         [0, 1, 2, 0, 0, 0, 0, 0, 8, 9, 10, 8, 8, 8, 8, 8])
        # an external refill of the buffer is only repaired when the epilogue state moved
        st.live_lens[:2].fill_(8)
        st.plan_step(bs=2, proposal=None)
        self.assertEqual(st.live_lens.tolist(), [8, 8, 0, 0])
        st.epilogue._static_step_state = (2, True)
        st.plan_step(bs=2, proposal=None)
        self.assertEqual(st.live_lens.tolist(), [3, 3, 0, 0])
        # above bsmax: full width, identity remap for the real rows
        st.plan_step(bs=3, proposal=None)
        self.assertEqual(st.live_lens.tolist(), [8, 8, 8, 0])
        self.assertEqual(st.src_index.tolist(), list(range(24)) + [0] * 8)

    def test_cost_switches_between_dynamic_and_full(self):
        st = self._state("cost:1=0.5,2=0.0")
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(1, 7, 30, generator=g)
        logits[0, :, 0] = 10.0  # confident draft at every position
        logits[0, 3, 0] = 0.0   # ... except position 3
        block = types.SimpleNamespace(draft_tokens=torch.zeros(1, 7, dtype=torch.long),
                                      corrected_logits=logits, temperatures=None,
                                      greedy_mask=None)
        st.plan_step(bs=1, proposal=types.SimpleNamespace(draft_block=block))
        self.assertEqual(st.live_lens.tolist(), [4, 0, 0, 0])
        st.plan_step(bs=2, proposal=None)  # T=0 at bs=2: no draft logits needed
        self.assertEqual(st.live_lens.tolist(), [8, 8, 0, 0])
        self.assertEqual(st.src_index[:16].tolist(), list(range(16)))


class StockCappedSamplingLawTest(unittest.TestCase):
    """Multi-step Monte Carlo of the engine's capped sampled accept, mirroring the
    stock kernels (speculative_sampling_classic_kernel / blockwise_rejection /
    the split variant): rejection runs over ALL rows, including dead rows whose target
    distribution here is deliberate garbage; correct = min(raw, L-1); when the cap
    trims (raw > L-1) the bonus is the draft accepted at row L-1 (the token sitting in
    the first dead row, but accepted by row L-1's live distribution), otherwise the
    residual / final sample at row raw <= L-1. L comes from the draft's max
    probabilities. The first three emitted tokens must follow the target law exactly;
    the boundary cases (raw == L-1, raw > L-1) are asserted to occur often."""

    P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
                     dtype=torch.float64)  # target: order-1 Markov over 3 tokens
    Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]],
                     dtype=torch.float64)  # draft
    G = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64)  # dead-row garbage

    def _simulate(self, n, bonus_rule, threshold=0.45, gamma=3, out_len=3, seed=0):
        gen = torch.Generator().manual_seed(seed)
        P, Q, G = self.P, self.Q, self.G
        out = torch.full((n, out_len), -1, dtype=torch.long)
        filled = torch.zeros(n, dtype=torch.long)
        anchor = torch.zeros(n, dtype=torch.long)
        stats = {"trim": 0, "boundary": 0, "rows": 0}
        rows = torch.arange(n)
        while bool((filled < out_len).any()):
            # draft chain d_0..d_{gamma-1}, q_j = Q[previous token]
            prev, drafts, qs = anchor, [], []
            for _ in range(gamma):
                q = Q[prev]
                d = torch.multinomial(q, 1, generator=gen).squeeze(1)
                drafts.append(d)
                qs.append(q)
                prev = d
            drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1)  # [n,g], [n,g,V]
            surv = torch.cumprod(qs.max(-1).values, 1)
            L = 1 + (surv >= threshold).sum(1)  # max-probability rule, 1..gamma+1
            # target rows: row 0 = anchor, row j = d_{j-1}; rows >= L are garbage
            row_tok = torch.cat([anchor[:, None], drafts], 1)
            ps = P[row_tok]  # [n, g+1, V]
            dead = torch.arange(gamma + 1)[None, :] >= L[:, None]
            ps = torch.where(dead[..., None], G.expand_as(ps), ps)
            # stock rejection chain over every row (dead ones included)
            u = torch.rand((n, gamma), generator=gen, dtype=torch.float64)
            pd = ps[:, :gamma].gather(2, drafts[..., None]).squeeze(2)
            qd = qs.gather(2, drafts[..., None]).squeeze(2)
            ok = u * qd < pd
            raw = torch.cumprod(ok.long(), 1).sum(1)
            capped = torch.minimum(raw, L - 1)
            # residual / final distribution at row raw (live whenever it is used)
            p_raw = ps[rows, raw]
            q_raw = torch.cat([qs, qs[:, :1]], 1)[rows, raw]
            resid = torch.where((raw == gamma)[:, None], p_raw,
                                (p_raw - q_raw).clamp(min=0))
            resid = resid / resid.sum(1, keepdim=True)
            resid_tok = torch.multinomial(resid, 1, generator=gen).squeeze(1)
            trimmed = capped < raw
            if bonus_rule == "stock":
                at_cap = drafts.gather(1, capped.clamp(max=gamma - 1)[:, None]).squeeze(1)
                bonus = torch.where(trimmed, at_cap, resid_tok)
            elif bonus_rule == "leak":  # negative control: residual at a dead row
                bonus = resid_tok
            else:  # "fresh": stop before verifying d_{L-1}; fresh draw from row L-1
                p_cap = ps[rows, capped]
                fresh = torch.multinomial(p_cap, 1, generator=gen).squeeze(1)
                stopped_at_cap = raw >= L - 1
                bonus = torch.where(stopped_at_cap, fresh, resid_tok)
            emitted = torch.cat([drafts, bonus[:, None]], 1)
            n_emit = capped + 1
            active = filled < out_len
            stats["trim"] += int((active & trimmed).sum())
            stats["boundary"] += int((active & (raw == L - 1) & (L <= gamma)).sum())
            stats["rows"] += int(active.sum())
            for k in range(gamma + 1):
                take = active & (k < n_emit) & (filled + k < out_len)
                tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
                idx = (filled + k).clamp(max=out_len - 1)
                out[take, idx[take]] = tok[take]
            filled = torch.where(active, filled + n_emit, filled)
            anchor = torch.where(active, bonus, anchor)
        return out, stats

    def _error(self, bonus_rule):
        n = 300_000
        out, stats = self._simulate(n, bonus_rule)
        emp = torch.zeros(3, 3, 3, dtype=torch.float64)
        emp.index_put_((out[:, 0], out[:, 1], out[:, 2]),
                       torch.ones(n, dtype=torch.float64), accumulate=True)
        emp /= n
        P = self.P
        exact = torch.einsum("a,ab,bc->abc", P[0], P, P)
        return float((emp - exact).abs().max()), stats

    def _check(self, bonus_rule):
        err, stats = self._error(bonus_rule)
        self.assertGreater(stats["trim"], stats["rows"] // 20, stats)
        self.assertGreater(stats["boundary"], stats["rows"] // 20, stats)
        self.assertLess(err, 3.5e-3, (bonus_rule, err, stats))

    def test_stock_capped_bonus_is_exact(self):
        self._check("stock")

    def test_fresh_bonus_alternative_is_exact(self):
        self._check("fresh")

    def test_negative_control_detects_dead_row_leak(self):
        err, _ = self._error("leak")
        self.assertGreater(err, 1e-2)


class SampledStoppingRuleTest(unittest.TestCase):
    """Chain speculative sampling capped at a live length L: the first two emitted
    tokens must follow the target's joint law when L depends only on information
    that does not include the token being verified (here: the draft's max
    probabilities). A cut on the proposed token's own probability is biased; the
    test asserts both facts. Vectorized Monte Carlo."""

    P1 = torch.tensor([0.5, 0.2, 0.2, 0.1])
    P2 = torch.tensor([[0.7, 0.1, 0.1, 0.1], [0.1, 0.6, 0.2, 0.1],
                       [0.25, 0.25, 0.25, 0.25], [0.1, 0.1, 0.1, 0.7]])
    Q1 = torch.tensor([0.3, 0.4, 0.2, 0.1])
    Q2 = torch.tensor([[0.4, 0.3, 0.2, 0.1], [0.2, 0.2, 0.5, 0.1],
                       [0.1, 0.1, 0.1, 0.7], [0.25, 0.25, 0.25, 0.25]])

    def _joint_error(self, rule: str) -> float:
        torch.manual_seed(0)
        p1, p2, q1, q2 = self.P1, self.P2, self.Q1, self.Q2
        V, N = 4, 400_000
        d1 = torch.multinomial(q1, N, replacement=True)
        d2 = torch.multinomial(q2[d1], 1).squeeze(1)
        if rule == "max":  # depends on the draft distributions, not on d2
            score = q1.max() * q2[d1].max(dim=1).values
            L = torch.where(score > 0.18, 3, 2)
        else:  # the proposed tokens' own probabilities
            L = torch.where(q1[d1] * q2[d1, d2] > 0.12, 3, 2)
        u1, u2 = torch.rand(N), torch.rand(N)
        acc1 = u1 < (p1[d1] / q1[d1]).clamp(max=1)
        resid1 = (p1 - q1).clamp(min=0)
        r1 = torch.multinomial(resid1 / resid1.sum(), N, replacement=True)
        x1 = torch.where(acc1, d1, r1)
        # Second token: by rejection sampling of d2 at row 1 when L == 3, else (cap,
        # or a rejection at row 0) from the target law p2[x1] (bonus / next step).
        p2x, q2d = p2[x1], q2[d1]
        ratio = p2x.gather(1, d2[:, None]).squeeze(1) / q2d.gather(1, d2[:, None]).squeeze(1)
        verify2 = acc1 & (L == 3)
        acc2 = verify2 & (u2 < ratio.clamp(max=1))
        resid2 = (p2x - q2d).clamp(min=0)
        resid2 = resid2 / resid2.sum(1, keepdim=True).clamp(min=1e-12)
        r2 = torch.multinomial(resid2 + 1e-12, 1).squeeze(1)
        fresh = torch.multinomial(p2x, 1).squeeze(1)
        x2 = torch.where(acc2, d2, torch.where(verify2, r2, fresh))
        emp = torch.zeros(V, V)
        emp.index_put_((x1, x2), torch.ones(N), accumulate=True)
        return float((emp / N - p1[:, None] * p2).abs().max())

    def test_max_probability_cut_is_exact(self):
        self.assertLess(self._joint_error("max"), 4e-3)

    def test_token_probability_cut_is_biased(self):
        self.assertGreater(self._joint_error("token"), 2e-2)


def _fake_env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("INKLING_VERIFY_CAP")}
    env["PYTHONPATH"] = os.pathsep.join([ROOT, os.path.join(HERE, "fake_sglang")])
    env.update(extra)
    return env


def _run(extra):
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_e2e.py")],
                         env=_fake_env(extra), capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-4000:])
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


class EndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = {path: _run({"FAKE_PATH": path, "FAKE_BS": "3"})[0]
                    for path in ("eager", "epilogue")}

    def test_baseline_unpatched(self):
        for res in self.base.values():
            self.assertFalse(res["enabled"])
            self.assertEqual(res["applied"], [])
        self.assertEqual(self.base["eager"]["seqs"], self.base["epilogue"]["seqs"])

    def test_greedy_byte_identity(self):
        for path in ("eager", "epilogue"):
            base = self.base[path]
            for mode in ("fixed:8", "fixed:3", "fixed:1", "conf:0.3", "confmax:0.6",
                         "conf:0.05,min=2,max=6", "cost:1=0.3,2=0.2,4=0.1",
                         "fixed:3,bsmax=2"):
                for remap in ("1", "0"):
                    res, err = _run({"FAKE_PATH": path, "FAKE_BS": "3",
                                     "INKLING_VERIFY_CAP": mode,
                                     "INKLING_VERIFY_CAP_REMAP": remap,
                                     "INKLING_VERIFY_CAP_LOG_EVERY": "5"})
                    tag = f"{path} {mode} remap={remap}"
                    self.assertTrue(res["enabled"], tag)
                    self.assertEqual(res["seqs"], base["seqs"], tag)
                    self.assertIn("[inkling-vcap] enabled", err, tag)
                    if mode == "fixed:8":  # full width everywhere: fully inert
                        self.assertEqual((res["remap_recorded"], res["cutoff_injected"],
                                          res["static_accept_recorded"]), (0, 0, 0), tag)
                        self.assertIn("capping=off (inert)", err, tag)
                    else:
                        if remap == "1":
                            self.assertGreater(res["remap_recorded"], 0, tag)
                        if path == "epilogue":
                            self.assertGreater(res["static_accept_recorded"], 0, tag)
                        else:
                            self.assertGreater(res["cutoff_injected"], 0, tag)
                    if mode == "fixed:3,bsmax=2":  # bs=3 > bsmax: never capped
                        self.assertEqual(res["steps"], base["steps"], tag)
                        self.assertEqual(res["mean_union"], base["mean_union"], tag)
                    if mode == "fixed:3" and remap == "1":
                        # dead rows touch no extra experts
                        self.assertLess(res["mean_union"], base["mean_union"], tag)

    def test_union_counter_logs(self):
        res, err = _run({"FAKE_BS": "1", "INKLING_VERIFY_CAP": "fixed:8",
                         "INKLING_VERIFY_CAP_UNION": "1",
                         "INKLING_VERIFY_CAP_LOG_EVERY": "3"})
        self.assertIn("union_mean=", err)
        self.assertIn("mean_live=8.000", err)


class PatchGuardTest(unittest.TestCase):
    def test_signature_change_fails_import(self):
        from inkling_vcap import patches

        mod = types.ModuleType("fake")

        class Inkling2Gate:
            def forward(self, x, extra):
                return x

        class Inkling2MoE:
            def forward(self, x, forward_batch=None, combine=True):
                return self.experts(self.gate(x))

        mod.Inkling2Gate, mod.Inkling2MoE = Inkling2Gate, Inkling2MoE
        with self.assertRaises(patches.PatchTargetError):
            patches.patch_moe(mod)
        with self.assertRaises(patches.PatchTargetError):
            patches.patch_draft(types.ModuleType("empty"))

    def test_post_import_hook(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "vcap_probe_mod.py"), "w") as f:
                f.write("VALUE = 1\n")
            sys.path.insert(0, tmp)
            seen = []
            finder = importhook.install({"vcap_probe_mod": lambda m: seen.append(m.VALUE)})
            try:
                import vcap_probe_mod  # noqa: F401
                import inspect
                self.assertEqual(seen, [1])
                self.assertIn("VALUE", inspect.getsource(vcap_probe_mod))
            finally:
                sys.meta_path.remove(finder)
                sys.path.remove(tmp)

    def test_malformed_env_stops_process(self):
        out = subprocess.run([sys.executable, "-c", "print('alive')"],
                             env=_fake_env({"INKLING_VERIFY_CAP": "conf:7"}),
                             capture_output=True, text=True)
        self.assertNotEqual(out.returncode, 0)
        self.assertTrue("[inkling-vcap] startup failed" in out.stderr or "[inkling-overlay" in out.stderr, out.stderr)
        out = subprocess.run([sys.executable, "-c", "import sys; print('sglang' in sys.modules)"],
                             env=_fake_env({"INKLING_VERIFY_CAP": "off"}),
                             capture_output=True, text=True)
        self.assertEqual(out.stdout.strip(), "False")


if __name__ == "__main__":
    unittest.main()
