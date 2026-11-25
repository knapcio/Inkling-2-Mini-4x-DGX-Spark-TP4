"""CPU tests of the ``policy:`` mode (calibrated, batch-size-priced verify width):

- the JSON format and its validation, and the env grammar (path keeps its case);
- the device rule equals the replay's numpy reference (bsreplay.pol_ltrc / pol_ltrb),
  decision for decision, on random batches with mixed greedy / sampled rows;
- the stopping-rule property, directly: perturbing a request's draft confidences at
  or beyond its chosen cut position (m_j for j >= L_r) never changes any width in
  the batch, with batch-aware rounds on;
- a Monte Carlo of the stock capped sampled accept (as StockCappedSamplingLawTest)
  with TWO requests per batch whose widths interact through the batch-aware rounds:
  the emitted tokens follow the target law; a look-ahead rule is detected as biased;
- plan_step on the CPU state, and the greedy byte identity end to end on the toy
  DSpark loop (eager and epilogue accept paths, remap on and off).
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # the overlay root (policy/)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

import bsreplay  # noqa: E402
from inkling_vcap import logic  # noqa: E402
from inkling_vcap.config import VcapConfigError, parse_config  # noqa: E402
from inkling_vcap.policyfile import PolicyFileError, parse_policy  # noqa: E402

EDGES = list(bsreplay.EDGES)
NB = len(EDGES) + 1
U1 = [8.0, 12.4, 15.9, 18.9, 21.4, 23.7, 25.6, 27.4]


def _calib(seed=0):
    rng = np.random.default_rng(seed)
    # increasing in the bin, decreasing a little with the position
    base = np.sort(rng.uniform(0.2, 0.98, (2, 7, NB)), axis=-1)
    return (base * np.linspace(1.0, 0.9, 7)[None, :, None]).tolist()


def _policy_obj(theta=((1, 0.03), (4, 0.05), (8, 0.0)), pool=None, rounds=0, **kw):
    cal = _calib()
    obj = {"format": "inkling-vcap-policy/1", "stride": 8, "bin_edges": EDGES,
           "calib": {"greedy": cal[0], "sampled": cal[1]}, "u1": U1,
           "theta": [list(t) for t in theta], "rounds": rounds}
    if pool:
        obj["pool"] = [list(p) for p in pool]
    obj.update(kw)
    return obj


def _write(obj):
    fd, path = tempfile.mkstemp(suffix=".json", prefix="VcapPolicy")
    with os.fdopen(fd, "w") as f:
        json.dump(obj, f)
    return path


class PolicyFileTest(unittest.TestCase):
    def test_parse_and_lookup(self):
        pf = parse_policy(_policy_obj(pool=((2, 60.0), (8, 70.0)), rounds=2), "x")
        self.assertEqual([pf.theta_for(b) for b in (1, 2, 4, 5, 8, 24)],
                         [0.03, 0.05, 0.05, 0.0, 0.0, 0.0])
        self.assertEqual([pf.pool_for(b) for b in (1, 2, 3, 24)], [60.0, 60.0, 70.0, 70.0])
        self.assertAlmostEqual(pf.row_cost[0], 4.4)
        self.assertEqual(len(pf.calib), 2)
        only = _policy_obj()
        del only["calib"]["sampled"]
        self.assertEqual(parse_policy(only).calib[1], parse_policy(only).calib[0])

    def test_shipped_policy_file(self):
        from inkling_vcap.policyfile import load_policy

        pf = load_policy(os.path.join(ROOT, "inkling_vcap", "policies", "r2-theta004.json"))
        self.assertEqual((pf.stride, pf.rounds), (8, 0))
        self.assertEqual([pf.theta_for(b) for b in (1, 8, 24, 64)], [0.04] * 4)

    def test_bad(self):
        bads = [
            dict(format="nope"),
            dict(stride=1),
            dict(bin_edges=[0.5, 0.4]),
            dict(u1=[8.0] * 7),
            dict(u1=[8.0, 7.0] + U1[2:]),
            dict(calib={"greedy": [[0.5] * NB] * 6}),
            dict(calib={"other": _calib()[0]}),
            dict(calib={"greedy": [[1.5] * NB] * 7}),
            dict(theta=[]),
            dict(theta=[[1, -0.1]]),
            dict(theta=[[1, 0.1], [1, 0.2]]),
            dict(pool=[[2, 20.0]], rounds=1),  # E must exceed u1[-1]
            dict(rounds=2),  # rounds without a pool
            dict(rounds=9, pool=[[2, 60.0]]),
        ]
        for kw in bads:
            with self.assertRaises(PolicyFileError, msg=str(kw)):
                parse_policy(_policy_obj(**kw), "bad")

    def test_env(self):
        path = _write(_policy_obj())
        try:
            self.assertIn("VcapPolicy", path)  # upper case survives the parser
            c = parse_config({"INKLING_VERIFY_CAP": f"policy:{path},MIN=1,max=8"})
            self.assertEqual(c.kind, "policy")
            self.assertTrue(c.uses_draft_probs)
            self.assertFalse(c.never_caps(8))
            self.assertEqual(c.policy(1, 8).threshold, 0.03)
            self.assertEqual(c.policy(6, 8).const_len, 8)  # theta 0: full width
            self.assertIn("policy:", c.describe())
            c.validate_for_stride(8)
            with self.assertRaises(VcapConfigError):
                c.validate_for_stride(6)
            inert = _write(_policy_obj(theta=((1, 0.0),)))
            ci = parse_config({"INKLING_VERIFY_CAP": f"policy:{inert}"})
            self.assertTrue(ci.never_caps(8))
            self.assertFalse(ci.uses_draft_probs)
            os.unlink(inert)
            for bad in ("policy:", f"policy:{path}.missing",
                        f"policy:{path},4=0.1"):  # bs entries belong to cost:
                with self.assertRaises(VcapConfigError, msg=bad):
                    parse_config({"INKLING_VERIFY_CAP": bad})
            # the old modes still parse as before
            self.assertEqual(parse_config({"INKLING_VERIFY_CAP": "COST:1=0.3,8=0"}).cost_table,
                             ((1, 0.3), (8, 0.0)))
        finally:
            os.unlink(path)


def _device_args(dtype=torch.float64):
    cal = _calib()
    return dict(table=torch.tensor(cal, dtype=dtype),
                edges=torch.tensor(EDGES, dtype=torch.float32),
                row_cost=torch.tensor(np.diff(U1), dtype=dtype),
                u1=torch.tensor(U1, dtype=dtype))


def _ref_group(m, sampled):
    """bsreplay Group for one batch per row of m [n, bs, 7]."""
    n, bs, _ = m.shape
    z = np.zeros((n, bs), np.int64)
    return bsreplay.Group(bs, m, z, ~sampled, z, np.full((n, 8), np.nan),
                          np.zeros(n, "<U2"), np.zeros((n, bs), "<U2"), np.zeros(n))


def _ref_calib():
    c = bsreplay.Calib(use_pf=False)
    t = np.asarray(_calib())  # make table() return exactly this
    c.trials[..., 0] = 1e12
    c.succ[..., 0] = t * (1e12 + 2.0) - 1.0
    return c


class DeviceRuleTest(unittest.TestCase):
    def _random(self, n, bs, seed):
        g = torch.Generator().manual_seed(seed)
        hi = torch.rand(n, bs, 7, generator=g) < 0.6
        m = torch.where(hi, 0.85 + 0.15 * torch.rand(n, bs, 7, generator=g),
                        torch.rand(n, bs, 7, generator=g))
        sampled = torch.rand(n, bs, generator=g) < 0.5
        return m.to(torch.float64), sampled

    def test_matches_replay_reference(self):
        args = _device_args()
        calib = _ref_calib()
        np.testing.assert_allclose(calib.table()[:, :, :, 0], np.asarray(_calib()), rtol=1e-9)
        u1 = np.asarray(U1)
        c = np.diff(u1)
        for bs, rounds, theta, pool in ((1, 0, 0.03, 0.0), (4, 0, 0.05, 0.0),
                                        (4, 1, 0.08, 60.0), (8, 3, 0.1, 70.0),
                                        (24, 3, 0.3, 300.0)):
            m, sampled = self._random(400, bs, seed=bs + rounds)
            dev = torch.stack([
                logic.calibrated_live_lens(m[i], sampled[i], theta=theta, min_len=1,
                                           max_len=8, pool=pool, rounds=rounds, **args)
                for i in range(len(m))])
            batched = logic.calibrated_live_lens(m, sampled, theta=theta, min_len=1,
                                                 max_len=8, pool=pool, rounds=rounds, **args)
            self.assertTrue(torch.equal(dev, batched))
            S = calib.surv(_ref_group(m.numpy(), sampled.numpy()))
            ref = (bsreplay.pol_ltrb(S, c, theta, u1, pool, rounds) if rounds
                   else bsreplay.pol_ltrc(S, c, theta))
            np.testing.assert_array_equal(dev.numpy(), ref, err_msg=f"bs={bs} rounds={rounds}")
            self.assertGreater(len(set(ref.reshape(-1).tolist())), 3)  # widths vary

    def test_rounds_are_monotone_and_cheaper_rows_widen(self):
        args = _device_args()
        m, sampled = self._random(300, 8, seed=7)
        prev = None
        for rounds in (1, 2, 3, 4):
            L = logic.calibrated_live_lens(m, sampled, theta=0.1, min_len=1, max_len=8,
                                           pool=60.0, rounds=rounds, **args)
            if prev is not None:
                self.assertTrue(bool((L >= prev).all()))
            prev = L
        solo = logic.calibrated_live_lens(m, sampled, theta=0.1, min_len=1, max_len=8, **args)
        self.assertTrue(bool((prev >= solo).all()))  # overlap makes rows cheaper

    def test_width_ignores_confidences_at_and_past_the_cut(self):
        """The law-preservation property itself: L_r may depend on m_r[0..L_r-1]
        (functions of d_0..d_{L_r-2}), never on m_r[j] for j >= L_r (which depend on
        d_{L_r-1}, the first excluded draft, or later ones)."""
        args = _device_args()
        g = torch.Generator().manual_seed(3)
        m, sampled = self._random(200, 6, seed=11)
        for rounds, pool in ((0, 0.0), (3, 60.0)):
            base = logic.calibrated_live_lens(m, sampled, theta=0.08, min_len=1, max_len=8,
                                              pool=pool, rounds=rounds, **args)
            for trial in range(20):
                pos = torch.arange(7)[None, None, :]
                beyond = pos >= base[..., None]  # m index j >= L (d_{L-1} onwards)
                noise = torch.rand(m.shape, generator=g, dtype=torch.float64)
                m2 = torch.where(beyond, noise, m)
                again = logic.calibrated_live_lens(m2, sampled, theta=0.08, min_len=1,
                                                   max_len=8, pool=pool, rounds=rounds,
                                                   **args)
                self.assertTrue(torch.equal(base, again), (rounds, trial))


class BatchSampledLawTest(unittest.TestCase):
    """Two requests per batch, each an order-1 Markov chain (gamma = 3), widths from
    the batch-aware calibrated rule (rounds = 2: each request's threshold depends on
    the other request's width). Stock capped sampled accept: rejection over every
    row (dead rows get a garbage target), correct = min(raw, L-1), trimmed -> the
    draft accepted at row L-1, else the residual at row raw. The first three emitted
    tokens of every request must follow the target law."""

    P = torch.tensor([[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.3, 0.1, 0.6]],
                     dtype=torch.float64)
    Q = torch.tensor([[0.3, 0.6, 0.1], [0.1, 0.8, 0.1], [0.5, 0.2, 0.3]],
                     dtype=torch.float64)
    G = torch.tensor([0.98, 0.01, 0.01], dtype=torch.float64)
    EDGES = torch.tensor([0.55, 0.7], dtype=torch.float32)
    TABLE = torch.tensor([[[0.3, 0.6, 0.9]] * 3] * 2, dtype=torch.float64)
    U1 = torch.tensor([8.0, 12.0, 15.0, 17.0], dtype=torch.float64)

    def _widths(self, m, rule):
        if rule == "lookahead":  # include everything iff the LAST confidence is high
            return torch.where(m[..., 2] >= 0.7, 4, 1)
        return logic.calibrated_live_lens(
            m, torch.ones(m.shape[:-1], dtype=torch.bool), table=self.TABLE,
            edges=self.EDGES, row_cost=self.U1[1:] - self.U1[:-1], theta=0.25,
            min_len=1, max_len=4, u1=self.U1, pool=20.0, rounds=2)

    def _simulate(self, n_batches, rule, gamma=3, out_len=3, seed=0):
        gen = torch.Generator().manual_seed(seed)
        P, Q, G = self.P, self.Q, self.G
        n = 2 * n_batches
        out = torch.full((n, out_len), -1, dtype=torch.long)
        filled = torch.zeros(n, dtype=torch.long)
        anchor = torch.zeros(n, dtype=torch.long)
        rows = torch.arange(n)
        stats = {"trim": 0, "boundary": 0, "rows": 0, "widths": set(), "coupled": 0}
        while bool((filled < out_len).any()):
            prev, drafts, qs = anchor, [], []
            for _ in range(gamma):
                q = Q[prev]
                d = torch.multinomial(q, 1, generator=gen).squeeze(1)
                drafts.append(d)
                qs.append(q)
                prev = d
            drafts, qs = torch.stack(drafts, 1), torch.stack(qs, 1)
            m = qs.max(-1).values.view(n_batches, 2, gamma)
            L = self._widths(m, rule).reshape(n)
            if rule != "lookahead":
                # does a request's width depend on its partner? (partner shuffled)
                perm = torch.randperm(n_batches, generator=gen)
                m2 = m.clone()
                m2[:, 1] = m[perm, 1]
                L2 = self._widths(m2, rule)
                stats["coupled"] += int((L2[:, 0] != L.view(n_batches, 2)[:, 0]).sum())
            row_tok = torch.cat([anchor[:, None], drafts], 1)
            ps = P[row_tok]
            dead = torch.arange(gamma + 1)[None, :] >= L[:, None]
            ps = torch.where(dead[..., None], G.expand_as(ps), ps)
            u = torch.rand((n, gamma), generator=gen, dtype=torch.float64)
            pd = ps[:, :gamma].gather(2, drafts[..., None]).squeeze(2)
            qd = qs.gather(2, drafts[..., None]).squeeze(2)
            raw = torch.cumprod((u * qd < pd).long(), 1).sum(1)
            capped = torch.minimum(raw, L - 1)
            p_raw = ps[rows, raw]
            q_raw = torch.cat([qs, qs[:, :1]], 1)[rows, raw]
            resid = torch.where((raw == gamma)[:, None], p_raw, (p_raw - q_raw).clamp(min=0))
            resid = resid / resid.sum(1, keepdim=True)
            resid_tok = torch.multinomial(resid, 1, generator=gen).squeeze(1)
            trimmed = capped < raw
            at_cap = drafts.gather(1, capped.clamp(max=gamma - 1)[:, None]).squeeze(1)
            bonus = torch.where(trimmed, at_cap, resid_tok)
            emitted = torch.cat([drafts, bonus[:, None]], 1)
            n_emit = capped + 1
            active = filled < out_len
            stats["trim"] += int((active & trimmed).sum())
            stats["boundary"] += int((active & (raw == L - 1) & (L <= gamma)).sum())
            stats["rows"] += int(active.sum())
            stats["widths"] |= set(L[active].tolist())
            for k in range(gamma + 1):
                take = active & (k < n_emit) & (filled + k < out_len)
                tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
                idx = (filled + k).clamp(max=out_len - 1)
                out[take, idx[take]] = tok[take]
            filled = torch.where(active, filled + n_emit, filled)
            anchor = torch.where(active, bonus, anchor)
        return out, stats

    def _error(self, rule):
        out, stats = self._simulate(150_000, rule)
        n = out.shape[0]
        emp = torch.zeros(3, 3, 3, dtype=torch.float64)
        emp.index_put_((out[:, 0], out[:, 1], out[:, 2]),
                       torch.ones(n, dtype=torch.float64), accumulate=True)
        exact = torch.einsum("a,ab,bc->abc", self.P[0], self.P, self.P)
        return float((emp / n - exact).abs().max()), stats

    def test_batch_aware_rule_is_exact(self):
        err, stats = self._error("batch")
        self.assertGreater(stats["trim"], stats["rows"] // 20, stats)
        self.assertGreater(stats["boundary"], stats["rows"] // 40, stats)
        self.assertGreater(stats["coupled"], stats["rows"] // 40, stats)  # partner matters
        self.assertGreaterEqual(len(stats["widths"]), 3, stats)
        self.assertLess(err, 3.5e-3, (err, stats))

    def test_lookahead_rule_is_detected(self):
        err, _ = self._error("lookahead")
        self.assertGreater(err, 1e-2)


class PolicyPlanStepTest(unittest.TestCase):
    def test_plan_step(self):
        from inkling_vcap.state import VcapState

        path = _write(_policy_obj(theta=((1, 0.03), (2, 0.1), (3, 0.0))))
        try:
            st = VcapState()
            st.config = parse_config({"INKLING_VERIFY_CAP": f"policy:{path}"})
            st.epilogue = types.SimpleNamespace(_static_step_state=None)
            st.allocate(stride=8, capacity=4, device="cpu")
            g = torch.Generator().manual_seed(0)
            logits = torch.randn(2, 7, 50, generator=g)
            logits[:, :, 0] = 12.0  # confident everywhere ...
            logits[1, 2, :] = 0.0   # ... except request 1, position 2 (uniform)
            block = types.SimpleNamespace(
                draft_tokens=torch.zeros(2, 7, dtype=torch.long), corrected_logits=logits,
                temperatures=torch.tensor([1.0, 0.7]),
                greedy_mask=torch.tensor([True, False]))
            st.plan_step(bs=2, proposal=types.SimpleNamespace(draft_block=block))
            probs = logic.draft_token_probs(logits, block.draft_tokens,
                                            temperatures=block.temperatures,
                                            greedy_mask=block.greedy_mask, use_max=True)
            want = logic.calibrated_live_lens(
                probs, torch.tensor([False, True]), theta=0.1, min_len=1, max_len=8,
                **{k: v.to(torch.float32) for k, v in _device_args().items()})
            self.assertEqual(st.live_lens[:2].tolist(), want.tolist())
            self.assertEqual(st.live_lens[2:].tolist(), [0, 0])
            self.assertLessEqual(int(st.live_lens[1]), 3)  # cut at or before position 2
            L0 = int(st.live_lens[0])
            self.assertEqual(st.src_index[:8].tolist(),
                             list(range(L0)) + [0] * (8 - L0))
            st.plan_step(bs=3, proposal=None)  # theta 0 at bs=3: full width, no logits
            self.assertEqual(st.live_lens.tolist(), [8, 8, 8, 0])
        finally:
            os.unlink(path)


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


class PolicyEndToEndTest(unittest.TestCase):
    def test_greedy_byte_identity(self):
        paths = {
            "solo": _write(_policy_obj(theta=((8, 0.05),))),
            "batch": _write(_policy_obj(theta=((8, 0.2),), pool=((8, 40.0),), rounds=3)),
            "inert": _write(_policy_obj(theta=((8, 0.0),))),
        }
        try:
            for path in ("eager", "epilogue"):
                base, _ = _run({"FAKE_PATH": path, "FAKE_BS": "3"})
                for name, pol in paths.items():
                    for remap in ("1", "0"):
                        res, err = _run({"FAKE_PATH": path, "FAKE_BS": "3",
                                         "INKLING_VERIFY_CAP": f"policy:{pol}",
                                         "INKLING_VERIFY_CAP_REMAP": remap,
                                         "INKLING_VERIFY_CAP_LOG_EVERY": "5"})
                        tag = f"{path} {name} remap={remap}"
                        self.assertTrue(res["enabled"], tag)
                        self.assertEqual(res["seqs"], base["seqs"], tag)
                        if name == "inert":
                            self.assertIn("capping=off (inert)", err, tag)
                            self.assertEqual(res["remap_recorded"], 0, tag)
                        else:
                            if remap == "1":
                                self.assertGreater(res["remap_recorded"], 0, tag)
                                self.assertLess(res["mean_union"], base["mean_union"], tag)
                            if path == "epilogue":
                                self.assertGreater(res["static_accept_recorded"], 0, tag)
                            else:
                                self.assertGreater(res["cutoff_injected"], 0, tag)
        finally:
            for p in paths.values():
                os.unlink(p)


if __name__ == "__main__":
    unittest.main()
