"""Draft temperature (tau:X / INKLING_DRAFT_TAU): option parsing, the patch plumbing
against the CPU stubs (fresh interpreter per case), and a Monte Carlo law test.

  python -m unittest discover -s tests/draftbytes -p 'test_tau.py' -v   (from overlay/)
"""

import json
import os
import subprocess
import sys
import textwrap
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

from inkling_draftbytes import _opts  # noqa: E402


def _run(code, options="tau:0.7", extra_env=None, expect_ok=True):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(HERE, "stubs"), ROOT])
    env.pop("INKLING_DRAFT_TAU", None)
    if options is None:
        env.pop("INKLING_DRAFT_BYTES", None)
    else:
        env["INKLING_DRAFT_BYTES"] = options
    env.update(extra_env or {})
    proc = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                          capture_output=True, text=True, timeout=300)
    if not expect_ok:
        return proc
    if proc.returncode != 0:
        raise AssertionError(f"[{options}] failed:\n{proc.stderr[-3000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


_RESOLVE = """
import json, types, torch
import inkling_draftbytes
opts = inkling_draftbytes.install()
from sglang.srt.speculative.dspark_components import dspark_draft as dd
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds
from inkling_draftbytes import patches
si = types.SimpleNamespace(temperatures=torch.tensor([0.8, 0.0, 1.0, 1e-7]),
                           top_ks=torch.tensor([1 << 30, 1, 50, 1 << 30]))
fresh = dd.resolve_draft_params(sampling_info=si, bs=4, device="cpu")
sampler = ds.DsparkDraftSampler(max_bs=6)
sampler.params.step_temperatures[4:].fill_(9.0)
sampler.stage_sampling_params(bs=4, sampling_info=si)
sampler.stage_sampling_params(bs=4, sampling_info=si)  # restage: no compounding
greedy = dd.resolve_draft_params(sampling_info=None, bs=3, device="cpu")
print(json.dumps({
    "describe": opts.describe() if opts else None,
    "patched": patches.STATE.patched,
    "temps": fresh.temperatures.tolist(),
    "step": fresh.step_temperatures.tolist(),
    "same_bits": bool(torch.equal(fresh.temperatures[fresh.step_temperatures != 0],
                                  fresh.step_temperatures[fresh.step_temperatures != 0])),
    "sampler_temps": sampler.params.temperatures[:4].tolist(),
    "sampler_step": sampler.params.step_temperatures.tolist(),
    "sampler_is_patched": ds.resolve_draft_params is dd.resolve_draft_params,
    "greedy_temps": greedy.temperatures.tolist(),
    "greedy_step": greedy.step_temperatures.tolist(),
}))
"""


class TauOptionTests(unittest.TestCase):
    def test_token_and_env(self):
        self.assertEqual(_opts.parse("tau:0.7", environ={}).tau, 0.7)
        self.assertTrue(_opts.parse("tau:0.7", environ={}).active)
        self.assertEqual(_opts.parse("", environ={"INKLING_DRAFT_TAU": "0.8"}).tau, 0.8)
        o = _opts.parse("gemv", environ={"INKLING_DRAFT_TAU": "0.75"})
        self.assertEqual((o.gemv, o.tau), (True, 0.75))
        self.assertIn("tau:0.75", o.describe())
        self.assertEqual(_opts.parse("gemv", environ={}).tau, 1.0)
        self.assertFalse(_opts.parse("tau:1", environ={}).active)
        # same value in both places is fine
        self.assertEqual(_opts.parse("tau:0.7", environ={"INKLING_DRAFT_TAU": "0.7"}).tau, 0.7)

    def test_errors(self):
        for text, env in (("tau", {}), ("tau:x", {}), ("tau:0", {}), ("tau:9", {}),
                          ("tau:0.7", {"INKLING_DRAFT_TAU": "0.8"}),
                          ("", {"INKLING_DRAFT_TAU": "-1"})):
            with self.assertRaises(_opts.OptionError, msg=(text, env)):
                _opts.parse(text, environ=env)


class TauPatchTests(unittest.TestCase):
    def test_resolve_scaled_consistently(self):
        r = _run(_RESOLVE)
        self.assertEqual(r["describe"], "tau:0.7")
        for name in ("dspark_draft.resolve_draft_params",):
            self.assertIn(name, r["patched"])
        f32 = lambda x: torch.tensor(x, dtype=torch.float32)  # noqa: E731
        want = f32([0.8, 1e-5, 1.0, 1e-5]) * 0.7
        self.assertTrue(torch.equal(f32(r["temps"]), want), r["temps"])
        # greedy row (top_k 1) keeps step temperature 0; sampled rows draft at tau*T
        self.assertEqual(r["step"][1], 0.0)
        self.assertTrue(torch.equal(f32(r["step"])[[0, 2, 3]], want[[0, 2, 3]]), r["step"])
        self.assertTrue(r["same_bits"])
        self.assertTrue(torch.equal(f32(r["sampler_temps"]), want))
        # rows past bs untouched; staged twice -> still tau^1
        self.assertEqual(r["sampler_step"][4:], [9.0, 9.0])
        self.assertTrue(torch.equal(f32(r["sampler_step"][:4]), f32(r["step"])))
        self.assertTrue(r["sampler_is_patched"])
        # no sampling_info (all greedy): stock values
        self.assertEqual(r["greedy_temps"], [1.0, 1.0, 1.0])
        self.assertEqual(r["greedy_step"], [0.0, 0.0, 0.0])

    def test_env_form_and_stack_with_hotvocab_hook(self):
        r = _run(_RESOLVE, options="gemv,hotvocab:2000",
                 extra_env={"INKLING_DRAFT_TAU": "0.7",
                            "INKLING_DRAFT_HOTVOCAB_FILE": "/nonexistent-not-read-here"})
        self.assertIn("tau:0.7", r["describe"])
        self.assertTrue(r["same_bits"])
        # both dspark_draft patches applied through the one post-import callback
        self.assertIn("dspark_draft._draft_step", r["patched"])
        self.assertIn("dspark_draft.resolve_draft_params", r["patched"])

    def test_off_is_stock(self):
        r = _run(_RESOLVE, options="gemv")
        self.assertNotIn("dspark_draft.resolve_draft_params", r["patched"])
        self.assertEqual(r["temps"][0], torch.tensor(0.8).item())

    def test_sampling_accept_graph_refused(self):
        code = """
        import inkling_draftbytes
        inkling_draftbytes.install()
        from sglang.srt.speculative.dspark_components import dspark_sampling_accept_graph as m
        m.DsparkSamplingAcceptGraph(epilogue=None)
        """
        proc = _run(code, expect_ok=False)
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("sampling-accept CUDA graph", proc.stderr)
        ok = _run("""
        import json, inkling_draftbytes
        inkling_draftbytes.install()
        from sglang.srt.speculative.dspark_components import dspark_sampling_accept_graph as m
        m.DsparkSamplingAcceptGraph(epilogue=None)
        print(json.dumps({"ok": True}))
        """, options="gemv")
        self.assertTrue(ok["ok"])


class TauLawTest(unittest.TestCase):
    """Multi-step Monte Carlo of the stock chain accept (accept d with prob
    min(1, p(d)/q(d)), else a residual draw from max(p - q, 0); all accepted -> draw
    from the last row's p). Target: order-1 Markov at T = 0.8 from fixed logits; the
    draft draws at tau x T from different logits. The accept reads q at tau x T (what
    the patch does): the first three tokens follow the target law. Negative control:
    the accept reads q at T while the draft drew at tau x T -> detectably biased."""

    ZT = torch.tensor([[1.2, 0.4, -0.8, 0.1], [0.0, 1.0, 0.3, -0.5],
                       [-0.3, 0.2, 1.1, 0.4], [0.6, -0.6, 0.2, 0.9]], dtype=torch.float64)
    ZD = torch.tensor([[1.0, 0.6, -0.5, 0.0], [0.2, 1.2, 0.0, -0.2],
                       [-0.5, 0.4, 0.9, 0.6], [0.8, -0.3, 0.0, 1.0]], dtype=torch.float64)
    T = 0.8

    def _simulate(self, n, tau, q_temp_factor, gamma=4, out_len=3, seed=0):
        gen = torch.Generator().manual_seed(seed)
        P = torch.softmax(self.ZT / self.T, -1)
        Qdraw = torch.softmax(self.ZD / (tau * self.T), -1)
        Qacc = torch.softmax(self.ZD / (q_temp_factor * self.T), -1)
        out = torch.full((n, out_len), -1, dtype=torch.long)
        filled = torch.zeros(n, dtype=torch.long)
        anchor = torch.zeros(n, dtype=torch.long)
        rows = torch.arange(n)
        accepted = 0
        steps = 0
        while bool((filled < out_len).any()):
            prev, drafts, prevs = anchor, [], []
            for _ in range(gamma):
                d = torch.multinomial(Qdraw[prev], 1, generator=gen).squeeze(1)
                prevs.append(prev)
                drafts.append(d)
                prev = d
            drafts, prevs = torch.stack(drafts, 1), torch.stack(prevs, 1)
            pd = P[prevs, drafts]
            qd = Qacc[prevs, drafts]
            u = torch.rand((n, gamma), generator=gen, dtype=torch.float64)
            ok = u * qd < pd
            acc = torch.cumprod(ok.long(), 1).sum(1)
            row_tok = torch.cat([anchor[:, None], drafts], 1)[rows, acc]
            p_row = P[row_tok]
            q_row = Qacc[row_tok]
            resid = torch.where((acc == gamma)[:, None], p_row, (p_row - q_row).clamp(min=0))
            bonus = torch.multinomial(resid / resid.sum(1, keepdim=True), 1,
                                      generator=gen).squeeze(1)
            emitted = torch.cat([drafts, bonus[:, None]], 1)
            n_emit = acc + 1
            active = filled < out_len
            accepted += int(acc[active].sum())
            steps += int(active.sum())
            for k in range(gamma + 1):
                take = active & (k < n_emit) & (filled + k < out_len)
                tok = torch.where(k == n_emit - 1, bonus, emitted[:, k])
                idx = (filled + k).clamp(max=out_len - 1)
                out[take, idx[take]] = tok[take]
            filled = torch.where(active, filled + n_emit, filled)
            anchor = torch.where(active, bonus, anchor)
        V = P.shape[0]
        emp = torch.zeros(V, V, V, dtype=torch.float64)
        emp.index_put_((out[:, 0], out[:, 1], out[:, 2]),
                       torch.ones(n, dtype=torch.float64), accumulate=True)
        emp /= n
        exact = torch.einsum("a,ab,bc->abc", P[0], P, P)
        return float((emp - exact).abs().max()), accepted / steps

    def test_tau_with_matching_q_is_exact(self):
        for tau in (0.7, 0.5, 1.3):
            err, _ = self._simulate(300_000, tau, tau, seed=int(tau * 10))
            self.assertLess(err, 3.5e-3, (tau, err))
            print(f"tau {tau}: max abs error {err:.2e}", file=sys.stderr)

    def test_mismatched_q_is_biased(self):
        err, _ = self._simulate(300_000, 0.5, 1.0, seed=3)
        print(f"negative control: max abs error {err:.2e}", file=sys.stderr)
        self.assertGreater(err, 1e-2, err)


if __name__ == "__main__":
    unittest.main()
