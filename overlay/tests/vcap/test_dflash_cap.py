"""The verify cap on the DFLASH worker, CPU stand-in:

  python -m unittest discover -s tests/vcap -p 'test_dflash_cap.py' -v   (from overlay/)

- fixed:3: every step verifies 3 rows; the accept is exactly min(uncapped accept, 2) (the stock cutoff contract);
  the MoE experts read per visit are exactly the union of the LIVE rows (dead rows remapped to their anchor, which
  is live), recomputed on the unpatched toy target.
- fixed:8 never caps: the overlay is inert (tokens identical to off).
- a capped step that leaves without _accept_block is refused (exactness guard).
- dfpolicy: tools/dfreplay.py's export loads in dfcap, and dfcap.lens_policy (runtime, torch) equals the replay's
  widths for the same requests (survival tables, thresholds, bins)."""

import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
TOOLS = os.path.join(os.path.dirname(OVERLAY), "tools")
sys.path.insert(0, OVERLAY)
sys.path.insert(0, TOOLS)

from inkling_vcap import dfcap  # noqa: E402


def _run(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("INKLING_")}
    env["PYTHONPATH"] = os.pathsep.join([OVERLAY, os.path.join(HERE, "fake_sglang")])
    env.update(extra)
    return subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=env,
                          capture_output=True, text=True, timeout=600)


def _ok(extra):
    out = _run(extra)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


class CapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfcap-")
        base = {"FAKE_TEMPS": "0,0,0", "FAKE_STEPS": "30"}
        cls.off, _ = _ok(base)
        cls.hist3 = os.path.join(cls.tmp, "h3.pt")
        cls.cap3, cls.err3 = _ok(dict(base, INKLING_VERIFY_CAP="fixed:3", FAKE_HISTORY=cls.hist3))
        cls.cap8, _ = _ok(dict(base, INKLING_VERIFY_CAP="fixed:8"))

    def test_fixed3_accept_contract(self):
        self.assertTrue(self.cap3["enabled"])
        self.assertIn("capping=on", self.err3)
        hist = torch.load(self.hist3, weights_only=False)
        capped_any = False
        for h in hist:
            self.assertTrue(torch.equal(h["live_lens"], torch.full((h["bs"],), 3)))
            want = torch.minimum(h["uncapped_accept"], torch.tensor(2, dtype=h["uncapped_accept"].dtype))
            self.assertTrue(torch.equal(h["accept_len"], want))
            capped_any |= bool((h["uncapped_accept"] > 2).any())
        self.assertTrue(capped_any)  # the cap actually trimmed something

    def test_remap_reads_only_live_experts(self):
        sys.path.insert(0, os.path.join(HERE, "fake_sglang"))
        from sglang.srt.speculative.dflash_worker_v2 import ToyTarget  # unpatched

        model = ToyTarget()
        hist = torch.load(self.hist3, weights_only=False)
        for h in hist[:10]:
            bs = h["bs"]
            x = model.emb[h["draft_tokens"].reshape(-1)]
            live = (torch.arange(8)[None, :] < 3).expand(bs, 8).reshape(-1)
            for v, li in enumerate(model.schedule):
                _, ids = model.layers[li].gate(x)
                want = sorted(set(ids[live].reshape(-1).tolist()))
                self.assertEqual(h["touched"][v], want)
                x = x + model.layers[li](x, None)  # live rows only matter (dead rows differ, never read)

    def test_dead_rows_ungraded_in_statistics(self):
        """The capped greedy accept grades only live rows: cap_trim_lens = min(u, L) - min(u, L - 1), where u is the
        count over every row (dead rows included, as the stock AcceptGreedy would count); tokens are as before."""
        hist = torch.load(self.hist3, weights_only=False)
        through_dead = 0
        for h in hist:
            u, L = h["uncapped_accept"].to(torch.int64), h["live_lens"].to(torch.int64)
            want = torch.minimum(u, L) - torch.minimum(u, L - 1)
            self.assertTrue(torch.equal(h["cap_trim_lens"].to(torch.int64), want))
            through_dead += int((u > L).sum())
        self.assertGreater(through_dead, 0)  # without the mask cap_trim_lens would have counted dead-row matches

    def test_mask_dead_predictions(self):
        from inkling_vcap import dflash, logic

        lens = torch.tensor([3, 8, 1, 0])  # the last one: a graph-padding request
        layout = logic.RowLayout(4, 8, "cpu")
        src = layout.rows.clone()
        logic.build_src_index(lens, layout, out=src)
        tp = torch.arange(24).view(3, 8) + 100
        got = dflash.mask_dead_predictions(tp, src, 3, 8)
        want = tp.clone()
        want[0, 3:] = -1
        want[2, 1:] = -1
        self.assertTrue(torch.equal(got, want))

    def test_fixed8_inert(self):
        self.assertEqual(self.cap8["emitted"], self.off["emitted"])

    def test_custom_logit_processor_runs_uncapped(self):
        hist = os.path.join(self.tmp, "hlp.pt")
        res, _ = _ok({"FAKE_TEMPS": "0,0", "FAKE_STEPS": "10", "INKLING_VERIFY_CAP": "fixed:3", "FAKE_CUSTOM_LP": "1",
                      "FAKE_HISTORY": hist})
        self.assertTrue(res["enabled"])
        for h in torch.load(hist, weights_only=False):
            self.assertIsNone(h["live_lens"])
            self.assertTrue(torch.equal(h["accept_len"], h["uncapped_accept"]))

    def test_bypass_refused(self):
        out = _run({"FAKE_TEMPS": "0,0", "FAKE_STEPS": "6", "INKLING_VERIFY_CAP": "fixed:3", "FAKE_SKIP": "3"})
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("bypassed _accept_block", out.stderr)


class PolicyParityTest(unittest.TestCase):
    def test_replay_export_equals_runtime(self):
        import dfreplay
        import test_dfreplay

        d = tempfile.mkdtemp(prefix="dfpar-")
        test_dfreplay.synth(d, runs=4, steps=150, bs_list=(1, 4))
        pol = os.path.join(d, "policy.json")
        subprocess.run([sys.executable, os.path.join(TOOLS, "dfreplay.py"), d, "--visits", "8", "--bootstrap", "10",
                        "--export-policy", pol, "--json", os.path.join(d, "r.json")], check=True,
                       capture_output=True, text=True)
        p = dfcap.load_df_policy(pol)
        dev = dfcap.DfPolicyDevice(p, "cpu")
        self.assertTrue(any(t > 0 for _, t in p.theta))
        # replay side, same tables
        R, S = dfreplay.load([d])
        dfreplay.prepare(R, S)
        m, gap = dfreplay.features(R)
        bins = dfreplay.feature_bins(R, m, gap)
        calib = dfreplay.Calib(tuple(p.features), band=p.band_calib)
        calib.tables = [np.asarray(p.calib)]
        n = len(m)
        idx = np.arange(n)
        surv = calib.survival(R, idx, bins)
        checked = 0
        for s_ in range(len(S["bs"])):
            rows = R["by_step"][s_]
            bs = len(rows)
            th = p.theta_for(bs)
            if th <= 0:
                continue
            b = p.band_index(bs)
            want = dfreplay.widths_for(surv[rows], th, np.asarray(p.rowcost[p.bands[b]]))
            got = dfcap.lens_policy(torch.tensor(m[rows], dtype=torch.float32),
                                    torch.tensor(gap[rows], dtype=torch.float32),
                                    torch.tensor(~R["greedy"][rows]), b, dev)
            self.assertEqual(got.tolist(), want.tolist())
            checked += 1
        self.assertGreater(checked, 100)


class ModeContextPriceParityTest(unittest.TestCase):
    """theta_x (per band, greedy/sampled, context bucket) + ctx_edges: dfcap.lens_policy (runtime) equals the
    replay's per-request price lookup (dfreplay.theta_of) and widths, request by request."""

    def test_runtime_equals_replay(self):
        import argparse

        import dfreplay

        g = np.random.default_rng(5)
        G, nm = 7, len(dfreplay.M_EDGES) + 1
        calib = g.uniform(0.3, 0.98, size=2 * G * nm)  # strata (mode) x slot x m-bin
        bands = list(dfreplay.BANDS)
        rowcost = {b: list(g.uniform(0.5, 4.0, size=G)) for b in bands}
        theta = [[b, float(g.uniform(0.02, 0.3))] for b in bands]
        theta_x = [[1, 0, 2, 0.05], [1, 1, 0, 0.0], [4, 1, 0, 0.12], [4, 0, 1, 0.2], [8, 1, 0, 0.0]]
        obj = dict(format=dfcap.FORMAT, stride=8, bands=bands, features=["slot", "m"], band_calib=False,
                   edges=dict(m=list(dfreplay.M_EDGES)), calib=calib.tolist(), theta=theta, theta_x=theta_x,
                   ctx_edges=list(dfreplay.CTX_EDGES), rowcost={str(b): v for b, v in rowcost.items()}, bsmax=48)
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(obj, fh)
        pol = dfcap.load_df_policy(fh.name)
        dev = dfcap.DfPolicyDevice(pol, "cpu")
        rp = dict(theta={bands.index(b): t for b, t in theta},
                  theta_x={(bands.index(b), m, c): t for b, m, c, t in theta_x},
                  rowcost={bands.index(b): np.asarray(v) for b, v in rowcost.items()}, bsmax=48)
        table = calib.reshape(2, G, nm)
        checked = 0
        for bs in (1, 3, 4, 8):
            for _ in range(40):
                m = g.uniform(0, 1, size=(bs, G)).astype(np.float32)
                sampled = g.random(bs) < 0.5
                ctx = g.choice([100, 16383, 16384, 40000, 65535, 65536, 200000], size=bs)
                band = dfreplay.band_of(bs)
                # replay side
                mb = np.digitize(m, dfreplay.M_EDGES)
                surv = np.cumprod(table[sampled.astype(int)[:, None], np.arange(G)[None, :], mb], axis=1)
                cb = np.digitize(ctx, dfreplay.CTX_EDGES)
                ths = np.array([dfreplay.theta_of(rp, band, int(sampled[i]), int(cb[i])) for i in range(bs)])
                want = dfreplay.widths_for(surv, ths, rp["rowcost"][band])
                got = dfcap.lens_policy(torch.tensor(m), torch.zeros(bs, G), torch.tensor(sampled), band, dev,
                                        context=torch.tensor(ctx))
                self.assertEqual(got.tolist(), want.tolist())
                checked += 1
        self.assertEqual(checked, 160)
        # host decision: a band caps when any (mode, bucket) price is positive
        self.assertGreater(pol.theta_for(8), 0.0)  # greedy band-8 price from theta
        obj2 = dict(obj, theta=[[b, 0.0] for b in bands], theta_x=[[8, 1, 0, 0.0]])
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump(obj2, fh)
        self.assertEqual(dfcap.load_df_policy(fh.name).theta_for(8), 0.0)


if __name__ == "__main__":
    unittest.main()
