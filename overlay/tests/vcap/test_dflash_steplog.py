"""DFLASH full-width capture (inkling_vcap/dflash.py + dfsteplog.py), CPU.

  python -m unittest discover -s tests/vcap -p 'test_dflash_steplog.py' -v   (from overlay/)

- features: the walked selector rows and (m, gap) against a per-request loop reference; greedy rows at T = 1;
  the sample_path wrapper returns the stock walk's outputs unchanged.
- end to end on the CPU stand-in (fake_sglang .../dflash_worker_v2.py) with the overlay armed by sitecustomize:
  emitted tokens identical with the capture on and off (measurement only); every logged column against the
  worker's own reference history (rids, contexts, temperatures, rows, m, gap, accepted, unions per visit and width);
  a step that bypasses _accept_block is dropped without corrupting the next ones.
- source pins: the hooks' signature / source checks run against the pinned image's dflash_worker_v2.py and
  models/dflash.py AFTER the DFlash2 compatibility patch (applied to a temporary copy), when both are available
  (INKLING_IMG_SRC, INKLING_DFLASH2_PATCH)."""

import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

from inkling_vcap import dflash  # noqa: E402
from inkling_vcap.dflash import DflashFeatures, make_sample_path, row_features, walked_rows  # noqa: E402

# A copy of the pinned image's sglang package directory and the DFlash2 compatibility patch (downloaded with the
# drafter); the source tests below skip when either is not given.
IMG_SRC = os.environ.get("INKLING_IMG_SRC", "")
DFLASH2_PATCH = os.environ.get("INKLING_DFLASH2_PATCH", "")


def _ref_rows(candidate_ids, scores, tokens):
    bs, gamma, k = candidate_ids.shape
    out = torch.empty(bs, gamma, k)
    for b in range(bs):
        prev = None
        for j in range(gamma):
            lane = int((candidate_ids[b, j] == tokens[b, j]).nonzero()[0, 0])
            out[b, j] = scores[b, 0, 0] if j == 0 else scores[b, j, prev]
            prev = lane
    return out


def _ref_features(rows, t_eff):
    m = torch.empty(rows.shape[:2])
    gap = torch.empty(rows.shape[:2])
    for b in range(rows.shape[0]):
        for j in range(rows.shape[1]):
            q = torch.softmax(rows[b, j].double() / float(t_eff[b]), dim=-1)
            s = torch.sort(q, descending=True).values
            m[b, j], gap[b, j] = float(s[0]), float(s[0] - s[1])
    return m, gap


class FeatureTest(unittest.TestCase):
    def setUp(self):
        from sglang_fake_selector import CandidateSelector  # noqa: F401 (path set in setUpModule)

    def _walk(self, bs=5, gamma=7, k=16, seed=0):
        from sglang_fake_selector import CandidateSelector

        g = torch.Generator().manual_seed(seed)
        cand = torch.stack([torch.stack([torch.randperm(64, generator=g)[:k] for _ in range(gamma)])
                            for _ in range(bs)])
        scores = torch.randn(bs, gamma, k, k, generator=g) * 2
        scores[0, 2, :, 5] = float("-inf")  # masked candidate lanes are allowed
        temps = torch.tensor([1e-5, 0.8, 1.3, 0.6, 1.0])[:bs]
        greedy = torch.tensor([True, False, False, True, False])[:bs]
        uni = torch.rand(bs, gamma, generator=g)
        sel = CandidateSelector()
        tokens, q = sel.sample_path(candidate_ids=cand, scores=scores, uniforms=uni, temperatures=temps,
                                    greedy_mask=greedy)
        return sel, cand, scores, uni, temps, greedy, tokens, q

    def test_rows_and_features_match_reference(self):
        _, cand, scores, _, temps, greedy, tokens, _ = self._walk()
        rows = walked_rows(cand, scores, tokens)
        self.assertTrue(torch.equal(rows, _ref_rows(cand, scores, tokens)))
        t_eff = dflash.effective_temperatures(temps, greedy)
        self.assertTrue(torch.equal(t_eff, torch.where(greedy, torch.ones(5), temps)))
        m, gap = row_features(rows, t_eff)
        rm, rgap = _ref_features(rows, t_eff)
        self.assertLess(float((m - rm).abs().max()), 1e-6)
        self.assertLess(float((gap - rgap).abs().max()), 1e-6)
        self.assertTrue(bool(((m >= 1 / 16 - 1e-6) & (m <= 1)).all()))

    def test_sampled_m_is_the_walked_q_max(self):
        # For sampled rows the walk's own q row is softmax(row / T): m must be its max.
        _, cand, scores, _, temps, greedy, tokens, q = self._walk()
        m, _ = row_features(walked_rows(cand, scores, tokens), dflash.effective_temperatures(temps, greedy))
        sampled = ~greedy
        self.assertLess(float((m[sampled] - q[sampled].max(dim=-1).values).abs().max()), 1e-6)

    def test_buffers_wrapper_and_capacity(self):
        sel, cand, scores, uni, temps, greedy, tokens, q = self._walk()
        feats = DflashFeatures(capacity=8, gamma=7, top_k=16, device="cpu", keep_rows=True)
        from inkling_vcap.state import STATE

        old = (STATE.enabled, STATE.df)
        STATE.enabled, STATE.df = True, feats
        try:
            wrapped = make_sample_path(sel.sample_path)
            t2, q2 = wrapped(candidate_ids=cand, scores=scores, uniforms=uni, temperatures=temps, greedy_mask=greedy)
            self.assertTrue(torch.equal(t2, tokens) and torch.equal(q2, q))
            self.assertEqual(int(feats.seq), 1)
            rows = walked_rows(cand, scores, tokens)
            self.assertTrue(torch.equal(feats.rows[:5], rows))
            self.assertTrue(torch.equal(feats.greedy[:5], greedy))
            m, gap = row_features(rows, dflash.effective_temperatures(temps, greedy))
            self.assertTrue(torch.equal(feats.m[:5], m) and torch.equal(feats.gap[:5], gap))
            small = DflashFeatures(capacity=4, gamma=7, top_k=16, device="cpu", keep_rows=False)
            STATE.df = small
            wrapped(candidate_ids=cand, scores=scores, uniforms=uni, temperatures=temps, greedy_mask=greedy)
            self.assertEqual((small.skipped, int(small.seq)), (1, 0))
        finally:
            STATE.enabled, STATE.df = old


def _env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("INKLING_")}
    env["PYTHONPATH"] = os.pathsep.join([ROOT, os.path.join(HERE, "fake_sglang")])
    env.update(extra)
    return env


def _run(extra):
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=_env(extra),
                         capture_output=True, text=True, timeout=600)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-4000:])
    return json.loads(out.stdout.strip().splitlines()[-1]), out.stderr


def _load(directory):
    files = sorted(glob.glob(os.path.join(directory, "dflash1-*.npz")))
    parts = [dict(np.load(f, allow_pickle=False)) for f in files]
    cat = {}
    for key in ("step", "bs", "rid", "rows", "m", "gap", "t_eff", "greedy", "temp", "context", "accepted", "union",
                "visits", "walks", "t_draft_ms"):
        cat[key] = np.concatenate([p[key] for p in parts])
    cat["format"] = {str(p["format"]) for p in parts}
    return files, cat


class EndToEndTest(unittest.TestCase):
    TEMPS = "0,0.8,0,1.2"
    STEPS = "40"

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfsteplog-")
        cls.off, _ = _run({"FAKE_TEMPS": cls.TEMPS, "FAKE_STEPS": cls.STEPS})
        cls.dir_on = os.path.join(cls.tmp, "on")
        cls.hist_on = os.path.join(cls.tmp, "hist_on.pt")
        cls.on, cls.err_on = _run({"FAKE_TEMPS": cls.TEMPS, "FAKE_STEPS": cls.STEPS, "INKLING_VERIFY_CAP": "steplog",
                                   "INKLING_VERIFY_CAP_STEPLOG": cls.dir_on, "FAKE_HISTORY": cls.hist_on})
        cls.dir_skip = os.path.join(cls.tmp, "skip")
        cls.hist_skip = os.path.join(cls.tmp, "hist_skip.pt")
        cls.skip, _ = _run({"FAKE_TEMPS": cls.TEMPS, "FAKE_STEPS": "20", "FAKE_SKIP": "5,6",
                            "INKLING_VERIFY_CAP": "steplog", "INKLING_VERIFY_CAP_STEPLOG": cls.dir_skip,
                            "FAKE_HISTORY": cls.hist_skip})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_armed_and_measurement_only(self):
        self.assertFalse(self.off["enabled"])
        self.assertTrue(self.on["enabled"])
        self.assertEqual(self.on["algorithm"], "dflash")
        for p in dflash.REQUIRED_DFLASH:
            self.assertIn(p, self.on["applied"])
        self.assertEqual(self.on["emitted"], self.off["emitted"])
        self.assertIn("DFLASH attached", self.err_on)
        self.assertEqual(self.on["steps"], int(self.STEPS))
        self.assertEqual(sum(self.on["dropped"].values()), 0)
        self.assertEqual((self.on["missing_routes"], self.on["stale"]), (0, 0))

    def test_columns_match_reference(self):
        files, d = _load(self.dir_on)
        self.assertEqual(d["format"], {"inkling-dflash-steplog/1"})
        hist = torch.load(self.hist_on, weights_only=False)
        self.assertEqual(len(d["step"]), len(hist))
        bs = hist[0]["bs"]
        self.assertTrue((d["bs"] == bs).all())
        self.assertTrue((d["walks"] == 1).all())
        self.assertTrue(np.isnan(d["t_draft_ms"]).all())  # CPU: no CUDA events
        self.assertTrue((d["visits"] == 3).all())
        for s, h in enumerate(hist):
            rs = slice(s * bs, (s + 1) * bs)
            self.assertEqual(list(d["rid"][rs]), h["rids"])
            self.assertTrue(np.array_equal(d["context"][rs], h["prefix_lens"].numpy()))
            greedy = h["greedy"].numpy()
            self.assertTrue(np.array_equal(d["greedy"][rs], greedy))
            self.assertTrue(np.allclose(d["temp"][rs], np.where(greedy, 0.0, h["req_temps"].numpy())))
            rows = walked_rows(h["candidate_ids"], h["scores"], h["tokens"])
            self.assertTrue(np.array_equal(d["rows"][rs], rows.numpy()))
            t_eff = dflash.effective_temperatures(h["temps"], h["greedy"])
            self.assertTrue(np.array_equal(d["t_eff"][rs], t_eff.numpy()))
            m, gap = row_features(rows, t_eff)
            self.assertTrue(np.array_equal(d["m"][rs], m.numpy()) and np.array_equal(d["gap"][rs], gap.numpy()))
            self.assertTrue(np.array_equal(d["accepted"][rs], h["accept_len"].numpy()))
            u = d["union"][s]
            for v, experts in enumerate(h["touched"]):
                self.assertEqual(int(u[v, 7]), len(experts))  # width 8 = every real row
            self.assertTrue((u[3:] == -1).all())
            self.assertTrue((np.diff(u[:3], axis=1) >= 0).all())
            self.assertTrue((u[:3, 0] >= 2).all())  # top-2 toy router: >= 2 experts at width 1
        acc = d["accepted"]
        self.assertTrue(acc.min() == 0 and acc.max() >= 3)  # the toy target accepts a spread of lengths

    def test_lattice_and_target_fields(self):
        sys.path.insert(0, os.path.join(HERE, "fake_sglang"))
        from sglang.srt.models import dflash as mdf  # the unpatched CPU stand-in

        files = sorted(glob.glob(os.path.join(self.dir_on, "dflash1-*.npz")))
        z = {k: np.concatenate([np.load(f)[k] for f in files]) for k in
             ("cand_ids", "scores", "unary", "lse", "drafted", "committed", "commit_len", "tgt_p1", "tgt_gap",
              "prompt_len", "prompt_hash", "accepted", "tgt_cand_p")}
        hist = torch.load(self.hist_on, weights_only=False)
        dm = mdf.DFlash2DraftModel(64, 16, 16)
        bs = hist[0]["bs"]
        for s, h in enumerate(hist):
            rs = slice(s * bs, (s + 1) * bs)
            self.assertTrue(np.array_equal(z["cand_ids"][rs], h["candidate_ids"].numpy()))
            self.assertTrue(np.array_equal(z["scores"][rs], h["scores"].numpy().astype(np.float16)))
            self.assertTrue(np.allclose(z["unary"][rs], h["unary"].numpy(), atol=1e-6))
            lse = torch.logsumexp(dm._transform_unary_logits(h["plane"].clone()), dim=-1).view(bs, 7)
            self.assertTrue(np.allclose(z["lse"][rs], lse.numpy(), atol=1e-5))
            # tail mass is a probability: the top-16 never exceed the plane
            tail = 1 - np.exp(z["unary"][rs] - z["lse"][rs][..., None]).sum(-1)
            self.assertTrue(((tail >= -1e-5) & (tail <= 1)).all())
            self.assertTrue(np.array_equal(z["drafted"][rs], h["tokens"].numpy()))
            self.assertTrue(np.array_equal(z["committed"][rs], h["out_tokens"].numpy()))
            self.assertTrue(np.array_equal(z["commit_len"][rs], h["commit_lens"].numpy()))
            x = h["logits"].float().view(bs, 8, -1)
            p = torch.softmax(x, dim=-1)
            top = p.topk(2, dim=-1).values
            self.assertTrue(np.allclose(z["tgt_p1"][rs], top[..., 0].numpy(), atol=1e-5))
            self.assertTrue(np.allclose(z["tgt_gap"][rs], (top[..., 0] - top[..., 1]).numpy(), atol=1e-5))
            self.assertEqual(z["prompt_len"][rs].tolist(), [17 + 5 * i for i in range(bs)])
            # exact-BV replay input: P of row j (request temperature) on slot j's candidates; greedy rows NaN
            T = torch.where(h["req_temps"] > 0, h["req_temps"], torch.ones_like(h["req_temps"]))
            pref = torch.softmax(x[:, :7] / T[:, None, None], dim=-1).gather(-1, h["candidate_ids"])
            for i in range(bs):
                if bool(h["greedy"][i]):
                    self.assertTrue(np.isnan(z["tgt_cand_p"][rs][i]).all())
                else:
                    self.assertTrue(np.allclose(z["tgt_cand_p"][rs][i], pref[i].numpy(), atol=1e-6))
            self.assertEqual(len(set(z["prompt_hash"][rs].tolist())), bs)

    def test_width_unions_recomputed(self):
        # Width-w union = distinct experts over rows 0..w-1 of every request: recompute from the gate directly.
        _, d = _load(self.dir_on)
        hist = torch.load(self.hist_on, weights_only=False)
        sys.path.insert(0, os.path.join(HERE, "fake_sglang"))
        from sglang.srt.speculative.dflash_worker_v2 import ToyTarget  # the unpatched CPU stand-in

        model = ToyTarget()
        for s in (0, 7, 19):
            h = hist[s]
            ids_seen = []
            x = model.emb[h["draft_tokens"].reshape(-1)]
            for li in model.schedule:
                _, ids = model.layers[li].gate(x)
                ids_seen.append(ids.view(h["bs"], 8, -1))
                x = x + model.layers[li](x, None)
            for v, ids in enumerate(ids_seen):
                for w in range(1, 9):
                    self.assertEqual(int(d["union"][s][v, w - 1]), len(set(ids[:, :w].reshape(-1).tolist())))

    def test_raw_ids(self):
        # bs = 4 <= RAW_IDS_MAX_BS: every request-step keeps its routed ids; they must equal the gate's own ids.
        sys.path.insert(0, os.path.join(HERE, "fake_sglang"))
        from sglang.srt.speculative.dflash_worker_v2 import ToyTarget  # the unpatched CPU stand-in

        files, d = _load(self.dir_on)
        hist = torch.load(self.hist_on, weights_only=False)
        model = ToyTarget()
        bs = hist[0]["bs"]
        seen = 0
        base = 0
        for f in files:
            part = np.load(f)
            n = len(part["rid"])
            self.assertTrue(np.array_equal(part["ids_req"], np.arange(n)))
            for i in range(0, n, bs):
                s = (base + i) // bs
                h = hist[s]
                x = model.emb[h["draft_tokens"].reshape(-1)]
                for v, li in enumerate(model.schedule):
                    _, ids = model.layers[li].gate(x)
                    got = part["ids"][i:i + bs, v, :, :ids.shape[1]].astype(np.int64)
                    self.assertTrue(np.array_equal(got, ids.view(bs, 8, -1).numpy()))
                    self.assertTrue((part["ids"][i:i + bs, v, :, ids.shape[1]:] == 255).all())
                    x = x + model.layers[li](x, None)
                self.assertTrue((part["ids"][i:i + bs, 3:] == 255).all())
                seen += 1
            base += n
        self.assertEqual(seen, len(hist))

    def test_dropped_steps(self):
        self.assertEqual(self.skip["dropped"]["no_accept"], 2)
        self.assertEqual(self.skip["steps"], 18)
        self.assertEqual(self.skip["stale"], 0)
        _, d = _load(self.dir_skip)
        hist = [h for h in torch.load(self.hist_skip, weights_only=False) if h["accept_len"] is not None]
        bs = hist[0]["bs"]
        self.assertEqual(len(d["step"]), len(hist))
        for s, h in enumerate(hist):
            rs = slice(s * bs, (s + 1) * bs)
            self.assertTrue(np.array_equal(d["accepted"][rs], h["accept_len"].numpy()))
            self.assertTrue(np.array_equal(d["context"][rs], h["prefix_lens"].numpy()))
            self.assertEqual(int(d["union"][s][0, 7]), len(h["touched"][0]))


class FailureIsolationTest(unittest.TestCase):
    """Capture failures never change or stop serving."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfsteplog-fail-")
        cls.off, _ = _run({"FAKE_TEMPS": "0,0.8,0,1.2", "FAKE_STEPS": "30"})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _on(self, name, **extra):
        env = {"FAKE_TEMPS": "0,0.8,0,1.2", "FAKE_STEPS": "30", "INKLING_VERIFY_CAP": "steplog",
               "INKLING_VERIFY_CAP_STEPLOG": os.path.join(self.tmp, name)}
        env.update(extra)
        return _run(env)

    def test_writer_failure_keeps_serving(self):
        res, err = self._on("wfail", FAKE_FAIL_FLUSH="1", INKLING_VERIFY_CAP_STEPLOG_CHUNK="4", FAKE_STEP_SLEEP="0.02")
        self.assertEqual(res["emitted"], self.off["emitted"])
        self.assertIn("No space left", res["writer_error"])
        self.assertIsNotNone(res["failed"])  # the next mark sees the writer error and disables the capture
        self.assertIn("CAPTURE DISABLED", err)
        self.assertEqual(res["written"], 0)

    def test_over_capacity_is_skipped(self):
        res, err = self._on("cap", FAKE_CAPACITY="2")
        self.assertEqual(res["emitted"], self.off["emitted"])
        self.assertEqual(res["steps"], 0)
        self.assertEqual(res["dropped"]["over_capacity"], 30)
        self.assertIsNone(res["failed"])

    def test_final_bypass_dropped_and_status(self):
        res, _ = self._on("tail", FAKE_SKIP="29")
        self.assertEqual(res["emitted"][:29], self.off["emitted"][:29])  # step 29 leaves without an accept
        self.assertEqual(res["dropped"]["no_accept"], 1)
        self.assertEqual((res["steps"], res["written"]), (29, 29))
        status = json.load(open(glob.glob(os.path.join(self.tmp, "tail", "status-*.json"))[0]))
        self.assertEqual((status["logged"], status["written"], status["idle"]), (29, 29, True))
        self.assertEqual(status["visits"], {"3": 29})

    def test_backpressure_does_not_block(self):
        import time as _t

        from inkling_vcap import dfsteplog
        from inkling_vcap.dflash import DflashFeatures

        feats = DflashFeatures(capacity=4, gamma=7, top_k=16, device="cpu", keep_rows=True)
        log = dfsteplog.DfStepLog(os.path.join(self.tmp, "bp"), stride=8, top_k=16, capacity=4, device="cpu",
                                  features=feats)
        held = []
        try:
            while not log.free.empty():
                held.append(log.free.get_nowait())
            t0 = _t.time()
            log.mark_draft_start(2)
            self.assertLess(_t.time() - t0, 0.05)  # never waits on the scheduler thread
            self.assertEqual(log.dropped["backpressure"], 1)
            self.assertIsNone(log.slot)
            self.assertIsNone(log.failed)
            log.start(batch=types_ns(reqs=[types_ns(rid="a"), types_ns(rid="b")], sampling_info=None),
                      prefix_lens=torch.tensor([5, 6]))
            self.assertFalse(log.active)
            self.assertEqual(int(log.control[1]), 0)
            self.assertEqual(log.dropped["no_draft_mark"], 0)  # the mark happened; only the slot was missing
            self.assertEqual((log.step, log.step_ids), (0, 1))  # a dropped step consumes a step id
        finally:
            for slot in held:
                log.free.put(slot)
            log.close()

    def test_reused_rid_gets_its_own_prompt(self):
        from inkling_vcap import dfsteplog
        from inkling_vcap.dflash import DflashFeatures

        feats = DflashFeatures(capacity=2, gamma=7, top_k=16, device="cpu", keep_rows=True)
        log = dfsteplog.DfStepLog(os.path.join(self.tmp, "rid"), stride=8, top_k=16, capacity=2, device="cpu",
                                  features=feats)
        try:
            a = log._prompt_key(types_ns(rid="x", origin_input_ids=[1, 2]))
            b = log._prompt_key(types_ns(rid="x", origin_input_ids=[1, 2, 3]))  # same rid, a new request
            self.assertEqual((a[0], b[0]), (2, 3))
            self.assertNotEqual(a[1], b[1])

            class Slotted:
                __slots__ = ("rid", "origin_input_ids")

                def __init__(self, rid, ids):
                    self.rid, self.origin_input_ids = rid, ids

            c = log._prompt_key(Slotted("y", [4, 5, 6, 7]))
            d = log._prompt_key(Slotted("y", [4, 5]))
            self.assertEqual((c[0], d[0]), (4, 2))
        finally:
            log.close()

    def test_target_cand_p_numerics_and_filtering(self):
        from inkling_vcap import dfsteplog
        from inkling_vcap.dflash import DflashFeatures

        feats = DflashFeatures(capacity=2, gamma=7, top_k=16, device="cpu", keep_rows=True)
        log = dfsteplog.DfStepLog(os.path.join(self.tmp, "tcp"), stride=8, top_k=16, capacity=2, device="cpu",
                                  features=feats)
        try:
            V = 40
            feats.cand[:2] = torch.arange(16).expand(2, 7, 16)
            logits = torch.full((16, V), -5.0)
            logits[:, 0] = 3.0
            logits[:, 1] = 3.0                     # a tie at the top
            meta = torch.zeros(2, log.col["filtered"].stop)
            meta[:, log.col["temp"]] = torch.tensor([[1e-6], [0.8]])
            out = log._target_cand_p(logits, 2, meta)
            self.assertTrue(torch.allclose(out[0, :, :2], torch.full((7, 2), 0.5), atol=1e-6))  # low temperature: shift before scaling
            self.assertTrue(torch.allclose(out.sum(-1)[1], torch.softmax(logits[:7] / 0.8, -1)[:, :16].sum(-1)))
            meta[1, log.col["filtered"]] = 1.0     # top-k / top-p / min-p: not recorded
            out = log._target_cand_p(logits, 2, meta)
            self.assertTrue(torch.isnan(out[1]).all() and not torch.isnan(out[0]).any())
        finally:
            log.close()

    def test_partial_journal_invalidates_step(self):
        from inkling_vcap import dfsteplog
        from inkling_vcap.dflash import DflashFeatures

        feats = DflashFeatures(capacity=2, gamma=7, top_k=16, device="cpu", keep_rows=True)
        log = dfsteplog.DfStepLog(os.path.join(self.tmp, "partial"), stride=8, top_k=16, capacity=2, device="cpu",
                                  features=feats)
        try:
            slot = log.free.get_nowait()
            n, V = 2, 48
            routes = torch.randint(0, 128, (V, n, 8, 8), dtype=torch.int16)
            slot.control.copy_(torch.tensor([V, n], dtype=torch.int32))
            slot.seq.copy_(torch.tensor([0, 1]))
            slot.routes[:routes.numel()].copy_(routes.reshape(-1))
            rec = log._digest(0, ("a", "b"), 0.0, slot, [(1, 1), (2, 2)])
            self.assertEqual(rec[4], V)  # complete journal: visits 48
            routes[5, 1, 0, :] = -1      # one real row of one visit never written
            slot.routes[:routes.numel()].copy_(routes.reshape(-1))
            rec = log._digest(1, ("a", "b"), 0.0, slot, [(1, 1), (2, 2)])
            self.assertEqual(rec[4], 0)
            self.assertTrue((rec[3] == -1).all())
            self.assertIsNone(rec[10])
            slot.control.copy_(torch.tensor([V, 3], dtype=torch.int32))  # bs mismatch: not this step's journal
            self.assertEqual(log._digest(2, ("a", "b"), 0.0, slot, [(1, 1), (2, 2)])[4], 0)
            slot.control.copy_(torch.tensor([65, 2], dtype=torch.int32))  # invalid cursor (sharded rows)
            self.assertEqual(log._digest(3, ("a", "b"), 0.0, slot, [(1, 1), (2, 2)])[4], 0)
            log.free.put(slot)
        finally:
            log.close()

    def test_dfcheck_requires_fresh_drain(self):
        import time as _t

        d = os.path.join(self.tmp, "chk")
        res, _ = _run({"FAKE_TEMPS": "0,0.8", "FAKE_STEPS": "120", "INKLING_VERIFY_CAP": "steplog",
                       "INKLING_VERIFY_CAP_STEPLOG": d, "FAKE_STEP_SLEEP": "0.01"})
        self.assertEqual(sum(res["dropped"].values()), 0, res["dropped"])
        env = _env({})

        def check(*args):
            out = subprocess.run([sys.executable, "-m", "inkling_vcap.dfcheck", d, "--visits", "3", *args], env=env,
                                 capture_output=True, text=True)
            return out.returncode, json.loads(out.stdout.strip().splitlines()[-1])

        rc, r = check("--min-steps", "100", "--require-sampled", "--require-batched")
        self.assertEqual((rc, r["ok"], r["complete"], r["drained"]), (0, True, 1.0, True))
        rc, r = check("--min-steps", "100", "--after", str(_t.time() + 5))  # no acknowledgement after that time
        self.assertEqual((rc, r["ok"], r["drained"]), (1, False, False))
        rc, r = check("--min-steps", "100", "--visits", "48")
        self.assertEqual((rc, r["ok"]), (1, False))
        self.assertEqual((r["sampled_req_steps"], r["bs"]), (0, {}))  # coverage counts complete steps only
        rc, r = check("--since-step", "100", "--min-steps", "10")  # a phase's own steps
        self.assertEqual((rc, r["ok"], r["steps"]), (0, True, 20))
        rc, r = check("--since-step", "120", "--min-steps", "0")  # no new steps: never ok
        self.assertEqual((rc, r["ok"], r["steps"]), (1, False, 0))


def types_ns(**kw):
    import types

    return types.SimpleNamespace(**kw)


def _patched_image(tmp):
    """Copy the files the DFlash2 compatibility patch edits, run its apply() on the copy, return the root."""
    rels = ["srt/configs/inkling2.py", "srt/model_executor/model_runner_components/spec_aux_hidden_state.py",
            "srt/models/dflash.py", "srt/models/inkling2.py", "srt/speculative/dflash_utils.py",
            "srt/speculative/dflash_worker_v2.py"]
    root = os.path.join(tmp, "sglang")
    for rel in rels:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        shutil.copy(os.path.join(IMG_SRC, rel), os.path.join(root, rel))
    open(os.path.join(root, "__init__.py"), "w").close()
    code = ("import importlib.util,sys;sys.path.insert(0,sys.argv[2]);"
            "s=importlib.util.spec_from_file_location('p',sys.argv[1]);"
            "m=importlib.util.module_from_spec(s);s.loader.exec_module(m);print(len(m.apply()))")
    out = subprocess.run([sys.executable, "-I", "-c", code, DFLASH2_PATCH, tmp], cwd=tmp, capture_output=True, text=True,
                         env={"PATH": os.environ.get("PATH", "")})
    if out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return root


def _methods(path, cls_name):
    with open(path) as fh:
        src = fh.read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == cls_name:
            out = {}
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    a = item.args
                    out[item.name] = (ast.get_source_segment(src, item),
                                      [x.arg for x in a.posonlyargs + a.args + a.kwonlyargs],
                                      [x.arg for x in a.kwonlyargs])
            return out
    raise AssertionError(f"{cls_name} not in {path}")


def _ordered(src, needles):
    last = -1
    for n in needles:
        i = src.find(n)
        if i < 0 or i < last:
            return False
        last = i
    return True


@unittest.skipUnless(os.path.isdir(IMG_SRC) and os.path.isfile(DFLASH2_PATCH), "pinned image source not available")
class ImagePinTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfpin-")
        cls.root = _patched_image(cls.tmp)
        cls.worker = _methods(os.path.join(cls.root, "srt/speculative/dflash_worker_v2.py"), "DFlashWorkerV2")
        cls.sampler = _methods(os.path.join(cls.root, "srt/speculative/dflash_worker_v2.py"), "_SelectorDraftSampler")
        cls.selector = _methods(os.path.join(cls.root, "srt/models/dflash.py"), "CandidateSelector")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_patch_applied(self):
        self.assertIn("get_dflash_valid_vocab_size", self.worker["__init__"][0])

    def test_forward_order(self):
        self.assertTrue(_ordered(self.worker["forward_batch_generation"][0], dflash.FBG_NEEDLES))

    def test_signatures_and_sources(self):
        src, params, kw = self.worker["_plan_verify"]
        self.assertEqual(params, ["self", "batch", "draft_input", "prefix_lens"])
        self.assertEqual(kw, ["batch", "draft_input", "prefix_lens"])
        self.assertTrue(_ordered(src, ["if self._verify_planner is None:", "return None, None, None"]))
        src, params, kw = self.worker["_accept_block"]
        self.assertEqual(params, ["self", "candidates", "next_token_logits", "sampling_info", "draft_input",
                                  "prefix_lens", "bs", "target_predict", "cutoff_verify_lens", "verify_shard"])
        self.assertIn("candidates", kw)
        self.assertIn("return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, cap_trim_lens", src)
        self.assertEqual(self.worker["_stage_draft_sampler"][1], ["self", "bs", "sampling_info"])
        self.assertIn("self.selector.sample_path(", self.sampler["__call__"][0])
        self.assertIn("self.selector.sample_path(", self.worker["_propose_selector_block"][0])
        src, params, kw = self.selector["sample_path"]
        self.assertEqual(params, dflash.SAMPLE_PATH_PARAMS)
        self.assertEqual(kw, dflash.SAMPLE_PATH_PARAMS[1:])

    def test_fake_matches_image_shape(self):
        fake = _methods(os.path.join(HERE, "fake_sglang/sglang/srt/speculative/dflash_worker_v2.py"),
                        "DFlashWorkerV2")
        for name in ("_plan_verify", "_accept_block", "_stage_draft_sampler", "_propose_selector_block"):
            self.assertEqual(fake[name][1], self.worker[name][1], name)


def setUpModule():
    # The feature tests reuse the fake selector without importing the fake sglang package into this process.
    path = os.path.join(HERE, "fake_sglang/sglang/srt/speculative/dflash_worker_v2.py")
    with open(path) as fh:
        src = fh.read()
    mod = type(sys)("sglang_fake_selector")
    exec(compile("import torch\nimport torch.nn.functional as F\nfrom torch import nn\nK = 16\n"
                 + ast.get_source_segment(src, next(n for n in ast.parse(src).body
                                                    if isinstance(n, ast.ClassDef) and n.name == "CandidateSelector")),
                 path, "exec"), mod.__dict__)
    sys.modules["sglang_fake_selector"] = mod


if __name__ == "__main__":
    unittest.main()
