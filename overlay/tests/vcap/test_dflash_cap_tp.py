"""The DFLASH verify cap at TP4 / EP4, CPU stand-in with four real processes (gloo):

  python -m unittest discover -s tests/vcap -p 'test_dflash_cap_tp.py' -v   (from overlay/)

Each rank runs the toy DFLASH loop of run_fake_dflash.py with the overlay armed as in the container; fake_tp.py gives
it the image's TP hooks (runtime_context.get_parallel, a TP group), experts split over the ranks with an all-reduce
(Marlin _forward_source at EP4) and vocab-sharded verify logits (the deferred gather: cross-shard argmax or full
gather). Checked:
- rank invariance: live lengths, remap index, accept lengths and emitted tokens identical on all four ranks, and every
  rank issued the same collectives (a rank-dependent decision would hang the run: each run has a timeout);
- the vocab-sharded accept: the cross-shard argmax path and the full-gather path both reproduce the TP1 run token for
  token, with the cap active (dfpolicy df-SLDF2x and fixed:3);
- the guard: a rank-local feature difference (FAKE_PERTURB_RANK) changes nothing with the default gather and with
  check (which counts it); a rank whose features failed, or whose length computation raised, still joins the
  collective proposing full width (not clamped to max=), and rank 0's lengths win (full width when rank 0 failed);
- the one-time settings check at attach: ranks with different cap settings all refuse to start, none hangs;
- EP: each rank's experts touched in the verify are exactly the live rows' experts within its own slice (the dead-row
  remap shrinks every rank's expert set);
- the full-width capture (INKLING_VERIFY_CAP_STEPLOG, the refit input) runs at TP and every rank writes the same
  features, accepts and unions;
- refusals: DP attention at TP > 1; INKLING_VERIFY_CAP_TP_SYNC other than gather / check for an adaptive cap (while
  TP1 and fixed:K ignore the variable); INKLING_VERIFY_CAP_BV=stock with block verification; a
  forward_batch_generation whose shard-only consumers are not layout-gated.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, OVERLAY)

POLICY = os.path.join(OVERLAY, "inkling_vcap", "policies", "df-SLDF2x.json")
DFPOLICY = f"dfpolicy:{POLICY},bsmax=8"
WORLD = 4
TIMEOUT = 240


def _env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INKLING_", "FAKE_"))}
    env["PYTHONPATH"] = os.pathsep.join([OVERLAY, os.path.join(HERE, "fake_sglang")])
    env.update(extra)
    return env


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run_tp1(extra, tmp):
    hist = os.path.join(tmp, f"tp1-{len(os.listdir(tmp))}.pt")
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")],
                         env=_env(dict(extra, FAKE_HISTORY=hist)), capture_output=True, text=True, timeout=TIMEOUT)
    if out.returncode != 0:
        raise AssertionError(out.stderr[-3000:])
    return json.loads(out.stdout.strip().splitlines()[-1]), torch.load(hist, weights_only=False)


def run_tp(extra, tmp, world=WORLD, per_rank_dir=False, rank_env=None, expect_fail=False):
    """[(summary, history, stderr)] per rank; raises when any rank fails or the group hangs. per_rank_dir: values
    of ``extra`` may hold "{rank}" (one capture directory per rank, as one per node in the fleet). rank_env: extra
    variables per rank. expect_fail: return [(returncode, stderr)] instead (every rank must still exit)."""
    init = f"tcp://127.0.0.1:{_free_port()}"
    tag = len(os.listdir(tmp))
    procs, hists = [], []
    for r in range(world):
        hist = os.path.join(tmp, f"tp{tag}-r{r}.pt")
        hists.append(hist)
        mine = {k: v.replace("{rank}", str(r)) for k, v in extra.items()} if per_rank_dir else dict(extra)
        mine.update((rank_env or {}).get(r, {}))
        env = _env(dict(mine, FAKE_TP=str(world), FAKE_RANK=str(r), FAKE_DIST=init, FAKE_HISTORY=hist))
        procs.append(subprocess.Popen([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=env,
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True))
    outs = []
    try:
        for p in procs:
            outs.append(p.communicate(timeout=TIMEOUT))
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        raise AssertionError("TP stand-in timed out: the ranks issued different collectives (deadlock)")
    if expect_fail:
        return [(p.returncode, se) for p, (_, se) in zip(procs, outs)]
    res = []
    for r, (p, (so, se)) in enumerate(zip(procs, outs)):
        if p.returncode != 0:
            raise AssertionError(f"rank {r} failed:\n{se[-3000:]}")
        res.append((json.loads(so.strip().splitlines()[-1]), torch.load(hists[r], weights_only=False), se))
    return res


def _eq(a, b):
    if a is None or b is None:
        return a is None and b is None
    return torch.equal(a, b)


class TpCapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="dfcap-tp-")
        cls.base = {"FAKE_TEMPS": "0,0,0", "FAKE_STEPS": "24"}
        cls.tp1_pol, cls.tp1_pol_hist = run_tp1(dict(cls.base, INKLING_VERIFY_CAP=DFPOLICY), cls.tmp)
        cls.tp4_pol = run_tp(dict(cls.base, INKLING_VERIFY_CAP=DFPOLICY), cls.tmp)

    def assert_ranks_identical(self, res):
        s0, h0, _ = res[0]
        for s, h, _ in res[1:]:
            self.assertEqual(s["emitted"], s0["emitted"])
            self.assertEqual(s["collectives"], s0["collectives"])
            self.assertEqual(len(h), len(h0))
            for a, b in zip(h, h0):
                for key in ("live_lens", "accept_len", "commit_lens", "bonus", "src_index", "cap_trim_lens"):
                    self.assertTrue(_eq(a.get(key), b.get(key)), key)

    def assert_matches_tp1(self, res, tp1, tp1_hist):
        for s, h, _ in res:
            self.assertEqual(s["emitted"], tp1["emitted"])
            for a, b in zip(h, tp1_hist):
                self.assertTrue(_eq(a["live_lens"], b["live_lens"]))
                self.assertTrue(torch.equal(a["accept_len"], b["accept_len"]))

    # ---------------------------------------------------------------- invariance + vocab-sharded accept
    def test_policy_rank_invariant_and_equals_tp1(self):
        res = self.tp4_pol
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)
        capped = sum(int((h["live_lens"] < 8).sum()) for h in res[0][1] if h["live_lens"] is not None)
        self.assertGreater(capped, 0)  # the policy actually cut rows
        trimmed = sum(int((h["uncapped_accept"] > h["accept_len"]).sum()) for h in res[0][1])
        self.assertGreater(trimmed, 0)  # and the cut changed accepts (the cutoff reached the sharded accept)
        for h in res[0][1]:  # dead rows ungraded in the statistics (cross-shard argmax path)
            u, L = h["uncapped_accept"].to(torch.int64), h["live_lens"].to(torch.int64)
            self.assertTrue(torch.equal(h["cap_trim_lens"].to(torch.int64), torch.minimum(u, L) - torch.minimum(u, L - 1)))
        for s, _, err in res:
            self.assertEqual(s["tp_sync"]["calls"], 24)  # one live-length gather per capped step
            self.assertIn("tp=4 tp_sync=gather", err)
            # once: the settings check; per step: deferred argmax (1 gather) + live-length gather (1); 3 MoE visits
            self.assertEqual(s["collectives"], {"all_gather_into_tensor": 49, "all_reduce": 72, "all_gather": 0})
            self.assertEqual(s["tp_plan_errors"], 0)

    def test_full_gather_path_equals_tp1(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_FULL_GATHER="1"), self.tmp)
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)
        self.assertEqual(res[0][0]["collectives"]["all_gather"], 24)  # finish_deferred_gather each step

    def test_fixed_needs_no_collective(self):
        extra = dict(self.base, INKLING_VERIFY_CAP="fixed:3")
        tp1, tp1_hist = run_tp1(extra, self.tmp)
        res = run_tp(dict(extra, INKLING_VERIFY_CAP_TP_SYNC="off"), self.tmp)  # ignored by fixed:K
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, tp1, tp1_hist)
        for s, h, err in res:
            self.assertIsNone(s["tp_sync"])
            self.assertIn("tp_sync=none (fixed)", err)
            self.assertEqual(s["collectives"]["all_gather_into_tensor"], 25)  # settings check + deferred argmax
            for rec in h:
                want = torch.minimum(rec["uncapped_accept"], torch.tensor(2, dtype=rec["uncapped_accept"].dtype))
                self.assertTrue(torch.equal(rec["accept_len"], want))

    def test_bsmax_steps_skip_the_collective(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=f"dfpolicy:{POLICY},bsmax=2"), self.tmp)
        self.assert_ranks_identical(res)
        for s, h, _ in res:
            self.assertEqual(s["tp_sync"]["calls"], 0)
            self.assertTrue(all(rec["live_lens"] is None for rec in h))
            self.assertTrue(all(torch.equal(rec["src_index"], torch.arange(24)) for rec in h))

    # ---------------------------------------------------------------- the guard
    def test_rank_local_difference_gather(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_PERTURB_RANK="2"), self.tmp)
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)  # rank 0's (unperturbed) lengths won

    def test_rank_local_difference_check_counts(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_PERTURB_RANK="1",
                          INKLING_VERIFY_CAP_TP_SYNC="check"), self.tmp)
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)
        for s, _, err in res:
            self.assertGreater(s["tp_sync"]["mismatch_steps"], 0)
            self.assertIn("live lengths differ across ranks", err)

    def test_tp_sync_off_refused_for_adaptive(self):
        env = _env(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, INKLING_VERIFY_CAP_TP_SYNC="off", FAKE_TP="4",
                        FAKE_RANK="0", FAKE_NO_DIST="1"))  # refused before any collective
        out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=env,
                             capture_output=True, text=True, timeout=TIMEOUT)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("INKLING_VERIFY_CAP_TP_SYNC='off'", out.stderr)
        tp1, _ = run_tp1(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, INKLING_VERIFY_CAP_TP_SYNC="off"), self.tmp)
        self.assertEqual(tp1["emitted"], self.tp1_pol["emitted"])  # TP1 ignores the variable

    def test_failed_features_on_one_rank(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_FAIL_FEATURES_RANK="3"), self.tmp)
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)

    def test_failed_features_on_rank0_full_width(self):
        # max=3 caps every proposal at 3 rows, but the fallback proposes full width (8), unclamped
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY + ",max=3", FAKE_FAIL_FEATURES_RANK="0"), self.tmp)
        self.assert_ranks_identical(res)
        for rec in res[0][1]:
            self.assertTrue(torch.equal(rec["live_lens"], torch.full((3,), 8)))
            self.assertTrue(torch.equal(rec["accept_len"], rec["uncapped_accept"]))

    def test_planning_exception_on_one_rank(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_PLAN_RAISE_RANK="2"), self.tmp)
        self.assert_ranks_identical(res)
        self.assert_matches_tp1(res, self.tp1_pol, self.tp1_pol_hist)  # rank 0's proposal won
        self.assertEqual([s["tp_plan_errors"] for s, _, _ in res], [0, 0, 24, 0])
        self.assertIn("rank-local planning failure", res[2][2])

    def test_planning_exception_on_rank0_full_width(self):
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY + ",max=3", FAKE_PLAN_RAISE_RANK="0"), self.tmp)
        self.assert_ranks_identical(res)
        for rec in res[0][1]:
            self.assertTrue(torch.equal(rec["live_lens"], torch.full((3,), 8)))

    def test_settings_mismatch_refused_on_every_rank(self):
        for skew in ({"INKLING_VERIFY_CAP": f"dfpolicy:{POLICY},bsmax=4"},
                     {"INKLING_VERIFY_CAP": DFPOLICY + ",rowscale=0.5"},
                     {"INKLING_VERIFY_CAP_TP_SYNC": "check"}):
            out = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY), self.tmp, rank_env={2: skew},
                         expect_fail=True)
            for rc, err in out:
                self.assertNotEqual(rc, 0)
                self.assertIn("verify-cap settings differ across TP ranks (ranks [2] differ from rank 0)", err)

    def test_settings_mismatch_policy_contents(self):
        import shutil

        d = tempfile.mkdtemp(prefix="dfpol-")
        other = os.path.join(d, "df.json")
        shutil.copy(POLICY, other)
        with open(other) as fh:
            obj = json.load(fh)
        obj["theta"][0][1] *= 1.01
        with open(other, "w") as fh:
            json.dump(obj, fh)
        out = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY), self.tmp,
                     rank_env={1: {"INKLING_VERIFY_CAP": f"dfpolicy:{other},bsmax=8"}}, expect_fail=True)
        self.assertTrue(all(rc != 0 and "ranks [1] differ" in err for rc, err in out))

    def test_block_verification(self):
        env = _env(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, INKLING_VERIFY_CAP_BV="stock", FAKE_BLOCK_VERIFY="1",
                        FAKE_TP="4", FAKE_RANK="0", FAKE_NO_DIST="1"))
        out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=env,
                             capture_output=True, text=True, timeout=TIMEOUT)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("INKLING_VERIFY_CAP_BV=stock is refused", out.stderr)
        res = run_tp(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_BLOCK_VERIFY="1"), self.tmp)  # mask: allowed
        self.assert_ranks_identical(res)
        self.assertIn("block_verification=on bv_mask=on", res[0][2])

    # ---------------------------------------------------------------- EP: per-rank expert sets
    def test_ep_remap_reads_only_live_local_experts(self):
        sys.path.insert(0, os.path.join(HERE, "fake_sglang"))
        from sglang.srt.speculative.dflash_worker_v2 import ToyTarget  # unpatched, EP off in this process

        model = ToyTarget()
        n_exp = 16 // WORLD
        shrunk = 0
        for r, (_, hist, _) in enumerate(self.tp4_pol):
            for h in hist:
                bs = h["bs"]
                lens = h["live_lens"] if h["live_lens"] is not None else torch.full((bs,), 8)
                live = (torch.arange(8)[None, :] < lens[:, None]).reshape(-1)
                x = model.emb[h["draft_tokens"].reshape(-1)]
                for v, li in enumerate(model.schedule):
                    _, ids = model.layers[li].gate(x)
                    lo = r * n_exp
                    want = sorted(e for e in set(ids[live].reshape(-1).tolist()) if lo <= e < lo + n_exp)
                    every = sorted(e for e in set(ids.reshape(-1).tolist()) if lo <= e < lo + n_exp)
                    self.assertEqual(h["touched"][v], want)
                    shrunk += len(every) - len(want)
                    x = x + model.layers[li](x, None)
        self.assertGreater(shrunk, 0)  # dead rows' experts really dropped out of some rank's set

    # ---------------------------------------------------------------- the full-width capture (refit input)
    def test_steplog_at_tp(self):
        import glob

        import numpy as np

        d = tempfile.mkdtemp(prefix="dfsl-tp-")
        res = run_tp({"FAKE_TEMPS": "0,0.8,0", "FAKE_STEPS": "16", "INKLING_VERIFY_CAP": "steplog",
                      "INKLING_VERIFY_CAP_STEPLOG": os.path.join(d, "r{rank}")}, self.tmp, per_rank_dir=True)
        caps = []
        for r, (s, _, _) in enumerate(res):
            self.assertEqual((s["steps"], s["written"], s["failed"], s["missing_routes"]), (16, 16, None, 0))
            self.assertIsNone(s["tp_sync"])  # full width: no live-length collective
            files = sorted(glob.glob(os.path.join(d, f"r{r}", "dflash1-*.npz")))
            z = [np.load(f) for f in files]
            caps.append({k: np.concatenate([x[k] for x in z]) for k in ("m", "gap", "accepted", "union")})
        for c in caps[1:]:
            for k in c:
                self.assertTrue(np.array_equal(c[k], caps[0][k]), k)

    # ---------------------------------------------------------------- refusals
    def test_attn_dp_refused(self):
        env = _env(dict(self.base, INKLING_VERIFY_CAP=DFPOLICY, FAKE_TP="4", FAKE_RANK="0", FAKE_NO_DIST="1",
                        FAKE_ATTN_DP="1"))
        out = subprocess.run([sys.executable, os.path.join(HERE, "run_fake_dflash.py")], env=env,
                             capture_output=True, text=True, timeout=TIMEOUT)
        self.assertNotEqual(out.returncode, 0)
        self.assertIn("needs attention DP off", out.stderr)

    def test_unguarded_shard_consumer_refused(self):
        from inkling_vcap import dflash

        def good(self, layout, target_lp, logits_output, bs):
            if target_lp is not None and target_lp.is_deferred_shard(logits_output.next_token_logits):
                if layout is None and self._tail_covered(bs):
                    pass
                elif (layout is None
                      and self._megakernel_accept is not None):
                    pass
                else:
                    target_predict = target_lp.deferred_argmax(logits_output.next_token_logits)
            elif layout is None and self._can_megakernel_sample(bs):
                pass
            else:
                logits_output.next_token_logits = target_lp.finish_deferred_gather(logits_output.next_token_logits)
            (a, b) = self._accept_block(target_predict=target_predict,
                                        cutoff_verify_lens=None if layout is None else layout.verify_lens,)

        def bad(self, layout, target_lp, logits_output, bs):
            if target_lp is not None and target_lp.is_deferred_shard(logits_output.next_token_logits):
                if layout is None and self._tail_covered(bs):
                    pass
                elif self._megakernel_accept is not None:  # no longer gated on layout is None
                    pass
                else:
                    target_predict = target_lp.deferred_argmax(logits_output.next_token_logits)
            elif layout is None and self._can_megakernel_sample(bs):
                pass
            else:
                logits_output.next_token_logits = target_lp.finish_deferred_gather(logits_output.next_token_logits)
            (a, b) = self._accept_block(target_predict=target_predict,
                                        cutoff_verify_lens=None if layout is None else layout.verify_lens,)

        dflash._require_source_flat(good, dflash.FBG_SHARD_NEEDLES, "good")
        with self.assertRaises(dflash.DflashPatchError):
            dflash._require_source_flat(bad, dflash.FBG_SHARD_NEEDLES, "bad")


class TpSyncUnitTest(unittest.TestCase):
    """tpsync.LiveLenSync on a single-process group whose all-gather returns preset per-rank rows."""

    class Group:
        def __init__(self, rows):
            self.rows, self.world_size, self.rank_in_group = rows, len(rows), 2

        def all_gather_into_tensor(self, out, inp):
            out.copy_(torch.cat(self.rows))

    def test_takes_rank0(self):
        from inkling_vcap import tpsync

        rows = [torch.tensor([3, 8, 1]), torch.tensor([3, 7, 1]), torch.tensor([2, 8, 1]), torch.tensor([3, 8, 1])]
        for mode in ("gather", "check"):
            logs = []
            sync = tpsync.LiveLenSync(self.Group(rows), mode, 8, "cpu", log=logs.append)
            buf = torch.zeros(8, dtype=torch.int64)
            buf[:3] = torch.tensor([2, 8, 1])  # this rank's own (different) view
            sync(buf[:3])
            self.assertEqual(buf[:3].tolist(), [3, 8, 1])
            self.assertEqual(buf[3:].tolist(), [0] * 5)
            if mode == "check":
                self.assertEqual((sync.mismatch_steps, sync.mismatch_rows), (1, 2))
                self.assertEqual(len(logs), 1)

    def test_rejects_bad_input(self):
        from inkling_vcap import tpsync

        rows = [torch.tensor([3])] * 4
        sync = tpsync.LiveLenSync(self.Group(rows), "gather", 2, "cpu")
        with self.assertRaises(RuntimeError):
            sync(torch.tensor([3], dtype=torch.int32))
        with self.assertRaises(RuntimeError):
            sync(torch.zeros(3, dtype=torch.int64))  # above capacity
        for mode in ("bcast", "off"):
            with self.assertRaises(ValueError):
                tpsync.LiveLenSync(self.Group(rows), mode, 2, "cpu")

    def test_settings_check(self):
        from inkling_vcap import tpsync

        a = dict(kind="dfpolicy", bsmax=8, rowscale=1.0, policy_sha256="ab")
        da = tpsync.settings_digest(a)
        self.assertTrue(torch.equal(da, tpsync.settings_digest(dict(reversed(list(a.items()))))))
        self.assertFalse(torch.equal(da, tpsync.settings_digest(dict(a, policy_sha256="ac"))))
        tpsync.check_settings(self.Group([da] * 4), a, "cpu")
        with self.assertRaises(RuntimeError):
            tpsync.check_settings(self.Group([da, da, tpsync.settings_digest(dict(a, bsmax=4)), da]), a, "cpu")


class ConfigTest(unittest.TestCase):
    def test_rowscale_and_tp_sync_parse(self):
        from inkling_vcap.config import VcapConfigError, parse_config

        cfg = parse_config({"INKLING_VERIFY_CAP": DFPOLICY + ",rowscale=0.5"})
        self.assertEqual((cfg.rowscale, cfg.tp_sync, cfg.bsmax), (0.5, "gather", 8))
        self.assertIn("rowscale=0.5", cfg.describe())
        cfg = parse_config({"INKLING_VERIFY_CAP": DFPOLICY})
        self.assertEqual(cfg.rowscale, 1.0)
        self.assertNotIn("rowscale", cfg.describe())
        self.assertEqual(parse_config({"INKLING_VERIFY_CAP": DFPOLICY,
                                       "INKLING_VERIFY_CAP_TP_SYNC": "Check"}).tp_sync, "check")
        # validated only where it applies (adaptive cap at TP > 1, at attach): parsing never fails on it
        self.assertEqual(parse_config({"INKLING_VERIFY_CAP": DFPOLICY,
                                       "INKLING_VERIFY_CAP_TP_SYNC": "off"}).tp_sync, "off")
        for bad in ({"INKLING_VERIFY_CAP": "fixed:3,rowscale=0.5"},
                    {"INKLING_VERIFY_CAP": DFPOLICY + ",rowscale=0"},
                    {"INKLING_VERIFY_CAP": DFPOLICY + ",rowscale=nan"},
                    {"INKLING_VERIFY_CAP": DFPOLICY + ",rowscale=x"}):
            with self.assertRaises(VcapConfigError):
                parse_config(bad)

    def test_rowscale_scales_thresholds_only(self):
        from inkling_vcap import dfcap

        pol = dfcap.load_df_policy(POLICY)
        base = dfcap.DfPolicyDevice(pol, "cpu")
        one = dfcap.DfPolicyDevice(pol, "cpu", scale=1.0)
        half = dfcap.DfPolicyDevice(pol, "cpu", scale=0.5)
        self.assertTrue(torch.equal(base.thr, one.thr))
        self.assertTrue(torch.equal(half.thr, base.thr * 0.5))
        g = torch.Generator().manual_seed(3)
        m = torch.rand(4, 7, generator=g)
        sampled = torch.tensor([False, True, False, True])
        b = pol.band_index(4)
        l1 = dfcap.lens_policy(m, torch.zeros(4, 7), sampled, b, base)
        l2 = dfcap.lens_policy(m, torch.zeros(4, 7), sampled, b, half)
        self.assertTrue(bool((l2 >= l1).all()))  # a lower price never verifies fewer rows


if __name__ == "__main__":
    unittest.main()
