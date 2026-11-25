"""inkling_phoist at TP4: the start-up decision is the same on every rank (CPU stand-in, four gloo processes).

    python -m unittest discover -s tests/phoist -p 'test_phoist_tp.py' -v   (from overlay/)

Each rank runs run_tp_phoist.py: install() from the switch, the patched prepare_loop (real build_pack, real
startup_check over stand-in kernels, the TP agreement), then the patched loop_mix at several row counts. Checked:
- all ranks pass: every rank hoists the armed row counts and runs pfuse at the others, identically;
- a fault on ONE rank disables the hoist on EVERY rank, and nothing hangs: OOM inside the self-test, a tolerance
  failure, an exception or a refused layout while packing, a different armed regime set, an exception outside the
  self-test's own handler (generator construction, the final CUDA cleanup), a model without loop queries, other
  effective pfuse settings (LOOP_NORM=stock, MIN_ROWS), the switch off on that rank;
- control: without the agreement (the single-GPU behaviour) the same fault leaves the ranks on different paths;
- configuration refusals are collective: an invalid switch on one rank, `exact` at TP > 1 and an unreadable TP size
  under a multi-rank process group stop every rank (no rank continues);
- a weight reload with the TP hoist armed stops every rank and keeps the pack's buffers; with the hoist disarmed
  (or at TP1) a reload goes on;
- TP1: no collective is issued and the single-GPU behaviour is unchanged.
"""
import json
import os
import socket
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(HERE, "run_tp_phoist.py")
WORLD = 4
TIMEOUT = 180
HOIST = ["hoisted", "hoisted", "hoisted", "hoisted", "pfuse", "pfuse", "hoisted", "hoisted", "pfuse", "pfuse"]
PFUSE = ["pfuse"] * len(HOIST)


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(extra, world=WORLD):
    """[(returncode, summary or None, stderr)] per rank; AssertionError if the ranks hang."""
    init = f"tcp://127.0.0.1:{_free_port()}"
    base = {k: v for k, v in os.environ.items() if not k.startswith(("INKLING_", "FAKE_", "PH_", "PYTHONPATH"))}
    procs = []
    for r in range(world):
        env = dict(base, FAKE_TP=str(world), FAKE_RANK=str(r), FAKE_DIST=init, **extra)  # no sitecustomize: the runner installs
        procs.append(subprocess.Popen([sys.executable, RUNNER], env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.PIPE, text=True))
    outs = []
    try:
        for p in procs:
            outs.append(p.communicate(timeout=TIMEOUT))
    except subprocess.TimeoutExpired:
        for p in procs:
            p.kill()
        raise AssertionError("ranks hung: different collectives on different ranks")
    res = []
    for p, (so, se) in zip(procs, outs):
        lines = [json.loads(ln) for ln in so.strip().splitlines() if ln.startswith("{")]
        summ = next((d for d in lines if "decisions" in d), None)
        if summ is not None:
            summ["reload"] = next((d for d in lines if "reload" in d), None)
        res.append((p.returncode, summ, se))
    return res


class PhoistTp(unittest.TestCase):
    def ok(self, res):
        for r, (code, s, se) in enumerate(res):
            self.assertEqual(code, 0, f"rank {r}:\n{se[-2000:]}")
        return [s for _, s, _ in res]

    def assert_same_everywhere(self, sums, want):
        for s in sums:
            self.assertEqual(s["decisions"], want, s)
        self.assertEqual(len({json.dumps(s["decisions"]) for s in sums}), 1)
        self.assertEqual(len({s["startup_ok"] for s in sums}), 1)

    def test_all_ranks_pass_all_hoist(self):
        sums = self.ok(run({}))
        self.assert_same_everywhere(sums, HOIST)
        self.assertTrue(all(s["startup_ok"] and s["failed"] == "None" for s in sums))

    def test_one_rank_fault_disables_every_rank(self):
        for fault, rank, local in (("oom", 2, "OutOfMemoryError"), ("mismatch", 1, "(8, 0, 'attn')"),
                                   ("pack", 3, "'pack'"), ("layout", 0, None), ("tiles", 0, None),
                                   ("generator", 1, "'setup'"), ("cleanup", 2, "'setup'"), ("noloop", 3, None),
                                   ("loopnorm", 0, None), ("minrows", 3, None), ("off", 1, None)):
            with self.subTest(fault=fault):
                sums = self.ok(run({"PH_FAULT": fault, "PH_FAULT_RANK": str(rank)}))
                self.assert_same_everywhere(sums, PFUSE)
                for s in sums:
                    self.assertFalse(s["startup_ok"])
                    if s["rank"] == rank and local is not None:
                        self.assertIn(local, s["failed"])        # the rank's own reason is kept
                    else:
                        self.assertIn("'tp'", s["failed"])       # the peers learned it from the agreement
                if fault in ("tiles", "loopnorm", "minrows"):
                    self.assertIn("effective pfuse settings", sums[(rank + 1) % WORLD]["failed"])

    def test_control_without_agreement_ranks_diverge(self):
        sums = self.ok(run({"PH_FAULT": "oom", "PH_FAULT_RANK": "2", "PH_NO_AGREE": "1"}))
        self.assertEqual(sums[2]["decisions"], PFUSE)
        self.assertEqual(sums[0]["decisions"], HOIST)       # the hazard the agreement closes

    def test_exact_refused_on_every_rank(self):
        res = run({"PH_VARIANT": "exact"})
        for code, s, se in res:
            self.assertNotEqual(code, 0)
            self.assertIsNone(s)
            self.assertIn("refused on every rank", se)
            self.assertIn("TP1 only", se)

    def test_invalid_switch_on_one_rank_refuses_on_every_rank(self):
        res = run({"PH_FAULT": "invalid", "PH_FAULT_RANK": "2"})
        for r, (code, s, se) in enumerate(res):
            self.assertNotEqual(code, 0)
            self.assertIsNone(s)
            self.assertIn("refused on every rank", se)
            self.assertIn("INKLING_PFUSE_HOIST='x'" if r == 2 else "a peer rank refused", se)

    def test_reload_with_tp_hoist_armed_stops_every_rank(self):
        res = run({"PH_RELOAD": "1"})
        for code, s, se in res:
            self.assertEqual(s["decisions"], HOIST)
            self.assertNotEqual(code, 0)
            self.assertIn("weight reload with the TP hoist armed is refused", se)
            self.assertTrue(s["reload"]["pack_kept"])
        sums = self.ok(run({"PH_RELOAD": "1", "PH_FAULT": "oom", "PH_FAULT_RANK": "2"}))  # disarmed: reload goes on
        self.assert_same_everywhere(sums, PFUSE)

    def test_unreadable_tp_size_under_a_process_group_refuses(self):
        res = run({"PH_NO_CONTEXT": "1"})
        for code, s, se in res:
            self.assertNotEqual(code, 0)
            self.assertIn("TP size is unreadable", se)

    def test_tp1_unchanged_no_collective(self):
        (code, s, se), = run({}, world=1)
        self.assertEqual(code, 0, se)
        self.assertEqual(s["decisions"], HOIST)
        (code, s, se), = run({"PH_FAULT": "oom", "PH_FAULT_RANK": "0"}, world=1)
        self.assertEqual(code, 0, se)
        self.assertEqual(s["decisions"], PFUSE)
        self.assertNotIn("TP", s["failed"])
        (code, s, se), = run({"PH_RELOAD": "1"}, world=1)           # TP1: reload re-runs the self-test as on 1x
        self.assertEqual(code, 0, se)
        self.assertEqual(s["reload"]["pack_kept"], True)
        (code, s, se), = run({"PH_FAULT": "invalid", "PH_FAULT_RANK": "0"}, world=1)
        self.assertNotEqual(code, 0)
        self.assertIn("INKLING_PFUSE_HOIST='x'", se)


if __name__ == "__main__":
    unittest.main()
