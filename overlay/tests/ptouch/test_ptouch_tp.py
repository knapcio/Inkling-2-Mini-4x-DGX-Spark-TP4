"""inkling_ptouch at TP4: every rank's prefix cache evolves identically (CPU stand-in, four gloo processes).

    python -m unittest discover -s tests/ptouch -p 'test_ptouch_tp.py' -v   (from overlay/)

Each rank runs run_tp_ptouch.py: the overlay installed on a toy cache with a logical LRU clock, locks and eviction,
the same scheduler script on every rank (agents' turn 1 under filler pressure, then turn 2). Checked:
- armed on every rank: identical trees (digest over every node's key, stamp and kind, and the clock), identical
  turn-2 hits (8 / 8 against stock's 0 / 8 in this script) and identical consensus records (rid, touched, depth);
- start-up agreement: one rank with another INKLING_PROMPT_TOUCH_MIN, a switched-off rank, a threshold that does
  not fit int64, or a rank inert at install (changed SWA premise) makes every rank inert (every rank still votes),
  and every rank then equals the stock run;
- control: without the agreement the same rank-local inert state gives different trees (the hazard);
- runtime failure of the overlay on one rank (the touch walk or the pre-read raises) stops that rank with
  PtouchFatal after the original insert, instead of continuing inert alone;
- an unreadable TP size under a multi-rank process group refuses at cache construction on every rank;
- TP1: no collective; an overlay failure goes inert as in the single-GPU recipe.
"""
import json
import os
import socket
import subprocess
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
RUNNER = os.path.join(HERE, "run_tp_ptouch.py")
WORLD = 4
TIMEOUT = 180


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def run(extra, world=WORLD):
    """[(returncode, summary or None, stderr)] per rank; AssertionError if the ranks hang."""
    init = f"tcp://127.0.0.1:{_free_port()}"
    base = {k: v for k, v in os.environ.items() if not k.startswith(("INKLING_", "FAKE_", "PT_", "PYTHONPATH"))}
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
        lines = [ln for ln in so.strip().splitlines() if ln.startswith("{")]
        res.append((p.returncode, json.loads(lines[-1]) if lines else None, se))
    return res


class PtouchTp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        (code, cls.stock, se), = run({"PT_OFF": "1"}, world=1)
        assert code == 0, se

    def ok(self, res):
        for r, (code, s, se) in enumerate(res):
            self.assertEqual(code, 0, f"rank {r}:\n{se[-2000:]}")
        return [s for _, s, _ in res]

    def assert_identical(self, sums):
        for key in ("digest", "hits", "records", "inert"):
            self.assertEqual(len({json.dumps(s[key]) for s in sums}), 1, f"ranks differ in {key}")

    def test_armed_on_every_rank_identical_trees(self):
        sums = self.ok(run({}))
        self.assert_identical(sums)
        self.assertEqual(sums[0]["hits"], [True] * 8)
        self.assertEqual(self.stock["hits"], [False] * 8)         # the toy reproduces the miss the touch removes
        self.assertNotEqual(sums[0]["digest"], self.stock["digest"])
        self.assertEqual(sums[0]["inert"], "")
        self.assertEqual(sums[0]["tp"], WORLD)
        self.assertGreater(sums[0]["calls"], 0)
        self.assertTrue(any("touch=True" in r for r in sums[0]["records"]))
        self.assertTrue(any("touch=False" in r for r in sums[0]["records"]))   # fillers below the threshold

    def test_rank_local_start_up_difference_makes_every_rank_inert(self):
        for fault, rank, why in (("min", 1, "INKLING_PROMPT_TOUCH_MIN"), ("premise", 3, "a rank is inert"),
                                 ("off", 2, "a rank is inert"), ("badmin", 0, "a rank is inert")):
            with self.subTest(fault=fault):
                sums = self.ok(run({"PT_FAULT": fault, "PT_FAULT_RANK": str(rank)}))
                self.assertEqual(len({s["digest"] for s in sums}), 1)
                self.assertEqual(len({json.dumps(s["hits"]) for s in sums}), 1)
                for s in sums:
                    self.assertTrue(s["inert"], s)
                    self.assertEqual(s["calls"], 0)
                    self.assertEqual(s["digest"], self.stock["digest"])    # all inert == stock, on every rank
                    if s["rank"] != rank:
                        self.assertIn(why, s["inert"])

    def test_control_without_agreement_trees_diverge(self):
        sums = self.ok(run({"PT_FAULT": "premise", "PT_FAULT_RANK": "3", "PT_NO_AGREE": "1"}))
        self.assertEqual(sums[3]["digest"], self.stock["digest"])
        self.assertNotEqual(sums[0]["digest"], sums[3]["digest"])        # the hazard the agreement closes
        self.assertNotEqual(sums[0]["hits"], sums[3]["hits"])

    def test_runtime_failure_on_one_rank_stops_it(self):
        for fault in ("touch", "preread"):
            with self.subTest(fault=fault):
                res = run({"PT_FAULT": fault, "PT_FAULT_RANK": "2"})
                code, s, se = res[2]
                self.assertNotEqual(code, 0)
                self.assertIsNone(s)                                      # never went on inert
                self.assertIn("PtouchFatal", se)
                self.assertIn("stopping instead of going inert", se)
                peers = [x for i, (c, x, e) in enumerate(res) if i != 2]
                self.assertTrue(all(c == 0 for i, (c, x, e) in enumerate(res) if i != 2))
                self.assertEqual(len({p["digest"] for p in peers}), 1)

    def test_unreadable_tp_size_refuses(self):
        for code, s, se in run({"PT_NO_CONTEXT": "1"}):
            self.assertNotEqual(code, 0)
            self.assertIn("TP size is unreadable", se)

    def test_tp1(self):
        (code, s, se), = run({}, world=1)
        self.assertEqual(code, 0, se)
        self.assertEqual((s["tp"], s["hits"], s["inert"]), (1, [True] * 8, ""))
        (code, s, se), = run({"PT_FAULT": "touch", "PT_FAULT_RANK": "0"}, world=1)
        self.assertEqual(code, 0, se)                                     # single-GPU behaviour: inert, serving on
        self.assertEqual(s["inert"], "touch failed")


if __name__ == "__main__":
    unittest.main()
