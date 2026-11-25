"""dfcap: the DFLASH verify-cap width rules and the remap-identity invariant (CPU).

  python -m unittest discover -s tests/vcap -p 'test_dfcap.py' -v   (from overlay/)"""

import json
import os
import sys
import tempfile
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inkling_vcap import dfcap, logic  # noqa: E402
from inkling_vcap.dflash import DflashFeatures  # noqa: E402


def _state(cap=4):
    st = types.SimpleNamespace(capacity=cap, stride=8, live_lens=torch.zeros(cap, dtype=torch.int64),
                               layout=logic.RowLayout(cap, 8, "cpu"))
    st.src_index = st.layout.rows.clone()
    st.df = DflashFeatures(capacity=cap, gamma=7, top_k=16, device="cpu", keep_rows=False)
    return st


class WidthRuleTest(unittest.TestCase):
    def test_conf_is_leading_run(self):
        m = torch.tensor([[0.9, 0.9, 0.5, 0.99, 0.99, 0.99, 0.99], [0.2] * 7, [1.0] * 7])
        self.assertEqual(dfcap.lens_conf(m, 0.5).tolist(), [3, 1, 8])

    # The calibrated rule is checked against tools/dfreplay.py's export in test_dflash_cap.PolicyParityTest.


class PlanTest(unittest.TestCase):
    def test_capped_then_uncapped_restores_identity(self):
        st = _state()
        cut = dfcap.plan(st, bs=2, mode="fixed", arg=3)
        self.assertEqual(cut.verify_lens.tolist(), [3, 3])
        self.assertEqual(cut.graph_num_tokens, 16)
        self.assertEqual(st.live_lens.tolist(), [3, 3, 0, 0])
        src = st.src_index.view(4, 8)
        self.assertEqual(src[0].tolist(), [0, 1, 2, 0, 0, 0, 0, 0])
        self.assertEqual(src[1].tolist(), [8, 9, 10, 8, 8, 8, 8, 8])
        self.assertTrue((src[2:] == 0).all())  # padding requests read row 0
        self.assertIsNone(dfcap.plan(st, bs=5, mode="fixed", arg=3))  # above capacity: uncapped
        self.assertTrue(torch.equal(st.src_index, st.layout.rows))
        dfcap.plan(st, bs=2, mode="fixed", arg=8)
        self.assertTrue(torch.equal(st.src_index.view(4, 8)[:2].reshape(-1), st.layout.rows[:16]))


if __name__ == "__main__":
    unittest.main()
