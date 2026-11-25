"""logic.mask_dead_q_rows_sparse == logic.mask_dead_draft_rows after the DFLASH dense scatter (CPU).

  python -m unittest discover -s tests/vcap -p 'test_bv_sparse.py' -v   (from overlay/)"""

import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inkling_vcap import logic  # noqa: E402


def dense(q, cand, vocab):
    out = torch.zeros(q.shape[0], q.shape[1], vocab, dtype=torch.float32)
    out.scatter_(-1, cand, q.float())
    return out


class SparseMaskTest(unittest.TestCase):
    def test_equals_dense_mask(self):
        g = torch.Generator().manual_seed(0)
        bs, G, K, V = 6, 7, 16, 97
        for trial in range(50):
            cand = torch.stack([torch.stack([torch.randperm(V, generator=g)[:K] for _ in range(G)])
                                for _ in range(bs)])
            q = torch.softmax(torch.randn(bs, G, K, generator=g) * 3, dim=-1)
            q[0, 3, 4] = 0.0  # zero lanes are allowed
            lane = torch.randint(0, K, (bs, G), generator=g)
            tokens = cand.gather(-1, lane[..., None])[..., 0]
            lens = torch.randint(1, 9, (bs,), generator=g)
            lens[0], lens[1] = 1, 8  # boundary: only the anchor live / every row live
            got = dense(logic.mask_dead_q_rows_sparse(q, cand, tokens, lens), cand, V)
            ref = logic.mask_dead_draft_rows(dense(q, cand, V), tokens.clone(), lens)
            self.assertTrue(torch.equal(got, ref), trial)

    def test_input_untouched_and_full_width_identity(self):
        g = torch.Generator().manual_seed(1)
        cand = torch.stack([torch.randperm(50, generator=g)[:16] for _ in range(7)])[None]
        q = torch.softmax(torch.randn(1, 7, 16, generator=g), dim=-1)
        tokens = cand[..., 0]
        before = q.clone()
        out = logic.mask_dead_q_rows_sparse(q, cand, tokens, torch.tensor([8]))
        self.assertTrue(torch.equal(q, before))
        self.assertTrue(torch.equal(out, q))
        self.assertNotEqual(out.data_ptr(), q.data_ptr())


if __name__ == "__main__":
    unittest.main()
