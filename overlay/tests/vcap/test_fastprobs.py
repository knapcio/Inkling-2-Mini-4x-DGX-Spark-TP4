"""fastprobs combine math (CPU): a torch emulation of the image's split stats kernel
(_softmax_temp_stats_kernel: per chunk online max / sum-exp of x/T) feeds
combine_max_prob, which must equal the torch reference draft_token_probs(use_max=True).
The kernels themselves run only on the GPU (tools/gpu_check_fastprobs.py)."""
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))

from inkling_vcap import fastprobs, logic  # noqa: E402


def emulate_stats(flat, temps, rows_per_request, chunk):
    rows, vocab = flat.shape
    splits = -(-vocab // chunk)
    out = torch.empty(rows, splits, 2)
    for r in range(rows):
        t = temps[r // rows_per_request]
        for s in range(splits):
            x = flat[r, s * chunk:(s + 1) * chunk].float() / t
            m = x.max()
            if m == float("-inf"):
                out[r, s] = torch.tensor([float("-inf"), 0.0])
            else:
                out[r, s] = torch.stack([m, torch.exp(x - m).sum()])
    return out


class CombineTest(unittest.TestCase):
    def test_matches_reference(self):
        g = torch.Generator().manual_seed(0)
        bs, gamma, vocab = 3, 7, 5000
        logits = (torch.randn(bs, gamma, vocab, generator=g) * 4).to(torch.bfloat16)
        logits[:, :, 3000:] = float("-inf")
        logits[1, 2, :] = float("-inf")
        logits[1, 2, 17] = 3.0  # one finite entry: m = 1
        temps = torch.tensor([0.8, 0.3, 1.3])
        greedy = torch.tensor([False, True, False])
        ref = logic.draft_token_probs(logits, torch.zeros(bs, gamma, dtype=torch.long),
                                      temperatures=temps, greedy_mask=greedy, use_max=True)
        rt = fastprobs.request_temps(bs, temps, greedy, "cpu")
        self.assertEqual(rt.tolist(), [torch.tensor(0.8).item(), 1.0, torch.tensor(1.3).item()])
        for chunk in (2048, 1024, 700):
            stats = emulate_stats(logits.reshape(bs * gamma, vocab), rt, gamma, chunk)
            got = fastprobs.combine_max_prob(stats).view(bs, gamma)
            self.assertLess(float((got - ref).abs().max()), 1e-6, chunk)
        self.assertEqual(float(got[1, 2]), 1.0)

    def test_all_neg_inf_row_is_zero(self):
        stats = torch.tensor([[[float("-inf"), 0.0], [float("-inf"), 0.0]]])
        self.assertEqual(fastprobs.combine_max_prob(stats).tolist(), [0.0])

    def test_unavailable_on_cpu(self):
        self.assertFalse(fastprobs.available("cpu"))


if __name__ == "__main__":
    unittest.main()
