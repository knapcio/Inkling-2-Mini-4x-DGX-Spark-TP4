"""CPU checks of inkling_attn at the attention-TP4 per-rank shapes (24 q / 8 kv heads over 4 ranks: 6 q / 2 kv heads,
GQA 3; qkvr rank row [q 768 | k 256 | v 256 | r 96 | gate 768 | sinks 6 | pad 2] = 2152 columns,
inkling2_attn.py:139-149).

1. Kernel control flow and lane-level index math (emulate_kernel.py: work items, KV head = head // group, key tiles,
   in-kernel sheared bias, masks, online softmax) against a dense reference at 6 / 2 heads (TP4) and 24 / 8 (TP1).
2. The patch's eligibility at the TP4 shapes: _kernel_call accepts the FA4 call (so the kernel does run at TP4) and
   _gate_ok accepts the gate / sinks column views of the [T, 2152] qkvr output (offsets 1376 and 2144, row stride
   2152); the gated epilogue's addressing (gate + t * g_st + head * 128 + c, sink[t * s_st + head]) reads exactly
   gate[t, head * 128 + c] and sinks[t, head] of those views.

Run: python3 overlay/inkling_attn/tests/test_tp_shapes_cpu.py   (or python -m unittest; needs numpy and torch)
"""
import os
import sys
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # overlay/

TP4 = dict(q=768, k=256, v=256, r=96, gate=768, sinks=6, pad=2)


def qkvr_views(torch, t, sizes):
    names = list(sizes)
    buf = torch.arange(t * sum(sizes.values()), dtype=torch.float32).reshape(t, -1).to(torch.bfloat16)
    views = dict(zip(names, buf.split(list(sizes.values()), dim=-1)))
    return buf, views


class EmulatedKernelTests(unittest.TestCase):
    def test_tp4_and_tp1_heads(self):
        import emulate_kernel as ek

        for heads, group in ((6, 3), (24, 3)):
            ek.run([(36, 36)], True, 128, 127, heads=heads, group=group)
            ek.run([(150, 700), (0, 0), (70, 70), (3, 300)], True, 128, 127, heads=heads, group=group, seed=1)
            ek.run([(150, 700), (0, 0), (70, 70), (3, 300)], False, 128, -1, heads=heads, group=group, seed=2)


class PatchShapeTests(unittest.TestCase):
    def setUp(self):
        import torch

        from inkling_attn import patch

        self.torch, self.patch = torch, patch

    def _fa4_kwargs(self, t, heads, kv_heads, window_left):
        torch = self.torch
        pages, page = 4, 128
        return dict(
            q=torch.zeros((t, heads, 128), dtype=torch.bfloat16),
            k=torch.zeros((pages, page, kv_heads, 128), dtype=torch.bfloat16),
            v=torch.zeros((pages, page, kv_heads, 128), dtype=torch.bfloat16),
            page_table=torch.zeros((1, pages), dtype=torch.int32),
            cu_seqlens_q=torch.tensor([0, t], dtype=torch.int32),
            seqused_k=torch.tensor([t], dtype=torch.int32),
            out=torch.empty((t, heads, 128), dtype=torch.float32),
            softmax_scale=128 ** -0.5,
            causal=True,
            window_size=(window_left, 0) if window_left is not None else (None, None),
            num_splits=1,
        )

    def test_kernel_call_accepts_tp4_shapes(self):
        torch, patch = self.torch, self.patch
        t = 36
        for heads, kv_heads in ((6, 2), (24, 8)):
            for local in (True, False):
                proj = torch.zeros((16, 1024), dtype=torch.bfloat16)
                r = torch.zeros((t, heads, 16), dtype=torch.bfloat16)
                visit = patch._Visit(None, local, 1023 if local else None, proj)
                visit.r = r
                kw = self._fa4_kwargs(t, heads, kv_heads, 1023 if local else None)
                call = patch._kernel_call(visit, r, kw)
                self.assertIsNotNone(call, (heads, kv_heads, local))
                self.assertEqual(tuple(call["q"].shape), (t, heads, 128))
                self.assertEqual(call["window_left"], 1023 if local else None)
        # a q / KV head count that is not a GQA multiple is left to stock FA4
        visit = patch._Visit(None, False, None, torch.zeros((16, 1024), dtype=torch.bfloat16))
        r = torch.zeros((t, 6, 16), dtype=torch.bfloat16)
        visit.r = r
        self.assertIsNone(patch._kernel_call(visit, r, self._fa4_kwargs(t, 6, 4, None)))

    def test_gate_views_tp4(self):
        torch, patch = self.torch, self.patch
        t, heads = 40, 6
        buf, v = qkvr_views(torch, t, TP4)
        gate, sinks = v["gate"], v["sinks"]
        self.assertEqual((gate.storage_offset(), gate.stride()), (1376, (2152, 1)))
        self.assertEqual((sinks.storage_offset(), sinks.stride()), (2144, (2152, 1)))
        visit = patch._Visit(None, True, 1023, None)
        visit.gate, visit.sinks, visit.tokens = gate, sinks, t
        q = torch.zeros((t, heads, 128), dtype=torch.bfloat16)
        out = torch.empty((t, heads, 128), dtype=torch.float32)
        self.assertTrue(patch._gate_ok(visit, {"q": q, "out": out}))
        # the kernel's pointer arithmetic on the flat storage (attn_kernel.cuh gated epilogue)
        flat = buf.reshape(-1)
        for tt in (0, 17, t - 1):
            for head in range(heads):
                for c in (0, 8, 120):
                    self.assertEqual(flat[gate.storage_offset() + tt * gate.stride(0) + head * 128 + c].item(),
                                     gate[tt, head * 128 + c].item())
                self.assertEqual(flat[sinks.storage_offset() + tt * sinks.stride(0) + head].item(),
                                 sinks[tt, head].item())
        # a gate that does not hold heads * 128 columns is refused (stock gate_sink then runs)
        visit.gate = buf[:, 1376:2150]
        self.assertFalse(patch._gate_ok(visit, {"q": q, "out": out}))


if __name__ == "__main__":
    unittest.main()
