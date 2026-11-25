"""CPU correctness checks. Run with PYTHONPATH=overlay python -m unittest discover -s overlay/tests/kv12s."""

import math
import unittest

import torch

from inkling_kv12s.check_data import kv_values
from inkling_kv12s.storage import (
    CAP, DIM, EMPTY, PACK_BATCH, PACKED, PAGE, STAGE_SHIFT, CapacityError, Layer, Layout, validate_paged,
)


def slots(page, start=0, end=PAGE):
    return list(range(page * PAGE + start, page * PAGE + end))


def word_values(n, word):
    return torch.full((n, 1, DIM), word, dtype=torch.int16).view(torch.bfloat16)


def reference(q, k, v, bias):
    """Global causal attention with distance bias and BF16 probability rounding."""
    qlen, heads, dim = q.shape
    seqlen = k.shape[0]
    k = k.repeat_interleave(heads // k.shape[1], dim=1)
    v = v.repeat_interleave(heads // v.shape[1], dim=1)
    dist = torch.arange(seqlen - qlen, seqlen)[:, None] - torch.arange(seqlen)[None, :]
    rb = bias.index_select(-1, dist.clamp(0, bias.shape[-1] - 1).flatten())
    rb = rb.view(qlen, heads, qlen, seqlen)[torch.arange(qlen), :, torch.arange(qlen)]
    scores = torch.einsum("qhd,khd->qhk", q.float(), k.float()) / math.sqrt(dim)
    scores += torch.where((dist[:, None, :] < bias.shape[-1]), rb, 0).float()
    scores.masked_fill_(dist[:, None, :] < 0, float("-inf"))
    p = torch.softmax(scores, dim=-1).to(torch.bfloat16)
    return torch.einsum("qhk,khd->qhd", p.float(), v.float()), torch.logsumexp(scores, dim=-1)


class StorageTests(unittest.TestCase):
    def assert_bits(self, actual, expected):
        self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))

    def test_all_65536_patterns_in_both_paths(self):
        # 64 escape slots per packed page-head are enough to exercise every
        # bit pattern in the streaming path without tripping overflow.
        words = torch.arange(65536, dtype=torch.int32).to(torch.int16)
        pages = words.numel() // (2 * CAP)
        layer = Layer.standalone(pages + 1, 1, overflow_heads=1)
        raw = torch.full((pages, 2, PAGE, 1, DIM), 0x3F80, dtype=torch.int16)
        raw[:, :, 0, 0, :CAP] = words.view(pages, 2, CAP)
        self._write_pages(layer, raw)
        self.assertFalse(layer.spilled)
        self._check_all(layer, raw)

        layer = Layer.standalone(3, 1, overflow_heads=4)
        raw = words.view(2, 2, PAGE, 1, DIM)
        self._write_pages(layer, raw)
        self.assertEqual(len(layer.spilled), 2)
        self._check_all(layer, raw)

    def test_escape_capacity_boundary_and_exhaustion(self):
        layer = Layer.standalone(5, 1, overflow_heads=1)
        k, v = word_values(PAGE, 0x3F80), word_values(PAGE, 0x3F80)
        k.view(torch.int16)[0, 0, :CAP] = -1  # NaN with sign and payload
        layer.write(slots(1), k, v)
        self.assertFalse(layer.spilled)
        k.view(torch.int16)[32, 0, 127] = -32767  # -subnormal, exponent zero
        layer.write(slots(2), k, v)
        self.assertFalse(layer.spilled)
        k.view(torch.int16)[32, 0, 126] = 0x7F80  # +inf
        layer.write(slots(3), k, v)
        self.assertEqual(layer.spilled, {3: [0]})
        self.assert_bits(layer.read_rows(slots(3))[:, 0], k)
        with self.assertRaises(CapacityError):
            layer.write(slots(4), k, v)
        self.assertEqual(layer.flags[4].item(), EMPTY)
        layer.release([3])
        layer.write(slots(4), k, v)
        self.assertEqual(layer.spilled, {4: [0]})
        self.assert_bits(layer.read_rows(slots(4))[:, 0], k)

    def test_second_batch_overflow_is_atomic_and_retryable(self):
        layer = Layer.standalone(PACK_BATCH + 4, 1, staging_pages=1, overflow_heads=1)
        wide = word_values(PAGE, 0x7FC1)
        plain = word_values(PAGE, 0x3F80)
        layer.write(slots(1), wide, plain)
        layer.write(slots(2, 0, 7), plain[:7], plain[:7])
        end_page = 2 + PACK_BATCH
        locations = list(range(2 * PAGE + 7, (end_page + 1) * PAGE + 3))
        k = word_values(len(locations), 0x4000)
        v = k.clone()
        k[-PAGE - 3:-3] = wide
        before = layer.raw.clone(), layer.stage.clone()
        with self.assertRaisesRegex(CapacityError, "overflow"):
            layer.write(locations, k, v)
        self.assertTrue(torch.equal(layer.raw, before[0]))
        self.assertTrue(torch.equal(layer.stage, before[1]))
        self.assertEqual(layer.open[2].end, 7)
        self.assertEqual(layer.spilled, {1: [0]})
        layer.release([1])
        layer.write(locations, k, v)
        self.assert_bits(layer.read_rows(locations)[:, 0], k)
        self.assertEqual(layer.spilled, {end_page: [0]})
        self.assertEqual(layer.open[end_page + 1].end, 3)

    def test_staging_capacity_failure_leaves_packet_unpublished(self):
        layer = Layer.standalone(5, 1, staging_pages=1)
        layer.write(slots(1, 0, 1), word_values(1, 0x3F80), word_values(1, 0x3F80))
        locs = slots(2) + slots(3, 0, 1)
        words = word_values(len(locs), 0x4000)
        with self.assertRaisesRegex(CapacityError, "staging"):
            layer.write(locs, words, words)
        self.assertEqual(layer.flags[2:].tolist(), [EMPTY, EMPTY, EMPTY])
        layer.release([1])
        layer.write(locs, words, words)
        self.assert_bits(layer.read_rows(locs)[:, 0], words)

    def test_stale_descriptors_do_not_read_reassigned_slots(self):
        layer = Layer.standalone(7, 1, staging_pages=1, overflow_heads=1)
        wide, plain = word_values(PAGE, 0x7FC1), word_values(PAGE, 0x3F80)
        layer.write(slots(1), wide, plain)
        layer.write(slots(2, 0, 1), plain[:1], plain[:1])
        raw = layer.raw.view(layer.pages, layer.page_stride)
        raw[[3, 4]] = raw[[1, 2]]
        layer.transfer([1, 2], [3, 4])
        for page in (1, 2):
            self.assertEqual(layer.flags[page].item(), EMPTY)
            with self.assertRaisesRegex(RuntimeError, "unwritten"):
                layer.read_rows(slots(page, 0, 1))
        layer.release([3, 4])
        layer.write(slots(5), word_values(PAGE, 0x7FC2), plain)
        layer.write(slots(6, 0, 1), plain[:1], plain[:1])
        for page in (1, 2, 3, 4):
            with self.assertRaisesRegex(RuntimeError, "unwritten"):
                layer.read_rows(slots(page, 0, 1))

    def test_reader_metadata_bounds(self):
        layer = Layer.standalone(3, 1)
        q = word_values(8, 0x3F80)
        table = torch.tensor([[1], [2]], dtype=torch.int32)
        seq = torch.tensor([PAGE, 11], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 7], dtype=torch.int32)
        self.assertEqual(validate_paged(q, layer, table, seq, offsets), ([PAGE, 11], [1, 6]))
        for kv, cu, message in (
            ([PAGE + 1, 11], offsets, "page-table"),
            ([-1, 11], offsets, "page-table"),
            (seq, [0, 1, 9], "query bounds"),
            (seq, [0, -1, 7], "query length"),
            (seq, [0, 8, 7], "query length"),
            (seq, [0, 7], "query offset tensor"),
        ):
            with self.subTest(kv=kv, cu=cu), self.assertRaisesRegex(ValueError, message):
                validate_paged(q, layer, table, torch.as_tensor(kv, dtype=torch.int32),
                               torch.as_tensor(cu, dtype=torch.int32))
        with self.assertRaisesRegex(ValueError, "query length"):
            validate_paged(q, layer, table, seq, offsets, qmax=5)

    def test_open_page_verify_rejection_commit_and_reuse(self):
        layer = Layer.standalone(5, 1, staging_pages=2)
        prefix = word_values(120, 0x3F80)
        layer.write(slots(1, 0, 120), prefix, prefix)
        self.assertEqual(layer.flags[1].item(), 121)
        self.assert_bits(layer.read_rows(slots(1, 0, 120))[:, 0], prefix)
        verify = word_values(16, 0x4000)
        layer.write(slots(1, 120) + slots(2, 0, 8), verify, verify, tentative=True)
        self.assertEqual(len(layer.open), 2)
        self.assert_bits(layer.read_rows(slots(1, 120) + slots(2, 0, 8))[:, 0], verify)
        # Accept 3, free the speculative next page, and overwrite the rejected suffix.
        layer.release([2])
        new = word_values(5, 0x4080)
        layer.write(slots(1, 123), new, new)
        self.assertEqual(layer.flags[1].item(), PACKED)
        expected = torch.cat((prefix, verify[:3], new))
        self.assert_bits(layer.read_rows(slots(1))[:, 1], expected)
        with self.assertRaisesRegex(RuntimeError, "immutable"):
            layer.write(slots(1, 0, 1), expected[:1], expected[:1])
        layer.write(slots(2), expected, expected, tentative=True)
        self.assertIn(2, layer.open)
        layer.commit([2, 2])
        self.assertEqual(layer.flags[2].item(), PACKED)
        self.assert_bits(layer.read_rows(slots(2))[:, 1], expected)
        layer.release([1, 2])
        reused = word_values(1, -32768)
        layer.write([0, -1, 128], reused.expand(3, -1, -1), reused.expand(3, -1, -1))
        self.assert_bits(layer.read_rows([128])[:, 0], reused)
        self.assert_bits(layer.read_rows([0])[:, 0], word_values(1, 0))
        with self.assertRaisesRegex(RuntimeError, "unwritten"):
            layer.read_rows([129])

    def test_move_and_rollback_preserve_external_ownership(self):
        layer = Layer.standalone(5, 1)
        spill = word_values(PAGE, 0x7FC1)
        partial = word_values(7, 0x3F80)
        layer.write(slots(1), spill, spill)
        layer.write(slots(2, 0, 7), partial, partial)
        for source, dest in (([1, 2], [3, 4]), ([3, 4], [1, 2])):
            raw = layer.raw.view(layer.pages, layer.page_stride)
            raw[dest] = raw[source]
            layer.transfer(source, dest)
            layer.release(source)
            self.assert_bits(layer.read_rows(slots(dest[0]))[:, 0], spill)
            self.assert_bits(layer.read_rows(slots(dest[1], 0, 7))[:, 1], partial)
        self.assertEqual(len(layer._free_overflow), layer.overflow.shape[0] - 2)

    def test_bounded_snapshot_rows_restore_to_new_physical_ids(self):
        layer = Layer.standalone(8, 1)
        packed = word_values(PAGE, 0x7F81)
        tail = word_values(29, -32767)
        loc = slots(5) + slots(6, 0, 29)
        k = torch.cat((packed, tail))
        v = k.flip(0)
        layer.write(loc, k, v)
        saved = layer.read_rows(loc).cpu()
        layer.release([5, 6])
        target = slots(2) + slots(1, 0, 29)
        layer.write(target, saved[:, 0], saved[:, 1])
        self.assert_bits(layer.read_rows(target), saved)

    def test_reference_attention_equality(self):
        generator = torch.Generator().manual_seed(217)
        for real_like in (False, True):
            with self.subTest(real_like=real_like):
                layer = Layer.standalone(5, 2)
                lengths = 2 * PAGE + 19
                locations = slots(3) + slots(1) + slots(2, 0, 19)
                k = kv_values(lengths, 2, "cpu", generator, real_like=real_like)
                v = kv_values(lengths, 2, "cpu", generator, real_like=real_like)
                layer.write(locations, k, v)
                for qlen in (1, 8, 64):
                    q = torch.randn((qlen, 6, DIM), generator=generator).bfloat16()
                    bias = torch.randn((qlen, 6, 119), generator=generator).bfloat16()
                    gathered = layer.read_rows(locations)
                    expect = reference(q, k, v, bias)
                    actual = reference(q, gathered[:, 0], gathered[:, 1], bias)
                    for a, b in zip(actual, expect):
                        self.assertTrue(torch.equal(a.view(torch.int32), b.view(torch.int32)))

    def test_shared_envelope_has_independent_visit_flags_and_planes(self):
        stride = 2 * Layout(1).size
        raw = torch.zeros(3 * stride, dtype=torch.uint8)
        first = Layer(raw, 1, stride, 0, 1, 1)
        second = Layer(raw, 1, stride, Layout(1).size, 1, 1)
        full = word_values(PAGE, 0x3F80)
        partial = word_values(11, 0x4000)
        first.write(slots(1), full, full)
        second.write(slots(1, 0, 11), partial, partial)
        self.assertEqual(first.flags[1].item(), PACKED)
        self.assertEqual(second.flags[1].item(), 12)
        self.assert_bits(first.read_rows(slots(1))[:, 0], full)
        self.assert_bits(second.read_rows(slots(1, 0, 11))[:, 0], partial)

    def test_layout_price(self):
        self.assertEqual(Layout(8).size, 397824)
        self.assertTrue(all(off % 4 == 0 for off in Layout(8).offsets))
        self.assertEqual(8 * Layout(8).size // PAGE, 24864)

    def _write_pages(self, layer, raw):
        inputs = raw.transpose(0, 1).reshape(2, -1, 1, DIM).view(torch.bfloat16)
        layer.write(list(range(PAGE, PAGE * (raw.shape[0] + 1))), *inputs)

    def _check_all(self, layer, raw):
        for n in range(0, raw.shape[0], 16):
            block = raw[n:n + 16]
            locations = list(range((n + 1) * PAGE, (n + len(block) + 1) * PAGE))
            got = layer.read_rows(locations).view(len(block), PAGE, 2, 1, DIM).transpose(1, 2)
            self.assert_bits(got, block)

    def test_triton_loader_if_interpreter_is_available(self):
        import importlib.util
        import os

        if importlib.util.find_spec("triton") is None or os.getenv("TRITON_INTERPRET") != "1":
            self.skipTest("requires Triton and TRITON_INTERPRET=1")
        from inkling_kv12s.attention import gather_for_fa4, load_test_pages

        layer = Layer.standalone(4, 1)
        normal = word_values(PAGE, 0x3F80)
        normal.view(torch.int16)[32, 0, -3:] = torch.tensor([0x7FC1, -1, -32767], dtype=torch.int16)
        overflow = word_values(PAGE, 0x7FC1)
        layer.stage.fill_(0x4000)
        for page, words, end, tentative in ((1, normal, PAGE, False), (2, normal, 37, True),
                                            (3, overflow, PAGE, False)):
            layer.write(slots(page, 0, end), words[:end], words[:end], tentative=tentative)
        stage = normal.clone()
        stage[37:].zero_()
        for block in (32, 64, 128):
            actual = load_test_pages(layer, 3, block=block)
            self.assert_bits(actual[0, 0], normal)
            self.assert_bits(actual[1, 1], stage)
            self.assert_bits(actual[2, 0], overflow)

        table = torch.tensor([[3, 1, -999], [3, 2, 999]], dtype=torch.int32)
        seq = torch.tensor([PAGE + 5, PAGE + 37], dtype=torch.int32)
        offsets = torch.tensor([0, 1, 2], dtype=torch.int32)
        k, v, remapped = gather_for_fa4(layer, normal[:2], table, seq, offsets)
        self.assertEqual(k.shape[0], 4)
        self.assertEqual(remapped.tolist(), [[1, 2, 0], [1, 3, 0]])
        self.assertFalse(k[0].any())
        self.assert_bits(k[1], overflow)
        self.assert_bits(k[2, :5], normal[:5])
        self.assertFalse(k[2, 5:].any())
        self.assert_bits(v[3], stage)

        for bad in (-1, layer.overflow.shape[0] + 1):
            layer.planes[4][3, 0, 0] = bad
            self.assertFalse(load_test_pages(layer, 3)[2, 0].any())
        layer.flags[2] = (layer.stage.shape[0] << STAGE_SHIFT) | 38
        self.assertFalse(load_test_pages(layer, 3)[1].any())
        for bad in (-1, CAP + 1):
            layer.planes[3][1, 0, 0, 1:3] = torch.tensor([bad, CAP + 1])
            corrupted = load_test_pages(layer, 3)[0, 0]
            self.assert_bits(corrupted[32, 0, -3:], torch.tensor([0, -32768, -32767], dtype=torch.int16))


if __name__ == "__main__":
    unittest.main()
