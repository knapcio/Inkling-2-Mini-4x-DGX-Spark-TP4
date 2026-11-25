"""Batched bound, serving handoff and fallback tests (CPU)."""

import threading
import types
import unittest
from unittest import mock

import numpy as np
import torch

from inkling_cert import cert_math as cm, certwide, certwide_math as wm, patches


class WideBoundTest(unittest.TestCase):
    def test_weight_radius_covers_both_paths_and_matches_torch(self):
        rng = np.random.default_rng(45)
        w = cm.bf16_round(rng.normal(0, 0.021, (80, cm.K)))
        _, _, d = cm.quantize(w)
        rt, nv = cm.tables(w, d)
        r = wm.weight_radius(rt, nv)
        torch_r = certwide.weight_radius({"rt": torch.from_numpy(rt.T.copy()), "nv": torch.from_numpy(nv)})
        np.testing.assert_array_equal(r, torch_r.numpy())
        self.assertTrue(np.all(r.astype(np.float64) >= np.linalg.norm(w - d, axis=1) + certwide.CN * nv))
        x = cm.bf16_round(rng.normal(0, 5, (192, cm.K)))
        norm = wm.activation_norm(x)
        for sign in (-1, 1):
            err = sign * 2 * cm.C_ACC * norm[:, None] * nv[None, :]
            lo, hi = wm.intervals(x.astype(np.float64) @ d.T + err, norm, r)
            exact = x.astype(np.float64) @ w.T - err
            self.assertTrue(np.all((lo <= exact) & (exact <= hi)))

    def test_disjoint_rows_union_capped_ties_and_pad(self):
        # Distinct m16 groups need distinct vocab tiles. The lower index lies
        # in a different tile and has a slightly smaller *raw* logit.
        def plateau(x):
            return np.round(cm.softcap(x) * 2.0**15) * 2.0**-15

        for m in (17, 24, 64, 128, 192):
            with self.subTest(m=m):
                logits = np.full((m, 512), -9, np.float32)
                shift = 32 * (np.arange(m) // 16)
                logits[np.arange(m), 4 + shift] = 100
                logits[np.arange(m), 19 + shift] = np.nextafter(np.float32(100), np.float32(np.inf))
                logits[:, -1] = 500
                lo, hi = wm.intervals(logits, np.ones(m, np.float32), np.full(512, 0.02, np.float32))
                rows, union, groups, safe = wm.select(lo, hi, 511, 2048, np.ones(m))
                self.assertTrue(safe)
                self.assertTrue(np.all(rows.sum(1) == 2))
                self.assertEqual(union.sum(), 2 * ((m + 15) // 16))
                self.assertFalse(union[-1])
                np.testing.assert_array_equal(wm.compact_token(logits, 511, groups, plateau), 4 + shift)
                np.testing.assert_array_equal(wm.compact_token(logits, 511, groups, plateau),
                                              cm.stock_token(logits, 511, plateau))

    def test_inclusive_threshold_and_overcapacity_fallback(self):
        logit = np.zeros((24, 128), np.float32)
        logit[:, 2] = 50
        cut, ok = wm.thresholds(logit.max(1), logit.max(1))
        self.assertTrue(ok.all())
        hi = logit.copy()
        hi[:, 17] = cut
        norm = np.ones(24)
        rows, union, _, safe = wm.select(logit, hi, 128, 2, norm)
        self.assertEqual(union.sum(), 2)
        self.assertTrue(rows[:, 1].all())
        self.assertTrue(safe)
        self.assertFalse(wm.select(logit, hi, 128, 1, norm)[-1])
        self.assertFalse(wm.select(logit, hi, 128, 2, norm, on=False)[-1])
        flat = np.zeros_like(logit)
        self.assertFalse(wm.select(flat, flat, 128, 8, norm)[-1])

    def test_derivative_and_cutoff_round_conservatively(self):
        L = np.linspace(0, 500, 2049, dtype=np.float32)
        self.assertTrue(np.all(wm.derivative_lower(L) <= np.exp(-2 * L.astype(np.float64) / cm.CAP)))
        T = np.array([-150.4, -150.3, -100, -1, 0, 1, 100, 150.3, 150.4], np.float32)
        H = np.maximum(T, 1)
        cutoff, ok = wm.thresholds(T, H)
        np.testing.assert_array_equal(ok, [False, True, True, True, True, True, True, True, False])
        ideal_slope = np.exp(-2 * np.maximum(np.abs(T.astype(np.float64)) + cm.DMAX, H) / cm.CAP)
        self.assertTrue(np.all((T.astype(np.float64) - cutoff)[ok] * ideal_slope[ok] >= 2 * cm.TWO_E))

    def test_one_unsafe_row_forces_full_head(self):
        logits = np.zeros((192, 64), np.float32)
        logits[:, 1] = 50
        norm = np.ones(192)
        self.assertTrue(wm.select(logits, logits, 64, 4, norm)[-1])
        for bad in (200, np.inf, np.nan):
            altered = logits.copy()
            altered[-1, 1] = bad
            self.assertFalse(wm.select(altered, altered, 64, 4, norm)[-1])
        norm[-1] = 2.0**21
        self.assertFalse(wm.select(logits, logits, 64, 4, norm)[-1])

    def test_softcap_reversal_proof_uses_margin(self):
        # The bounded softcap error can cause a slightly smaller raw logit
        # to win outright. Its lower index and separate tile are incidental.
        logits = np.full((17, 64), -np.inf, np.float32)
        logits[:, 0], logits[:, 19] = 99.9999, 100
        def adversarial_softcap(x):
            return cm.softcap(x) + np.where(x < 100, cm.E_SOFTCAP, -cm.E_SOFTCAP)

        rows, _, groups, safe = wm.select(logits, logits, 64, 4, np.ones(17))
        self.assertTrue(safe)
        self.assertTrue(rows[:, 0].all())
        np.testing.assert_array_equal(wm.compact_token(logits, 64, groups, adversarial_softcap), 0)


class WideHookTest(unittest.TestCase):
    def test_both_graph_accept_paths_consume_the_ids(self):
        class AcceptGreedy:
            @classmethod
            def execute(cls, *, target_logits, target_predict=None):
                return target_predict

        stock_ids = torch.tensor([3, 4])
        direct_ids = torch.tensor([8, 9])
        mod = types.SimpleNamespace(AcceptGreedy=AcceptGreedy, sampling_argmax=lambda logits: stock_ids)
        patches._patch_wide_accept(mod)
        shared = torch.empty(2, 32)
        captured = shared.view_as(shared)
        captured._inkling_certwide_ids = direct_ids
        self.assertIs(mod.sampling_argmax(captured), direct_ids)
        self.assertIs(mod.AcceptGreedy.execute(target_logits=captured), direct_ids)
        self.assertIs(mod.AcceptGreedy.execute(target_logits=captured, target_predict=stock_ids), stock_ids)
        self.assertIs(mod.sampling_argmax(shared), stock_ids)
        self.assertIsNone(mod.AcceptGreedy.execute(target_logits=shared))

    def test_head_handoff_is_target_verify_only(self):
        mode = mock.Mock()
        metadata = types.SimpleNamespace(forward_mode=mode)
        stock = object()
        direct = object()
        wrapped = patches.make_get_logits(lambda *args: stock)
        with mock.patch.object(patches, "_wide_logits", return_value=direct) as wide:
            mode.is_target_verify.return_value = True
            self.assertIs(wrapped(None, None, None, metadata), direct)
            mode.is_target_verify.return_value = False
            self.assertIs(wrapped(None, None, None, metadata), stock)
        self.assertEqual(wide.call_count, 1)

    def test_grammar_and_logprobs_never_arm(self):
        for blocked in ("has_grammar", "return_logprob"):
            batch = types.SimpleNamespace(**{blocked: True})
            self.assertFalse(patches.batch_ok(batch, sampling_info=None))

    def test_replay_workspace_rejects_other_streams_and_threads(self):
        stack = certwide.CertWide.__new__(certwide.CertWide)
        stack.q = types.SimpleNamespace(device="cuda:0")
        stack._replay_owner = None
        stack._owner_lock = threading.Lock()
        with mock.patch.object(torch.cuda, "current_stream", return_value=1) as current:
            stack.require_replay_stream()
            stack.require_replay_stream()
            with mock.patch.object(threading, "current_thread", return_value=object()):
                with self.assertRaisesRegex(RuntimeError, "owning thread and CUDA stream"):
                    stack.require_replay_stream()
            current.return_value = 2
            with self.assertRaisesRegex(RuntimeError, "owning thread and CUDA stream"):
                stack.require_replay_stream()

    def test_target_graph_checks_owner_before_replay(self):
        class Runner:
            model_runner = types.SimpleNamespace(is_draft_worker=False)
            capture_forward_mode = types.SimpleNamespace(is_target_verify=lambda: True)
            replay = mock.Mock(return_value="output")

            def execute(self):
                return self.replay()

            def _process_output_after_replay(self, output, forward_batch):
                return output

        wide = mock.Mock()
        with mock.patch.dict(patches.STATE, cert=types.SimpleNamespace(wide=wide), patched=[]):
            patches.patch_graph_runner(types.SimpleNamespace(DecodeCudaGraphRunner=Runner))
            wide.require_replay_stream.side_effect = RuntimeError("wrong stream")
            with self.assertRaisesRegex(RuntimeError, "wrong stream"):
                Runner().execute()
            Runner.replay.assert_not_called()
            wide.require_replay_stream.side_effect = None
            self.assertEqual(Runner().execute(), "output")

    def test_exact_passes_registered_width_to_binding(self):
        stack = certwide.CertWide.__new__(certwide.CertWide)
        stack.wide_ext = mock.Mock()
        stack.entry = types.SimpleNamespace(n=128, buf=object())
        stack.sparse = torch.empty((24, 32))
        stack.state, stack.tiles, stack.groups = object(), object(), object()
        x, wrong_width = torch.empty((24, cm.K)), torch.empty((24, 112))
        stack.exact(x, wrong_width, 120, types.SimpleNamespace(mt=0, mg=2, pfd=1))
        self.assertEqual(stack.wide_ext.certwide_exact.call_args.args[7:], (128, 120, 0, 2, 1))


if __name__ == "__main__":
    unittest.main()
