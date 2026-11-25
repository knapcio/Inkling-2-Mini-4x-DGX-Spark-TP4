"""CPU tests of inkling_cert's SGLang hooks with stand-in objects (no sglang, no GPU)."""

import os
import sys
import types
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inkling_cert import patches  # noqa: E402


class FakeCert:
    def __init__(self, pack):
        self.pack = pack
        self.on_values = []
        self.runs = []
        self.warm = set()
        self.on, self.off = "ON", "OFF"

    def set_on(self, v):
        self.on_values.append(1 if v else 0)

    def run(self, x2, out, on, v_real, cap):
        self.runs.append((x2.shape[0], on, v_real, cap))
        out.fill_(1.0)
        return out


class Pack:
    def __init__(self, n, k):
        self.n, self.k = n, k


def install_fake_pico(weight, pack):
    pp = types.ModuleType("inkling_pico12.patches")
    pp.STATE = {"levers": ("lmhead",)}
    pp.HEADS = {weight.data_ptr(): pack}
    po = types.ModuleType("inkling_pico12.ops")
    po.rows_of = lambda x: x.numel() // x.shape[-1]
    po.eligible = lambda x, p: 0 < po.rows_of(x) <= 16
    po.out_ok = lambda out, m, p, dt, dev: tuple(out.shape) == (m, p.n)
    po.as_rows = lambda x, k: x.reshape(-1, k)
    sys.modules["inkling_pico12.patches"] = pp
    sys.modules["inkling_pico12.ops"] = po


class Mode:
    def __init__(self, verify):
        self.verify = verify

    def is_target_verify(self):
        return self.verify


class LP:
    shard_softcap = 30.0
    logit_scale = None
    do_tensor_parallel_all_gather = False
    do_tensor_parallel_all_gather_dp_attn = False

    def __init__(self, vocab):
        self.config = types.SimpleNamespace(vocab_size=vocab)


class HookTest(unittest.TestCase):
    def setUp(self):
        self.V, self.K = 64, 2304
        self.w = torch.zeros((self.V, self.K), dtype=torch.bfloat16)
        self.pack = Pack(self.V, self.K)
        install_fake_pico(self.w, self.pack)
        self.cert = FakeCert(self.pack)
        patches.STATE.update(mode="1", cfg={"log_every": 0, "cache_dir": "/tmp"}, cert=self.cert, disabled=None,
                             log=lambda m: None)
        self.inner_calls = []

        def inner(x, weight, out):
            self.inner_calls.append(x.shape[0])
            return torch.zeros((x.shape[0], self.V))

        inner._pico12 = True
        self.head = patches.make_lm_head(inner)
        self.lp = LP(60)

        def get_logits(lp, hidden, lm_head, meta):
            return self.head(hidden, self.w, None)

        self.get_logits = patches.make_get_logits(get_logits)

    def x(self, m):
        return torch.zeros((m, self.K), dtype=torch.bfloat16)

    def test_only_target_verify(self):
        self.get_logits(self.lp, self.x(8), None, types.SimpleNamespace(forward_mode=Mode(False)))
        self.assertEqual(self.inner_calls, [8])
        self.assertEqual(self.cert.runs, [])
        self.assertIsNone(patches.CTX["lp"])

    def test_eager_verify_first_call_warms_then_pico(self):
        meta = types.SimpleNamespace(forward_mode=Mode(True))
        self.get_logits(self.lp, self.x(8), None, meta)          # first eager M <= 8: kernels, full fallback
        self.get_logits(self.lp, self.x(8), None, meta)          # then pico directly
        self.get_logits(self.lp, self.x(16), None, meta)         # first eager M 9..16
        self.assertEqual([r[1] for r in self.cert.runs], ["OFF", "OFF"])
        self.assertEqual(self.inner_calls, [8])
        self.assertEqual(self.cert.runs[0][2:], (60, 30.0))

    def test_rows_above_16_and_bad_processor(self):
        meta = types.SimpleNamespace(forward_mode=Mode(True))
        self.get_logits(self.lp, self.x(24), None, meta)
        self.assertEqual(self.inner_calls, [24])
        lp = LP(60)
        lp.logit_scale = 2.0
        self.get_logits(lp, self.x(8), None, meta)
        self.assertIsNotNone(patches.STATE["disabled"])
        self.get_logits(self.lp, self.x(8), None, meta)          # disabled for good
        self.assertEqual(self.cert.runs, [])

    def test_mode_off(self):
        patches.STATE["mode"] = "0"
        self.get_logits(self.lp, self.x(8), None, types.SimpleNamespace(forward_mode=Mode(True)))
        self.assertEqual(self.cert.runs, [])


class VerifyFlagTest(unittest.TestCase):
    def setUp(self):
        self.cert = FakeCert(None)
        patches.STATE.update(mode="1", cfg={"log_every": 0, "cache_dir": "/tmp"}, cert=self.cert, disabled=None,
                             log=lambda m: None)
        calls = []

        class Epi:
            simulate_fold = False

            def begin_static_step(self, bs, armed):
                calls.append(("static", armed))

            def begin_step(self, verify_lens, armed):
                calls.append(("step", armed))

        class Exec:
            _simulated_correct_len = None

            def __init__(self):
                self.verify_epilogue = Epi()

            def run_non_compact(self, *, batch, draft_input, verify_ids_2d, verify_window, sampling_info):
                calls.append(("run",))
                return "result"

        mod = types.ModuleType("dspark_verify")
        mod.DsparkVerifyEpilogue, mod.TargetVerifyExecutor = Epi, Exec
        patches.patch_verify(mod)
        self.ex, self.calls = Exec(), calls

    def run_step(self, armed, batch=None, si=None):
        self.ex.verify_epilogue.begin_static_step(1, armed)
        batch = batch or types.SimpleNamespace(return_logprob=False, has_grammar=False, sampling_info=si)
        r = self.ex.run_non_compact(batch=batch, draft_input=None, verify_ids_2d=None, verify_window=None,
                                    sampling_info=si)
        self.assertEqual(r, "result")
        return self.cert.on_values[-1]

    def greedy(self, **kw):
        d = dict(is_all_greedy=True, has_watermark_candidates=False, sampling_mask_batch_indices=None)
        d.update(kw)
        return types.SimpleNamespace(**d)

    def test_flag(self):
        self.assertEqual(self.run_step(True, si=self.greedy()), 1)
        self.assertEqual(self.run_step(False, si=self.greedy()), 0)
        self.assertEqual(self.run_step(True, si=self.greedy(is_all_greedy=False)), 0)
        self.assertEqual(self.run_step(True, si=self.greedy(has_watermark_candidates=True)), 0)
        self.assertEqual(self.run_step(True, si=self.greedy(sampling_mask_batch_indices=torch.zeros(1))), 0)
        b = types.SimpleNamespace(return_logprob=True, has_grammar=False, sampling_info=None)
        self.assertEqual(self.run_step(True, batch=b, si=None), 0)
        b = types.SimpleNamespace(return_logprob=False, has_grammar=False, sampling_info=self.greedy(is_all_greedy=False))
        self.assertEqual(self.run_step(True, batch=b, si=None), 0)      # the batch samples although si is None
        self.assertEqual(self.run_step(True, si=None), 1)

    def test_begin_step_clears(self):
        self.assertEqual(self.run_step(True, si=self.greedy()), 1)
        self.ex.verify_epilogue.begin_step(None, armed=True)
        self.assertEqual(self.cert.on_values[-1], 0)
        b = types.SimpleNamespace(return_logprob=False, has_grammar=False, sampling_info=None)
        self.ex.run_non_compact(batch=b, draft_input=None, verify_ids_2d=None, verify_window=None, sampling_info=None)
        self.assertEqual(self.cert.on_values[-1], 0)                     # not re-armed without begin_static_step

    def test_simulated_never(self):
        self.ex._simulated_correct_len = torch.zeros(1)
        self.assertEqual(self.run_step(True, si=self.greedy()), 0)


class TensorParallel(unittest.TestCase):
    """prepare() at TP > 1 disables the certificate (logged once), before any pico requirement; TP1 goes on."""

    def setUp(self):
        self.logs = []
        patches.STATE.update(mode="1", cfg={"cache_dir": "/tmp"}, cert=None, disabled=None, log=self.logs.append)
        self.saved = {k: sys.modules.get(k) for k in ("sglang", "sglang.srt", "sglang.srt.runtime_context",
                                                      "inkling_pico12.patches", "inkling_pico12.ops")}
        for k in ("inkling_pico12.patches", "inkling_pico12.ops"):  # pico not armed
            sys.modules.pop(k, None)

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        patches.STATE.update(disabled=None, log=None)

    def runtime(self, tp):
        rc = types.ModuleType("sglang.srt.runtime_context")
        rc.get_parallel = lambda: types.SimpleNamespace(tp_size=tp)
        sys.modules.update({"sglang": types.ModuleType("sglang"), "sglang.srt": types.ModuleType("sglang.srt"),
                            "sglang.srt.runtime_context": rc})

    @staticmethod
    def model(head_tp):
        head = types.SimpleNamespace(weight=torch.zeros((4, 8), dtype=torch.bfloat16))
        if head_tp is not None:
            head.tp_size = head_tp
        return types.SimpleNamespace(llm=types.SimpleNamespace(lm_head=head))

    def test_tp4_disables_from_runtime_context(self):
        self.runtime(4)
        for _ in range(3):  # every call (e.g. a reload) stays disabled; logged once
            patches.prepare(self.model(None))
        self.assertEqual(patches.STATE["disabled"], "tensor parallel size 4 (TP1 only)")
        self.assertEqual(len(self.logs), 1)
        self.assertIsNone(patches.STATE["cert"])

    def test_tp4_from_the_head_without_runtime_context(self):
        sys.modules.pop("sglang.srt.runtime_context", None)
        sys.modules["sglang.srt.runtime_context"] = None  # import raises
        patches.prepare(self.model(4))
        self.assertEqual(patches.STATE["disabled"], "tensor parallel size 4 (TP1 only)")

    def test_tp1_continues(self):
        self.runtime(1)
        with self.assertRaises(patches.ops.CertFatal):  # TP1: on to the pico requirement, as before
            patches.prepare(self.model(1))
        self.assertIsNone(patches.STATE["disabled"])
        self.assertEqual(patches.tp_size(self.model(1).llm.lm_head), 1)


if __name__ == "__main__":
    unittest.main()
