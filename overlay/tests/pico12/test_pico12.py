"""CPU unit tests of inkling_pico12's Python side (torch CPU; no GPU, no nvcc, no sglang).

    python -m unittest discover -s tests/pico12 -v        (from overlay/)

The CUDA kernels are not run here: tools/gpu_check_pico12.py checks them on the Spark. A stand-in extension
(fake_ext.py) keeps the kernel's call contract (rows <= 16, contiguous aligned rows, [M, N] outputs) and
computes reference math, and a stand-in sglang package (fake_sglang/) carries the stock code of the patched
functions. Covered: settings and lever parsing, the source checks (also against a real sglang tree when
PICO12_SGLANG_SRC names one), the dispatch decisions of every patch point, the stacked
qkvr path (values, call counts, and the exact view layout of the stock path), the LM head and draft head
wrappers, reloads in place, the composition with inkling_kernels.qkvr_gemv and with a subclassing overlay, the
no-op contract, the loud failures (exit 78), and the whole flow through sitecustomize in a subprocess.
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
FAKE_ROOT = os.path.join(HERE, "fake_sglang")
for p in (OVERLAY, FAKE_ROOT, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import inkling_pico12  # noqa: E402
from fake_ext import FakeExt  # noqa: E402
from inkling_pico12 import ops, patches  # noqa: E402
from sglang.srt import batch_invariant_ops  # noqa: E402
from sglang.srt.layers.quantization import unquant as uq  # noqa: E402
from sglang.srt.models import dspark, inkling2, inkling2_mtp  # noqa: E402
from sglang.srt.models.inkling_common import inkling2_attn as attn_mod  # noqa: E402

STOCK_PROJECT = attn_mod.Inkling2Attention._project_sources
REAL_SGLANG = os.environ.get("PICO12_SGLANG_SRC", "")  # a real sglang package dir (e.g. copied from the image)
LOGS = []
EXT = FakeExt()
ALL = ("dense", "lmhead", "drafthead", "draft")


def configure(levers=ALL, **settings):
    ops.EXT = EXT
    ops.SET.require_cuda = False
    ops.SET.max_m = settings.get("max_m", 16)
    ops.SET.stacked_max_m = settings.get("stacked_max_m", 16)
    ops.SET.wk = settings.get("wk", 1)  # K = 64 in the stand-ins: 2 steps of 32
    ops.SET.pdl = True
    ops.LOG = LOGS.append
    ops._batch_invariant = batch_invariant_ops.is_batch_invariant_mode_enabled
    ops._warned.clear()
    patches.STATE.update(levers=tuple(levers), cfg={"cache_dir": "/nonexistent", "std": "c++20", "verbose": False},
                         log=LOGS.append, patched=[])
    patches.HEADS.clear()
    patches._spec_dtype[0] = patches._SPEC_UNSET
    EXT.reset()
    EXT.pad = 64
    for key in uq.CALLS:
        uq.CALLS[key] = 0
    dspark.CALLS["project"] = 0
    dspark.SPEC.speculative_dspark_lm_head_dtype = None
    uq.STOCK_GEMV[0] = True
    batch_invariant_ops.ENABLED[0] = False


def bf16(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g).to(torch.bfloat16)


def ref(x, w, dtype=torch.bfloat16):
    return (x.reshape(-1, x.shape[-1]).float() @ w.float().T).to(dtype).view(*x.shape[:-1], w.shape[0])


def bits_equal(a, b):
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.dtype == torch.bfloat16 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def model_with_packs(levers=ALL, layers=1, **settings):
    configure(levers, **settings)
    model = inkling2.Inkling2ForConditionalGeneration(layers=layers)
    patches.prepare_target(model)
    return model


class Settings(unittest.TestCase):
    def test_levers(self):
        self.assertEqual(inkling_pico12.parse_levers(""), ())
        self.assertEqual(inkling_pico12.parse_levers(" dense , lmhead,dense"), ("dense", "lmhead"))
        self.assertEqual(inkling_pico12.parse_levers("all"), ALL)
        self.assertEqual(inkling_pico12.parse_levers("off"), ())
        with self.assertRaises(RuntimeError):
            inkling_pico12.parse_levers("dense,wq")
        with self.assertRaises(RuntimeError):
            inkling_pico12.parse_levers("off,dense")

    def test_config(self):
        cfg = inkling_pico12.config_from_env({})
        self.assertEqual((cfg["max_m"], cfg["stacked_max_m"], cfg["wk"], cfg["pdl"], cfg["std"]),
                         (16, 16, 16, True, "c++20"))
        self.assertTrue(cfg["cache_dir"].endswith(os.path.join(".cache", "inkling-pico12")))
        cfg = inkling_pico12.config_from_env({"INKLING_PICO12_MAX_M": "8", "INKLING_PICO12_STACKED_MAX_M": "0",
                                              "INKLING_PICO12_PDL": "0", "INKLING_PICO12_CACHE": "/c",
                                              "INKLING_PICO12_STD": "c++17"})
        self.assertEqual((cfg["max_m"], cfg["stacked_max_m"], cfg["pdl"], cfg["cache_dir"], cfg["std"]),
                         (8, 0, False, "/c", "c++17"))
        for name, value in (("INKLING_PICO12_MAX_M", "17"), ("INKLING_PICO12_MAX_M", "0"),
                            ("INKLING_PICO12_WK", "x"), ("INKLING_PICO12_STD", "c++14"),
                            ("INKLING_PICO12_STACKED_MAX_M", "-1")):
            with self.assertRaises(ValueError, msg=f"{name}={value}"):
                inkling_pico12.config_from_env({name: value})

    def test_hook_plan(self):
        plan = dict(inkling_pico12.hook_plan(("lmhead",)))
        self.assertEqual(set(plan), {inkling_pico12.UNQUANT, inkling_pico12.MODEL})
        plan = dict(inkling_pico12.hook_plan(ALL))
        self.assertEqual(set(plan), {inkling_pico12.UNQUANT, inkling_pico12.ATTN, inkling_pico12.MODEL,
                                     inkling_pico12.MTP})
        for _, name in inkling_pico12.hook_plan(ALL):
            self.assertTrue(callable(getattr(patches, name)))

    def test_pure_helpers(self):
        self.assertEqual(ops.choose_mt(8600), 1)
        self.assertEqual(ops.choose_mt(2304), 2)
        self.assertEqual(ops.choose_mt(201024), 2)
        self.assertIsNone(ops.shape_reason(8600, 2304, 16))
        self.assertIsNone(ops.shape_reason(2304, 3072, 16))
        self.assertIn("multiple of 8", ops.shape_reason(12, 64, 1))
        self.assertIn("multiple of 32", ops.shape_reason(16, 48, 1))
        self.assertIn("wk", ops.shape_reason(16, 64, 4))
        self.assertEqual(ops.kv_bounds([3072, 1024, 1024, 384, 3072, 24]), (3072, 5120))
        self.assertIsNone(ops.kv_bounds([3070, 1024, 1024]))
        self.assertIsNone(ops.kv_bounds([8]))
        self.assertFalse(ops.stacked_needs_kv(5))
        self.assertTrue(ops.stacked_needs_kv(6))


class SourceChecks(unittest.TestCase):
    def test_fake_tree_passes(self):
        inkling_pico12.check_sources(ALL, os.path.join(FAKE_ROOT, "sglang"))

    def test_missing_needle_and_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "sglang")
            shutil.copytree(os.path.join(FAKE_ROOT, "sglang"), root)
            path = os.path.join(root, "srt/models/inkling_common/inkling2_attn.py")
            with open(path) as f:
                text = f.read()
            with open(path, "w") as f:
                f.write(text.replace("def _stacked(", "def _stacked_renamed("))
            with self.assertRaisesRegex(RuntimeError, "_stacked"):
                inkling_pico12.check_sources(("dense",), root)
            inkling_pico12.check_sources(("lmhead",), root)  # lmhead does not need inkling2_attn
            os.remove(os.path.join(root, "srt/layers/logits_processor.py"))
            with self.assertRaisesRegex(RuntimeError, "missing"):
                inkling_pico12.check_sources(("lmhead",), root)

    @unittest.skipUnless(REAL_SGLANG and os.path.isdir(REAL_SGLANG), "PICO12_SGLANG_SRC not set")
    def test_real_sglang_tree(self):
        inkling_pico12.check_sources(ALL, REAL_SGLANG)


class Dense(unittest.TestCase):
    def setUp(self):
        configure()
        self.layer = attn_mod._Linear(64, 96, seed=3)
        self.pack, why = patches.register_linear(self.layer, "t.wo")
        self.assertIsNone(why)

    def test_register(self):
        self.assertEqual((self.pack.n, self.pack.k, self.pack.mt), (64, 96, 2))
        self.assertTrue(type(self.layer.quant_method).__name__.startswith("Pico12"))
        self.assertIsInstance(self.layer.quant_method, uq.UnquantizedLinearMethod)
        again, _ = patches.register_linear(self.layer, "t.wo")  # a reload: same pack and buffer
        self.assertIs(again, self.pack)

    def test_apply_small_m_goes_to_pico(self):
        for m in (1, 2, 8, 16):
            EXT.reset()
            x = bf16(m, 96, seed=m)
            y, _ = self.layer(x)
            self.assertEqual(EXT.calls, [("gemv", m, 64, 96, torch.bfloat16)])
            self.assertTrue(bits_equal(y, ref(x, self.layer.weight)))
        self.assertEqual(uq.CALLS["apply"], 0)

    def test_apply_leading_dims_and_strided_rows(self):
        x = bf16(2, 4, 96)
        y, _ = self.layer(x)
        self.assertEqual(tuple(y.shape), (2, 4, 64))
        self.assertTrue(bits_equal(y, ref(x, self.layer.weight)))
        wide = bf16(8, 128)
        y, _ = self.layer(wide[:, :96])  # rows 256 B apart: copied to contiguous rows first
        self.assertTrue(bits_equal(y, ref(wide[:, :96], self.layer.weight)))
        flat = bf16(8 * 96 + 1)
        x = flat[1:].view(8, 96)  # contiguous but 2 bytes off 16-byte alignment
        self.assertNotEqual(x.data_ptr() % 16, 0)
        y, _ = self.layer(x)
        self.assertTrue(bits_equal(y, ref(x, self.layer.weight)))
        self.assertEqual(uq.CALLS["apply"], 0)

    def test_apply_falls_back(self):
        cases = [
            ("17 rows", bf16(17, 96)),
            ("0 rows", bf16(0, 96)),
            ("fp32 x", bf16(4, 96).float()),
            ("grad", bf16(4, 96).float().requires_grad_().to(torch.bfloat16)),
        ]
        for what, x in cases:
            EXT.reset()
            before = uq.CALLS["apply"]
            y = self.layer.quant_method.apply(self.layer, x, None)
            self.assertEqual(EXT.calls, [], what)
            self.assertEqual(uq.CALLS["apply"], before + 1, what)
            if x.numel():
                self.assertEqual(tuple(y.shape), (x.shape[0], 64))
        EXT.reset()
        self.layer.quant_method.apply(self.layer, bf16(4, 96), torch.zeros(64, dtype=torch.bfloat16))
        self.assertEqual(EXT.calls, [], "bias")
        ops.SET.max_m = 4
        self.layer.quant_method.apply(self.layer, bf16(5, 96), None)
        self.assertEqual(EXT.calls, [], "max_m")
        batch_invariant_ops.ENABLED[0] = True
        self.layer.quant_method.apply(self.layer, bf16(2, 96), None)
        self.assertEqual(EXT.calls, [], "batch-invariant mode")

    def test_replaced_weight_falls_back_with_one_warning(self):
        self.layer.weight = torch.nn.Parameter(bf16(64, 96, seed=7), requires_grad=False)
        LOGS.clear()
        for _ in range(3):
            y = self.layer.quant_method.apply(self.layer, bf16(2, 96), None)
        self.assertEqual(EXT.calls, [])
        self.assertTrue(bits_equal(y, ref(bf16(2, 96), self.layer.weight)))
        self.assertEqual(len([m for m in LOGS if "storage changed" in m]), 1)

    def test_apply_into(self):
        x = bf16(4, 96)
        out = torch.empty(4, 64, dtype=torch.bfloat16)
        got = self.layer.quant_method.apply_into(self.layer, x, out)
        self.assertIs(got, out)
        self.assertEqual(EXT.calls, [("gemv", 4, 64, 96, torch.bfloat16)])
        self.assertTrue(bits_equal(out, ref(x, self.layer.weight)))
        EXT.reset()
        wide = torch.empty(4, 80, dtype=torch.bfloat16)
        got = self.layer.quant_method.apply_into(self.layer, x, wide[:, :64])  # row stride 80: stock writes it
        self.assertEqual(EXT.calls, [])
        self.assertEqual(uq.CALLS["apply_into"], 1)
        self.assertTrue(bits_equal(got, ref(x, self.layer.weight)))

    def test_not_registered(self):
        biased = attn_mod._Linear(64, 96, bias=True)
        self.assertEqual(patches.register_linear(biased, "b")[1], "has a bias")
        odd = attn_mod._Linear(12, 96)
        self.assertIn("multiple of 8", patches.register_linear(odd, "o")[1])
        lin = attn_mod._Linear(64, 96)
        lin.quant_method = object()
        self.assertIn("quant method", patches.register_linear(lin, "q")[1])
        lin = attn_mod._Linear(64, 96)
        lin.weight = torch.nn.Parameter(lin.weight.float(), requires_grad=False)
        self.assertIn("BF16", patches.register_linear(lin, "f")[1])

    def test_debug_wrapper_dropped_and_shared_method_untouched(self):
        lin = attn_mod._Linear(64, 96)
        shared = lin.quant_method
        other = attn_mod._Linear(64, 96)
        other.quant_method = shared
        shared.apply = lambda *a, **k: (_ for _ in ()).throw(AssertionError("instance wrapper ran"))
        patches.register_linear(lin, "d")
        self.assertNotIn("apply", lin.quant_method.__dict__)
        self.assertIs(type(shared), uq.UnquantizedLinearMethod)  # the other layer keeps the stock method
        lin(bf16(2, 96))
        self.assertEqual(len(EXT.calls), 1)

    def test_composes_with_subclassing_overlay(self):
        seen = []

        def overlay_class(cls):  # the pattern inkling_draftbytes uses (draftmx8)
            inner = cls.apply

            def apply(self, layer, x, bias=None):
                if getattr(layer, "_w8", False) and x.shape[0] <= 4:
                    seen.append("w8")
                    return torch.zeros(x.shape[0], layer.weight.shape[0], dtype=x.dtype)
                return inner(self, layer, x, bias)

            return type("DraftBytesW8" + cls.__name__, (cls,), {"apply": apply})

        lin = attn_mod._Linear(64, 96)
        patches.register_linear(lin, "draft.x")  # pico first (draft load) ...
        lin.quant_method.__class__ = overlay_class(type(lin.quant_method))  # ... then the w8 overlay (attach)
        lin._w8 = True
        lin(bf16(4, 96))
        self.assertEqual((seen, EXT.calls), (["w8"], []))
        lin(bf16(8, 96))
        self.assertEqual(len(EXT.calls), 1)
        patches.register_linear(lin, "draft.x")  # a reload keeps the outer class
        self.assertTrue(type(lin.quant_method).__name__.startswith("DraftBytesW8"))


class Stacked(unittest.TestCase):
    def check(self, attn, m, layout, expect_calls, pico=True):
        g = torch.Generator().manual_seed(m)
        buf = (torch.randn((3, m, 64), generator=g)).to(torch.bfloat16)
        if layout == "adjacent":
            srcs = attn_mod.AttentionSources(*buf.unbind(0))
        else:
            srcs = attn_mod.AttentionSources(*(t.clone() for t in buf.unbind(0)))
        wrapped = patches.make_project_sources(STOCK_PROJECT, attn_mod._stacked)
        EXT.reset()
        uq.CALLS["apply"] = 0
        got = wrapped(attn, srcs)
        calls = list(EXT.calls)
        stock_apply = uq.CALLS["apply"]
        want = STOCK_PROJECT(attn, srcs)
        for a, b, name in zip(got, want, ("q", "k", "v", "r", "gate", "sinks")):
            self.assertEqual((a.shape, a.stride(), a.storage_offset()), (b.shape, b.stride(), b.storage_offset()),
                             f"M={m} {layout} {name}: view layout differs from the stock path")
            self.assertTrue(bits_equal(a, b), f"M={m} {layout} {name}: values differ")
        if pico:
            self.assertEqual(len(calls), expect_calls, f"M={m}")
            self.assertEqual(stock_apply, 0)
        return calls

    def test_stacked_paths(self):
        model = model_with_packs(("dense",))
        attn = model.llm.layers[0].attn
        plan = attn.__dict__["_pico12_qkvr"]
        self.assertEqual((plan.kv_lo, plan.kv_hi, plan.kv.n, plan.full.n), (3072, 5120, 2048, 8600))
        for layout in ("adjacent", "separate"):
            calls = self.check(attn, 1, layout, 1)
            self.assertEqual(calls[0][1:3], (3, 8600))
            calls = self.check(attn, 5, layout, 1)  # 15 rows: one call over all columns
            self.assertEqual(calls[0][1:3], (15, 8600))
            calls = self.check(attn, 8, layout, 2)  # q rows over all columns, then 16 k+v rows on the kv pack
            self.assertEqual([c[1:3] for c in calls], [(8, 8600), (16, 2048)])
            calls = self.check(attn, 9, layout, 3)
            self.assertEqual([c[1:3] for c in calls], [(9, 8600), (16, 2048), (2, 2048)])
            calls = self.check(attn, 16, layout, 3)
            self.assertEqual([c[1:3] for c in calls], [(16, 8600), (16, 2048), (16, 2048)])
            self.check(attn, 17, layout, 0, pico=False)
            self.assertEqual(EXT.calls, [])  # 51 stacked rows: the stock projection

    def test_stacked_max_m_without_kv_pack(self):
        model = model_with_packs(("dense",), stacked_max_m=5)
        attn = model.llm.layers[0].attn
        self.assertIsNone(attn.__dict__["_pico12_qkvr"].kv)
        self.check(attn, 5, "adjacent", 1)
        self.check(attn, 6, "adjacent", 0, pico=False)
        self.assertEqual(EXT.calls, [])

    def test_shared_sources_use_the_plain_path(self):
        model = model_with_packs(("dense",))
        attn = model.llm.layers[0].attn
        wrapped = patches.make_project_sources(STOCK_PROJECT, attn_mod._stacked)
        x = bf16(4, 64)
        got = wrapped(attn, attn_mod.AttentionSources(x, x, x))
        self.assertEqual([c[1:3] for c in EXT.calls], [(4, 8600)])
        want = STOCK_PROJECT(attn, attn_mod.AttentionSources(x, x, x))
        self.assertTrue(all(bits_equal(a, b) for a, b in zip(got, want)))

    def test_unpackable_kv_bounds(self):
        configure(("dense",))
        model = inkling2.Inkling2ForConditionalGeneration(layers=1, splits=[3076, 1024, 1020, 384, 3072, 24])
        patches.prepare_target(model)
        attn = model.llm.layers[0].attn
        plan = attn.__dict__["_pico12_qkvr"]
        self.assertIsNone(plan.kv)
        self.check(attn, 4, "adjacent", 1)
        self.check(attn, 8, "adjacent", 0, pico=False)


class StackedOff(unittest.TestCase):
    def test_stacked_off_still_plain(self):
        # With stacked calls off, 3M <= 16 stacked rows still reach qkvr's own pico method via the stock code.
        model = model_with_packs(("dense",), stacked_max_m=0)
        attn = model.llm.layers[0].attn
        buf = bf16(3, 2, 64)
        STOCK_PROJECT(attn, attn_mod.AttentionSources(*buf.unbind(0)))
        self.assertEqual([c[1:3] for c in EXT.calls], [(6, 8600)])


class Heads(unittest.TestCase):
    def setUp(self):
        self.model = model_with_packs(("lmhead", "drafthead"))
        self.w = self.model.lm_head.weight
        self.fn, self.applies = patches.make_lm_head(uq.sm12x_fp32_lm_head, uq.sm12x_fp32_lm_head_applies)

    def test_registered(self):
        self.assertEqual(list(patches.HEADS), [self.w.data_ptr()])
        self.assertIn("head [256x64] for lmhead+drafthead", LOGS[-1])

    def test_lm_head(self):
        x = bf16(8, 64)
        y = self.fn(x, self.w, None)
        self.assertEqual(EXT.calls, [("gemv", 8, 256, 64, torch.float32)])
        self.assertTrue(bits_equal(y, ref(x, self.w, torch.float32)))
        EXT.reset()
        out = torch.empty(8, 256)
        y = self.fn(x, self.w, out)
        self.assertEqual(y.data_ptr(), out.data_ptr())
        self.assertTrue(bits_equal(out, ref(x, self.w, torch.float32)))
        self.assertEqual(uq.CALLS["lm_head"], 0)
        EXT.reset()
        self.fn(bf16(17, 64), self.w, None)
        self.fn(x, self.w, torch.empty(8, 300)[:, :256])
        self.fn(x, bf16(256, 64), None)  # another weight
        self.assertEqual((EXT.calls, uq.CALLS["lm_head"]), ([], 3))

    def test_applies(self):
        self.assertTrue(self.applies(bf16(2, 64), self.w, None))
        uq.STOCK_GEMV[0] = False  # stock GEMV backend off: pico still takes its rows
        self.assertTrue(self.applies(bf16(2, 64), self.w, None))
        self.assertFalse(self.applies(bf16(17, 64), self.w, None))
        self.assertFalse(self.applies(bf16(2, 64), self.w, torch.empty(2, 300)[:, :256]))
        self.assertFalse(self.applies(bf16(2, 64), bf16(256, 64), None))

    def test_draft_head(self):
        draft = inkling2_mtp.Inkling2ForConditionalGenerationMTP()
        draft.attach_shared_modules(embed_tokens=None, lm_head=self.model.lm_head)
        wrapped = patches.make_project_through_lm_head(dspark.project_through_lm_head)
        x = bf16(8, 64)
        y = wrapped(x, draft.lm_head)
        self.assertEqual(EXT.calls, [("gemv", 8, 256, 64, torch.bfloat16)])
        self.assertTrue(bits_equal(y, dspark.project_through_lm_head(x, draft.lm_head)))
        EXT.reset()
        wrapped(x.float(), draft.lm_head)
        wrapped(bf16(17, 64), draft.lm_head)
        self.model.lm_head._idb_head = object()  # inkling_draftbytes head8 owns the draft head
        wrapped(x, draft.lm_head)
        del self.model.lm_head._idb_head
        patches._spec_dtype[0] = patches._SPEC_UNSET
        dspark.SPEC.speculative_dspark_lm_head_dtype = "fp8"
        wrapped(x, draft.lm_head)
        self.assertEqual(EXT.calls, [])


class Reload(unittest.TestCase):
    def test_reload_codes_in_place(self):
        model = model_with_packs(ALL)
        attn = model.llm.layers[0].attn
        packs = (attn.qkvr.__dict__["_pico12"], attn.__dict__["_pico12_qkvr"].kv, attn.wo_ud.__dict__["_pico12"],
                 patches.HEADS[model.lm_head.weight.data_ptr()])
        ptrs = [p.buf.data_ptr() for p in packs]
        new_wo = bf16(*attn.wo_ud.weight.shape, seed=42)
        model.load_weights([("llm.layers.0.attn.wo_ud.weight", new_wo)])
        self.assertEqual(model.post_loads, 1)  # stand-in load_weights calls post_load_weights; not patched here
        patches.prepare_target(model)
        self.assertEqual([p.buf.data_ptr() for p in packs], ptrs)
        self.assertIs(attn.wo_ud.__dict__["_pico12"], packs[2])
        x = bf16(3, 96)
        y, _ = attn.wo_ud(x)
        self.assertTrue(bits_equal(y, ref(x, new_wo)))

    def test_reload_that_needs_more_bytes_is_fatal(self):
        model_with_packs(("dense",))
        layer = attn_mod._Linear(64, 96)
        pack, _ = patches.register_linear(layer, "r")
        EXT.pad = 4096
        with self.assertRaises(ops.Pico12Fatal):
            patches.register_linear(layer, "r")


class Draft(unittest.TestCase):
    def test_prepare_draft(self):
        configure(("draft",))
        draft = inkling2_mtp.Inkling2ForConditionalGenerationMTP(layers=2)
        LOGS.clear()
        patches.prepare_draft(draft)
        self.assertRegex(LOGS[-1], r"active \(draft\): 8 linears packed .* skipped 2 \(0.odd: N=12")
        layer = draft.layers[1]
        x = bf16(8, 64)
        y, _ = layer.gate_up_proj(x)
        self.assertEqual(EXT.calls, [("gemv", 8, 256, 64, torch.bfloat16)])
        self.assertTrue(bits_equal(y, ref(x, layer.gate_up_proj.weight)))


class QkvrGemvComposition(unittest.TestCase):
    """inkling_kernels.qkvr_gemv patches the same method; pico must end up outermost in either hook order."""

    def fake_inkling_kernels(self, levers=("qkvr_gemv",)):
        ik = types.ModuleType("inkling_kernels")
        ik.STATE = {"levers": levers}
        ik.qkvr_config_from_env = lambda: {"max_m": 16, "bn": 32, "bk": 256, "warps": 4}
        qg = types.ModuleType("inkling_kernels.qkvr_gemv")
        qg.applied = 0

        def patch(module, cfg):  # the early return and wrapper of the real qkvr_gemv.patch
            cls = module.Inkling2Attention
            stock = cls._project_sources
            if getattr(stock, "inkling_cfg", None) is not None:
                return
            qg.applied += 1

            def _project_sources(self, sources):
                return stock(self, sources)

            _project_sources.inkling_cfg = cfg
            _project_sources.__wrapped__ = stock
            cls._project_sources = _project_sources

        qg.patch = patch
        ik.qkvr_gemv = qg
        return ik, qg

    def run_order(self, pico_first):
        configure(("dense",))
        ik, qg = self.fake_inkling_kernels()
        saved = {k: sys.modules.get(k) for k in ("inkling_kernels", "inkling_kernels.qkvr_gemv")}
        sys.modules["inkling_kernels"], sys.modules["inkling_kernels.qkvr_gemv"] = ik, qg
        cls = attn_mod.Inkling2Attention
        try:
            if pico_first:
                patches.patch_attn(attn_mod)
                qg.patch(attn_mod, ik.qkvr_config_from_env())  # inkling_kernels' own hook, later
            else:
                qg.patch(attn_mod, ik.qkvr_config_from_env())
                patches.patch_attn(attn_mod)
            outer = cls._project_sources
            self.assertTrue(getattr(outer, "inkling_pico12", False), "pico is not outermost")
            self.assertIsNotNone(outer.inkling_cfg)
            self.assertIsNotNone(getattr(outer.__wrapped__, "inkling_cfg", None))
            self.assertIs(outer.__wrapped__.__wrapped__, STOCK_PROJECT)
            self.assertEqual(qg.applied, 1)
            self.assertEqual(patches.STATE["qkvr_composed"], "qkvr_gemv")
        finally:
            cls._project_sources = STOCK_PROJECT
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

    def test_kernels_hook_first(self):
        self.run_order(pico_first=False)

    def test_pico_hook_first(self):
        self.run_order(pico_first=True)


def run_py(code, env_extra, root=FAKE_ROOT):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("INKLING_", "PYTHON"))}
    env.update(PYTHONPATH=os.pathsep.join([OVERLAY, root, HERE]), **env_extra)
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)


E2E = r"""
import json, sys, torch
import inkling_pico12
from inkling_pico12 import ops
from fake_ext import FakeExt
assert 'torch' in sys.modules
ops.EXT = FakeExt(); ops.SET.require_cuda = False
from sglang.srt.layers.quantization import unquant as uq
from sglang.srt.models import inkling2, inkling2_mtp
from sglang.srt.models.inkling_common import inkling2_attn as attn_mod
model = inkling2.Inkling2ForConditionalGeneration(layers=2)
model.load_weights()
draft = inkling2_mtp.Inkling2ForConditionalGenerationMTP()
draft.load_weights()
draft.attach_shared_modules(embed_tokens=None, lm_head=model.lm_head)
attn = model.llm.layers[1].attn
buf = torch.randn(3, 8, 64).to(torch.bfloat16)
srcs = attn_mod.AttentionSources(*buf.unbind(0))
out = attn._project_sources(srcs)
stacked_calls = [c[1:3] for c in ops.EXT.calls]
ops.EXT.reset()
logits = uq.sm12x_fp32_lm_head(buf[0], model.lm_head.weight, None)
base, _ = draft.compute_base_logits(buf[0])
y, _ = draft.layers[0].o_proj(buf[0])
print(json.dumps({
    "stacked_calls": stacked_calls,
    "other_calls": [[c[1], c[2], str(c[4])] for c in ops.EXT.calls],
    "levers": list(inkling_pico12.STATE["levers"]),
    "post_loads": model.post_loads,
}))
"""


class EndToEnd(unittest.TestCase):
    def test_through_sitecustomize(self):
        r = run_py(E2E, {"INKLING_PICO12": "all", "INKLING_PICO12_WK": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        res = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(res["levers"], list(ALL))
        self.assertEqual(res["post_loads"], 1)
        self.assertEqual(res["stacked_calls"], [[8, 8600], [16, 2048]])
        self.assertEqual(res["other_calls"], [[8, 256, "torch.float32"], [8, 256, "torch.bfloat16"],
                                              [8, 64, "torch.bfloat16"]])
        lines = [ln for ln in r.stderr.splitlines() if "[inkling_pico12" in ln]
        target = [ln for ln in lines if "active (target)" in ln]
        self.assertEqual(len(target), 1, r.stderr)
        for needle in ("dense: qkvr 2 (+kv 2) wo_ud 2", "head [256x64] for lmhead+drafthead",
                       "unquant.sm12x_fp32_lm_head", "Inkling2Attention._project_sources",
                       "Inkling2ForConditionalGeneration.post_load_weights",
                       "inkling2_mtp.project_through_lm_head", "packed", "MB extra", "max_m=16"):
            self.assertIn(needle, target[0])
        self.assertEqual(len([ln for ln in lines if "active (draft)" in ln]), 1, r.stderr)
        self.assertNotIn("armed", r.stderr)  # the armed line is verbose-only

    def test_noop_and_no_torch_at_install(self):
        code = ("import sys, inkling_pico12\n"
                "print(inkling_pico12.install(), 'torch' in sys.modules, len(sys.meta_path))")
        r = run_py(code, {})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split()[:2], ["()", "False"])
        code = ("import sys, inkling_pico12\n"
                "print(','.join(inkling_pico12.STATE['levers']), 'torch' in sys.modules)")
        r = run_py(code, {"INKLING_PICO12": "dense,lmhead", "INKLING_PICO12_VERBOSE": "1"})
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(r.stdout.split(), ["dense,lmhead", "False"])
        self.assertIn("armed levers=dense,lmhead", r.stderr)

    def test_bad_env_is_fatal_at_startup(self):
        r = run_py("print('served')", {"INKLING_PICO12": "dense,bogus"})
        self.assertEqual(r.returncode, 78)
        self.assertIn("FATAL at startup", r.stderr)
        self.assertNotIn("served", r.stdout)
        r = run_py("print('served')", {"INKLING_PICO12": "dense", "INKLING_PICO12_MAX_M": "32"})
        self.assertEqual(r.returncode, 78)

    def test_missing_attribute_is_fatal_at_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "fake")
            shutil.copytree(FAKE_ROOT, root)
            path = os.path.join(root, "sglang/srt/models/inkling_common/inkling2_attn.py")
            with open(path) as f:
                text = f.read()
            # the text check still passes (the name is in a comment), the attribute is gone
            text = text.replace("def _stacked(sources):", "# def _stacked(\ndef _stacked_gone(sources):")
            with open(path, "w") as f:
                f.write(text)
            code = "from sglang.srt.models import inkling2\nprint('served')"
            r = run_py(code, {"INKLING_PICO12": "dense"}, root=root)
            self.assertEqual(r.returncode, 78, r.stderr)
            self.assertIn("FATAL", r.stderr)
            self.assertIn("_stacked", r.stderr)
            self.assertNotIn("served", r.stdout)

    def test_failed_prepare_is_fatal(self):
        code = ("from inkling_pico12 import ops\n"
                "class Broken:\n"
                "    def encode(self, w): raise RuntimeError('nvcc says no')\n"
                "ops.EXT = Broken(); ops.SET.require_cuda = False\n"
                "from sglang.srt.models import inkling2\n"
                "inkling2.Inkling2ForConditionalGeneration(layers=1).load_weights()\n"
                "print('served')")
        r = run_py(code, {"INKLING_PICO12": "dense", "INKLING_PICO12_WK": "1"})
        self.assertEqual(r.returncode, 78, r.stderr)
        self.assertIn("nvcc says no", r.stderr)
        self.assertNotIn("served", r.stdout)


if __name__ == "__main__":
    unittest.main()
