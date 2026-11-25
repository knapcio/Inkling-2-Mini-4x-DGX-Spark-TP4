"""CPU unit tests of inkling_wide's Python side (torch CPU; no GPU, no nvcc, no sglang).

    python3 -m unittest discover -s inkling_wide/tests -v        (from overlay/)

The CUDA kernel is not run here (inkling_wide/tools/gpu_check_wide.py checks it on the Spark). A stand-in extension
(fake_wide_ext.py) keeps the kernel's call contract and computes reference math; the stand-in sglang package and
pico extension of tests/pico12 carry the stock code of the patched functions and inkling_pico12's packs. Covered:
lever / settings parsing, the source checks, bucket and split-K choice, stacked regions, the dispatch of every
patch point (row range, bias, grad, batch-invariant mode, out shapes), the stacked view layout, the LM head
wrapper, the composition with inkling_pico12 (pico <= 16 rows, wide above, bf16x buffers shared) and the wrong
install order (exit 78), reloads, and the no-op contract.
"""

import os
import sys
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
OVERLAY = os.path.dirname(os.path.dirname(HERE))
PICO_TESTS = os.path.join(OVERLAY, "tests", "pico12")
FAKE_ROOT = os.path.join(PICO_TESTS, "fake_sglang")
for p in (OVERLAY, FAKE_ROOT, PICO_TESTS, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import inkling_pico12  # noqa: E402
import inkling_wide  # noqa: E402
from fake_ext import FakeExt  # noqa: E402  (inkling_pico12's stand-in)
from fake_wide_ext import FakeWideExt  # noqa: E402
from inkling_pico12 import ops as pops  # noqa: E402
from inkling_pico12 import patches as ppatches  # noqa: E402
from inkling_wide import ops, patches  # noqa: E402
from sglang.srt import batch_invariant_ops  # noqa: E402
from sglang.srt.layers.quantization import unquant as uq  # noqa: E402
from sglang.srt.models import inkling2  # noqa: E402
from sglang.srt.models.inkling_common import inkling2_attn as attn_mod  # noqa: E402

STOCK_PROJECT = attn_mod.Inkling2Attention._project_sources
STOCK_HEAD = (uq.sm12x_fp32_lm_head, uq.sm12x_fp32_lm_head_applies)
STOCK_POST_LOAD = inkling2.Inkling2ForConditionalGeneration.post_load_weights
STOCK_FORWARD = inkling2.Inkling2ForConditionalGeneration.forward
WEXT = FakeWideExt()
PEXT = FakeExt()
LOGS = []
EXITS = []


def restore_sglang():
    attn_mod.Inkling2Attention._project_sources = STOCK_PROJECT
    uq.sm12x_fp32_lm_head, uq.sm12x_fp32_lm_head_applies = STOCK_HEAD
    inkling2.Inkling2ForConditionalGeneration.post_load_weights = STOCK_POST_LOAD
    inkling2.Inkling2ForConditionalGeneration.forward = STOCK_FORWARD


def configure(levers=("dense", "lmhead"), min_m=17, max_m=256, fmt="auto", splitk=None, scope="all"):
    restore_sglang()
    ops.SET.scope = scope
    ops.IN_SCOPE[0] = False
    ops.EXT = WEXT
    ops.SET.require_cuda = False
    ops.SET.min_m, ops.SET.max_m, ops.SET.fmt = min_m, max_m, fmt
    ops.SET.max_m_kind = {}
    ops.SET.splitk = dict(splitk or {})
    ops.SET.sms = 48
    ops.SET.pdl = True
    ops.LOG = LOGS.append
    ops._batch_invariant = batch_invariant_ops.is_batch_invariant_mode_enabled
    ops._warned.clear()
    ops.WS.update(ws=None, cnt=None)
    patches.STATE.update(levers=tuple(levers), cfg={"cache_dir": "/nonexistent", "verbose": False},
                         log=LOGS.append, patched=[])
    patches.HEADS.clear()
    patches.EXIT = EXITS.append
    WEXT.reset()
    for key in uq.CALLS:
        uq.CALLS[key] = 0
    uq.STOCK_GEMV[0] = True
    batch_invariant_ops.ENABLED[0] = False
    inkling_pico12.STATE["levers"] = ()


def configure_pico(levers=("dense", "lmhead")):
    pops.EXT = PEXT
    pops.SET.require_cuda = False
    pops.SET.max_m = pops.SET.stacked_max_m = 16
    pops.SET.wk = 1
    pops.LOG = LOGS.append
    pops._batch_invariant = batch_invariant_ops.is_batch_invariant_mode_enabled
    pops._warned.clear()
    ppatches.STATE.update(levers=tuple(levers), cfg={"cache_dir": "/x", "std": "c++20", "verbose": False},
                          log=LOGS.append, patched=[])
    ppatches.HEADS.clear()
    PEXT.reset()
    PEXT.pad = 64
    inkling_pico12.STATE["levers"] = tuple(levers)


def model(layers=1, **kw):
    kw.setdefault("hidden", 64)
    kw.setdefault("heads_dim", 128)
    return inkling2.Inkling2ForConditionalGeneration(layers=layers, **kw)


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


class Settings(unittest.TestCase):
    def test_levers(self):
        self.assertEqual(inkling_wide.parse_levers(""), ())
        self.assertEqual(inkling_wide.parse_levers(" dense , lmhead,dense"), ("dense", "lmhead"))
        self.assertEqual(inkling_wide.parse_levers("all"), ("dense", "lmhead"))
        self.assertEqual(inkling_wide.parse_levers("off"), ())
        for bad in ("dense,draft", "off,dense"):
            with self.assertRaises(RuntimeError):
                inkling_wide.parse_levers(bad)

    def test_config(self):
        cfg = inkling_wide.config_from_env({})
        self.assertEqual((cfg["min_m"], cfg["max_m"], cfg["pdl"], cfg["fmt"], cfg["splitk"]), (17, 256, True, "auto", {}))
        self.assertTrue(cfg["cache_dir"].endswith(os.path.join(".cache", "inkling-wide")))
        cfg = inkling_wide.config_from_env({"INKLING_WIDE_MIN_M": "9", "INKLING_WIDE_MAX_M": "128",
                                            "INKLING_WIDE_PDL": "0", "INKLING_WIDE_FORMAT": "raw",
                                            "INKLING_WIDE_SPLITK": "wo=4, head=1"})
        self.assertEqual((cfg["min_m"], cfg["max_m"], cfg["pdl"], cfg["fmt"], cfg["splitk"]),
                         (9, 128, False, "raw", {"wo": 4, "head": 1}))
        for env in ({"INKLING_WIDE_MAX_M": "257"}, {"INKLING_WIDE_MIN_M": "0"},
                    {"INKLING_WIDE_MIN_M": "64", "INKLING_WIDE_MAX_M": "32"}, {"INKLING_WIDE_FORMAT": "fp8"},
                    {"INKLING_WIDE_SPLITK": "wo"}, {"INKLING_WIDE_SPLITK": "lm=2"}, {"INKLING_WIDE_SPLITK": "wo=0"}):
            with self.assertRaises(ValueError, msg=str(env)):
                inkling_wide.config_from_env(env)

    def test_kind_caps(self):
        cfg = inkling_wide.config_from_env({"INKLING_WIDE_MAX_M_KIND": "wo=64"})
        self.assertEqual(cfg["max_m_kind"], {"qkvr": 96, "wo": 64, "head": 256})
        with self.assertRaises(ValueError):
            inkling_wide.config_from_env({"INKLING_WIDE_MAX_M_KIND": "wo=300"})
        configure()
        ops.SET.max_m_kind = {"wo": 32}
        self.assertTrue(ops.in_range(32, "wo") and not ops.in_range(33, "wo") and ops.in_range(200, "qkvr"))

    def test_source_checks(self):
        import shutil
        import tempfile

        # the stand-in sglang of tests/pico12 plus the strings the scope patch needs
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "sglang")
            shutil.copytree(os.path.join(FAKE_ROOT, "sglang"), root)
            with open(os.path.join(root, "srt", "models", "inkling2.py"), "a") as f:
                f.write("\n# forward_batch: ForwardBatch,\n")
            os.makedirs(os.path.join(root, "srt", "model_executor"), exist_ok=True)
            with open(os.path.join(root, "srt", "model_executor", "forward_mode.py"), "w") as f:
                f.write("def is_decode(self):\ndef is_target_verify(self):\n")
            inkling_wide.check_sources(("dense", "lmhead"), root)
            with self.assertRaises(RuntimeError):
                inkling_wide.check_sources(("dense", "lmhead"), os.path.join(FAKE_ROOT, "sglang"))
        with self.assertRaises(RuntimeError):
            inkling_wide.check_sources(("dense",), os.path.join(HERE))

    def test_noop_without_switch(self):
        saved = dict(inkling_wide.STATE)
        try:
            inkling_wide.STATE.update(installed=False, levers=(), cfg=None)
            self.assertEqual(inkling_wide.install(env={}), ())
        finally:
            inkling_wide.STATE.update(saved)


class Shapes(unittest.TestCase):
    def setUp(self):
        configure()

    def test_bucket(self):
        for m, b in ((1, 1), (16, 1), (17, 2), (32, 2), (33, 3), (48, 3), (49, 4), (64, 4), (65, 6), (96, 6),
                     (97, 8), (128, 8), (129, 12), (192, 12), (193, 16), (256, 16)):
            self.assertEqual(ops.bucket(m), b, m)
            self.assertEqual(WEXT.bucket(m), b, m)
        with self.assertRaises(ValueError):
            ops.bucket(257)

    def test_split(self):
        self.assertEqual(ops.choose_split(8600, 2304, 64, "qkvr"), 1)
        self.assertEqual(ops.choose_split(201024, 2304, 256, "head"), 1)
        self.assertEqual([ops.choose_split(2304, 3072, m, "wo") for m in (17, 32, 48, 64, 128, 256)], [3, 3, 3, 3, 2, 2])
        self.assertEqual(ops.choose_split(8600, 2304, 256, "qkvr"), 1)
        ops.SET.splitk = {"wo": 3}
        self.assertEqual(ops.choose_split(2304, 3072, 64, "wo"), 3)
        ops.SET.splitk = {"wo": 99}
        self.assertEqual(ops.choose_split(2304, 3072, 64, "wo"), 3072 // 64)

    def test_regions(self):
        self.assertEqual(ops.regions_of([3072, 1024, 1024, 384, 3072, 24]), (3072, 4096, 5120))
        self.assertIsNone(ops.regions_of([3000, 1024, 1024]))
        self.assertIsNone(ops.regions_of([3072, 1000, 1024]))
        self.assertIsNone(ops.regions_of([3072, 1024]))

    def test_shape_reason(self):
        self.assertIsNone(ops.shape_reason(8600, 2304))
        self.assertIsNotNone(ops.shape_reason(8601, 2304))
        self.assertIsNotNone(ops.shape_reason(8600, 2336))


class Dense(unittest.TestCase):
    def setUp(self):
        configure()
        patches.patch_attn(attn_mod)
        self.model = model()
        patches.prepare_target(self.model)
        self.attn = self.model.llm.layers[0].attn

    def tearDown(self):
        restore_sglang()

    def test_registered(self):
        self.assertEqual(type(self.attn.qkvr.quant_method).__name__, "WideUnquantizedLinearMethod")
        self.assertEqual(self.attn.qkvr.__dict__["_wide"].fmt, ops.RAW)
        self.assertIn("_wide", self.attn.wo_ud.__dict__)
        self.assertEqual(self.attn.__dict__["_wide_qkvr"][1], (3072, 4096, 5120))
        self.assertTrue(any("active (target)" in m for m in LOGS))

    def test_rows_range(self):
        w = self.attn.qkvr.weight
        for m, wide in ((8, False), (16, False), (17, True), (64, True), (256, True), (257, False)):
            WEXT.reset()
            uq.CALLS["apply"] = 0
            x = bf16(m, 64, seed=m)
            y, _ = self.attn.qkvr(x)
            self.assertEqual(len(WEXT.calls), int(wide), m)
            self.assertEqual(uq.CALLS["apply"], int(not wide), m)
            self.assertTrue(bits_equal(y, ref(x, w)), m)

    def test_3d_input_and_wo(self):
        x = bf16(3, 8, 128, seed=1)
        y, _ = self.attn.wo_ud(x)
        self.assertEqual(tuple(y.shape), (3, 8, 64))
        self.assertTrue(bits_equal(y, ref(x, self.attn.wo_ud.weight)))
        self.assertEqual(WEXT.calls[-1][:3], (24, 64, 128))

    def test_falls_back(self):
        x = bf16(32, 64)
        batch_invariant_ops.ENABLED[0] = True
        self.attn.qkvr(x)
        batch_invariant_ops.ENABLED[0] = False
        self.attn.qkvr(x)
        self.assertEqual(len(WEXT.calls), 1)
        self.assertEqual(uq.CALLS["apply"], 1)
        self.attn.qkvr.quant_method.apply(self.attn.qkvr, x, bias=torch.zeros(8600, dtype=torch.bfloat16))
        self.assertEqual(uq.CALLS["apply"], 2)

    def test_apply_into(self):
        x = bf16(40, 128, seed=2)
        out = torch.empty((40, 64), dtype=torch.bfloat16)
        self.attn.wo_ud(x, output_tensor=out)
        self.assertEqual(len(WEXT.calls), 1)
        self.assertTrue(bits_equal(out, ref(x, self.attn.wo_ud.weight)))
        bad = torch.empty((40, 128), dtype=torch.bfloat16)[:, :64]  # rows not contiguous
        self.attn.wo_ud(x, output_tensor=bad)
        self.assertEqual(uq.CALLS["apply_into"], 1)

    def test_stacked_layout_and_values(self):
        m = 20
        q, k, v = bf16(m, 64, seed=3), bf16(m, 64, seed=4), bf16(m, 64, seed=5)
        src = attn_mod.AttentionSources(q, k, v)
        got = self.attn._project_sources(src)
        self.assertEqual(len(WEXT.calls), 1)
        self.assertTrue(WEXT.calls[0][-1])  # stacked
        stock = STOCK_PROJECT(self.attn, src)
        for a, b in zip(got, stock):
            self.assertEqual((a.shape, a.stride(), a.storage_offset()), (b.shape, b.stride(), b.storage_offset()))
            self.assertTrue(bits_equal(a, b))

    def test_stacked_falls_back(self):
        for m in (8, 300):
            WEXT.reset()
            q, k, v = bf16(m, 64, seed=3), bf16(m, 64, seed=4), bf16(m, 64, seed=5)
            self.attn._project_sources(attn_mod.AttentionSources(q, k, v))
            self.assertFalse(any(c[-1] for c in WEXT.calls), m)
        WEXT.reset()  # shared sources go through the plain linear
        q = bf16(24, 64, seed=3)
        self.attn._project_sources(attn_mod.AttentionSources(q, q, q))
        self.assertEqual([c[-1] for c in WEXT.calls], [False])

    def test_reload_and_replaced_storage(self):
        w = self.attn.qkvr.weight
        w.data.copy_(bf16(*w.shape, seed=7))  # in place: still current
        x = bf16(32, 64, seed=8)
        y, _ = self.attn.qkvr(x)
        self.assertTrue(bits_equal(y, ref(x, w)))
        self.assertEqual(len(WEXT.calls), 1)
        self.attn.qkvr.weight = torch.nn.Parameter(bf16(*w.shape, seed=9), requires_grad=False)
        y, _ = self.attn.qkvr(x)
        self.assertEqual(len(WEXT.calls), 1)
        self.assertTrue(any("storage changed" in m for m in LOGS))
        patches.prepare_target(self.model)  # re-registration picks the new storage up
        self.attn.qkvr(x)
        self.assertEqual(len(WEXT.calls), 2)

    def test_split_workspace(self):
        configure(splitk={"wo": 1, "qkvr": 2})
        m2 = model(hidden=128)
        patches.prepare_target(m2)
        self.assertGreaterEqual(ops.WS["ws"].numel(), 2 * 256 * 8600)
        with self.assertRaises(ops.WideFatal):  # a reload needing more cannot replace graph-referenced buffers
            ops.ensure_workspace(list(patches.HEADS.values()), 17, 256, "cpu", floats=10**9)
        x = bf16(48, 128, seed=10)
        y, _ = m2.llm.layers[0].attn.qkvr(x)
        self.assertEqual(WEXT.calls[-1][5], 2)
        self.assertTrue(bits_equal(y, ref(x, m2.llm.layers[0].attn.qkvr.weight)))


class Scope(unittest.TestCase):
    """INKLING_WIDE_SCOPE=verify: wide takes calls only inside a target decode / target-verify forward."""

    class Mode:
        def __init__(self, name):
            self.name = name

        def is_decode(self):
            return self.name == "decode"

        def is_target_verify(self):
            return self.name == "verify"

    def setUp(self):
        configure(scope="verify")
        self.model = model()
        patches.prepare_target(self.model)
        self.attn = self.model.llm.layers[0].attn

    def tearDown(self):
        restore_sglang()

    def run_in(self, mode, m=32):
        used = []

        def inner(model_self, input_ids, positions, forward_batch, *a, **kw):
            WEXT.reset()
            self.attn.qkvr(bf16(m, 64, seed=1))
            used.append(len(WEXT.calls))
            return "out"

        fwd = patches.make_forward(inner)
        fb = types.SimpleNamespace(forward_mode=self.Mode(mode)) if mode else types.SimpleNamespace()
        self.assertEqual(fwd(self.model, None, None, fb), "out")
        self.assertFalse(ops.IN_SCOPE[0])  # restored
        return used[0]

    def test_scope(self):
        self.assertEqual(self.run_in("verify"), 1)
        self.assertEqual(self.run_in("decode"), 1)
        self.assertEqual(self.run_in("extend"), 0)
        self.assertEqual(self.run_in(None), 0)
        WEXT.reset()
        self.attn.qkvr(bf16(32, 64))  # outside any forward
        self.assertEqual(len(WEXT.calls), 0)

    def test_restored_on_error(self):
        def inner(*a, **kw):
            raise RuntimeError("boom")

        fwd = patches.make_forward(inner)
        with self.assertRaises(RuntimeError):
            fwd(self.model, None, None, types.SimpleNamespace(forward_mode=self.Mode("verify")))
        self.assertFalse(ops.IN_SCOPE[0])

    def test_patch_model_wraps_forward(self):
        patches.patch_model(inkling2)
        self.assertTrue(getattr(inkling2.Inkling2ForConditionalGeneration.forward, "_wide", False))

    def test_config(self):
        self.assertEqual(inkling_wide.config_from_env({})["scope"], "verify")
        self.assertEqual(inkling_wide.config_from_env({"INKLING_WIDE_SCOPE": "all"})["scope"], "all")
        with self.assertRaises(ValueError):
            inkling_wide.config_from_env({"INKLING_WIDE_SCOPE": "prefill"})


class Head(unittest.TestCase):
    def setUp(self):
        configure()
        patches.patch_unquant(uq)
        self.model = model()
        patches.prepare_target(self.model)
        self.w = self.model.llm.lm_head.weight

    def test_head(self):
        for m, wide in ((4, False), (20, True), (256, True)):
            WEXT.reset()
            uq.CALLS["lm_head"] = 0
            x = bf16(m, 64, seed=m)
            self.assertTrue(uq.sm12x_fp32_lm_head_applies(x, self.w, None))
            y = uq.sm12x_fp32_lm_head(x, self.w, None)
            self.assertEqual(y.dtype, torch.float32)
            self.assertTrue(bits_equal(y, ref(x, self.w, torch.float32)))
            self.assertEqual(len(WEXT.calls), int(wide))
            self.assertEqual(uq.CALLS["lm_head"], int(not wide))
        self.assertFalse(uq.sm12x_fp32_lm_head_applies(bf16(300, 64), self.w, None))

    def test_head_out(self):
        x = bf16(33, 64, seed=1)
        out = torch.empty((33, 256), dtype=torch.float32)
        y = uq.sm12x_fp32_lm_head(x, self.w, out)
        self.assertEqual(y.data_ptr(), out.data_ptr())
        self.assertTrue(bits_equal(out, ref(x, self.w, torch.float32)))
        wrong = torch.empty((33, 255), dtype=torch.float32)
        self.assertFalse(uq.sm12x_fp32_lm_head_applies(x, self.w, wrong))


class WithPico(unittest.TestCase):
    """inkling_pico12 installed first (as sitecustomize does): pico <= 16 rows, wide 17..256 on pico's buffers."""

    def setUp(self):
        configure()
        configure_pico()
        ppatches.patch_unquant(uq)
        patches.patch_unquant(uq)
        ppatches.patch_attn(attn_mod)
        patches.patch_attn(attn_mod)
        ppatches.patch_model(inkling2)
        patches.patch_model(inkling2)
        self.model = model()
        self.model.post_load_weights()
        self.attn = self.model.llm.layers[0].attn

    def tearDown(self):
        restore_sglang()

    def test_formats_and_split_of_work(self):
        e = self.attn.qkvr.__dict__["_wide"]
        self.assertEqual(e.fmt, ops.BF16X)
        self.assertIs(e.pack, self.attn.qkvr.__dict__["_pico12"])
        self.assertEqual(type(self.attn.qkvr.quant_method).__name__, "WidePico12UnquantizedLinearMethod")
        for m, who in ((8, "pico"), (16, "pico"), (17, "wide"), (128, "wide"), (257, "stock")):
            PEXT.reset(), WEXT.reset()
            uq.CALLS["apply"] = 0
            x = bf16(m, 64, seed=m)
            y, _ = self.attn.qkvr(x)
            self.assertTrue(bits_equal(y, ref(x, self.attn.qkvr.weight)), m)
            self.assertEqual((len(PEXT.calls) > 0, len(WEXT.calls) > 0, uq.CALLS["apply"] > 0),
                             (who == "pico", who == "wide", who == "stock"), m)
        self.attn.qkvr(bf16(40, 64))
        self.assertEqual(WEXT.calls[-1][3], 2)

    def test_head_bf16x(self):
        w = self.model.llm.lm_head.weight
        self.assertEqual(patches.HEADS[w.data_ptr()].fmt, ops.BF16X)
        x = bf16(40, 64, seed=3)
        y = uq.sm12x_fp32_lm_head(x, w, None)
        self.assertEqual(WEXT.calls[-1][3], 2)
        self.assertTrue(bits_equal(y, ref(x, w, torch.float32)))
        PEXT.reset()
        uq.sm12x_fp32_lm_head(bf16(8, 64), w, None)
        self.assertEqual(len(PEXT.calls), 1)

    def test_stacked_order(self):
        for m, who in ((4, "pico"), (17, "wide")):
            PEXT.reset(), WEXT.reset()
            q, k, v = bf16(m, 64, seed=3), bf16(m, 64, seed=4), bf16(m, 64, seed=5)
            src = attn_mod.AttentionSources(q, k, v)
            got = self.attn._project_sources(src)
            for a, b in zip(got, STOCK_PROJECT(self.attn, src)):
                self.assertTrue(bits_equal(a, b), m)
            self.assertEqual((len(PEXT.calls) > 0, len(WEXT.calls) > 0), (who == "pico", who == "wide"), m)
        self.assertTrue(getattr(attn_mod.Inkling2Attention._project_sources, "inkling_pico12", False))

    def test_raw_format_forced(self):
        configure(fmt="raw")
        configure_pico()
        m2 = model()
        ppatches.prepare_target(m2)
        patches.prepare_target(m2)
        self.assertEqual(m2.llm.layers[0].attn.qkvr.__dict__["_wide"].fmt, ops.RAW)


class WrongOrder(unittest.TestCase):
    def tearDown(self):
        restore_sglang()

    def test_wide_before_pico_exits(self):
        configure()
        configure_pico()
        EXITS.clear()
        patches.patch_model(inkling2)  # wide's wrapper inside ...
        ppatches.patch_model(inkling2)  # ... pico's: wide registers before pico coded anything
        ppatches.EXIT = lambda code: EXITS.append(("pico", code))
        with self.assertRaises(BaseException):
            model().post_load_weights()
        self.assertIn(78, EXITS)
        self.assertTrue(any("install inkling_wide after inkling_pico12" in m for m in LOGS))


if __name__ == "__main__":
    unittest.main()
