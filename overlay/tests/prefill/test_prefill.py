"""CPU tests of inkling_prefill (torch CPU, no Triton, no GPU, no sglang).

  python -m unittest discover -s tests/prefill -v        (from overlay/)

Covered here: env parsing and the no-op contract, the startup source checks, the
sheared-bias layout (element-wise description vs a literal transcription of the
CuTe ShearingBias kernel), the NVFP4 scale word mapping, the quant_once gather
index math (torch emulation vs the stock routed layout), the qkvr_split wrapper
(values, fallbacks, composition with other _project_sources wrappers), the
partial_bf16 and bias_layout routing logic, and the post-import patch plumbing on
fake sglang modules. The Triton kernels and all bitwise claims against the real
kernels are covered by tools/gpu_check_prefill.py on the Spark.
"""

import math
import os
import random
import subprocess
import sys
import tempfile
import textwrap
import types
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/
sys.path.insert(0, ROOT)

import inkling_prefill as P  # noqa: E402
from inkling_prefill import _layout as L  # noqa: E402
from inkling_prefill import bias_layout, partial_bf16, qkvr_split  # noqa: E402

SPLITS = [3072, 1024, 1024, 384, 3072, 24]


def fake_tree(levers=P.LEVERS, drop=None):
    """A temp sglang tree whose files contain exactly the needles."""
    root = tempfile.mkdtemp(prefix="ip_fake_sglang_")
    files = {}
    for lever in levers:
        for rel, needles in P._SOURCE_CHECKS[lever]:
            files.setdefault(rel, []).extend(needles)
    for rel, needles in files.items():
        path = os.path.join(root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write("\n".join(n for n in needles if n != drop))
    return root


class EnvTests(unittest.TestCase):
    def test_parse(self):
        self.assertEqual(P.parse_levers(""), ())
        self.assertEqual(P.parse_levers("0"), ())
        self.assertEqual(P.parse_levers("all"), P.LEVERS)
        self.assertEqual(P.parse_levers(" quant_once , qkvr_split,quant_once"), ("quant_once", "qkvr_split"))
        with self.assertRaises(RuntimeError):
            P.parse_levers("qkvr_split,bogus")

    def test_config(self):
        old = dict(os.environ)
        try:
            for bad in ("8", "64", "127"):
                os.environ["INKLING_PREFILL_QKVR_MIN_M"] = bad
                with self.assertRaises(RuntimeError):
                    P.config_from_env()
            os.environ["INKLING_PREFILL_QKVR_MIN_M"] = "512"
            os.environ["INKLING_PREFILL_QKVR_OUT"] = "strided"
            os.environ.pop("INKLING_PREFILL_QKVR_TP_MIN_M", None)
            cfg = P.config_from_env()
            self.assertEqual(cfg["qkvr_split"], {"min_m": 512, "out": "strided"})  # TP1 config unchanged
            self.assertEqual(P.describe(("qkvr_split",), cfg), "qkvr_split(min_m=512,out=strided)")
            os.environ["INKLING_PREFILL_QKVR_TP_MIN_M"] = "416"
            cfg = P.config_from_env()
            self.assertEqual(cfg["qkvr_split"], {"min_m": 512, "out": "strided", "tp_min_m": 416})
            self.assertEqual(P.describe(("qkvr_split",), cfg), "qkvr_split(min_m=512,out=strided,tp_min_m=416)")
            for bad in ("64", "x"):
                os.environ["INKLING_PREFILL_QKVR_TP_MIN_M"] = bad
                with self.assertRaises(RuntimeError):
                    P.config_from_env()
            os.environ.pop("INKLING_PREFILL_QKVR_TP_MIN_M")
            for bad in ("separate", "bad"):
                os.environ["INKLING_PREFILL_QKVR_OUT"] = bad
                with self.assertRaises(RuntimeError):
                    P.config_from_env()
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_source_checks(self):
        P.check_sources(P.LEVERS, fake_tree())
        needle = P._SOURCE_CHECKS["bias_layout"][0][1][1]
        with self.assertRaisesRegex(RuntimeError, "bias_layout"):
            P.check_sources(["bias_layout"], fake_tree(drop=needle))
        with self.assertRaisesRegex(RuntimeError, "missing"):
            P.check_sources(["quant_once"], tempfile.mkdtemp())

    def test_real_tree_if_present(self):
        root = os.environ.get("INKLING_PREFILL_TEST_SGLANG", "")
        if not root or not os.path.isdir(root):
            self.skipTest("set INKLING_PREFILL_TEST_SGLANG to an sglang package dir")
        P.check_sources(P.LEVERS, root)


class ShearLayoutTests(unittest.TestCase):
    def _cases(self):
        rnd = random.Random(1)
        cases = [(1, 0), (5, 0), (63, 0), (64, 0), (65, 0), (1024, 0), (1500, 0), (8192, 0),
                 (300, 700), (1024, 24576), (8192, 24576), (200, 1023), (77, 5000), (1, 130000),
                 (129, 1), (2, 1022)]
        cases += [(rnd.randint(1, 3000), rnd.randint(0, 40000)) for _ in range(12)]
        for sq, prefix in cases:
            ms = {0, 1, 2, 3, 62, 63, 64, 65, 127, 128, 129, 1022, 1023, 1024, 1025, sq - 1, sq - 2, sq // 2}
            ms |= {rnd.randrange(sq) for _ in range(4)}
            for m in sorted(x for x in ms if 0 <= x < sq):
                yield sq, sq + prefix, m

    def test_simple_equals_stock_kernel(self):
        pre = [float(i + 1) for i in range(1024)]
        n = 0
        for sq, sk, m in self._cases():
            for local in (True, False):
                a = L.stock_shear_row(pre, m, sq, sk, local, 1023)
                b = L.simple_shear_row(pre, m, sq, sk, local, 1023)
                self.assertFalse(any(isinstance(x, float) and math.isnan(x) for x in a), (sq, sk, m, local))
                self.assertEqual(a, b, (sq, sk, m, local))
                n += 1
        self.assertGreater(n, 400)

    def test_reference_matches_rows(self):
        torch.manual_seed(0)
        cu = torch.tensor([0, 70, 70, 200], dtype=torch.int32)  # a zero-length slot in between
        seqk = torch.tensor([70 + 900, 5, 130 + 3000], dtype=torch.int32)
        r = torch.randn(200, 2, 16)
        proj = torch.randn(16, 1024)
        for local in (True, False):
            ref = bias_layout.sheared_reference(r, proj, cu, seqk, local, 1023)
            self.assertEqual(tuple(ref.shape), (256, 2, 1280))
            pre = torch.einsum("thd,de->the", r, proj)
            for tok, (b, m) in ((0, (0, 0)), (69, (0, 69)), (70, (2, 0)), (199, (2, 129))):
                sq = int(cu[b + 1] - cu[b])
                row = L.simple_shear_row(pre[tok, 1].tolist(), m, sq, int(seqk[b]), local, 1023)
                got = ref[tok, 1].tolist()
                for x, y in zip(got, row):
                    self.assertTrue(x == y or abs(x - y) < 1e-6, (tok, local))
            self.assertTrue(torch.isnan(ref[200:]).all())


class ScaleLayoutTests(unittest.TestCase):
    def test_word_vs_byte(self):
        sk = L.scale_k_of(2304)
        self.assertEqual(sk, 144)
        for local in list(range(300)) + [1000, 4095, 65535]:
            for c4 in range(sk // 4):
                for j in range(4):
                    self.assertEqual(L.scale_word(local, c4, sk) * 4 + j, L.scale_offset(local, 4 * c4 + j, sk))

    def test_gather_math_vs_stock_layout(self):
        """Emulate the stock per-routed-row writes and the quant_once gather with
        the same per-token scale bytes; both layouts must agree byte for byte."""
        rnd = random.Random(3)
        width, top_k, experts, tokens = 256, 4, 8, 150
        sk = L.scale_k_of(width)
        tok_scale = [[rnd.randrange(1, 256) for _ in range(sk)] for _ in range(tokens)]
        picks = [rnd.sample(range(experts), top_k) for _ in range(tokens)]
        flat = [e for p in picks for e in p]
        order = sorted(range(len(flat)), key=lambda i: (flat[i], i))  # stable sort by expert
        dst2src = order
        counts = [flat.count(e) for e in range(experts)]
        offsets = [0]
        for c in counts:
            offsets.append(offsets[-1] + c)
        rows = len(flat)
        nbytes = L.scale_rows(rows, experts) * sk
        stock = [None] * nbytes
        for r in range(rows):
            e = max(i for i in range(experts) if offsets[i] <= r and counts[i] > 0 and r < offsets[i + 1])
            t = dst2src[r] // top_k
            for col in range(sk):
                stock[L.scale_base(offsets, e, sk) + L.scale_offset(r - offsets[e], col, sk)] = tok_scale[t][col]
        for e in range(experts):
            used = counts[e] % 128
            if used:
                for local in range(counts[e], counts[e] - used + 128):
                    for col in range(sk):
                        stock[L.scale_base(offsets, e, sk) + L.scale_offset(local, col, sk)] = 0
        # token layout (one expert, base 0), then the kernel's word moves
        tok_bytes = {}
        for t in range(tokens):
            for col in range(sk):
                tok_bytes[L.scale_offset(t, col, sk)] = tok_scale[t][col]
        mine = [None] * nbytes
        for r in range(rows):
            starts = offsets[1:experts]
            e = sum(1 for s in starts if s <= r)  # _gather_routed's expert lookup
            oe = offsets[e]
            base_words = (oe - offsets[0] + 127 * e) // 128 * 128 * (sk // 4)
            t = dst2src[r] // top_k
            for c4 in range(sk // 4):
                dst = base_words + L.scale_word(r - oe, c4, sk)
                src = L.scale_word(t, c4, sk)
                for j in range(4):
                    mine[dst * 4 + j] = tok_bytes[src * 4 + j]
        for e in range(experts):
            count = offsets[e + 1] - offsets[e]
            used = count % 128
            if used:
                base_words = (offsets[e] - offsets[0] + 127 * e) // 128 * 128 * (sk // 4)
                for row in range(used, 128):
                    local = count // 128 * 128 + row
                    for c4 in range(sk // 4):
                        for j in range(4):
                            mine[(base_words + L.scale_word(local, c4, sk)) * 4 + j] = 0
        self.assertEqual(stock, mine)


class _Method:
    pass


_Method.__name__ = "UnquantizedLinearMethod"


class _Linear:
    def __init__(self, weight):
        self.weight = weight
        self.bias = None
        self.quant_method = _Method()

    def __call__(self, x):
        return torch.nn.functional.linear(x, self.weight), None


class _Attn:
    def __init__(self, weight):
        self.split_sizes = list(SPLITS)
        self.qkvr = _Linear(weight)


def stock_project_sources(self, sources):
    """Replica of sglang Inkling2Attention._project_sources (three-source case)."""
    if len({id(sources.q), id(sources.k), id(sources.v)}) == 3:
        stacked = torch.cat([sources.q, sources.k, sources.v])
        q_proj, k_proj, v_proj = (
            p.split(self.split_sizes, dim=-1)[:6] for p in self.qkvr(stacked)[0].chunk(3)
        )
        q, _, _, r, gate, sinks = q_proj
        return q, k_proj[1], v_proj[2], r, gate, sinks
    q_proj = self.qkvr(sources.q)[0].split(self.split_sizes, dim=-1)[:6]
    q, k, v, r, gate, sinks = q_proj
    return q, k, v, r, gate, sinks


class QkvrSplitTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.k = 64
        n = sum(SPLITS)
        self.w = (torch.randn(n, self.k) * 0.05).to(torch.bfloat16)
        self.attn = _Attn(self.w)
        self.Src = types.SimpleNamespace

    def _sources(self, m):
        buf = torch.randn(3, m, self.k).to(torch.bfloat16)
        return self.Src(q=buf[0], k=buf[1], v=buf[2])

    def test_plan(self):
        self.assertEqual(qkvr_split.column_plan(SPLITS, 8600), ((0, 3072), (3072, 4096), (4096, 5120), (5120, 8600)))
        self.assertIsNone(qkvr_split.column_plan(SPLITS, 8608))

    def test_values_and_fallbacks(self):
        calls = []

        def prev(self, sources):
            calls.append(sources.q.shape[0])
            return stock_project_sources(self, sources)

        for _ in range(1):
            fn = qkvr_split.make_project_sources(prev, None, {"min_m": 32, "out": "strided", "allow_cpu": True})
            src = self._sources(40)
            got = fn(self.attn, src)
            ref = stock_project_sources(self.attn, src)
            for g, r in zip(got, ref):
                self.assertEqual(g.shape, r.shape)
                self.assertEqual(g.stride(), r.stride())  # stock row stride 8600, unit columns
                torch.testing.assert_close(g.float(), r.float(), rtol=2e-2, atol=2e-2)
            # attention_prologue's contract: q, k, v, rel share one row stride
            q, k, v, r = got[0], got[1], got[2], got[3]
            stride = q.stride(0)
            self.assertTrue(all(t.stride() == (stride, 1) for t in (q, k, v, r)))
            self.assertEqual(len({t.untyped_storage().data_ptr() for t in got}), 1)
            self.assertEqual(calls, [])
            fn(self.attn, self._sources(31))  # below min_m
            shared = self._sources(40)
            fn(self.attn, self.Src(q=shared.q, k=shared.q, v=shared.q))  # shared source
            self.assertEqual(calls, [31, 40])
            calls.clear()
        fn = qkvr_split.make_project_sources(prev, None, {"min_m": 32, "out": "strided"})
        fn(self.attn, self._sources(40))  # CPU tensors without allow_cpu -> stock
        self.assertEqual(calls, [40])

    def test_composition(self):
        seen = []

        def stock(self, sources):
            seen.append("stock")
            return stock_project_sources(self, sources)

        def small_wrapper(inner):
            def f(self, sources):
                if sources.q.shape[0] <= 16:
                    seen.append("small")
                    return stock_project_sources(self, sources)
                return inner(self, sources)

            f.inkling_cfg = {"max_m": 16}
            return f

        cfg = {"min_m": 32, "out": "strided", "allow_cpu": True}
        a = qkvr_split.make_project_sources(small_wrapper(stock), None, cfg)  # ours outside
        b = small_wrapper(qkvr_split.make_project_sources(stock, None, cfg))  # ours inside
        self.assertEqual(a.inkling_cfg, {"max_m": 16})  # marker visible through ours
        for fn in (a, b):
            seen.clear()
            fn(self.attn, self._sources(8))
            fn(self.attn, self._sources(20))
            fn(self.attn, self._sources(64))
            self.assertEqual(seen, ["small", "stock"])  # 64 rows: ours (no stock call)


# attention TP4 per-rank qkvr (Inkling2Attention.__init__, inkling2_attn.py:139-149): 24 q / 8 kv heads / 4 ranks
TP4_SPLITS = [768, 256, 256, 96, 768, 6, 2]  # q, k, v, r (6 heads x 16), gate, sinks, qkvr_pad_size


class QkvrSplitTpTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1)
        self.k = 64
        self.w32 = torch.randn(sum(TP4_SPLITS), self.k) * 0.05
        self.w = self.w32.to(torch.bfloat16)
        self.Src = types.SimpleNamespace

    def _attn(self, tp, weight=None):
        a = _Attn(self.w if weight is None else weight)
        a.split_sizes = list(TP4_SPLITS)
        if tp is not None:
            a.tp_size = tp
        return a

    def _sources(self, m, dtype=torch.bfloat16):
        buf = torch.randn(3, m, self.k).to(dtype)
        return self.Src(q=buf[0], k=buf[1], v=buf[2])

    def test_tp4_plan_keeps_pad_out_of_the_views(self):
        plan = qkvr_split.column_plan(TP4_SPLITS, 2152)
        self.assertEqual(plan, ((0, 768), (768, 1024), (1024, 1280), (1280, 2152)))
        stock_sizes = [768, 256, 256, 96, 768, 6]
        self.assertEqual(qkvr_split.column_plan(stock_sizes, 2150), ((0, 768), (768, 1024), (1024, 1280), (1280, 2150)))

    def test_tp4_values_equal_stock(self):
        """Every returned view at the TP4 rank shape: same columns, same source rows, same strides as stock
        (FP32 on the CPU, so a wrong column or source is a large error, not rounding)."""
        attn = self._attn(1, self.w32)
        plan = qkvr_split.column_plan(TP4_SPLITS, 2152)
        for m in (32, 128, 333):
            src = self._sources(m, torch.float32)
            got = qkvr_split.split_project(src.q, src.k, src.v, self.w32, plan, TP4_SPLITS[3:])
            ref = stock_project_sources(attn, src)
            for name, g, r in zip(("q", "k", "v", "r", "gate", "sinks"), got, ref):
                self.assertEqual((name, g.shape, g.stride()), (name, r.shape, r.stride()))
                torch.testing.assert_close(g, r, rtol=1e-5, atol=1e-5, msg=name)
            self.assertEqual([t.shape[1] for t in got], [768, 256, 256, 96, 768, 6])

    def test_tp_gate(self):
        calls = []

        def prev(self, sources):
            calls.append(sources.q.shape[0])
            return stock_project_sources(self, sources)

        base = {"min_m": 128, "out": "strided", "allow_cpu": True}
        off = qkvr_split.make_project_sources(prev, None, dict(base))
        floor = qkvr_split.make_project_sources(prev, None, dict(base, tp_min_m=416))
        # TP1 (no attribute, or tp_size 1): unchanged, the split takes M >= 128 with or without a TP floor
        for fn in (off, floor):
            for attn in (self._attn(None), self._attn(1)):
                calls.clear()
                fn(attn, self._sources(128))
                self.assertEqual(calls, [])
        # TP4 without a floor: every call goes to prev
        calls.clear()
        for m in (128, 416, 1024):
            off(self._attn(4), self._sources(m))
        self.assertEqual(calls, [128, 416, 1024])
        # TP4 with a floor: prev below it, the split from it on
        calls.clear()
        for m in (128, 384, 416, 1024):
            got = floor(self._attn(4), self._sources(m))
            self.assertEqual(got[1].shape, (m, 256))
        self.assertEqual(calls, [128, 384])


class PartialBf16Tests(unittest.TestCase):
    def test_routing(self):
        calls = []

        def stock(x, offsets, src2dst, top_k, *, nvfp4_scales=None):
            calls.append("stock")
            return x.float()

        orig = partial_bf16.routed_partial_bf16
        partial_bf16.routed_partial_bf16 = lambda *a, **k: calls.append("bf16") or a[0]
        try:
            rp = partial_bf16.make_routed_partial(stock)
            mod = types.SimpleNamespace(routed_partial=rp)

            def fwd(self, x, weights, ids, combine):
                return mod.routed_partial(x, None, None, 8).to(x.dtype)

            wrapped = partial_bf16.make_forward_sm120(fwd)
            x = torch.zeros(2, 4, dtype=torch.bfloat16)
            x.is_cuda  # CPU: the bf16 path requires CUDA tensors
            wrapped(None, x, None, None, True)
            self.assertEqual(calls, ["stock"])
            calls.clear()

            class FakeCuda(torch.Tensor):
                @property
                def is_cuda(self):
                    return True

            xc = x.as_subclass(FakeCuda)
            wrapped(None, xc, None, None, True)
            mod.routed_partial(xc, None, None, 8)  # outside _forward_sm120
            self.assertEqual(calls, ["bf16", "stock"])
            calls.clear()
            wrapped(None, xc.float().as_subclass(FakeCuda), None, None, True)  # FP32 x
            self.assertEqual(calls, ["stock"])
        finally:
            partial_bf16.routed_partial_bf16 = orig


class BiasLayoutLogicTests(unittest.TestCase):
    def _mode(self, extend=True, decode=False, verify=False, draft=False):
        return types.SimpleNamespace(
            is_extend=lambda: extend, is_decode=lambda: decode,
            is_target_verify=lambda: verify, is_draft_extend=lambda: draft,
        )

    def _setup(self, **kw):
        class FakeCuda(torch.Tensor):
            @property
            def is_cuda(self):
                return True

        cu = torch.tensor([0, 10], dtype=torch.int32).as_subclass(FakeCuda)
        sk = torch.tensor([10], dtype=torch.int32).as_subclass(FakeCuda)
        meta = types.SimpleNamespace(cu_seqlens_q=cu, cache_seqlens_int32=sk)
        backend = types.SimpleNamespace(forward_metadata=meta)
        fb = types.SimpleNamespace(forward_mode=self._mode(**kw), contains_mm_inputs=lambda: False)
        owner = types.SimpleNamespace(is_local=True, rel_extent=1024, rel_logits_proj=object())
        attn = types.SimpleNamespace(pack_gqa=False, sliding_window_size=1023)
        model = types.SimpleNamespace(kv_cache_dtype="auto")
        return owner, fb, attn, (lambda: backend), (lambda: model), meta

    def test_eligibility(self):
        owner, fb, attn, gab, gm, meta = self._setup()
        v = bias_layout._visit_for(owner, fb, attn, gab, gm)
        self.assertIsNotNone(v)
        self.assertEqual((v.local, v.window_left), (True, 1023))
        for kw in ({"verify": True}, {"decode": True}, {"draft": True}, {"extend": False}):
            o, f, a, g1, g2, _ = self._setup(**kw)
            self.assertIsNone(bias_layout._visit_for(o, f, a, g1, g2), kw)
        o, f, a, g1, g2, m = self._setup()
        m.swa_spec_metadata = object()
        self.assertIsNone(bias_layout._visit_for(o, f, a, g1, g2))
        o, f, a, g1, g2, _ = self._setup()
        self.assertIsNone(bias_layout._visit_for(o, f, a, g1, lambda: types.SimpleNamespace(kv_cache_dtype="mxfp8")))
        o, f, a, g1, g2, _ = self._setup()
        a.sliding_window_size = 511  # window + 1 != rel_extent
        self.assertIsNone(bias_layout._visit_for(o, f, a, g1, g2))
        o, f, a, g1, g2, _ = self._setup()
        o.is_local, a.sliding_window_size = False, -1
        self.assertEqual(bias_layout._visit_for(o, f, a, g1, g2).local, False)

    def test_rel_forward_and_varlen(self):
        made = []
        orig = bias_layout.sheared_rel_bias
        bias_layout.sheared_rel_bias = lambda *a, **k: made.append(a) or "SHEARED"
        try:
            prev = lambda self, r, tau=None: "STOCK"  # noqa: E731
            fwd = bias_layout.make_rel_forward(prev, empty=None)

            class FakeCuda(torch.Tensor):
                @property
                def is_cuda(self):
                    return True

            proj_mod = types.SimpleNamespace(proj=torch.zeros(16, 1024, dtype=torch.bfloat16))
            owner = types.SimpleNamespace(rel_logits_proj=proj_mod, rel_extent=1024)
            r = torch.zeros(4, 24, 16, dtype=torch.bfloat16).as_subclass(FakeCuda)
            self.assertEqual(fwd(proj_mod, r), "STOCK")  # no visit
            token = bias_layout._CTX.set(bias_layout._Visit(owner, True, 1023, None, None))
            try:
                self.assertEqual(fwd(proj_mod, r), "SHEARED")
                self.assertEqual(fwd(proj_mod, r, torch.ones(4)), "STOCK")  # tau given
                other = types.SimpleNamespace(proj=proj_mod.proj)
                self.assertEqual(fwd(other, r), "STOCK")  # another layer's proj
            finally:
                bias_layout._CTX.reset(token)
        finally:
            bias_layout.sheared_rel_bias = orig
        got = []
        varlen = bias_layout.make_varlen(lambda **kw: got.append(kw))
        plain = torch.zeros(100, 24, 1024)
        full = torch.zeros(192, 24, 1280)  # round_up(170, 64) rows
        q = torch.zeros(130, 24, 128)  # SGLang narrowed q (and the bias) to 130 real rows
        varlen(q=q, rel_bias=plain, rel_bias_prep_cache={})
        with bias_layout.use_presheared(full):
            varlen(q=q, rel_bias=full[:130], rel_bias_prep_cache={})
            varlen(q=q, rel_bias=plain, rel_bias_prep_cache={})  # another tensor: stock
            with self.assertRaises(RuntimeError):  # more query rows than the buffer holds
                varlen(q=torch.zeros(193, 24, 128), rel_bias=full, rel_bias_prep_cache={})
        varlen(q=1)
        self.assertNotIn("rel_bias_presheared", got[0])
        self.assertEqual(got[1].get("rel_bias_presheared"), True)
        self.assertEqual(tuple(got[1]["rel_bias"].shape), (192, 24, 1280))  # round_up(130, 64)
        self.assertEqual(got[1]["rel_bias"].data_ptr(), full.data_ptr())
        self.assertNotIn("rel_bias_prep_cache", got[1])
        self.assertNotIn("rel_bias_presheared", got[2])
        self.assertEqual(got[3], {"q": 1})
        bias_layout._PRODUCED_WIDTHS.add(1280)
        try:
            with self.assertRaisesRegex(RuntimeError, "outside the layer visit"):
                varlen(q=q, rel_bias=full[:130], rel_bias_prep_cache={})  # no visit
        finally:
            bias_layout._PRODUCED_WIDTHS.discard(1280)

    def test_visit_records_buffer_and_breakable(self):
        owner, fb, attn, gab, gm, _ = BiasLayoutLogicTests._setup(self)
        self.assertIsNone(bias_layout._visit_for(owner, fb, attn, gab, gm, lambda: True))
        self.assertIsNotNone(bias_layout._visit_for(owner, fb, attn, gab, gm, lambda: False))
        orig = bias_layout.sheared_rel_bias
        bias_layout.sheared_rel_bias = lambda *a, **k: torch.ones(1)
        try:
            class FakeCuda(torch.Tensor):
                @property
                def is_cuda(self):
                    return True

            proj_mod = types.SimpleNamespace(proj=torch.zeros(16, 1024, dtype=torch.bfloat16))
            owner = types.SimpleNamespace(rel_logits_proj=proj_mod, rel_extent=1024)
            visit = bias_layout._Visit(owner, True, 1023, None, None)
            fwd = bias_layout.make_rel_forward(lambda self, r, tau=None: None, empty=None)
            token = bias_layout._CTX.set(visit)
            try:
                out = fwd(proj_mod, torch.zeros(4, 24, 16, dtype=torch.bfloat16).as_subclass(FakeCuda))
            finally:
                bias_layout._CTX.reset(token)
            self.assertIs(visit.sheared, out)
        finally:
            bias_layout.sheared_rel_bias = orig


PLUMBING = textwrap.dedent(
    """
    import os, sys, types
    sys.path.insert(0, {root!r})
    import torch
    import inkling_prefill as P

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    for pkg in ["sglang", "sglang.srt", "sglang.srt.models", "sglang.srt.models.inkling_common",
                "sglang.srt.layers", "sglang.srt.layers.quantization", "sglang.kernels", "sglang.kernels.ops",
                "sglang.kernels.ops.moe", "sglang.kernels.ops.attention", "sglang.kernels.ops.attention.flash_attn",
                "sglang.kernels.ops.attention.flash_attn.cute"]:
        mod(pkg).__path__ = []
    class Inkling2Attention:
        def _project_sources(self, s): return "stock"
        def _forward(self, h, p, fb, *, attn, attn_padded): return "stock"
    class RelLogitsProj:
        def forward(self, r, log_scaling_tau=None): return "stock"
    class Inkling2Experts:
        def _forward_sm120(self, x, w, i, c): return "stock"
    def routed_partial(x, o, s, k, *, nvfp4_scales=None): return "stock"
    def nvfp4_routed_quantize(x, dst2src, top_k, offsets): return "stock"
    def _nvfp4_outputs(rows, width, experts, device): return None
    backend = types.SimpleNamespace(is_cutedsl=lambda: False, is_gemv=lambda: False)
    order = os.environ["ORDER"]
    if order == "before":
        mod("sglang.srt.models.inkling_common.inkling2_attn", Inkling2Attention=Inkling2Attention,
            get_attn_backend=lambda: None, get_model=lambda: None)
    mod("sglang.srt.layers.quantization.unquant", get_bf16_gemm_backend=lambda: backend)
    mod("sglang.kernels.ops.attention.flash_attn.cute.interface", _shear_bias_empty=lambda *a: None)
    for pkg in ["sglang.srt.model_executor", "sglang.srt.model_executor.runner_backend_utils",
                "sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph"]:
        mod(pkg).__path__ = []
    mod("sglang.srt.model_executor.runner_backend_utils.breakable_cuda_graph.context",
        is_in_breakable_cuda_graph=lambda: False)
    print(P.install(root={fake!r}))
    if order == "after":
        mod("sglang.srt.models.inkling_common.inkling2_attn", Inkling2Attention=Inkling2Attention,
            get_attn_backend=lambda: None, get_model=lambda: None)
        import importlib.util
    # the hook fires on real imports; simulate them for the remaining modules
    from inkling_prefill import _hook
    for name, attrs in [
        ("sglang.srt.models.inkling_common.attn", dict(RelLogitsProj=RelLogitsProj)),
        ("sglang.kernels.ops.attention.flash_attention_v4_sm120", dict(_flash_attn_varlen_func=lambda **k: k)),
        ("sglang.kernels.ops.moe.inkling2_experts_sm120",
         dict(nvfp4_routed_quantize=nvfp4_routed_quantize, _nvfp4_outputs=_nvfp4_outputs)),
        ("sglang.srt.models.inkling_common.inkling2_moe",
         dict(Inkling2Experts=Inkling2Experts, routed_partial=routed_partial)),
    ] + ([("sglang.srt.models.inkling_common.inkling2_attn", None)] if order == "after" else []):
        m = sys.modules[name] if attrs is None else mod(name, **attrs)
        _hook._run(name, m)
    print(sorted(P.STATE["applied"]))
    print(_hook.pending())
    import sglang.srt.models.inkling_common.inkling2_attn as A
    assert hasattr(A.Inkling2Attention._project_sources, "inkling_prefill_qkvr_split")
    assert hasattr(A.Inkling2Attention._forward, "inkling_prefill_bias_layout")
    print("OK")
    """
)


class PlumbingTests(unittest.TestCase):
    def test_install_and_patch(self):
        fake = fake_tree()
        for order in ("before", "after"):
            code = PLUMBING.format(root=ROOT, fake=fake)
            env = dict(os.environ, ORDER=order, INKLING_PREFILL="all")
            p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
            self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
            self.assertIn("OK", p.stdout)
            self.assertIn("enabled qkvr_split(min_m=128,out=strided) bias_layout quant_once(min_t=256) partial_bf16",
                          p.stderr)
            applied = p.stdout.splitlines()[1]
            for lever in P.LEVERS:
                self.assertIn(lever, applied)
            self.assertEqual(p.stdout.splitlines()[2], "[]")

    def test_noop_when_unset(self):
        code = (f"import sys; sys.path.insert(0, {ROOT!r}); import inkling_prefill as P; "
                "print(P.install(), 'torch' in sys.modules, 'sglang' in sys.modules)")
        env = {k: v for k, v in os.environ.items() if k != "INKLING_PREFILL"}
        p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(p.stdout.strip(), "() False False", p.stderr)
        self.assertEqual(p.stderr, "")

    def test_sitecustomize_fatal(self):
        env = dict(os.environ, INKLING_PREFILL="bogus", PYTHONPATH=ROOT)
        p = subprocess.run([sys.executable, "-c", "print('ran')"], capture_output=True, text=True, env=env)
        self.assertEqual(p.returncode, 78)
        self.assertIn("FATAL", p.stderr)


if __name__ == "__main__":
    unittest.main()
