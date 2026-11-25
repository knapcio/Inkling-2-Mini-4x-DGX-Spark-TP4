"""CPU tests for inkling_rsfix (no GPU, no triton, no sglang import).

Source checks read the image's sglang tree: RSFIX_SGLANG_SRC (the directory that holds
sglang's ``kernels/`` and ``srt/``), else the installed sglang package; skipped when
neither exists.

    python -m pytest -q tests/test_rsfix_cpu.py
"""

import importlib.util
import math
import os
import random
import shutil
import subprocess
import sys
import textwrap
import typing

import pytest
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))  # overlay/ (tests/rsfix/..)
sys.path.insert(0, ROOT)

import inkling_rsfix  # noqa: E402  (imports only _hook and _sources)
from inkling_rsfix import _sources  # noqa: E402
from inkling_rsfix import reference as ref  # noqa: E402

DATA = os.path.join(HERE, "data")
F32 = torch.float32


def _src_root():
    env = os.environ.get("RSFIX_SGLANG_SRC")
    if env:
        return env if os.path.isdir(env) else None
    spec = importlib.util.find_spec("sglang")
    if spec is not None and spec.submodule_search_locations:
        return list(spec.submodule_search_locations)[0]
    return None


SRC = _src_root()
needs_src = pytest.mark.skipif(SRC is None, reason="no sglang source tree (set RSFIX_SGLANG_SRC)")


def _read(rel):
    with open(os.path.join(SRC, rel), encoding="utf-8") as fh:
        return fh.read()


def _read_pkg(name):
    with open(os.path.join(ROOT, "inkling_rsfix", name), encoding="utf-8") as fh:
        return fh.read()


# ----------------------------------------------------------------------------- sources
TARGETS = tuple(t for p in inkling_rsfix.PARTS for t in inkling_rsfix.PART_TARGETS[p])


def test_parts_and_targets():
    # every part maps to pinned target modules; `selector` covers the three selector-walk modules
    assert inkling_rsfix.PARTS == ("reject", "markov", "blockwise", "selector")
    assert set(inkling_rsfix.PART_TARGETS) == set(inkling_rsfix.PARTS)
    assert inkling_rsfix.PART_TARGETS["selector"] == ("selector_kernel", "selector_model", "selector_lilicorr")
    assert set(TARGETS) == set(_sources.MODULES) == set(_sources.EXPECTED) == set(inkling_rsfix._PLAN)
    assert inkling_rsfix.parts({"INKLING_RSFIX": "1"}) == inkling_rsfix.PARTS
    assert inkling_rsfix.parts({"INKLING_RSFIX": "selector,reject"}) == ("reject", "selector")
    assert inkling_rsfix.parts({"INKLING_RSFIX": "0"}) == ()
    with pytest.raises(RuntimeError):
        inkling_rsfix.parts({"INKLING_RSFIX": "reject,bogus"})
    assert inkling_rsfix._targets(("selector",)) == inkling_rsfix.PART_TARGETS["selector"]
    for target in TARGETS:  # every target's plan is the 5-tuple apply() / _is_patched() unpack
        assert len(inkling_rsfix._PLAN[target]) == 5


@needs_src
@pytest.mark.parametrize("target", TARGETS)
def test_pins_match_image(target):
    assert _sources.mismatches(target, _read(_sources.MODULES[target][1])) == []


@needs_src
def test_check_sources_selector_part():
    inkling_rsfix.check_sources(root=SRC, which=("selector",))


@needs_src
def test_check_sources_accepts_image_and_refuses_an_edit(tmp_path):
    inkling_rsfix.check_sources(root=SRC)
    rel = _sources.REJECT
    text = _read(rel).replace("candidates_mask = total_cumsum > target_u\n",
                              "candidates_mask = total_cumsum >= target_u\n", 1)
    dst = tmp_path / rel
    dst.parent.mkdir(parents=True)
    dst.write_text(text)
    with pytest.raises(RuntimeError, match="differs from the pinned image"):
        _sources.check_part("reject", root=str(tmp_path))
    with pytest.raises(RuntimeError, match="is missing"):
        _sources.check_part("markov", root=str(tmp_path))


@needs_src
def test_reject_edits_are_exactly_the_golden_diff():
    import difflib

    stock = _read(_sources.REJECT)
    mine = _read_pkg("reject.py")
    out = []
    for n in ("speculative_sampling_classic_kernel", "_split_residual_tile", "_split_sum_kernel",
              "_split_cdf_kernel", "_split_scratch", "_chain_speculative_sampling_split"):
        a = _sources.segment(stock, n).splitlines(keepends=True)
        b = _sources.segment(mine, n).splitlines(keepends=True)
        out += list(difflib.unified_diff(a, b, f"stock/{n}", f"rsfix/{n}", n=0))
    with open(os.path.join(DATA, "reject_edits.diff"), encoding="utf-8") as fh:
        assert "".join(out) == fh.read()
    # Every removed stock line is one of the intended edit sites.
    removed = {ln[1:].strip() for ln in out if ln.startswith("-") and not ln.startswith("---")}
    assert removed <= {
        "candidates_mask = total_cumsum > target_u",
        "tl.store(Predicts + last_accepted_global_idx, final_token)",
        "# Pass 2: CDF. Degenerate residual (norm_sum == 0, i.e. p == q everywhere on",
        "# rejection) leaves the cumsum at 0 <= target_u, so final_token falls back to",
        "# VOCAB_SIZE - 1; acceptable since this case is numerically near-impossible.",
        'full-vocabulary passes above, including the non-probability guard on q."""',
        "return val",
        "# The stock fallback when no tile matches (degenerate residual).",
        "tl.store(Result + pid, VOCAB_SIZE - 1)",
        "val = _split_residual_tile(",
        'all tiles with an atomic reproduces the stock token."""',
        "# Arrival counter (acq_rel RMW, gpu scope): each program's atomic minimum above is",
        '"""Tile sums, sampled token and arrival counters, one set per (device, stream, tile',
        "count), made once and never replaced, so pointers captured in a CUDA graph stay",
        "valid. Returns ``None`` while capturing until an eager launch has made the set,",
        "tile_sums, result, arrived = scratch",
    }, removed


@needs_src
@pytest.mark.skipif(shutil.which("patch") is None, reason="needs the patch tool")
def test_markov_and_blockwise_are_the_pr_code(tmp_path):
    base = tmp_path / "python" / "sglang"
    for rel in (_sources.MARKOV, _sources.BLOCKWISE, "kernels/ops/speculative/dspark/sharded_accept_hip.py"):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(SRC, rel), base / rel)
    for pr in ("pr51681", "pr51692"):
        with open(os.path.join(DATA, f"{pr}.code.diff"), "rb") as fh:
            subprocess.run(["patch", "-p1", "--quiet"], cwd=tmp_path, stdin=fh, check=True)
    pm = (base / _sources.MARKOV).read_text()
    pb = (base / _sources.BLOCKWISE).read_text()
    mk = _read_pkg("markov.py")
    bw = _read_pkg("blockwise.py")
    for n in ("_exponential_from_philox", "_markov_sample_partial_kernel", "_select_block"):
        assert _sources.segment(mk, n) == _sources.segment(pm, n), n
    # The partial kernel is the stock text (it is copied only to call the fixed _select_block).
    assert _sources.segment(mk, "_markov_sample_partial_kernel") == _sources.segment(
        _read(_sources.MARKOV), "_markov_sample_partial_kernel")
    pr_torch = textwrap.dedent(_sources.segment(pm, "MarkovSampleStep.torch"))
    assert pr_torch.replace("@classmethod\ndef torch(", "def markov_torch(", 1) == \
        _sources.segment(mk, "markov_torch")
    for n in ("_log_normalizer_parts", "_stats_kernel", "_rejection_kernel", "_resample_kernel",
              "blockwise_rejection_triton", "blockwise_rejection_torch"):
        assert _sources.segment(bw, n) == _sources.segment(pb, n), n


def test_parts_env():
    assert inkling_rsfix.parts({}) == ()
    assert inkling_rsfix.parts({"INKLING_RSFIX": "0"}) == ()
    assert inkling_rsfix.parts({"INKLING_RSFIX": "1"}) == inkling_rsfix.PARTS
    assert inkling_rsfix.parts({"INKLING_RSFIX": "blockwise, reject"}) == ("reject", "blockwise")
    with pytest.raises(RuntimeError):
        inkling_rsfix.parts({"INKLING_RSFIX": "reject,bogus"})


# ----------------------------------------------------------------------- lane rule model
def _sparse_row(rng, vocab, valid, k):
    """k positive residual lanes (log-uniform 1e-9..1) among the valid lanes; the padded
    lanes valid..vocab-1 and the rest are 0. p is positive on a superset."""
    val = torch.zeros(vocab, dtype=F32)
    p = torch.zeros(vocab, dtype=F32)
    lanes = rng.sample(range(valid), k)
    for i in lanes:
        val[i] = 10.0 ** rng.uniform(-9, 0)
        p[i] = val[i] + 10.0 ** rng.uniform(-9, -1)
    for i in rng.sample(range(valid), min(valid, k)):
        p[i] = max(float(p[i]), 1e-6)
    return val, p


def _noisy_scan(rng, ulps):
    """A parallel-scan stand-in: the exact prefix with independent relative errors of up
    to ``ulps`` FP32 ulps per lane, so it need not be monotone and the zero lanes after
    the last positive one (padded rows, the masked tail) can sit above the scan's end."""

    def scan(v, t):
        c = torch.cumsum(v.double(), 0)
        noise = torch.tensor([rng.uniform(-ulps, ulps) for _ in range(v.numel())], dtype=torch.float64)
        c = c * (1.0 + noise * 2.0 ** -23)
        return c.to(F32)

    return scan


COINS = (1.0 - 2.0 ** -24, 1.0 - 2.0 ** -23, 1.0 - 2.0 ** -22, 1.0)


@pytest.mark.parametrize("vocab,valid,block_v", [(24, 24, 32), (300, 260, 64), (1000, 931, 128)])
def test_fixed_rule_never_emits_a_zero_mass_lane(vocab, valid, block_v):
    rng = random.Random(vocab)
    stock_bad = fixed_bad = trials = 0
    for _ in range(300):
        val, p = _sparse_row(rng, vocab, valid, rng.randint(1, min(12, valid)))
        scan = _noisy_scan(rng, rng.choice((0.0, 2.0, 8.0)))
        norm = ref.pass1(val, block_v) * (1.0 + rng.randint(0, 4) * 2.0 ** -23)
        for coin in COINS + (rng.random(),):
            trials += 1
            kw = dict(vocab=vocab, block_v=block_v, scan=scan)
            s = ref.final_token(val, p, coin, norm, fixed=False, **kw)
            f = ref.final_token(val, p, coin, norm, fixed=True, **kw)
            stock_bad += not ref.lane_is_valid(s, val, p, valid)
            fixed_bad += not ref.lane_is_valid(f, val, p, valid)
    assert fixed_bad == 0
    assert stock_bad > 0  # the inputs do reach the stock bug


def test_real_vocab_padded_tail():
    # Inkling: logits width 201,024, valid ids 0..200,057, BLOCK_V 4096 (50 tiles; the last
    # tile holds lanes 200,704..201,023 in range and 201,024..204,799 masked).
    vocab, valid, block_v = 201024, 200058, 4096
    rng = random.Random(7)
    val = torch.zeros(vocab, dtype=F32)
    for i in (0, 4095, 4096, 100000, 196607, 196608, 200057):
        val[i] = 10.0 ** rng.uniform(-6, 0)
    p = val.clone()
    norm = ref.pass1(val, block_v)
    kw = dict(vocab=vocab, block_v=block_v)
    # coin 1.0: target_u == norm_sum == the exact scan end -> no lane matches.
    s = ref.final_token(val, p, 1.0, norm, fixed=False, **kw)
    f = ref.final_token(val, p, 1.0, norm, fixed=True, **kw)
    assert s == vocab - 1 and not ref.lane_is_valid(s, val, p, valid)  # a padded row
    assert f == 200057
    # A normalizer one ulp above the scan end with the largest coin below 1.
    norm_up = torch.nextafter(norm, torch.tensor(2.0))
    s = ref.final_token(val, p, 1.0 - 2.0 ** -24, norm_up, fixed=False, **kw)
    f = ref.final_token(val, p, 1.0 - 2.0 ** -24, norm_up, fixed=True, **kw)
    assert f == 200057 and ref.lane_is_valid(f, val, p, valid)
    # The scan of the last tile returns a larger prefix on its masked lanes (>= vocab)
    # than on its in-range ones, as a tree scan may: the stock rule emits lane 201,024
    # (== VOCAB_SIZE, out of range), the failure seen on GPU at V = 24.
    def spike(v, t):
        c = torch.cumsum(v, 0)
        if t == 49:
            c[vocab - 49 * block_v:] += float(norm) * 2.0 ** -20
        return c
    s = ref.final_token(val, p, 1.0, norm, fixed=False, scan=spike, **kw)
    f = ref.final_token(val, p, 1.0, norm, fixed=True, scan=spike, **kw)
    assert s == vocab and not ref.lane_is_valid(s, val, p, valid)
    assert f == 200057


def test_degenerate_residual_falls_back_to_last_positive_target_lane():
    vocab, block_v = 300, 64
    val = torch.zeros(vocab, dtype=F32)
    p = torch.zeros(vocab, dtype=F32)
    p[[3, 70, 211]] = torch.tensor([0.5, 0.25, 0.25])
    for coin in (0.0, 0.5, 1.0 - 2.0 ** -24):
        assert ref.final_token(val, p, coin, 0.0, vocab=vocab, block_v=block_v, fixed=True) == 211
        assert ref.final_token(val, p, coin, 0.0, vocab=vocab, block_v=block_v, fixed=False) == vocab - 1
    # No positive target entry either (not a distribution): the stock token remains.
    z = torch.zeros(vocab, dtype=F32)
    assert ref.final_token(z, z, 0.5, 0.0, vocab=vocab, block_v=block_v, fixed=True) == vocab - 1


def test_fixed_equals_stock_when_stock_is_valid_and_exact():
    # Dyadic values: every partial sum is exact, so the scan, the tile sums and norm_sum
    # agree and the two rules must pick the same lane for every coin below 1.
    rng = random.Random(3)
    vocab, block_v = 500, 64
    for _ in range(40):
        val = torch.zeros(vocab, dtype=F32)
        for i in rng.sample(range(vocab - 40), 9):
            val[i] = rng.randint(1, 64) * 2.0 ** -10
        norm = ref.pass1(val, block_v)
        for k in range(64):
            coin = (k + rng.random()) / 64
            s = ref.final_token(val, val, coin, norm, vocab=vocab, block_v=block_v, fixed=False)
            f = ref.final_token(val, val, coin, norm, vocab=vocab, block_v=block_v, fixed=True)
            assert s == f and val[s] > 0


def test_fixed_rule_law_on_a_coin_grid():
    rng = random.Random(11)
    vocab, block_v, n = 200, 32, 20000
    val = torch.zeros(vocab, dtype=F32)
    lanes = rng.sample(range(150), 12)
    for i in lanes:
        val[i] = rng.uniform(0.01, 1.0)
    norm = ref.pass1(val, block_v)
    counts = torch.zeros(vocab)
    for k in range(n):
        counts[ref.final_token(val, val, (k + 0.5) / n, norm, vocab=vocab, block_v=block_v, fixed=True)] += 1
    law = val.double() / val.double().sum()
    assert float((counts / n - law).abs().max()) <= 2.0 / n
    assert float(counts[val == 0].sum()) == 0


# --------------------------------------------------------------- PR #51681 (CPU torch path)
class _Const:
    def __init__(self, value):
        self.value = value


def _markov_namespace(stock_text):
    ns = {"torch": torch, "Optional": typing.Optional, "Callable": typing.Callable,
          "triton": type("T", (), {"cdiv": staticmethod(lambda a, b: -(-a // b))}),
          "_UNIFORM_SCALE": _Const(2.3283064365386963e-10), "_NO_INDEX": _Const(2147483647),
          "_PHILOX_ROUNDS": 10}
    for n in ("_mul32", "_philox"):
        exec(_sources.segment(stock_text, n), ns)
    return ns


def _markov_torch(src_text, ns):
    exec(src_text, ns)
    fn = ns["markov_torch"]

    class Step:
        _check = staticmethod(lambda *a, **k: None)
        torch = classmethod(fn)

    return Step


@needs_src
def test_pr51681_philox_endpoint_cpu():
    stock_text = _read(_sources.MARKOV)
    key = 0x2D2A9C317E44B1F3
    ns = _markov_namespace(stock_text)
    # The word token 1 draws at the PR's counter rounds to 2**32 in FP32.
    words = ns["_philox"](key, torch.tensor([[0]]), torch.tensor([[0]]), torch.tensor([[0]]),
                          torch.tensor([[4_757_896]]))
    w1 = int(words[1].reshape(-1)[0])
    assert w1 == 0xFFFFFF9F
    assert float((torch.tensor([w1]).float() + 0.5) * 2.3283064365386963e-10) == 1.0

    stock_src = textwrap.dedent(_sources.segment(stock_text, "MarkovSampleStep.torch"))
    stock_src = stock_src.replace("@classmethod\ndef torch(", "def markov_torch(", 1)
    fixed_src = _sources.segment(_read_pkg("markov.py"), "markov_torch")
    out = {}
    for name, src in (("stock", stock_src), ("fixed", fixed_src)):
        step = _markov_torch(src, dict(ns))
        logits = torch.tensor([[-math.inf, 0.0]])
        out[name] = int(step.torch(base_logits=logits, bias=torch.zeros_like(logits),
                                   temperatures=torch.ones(1), key=key,
                                   offset=torch.tensor([4_757_896]), step=0,
                                   step_dtype=torch.float32))
    assert out == {"stock": 0, "fixed": 1}


@needs_src
def test_install_registers_one_hook_per_part(capsys):
    from inkling_rsfix import _hook

    st = dict(inkling_rsfix.STATE)
    try:
        inkling_rsfix.STATE.update(installed=False, parts=(), applied=[], stock={})
        assert inkling_rsfix.install(env={"INKLING_RSFIX": "reject,markov"}, root=SRC)
        assert inkling_rsfix.STATE["parts"] == ("reject", "markov")
        assert {_sources.MODULES[p][0] for p in ("reject", "markov")} <= set(_hook.pending())
        assert "[inkling_rsfix] enabled: parts=reject,markov" in capsys.readouterr().err
        assert inkling_rsfix.install(env={"INKLING_RSFIX": "1"}, root=SRC)  # idempotent
    finally:
        with _hook._lock:
            for t in TARGETS:
                _hook._callbacks.pop(_sources.MODULES[t][0], None)
                _hook._before.pop(_sources.MODULES[t][0], None)
        inkling_rsfix.STATE.clear()
        inkling_rsfix.STATE.update(st)


def test_fixed_modules_do_not_import_sglang():
    # The post-import hook imports them while the stock module is executing; an import of
    # the stock module from here would be a cycle (a review blocker).
    for name in ("reject.py", "markov.py", "blockwise.py", "selector.py"):
        text = _read_pkg(name)
        assert "import sglang" not in text and "from sglang" not in text, name
        assert "_STOCK_DEPS" in text


def test_is_stock_rejects_wrappers():
    import functools
    import types

    mod = types.ModuleType("rsfix_fake_stock")
    exec("def kernel():\n    return 1\nclass C:\n    @classmethod\n    def m(cls):\n        return 2\n", vars(mod))
    assert inkling_rsfix._is_stock(mod.kernel, mod)
    assert inkling_rsfix._is_stock(vars(mod.C)["m"], mod)

    @functools.wraps(mod.kernel)
    def wrapper():
        return mod.kernel()

    assert wrapper.__module__ == mod.kernel.__module__
    assert not inkling_rsfix._is_stock(wrapper, mod)
    wrapper.fn = mod.kernel  # what functools.wraps copies from a JITFunction's __dict__
    assert not inkling_rsfix._is_stock(wrapper, mod)


def _fake_target(tmp_path, monkeypatch, name):
    (tmp_path / f"{name}.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.setitem(_sources.MODULES, "reject", (name, _sources.REJECT))


@pytest.fixture
def clean_state():
    from inkling_rsfix import _hook

    st = {k: (list(v) if isinstance(v, list) else (dict(v) if isinstance(v, dict) else v))
          for k, v in inkling_rsfix.STATE.items()}
    inkling_rsfix.STATE.update(applied=[], fatal=None, stock={})
    added = []
    yield added
    with _hook._lock:
        for n in added:
            _hook._callbacks.pop(n, None)
    for n in added:
        sys.modules.pop(n, None)
    inkling_rsfix.STATE.clear()
    inkling_rsfix.STATE.update(st)


def test_failed_patch_keeps_the_hook_armed(tmp_path, monkeypatch, clean_state):
    """A failed patch must not let a retried import load the stock module unpatched."""
    import functools

    from inkling_rsfix import _hook

    name = "rsfix_fake_target_a"
    _fake_target(tmp_path, monkeypatch, name)
    clean_state.append(name)
    calls = []

    def boom(part, mod):
        calls.append(mod.__name__)
        raise RuntimeError("pin mismatch")

    monkeypatch.setattr(inkling_rsfix, "apply", boom)
    _hook.when_imported(name, functools.partial(inkling_rsfix._on_import, "reject"))
    for _ in range(2):
        with pytest.raises(inkling_rsfix.RsfixFatal):
            importlib.import_module(name)
        assert name not in sys.modules
    assert calls == [name]  # the retry is refused by the latched fatal error
    assert name in _hook.pending()


def test_outer_loader_failure_after_patch_repatches_the_fresh_module(tmp_path, monkeypatch, clean_state):
    """Review blocker: rsfix patches, a later hook raises, the module is discarded; the next
    import executes a fresh stock module, which must be patched again."""
    import functools

    from inkling_rsfix import _hook

    name = "rsfix_fake_target_b"
    _fake_target(tmp_path, monkeypatch, name)
    clean_state.append(name)
    calls = []

    def fake_apply(part, mod):
        calls.append(id(mod))
        setattr(mod, inkling_rsfix.MARK, True)
        return {"stock": {}, "fixed": {}}

    monkeypatch.setattr(inkling_rsfix, "apply", fake_apply)
    _hook.when_imported(name, functools.partial(inkling_rsfix._on_import, "reject"))
    fail = {"on": True}

    def outer(mod):
        if fail["on"]:
            raise RuntimeError("another overlay failed")

    _hook.when_imported(name, outer)
    with pytest.raises(RuntimeError, match="another overlay"):
        importlib.import_module(name)
    assert name not in sys.modules and len(calls) == 1
    fail["on"] = False
    mod = importlib.import_module(name)
    assert getattr(mod, inkling_rsfix.MARK, False) and len(calls) == 2 and calls[1] == id(mod)


@needs_src
def test_install_refuses_an_already_imported_target(monkeypatch, clean_state):
    monkeypatch.setitem(_sources.MODULES, "reject", ("json", _sources.REJECT))
    monkeypatch.setitem(inkling_rsfix.STATE, "installed", False)
    with pytest.raises(RuntimeError, match="install before"):
        inkling_rsfix.install(env={"INKLING_RSFIX": "reject"}, root=SRC)
    assert inkling_rsfix.STATE["installed"] is False


def test_constant_kinds():
    try:  # the real type when triton is installed (the check is by type identity then)
        from triton.language import constexpr
    except ImportError:
        class constexpr:  # noqa: N801  (stands in for triton.language.constexpr)
            def __init__(self, v):
                self.value = v

    assert inkling_rsfix._const_ok(10, "int", 10)
    assert not inkling_rsfix._const_ok(10.0, "int", 10)
    assert not inkling_rsfix._const_ok(True, "int", 1)
    assert inkling_rsfix._const_ok(constexpr(2147483647), "constexpr", 2147483647)
    assert not inkling_rsfix._const_ok(2147483647, "constexpr", 2147483647)


def test_reload_of_a_patched_module_is_refused_before_stock_code_runs(tmp_path, monkeypatch, clean_state):
    """Review blockers: importlib.reload re-executes the stock definitions into the same
    module dictionary. It is refused before execution, and the patched bindings stay."""
    import functools
    import types

    from inkling_rsfix import _hook

    name = "rsfix_fake_target_c"
    (tmp_path / f"{name}.py").write_text("def f():\n    return 'stock'\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    fixed_mod = types.ModuleType("rsfix_fake_fixed")
    exec("def f():\n    return 'fixed'\n", vars(fixed_mod))
    monkeypatch.setitem(sys.modules, "rsfix_fake_fixed", fixed_mod)
    monkeypatch.setitem(inkling_rsfix._PLAN, "reject", ("rsfix_fake_fixed", ("f",), (), {}, None))
    clean_state.append(name)
    calls = []

    def fake_apply(part, mod):
        calls.append(part)
        mod.f = fixed_mod.f
        return {"stock": {}, "fixed": {"f": fixed_mod.f}}

    monkeypatch.setattr(inkling_rsfix, "apply", fake_apply)
    _hook.when_imported(name, functools.partial(inkling_rsfix._on_import, "reject"))
    _hook.before_exec(name, functools.partial(inkling_rsfix._refuse_reload, "reject"))
    try:
        mod = importlib.import_module(name)
        assert mod.f() == "fixed" and len(calls) == 1
        with pytest.raises(inkling_rsfix.RsfixFatal, match="reloading"):
            importlib.reload(mod)
        assert sys.modules[name].f() == "fixed" and len(calls) == 1
        # a wrapper with a copied marker or name is not the fixed object ...
        mod.f = functools.wraps(fixed_mod.f)(lambda: "other")
        assert not inkling_rsfix._is_patched("reject", mod)
        # ... but the module instance was patched once, so a reload is still refused
        inkling_rsfix._PATCHED.add(mod)
        with pytest.raises(inkling_rsfix.RsfixFatal, match="reloading"):
            importlib.reload(mod)
    finally:
        with _hook._lock:
            _hook._before.pop(name, None)


def test_dependency_kinds():
    import functools
    import types

    mod = types.ModuleType("rsfix_fake_dep")
    exec("def _philox():\n    return 1\ndef helper():\n    return 2\nclass RejectionNoise:\n    pass\n"
         "def _split_chain_walk():\n    return 3\n", vars(mod))
    assert inkling_rsfix._dep_ok("_philox", mod._philox, mod, {})
    assert not inkling_rsfix._dep_ok("_philox", mod.helper, mod, {})  # the function of that name only
    assert not inkling_rsfix._dep_ok("_philox", functools.partial(mod._philox), mod, {})
    assert inkling_rsfix._dep_ok("RejectionNoise", mod.RejectionNoise, mod, {})
    assert not inkling_rsfix._dep_ok("RejectionNoise", mod.helper, mod, {})
    assert not inkling_rsfix._dep_ok("_split_chain_walk", mod._split_chain_walk, mod, {})  # must be a JITFunction
    assert not inkling_rsfix._dep_ok("unknown_name", 1, mod, {})
