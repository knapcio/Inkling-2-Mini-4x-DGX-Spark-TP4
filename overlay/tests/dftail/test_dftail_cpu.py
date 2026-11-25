"""CPU tests of inkling_dftail: switch parsing, the prefix-match bound, the input-size gate, the DFLASH scope, the
streaming-session guard, the source pin, patch idempotence and the install path (no GPU, no sglang import; a stand-in
Req class plays schedule_batch.Req). In the SGLang image the source pin is also checked against the real file.

    python3 -m unittest discover -s tests/dftail
"""
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/
import inkling_dftail as dt  # noqa: E402
from inkling_dftail import _hook, _sources  # noqa: E402

PINNED_TEXT = '''class ReqDllmMixin:
    pass


class Req(ReqDllmMixin):
    def init_next_round_input(self, tree_cache=None):
        key_limit = self._compute_max_prefix_len(len(self.full_untruncated_fill_ids))
        return key_limit

    def _compute_max_prefix_len(self, input_len: int) -> int:
        return max(input_len - 1, 0)
'''


def fake_module(dllm=False):
    class Req:
        session = None

        def is_dllm(self):
            return dllm

        def _compute_max_prefix_len(self, input_len):  # stock bound: everything but the last token
            return input_len - 1

    return types.SimpleNamespace(Req=Req)


def reset(dflash=True):
    dt.STATE.update(installed=False, tail=0, max_input=4096, bounded=0)
    dt.SCOPE.update(dflash=dflash, why="", logged=False)


def quiet():
    return redirect_stderr(io.StringIO())


class Switch(unittest.TestCase):
    def test_off(self):
        for raw in ("", "0", " 0 ", "  "):
            self.assertEqual(dt.tail_from_env({dt.ENV: raw}), 0)
        self.assertEqual(dt.tail_from_env({}), 0)

    def test_values(self):
        self.assertEqual(dt.tail_from_env({dt.ENV: "2048"}), 2048)
        self.assertEqual(dt.tail_from_env({dt.ENV: " 1024 "}), 1024)
        self.assertEqual(dt.tail_from_env({dt.ENV: str(1 << 20)}), 1 << 20)

    def test_invalid(self):
        for raw in ("abc", "2k", "1.5", "-1", str((1 << 20) + 1)):
            with self.assertRaises(RuntimeError, msg=raw):
                dt.tail_from_env({dt.ENV: raw})

    def test_max_input(self):
        key = dt.ENV + "_MAX_INPUT"
        self.assertEqual(dt.max_input_from_env({}), 4096)
        self.assertEqual(dt.max_input_from_env({key: " "}), 4096)
        self.assertEqual(dt.max_input_from_env({key: "8192"}), 8192)
        self.assertEqual(dt.max_input_from_env({key: "0"}), 0)  # 0: never bound
        for raw in ("8k", "1.5", "-1", "-4096"):
            with self.assertRaises(RuntimeError, msg=raw):
                dt.max_input_from_env({key: raw})


class Bound(unittest.TestCase):
    def setUp(self):
        reset()

    def patched(self, tail=2048, max_input=4096, dllm=False):
        mod = fake_module(dllm)
        dt.STATE.update(tail=tail, max_input=max_input)
        self.assertTrue(dt.patch(mod))
        return mod.Req()

    def test_bound_inside_gate(self):
        r = self.patched()
        self.assertEqual(r._compute_max_prefix_len(3000), 952)   # leaves the last 2048 tokens
        self.assertEqual(r._compute_max_prefix_len(4096), 2048)  # the gate is inclusive
        self.assertEqual(r._compute_max_prefix_len(2049), 1)
        self.assertEqual(r._compute_max_prefix_len(2048), 0)
        self.assertEqual(r._compute_max_prefix_len(1500), 0)     # shorter than the tail: nothing reused
        self.assertEqual(dt.STATE["bounded"], 5)

    def test_above_gate_is_stock(self):
        r = self.patched()
        for n in (4097, 8192, 32768, 131072):
            self.assertEqual(r._compute_max_prefix_len(n), n - 1)
        self.assertEqual(dt.STATE["bounded"], 0)

    def test_gate_override(self):
        r = self.patched(max_input=8192)
        self.assertEqual(r._compute_max_prefix_len(8192), 6144)
        self.assertEqual(r._compute_max_prefix_len(8193), 8192)

    def test_gate_zero_never_bounds(self):
        r = self.patched(max_input=0)
        self.assertEqual(r._compute_max_prefix_len(3000), 2999)

    def test_never_raises_the_stock_bound(self):
        mod = fake_module()
        mod.Req._compute_max_prefix_len = lambda self, n: 100  # e.g. a logprob request already bounds the match
        dt.STATE.update(tail=2048, max_input=4096)
        dt.patch(mod)
        self.assertEqual(mod.Req()._compute_max_prefix_len(3000), 100)
        self.assertEqual(mod.Req()._compute_max_prefix_len(2100), 52)

    def test_dllm_untouched(self):
        r = self.patched(dllm=True)
        self.assertEqual(r._compute_max_prefix_len(3000), 2999)

    def test_tail_zero_is_stock(self):
        r = self.patched(tail=0)
        self.assertEqual(r._compute_max_prefix_len(3000), 2999)

    def test_streaming_session_is_stock(self):
        r = self.patched()
        r.session = types.SimpleNamespace(streaming=True)
        self.assertEqual(r._compute_max_prefix_len(3000), 2999)
        r.session = types.SimpleNamespace(streaming=False)  # a non-streaming session is bounded as usual
        self.assertEqual(r._compute_max_prefix_len(3000), 952)
        self.assertEqual(dt.STATE["bounded"], 1)

    def test_patch_once(self):
        mod = fake_module()
        dt.STATE.update(tail=2048, max_input=4096)
        orig = mod.Req._compute_max_prefix_len
        self.assertTrue(dt.patch(mod))
        wrapped = mod.Req._compute_max_prefix_len
        self.assertIs(wrapped.__wrapped__, orig)
        self.assertFalse(dt.patch(mod))  # second patch is a no-op, never wraps twice
        self.assertIs(mod.Req._compute_max_prefix_len, wrapped)
        self.assertEqual(mod.Req()._compute_max_prefix_len(3000), 952)

    def test_missing_target(self):
        with self.assertRaises(RuntimeError):
            dt.patch(types.SimpleNamespace())
        with self.assertRaises(RuntimeError) as cm:
            dt.patch(types.SimpleNamespace(Req=type("Req", (), {})))
        self.assertIn("DFLASH_PREFIX_TAIL=0", str(cm.exception))


class Scope(unittest.TestCase):
    def setUp(self):
        reset(dflash=None)
        self.saved = dt._spec_algorithm

    def tearDown(self):
        dt._spec_algorithm = self.saved
        reset()

    def req(self):
        mod = fake_module()
        dt.STATE.update(tail=2048, max_input=4096)
        dt.patch(mod)
        return mod.Req()

    def test_dflash_string_and_enum(self):
        for alg in ("DFLASH", "dflash", types.SimpleNamespace(name="DFLASH")):
            reset(dflash=None)
            dt._spec_algorithm = lambda alg=alg: alg
            with quiet():
                self.assertEqual(self.req()._compute_max_prefix_len(3000), 952, msg=repr(alg))

    def test_other_algorithms_inert(self):
        for alg in (None, "DSPARK", "EAGLE", "NONE", types.SimpleNamespace(name="NGRAM")):
            reset(dflash=None)
            dt._spec_algorithm = lambda alg=alg: alg
            err = io.StringIO()
            with redirect_stderr(err):
                self.assertEqual(self.req()._compute_max_prefix_len(3000), 2999, msg=repr(alg))
            self.assertIn("inert", err.getvalue())
            self.assertEqual(dt.STATE["bounded"], 0)

    def test_resolved_once(self):
        calls = []

        def alg():
            calls.append(1)
            return "DFLASH"

        dt._spec_algorithm = alg
        r = self.req()
        with quiet():
            for _ in range(5):
                r._compute_max_prefix_len(3000)
        self.assertEqual(len(calls), 1)

    def test_unreadable_context_is_stock_and_retried(self):
        state = {"ready": False}

        def alg():
            if not state["ready"]:
                raise RuntimeError("runtime context not published")
            return "DFLASH"

        dt._spec_algorithm = alg
        r = self.req()
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(r._compute_max_prefix_len(3000), 2999)
            self.assertEqual(r._compute_max_prefix_len(3000), 2999)
            self.assertEqual(err.getvalue().count("cannot read"), 1)  # logged once
            state["ready"] = True
            self.assertEqual(r._compute_max_prefix_len(3000), 952)

    def test_scope_not_consulted_when_nothing_to_cap(self):
        def alg():
            raise AssertionError("must not be called")

        dt._spec_algorithm = alg
        r = self.req()
        self.assertEqual(r._compute_max_prefix_len(5000), 4999)  # above the gate
        self.assertEqual(r._compute_max_prefix_len(1), 0)       # stock bound already 0


class Sources(unittest.TestCase):
    def setUp(self):
        self.pins = _sources.EXPECTED[_sources.SCHEDULE_BATCH]
        self.saved = dict(self.pins)
        self.pins.clear()
        for name in self.saved:
            self.pins[name] = _sources.digest(PINNED_TEXT, name)

    def tearDown(self):
        self.pins.clear()
        self.pins.update(self.saved)

    def root_with(self, text):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d)
        p = os.path.join(d, _sources.SCHEDULE_BATCH)
        os.makedirs(os.path.dirname(p))
        with open(p, "w", encoding="utf-8") as fh:
            fh.write(text)
        return d

    def test_pinned_names(self):
        self.assertEqual(sorted(self.saved), ["Req._compute_max_prefix_len", "Req.init_next_round_input"])

    def test_match(self):
        self.assertEqual(_sources.mismatches(PINNED_TEXT), [])
        _sources.check(self.root_with(PINNED_TEXT))

    def test_changed_body_refuses(self):
        text = PINNED_TEXT.replace("max(input_len - 1, 0)", "max(input_len - 2, 0)")
        bad = _sources.mismatches(text)
        self.assertEqual([b[0] for b in bad], ["Req._compute_max_prefix_len"])
        with self.assertRaises(RuntimeError) as cm:
            _sources.check(self.root_with(text), dt.OFF_HINT)
        self.assertIn("DFLASH_PREFIX_TAIL=0", str(cm.exception))

    def test_changed_caller_refuses(self):
        text = PINNED_TEXT.replace("return key_limit", "return None")
        self.assertEqual([b[0] for b in _sources.mismatches(text)], ["Req.init_next_round_input"])

    def test_missing_or_duplicate(self):
        text = PINNED_TEXT.replace("def _compute_max_prefix_len", "def _other")
        self.assertIn(("Req._compute_max_prefix_len", "missing or defined twice"), _sources.mismatches(text))
        dup = PINNED_TEXT + "\n\nclass Req:\n    pass\n"
        self.assertEqual(len(_sources.mismatches(dup)), 2)

    def test_missing_file(self):
        with self.assertRaises(RuntimeError) as cm:
            _sources.check(tempfile.gettempdir() + "/no-such-sglang-root", dt.OFF_HINT)
        self.assertIn("DFLASH_PREFIX_TAIL=0", str(cm.exception))


@unittest.skipIf(importlib.util.find_spec("sglang") is None, "needs the SGLang image")
class PinnedImage(unittest.TestCase):
    def test_installed_sglang_matches(self):
        _sources.check()


class Install(unittest.TestCase):
    def setUp(self):
        reset()
        self.saved_check = _sources.check
        _sources.check = lambda *a, **k: None  # the pin itself is tested in Sources
        self.saved_mod = sys.modules.pop(dt.MODULE, None)

    def tearDown(self):
        _sources.check = self.saved_check
        _hook._callbacks.pop(dt.MODULE, None)
        if self.saved_mod is not None:
            sys.modules[dt.MODULE] = self.saved_mod
        reset()

    def imported(self):
        """simulate the first import of schedule_batch: run the registered post-import callbacks"""
        mod = fake_module()
        _hook._run(dt.MODULE, mod)
        return mod

    def test_off_installs_nothing(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(dt.install({}), 0)
        self.assertEqual(err.getvalue(), "")
        self.assertNotIn(dt.MODULE, _hook.pending())

    def test_off_skips_the_pin(self):
        def refuse(*a, **k):
            raise RuntimeError("pin checked")

        _sources.check = refuse
        self.assertEqual(dt.install({dt.ENV: "0", dt.ENV + "_MAX_INPUT": "8192"}), 0)

    def test_armed_default_gate(self):
        err = io.StringIO()
        with redirect_stderr(err):
            self.assertEqual(dt.install({dt.ENV: "2048"}), 2048)
        self.assertIn("last 2048 input tokens", err.getvalue())
        self.assertIn("at most 4096 tokens", err.getvalue())
        self.assertIn(dt.MODULE, _hook.pending())
        mod = self.imported()
        self.assertEqual(mod.Req()._compute_max_prefix_len(3000), 952)
        self.assertEqual(mod.Req()._compute_max_prefix_len(5000), 4999)

    def test_armed_gate_override(self):
        with quiet():
            dt.install({dt.ENV: "2048", dt.ENV + "_MAX_INPUT": "8192"})
        self.assertEqual(dt.STATE["max_input"], 8192)
        self.assertEqual(self.imported().Req()._compute_max_prefix_len(5000), 2952)

    def test_bad_gate(self):
        for raw in ("8k", "-1"):
            reset()
            with self.assertRaises(RuntimeError, msg=raw):
                dt.install({dt.ENV: "2048", dt.ENV + "_MAX_INPUT": raw})
            self.assertFalse(dt.STATE["installed"])

    def test_pin_mismatch_refuses(self):
        def refuse(root=None, off_hint="", files=()):
            raise RuntimeError("inkling_dftail: differs from the pinned image" + off_hint)

        _sources.check = refuse
        with self.assertRaises(RuntimeError) as cm:
            dt.install({dt.ENV: "2048"})
        self.assertIn("DFLASH_PREFIX_TAIL=0", str(cm.exception))
        self.assertFalse(dt.STATE["installed"])
        self.assertNotIn(dt.MODULE, _hook.pending())

    def test_refuses_after_import(self):
        sys.modules[dt.MODULE] = fake_module()
        try:
            with self.assertRaises(RuntimeError):
                dt.install({dt.ENV: "2048"})
        finally:
            sys.modules.pop(dt.MODULE, None)

    def test_install_once(self):
        with quiet():
            dt.install({dt.ENV: "2048"})
            self.assertEqual(dt.install({dt.ENV: "1024"}), 2048)  # the first install wins
        self.assertEqual(self.imported().Req()._compute_max_prefix_len(3000), 952)


if __name__ == "__main__":
    unittest.main()
