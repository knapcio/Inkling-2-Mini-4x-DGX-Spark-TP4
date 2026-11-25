"""CPU tests of inkling_ptouch: the wrapper calls the original checkpoint exactly once and returns its result on every
path, records the prompt-end depth at the checkpoint that completes the prompt, touches the prompt path after the
final insert only when it should, goes inert on its own failures and re-raises shutdown after the original ran
(no GPU, no sglang: stand-in modules play the cache, the key builder and the finish reasons).

    python3 -m unittest discover -s tests/ptouch
"""
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # overlay/
import inkling_ptouch as pt  # noqa: E402


class FINISH_ABORT:
    pass


class FINISH_LENGTH:
    pass


def _modules():
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.managers"):
        sys.modules.setdefault(name, types.ModuleType(name))
    common = types.ModuleType("sglang.srt.mem_cache.common")
    common._req_radix_key = lambda cache, req, ids, max_prefix_len=None: ("key", len(ids), max_prefix_len)
    sys.modules["sglang.srt.mem_cache.common"] = common
    rt = types.ModuleType("sglang.srt.managers.req_types")
    rt.FINISH_ABORT = FINISH_ABORT
    sys.modules["sglang.srt.managers.req_types"] = rt


class SWAComponent:
    pass


class MambaComponent:
    pass


class FullComponent:
    pass


def cache_module(publish=72064, hybrid=True, orig_raises=None):
    class UnifiedRadixCache:
        def __init__(self):
            self.disable = False
            self._components_tuple = (FullComponent(), SWAComponent(), MambaComponent()) if hybrid else (FullComponent(),)
            self.calls, self.touches = [], []
            self.touch_raises = None

        def _checkpoint_into_tree(self, req, *, up_to, **kwargs):
            self.calls.append(up_to)
            if orig_raises is not None:
                raise orig_raises
            if not req.finished():
                req.kv.cache_inserted_len = max(req.kv.cache_inserted_len, min(up_to, publish))
            return ("orig", up_to)

        def touch_prefix(self, key):
            if self.touch_raises is not None:
                raise self.touch_raises
            self.touches.append(key)

    m = types.ModuleType("fake_urc")
    m.UnifiedRadixCache = UnifiedRadixCache
    return m


class Req:
    def __init__(self, prompt=72083, done=False):
        self.origin_input_ids = list(range(prompt))
        self.retraction_count = 0
        self.session_id = None
        self.kv = types.SimpleNamespace(cache_inserted_len=0)
        self.finished_reason = None
        self._done = done
        self.finished_raises = None

    def finished(self):
        if self.finished_raises is not None:                                 # once: the overlay's pre-read
            exc, self.finished_raises = self.finished_raises, None
            raise exc
        return self._done

    def finish(self, reason=FINISH_LENGTH):
        self._done = True
        self.finished_reason = reason()


def lifecycle(cache, req, chunks=(8192, 72083), final=74131):
    outs = [cache._checkpoint_into_tree(req, up_to=u) for u in chunks]
    req.finish()
    outs.append(cache._checkpoint_into_tree(req, up_to=final))
    return outs


class Base(unittest.TestCase):
    def setUp(self):
        _modules()
        pt.STATE.update(calls=0, skipped=0, inert="", min=4096, tp=1, agreed=0, off=False, installed=False)
        self.err = io.StringIO()
        self._r = redirect_stderr(self.err)
        self._r.__enter__()

    def tearDown(self):
        self._r.__exit__(None, None, None)

    def wrapped(self, **kw):
        m = cache_module(**kw)
        pt.wrap(m)
        return m.UnifiedRadixCache()


class TestNormal(Base):
    def test_touch_after_final_insert(self):
        c = self.wrapped()
        req = Req()
        outs = lifecycle(c, req)
        self.assertEqual(c.calls, [8192, 72083, 74131])                      # orig exactly once per call
        self.assertEqual(outs, [("orig", 8192), ("orig", 72083), ("orig", 74131)])
        self.assertEqual(c.touches, [("key", 72083, 72064)])                 # the prompt, limited to the published depth
        self.assertEqual(pt.STATE["calls"], 1)
        self.assertEqual(pt.STATE["inert"], "")

    def test_no_touch_cases(self):
        for label, prep in (
                ("abort", lambda r: None),
                ("retracted", lambda r: setattr(r, "retraction_count", 1)),
                ("session", lambda r: setattr(r, "session_id", "s"))):
            pt.STATE.update(inert="")
            c = self.wrapped()
            req = Req()
            c._checkpoint_into_tree(req, up_to=72083)
            prep(req)
            if label == "abort":
                req.finish(FINISH_ABORT)
            else:
                req.finish()
            out = c._checkpoint_into_tree(req, up_to=74131)
            self.assertEqual(out, ("orig", 74131), label)
            self.assertEqual(c.touches, [], label)
            self.assertEqual(c.calls, [72083, 74131], label)

    def test_short_prompt_and_overlong_depth(self):
        c = self.wrapped()
        req = Req(prompt=1000)
        lifecycle(c, req, chunks=(1000,), final=1500)
        self.assertEqual(c.touches, [])
        c = self.wrapped(publish=80000)                                      # a depth past the prompt is never recorded
        req = Req()
        req.kv.cache_inserted_len = 72100                                    # e.g. a re-prefill that includes output
        c._checkpoint_into_tree(req, up_to=72083)
        req.finish()
        c._checkpoint_into_tree(req, up_to=74131)
        self.assertEqual(c.touches, [])

    def test_non_hybrid_never_touches(self):
        c = self.wrapped(hybrid=False)
        lifecycle(c, Req())
        self.assertEqual(c.touches, [])

    def test_wrap_is_idempotent(self):
        m = cache_module()
        pt.wrap(m)
        first = m.UnifiedRadixCache._checkpoint_into_tree
        pt.wrap(m)
        self.assertIs(m.UnifiedRadixCache._checkpoint_into_tree, first)


class TestFailures(Base):
    def test_orig_error_propagates_untouched(self):
        c = self.wrapped(orig_raises=ValueError("orig"))
        with self.assertRaises(ValueError):
            c._checkpoint_into_tree(Req(), up_to=72083)
        self.assertEqual(c.calls, [72083])
        self.assertEqual(pt.STATE["inert"], "")

    def test_pre_read_error_runs_orig_once_and_goes_inert(self):
        c = self.wrapped()
        req = Req()
        req.finished_raises = RuntimeError("boom")
        out = c._checkpoint_into_tree(req, up_to=72083)
        self.assertEqual(out, ("orig", 72083))
        self.assertEqual(c.calls, [72083])
        self.assertEqual(pt.STATE["inert"], "pre-read failed")

    def test_touch_error_keeps_result_and_goes_inert(self):
        c = self.wrapped()
        req = Req()
        c._checkpoint_into_tree(req, up_to=72083)
        req.finish()
        c.touch_raises = RuntimeError("touch")
        out = c._checkpoint_into_tree(req, up_to=74131)
        self.assertEqual(out, ("orig", 74131))
        self.assertEqual(pt.STATE["inert"], "touch failed")
        # inert: later requests run the original only
        req2 = Req()
        c.touch_raises = None
        lifecycle(c, req2)
        self.assertEqual(c.touches, [])

    def test_shutdown_reraised_after_orig(self):
        c = self.wrapped()
        req = Req()
        req.finished_raises = KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            c._checkpoint_into_tree(req, up_to=72083)
        self.assertEqual(c.calls, [72083])                                   # the original ran first
        self.assertEqual(pt.STATE["inert"], "pre-read failed")
        pt.STATE.update(inert="")
        c = self.wrapped()
        req = Req()
        c._checkpoint_into_tree(req, up_to=72083)
        req.finish()
        c.touch_raises = SystemExit(3)
        with self.assertRaises(SystemExit):
            c._checkpoint_into_tree(req, up_to=74131)
        self.assertEqual(c.calls, [72083, 74131])
        self.assertEqual(pt.STATE["inert"], "touch failed")

    def test_premise_mismatch_goes_inert(self):
        m = types.ModuleType("fake_swa")

        class Changed:
            def refresh_lru(self, phase, node, root):
                return None
        m.SWAComponent = Changed
        pt._premise(m)
        self.assertIn("refresh_lru changed", pt.STATE["inert"])
        c = self.wrapped()
        lifecycle(c, Req())
        self.assertEqual(c.touches, [])
        self.assertEqual(c.calls, [8192, 72083, 74131])

    def test_premise_holds(self):
        m = types.ModuleType("fake_swa")

        class Same:
            def refresh_lru(self, phase, node, root):
                match phase:
                    case "WALKDOWN":
                        return
                    case _:
                        self.lru.reset_node_and_window_ancestors_mru(node)
        m.SWAComponent = Same
        pt._premise(m)
        self.assertEqual(pt.STATE["inert"], "")

    def test_bad_min_env(self):
        os.environ["INKLING_PROMPT_TOUCH_MIN"] = "x"
        try:
            pt.install()
        finally:
            del os.environ["INKLING_PROMPT_TOUCH_MIN"]
        self.assertEqual(pt.STATE["inert"], "invalid INKLING_PROMPT_TOUCH_MIN")


if __name__ == "__main__":
    unittest.main()
