"""CPU tests of the wide8 option of inkling_draftbytes (parsing, ops wiring, the row-mode choice).

  python3 -m unittest discover -s tests/draftbytes -p 'test_wide8.py' -v      (from overlay/)

The kernels themselves are checked on a GPU by tools/gpu_check_wide8.py (bit-equality with the default kernel).
"""

import os
import sys
import types
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)

from inkling_draftbytes import _opts, ops  # noqa: E402


class Wide8Options(unittest.TestCase):
    def test_parse(self):
        o = _opts.parse("head8,wide8", environ={})
        self.assertTrue(o.wide8 and o.head8 is not None)
        self.assertEqual((o.wide8_mode, o.wide8_stages, o.wide8_bn, o.wide8_warps), (1, 3, 64, 8))
        self.assertIn("wide8", o.describe())
        self.assertFalse(_opts.parse("head8", environ={}).wide8)  # default unchanged

    def test_env_tunables(self):
        old = dict(os.environ)
        try:
            os.environ["INKLING_DRAFT_WIDE8_MODE"] = "1"
            os.environ["INKLING_DRAFT_WIDE8_STAGES"] = "2"
            o = _opts.parse("wide8", environ={})
            self.assertEqual((o.wide8_mode, o.wide8_stages), (1, 2))
            os.environ["INKLING_DRAFT_WIDE8_MODE"] = "3"
            with self.assertRaises(_opts.OptionError):
                _opts.parse("wide8", environ={})
        finally:
            os.environ.clear()
            os.environ.update(old)

    def test_ops_wiring(self):
        ops.configure(_opts.parse("head8", environ={}))
        self.assertEqual(ops._wide_kw(192), {})
        ops.configure(_opts.parse("head8,wide8", environ={}))
        self.assertEqual(ops._wide_kw(64), {})  # one row block: the default launch, unchanged
        self.assertEqual(ops._wide_kw(192), {"rows": 1, "num_stages": 3, "bn": 64, "num_warps": 8})
        ops.configure(_opts.parse("head8", environ={}))


class RowModes(unittest.TestCase):
    """_triton._rows without importing Triton (a stand-in module provides cdiv)."""

    def setUp(self):
        sys.modules.setdefault("triton", types.SimpleNamespace(cdiv=lambda a, b: -(-a // b), jit=lambda f: f))

    def test_rows(self):
        try:
            from inkling_draftbytes import _triton
        except Exception as exc:  # noqa: BLE001  (real Triton absent or partial: skip)
            self.skipTest(f"triton not importable on CPU: {exc!r}")
        self.assertEqual(_triton._rows(8, 16, 2), (0, 1, 1))
        self.assertEqual(_triton._rows(64, 64, 2), (0, 1, 1))
        self.assertEqual(_triton._rows(192, 64, 0), (0, 1, 3))  # default: one launch row per block
        self.assertEqual(_triton._rows(192, 64, 2), (2, 3, 1))
        self.assertEqual(_triton._rows(384, 64, 2), (2, 6, 1))
        self.assertEqual(_triton._rows(448, 64, 2), (1, 1, 1))  # more than 6 blocks: mode 1
        self.assertEqual(_triton._rows(192, 64, 1), (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
