"""CPU tests of inkling_attn's install logic (no torch, no GPU).

    python3 -m unittest discover -s inkling_attn/tests     (from the overlay directory)
    INKLING_ATTN_SGLANG_ROOT=/path/to/sglang additionally checks the patch targets of that source tree.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import inkling_attn  # noqa: E402


class ParseLevers(unittest.TestCase):
    def test_values(self):
        self.assertEqual(inkling_attn.parse_levers(""), ())
        self.assertEqual(inkling_attn.parse_levers("0"), ())
        self.assertEqual(inkling_attn.parse_levers("prefill_swa"), ("prefill_swa",))
        self.assertEqual(inkling_attn.parse_levers("all"), ("prefill_swa", "prefill_global"))
        self.assertEqual(inkling_attn.parse_levers(" prefill_global , prefill_swa,prefill_swa"),
                         ("prefill_global", "prefill_swa"))
        with self.assertRaises(RuntimeError):
            inkling_attn.parse_levers("prefill_swa,decode")


class SourceChecks(unittest.TestCase):
    def _tree(self, drop=None):
        root = tempfile.mkdtemp()
        for rel, needles in inkling_attn._SOURCE_CHECKS:
            path = os.path.join(root, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                for n in needles:
                    if n != drop:
                        f.write(n + "\n")
        return root

    def test_complete_tree_passes(self):
        inkling_attn.check_sources(self._tree())

    def test_missing_needle_fails(self):
        needle = inkling_attn._SOURCE_CHECKS[0][1][2]
        with self.assertRaises(RuntimeError):
            inkling_attn.check_sources(self._tree(drop=needle))

    @unittest.skipUnless(os.environ.get("INKLING_ATTN_SGLANG_ROOT"), "no sglang tree given")
    def test_real_tree(self):
        inkling_attn.check_sources(os.environ["INKLING_ATTN_SGLANG_ROOT"])


if __name__ == "__main__":
    unittest.main()
