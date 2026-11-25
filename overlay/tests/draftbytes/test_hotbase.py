"""hotbase: the compact base-logits window opens only inside the captured sampler's stock walk."""
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch  # noqa: F401
    from inkling_draftbytes import _opts, patches
except ImportError:  # pragma: no cover
    patches = None


@unittest.skipIf(patches is None, "needs torch")
class HotBase(unittest.TestCase):
    def test_option(self):
        with self.assertRaises(_opts.OptionError):
            _opts.parse("hotbase", environ={})
        o = _opts.parse("hotvocab:16,hotbase", environ={"INKLING_DRAFT_HOTVOCAB_FILE": "/x.json"})
        self.assertTrue(o.hotbase)
        self.assertIn("hotbase", o.describe())

    def test_window(self):
        seen = []

        class Sampler:
            _markov_walker = None

            def _markov_block(self, model_hidden, sample_hidden, anchor, bs):
                seen.append(patches.STATE.compact_base)
                return "ok"

        mod = types.SimpleNamespace(DsparkDraftSampler=Sampler)
        patches.patch_draft_sampler(mod)
        patches.patch_draft_sampler(mod)  # idempotent
        s = Sampler()
        self.assertEqual(s._markov_block(None, None, None, 8), "ok")
        s._markov_walker = types.SimpleNamespace(supports=lambda bs: bs <= 4)
        s._markov_block(None, None, None, 4)   # int8 walker serves: full-width base
        s._markov_block(None, None, None, 8)   # walker does not support the bucket: stock walk, compact
        self.assertEqual(seen, [1, 0, 1])
        self.assertEqual(patches.STATE.compact_base, 0)


if __name__ == "__main__":
    unittest.main()
