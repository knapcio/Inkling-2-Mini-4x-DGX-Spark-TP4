"""INKLING_ATTN_GATE vs SGLang kernel API logging: the gated epilogue leaves the FA4 FP32 out unwritten, and
SGLANG_KERNEL_API_LOGLEVEL > 0 logs / dumps that buffer, so the gate must be off (with a log line) whenever the
logging is on. CPU only: imports inkling_attn.patch (no torch / sglang at import) in a fresh interpreter."""
import os
import subprocess
import sys
import unittest

OVERLAY = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROBE = "import inkling_attn.patch as p; print(p.GATE, p.GATE_OFF_FOR_LOGGING)"


def probe(gate, loglevel):
    env = {k: v for k, v in os.environ.items() if k not in ("INKLING_ATTN_GATE", "SGLANG_KERNEL_API_LOGLEVEL")}
    env["PYTHONPATH"] = OVERLAY
    if gate is not None:
        env["INKLING_ATTN_GATE"] = gate
    if loglevel is not None:
        env["SGLANG_KERNEL_API_LOGLEVEL"] = loglevel
    r = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, timeout=60)
    if r.returncode != 0:
        raise AssertionError(r.stderr)
    return r.stdout.split(), r.stderr


class GateSwitch(unittest.TestCase):
    def test_gate_on_without_logging(self):
        for level in (None, "0", ""):
            out, err = probe("1", level)
            self.assertEqual(out, ["True", "False"], level)
            self.assertNotIn("gated epilogue OFF", err)

    def test_logging_turns_gate_off_with_a_line(self):
        for level in ("3", "10"):
            out, err = probe("1", level)
            self.assertEqual(out, ["False", "True"], level)
            self.assertIn("gated epilogue OFF: SGLANG_KERNEL_API_LOGLEVEL is set", err)

    def test_gate_off_stays_off(self):
        for level in (None, "3"):
            out, err = probe("0", level)
            self.assertEqual(out, ["False", "False"])
            self.assertNotIn("gated epilogue OFF", err)

    def test_unparsable_level_is_off_like_sglang(self):
        out, _ = probe("1", "abc")  # kernel_api_logging._get_env falls back to 0 on a parse error
        self.assertEqual(out, ["True", "False"])


if __name__ == "__main__":
    unittest.main()
