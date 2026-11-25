#!/usr/bin/env python3
"""In the image, no model: arm inkling_pico12 (lmhead) and inkling_cert as sitecustomize would, import the patched
sglang modules, and check that every hook landed in the expected order. Last line: `cert smoke_hooks PASS|FAIL`."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
os.environ.setdefault("INKLING_PICO12", "dense,lmhead")
os.environ.setdefault("INKLING_CERT_HEAD", "1")

import inkling_pico12  # noqa: E402

inkling_pico12.install()
import inkling_cert  # noqa: E402

inkling_cert.install()

import importlib  # noqa: E402

unq = importlib.import_module("sglang.srt.layers.quantization.unquant")
mdl = importlib.import_module("sglang.srt.models.inkling2")
ver = importlib.import_module("sglang.srt.speculative.dspark_components.dspark_verify")
from inkling_cert import patches  # noqa: E402

ok = True
f = unq.sm12x_fp32_lm_head
checks = {
    "lm head: cert outermost": getattr(f, "_inkling_cert", False),
    "lm head: pico inside": getattr(getattr(f, "__wrapped__", None), "_pico12", False),
    "_get_logits": getattr(mdl.Inkling2LogitsProcessor._get_logits, "_inkling_cert", False),
    "post_load_weights": getattr(mdl.Inkling2ForConditionalGeneration.post_load_weights, "_inkling_cert", False),
    "run_non_compact": getattr(ver.TargetVerifyExecutor.run_non_compact, "_inkling_cert", False),
    "begin_static_step": getattr(ver.DsparkVerifyEpilogue.begin_static_step, "_inkling_cert", False),
}
for k, v in checks.items():
    print(("  ok   " if v else "  FAIL ") + k)
    ok &= bool(v)
print("patched:", patches.STATE["patched"])
print(f"cert smoke_hooks {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
