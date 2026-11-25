# Usage (inside the SGLang image, overlay on PYTHONPATH): INKLING_DRAFT_BYTES=...,tau:0.7 INKLING_VERIFY_CAP=policy:... python3 tools/gpu_check_sampled_import.py
"""Run inside the SGLang image (no server), with PYTHONPATH=/work/overlay so sitecustomize arms the
overlays from the env. Checks the tau patch (and, with --bv, the vcap block-verify patch) against
the real engine modules. No model is loaded."""
import json
import sys
import types

import torch

out = {}
from sglang.srt.speculative.dspark_components import dspark_draft as dd  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_draft_sampler as ds  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_sampling_accept_graph as sag  # noqa: E402
from sglang.srt.speculative.dspark_components import dspark_verify as dv  # noqa: E402

try:
    from inkling_draftbytes import patches
    out["draftbytes_patched"] = patches.STATE.patched
    out["opts"] = patches.STATE.opts.describe() if patches.STATE.opts else None
except Exception as e:  # noqa: BLE001
    out["draftbytes"] = repr(e)
out["sampler_same_fn"] = ds.resolve_draft_params is dd.resolve_draft_params
si = types.SimpleNamespace(temperatures=torch.tensor([0.8, 0.0, 1.0], device="cuda"),
                           top_ks=torch.tensor([1 << 30, 1, 20], device="cuda"))
p = dd.resolve_draft_params(sampling_info=si, bs=3, device=torch.device("cuda"))
out["temps"] = p.temperatures.tolist()
out["step"] = p.step_temperatures.tolist()
out["sag_init"] = getattr(sag.DsparkSamplingAcceptGraph.__init__, "__module__", "?")
out["verify_draft_probs"] = getattr(dv._draft_probs, "__module__", "?")
out["verify_accept"] = getattr(dv.accept_draft_tokens, "__module__", "?")
print(json.dumps(out))
