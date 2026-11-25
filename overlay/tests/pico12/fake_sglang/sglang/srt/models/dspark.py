"""Stand-in of sglang/srt/models/dspark.py."""
from types import SimpleNamespace

from sglang.srt.layers.quantization.unquant import ref_linear

CALLS = {"project": 0}
SPEC = SimpleNamespace(speculative_dspark_lm_head_dtype=None)


def get_spec():
    return SPEC


def project_through_lm_head(hidden, lm_head):
    CALLS["project"] += 1
    weight = lm_head.weight
    return ref_linear(hidden.to(weight.dtype), weight, weight.dtype).view(*hidden.shape[:-1], weight.shape[0])
