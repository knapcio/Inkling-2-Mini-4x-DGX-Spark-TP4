"""Stand-in of sglang/srt/layers/quantization/unquant.py. Every "stock" path computes the same reference math
(BF16 operands, FP32 sums, one rounding) as the tests' fake pico extension, and counts its calls, so the tests
can check which path ran and compare results exactly."""
import torch

from sglang.srt.batch_invariant_ops import is_batch_invariant_mode_enabled  # noqa: F401

STOCK_GEMV = [True]  # stands in for "_sm12x_bf16_gemv is not None"
CALLS = {"apply": 0, "apply_into": 0, "lm_head": 0}


def ref_linear(x, weight, out_dtype):
    return (x.reshape(-1, x.shape[-1]).float() @ weight.float().T).to(out_dtype)


class UnquantizedLinearMethod(object):
    def apply(self, layer, x, bias=None):
        CALLS["apply"] += 1
        y = ref_linear(x, layer.weight, x.dtype)
        if bias is not None:
            y = y + bias
        return y.view(*x.shape[:-1], layer.weight.shape[0])

    def apply_into(self, layer, x, output, bias=None):
        CALLS["apply_into"] += 1
        output.copy_(ref_linear(x, layer.weight, output.dtype))
        return output


def sm12x_fp32_lm_head_applies(x, weight, out):
    return STOCK_GEMV[0] and x.numel() // x.shape[-1] <= 16


def sm12x_fp32_lm_head(x, weight, out):
    CALLS["lm_head"] += 1
    y = ref_linear(x, weight, torch.float32)
    if out is not None:
        out.copy_(y)
        return out
    return y
