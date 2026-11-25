"""CPU stand-in: the GEMV only takes CUDA tensors, so on CPU it never applies."""

import torch


def use_sm12x_bf16_gemv(x, weight, bias=None):
    return bool(x.is_cuda) and x.numel() // x.shape[-1] <= 16


def sm12x_bf16_gemv(x, weight, bias=None, out_dtype=None, out=None):
    return torch.matmul(x, weight.T)
