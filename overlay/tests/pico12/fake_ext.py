"""CPU stand-in of the built inkling_pico12 extension: the same call contract (shapes, strides, row limit,
in-place outputs) with reference math (BF16 operands, FP32 sums, one rounding), and a log of the calls.

Its "coded" buffer is the weight's own bytes followed by `pad` bytes, so that a reload written in place into an
older buffer (ops.encode(..., old=...)) changes what later calls compute, as on the GPU."""
import torch


class FakeExt:
    max_rows = 16

    def __init__(self):
        self.calls = []  # ("gemv", M, N, K, out dtype)
        self.pad = 64  # tests raise it to make a reload need more bytes

    def encode(self, w):
        assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous()
        n, k = w.shape
        assert n % 8 == 0 and k % 32 == 0
        raw = w.reshape(-1).view(torch.uint8)
        buf = torch.cat([raw, torch.zeros(self.pad, dtype=torch.uint8)])
        tiles = (n // 8) * (k // 32)
        return buf, self.pad // 512, tiles

    @staticmethod
    def weight_of(buf, n, k):
        return buf[: n * k * 2].view(torch.bfloat16).view(n, k)

    def gemv(self, x, buf, n, k, out, mt, wk, pdl):
        m = x.shape[0]
        assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[1] == k
        assert x.stride(1) == 1 and (m == 1 or x.stride(0) == k), "rows of x must be contiguous"
        assert x.data_ptr() % 16 == 0, "x must be 16-byte aligned"
        assert 1 <= m <= 16, f"{m} rows"
        assert out.dim() == 2 and tuple(out.shape) == (m, n) and out.stride(1) == 1 and (m == 1 or out.stride(0) == n)
        assert out.dtype in (torch.float32, torch.bfloat16)
        assert mt in (1, 2) and n % (8 * mt) == 0 and 1 <= wk <= 16 and wk <= k // 32
        w = self.weight_of(buf, n, k)
        out.copy_((x.float() @ w.float().T).to(out.dtype))
        self.calls.append(("gemv", m, n, k, out.dtype))

    def reset(self):
        self.calls.clear()
