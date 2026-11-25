"""CPU stand-in of the built inkling_wide extension: the call contract of wide_ext.cu's gemm() (rows, strides,
regions, split-K workspace) with reference math (BF16 operands, FP32 sums, one rounding), and a call log.

fmt 2 (bf16x) reads the weight through inkling_pico12's CPU stand-in buffer (the weight's own bytes, then pad)."""
import torch

NO_REGION = 2**31 - 1


class FakeWideExt:
    max_rows = 256
    block_n = 128

    def __init__(self):
        self.calls = []  # (M, N, K, fmt, out dtype, S, stacked)

    @staticmethod
    def bucket(m):
        need = (m + 15) // 16
        for b in (1, 2, 3, 4, 6, 8, 12, 16):
            if b >= need:
                return b
        return -1

    def prepare(self, device=0):
        pass

    @staticmethod
    def weight_of(W, fmt, n, k):
        if fmt == 0:
            assert W.dtype == torch.bfloat16 and tuple(W.shape) == (n, k) and W.is_contiguous()
            return W
        assert fmt == 2 and W.dtype == torch.uint8 and W.dim() == 1
        return W[: n * k * 2].view(torch.bfloat16).view(n, k)

    def gemm(self, x0, x1, x2, M, W, fmt, N, K, rb1, rb2, rb3, roff1, roff2, out, S, ws, cnt, pdl, mt, mg, pfd):
        assert mt in (0, 1, 2) and mg in (0, 1, 2) and 0 <= pfd <= 64
        assert 1 <= M <= 256, M
        assert N % 8 == 0 and K % 64 == 0
        for x in (x0, x1, x2):
            assert x.dtype == torch.bfloat16 and x.dim() == 2 and x.shape[0] >= M and x.shape[1] == K
            assert x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
        assert x0.stride(0) == x1.stride(0) == x2.stride(0)
        assert out.dim() == 2 and out.shape[1] == N and out.stride(1) == 1 and out.stride(0) >= N
        assert out.dtype in (torch.float32, torch.bfloat16)
        assert out.shape[0] >= M + max(0, roff1, roff2)
        for b in (rb1, rb2, rb3):
            assert b >= N or b % 128 == 0, b
        assert rb1 <= rb2 <= rb3
        assert 1 <= S <= K // 64
        if S > 1:
            assert ws.dtype == torch.float32 and ws.numel() >= S * M * N
            nblk = -(-N // 128)
            assert cnt.dtype == torch.int32 and cnt.numel() >= nblk and int(cnt[:nblk].abs().sum()) == 0
        w = self.weight_of(W, fmt, N, K).float()
        for c0 in range(0, N, 128):
            src = 1 if rb1 <= c0 < rb2 else 2 if rb2 <= c0 < rb3 else 0
            x = (x0, x1, x2)[src][:M].float()
            r0 = (0, roff1, roff2)[src]
            c1 = min(N, c0 + 128)
            out[r0 : r0 + M, c0:c1] = (x @ w[c0:c1].T).to(out.dtype)
        self.calls.append((M, N, K, fmt, out.dtype, S, rb1 < N))

    def reset(self):
        self.calls.clear()
