"""CPU stand-in for the FA backend surface dwin patches (written for the tests): FlashAttentionMetadata with the
fields forward_extend reads, and a forward_extend that runs paged bidirectional attention over
page_table / cache_seqlens_int32 of self.forward_metadata, like the image's TARGET_VERIFY path for a full layer."""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class FlashAttentionMetadata:
    cache_seqlens_int32: Optional[torch.Tensor] = None
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seq_len_q: int = 0
    page_table: Optional[torch.Tensor] = None
    swa_out_cache_loc: Optional[torch.Tensor] = None
    local_attn_metadata: Optional[object] = None


class ForwardMode:
    def __init__(self, name):
        self.name = name

    def is_target_verify(self):
        return self.name == "TARGET_VERIFY"

    def __repr__(self):
        return self.name


class ForwardBatch:
    def __init__(self, mode="TARGET_VERIFY"):
        self.forward_mode = ForwardMode(mode)


class RadixAttention:
    def __init__(self, layer_id, sliding_window_size=-1, window_anchor_at_start=False):
        self.layer_id = layer_id
        self.sliding_window_size = sliding_window_size
        self.window_anchor_at_start = window_anchor_at_start
        self.scaling = 0.25


class FlashAttentionBackend:
    def __init__(self, k_cache, v_cache, page_size, topk=1):
        self.k_cache, self.v_cache = k_cache, v_cache  # [pages, page, heads, dim]
        self.page_size = page_size
        self.topk = topk
        self.use_mla = False
        self.forward_metadata = None
        self.seen = []
        self.seen_max_k = []

    def forward_extend(self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs):
        if kwargs.get("boom"):
            raise RuntimeError("boom")
        m = self.forward_metadata
        self.seen.append(m)
        return self.flash_attn_with_kvcache(
            q=q, page_table=m.page_table, cache_seqlens=m.cache_seqlens_int32, softmax_scale=layer.scaling, ver=4
        )

    def flash_attn_with_kvcache(self, *, q, page_table, cache_seqlens, softmax_scale, ver, max_seqlen_k=None):
        self.seen_max_k.append(max_seqlen_k)
        P = self.page_size
        out = []
        qlen = q.shape[0] // page_table.shape[0]
        for r in range(page_table.shape[0]):
            n = int(cache_seqlens[r])
            idx = torch.arange(n)
            pages = page_table[r].long()[idx // P]
            kk = self.k_cache[pages, idx % P].float()
            vv = self.v_cache[pages, idx % P].float()
            qq = q[r * qlen:(r + 1) * qlen].float()
            s = torch.einsum("thd,nhd->thn", qq, kk) * softmax_scale
            out.append(torch.einsum("thn,nhd->thd", torch.softmax(s, -1), vv))
        return torch.cat(out)
