"""inkling_dfbytes patches: the DFlash2 drafter's layers (mx8) and its candidate projection (hot8 / hot16).

Patch points (draft side only):
  sglang.srt.models.dflash
    DFlashDraftModel.load_weights        -> mx8: MXFP8 copies of the decoder layers' BF16 linears, their
                                            quant_method swapped for inkling_draftbytes' W8 method (instances)
    DFlash2DraftModel.compute_candidates -> hot8 / hot16: top-k over the hot rows of the target head (TP1); under
                                            TP each rank's top-k over the hot rows of its vocab shard, then the
                                            stock K-wide candidate gather and global top-k
  sglang.srt.speculative.dflash_worker_v2
    DFlashWorkerV2._maybe_build_draft_sampler -> build the hot plane from the target head before the draft
                                                 graph is captured
"""

import bisect
import functools
import os
import sys

import torch

from . import STATE, DfbytesFatal, _log

MARK = "_inkling_dfbytes"


def _require(owner, names, where):
    missing = [n for n in names if not hasattr(owner, n)]
    if missing:
        raise DfbytesFatal(f"inkling_dfbytes: {where} lacks {missing}")


def _draftbytes():
    from inkling_draftbytes import _opts, ops, patches as dbp, quant, w8gemv

    return _opts, ops, dbp, quant, w8gemv


def configure_draftbytes(opts):
    """inkling_draftbytes' kernels read their settings from its STATE.opts. Under SPEC=dflash2 that package is not
    installed (no DSpark draft); configure it here for the W8 kernels only (no DSpark patch is armed)."""
    _opts, ops, dbp, _, _ = _draftbytes()
    if dbp.STATE.opts is not None:
        raise DfbytesFatal("inkling_dfbytes: inkling_draftbytes is armed (DSpark draft); the two are exclusive")
    text = "draftmx8,wide8,w8tile,w8gemv" if "mx8" in opts else "head8,wide8"
    db_opts = _opts.parse(text, environ=os.environ)
    if "mx8" in opts and not os.environ.get("INKLING_DRAFT_W8_MAX_M", "").strip():
        # GB10 bench 2026-11-24 (tools/bench_dfbytes.py, real drafter weights, cold L2): the 20 drafter linears at
        # 8 rows 3.17 -> 2.04 ms (w8gemv); at 56 rows the Triton W8 path is slower than stock (3.47 vs 3.32 ms)
        import dataclasses

        db_opts = dataclasses.replace(db_opts, w8_max_m=16)
    dbp.STATE.opts = db_opts
    ops.configure(db_opts)
    STATE["db_opts"] = db_opts


class HotPlane:
    def __init__(self, ids64, pack, head_ptr, n_real=None, pad=0):
        self.ids64 = ids64  # global token id of each plane row (TP: pad rows carry a sentinel id)
        self.pack = pack
        self.n = int(ids64.numel())
        self.head_ptr = head_ptr
        self.n_real = self.n if n_real is None else int(n_real)  # rows [n_real, n) are pad rows (TP only)
        self.pad = int(pad)


# ------------------------------------------------------------------ hot plane under TP (vocab-sharded target head)
_SHARD_ATTRS = ("num_embeddings_padded", "org_vocab_size_padded", "num_embeddings", "org_vocab_size")


def shard_ranges(lm_head, limit):
    """Every rank's real hot-eligible ids [start, end) (end clamped to `limit`, as tp4_dflash2_prepatch clamps the
    stock projection) and this rank's index in that list. Every rank derives the same list from the head's
    configuration (VocabParallelEmbedding._get_indices for each rank): no collective, no rank-local input."""
    tp = int(lm_head.tp_size)
    get = getattr(lm_head, "_get_indices", None)
    si = getattr(lm_head, "shard_indices", None)
    if get is None or si is None or any(getattr(lm_head, n, None) is None for n in _SHARD_ATTRS):
        raise DfbytesFatal("inkling_dfbytes: a hot plane under TP needs a VocabParallelEmbedding lm_head")
    dims = [int(getattr(lm_head, n)) for n in _SHARD_ATTRS]
    ranges = []
    for r in range(tp):
        idx = get(*dims, r, tp)
        start = int(idx.org_vocab_start_index)
        ranges.append((start, max(start, min(int(idx.org_vocab_end_index), limit))))
    start = int(si.org_vocab_start_index)
    own = (start, max(start, min(start + int(si.num_org_elements), limit)))
    if own not in ranges:
        raise DfbytesFatal(f"inkling_dfbytes: own vocab shard {own} is not one of the derived shards {ranges}")
    return ranges, ranges.index(own)


def split_hot(ids, ranges, rank, top_k):
    """ids: the TP1 hot set, ascending. -> (this rank's hot ids, per-rank counts, pad).

    The K-wide candidate gather needs exactly top_k candidates from every rank. If any shard holds fewer than top_k
    hot ids, EVERY rank appends top_k pad rows to its plane (decided from all ranks' counts, which every rank
    computes identically), whose logits are set to -inf: they never beat a real candidate, and the ranks together
    always hold >= top_k real ones (sum_r min(top_k, n_r) >= top_k because sum_r n_r = len(ids) >= top_k)."""
    counts = [bisect.bisect_left(ids, e) - bisect.bisect_left(ids, s) for s, e in ranges]
    if sum(counts) != len(ids):
        raise DfbytesFatal(f"inkling_dfbytes: vocab shards {ranges} cover {sum(counts)} of {len(ids)} hot ids")
    pad = top_k if min(counts) < top_k else 0
    s, e = ranges[rank]
    return ids[bisect.bisect_left(ids, s):bisect.bisect_left(ids, e)], counts, pad


def _build_hot_tp(lm_head, weight, ids, limit, top_k, quant, mode):
    ranges, rank = shard_ranges(lm_head, limit)
    mine, counts, pad = split_hot(ids, ranges, rank, top_k)
    if pad:
        # every rank sees the same counts (computed from the head's config, no collective), so every rank refuses
        # alike: a shard with fewer than top_k hot rows would need sentinel rows that can tie with real -inf scores
        raise DfbytesFatal(f"inkling_dfbytes: TP hot plane needs >= {top_k} hot ids per shard, got {counts}; "
                           f"raise INKLING_DFLASH_HOT_N or drop hot8")
    start = ranges[rank][0]
    if weight.shape[0] < ranges[rank][1] - start:
        raise DfbytesFatal(f"inkling_dfbytes: head shard has {weight.shape[0]} rows < {ranges[rank]}")
    local = torch.tensor([t - start for t in mine], dtype=torch.int64, device=weight.device)
    rows = weight.index_select(0, local)  # the same rows (same bytes) as the TP1 plane's for these ids
    if pad:
        rows = torch.cat([rows, rows.new_zeros((pad, rows.shape[1]))])
    # pad rows: sentinel id = the smallest hot id (a real token id, so even an impossible leak stays a valid index);
    # their -inf logits are never selected
    ids64 = torch.tensor(mine + [ids[0]] * pad, dtype=torch.int64, device=weight.device)
    if "hot16" in STATE["opts"]:
        pack = quant.bf16_pack(rows.contiguous())
    else:
        pack = quant.quantize_mode(rows, mode)  # per-row scales: each row's bytes equal its TP1 bytes
    del rows
    hot = HotPlane(ids64, pack, weight.data_ptr(), n_real=len(mine), pad=pad)
    STATE["hot_built"] += 1
    STATE["hot_tp"] = {"tp": len(ranges), "rank": rank, "ranges": ranges, "counts": counts, "pad": pad}
    _log(f"hot plane TP{len(ranges)} shard {rank} ids [{ranges[rank][0]}, {ranges[rank][1]}): {len(mine)} of "
         f"{len(ids)} hot ids (per shard {counts}{f', +{pad} pad rows on every rank' if pad else ''}), "
         f"{'BF16' if 'hot16' in STATE['opts'] else 'FP8 row'} {pack.nbytes() / 1e6:.1f} MB")
    return hot


def build_hot(model, lm_head):
    _opts, ops, dbp, quant, _ = _draftbytes()
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2 or weight.dtype != torch.bfloat16:
        raise DfbytesFatal("inkling_dfbytes: hot plane needs the target's dense BF16 lm_head weight")
    limit = int(getattr(lm_head, "org_vocab_size", weight.shape[0]))
    valid = getattr(model, "valid_vocab_size", None)
    if valid is not None:
        limit = min(limit, int(valid))
    ids = dbp.load_hot_ids(STATE["hot_file"], STATE["hot_n"], limit)
    top_k = int(getattr(getattr(model, "candidate_selector", None), "top_k", 16) or 16)
    if len(ids) < max(1024, top_k):
        raise DfbytesFatal(f"inkling_dfbytes: only {len(ids)} usable hot ids in {STATE['hot_file']}")
    if int(getattr(lm_head, "tp_size", 1)) != 1:
        return _build_hot_tp(lm_head, weight, ids, limit, top_k, quant, _opts.parse_mode("row", "row"))
    ids64 = torch.tensor(ids, dtype=torch.int64, device=weight.device)
    rows = weight.index_select(0, ids64)
    if "hot16" in STATE["opts"]:
        pack = quant.bf16_pack(rows.contiguous())
    else:
        pack = quant.quantize_mode(rows, _opts.parse_mode("row", "row"))
    del rows
    hot = HotPlane(ids64, pack, weight.data_ptr())
    STATE["hot_built"] += 1
    _log(f"hot plane: {hot.n} of {limit} ids, {'BF16' if 'hot16' in STATE['opts'] else 'FP8 row'} "
         f"{pack.nbytes() / 1e6:.1f} MB")
    return hot


def hot_logits(h, pack):
    """[N, n_hot] BF16 logits of the hot plane. A BF16 pack runs in <= 64-row launches: the multi-row-block wide
    mode with a BF16 weight needs more shared memory than GB10 has (as in inkling_draftbytes.rnnhoist); rows are
    independent, so the chunking changes nothing numerically."""
    _, ops, _, _, _ = _draftbytes()
    if pack.scale_kind != 0 or h.shape[0] <= 64:
        return ops.w8_linear(h, pack, None, torch.bfloat16)
    out = torch.empty((h.shape[0], pack.n), dtype=torch.bfloat16, device=h.device)
    for i in range(0, h.shape[0], 64):
        out[i:i + 64] = ops.w8_linear(h[i:i + 64], pack, None, torch.bfloat16)
    return out


def hot_local_topk(hidden, hot, k, radix_topk):
    """TP: this rank's top-k over its hot rows -> vals [N, k] (BF16, as the stock local top-k), global ids [N, k]."""
    logits = hot_logits(hidden.to(torch.bfloat16), hot.pack)
    if hot.pad:
        logits[:, hot.n_real:] = float("-inf")
    vals, idx = radix_topk(logits, k)
    return vals, hot.ids64.index_select(0, idx.reshape(-1).long()).view(idx.shape)


def gather_topk(mod, group, vals, global_ids, k):
    """The stock candidate_topk tail (sglang.srt.models.dflash): the same collectives in the same order -- the fused
    candidate all-gather when the image arms it, else NCCL all-gathers of the fp32 vals and the int64 ids -- then the
    global top-k over the tp * k gathered candidates."""
    fused = getattr(mod, "_fused_candidate_all_gather", None)
    gathered = fused(group, vals, global_ids) if fused is not None else None
    if gathered is not None:
        gathered_vals, gathered_ids = gathered
    else:
        gathered_vals = group.all_gather(vals.float(), dim=-1)
        gathered_ids = group.all_gather(global_ids, dim=-1)
    top_vals, sel = torch.topk(gathered_vals, k, dim=-1)
    return torch.gather(gathered_ids, -1, sel).long(), top_vals


def _tp_head_ok(mod, lm_head):
    """The stock path refuses a head sharded over a group other than the draft's TP group: leave that to it."""
    get_parallel = getattr(mod, "get_parallel", None)
    if getattr(lm_head, "tp_group", None) is None or getattr(lm_head, "shard_indices", None) is None:
        return False
    return get_parallel is None or int(lm_head.tp_size) == int(get_parallel().tp_size)


def _ensure_hot(model, lm_head):
    hot = getattr(model, "_idf_hot", None)
    weight = getattr(lm_head, "weight", None)
    if hot is not None and isinstance(weight, torch.Tensor) and hot.head_ptr == weight.data_ptr():
        return hot
    if hot is not None:
        # Captured draft graphs read the existing plane: a moved target head (online weight update) is unsupported.
        raise DfbytesFatal("inkling_dfbytes: the target lm_head moved after the hot plane was built")
    if torch.cuda.is_current_stream_capturing():
        raise DfbytesFatal("inkling_dfbytes: hot plane missing at capture (built before the draft graph)")
    model._idf_hot = build_hot(model, lm_head)
    return model._idf_hot


def prepare_layers(model):
    _, _, dbp, _, _ = _draftbytes()
    size = dbp._prepare_draftmx8(model, refresh=False)
    linears = dbp._draft_linears(model)
    STATE["prepared"] += 1
    shapes = sorted({tuple(m.weight.shape) for _, m in linears})
    _log(f"mx8: {len(linears)} drafter linears {shapes} -> {size / 1e6:.1f} MB MXFP8")
    if "mx8w" in STATE["opts"]:
        prepare_mx8w(linears)


# ------------------------------------------------------------------ mx8w: stock b12x MXFP8 GEMM on the eligible linears
_MX8W_CLASSES = {}


def b12x_eligible(pack) -> bool:
    """b12x mm_mxfp8 needs K % 128 == 0 (one BK128 tile) and an MXFP8 pack (E8M0 per 32 inputs, FP8 e4m3)."""
    from inkling_draftbytes.quant import SCALE_E8M0

    return (pack.scale_kind == SCALE_E8M0 and pack.group == 32 and pack.k % 128 == 0 and pack.n % 128 == 0
            and pack.q.dtype == torch.float8_e4m3fn and pack.q.is_contiguous())


def b12x_linear(x2, q, sf):
    """[M, K] BF16 -> [M, N] BF16 via flashinfer.mxfp8_quantize + mm_mxfp8(backend="b12x"), default tile plan (no
    autotuning: a fixed tactic per shape)."""
    from flashinfer import mm_mxfp8, mxfp8_quantize
    from flashinfer.autotuner import autotune

    xq, xs = mxfp8_quantize(x2, is_sf_swizzled_layout=True)
    with autotune(False):
        return mm_mxfp8(xq, q.t(), xs, sf, out_dtype=torch.bfloat16, backend="b12x")


def _mx8w_method_class(cls):
    sub = _MX8W_CLASSES.get(cls)
    if sub is None:
        inner_apply = cls.apply

        def apply(self, layer, x, bias=None):
            sf = getattr(layer, "_idf_b12x_sf", None)
            pack = getattr(layer, "_idb_w8", None)
            # the stock MXFP8 quantizer needs a contiguous, 16-byte aligned [rows, K] input; anything else falls back
            if (sf is not None and pack is not None and bias is None and x.is_cuda and x.dtype == torch.bfloat16
                    and x.dim() >= 1 and x.shape[-1] == pack.k and x.is_contiguous() and x.data_ptr() % 16 == 0):
                rows = x.numel() // x.shape[-1] if x.shape[-1] else 0
                if 0 < rows <= STATE["mx8w_max_m"]:
                    STATE["mx8w_calls"] += 1
                    y = b12x_linear(x.reshape(rows, x.shape[-1]), pack.q, sf)
                    return y.view(*x.shape[:-1], pack.n)
            return inner_apply(self, layer, x, bias)

        sub = type(f"Dfbytes{cls.__name__}", (cls,), {"apply": apply})
        _MX8W_CLASSES[cls] = sub
    return sub


def prepare_mx8w(linears):
    import copy

    from flashinfer import block_scale_interleave

    n = 0
    extra = 0
    for _, module in linears:
        pack = getattr(module, "_idb_w8", None)
        if pack is None or not b12x_eligible(pack):
            continue
        sf = block_scale_interleave(pack.scale.contiguous())
        old = getattr(module, "_idf_b12x_sf", None)
        if old is not None and old.shape == sf.shape and old.dtype == sf.dtype and old.device == sf.device:
            old.copy_(sf)  # reload: keep the address captured draft graphs read (q is refreshed in place by mx8)
        else:
            module._idf_b12x_sf = sf
        extra += sf.numel()
        if not type(module.quant_method).__name__.startswith("Dfbytes"):
            method = copy.copy(module.quant_method)
            method.__class__ = _mx8w_method_class(type(module.quant_method))
            module.quant_method = method
        n += 1
    if n == 0:
        raise DfbytesFatal("inkling_dfbytes: mx8w found no b12x-eligible drafter linear")
    STATE["mx8w_linears"] = n
    _log(f"mx8w: {n} of {len(linears)} drafter linears on b12x MXFP8 for rows 1..{STATE['mx8w_max_m']} "
         f"(the rest keep mx8; +{extra / 1e6:.1f} MB swizzled scales)")


def patch_dflash(mod):
    _require(mod, ["DFlashDraftModel", "DFlash2DraftModel", "radix_topk"], "sglang.srt.models.dflash")
    base, sel = mod.DFlashDraftModel, mod.DFlash2DraftModel
    _require(base, ["load_weights"], "DFlashDraftModel")
    _require(sel, ["compute_candidates", "_transform_unary_logits"], "DFlash2DraftModel")
    opts = STATE["opts"]
    if "mx8" in opts and not getattr(base.load_weights, MARK, False):
        orig_load = base.load_weights

        @functools.wraps(orig_load)
        def load_weights(self, weights):
            out = orig_load(self, weights)
            prepare_layers(self)
            return out

        setattr(load_weights, MARK, True)
        base.load_weights = load_weights
    if any(p.startswith("hot") for p in opts) and not getattr(sel.compute_candidates, MARK, False):
        orig_cc = sel.compute_candidates
        radix_topk = mod.radix_topk

        @functools.wraps(orig_cc)
        def compute_candidates(self, hidden):
            # every test here is rank-invariant (same head configuration, shapes, env on every rank)
            lm_head = self.lm_head
            if (lm_head is None or hidden.dim() != 2 or not hidden.is_cuda
                    or getattr(self, "merge_candidates_fused", False)):
                return orig_cc(self, hidden)
            tp = int(getattr(lm_head, "tp_size", 1))
            if tp != 1 and not _tp_head_ok(mod, lm_head):
                return orig_cc(self, hidden)
            hot = _ensure_hot(self, lm_head)
            if tp != 1:
                k = self.candidate_selector.top_k
                vals, global_ids = hot_local_topk(hidden, hot, k, radix_topk)
                ids, top_vals = gather_topk(mod, lm_head.tp_group, vals, global_ids, k)
                return ids, self._transform_unary_logits(top_vals)
            logits = hot_logits(hidden.to(torch.bfloat16), hot.pack)
            vals, idx = radix_topk(logits, self.candidate_selector.top_k)
            ids = hot.ids64.index_select(0, idx.reshape(-1).long()).view(idx.shape)
            return ids, self._transform_unary_logits(vals)

        setattr(compute_candidates, MARK, True)
        sel.compute_candidates = compute_candidates
    return True


def patch_worker(mod):
    _require(mod, ["DFlashWorkerV2"], "sglang.srt.speculative.dflash_worker_v2")
    cls = mod.DFlashWorkerV2
    _require(cls, ["_maybe_build_draft_sampler"], "DFlashWorkerV2")
    if not any(p.startswith("hot") for p in STATE["opts"]) or getattr(cls._maybe_build_draft_sampler, MARK, False):
        return False
    orig = cls._maybe_build_draft_sampler
    unwrap = getattr(mod, "unwrap_lora_layer", None)

    @functools.wraps(orig)
    def _maybe_build_draft_sampler(self, *args, **kwargs):
        # Build the hot plane eagerly, before the draft graph that calls compute_candidates is captured.
        model = getattr(self, "draft_model", None)
        target = getattr(getattr(getattr(self, "_target_worker", None), "model_runner", None), "model", None)
        lm_head = getattr(target, "lm_head", None)
        if lm_head is not None and unwrap is not None:
            lm_head = unwrap(lm_head)  # as the worker does
        megakernel = getattr(mod, "megakernel_enabled", None)
        # Any TP size: under TP each rank builds the plane of its own vocab shard (no collective; rank-invariant gates).
        if (model is not None and getattr(model, "candidate_selector", None) is not None and lm_head is not None
                and not getattr(model, "merge_candidates_fused", False)
                and (int(getattr(lm_head, "tp_size", 1)) == 1 or _tp_head_ok(mod, lm_head))
                and not (callable(megakernel) and megakernel())):
            _ensure_hot(model, lm_head)
        return orig(self, *args, **kwargs)

    setattr(_maybe_build_draft_sampler, MARK, True)
    cls._maybe_build_draft_sampler = _maybe_build_draft_sampler
    return True
