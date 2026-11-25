"""inkling_cert under tensor parallelism (INKLING_CERT_TP=1): the certified head on each rank's vocab shard.

At TP the LM head is vocab-parallel: rank r holds ids [r * n, (r + 1) * n) (n = 50,256 at TP4; the real vocabulary
ends at 200,058, so the last shard has 49,290 real rows). The stock path computes every rank's shard (pico's GEMV),
softcaps it, all-gathers the full [M, 201,024] logits and masks the padded ids; the DFLASH accept takes the argmax.

Certified (cert_ext.cu screen_tp / select_tp, ops.Cert.run_tp), per captured target-verify head call of 1..16 rows:
  1. every rank screens its shard and sends (T_r, H_r) = (max lo, max hi) over its real ids: one [M, 2] all-gather
  2. every rank takes T = max_r T_r, H = max_r H_r (the same bits everywhere) and computes the tiles of its shard
     with hi >= T - delta(T, H); every other id gets -inf. This is the TP1 certificate with the global T: the
     stock winner's tile is computed on the rank that holds it, and no -inf beats it after the softcap.
  3. the shard then continues down the stock path unchanged. With SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1 the target
     verify keeps its shard and a greedy step resolves it with the stock deferred_argmax (local (max, id) and an
     [M, 2] all-gather) instead of the full logits gather; deferred_local_argmax and finish_deferred_gather are
     limited to the real vocabulary here (stock uses the padded size there, unlike the non-deferred path's mask).
     Without the deferral the stock in-graph gather publishes the certified logits and the argmax is the same.

Rank invariance: the collectives (the (T, H) all-gather, plus a second [M, 4] one in check mode) are owed by every
captured target-verify call of 1..16 rows once TP mode is armed. The decision is made in
Inkling2LogitsProcessor._compute_lm_head (make_compute_lm_head), which every rank enters before SGLang picks a head
kernel; if the certified head did not issue them (our head function not reached, e.g. sm12x_fp32_lm_head_applies
false -> torch.mm, or not eligible, or disabled), they are issued there with +inf, which makes every rank fall back
for that step. Arming is agreed at load by certtp.agree (one MIN all-reduce over the TP group's CPU group), which
every rank reaches whatever its local outcome (an observer flag, a refused layout, a failed build or self-test vote
no instead of returning early or exiting): all ranks arm only if all prepared, with the same head mode (1 / check)
and the same deferral setting. The device flag `on` comes from batch metadata (dflash.py), identical on every rank;
a rank with on = 0 sends +inf too. No rank-local outcome changes the number or order of collectives. (A rank whose
INKLING_CERT_HEAD / INKLING_CERT_TP env is unset never loads this package and never votes: the launch passes the same
INKLING_* environment to every rank.)
"""

import functools

import torch

LOGITS = "sglang.srt.layers.logits_processor"

# The stock deferral methods we replace (patch_deferral; matched in the image's logits_processor.py)
DEFER_NEEDLES = (
    "def deferred_argmax(self, logits: torch.Tensor) -> torch.Tensor:",
    "local = self.deferred_local_argmax(logits)",
    "group.all_gather_into_tensor(gathered.view(-1, 2), local)",
    "def deferred_local_argmax(self, logits: torch.Tensor) -> torch.Tensor:",
    "col_offset = group.rank_in_group * shard_width",
    "valid_width = max(0, min(shard_width, self.vocab_size - col_offset))",
    "logits, self.final_logit_softcapping, valid_width, col_offset",
    "def finish_deferred_gather(self, logits: torch.Tensor) -> torch.Tensor:",
    "return self._logits_gatherer(logits)[:, : self.vocab_size].float()",
)


def tp_group():
    from sglang.srt.runtime_context import get_parallel

    return get_parallel().tp_group


def shard_info(lm_head, lp, weight, tp, group):
    """(info, None) for a head this package can certify per shard, else (None, why). Every check depends on the
    configuration and the shard layout only, the same on every rank."""
    if group is None or int(getattr(group, "world_size", 0)) != tp:
        return None, f"TP group size {getattr(group, 'world_size', None)} != {tp}"
    rank = int(group.rank_in_group)
    if lp is None or not getattr(lp, "use_fp32_lm_head", False):
        return None, "the logits processor has no FP32 head"
    for name, want in (("do_tensor_parallel_all_gather", True), ("do_tensor_parallel_all_gather_dp_attn", False),
                       ("use_attn_tp_group", False), ("use_tp_lm_head_all_to_all", False)):
        if bool(getattr(lp, name, not want)) != want:
            return None, f"logits processor {name} is not {want}"
    if getattr(lp, "final_logit_softcapping", None) or getattr(lp, "logit_scale", None) is not None:
        return None, "logits processor softcap / logit_scale not as assumed"
    if not isinstance(weight, torch.Tensor) or weight.dim() != 2:
        return None, "no head weight"
    n = int(weight.shape[0])
    padded = int(getattr(lp, "vocab_size", 0) or 0)
    real = int(getattr(getattr(lp, "config", None), "vocab_size", 0) or 0)
    if n % 16 or n * tp != padded:
        return None, f"shard rows {n} x {tp} != padded vocab {padded} (or not a multiple of 16)"
    if getattr(lm_head, "tp_size", tp) != tp:
        return None, f"head tp_size {getattr(lm_head, 'tp_size', None)} != {tp}"
    idx = getattr(lm_head, "shard_indices", None)
    if idx is not None:
        if (int(idx.org_vocab_start_index) != rank * n or int(idx.num_org_elements) != n
                or int(getattr(idx, "num_added_elements_padded", 0)) != 0):
            return None, "vocab shard layout is not rank * rows (padding or added vocab)"
    if not (tp - 1) * n < real <= padded:
        return None, f"real vocab {real} leaves a shard without real ids"
    offset = rank * n
    return {"world": tp, "rank": rank, "n": n, "offset": offset, "v_real": min(n, real - offset), "real": real}, None


MODES = {"1": 1, "check": 2}


def agree(group, ok, mode, defer):
    """Load-time agreement over the TP group's CPU (gloo) group, reached by every rank whatever its local outcome:
    True iff every rank prepared its shard (ok), and every rank has the same head mode (1 / check: different numbers
    of collectives per step) and the same deferral setting. One MIN all-reduce of
    [ok, mode, -mode, defer, -defer]: min(mode) == max(mode) iff -min(-mode) == min(mode)."""
    import torch.distributed as dist

    t = torch.tensor([1 if ok else 0, int(mode), -int(mode), int(defer), -int(defer)], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group.cpu_group)
    v = [int(a) for a in t.tolist()]
    same_mode, same_defer = v[1] == -v[2], v[3] == -v[4]
    why = None
    if v[0] != 1:
        why = "a rank could not prepare its shard"
    elif not same_mode:
        why = "ranks differ in INKLING_CERT_HEAD (1 / check)"
    elif not same_defer:
        why = "ranks differ in SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER"
    return why is None, why


def first_nan(local, logits, valid_width, col_offset):
    """Replace a row's (max, id) by (NaN, its first NaN id) when the row's first valid_width logits hold a NaN: the
    gathered-row torch.argmax of the stock path returns the first NaN (softcap_row_argmax skips NaNs)."""
    if valid_width <= 0:
        return local
    nan = torch.isnan(logits[:, :valid_width])
    has = nan.any(dim=1, keepdim=True)
    idx = nan.to(torch.uint8).argmax(dim=1)  # the first maximal entry: the first NaN
    fix = torch.stack([torch.full_like(local[:, 0], float("nan")), (idx + col_offset).to(local.dtype)], dim=1)
    return torch.where(has, fix, local)


def merge_shards(gathered):
    """[world, M, 2] rank-major (max, id) -> [M] int64 ids: the lowest rank holding a NaN, else the lowest rank holding
    the largest maximum (ranks hold increasing id ranges): torch.argmax over the gathered row, NaNs included. Both
    argmaxes below are over values without NaN ties to break (uint8 flags, NaN-free maxima)."""
    v = gathered[..., 0]
    nan = torch.isnan(v)
    first_nan_rank = nan.to(torch.uint8).argmax(dim=0)
    best_value_rank = torch.where(nan, torch.full_like(v, float("-inf")), v).argmax(dim=0)
    best = torch.where(nan.any(dim=0), first_nan_rank, best_value_rank)
    return gathered[..., 1].gather(0, best.unsqueeze(0)).squeeze(0).long()


def patch_deferral(lp_cls, lp_mod, log):
    """The stock deferral (SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1) on Inkling2LogitsProcessor, made to return what the
    non-deferred path returns:
    - deferred_local_argmax / finish_deferred_gather over the real vocabulary (config.vocab_size): the non-deferred
      path masks ids >= 200,058 after the gather (Inkling2LogitsProcessor._get_logits); the deferred one ran over the
      padded 201,024 (its mask lands on the shard, where it is empty);
    - NaN: the non-deferred accept is torch.argmax over the gathered row (the first NaN wins); the local maximum
      reports a row's first NaN, and deferred_argmax merges NaNs first, then values, lowest rank on ties.
    Inert unless the DFLASH worker defers the verify gather."""
    if getattr(lp_cls.__dict__.get("deferred_local_argmax"), "_inkling_cert", False):
        return False
    finish = lp_cls.finish_deferred_gather
    argmax_fn = lp_mod.softcap_row_argmax
    get_parallel = lp_mod.get_parallel

    def deferred_local_argmax(self, logits):
        real = min(int(self.config.vocab_size), int(self.vocab_size))
        group = get_parallel().tp_group
        shard_width = logits.shape[1]
        col_offset = group.rank_in_group * shard_width
        valid_width = max(0, min(shard_width, real - col_offset))
        local = argmax_fn(logits, self.final_logit_softcapping, valid_width, col_offset)
        return first_nan(local, logits, valid_width, col_offset)

    def deferred_argmax(self, logits):
        group = get_parallel().tp_group
        local = self.deferred_local_argmax(logits)
        gathered = torch.empty((group.world_size, logits.shape[0], 2), dtype=local.dtype, device=local.device)
        group.all_gather_into_tensor(gathered.view(-1, 2), local)
        return merge_shards(gathered)

    @functools.wraps(finish)
    def finish_deferred_gather(self, logits):
        out = finish(self, logits)
        real = int(self.config.vocab_size)
        if out.shape[-1] > real:
            out[..., real:] = float("-inf")
        return out

    for f in (deferred_local_argmax, deferred_argmax, finish_deferred_gather):
        f._inkling_cert = True
    lp_cls.deferred_local_argmax = deferred_local_argmax
    lp_cls.deferred_argmax = deferred_argmax
    lp_cls.finish_deferred_gather = finish_deferred_gather
    log("Inkling2LogitsProcessor deferral: real vocabulary, NaN-first merge")
    return True
