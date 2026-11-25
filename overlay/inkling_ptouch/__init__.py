"""Prompt-end checkpoint recency (INKLING_PROMPT_TOUCH=1): LRU order only, nothing the model computes changes.

A request's prompt-end checkpoint (the conv state at the end of its prompt and the sliding window behind it) gets its
LRU recency when it is inserted, at prefill time. The final insert of the finished request refreshes only the window
behind the OUTPUT end (SWA refreshes the end node's window ancestors), and the request's lock moves there. So once the
request finishes, its prompt-end window is the oldest evictable window in the tree, older than every stale chunk
window that later prefills left (their states are gone after the per-path cap; the windows can never serve a match).
Under window-pool pressure the prompt-end windows go first, in prefill order: the state survives without its window,
the next turn of that conversation (which shares exactly the prompt when the template drops the reasoning) falls
back to the root and re-prefills everything, and each such re-prefill evicts the next-oldest prompt-end window.

This module bumps the recency of the prompt path right after the final insert, with the cache's own touch_prefix
(the tree walk the scheduler uses for waiting requests: no allocation, no lock, no state slot), keyed by the prompt up
to the deepest depth the request published by the end of its prefill. Requests without such a depth (whole-prompt
cache hits, retracted since, streaming sessions, PD) are left alone. Scope: the hybrid window + state tree.

Tensor parallelism (TP4 recipe). Every TP rank's scheduler keeps its own radix tree and must make the same cache
decisions (SGLang marks checkpoint / insert / evict @rank_consensus; requests and every rank-local abort reach all
ranks through rank 0's broadcast in recv_requests; the LRU clock is a logical counter, not time). The touch is a
deterministic function of scheduler state that is identical on every rank: req.finished() / finished_reason (from
the same token ids and the broadcast aborts), origin_input_ids, retraction_count, session_id, req.kv.cache_inserted_len
(set by the consensus-checked insert), up_to (a consensus parameter of checkpoint) and the env threshold. It runs
inside _checkpoint_into_tree, i.e. at the same position of every rank's checkpoint sequence, and walks each rank's
tree with the same key, so the trees (and their logical clocks) stay identical. No timing, no device memory, nothing
from the tokenizer / detokenizer side (rank 0 only) is read. Two rank-local outcomes are closed off at TP > 1:
  - the on / inert decision at start-up is agreed: every UnifiedRadixCache.__init__ joins one MIN all-reduce of
    [armed, min, -min] over the TP group's CPU (gloo) group; all ranks arm or all go inert (same threshold required).
    sitecustomize installs the hook whatever the switch, so a switched-off rank or one with an invalid
    INKLING_PROMPT_TOUCH_MIN (bounded to [-1, 2^40] in the payload) still votes, and votes no;
  - a runtime failure in the overlay (pre-read or touch) would make one rank inert alone. At TP > 1 it raises
    PtouchFatal (a RuntimeError) after the original insert ran: SGLang's scheduler exits on it (SIGQUIT to its
    parent) and launch-tp4.sh's watch takes every rank down. Silent cache divergence is never an option.
With SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=1 every finished-request decision is also recorded through SGLang's
rank_consensus_checker.assert_same (rid, touched, depth), so a divergence stops the server. TP1 is unchanged.
"""

import os
import sys

from . import _hook

URC = "sglang.srt.mem_cache.unified_radix_cache"
SWA = "sglang.srt.mem_cache.unified_cache.components.swa"
STATE = {"calls": 0, "skipped": 0, "inert": "", "min": 4096, "tp": 1, "agreed": 0, "off": False, "installed": False}
MARK = "_inkling_prompt_ckpt"


GUARD = BaseException   # caught at every overlay boundary (orig() is never inside one)
SHUTDOWN = (SystemExit, KeyboardInterrupt)   # marked inert, then re-raised: shutdown is never swallowed


class PtouchFatal(RuntimeError):
    """TP > 1: an overlay failure that would leave this rank's cache policy different from its peers'. A RuntimeError
    on purpose: SGLang's run_scheduler_process catches Exception, logs it and signals its parent (the server stops)."""


def _log(msg):
    try:
        sys.stderr.write(f"[inkling_ptouch pid={os.getpid()}] {msg}\n")
        sys.stderr.flush()
    except SHUTDOWN:
        raise
    except BaseException:  # noqa: BLE001  (diagnostics are best effort)
        pass


def _inert(why):
    """why: a fixed literal. Marked inert first, then logged best effort."""
    STATE["inert"] = why
    _log("INERT: " + why)


def _fail(why, exc=None):
    """A runtime overlay failure (the original insert already ran). TP1: go inert, as the single-GPU recipe. TP > 1:
    stop (an inert rank would stop touching while its peers go on, and the trees would diverge)."""
    _inert(why)
    if STATE["tp"] > 1:
        raise PtouchFatal(f"inkling_ptouch: {why} at TP{STATE['tp']}; stopping instead of going inert on one rank "
                          f"(the ranks' prefix caches would diverge)") from exc


def _premise(mod):
    """The SWA window refresh happens at the END node of a match / insert only (not on the walk down)."""
    try:
        import inspect
        src = inspect.getsource(mod.SWAComponent.refresh_lru)
        if "reset_node_and_window_ancestors_mru" not in src or "WALKDOWN" not in src:
            _inert("SWA refresh_lru changed (the window refresh is no longer end-node only)")
    except GUARD as exc:  # noqa: BLE001
        _inert("premise check failed")
        if isinstance(exc, SHUTDOWN):
            raise


def _hybrid(cache):
    names = {type(c).__name__ for c in getattr(cache, "_components_tuple", ())}
    return "SWAComponent" in names and "MambaComponent" in names


def tp_context():
    """(tp size, TP GroupCoordinator or None) from the image's runtime context. A failed read counts as TP1 only when
    no multi-process torch.distributed group exists (otherwise refusing beats skipping the agreement)."""
    try:
        from sglang.srt.runtime_context import get_parallel

        par = get_parallel()
        tp = int(par.tp_size)
        group = par.tp_group if tp > 1 else None
    except Exception as exc:  # noqa: BLE001
        try:
            import torch.distributed as dist

            multi = dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1
        except Exception:  # noqa: BLE001
            multi = False
        if multi:
            raise PtouchFatal(f"inkling_ptouch: torch.distributed has several ranks but the TP size is unreadable "
                              f"({exc!r:.120}); refusing") from exc
        return 1, None
    if tp > 1 and getattr(group, "cpu_group", None) is None:
        raise PtouchFatal(f"inkling_ptouch: TP{tp} without a TP group / CPU group; refusing")
    return tp, group


def agree(group, armed, min_tokens):
    """One MIN all-reduce over the TP group's CPU (gloo) group of [armed, min, -min]: (True, None) iff every rank is
    armed with the same threshold, else (False, why). Reached by every rank whatever its local state."""
    import torch
    import torch.distributed as dist

    m = int(min_tokens) if -1 <= int(min_tokens) <= MIN_MAX else -1  # bounded: always fits int64
    t = torch.tensor([1 if armed else 0, m, -m], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN, group=group.cpu_group)
    v = [int(a) for a in t.tolist()]
    if v[0] != 1:
        return False, "a rank is inert"
    if v[1] != -v[2]:
        return False, "ranks differ in INKLING_PROMPT_TOUCH_MIN"
    return True, None


def tp_agree():
    """Called after every UnifiedRadixCache.__init__ (all ranks build the same caches at the same point of start-up).
    Sets STATE["tp"]; at TP > 1 all ranks end armed or all inert."""
    tp, group = tp_context()
    STATE["tp"] = tp
    if tp <= 1:
        return True
    try:
        ok, why = agree(group, not STATE["inert"], STATE["min"])
    except BaseException as exc:  # a broken collective: stop (peers would hang or diverge)
        raise PtouchFatal(f"inkling_ptouch: TP{tp} start-up agreement failed ({exc!r:.200})") from exc
    STATE["agreed"] += 1
    if not ok:
        if not STATE["inert"]:
            _inert(f"TP{tp} agreement: {why}")
        elif not STATE.get("off"):
            _log(f"TP{tp} agreement: {why} (inert on every rank)")
        return False
    _log(f"TP{tp}: armed on every rank (prompts >= {STATE['min']} tokens)")
    return True


def wrap(mod):
    try:
        _wrap(mod)
    except PtouchFatal:
        raise
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise


def _wrap_init(cls):
    """First, before anything that can make this rank inert: every rank's caches join the start-up agreement."""
    init = cls.__init__
    if getattr(init, "_inkling_ptouch", False):
        return

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        tp_agree()  # even when inert: every rank joins the agreement

    __init__._inkling_ptouch = True
    __init__.__wrapped__ = init
    cls.__init__ = __init__


def _wrap(mod):
    try:
        cls = mod.UnifiedRadixCache
        _wrap_init(cls)
        if STATE.get("off"):
            return                                   # switched off: the agreement hook only
        orig = cls._checkpoint_into_tree
        need = ("touch_prefix", "disable")
        if not callable(getattr(cls, "touch_prefix", None)):
            _inert(f"UnifiedRadixCache lacks {need[0]}")
            return
        from sglang.srt.mem_cache.common import _req_radix_key
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise
        return
    try:  # optional: SGLang's cross-rank divergence checker (a no-op unless SGLANG_ENABLE_RANK_CONSENSUS_CHECKER=1)
        from sglang.srt.utils import rank_consensus_checker as rcc
        same = rcc.assert_same
    except Exception:  # noqa: BLE001
        same = None
    if getattr(orig, "_inkling_ptouch", False):
        return

    def _checkpoint_into_tree(self, req, *, up_to, **kwargs):
        pre, failed, pending, cause = None, False, None, None
        try:
            if not STATE["inert"] and not self.disable and _hybrid(self) and not getattr(req, "session_id", None):
                pre = (bool(req.finished()), len(req.origin_input_ids), int(req.retraction_count))
        except GUARD as exc:  # noqa: BLE001
            STATE["inert"] = "pre-read failed"                # no call that could raise before orig()
            pre, failed, cause = None, True, exc
            if isinstance(exc, SHUTDOWN):
                pending = exc
        out = orig(self, req, up_to=up_to, **kwargs)          # exactly once, outside the overlay's handlers
        if pending is not None:
            _inert("pre-read failed")
            raise pending                                     # the original insert ran; shutdown goes on
        if failed:
            _fail("pre-read failed", cause)                   # TP > 1: raises PtouchFatal
        if pre is None:
            return out
        try:
            if STATE["inert"]:
                return out
            finished, prompt, retractions = pre
            if not finished:
                # the checkpoint that completes the prompt: the deepest depth published on the prompt path so far
                depth = int(req.kv.cache_inserted_len)
                if up_to >= prompt >= STATE["min"] and 0 < depth <= prompt:
                    setattr(req, MARK, (depth, retractions))
            else:
                from sglang.srt.managers.req_types import FINISH_ABORT
                mark = getattr(req, MARK, None)
                touch = (mark is not None and mark[1] == retractions and int(req.retraction_count) == retractions
                         and not isinstance(req.finished_reason, FINISH_ABORT))
                if same is not None:
                    same("inkling_ptouch finished rid=%s touch=%s depth=%s", getattr(req, "rid", None), touch,
                         mark[0] if touch else None)
                if touch:
                    self.touch_prefix(_req_radix_key(self, req, req.origin_input_ids, max_prefix_len=mark[0]))
                    STATE["calls"] += 1
                    if STATE["calls"] in (1, 24, 64) or STATE["calls"] % 1024 == 0:
                        _log(f"prompt paths touched after the final insert: {STATE['calls']} "
                             f"(skipped {STATE['skipped']})")
                else:
                    STATE["skipped"] += 1
        except GUARD as exc:  # noqa: BLE001
            if isinstance(exc, SHUTDOWN):
                _inert("touch failed")
                raise
            _fail("touch failed", exc)                        # TP > 1: raises PtouchFatal
        return out

    _checkpoint_into_tree._inkling_ptouch = True
    _checkpoint_into_tree.__wrapped__ = orig
    cls._checkpoint_into_tree = _checkpoint_into_tree
    _log(f"armed: prompt paths (prompts >= {STATE['min']} tokens) touched after the final insert")


MIN_MAX = 1 << 40  # a threshold above any context; bounds the vote payload (int64)


def parse_min(raw):
    """INKLING_PROMPT_TOUCH_MIN -> (tokens, None) or (-1, why). Never raises."""
    try:
        v = int(str(raw).strip() or 4096)
    except (TypeError, ValueError):
        return -1, "invalid INKLING_PROMPT_TOUCH_MIN"
    if not 0 <= v <= MIN_MAX:
        return -1, "invalid INKLING_PROMPT_TOUCH_MIN"
    return v, None


def install(on=True):
    """sitecustomize calls it whatever the switch (on = INKLING_PROMPT_TOUCH), so that at TP > 1 every rank's cache
    joins the start-up agreement. Off: only UnifiedRadixCache.__init__ is wrapped (votes no). An invalid
    INKLING_PROMPT_TOUCH_MIN makes this rank inert, and at TP > 1 every rank with it. Once per process."""
    if STATE.get("installed"):
        return
    STATE["installed"] = True
    STATE["min"], bad = parse_min(os.environ.get("INKLING_PROMPT_TOUCH_MIN", "4096"))
    STATE["off"] = not on
    if not on:
        STATE["inert"] = "disabled"
    elif bad:
        _inert(bad)
    try:
        _hook.when_imported(URC, wrap)  # first: the agreement hook
        if on:
            _hook.when_imported(SWA, _premise)
    except GUARD as exc:  # noqa: BLE001
        _inert("install failed")
        if isinstance(exc, SHUTDOWN):
            raise
