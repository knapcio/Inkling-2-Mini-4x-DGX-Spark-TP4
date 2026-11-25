"""One rank of the inkling_ptouch TP stand-in (test_ptouch_tp.py starts one process per rank; CPU, gloo).

A toy prefix cache plays UnifiedRadixCache with the properties the overlay depends on: one node per checkpoint
(prefill chunk ends, the prompt end, the output end), a logical LRU clock (as the tree's get_and_increase_time_counter),
the running request's lock on its latest node, eviction of the oldest unlocked node over a capacity, and
touch_prefix re-stamping the node of a key. Every rank replays the same scheduler script (as SGLang's ranks do: the
same broadcast requests, the same checkpoint calls), agents' turn 1 then turn 2 with pressure from filler requests.
Prints one JSON line: the tree digest, the turn-2 hits, the overlay state and the consensus records.

Environment: FAKE_TP, FAKE_RANK, FAKE_DIST, PT_OFF=1 (overlay not installed: stock), PT_FAULT_RANK + PT_FAULT
(min | badmin | off | premise | touch | preread), PT_NO_AGREE=1 (control: skip the agreement), PT_NO_CONTEXT=1.
"""
import hashlib
import json
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # overlay/

import torch.distributed as dist  # noqa: E402

WORLD, RANK = int(os.environ.get("FAKE_TP", "1")), int(os.environ.get("FAKE_RANK", "0"))
FAULT = os.environ.get("PT_FAULT", "") if int(os.environ.get("PT_FAULT_RANK", "-1")) == RANK else ""
AGENTS, PROMPT, OUT, FILLER = 8, 5000, 300, 3000
CAP, FILLERS = 40, 1   # turn 1 leaves 56 nodes: stock loses every prompt end (0 / 8 turn-2 hits), the touch keeps 8 / 8
RECORDS = []


class FINISH_ABORT:
    pass


class FINISH_LENGTH:
    pass


class SWAComponent:
    def refresh_lru(self, phase, node, root):
        match phase:
            case "WALKDOWN":
                return
            case _:
                self.lru.reset_node_and_window_ancestors_mru(node)


class ChangedSWAComponent:
    def refresh_lru(self, phase, node, root):
        return None


class MambaComponent:
    pass


class FullComponent:
    pass


class UnifiedRadixCache:
    def __init__(self):
        self.disable = False
        self._components_tuple = (FullComponent(), SWAComponent(), MambaComponent())
        self.nodes = {}           # key (tuple of token ids) -> [stamp, kind]
        self.locks = {}           # rid -> key the request's lock sits on
        self.clock = 0
        self.touches = 0

    def _tick(self):
        self.clock += 1
        return self.clock

    def _evict(self):
        while len(self.nodes) > CAP:
            locked = set(self.locks.values())
            free = [(v[0], k) for k, v in self.nodes.items() if k not in locked]
            if not free:
                return
            del self.nodes[min(free)[1]]

    def _checkpoint_into_tree(self, req, *, up_to, **kwargs):
        key = tuple(req.fill_ids()[:up_to])
        prompt = len(req.origin_input_ids)
        kind = "out" if up_to > prompt else ("prompt" if up_to == prompt else "chunk")
        self.nodes[key] = [self._tick(), kind]
        if req.finished():
            self.locks.pop(req.rid, None)
        else:
            self.locks[req.rid] = key
            req.kv.cache_inserted_len = max(req.kv.cache_inserted_len, min(up_to, prompt))
        self._evict()

    def touch_prefix(self, key):
        if FAULT == "touch":
            self.touches += 1
            if self.touches == 3:
                raise RuntimeError("touch walk failed on this rank (simulated)")
        if key in self.nodes:
            self.nodes[key][0] = self._tick()

    def digest(self):
        items = sorted((k[0], len(k), v[0], v[1]) for k, v in self.nodes.items())
        return hashlib.sha256(repr((items, self.clock)).encode()).hexdigest()[:16]


class Req:
    def __init__(self, rid, base, prompt):
        self.rid = rid
        self.origin_input_ids = list(range(base, base + prompt))
        self.output_ids = []
        self.retraction_count = 0
        self.session_id = None
        self.kv = types.SimpleNamespace(cache_inserted_len=0)
        self.finished_reason = None
        self._raise_once = FAULT == "preread" and rid == "a3-t1"

    def fill_ids(self):
        return self.origin_input_ids + self.output_ids

    def finished(self):
        if self._raise_once:
            self._raise_once = False
            raise RuntimeError("req.finished() failed on this rank (simulated)")
        return self.finished_reason is not None

    def finish(self):
        self.output_ids = list(range(-1, -OUT - 1, -1))
        self.finished_reason = FINISH_LENGTH()


def prefill(cache, req):
    n = len(req.origin_input_ids)
    for up_to in [c for c in (2048, 4096) if c < n] + [n]:
        cache._checkpoint_into_tree(req, up_to=up_to)


def finish(cache, req):
    req.finish()
    cache._checkpoint_into_tree(req, up_to=len(req.fill_ids()))


def script(cache):
    """The scheduler script every rank replays: turn 1 of all agents, finishes with filler pressure, turn 2."""
    filler = 0
    reqs = [Req(f"a{i}-t1", 100000 * (i + 1), PROMPT) for i in range(AGENTS)]
    for r in reqs:
        prefill(cache, r)
    for r in reqs:
        finish(cache, r)
        for _ in range(FILLERS):
            f = Req(f"f{filler}", 10_000_000 + 10000 * filler, FILLER)  # below the threshold: never touched
            filler += 1
            prefill(cache, f)
            finish(cache, f)
    hits = []
    for i in range(AGENTS):
        key = tuple(reqs[i].origin_input_ids)
        hit = key in cache.nodes
        hits.append(hit)
        r2 = Req(f"a{i}-t2", 100000 * (i + 1), PROMPT)
        if not hit:
            prefill(cache, r2)                                   # a miss re-prefills the whole prompt
        else:
            cache._checkpoint_into_tree(r2, up_to=PROMPT)        # a hit only publishes the prompt end again
        finish(cache, r2)
    return hits


def fake_modules():
    for name in ("sglang", "sglang.srt", "sglang.srt.mem_cache", "sglang.srt.managers", "sglang.srt.utils",
                 "sglang.srt.mem_cache.unified_cache", "sglang.srt.mem_cache.unified_cache.components"):
        sys.modules.setdefault(name, types.ModuleType(name))
    common = types.ModuleType("sglang.srt.mem_cache.common")
    common._req_radix_key = lambda cache, req, ids, max_prefix_len=None: tuple(ids[:max_prefix_len])
    rt = types.ModuleType("sglang.srt.managers.req_types")
    rt.FINISH_ABORT = FINISH_ABORT
    rcc = types.ModuleType("sglang.srt.utils.rank_consensus_checker")
    rcc.assert_same = lambda fmt, *args: RECORDS.append(fmt % args)
    swa = types.ModuleType("sglang.srt.mem_cache.unified_cache.components.swa")
    swa.SWAComponent = ChangedSWAComponent if FAULT == "premise" else SWAComponent
    urc = types.ModuleType("sglang.srt.mem_cache.unified_radix_cache")
    urc.UnifiedRadixCache = UnifiedRadixCache
    sys.modules.update({"sglang.srt.mem_cache.common": common, "sglang.srt.managers.req_types": rt,
                        "sglang.srt.utils.rank_consensus_checker": rcc, swa.__name__: swa, urc.__name__: urc})
    if not os.environ.get("PT_NO_CONTEXT"):
        rc = types.ModuleType("sglang.srt.runtime_context")
        group = types.SimpleNamespace(world_size=WORLD, rank_in_group=RANK,
                                      cpu_group=dist.group.WORLD if WORLD > 1 else None)
        par = types.SimpleNamespace(tp_size=WORLD, tp_group=group)
        rc.get_parallel = lambda: par
        sys.modules["sglang.srt.runtime_context"] = rc
    return urc


def main():
    if WORLD > 1:
        dist.init_process_group("gloo", init_method=os.environ["FAKE_DIST"], rank=RANK, world_size=WORLD)
    else:
        def no_collective(*a, **kw):
            raise AssertionError("TP1 must not issue a collective")
        dist.all_reduce = no_collective
    urc = fake_modules()
    pt = None
    if not os.environ.get("PT_OFF"):
        if FAULT == "min":
            os.environ["INKLING_PROMPT_TOUCH_MIN"] = "8192"
        if FAULT == "badmin":
            os.environ["INKLING_PROMPT_TOUCH_MIN"] = str(1 << 70)  # does not fit int64
        import inkling_ptouch as pt
        if os.environ.get("PT_NO_AGREE"):
            pt.tp_agree = lambda: True
        pt.install(FAULT != "off")        # the targets are already imported: the hooks run at once
    cache = urc.UnifiedRadixCache()       # the start-up agreement runs here (wrapped __init__)
    hits = script(cache)
    out = {"rank": RANK, "digest": cache.digest(), "hits": hits, "records": RECORDS}
    if pt is not None:
        out.update(inert=pt.STATE["inert"], calls=pt.STATE["calls"], tp=pt.STATE["tp"])
    print(json.dumps(out), flush=True)
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
