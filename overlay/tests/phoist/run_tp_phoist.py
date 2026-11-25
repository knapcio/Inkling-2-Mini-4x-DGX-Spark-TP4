"""One rank of the inkling_phoist TP stand-in (test_phoist_tp.py starts one process per rank; CPU, gloo).

Runs the patched Inkling2CausalLLM.prepare_loop (the real build_pack, startup_check and TP agreement) on a stand-in
model, with stand-ins for pfuse's loop_mix and phoist's kernels, then asks the patched loop_mix for its path at a
list of row counts. Prints one JSON line: this rank's decisions and state.

Environment: FAKE_TP, FAKE_RANK, FAKE_DIST (init method), PH_VARIANT (INKLING_PFUSE_HOIST, default 1),
PH_FAULT_RANK + PH_FAULT (oom | mismatch | pack | layout | tiles | generator | cleanup | noloop | loopnorm | minrows |
off | invalid), PH_NO_AGREE=1 (control: skip the agreement, the single-GPU behaviour), PH_NO_CONTEXT=1 (no
sglang.srt.runtime_context), PH_RELOAD=1 (prepare_loop again after the decisions: a weight reload).
"""
import json
import os
import tempfile
import sys
import types
import unittest.mock as um

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # overlay/

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

import inkling_phoist as ph  # noqa: E402

WORLD, RANK = int(os.environ.get("FAKE_TP", "1")), int(os.environ.get("FAKE_RANK", "0"))
FAULT = os.environ.get("PH_FAULT", "") if int(os.environ.get("PH_FAULT_RANK", "-1")) == RANK else ""
ROWS = (8, 33, 1040, 1025, 4100)        # armed keys: 8, 33, 1025 (see STARTUP_ROWS); 1040 and 4100 are not


def main():
    if WORLD > 1:
        dist.init_process_group("gloo", init_method=os.environ["FAKE_DIST"], rank=RANK, world_size=WORLD)
    else:
        def no_collective(*a, **kw):
            raise AssertionError("TP1 must not issue a collective")
        dist.all_reduce = no_collective
    for name in ("sglang", "sglang.srt"):
        sys.modules.setdefault(name, types.ModuleType(name))
    if not os.environ.get("PH_NO_CONTEXT"):
        rc = types.ModuleType("sglang.srt.runtime_context")
        group = types.SimpleNamespace(world_size=WORLD, rank_in_group=RANK,
                                      cpu_group=dist.group.WORLD if WORLD > 1 else None)
        par = types.SimpleNamespace(tp_size=WORLD, tp_group=group)
        rc.get_parallel = lambda: par
        rc.get_server_args = lambda: types.SimpleNamespace(enable_mixed_chunk=False)
        sys.modules["sglang.srt.runtime_context"] = rc

    # pfuse's loop_mix (the reference) and phoist's kernels: deterministic stand-ins
    def pfuse_loop_mix(sources, addend, queries, normalize_values, bank, bank_queries, down, gain, forget, **kw):
        return sources[0][:4, :8].float() * 0.5 + queries.shape[0]

    mixmod = types.ModuleType("inkling_pfuse.mix")
    mixmod.loop_mix = pfuse_loop_mix
    mixmod.MIX = {"bm": 32, "bn": 64, "split_down": True, "per_ctrl": True}
    laymod = types.ModuleType("inkling_pfuse.layer")   # pfuse's effective settings, as pfuse.install() leaves them
    laymod.CFG = {"levers": ("loopmix", "resnorm", "sconvres"), "min_rows": 1024 if FAULT == "minrows" else 512,
                  "loop_norm": "stock" if FAULT == "loopnorm" else "fused"}
    laymod._MODES = ("EXTEND", "MIXED")
    pf = types.ModuleType("inkling_pfuse")
    pf.mix, pf.layer = mixmod, laymod
    sys.modules.update({"inkling_pfuse": pf, "inkling_pfuse.mix": mixmod, "inkling_pfuse.layer": laymod})
    fake = types.ModuleType("inkling_phoist.phoist")
    mode = {"startup": True}

    def prepare(bank, sq, down_all, tokens, variant=None):
        if FAULT == "oom" and mode["startup"]:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated on this rank)")
        return {"T": tokens}

    def loop_mix_hoisted(prep, layer, side, sources, addend, queries, *a, **kw):
        if mode["startup"]:
            y = pfuse_loop_mix(sources, addend, queries, True, None, None, None, None, None)
            return y + 1.0 if FAULT == "mismatch" else y
        return "hoisted"

    bk_small = 128 if FAULT == "tiles" else 256  # e.g. a different shared-memory limit on one rank
    fake.prepare, fake.loop_mix_hoisted = prepare, loop_mix_hoisted
    fake._down_tiles = lambda tokens, device: (tokens <= 4096, 64, 64, bk_small if tokens <= 1024 else 64)
    sys.modules["inkling_phoist.phoist"] = fake
    ph.phoist = fake
    ph.STARTUP_ROWS = (8, 17, 1025)
    root = tempfile.mkdtemp(prefix="phoist-src-")     # the patch-target needles check_sources looks for
    for rel, needles in ph._SOURCES:
        os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
        with open(os.path.join(root, rel), "w") as f:
            f.write("\n".join(needles))
    hoist = {"off": "0", "invalid": "x"}.get(FAULT, os.environ.get("PH_VARIANT", "1"))
    ph.install(env={"INKLING_PFUSE_HOIST": hoist, "INKLING_PFUSE": "all"}, root=root)  # patches the fake mix at once
    if os.environ.get("PH_NO_AGREE"):
        ph.tp_agree = lambda group, tp, pack, refuse=None: None

    class Inkling2CausalLLM:
        def prepare_loop(self):
            pass

    ph._patch_model(types.SimpleNamespace(Inkling2CausalLLM=Inkling2CausalLLM))

    class Raising:
        def __getitem__(self, i):
            raise RuntimeError("CUDA out of memory while stacking (simulated)")

    model = Inkling2CausalLLM()
    cw = [(torch.full((4, 64, 4608), 1.0 + i, dtype=torch.bfloat16), torch.ones(4, 8), torch.ones(4, 8))
          for i in range(12)]  # (down, gain, forget) as stack_controller_weights
    model.layers = ([types.SimpleNamespace() for _ in range(6)]
                    + [types.SimpleNamespace(controller_weights=c) for c in cw] + [types.SimpleNamespace()] * 6)
    if FAULT == "pack":
        model.layers[9] = types.SimpleNamespace(controller_weights=Raising())
    if FAULT == "layout":
        model.layers[9] = types.SimpleNamespace(controller_weights=None)
    model.config = types.SimpleNamespace(loop_core_layers=range(6, 18), loop_config={})
    sq = torch.zeros(12, 4, ph.D)
    model.loop_scaled_queries = None if FAULT == "noloop" else sq

    def cleanup(*a):
        if FAULT == "cleanup":
            raise RuntimeError("CUDA error during empty_cache (simulated)")

    def generator(*a, **kw):
        raise RuntimeError("CUDA generator construction failed (simulated)")

    with um.patch.object(torch.cuda, "synchronize", lambda *a: None), \
            um.patch.object(torch.cuda, "empty_cache", cleanup), \
            um.patch.object(torch, "Generator", generator if FAULT == "generator" else torch.Generator):
        model.prepare_loop()
    mode["startup"] = False

    fn = ph.make_loop_mix(lambda *a, **kw: "pfuse")
    decisions = []
    with um.patch.object(torch.cuda, "is_current_stream_capturing", lambda: False):
        for rows in ROWS:
            for li, side in ((0, "attn"), (11, "mlp")):
                bl = ph.BankList([torch.zeros(rows, ph.D, dtype=torch.bfloat16) for _ in range(2)])
                bl.sq, bl.prep = sq, None
                r = slice(0, 3) if side == "attn" else slice(3, 4)
                down = model.layers[6 + li].controller_weights
                down = down[0][r] if isinstance(down, tuple) else torch.zeros(1)
                decisions.append(fn([torch.zeros(rows, ph.D, dtype=torch.bfloat16)], None, torch.zeros(1, ph.D),
                                    True, bl, bl.sq[li, r], down, None, None))
    pack = ph.PACKS.get(sq.data_ptr())
    print(json.dumps({"rank": RANK, "decisions": decisions, "failed": repr(ph.STATE["failed"]),
                      "startup_ok": bool(pack and pack.get("startup_ok"))}), flush=True)
    if os.environ.get("PH_RELOAD"):
        down_before = pack["down_all"] if pack else None
        try:
            with um.patch.object(torch.cuda, "synchronize", lambda *a: None), \
                    um.patch.object(torch.cuda, "empty_cache", lambda *a: None):
                model.prepare_loop()
        finally:
            after = ph.PACKS.get(sq.data_ptr())
            print(json.dumps({"rank": RANK, "reload": "done", "pack_kept": down_before is None or
                              (after is not None and after["down_all"] is down_before)}), flush=True)
    if WORLD > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
