#!/usr/bin/env python3
"""GPU check + microbench of inkling_noexp's fused dead-row mask (serving assembler; server stopped).

  TRITON_PTXAS_BLACKWELL_PATH=/usr/local/cuda/bin/ptxas python3 tools/gpu_check_noexp_mask.py

Installs inkling_noexp (and imports inkling_vcap.state + the model module, so both patches apply), then under CUDA
graph capture compares, for live widths at c1 / c4 / c8 / c16 (8-128 rows):
  v2.7.2 path: vcap fused remap (ids[src[r]]) -> noexp's live mask + torch.where -> route_metadata(E + 1)
  new path:    noexp's _remap_mask_kernel (one launch) -> route_metadata(E + 1) on the remembered ids
bit for bit on src2dst, dst2src, gammas, offsets and the gathered weights, after replays with changed widths.
Times 48 visits per graph (remap + routing per visit) for both paths: the per-step overhead saved.
"""
import os
import statistics
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FAILED = []


def log(*a):
    print(*a, flush=True)


def main():
    os.environ.setdefault("INKLING_NOEXP", "1")
    os.environ.setdefault("INKLING_VERIFY_CAP", "policy:x")
    import inkling_noexp

    inkling_noexp.install()
    import torch
    import inkling_vcap.state as VS
    import inkling_vcap.fused as VF
    import sglang.srt.models.inkling_common.inkling2_moe as M
    from sglang.kernels.ops.moe.inkling2_route import route_metadata

    ok = getattr(VS.VcapState.remap, inkling_noexp.MARK, False) and getattr(M.Inkling2Experts._sm120_route,
                                                                           inkling_noexp.MARK, False)
    log("ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH", "bundled"), "| patched:", ok)
    if not ok:
        FAILED.append("not patched")
    import inkling_vcap.fused as VFm

    err = VFm.selfcheck(torch.device("cuda"))  # vcap's check + (patched) the phantom-mask self-check
    log("vcap fused self-check:", err or "pass", "| mask_ok:", inkling_noexp.STATE["mask_ok"])
    if err is not None or inkling_noexp.STATE["mask_ok"] is not True:
        FAILED.append("self-check")
    host = types.SimpleNamespace(num_global_experts=128)
    g = torch.Generator(device="cuda").manual_seed(8)

    def vstate(src):
        o = types.SimpleNamespace(stride=8, src_index=src, remap_recorded=0, remap_skipped=0, use_fused=True)
        return o

    def mk(lens):
        return torch.tensor([(r // 8) * 8 if r % 8 >= lens[r // 8] else r for r in range(8 * len(lens))],
                            device="cuda", dtype=torch.int64)

    def old_path(obj, weights, ids):  # v2.7.2: vcap remap, then noexp's mask in _sm120_route
        w, i = VF.remap_rows_fused(weights, ids, obj.src_index)
        live = obj.src_index[: i.shape[0]] == torch.arange(i.shape[0], device="cuda", dtype=obj.src_index.dtype)
        return (w, i), inkling_noexp.route_dead_free(route_metadata, i, w, live, 128)

    def new_path(obj, weights, ids):
        w, i = VS.VcapState.remap(obj, weights, ids)
        return (w, i), M.Inkling2Experts._sm120_route(host, i, w)

    cases = {"c1": [5], "c4": [8, 1, 3, 6], "c8": [4, 5, 5, 4, 5, 4, 5, 5], "c16": [3] * 10 + [4] * 6,
             "c24/int32": [3] * 12 + [5] * 12}
    flush = torch.ones(32 << 20, device="cuda")
    fo = torch.empty((), device="cuda")
    with torch.no_grad():
        for name, lens in cases.items():
            T = 8 * len(lens)
            src = mk(lens)
            obj = vstate(src)
            logits = torch.rand(T, 128, device="cuda", generator=g)
            weights, ids = logits.topk(8, -1)
            ids = ids.contiguous().to(torch.int32 if "int32" in name else torch.int64)
            weights = weights.contiguous()

            def graph(fn):
                s_ = torch.cuda.Stream()
                s_.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s_):
                    fn()
                torch.cuda.current_stream().wait_stream(s_)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    out = fn()
                return gr, out

            c0 = inkling_noexp.STATE["fused_mask"]
            go, oo = graph(lambda: [old_path(obj, weights, ids) for _ in range(48)])
            gn, on = graph(lambda: [new_path(obj, weights, ids) for _ in range(48)])
            served = inkling_noexp.STATE["fused_mask"] - c0
            if served != 48:
                FAILED.append(f"{name} fused mask served {served}")
            good = True
            for lens2 in (lens, [max(1, x - 1) for x in lens], [8] * len(lens)):
                src.copy_(mk(lens2))
                go.replay(); gn.replay(); torch.cuda.synchronize()
                for (wo, ro), (wn, rn) in zip(oo, on):
                    good &= all(torch.equal(x, y) for x, y in zip(wo, wn))  # gate outputs = vcap's tensors
                    good &= all(torch.equal(x.view(torch.int32), y.view(torch.int32)) for x, y in zip(ro, rn))
            log(f"  {name} ({T} rows): {'bit-identical' if good else 'DIFFERS'} (3 width sets, 48 visits); fused mask "
                f"served {served}/48 at capture")
            if not good:
                FAILED.append(name)
            ts = {"old": [], "new": []}
            for _ in range(32):
                for k, gr in (("old", go), ("new", gn)):
                    torch.sum(flush, 0, out=fo)
                    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
                    e0.record(); gr.replay(); e1.record(); torch.cuda.synchronize()
                    ts[k].append(e0.elapsed_time(e1) * 1e3)
            o, n = statistics.median(ts["old"][2:]), statistics.median(ts["new"][2:])
            log(f"  {name}: 48 visits remap+mask+route v2.7.2 {o:7.1f} us, fused {n:7.1f} us -> {(o - n) / 1000:.3f} ms "
                f"per step")
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
