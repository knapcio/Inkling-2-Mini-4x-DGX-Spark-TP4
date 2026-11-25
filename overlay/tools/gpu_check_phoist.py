#!/usr/bin/env python3
"""GPU check of the installed inkling_phoist patch (SGLang image, model server stopped).

  python3 tools/gpu_check_phoist.py [--model /model] [--tokens 8192,2048,512]

Installs inkling_pfuse (INKLING_PFUSE=all) and inkling_phoist (INKLING_PFUSE_HOIST=1) exactly as sitecustomize
does, then drives one loop pass the way pfuse's patched _forward does: a real LoopMessageBank (patched) over two
bank sources and the model's scaled loop queries, and per core layer the attention-side and mlp-side
mix.loop_mix calls with the layer's controller weights (real checkpoint values). Option A (default, INKLING_PFUSE_HOIST=1)
is exact up to FP32 summation order: every hoisted output is compared with pfuse's own loop_mix (the wrapped function)
and both with an FP64 reference of the same math and BF16 rounding points (gpu_check_pfuse's references); the hoist's
error must stay within 1.5x pfuse's (max and mean). INKLING_PFUSE_HOIST=exact requires bit identity. Also checks that the hoist served all 24 calls with one prepare per
pass, that a second pass (new bank object) prepares again, and that a CUDA graph capturing bank construction +
the 24 calls replays bit for bit like the eager hoist after the bank and sources change, and times one pass
(pfuse vs hoist, CUDA events) per row count. Also checks pfuse's eligibility gate: below
INKLING_PFUSE_MIN_ROWS (512) pfuse never reaches loop_mix (stock path), at 512 it does.
"""
import argparse
import glob
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
FAILED = []
D = 2304


def log(*a):
    print(*a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/model")
    ap.add_argument("--tokens", default="8192,4097,4096,2048,1025,1024,512")
    ap.add_argument("--ref-rows", type=int, default=256)
    ap.add_argument("--expect-variant", choices=("a", "exact"), required=True)
    a = ap.parse_args()
    os.environ.setdefault("INKLING_PFUSE", "all")
    os.environ["INKLING_PFUSE_HOIST"] = {"a": "1"}.get(a.expect_variant, a.expect_variant)
    import inkling_pfuse
    import inkling_phoist

    inkling_pfuse.install()
    inkling_phoist.install()
    import torch
    from safetensors import safe_open
    import sglang.srt.models.inkling2  # noqa: F401  (pfuse patches _forward, phoist prepare_loop)
    import sglang.srt.models.inkling_common.inkling2_loop as LM
    from sglang.srt.models.inkling_common.inkling2_loop import scaled_loop_queries
    import inkling_pfuse.mix as mix

    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "inkling_pfuse", "tools"))
    import gpu_check_pfuse as GP  # FP64 references + compare()

    GP.log = log
    variant = inkling_phoist.STATE["variant"]
    log("variant:", variant, "ptxas:", os.environ.get("TRITON_PTXAS_BLACKWELL_PATH"),
        "triton cache:", os.environ.get("TRITON_CACHE_DIR"))
    if variant != a.expect_variant:
        log(f"FAILED: variant {variant} installed, {a.expect_variant} expected")
        return 1

    log("applied:", inkling_phoist.STATE["applied"])
    if not getattr(mix.loop_mix, "_inkling_phoist", False) or not getattr(LM.LoopMessageBank.__init__,
                                                                            "_inkling_phoist", False):
        FAILED.append("not patched")
    orig = mix.loop_mix.__wrapped__
    h = {}
    for f in sorted(glob.glob(os.path.join(a.model, "model*.safetensors"))):
        hh = safe_open(f, "pt", device="cpu")
        for k in hh.keys():
            h[k] = hh

    def ck(n):
        k = "model.llm." + n
        return h[k].get_tensor(k).cuda()

    s = D ** -0.5
    sq = scaled_loop_queries(ck("middle_cycle_cross_pass_queries"), ck("middle_cycle_cross_pass_kv_queries"))
    sq = sq.contiguous()
    layers = []
    for L in range(6, 18):
        c = [torch.stack([ck(f"layers.{L}.controllers.{m}.{w}") for m in ("attn", "k", "v", "mlp")]).contiguous()
             for w in ("down_weight", "gain_weight", "forget_weight")]
        qa = ck(f"attn_res.attn_queries.{L - 1}").float() * s
        glob_ = L in (11, 17)
        layers.append(dict(ctrl=c, qa=(qa.unsqueeze(0) if qa.ndim == 1 else qa).contiguous(),
                           qm=(ck(f"attn_res.mlp_queries.{L}").float() * s).reshape(1, D).contiguous(),
                           norms=([ck(f"layers.{L}.{n}_input_norm.weight") for n in ("q", "k", "v")] if glob_
                                  else [ck(f"layers.{L}.attn_norm.weight")] * 3)))
    # the model side as prepare_loop sees it (stub model with the real controller weights and scaled queries)
    stub = types.SimpleNamespace(
        config=types.SimpleNamespace(loop_core_layers=range(6, 18), loop_config={}),
        layers=[None] * 6 + [types.SimpleNamespace(controller_weights=tuple(l_["ctrl"])) for l_ in layers] + [None] * 6,
        loop_scaled_queries=sq)
    pack = inkling_phoist.build_pack(stub)
    buf = pack["down_all"]
    if inkling_phoist.build_pack(stub)["down_all"] is not buf:  # reload: same buffer
        FAILED.append("reload buffer")
    if variant != "exact":
        # what prepare_loop runs at model load in serving (before capture); without it Option A never hoists
        prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)  # serving runs prepare_loop under SGLang's model dtype
        try:
            if not inkling_phoist.startup_check(stub, pack):
                FAILED.append("startup self-test")
        finally:
            torch.set_default_dtype(prev_dtype)
    st = inkling_phoist.STATE
    tol = inkling_phoist.tolerance_ok
    g = torch.Generator(device="cuda").manual_seed(5)
    graphs = []

    NSRC = (2, 3, 4, 2, 3, 3, 2, 3, 4, 2, 3, 3)  # 2-4 state sources; the global visits (L11, L17) get 3

    def nsrc(li):
        return NSRC[li]

    def args_of(li, side, srcs):
        """(sources, addend, queries, norm weights, rows) of one call, mirroring pfuse's layer.py: the mlp side of
        some visits is a block boundary (attention appended as a source, no addend)."""
        lay = layers[li]
        n = nsrc(li)
        if side == "attn":
            return srcs[li][:n], None, lay["qa"], lay["norms"], slice(0, 3)
        if li % 4 == 3:
            return srcs[li][:n] + [srcs[li][4]], None, lay["qm"], None, slice(3, 4)
        return srcs[li][:n], srcs[li][4], lay["qm"], None, slice(3, 4)

    def run_pass(fn, bank_in, srcs):
        bank = LM.LoopMessageBank(tuple(bank_in), sq)
        out = []
        for li, lay in enumerate(layers):
            for side in ("attn", "mlp"):
                src, add, q, nw, rows = args_of(li, side, srcs)
                d, gn, fg = (w[rows] for w in lay["ctrl"])
                kw = dict(norm_weights=nw, eps_norm=1e-6) if nw is not None else {}
                out.append(fn(src, add, q, True, bank.blocks, bank.scaled_queries[li, rows], d, gn, fg, **kw))
        return out

    def same(x, y):
        return len(x) == len(y) and all(
            u.dtype == v.dtype and torch.equal(u.view(torch.int32 if u.dtype == torch.float32 else torch.int16),
                                               v.view(torch.int32 if v.dtype == torch.float32 else torch.int16))
            for u, v in zip(x, y))

    def close(x, y):
        return len(x) == len(y) and all(tol(u, v)[0] for u, v in zip(x, y))

    with torch.no_grad():
        for T in [int(t) for t in a.tokens.split(",")]:
            bf = lambda sc: (torch.randn(T, D, device="cuda", generator=g) * sc).to(torch.bfloat16)  # noqa: E731
            srcs = [[bf(0.6), bf(2.0), bf(1.0), bf(1.2), bf(0.9)] for _ in layers]
            bank_in = [bf(1.5), bf(0.8)]
            ref = run_pass(orig, bank_in, srcs)
            served = []
            outs = []
            for _ in range(2):  # two passes = two bank objects: one prepare each
                h0, p0, c0 = st["hoisted"], st["prepared"], dict(st["checked"])
                outs.append(run_pass(mix.loop_mix, bank_in, srcs))
                served.append((st["hoisted"] - h0, st["prepared"] - p0, st["checked"].get(T, 0) - c0.get(T, 0)))
            torch.cuda.synchronize()
            if variant == "exact":
                ok = same(ref, outs[0]) and same(ref, outs[1])
                acct = all(p == 1 for _, p, _ in served) and sum(h for h, _, _ in served) + served[0][2] >= 24
            else:
                ok = same(outs[0], outs[1])  # the hoist is deterministic
                acct = served == [(24, 1, 0), (24, 1, 0)]
                nf = len(GP.FAILED)
                # FP64 reference on the first ref_rows rows and the last 16 rows (the last partial row tile)
                idx = torch.unique(torch.cat([torch.arange(min(a.ref_rows, T)), torch.arange(max(T - 16, 0), T)]))
                idx = idx.cuda()
                rr = int(idx.numel())
                for li, lay in enumerate(layers):
                    for side in ("attn", "mlp"):
                        i = 2 * li + (side == "mlp")
                        src, add, q, nw, rows = args_of(li, side, srcs)
                        r = GP.ref_attn_res([x[idx] for x in src], None if add is None else add[idx], q)
                        rm = GP.ref_attn_res([b[idx] for b in bank_in], None, sq[li, rows])
                        dn, gn, fg = (w[rows] for w in lay["ctrl"])
                        shared = q.shape[0] == 1 and rows.stop - rows.start == 3
                        mixed = GP.ref_loop(r, rm, dn, gn, fg, shared)
                        refv = torch.stack([GP.ref_norm(mixed[j], nw[j]) for j in range(3)]) if nw else mixed
                        GP.compare(f"T={T} core {li} {side} n={len(src)}{' +add' if add is not None else ''}",
                                   ref[i][:, idx], outs[1][i][:, idx], refv, rr)
                ok = ok and len(GP.FAILED) == nf and close(outs[1], ref)
                bits = sum(int((u.view(torch.int16 if u.dtype == torch.bfloat16 else torch.int32) ==
                                v.view(torch.int16 if v.dtype == torch.bfloat16 else torch.int32)).sum())
                           for u, v in zip(ref, outs[1]))
                tot = sum(u.numel() for u in ref)
                worst = max(tol(u, v)[1] for u, v in zip(outs[1], ref))
                log(f"T={T}: hoist vs pfuse bit-equal {100 * bits / tot:.3f} % of elements, worst rel max err "
                    f"{worst:.2e}")
            log(f"T={T}: eager passes {'OK' if ok else 'DIFFER'}; per pass (hoisted, prepares, self-checks) {served}")
            if not ok or not acct or st["failed"] is not None:
                FAILED.append(f"eager T={T}")
            # timing: one loop pass (bank construction + 24 loop_mix calls), pfuse vs hoist
            ev = [torch.cuda.Event(enable_timing=True) for _ in range(4)]
            tp, th = [], []
            for _ in range(5):
                ev[0].record(); run_pass(orig, bank_in, srcs); ev[1].record()  # noqa: E702
                ev[2].record(); run_pass(mix.loop_mix, bank_in, srcs); ev[3].record()  # noqa: E702
                torch.cuda.synchronize()
                tp.append(ev[0].elapsed_time(ev[1]))
                th.append(ev[2].elapsed_time(ev[3]))
            tp.sort()
            th.sort()
            log(f"T={T}: one pass pfuse {tp[2]:.2f} ms, hoist {th[2]:.2f} ms -> saved {tp[2] - th[2]:+.2f} ms per "
                f"pass, {2 * (tp[2] - th[2]):+.2f} ms per chunk (2 looped passes)")
            # CUDA graph: bank construction + 24 calls captured; replays after bank / source changes
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                run_pass(mix.loop_mix, bank_in, srcs)
            torch.cuda.current_stream().wait_stream(s_)
            gr = torch.cuda.CUDAGraph()
            h0, p0 = st["hoisted"], st["prepared"]
            with torch.cuda.graph(gr):
                gout = run_pass(mix.loop_mix, bank_in, srcs)
            if (st["hoisted"] - h0, st["prepared"] - p0) != (24, 1):
                FAILED.append(f"capture fell through T={T}")
                log(f"T={T}: capture hoisted {st['hoisted'] - h0} / prepared {st['prepared'] - p0} (expect 24 / 1)")
            graphs.append((T, gr, gout, bank_in, srcs))
            for i in range(2):
                for b in bank_in:
                    b.copy_((torch.randn(T, D, device="cuda", generator=g) * (1.2 + i)).to(torch.bfloat16))
                srcs[3][1].copy_((torch.randn(T, D, device="cuda", generator=g)).to(torch.bfloat16))
                gr.replay()
                torch.cuda.synchronize()
                eager = run_pass(mix.loop_mix if variant != "exact" else orig, bank_in, srcs)
                ok = same(gout, eager)
                if variant != "exact":
                    ok = ok and close(gout, run_pass(orig, bank_in, srcs))
                log(f"T={T}: graph replay {i} after bank / source change {'OK' if ok else 'DIFFER'}")
                if not ok:
                    FAILED.append(f"graph T={T}")

        # all buckets' graphs replayed again after the other buckets and eager passes at other sizes
        for T, gr, gout, bank_in, srcs in graphs:
            for b in bank_in:
                b.copy_((torch.randn(T, D, device="cuda", generator=g) * 0.9).to(torch.bfloat16))
            gr.replay()
            torch.cuda.synchronize()
            fresh = run_pass(mix.loop_mix if variant != "exact" else orig, bank_in, srcs)
            ok = same(gout, fresh) and (variant == "exact" or close(gout, run_pass(orig, bank_in, srcs)))
            log(f"T={T}: late replay (after other buckets + eager passes) {'OK' if ok else 'DIFFER'}")
            if not ok:
                FAILED.append(f"late replay T={T}")
        log("peak memory allocated %.0f MiB, reserved %.0f MiB" % (torch.cuda.max_memory_allocated() / 2 ** 20,
                                                                  torch.cuda.max_memory_reserved() / 2 ** 20))
        # pfuse eligibility (the hoist runs only inside loop_mix, which pfuse calls only when eligible)
        from inkling_pfuse import layer as PL
        from sglang.srt.model_executor.forward_mode import ForwardMode

        lay = types.SimpleNamespace(is_local=True,
                                    attn_norm=types.SimpleNamespace(pre_weight=None, weight=layers[0]["norms"][0],
                                                                    output_dtype=torch.bfloat16, eps=1e-6))
        lay.mlp_norm = lay.attn_norm
        fb = types.SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        res = {}
        for t in (256, 511, 512):
            stt = types.SimpleNamespace(partial=torch.zeros(t, D, dtype=torch.bfloat16, device="cuda"), blocks=[1])
            res[t] = PL._eligible(lay, stt, fb)
        log(f"pfuse eligibility (min rows {PL.CFG['min_rows']}): {res}")
        if res.get(256) or res.get(511) or not res.get(512):
            FAILED.append("eligibility")
    log("state:", {k: v for k, v in inkling_phoist.STATE.items() if k in ("applied", "hoisted", "fallthrough",
                                                                         "prepared")})
    log("FAILED: " + ", ".join(FAILED) if FAILED else "ALL OK")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
