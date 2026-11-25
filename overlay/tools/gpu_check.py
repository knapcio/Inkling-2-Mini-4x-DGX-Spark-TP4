#!/usr/bin/env python3
"""GPU correctness + quick timing for inkling_draftbytes. Run inside the SGLang
container on the Spark, with the server STOPPED (it allocates ~3 GB on the GPU,
nothing pinned):

  docker run --rm --gpus all -v <repo>/overlay:/opt/inkling-draft:ro \
      -e PYTHONPATH=/opt/inkling-draft lmsysorg/sglang:inkling-2-mini \
      python3 /opt/inkling-draft/gpu_check.py [--vocab 201024] [--hot 32768]

Part 1 (kernels, random weights at the real shapes): Triton kernels vs the
pure-torch reference (must agree within 1 BF16 ulp on >= 99.9 % of outputs),
and the FP8 variants vs the BF16 stock math (reported, not gated).
Part 2 (integration, real sglang classes): the patched Inkling2MTPHead.step_bias,
compute_base_logits and the real dspark_draft.draft_markov_block +
MarkovSampleStep, for every variant vs the stock path; then each variant is
captured in a CUDA graph and its replay must equal the eager run bit for bit.
Part 3: per-step time of the draft vocab work (base logits + 7 Markov steps),
stock vs each variant, CUDA-graph replays, L2 flushed before each block.
Exit code 0 = all gated checks passed.
"""

import argparse
import dataclasses
import json
import os
import sys
import tempfile
import traceback

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

FAILS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def info(name, detail):
    print(f"  [info] {name} {detail}")


def ulp_frac(a, b):
    """Fraction of a's elements within one BF16 ulp of b (finite b)."""
    a, b = a.float(), b.float()
    fin = torch.isfinite(b)
    tol = b.abs() * 2 ** -7 + 1e-6 * b[fin].abs().max().clamp_min(1e-30)
    ok = ((a - b).abs() <= tol) & fin
    same_inf = (~fin) & (a == b)
    return ((ok | same_inf).float().mean()).item()


def rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).norm() / b.norm()).item()


def model_config(path):
    cfg = {"padded_vocab_size": 201024, "unpadded_vocab_size": 200058,
           "final_logit_softcapping": 30.0, "hidden_size": 2304, "markov_rank": 256}
    try:
        with open(path) as f:
            raw = json.load(f)
        text = raw.get("text_config", raw)
        cfg["padded_vocab_size"] = int(text.get("padded_vocab_size", cfg["padded_vocab_size"]))
        cfg["unpadded_vocab_size"] = int(text.get("vocab_size", cfg["unpadded_vocab_size"]))
        cfg["final_logit_softcapping"] = text.get("final_logit_softcapping", 30.0)
        cfg["hidden_size"] = int(text.get("hidden_size", 2304))
        print(f"model config from {path}: {cfg}")
    except Exception as e:  # noqa: BLE001
        print(f"no model config at {path} ({e!r}); using defaults {cfg}")
    return cfg


def random_weights(vocab, hidden, rank, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    lm = (torch.randn(vocab, hidden, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    proj = (torch.randn(hidden, rank, device="cuda", generator=g) / rank ** 0.5).to(torch.bfloat16)
    pbias = (torch.randn(hidden, device="cuda", generator=g) * 0.02).to(torch.bfloat16)
    fold = torch.empty(vocab, rank, device="cuda", dtype=torch.bfloat16)
    for s in range(0, vocab, 16384):
        fold[s:s + 16384] = (lm[s:s + 16384].float() @ proj.float()).to(torch.bfloat16)
    fold_bias = (lm @ pbias.unsqueeze(1)).squeeze(1)
    return lm, proj, pbias, fold, fold_bias


# --------------------------------------------------------------------------


def part1_kernels(args, cfg):
    from inkling_draftbytes import ops
    from inkling_draftbytes.quant import bf16_pack, linear_ref, quantize

    print("\n== Part 1: kernels vs reference")
    v, h, r = cfg["padded_vocab_size"], cfg["hidden_size"], 256
    lm, _, _, fold, fold_bias = random_weights(v, h, r)
    lat = torch.tanh(torch.randn(8, r, device="cuda")).to(torch.bfloat16)
    stock = torch.addmm(fold_bias, lat, fold.T)

    for m in (1, 2, 4, 8):
        out = ops.w8_linear(lat[:m], bf16_pack(fold), fold_bias, narrow_k=True)
        check(f"foldk M={m} vs addmm", ulp_frac(out, stock[:m]) >= 0.999,
              f"within-1ulp={ulp_frac(out, stock[:m]):.5f}")
    for kind, grp in (("row", 0), ("group", 32), ("mx", 0)):
        pack = quantize(fold, kind, grp)
        ref = linear_ref(lat, pack, fold_bias)
        for m in (1, 8):
            out = ops.w8_linear(lat[:m], pack, fold_bias, narrow_k=True)
            check(f"fold8:{pack.label} M={m} kernel vs reference", ulp_frac(out, ref[:m]) >= 0.999,
                  f"within-1ulp={ulp_frac(out, ref[:m]):.5f}")
        info(f"fold8:{pack.label} vs BF16 stock", f"rel={rel(ref, stock):.4f}")
        del pack, ref

    x = torch.randn(64, h, device="cuda").to(torch.bfloat16)
    lm_stock = torch.matmul(x, lm.T)
    try:
        from sglang.kernels.ops.gemm import sm12x_bf16_gemv as gm
    except Exception as e:  # noqa: BLE001
        gm = None
        check("import sm12x_bf16_gemv", False, repr(e))
    for m in (1, 7, 14, 21, 28, 56):
        out = ops.bf16_gemv(x[:m], lm, 64, gm)
        check(f"gemv M={m} vs matmul", ulp_frac(out, lm_stock[:m]) >= 0.999,
              f"within-1ulp={ulp_frac(out, lm_stock[:m]):.5f}")
    for kind in ("row", "mx"):
        pack = quantize(lm, kind, 0)
        ref = linear_ref(x[:14], pack)
        for m in (7, 14):
            out = ops.w8_linear(x[:m], pack)
            check(f"head8:{pack.label} M={m} kernel vs reference", ulp_frac(out, ref[:m]) >= 0.999,
                  f"within-1ulp={ulp_frac(out, ref[:m]):.5f}")
        info(f"head8:{pack.label} vs BF16", f"rel={rel(ref, lm_stock[:14]):.4f} argmax-agree="
             f"{(ref.argmax(-1) == lm_stock[:14].argmax(-1)).float().mean().item():.3f}")
        del pack, ref

    for n, k in ((5120, h), (h, 3072), (18432, h), (h, 9216)):
        w = (torch.randn(n, k, device="cuda") * 0.02).to(torch.bfloat16)
        xx = torch.randn(16, k, device="cuda").to(torch.bfloat16)
        for kind in ("mx", "row"):
            pack = quantize(w, kind, 0)
            ref = linear_ref(xx, pack)
            out = ops.w8_linear(xx[:8], pack)
            check(f"draftmx8:{pack.label} [{n}x{k}] M=8 kernel vs reference",
                  ulp_frac(out, ref[:8]) >= 0.999, f"within-1ulp={ulp_frac(out, ref[:8]):.5f}")
            info(f"draftmx8:{pack.label} [{n}x{k}] vs BF16", f"rel={rel(ref, xx @ w.T):.4f}")

    # Fused hot step vs its reference.
    n_hot = min(args.hot, v)
    ids = torch.randperm(cfg["unpadded_vocab_size"], device="cuda")[:n_hot].sort().values
    width = v
    base = torch.randn(4, width, device="cuda").to(torch.bfloat16)
    for fold8 in (False, True):
        sub = fold.index_select(0, ids)
        pack = quantize(sub, "row") if fold8 else bf16_pack(sub.contiguous())
        fb = fold_bias.index_select(0, ids)
        for m in (1, 4):
            a = torch.full((m, width), float("-inf"), device="cuda", dtype=torch.bfloat16)
            b = a.clone()
            ops.hot_step(lat[:m], pack, fb, base[:m], ids.to(torch.int32), a,
                         softcap=cfg["final_logit_softcapping"], vocab_limit=cfg["unpadded_vocab_size"])
            ops.hot_step_ref(lat[:m], pack, fb, base[:m], ids, b,
                             softcap=cfg["final_logit_softcapping"], vocab_limit=cfg["unpadded_vocab_size"])
            check(f"hot_step {'fp8' if fold8 else 'bf16'} M={m} kernel vs reference",
                  ulp_frac(a, b) >= 0.999, f"within-1ulp={ulp_frac(a, b):.5f}")
    del lm, fold, fold_bias
    torch.cuda.empty_cache()


# --------------------------------------------------------------------------


class _TpSync:
    def sync(self, site, values):
        return values


def part2_and_3(args, cfg):
    print("\n== Part 2: integration with the real sglang classes")
    hot_file = os.path.join(tempfile.gettempdir(), "idb_gpu_check_hot.json")
    gperm = torch.Generator().manual_seed(3)
    perm = torch.randperm(cfg["unpadded_vocab_size"], generator=gperm).tolist()
    with open(hot_file, "w") as f:
        json.dump(perm, f)
    os.environ["INKLING_DRAFT_HOTVOCAB_FILE"] = hot_file
    import inkling_draftbytes
    from inkling_draftbytes import patches
    from inkling_draftbytes.quant import bf16_pack, quantize_mode

    inkling_draftbytes.install(f"gemv,fold8,hotvocab:{args.hot}")
    from sglang.srt.models import dspark as dspark_mod
    from sglang.srt.models import inkling2_mtp as mtp
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    try:
        dspark_mod.get_spec()
    except Exception:  # noqa: BLE001 - no server runtime here: the flag is unset
        from types import SimpleNamespace

        dspark_mod.get_spec = lambda: SimpleNamespace(speculative_dspark_lm_head_dtype=None)

    v, h, r = cfg["padded_vocab_size"], cfg["hidden_size"], 256
    lm, proj, pbias, _, _ = random_weights(v, h, r, seed=1)
    head = mtp.Inkling2MTPHead(v, r, h, 1e-6).to(device="cuda", dtype=torch.bfloat16)
    with torch.no_grad():
        head.hidden_out.weight.copy_(proj)
        head.hidden_out.bias.copy_(pbias)
        head.markov_w1.weight.normal_(0, 0.5)
    head.step_softcap = cfg["final_logit_softcapping"] or None
    head.step_vocab_limit = cfg["unpadded_vocab_size"]
    org = v

    class LMHead(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(lm, requires_grad=False)
            self.org_vocab_size, self.tp_size, self.quant_method = org, 1, None

    class Fake:  # the attributes compute_base_logits / prepare read
        pass

    model = Fake()
    model.lm_head = LMHead()
    model.markov_head = head
    model.logits_mup_width_multiplier = None
    model.layers = torch.nn.ModuleList()
    model._gather_logits = lambda x: x[..., :org]
    head.attach_fold(model.lm_head.weight, None)  # patched refresh_fold: no-op before prepare
    head._gather_logits = model._gather_logits
    opts_all = patches.STATE.opts
    patches.prepare(model)
    packs = {
        "fold_fp8": head._idb_fold,
        "fold_bf16": bf16_pack(head._fold_weight),
        "hot_fp8": head._idb_hot,
    }
    patches.STATE.opts = dataclasses.replace(opts_all, fold8=None)
    packs["hot_bf16"] = patches._build_hot(model, head, model.lm_head, None)
    patches.STATE.opts = dataclasses.replace(opts_all, fold8=opts_all.fold8, head8=opts_all.fold8)
    packs["hot_fp8_head8"] = patches._build_hot(model, head, model.lm_head, None)
    patches.STATE.opts = opts_all

    variants = {
        "stock": (False, None, None),
        "gemv": (True, None, None),
        "foldk": (False, "fold_bf16", None),
        "fold8": (False, "fold_fp8", None),
        "gemv+fold8": (True, "fold_fp8", None),
        f"hot{args.hot}": (False, None, "hot_bf16"),
        f"hot{args.hot}+fold8": (False, None, "hot_fp8"),
        f"hot{args.hot}+fold8+gemv": (True, None, "hot_fp8"),
        f"hot{args.hot}+fold8+head8": (False, None, "hot_fp8_head8"),
    }

    def use(name):
        gemv, fold, hot = variants[name]
        patches.STATE.opts = dataclasses.replace(opts_all, gemv=gemv)
        head._idb_fold = packs[fold] if fold else None
        head._idb_hot = packs[hot] if hot else None

    gamma = 7

    def make_inputs(bs, seed):
        g = torch.Generator(device="cuda").manual_seed(seed)
        return dict(
            hidden=(torch.randn(bs * gamma, h, device="cuda", generator=g) * 2).to(torch.bfloat16),
            draft_hidden=torch.randn(bs, gamma, h, device="cuda", generator=g).to(torch.bfloat16),
            f_rnn=torch.randn(bs, h, device="cuda", generator=g).to(torch.bfloat16),
            anchor=torch.randint(0, 50000, (bs,), device="cuda", generator=g),
        )

    def block(inp, logits_out, tokens_out):
        bs = inp["anchor"].shape[0]
        head._get_staged_f = lambda: inp["f_rnn"]
        base, _ = mtp.Inkling2ForConditionalGenerationMTP.compute_base_logits(model, inp["hidden"])
        base = base.view(bs, gamma, -1)
        noise = dd.DraftNoise(key=11, offset=torch.zeros(1, dtype=torch.int64, device="cuda"))
        tokens, _ = dd.draft_markov_block(
            markov_head=head, base_logits=base, anchor_tokens=inp["anchor"],
            draft_hidden=inp["draft_hidden"],
            step_temperatures=torch.zeros(bs, device="cuda"), noise=noise,
            tp_sync=_TpSync(), logits_out=logits_out,
        )
        tokens_out.copy_(tokens)

    results = {}
    for bs in args.bs:
        inp = make_inputs(bs, 100 + bs)
        outs = {}
        for name in variants:
            use(name)
            lo = torch.empty(bs, gamma, org, device="cuda", dtype=torch.bfloat16)
            to = torch.empty(bs, gamma, device="cuda", dtype=torch.int64)
            block(inp, lo, to)
            torch.cuda.synchronize()
            outs[name] = (to.clone(), lo.clone())
            # Graph capture safety: replay must equal the eager run.
            lo2, to2 = torch.empty_like(lo), torch.empty_like(to)
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                block(inp, lo2, to2)
            torch.cuda.current_stream().wait_stream(s)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                block(inp, lo2, to2)
            lo2.zero_()
            to2.zero_()
            graph.replay()
            torch.cuda.synchronize()
            check(f"bs={bs} {name}: graph replay == eager",
                  torch.equal(to2, to) and torch.equal(lo2, lo))
            del graph
        tok_ref, log_ref = outs["stock"]
        for name, (tok, lo) in outs.items():
            if name == "stock":
                continue
            same_prefix = torch.ones_like(tok, dtype=torch.bool)
            same_prefix[:, 1:] = (tok == tok_ref).cumprod(1)[:, :-1].bool()
            agree = (tok == tok_ref).float().mean().item()
            hot = packs[variants[name][2]] if variants[name][2] else None
            if hot is not None:
                in_hot = torch.zeros(org, dtype=torch.bool, device="cuda")
                in_hot[hot.ids64] = True
                check(f"bs={bs} {name}: off-hot step logits are -inf",
                      bool(torch.isneginf(lo[..., ~in_hot].float()).all()))
                check(f"bs={bs} {name}: every drafted token is hot", bool(in_hot[tok].all()))
                a, b = lo[..., in_hot][same_prefix], log_ref[..., in_hot][same_prefix]
                ref_hot = in_hot[tok_ref] & same_prefix
                agree = (tok[ref_hot] == tok_ref[ref_hot]).float().mean().item() if ref_hot.any() else float("nan")
                if name == f"hot{args.hot}":
                    check(f"bs={bs} {name}: hot logits within 1 ulp of stock (same-prefix steps)",
                          ulp_frac(a, b) >= 0.999, f"{ulp_frac(a, b):.5f}")
                info(f"bs={bs} {name}", f"token agreement where stock token is hot: {agree:.3f}")
            else:
                a, b = lo[same_prefix], log_ref[same_prefix]
                if name in ("gemv", "foldk"):
                    check(f"bs={bs} {name}: step logits within 1 ulp of stock (same-prefix steps)",
                          ulp_frac(a, b) >= 0.999, f"{ulp_frac(a, b):.5f}")
                info(f"bs={bs} {name}", f"token agreement {agree:.3f}; step-logit rel err "
                     f"{rel(a[torch.isfinite(b)], b[torch.isfinite(b)]):.4f}")

        print(f"\n== Part 3: draft vocab work per step, bs={bs} (base logits + {gamma} Markov steps)")
        from inkling_draftbytes.timing import time_ms

        lo = torch.empty(bs, gamma, org, device="cuda", dtype=torch.bfloat16)
        to = torch.empty(bs, gamma, device="cuda", dtype=torch.int64)
        for name in variants:
            use(name)
            ms = time_ms(lambda: block(inp, lo, to), cold=True, iters=10, reps=5)
            results.setdefault(bs, {})[name] = ms
            base_ms = results[bs].get("stock", ms)
            print(f"  {name:28s} {ms:7.3f} ms   saves {base_ms - ms:+.3f} ms vs stock")
    use("stock")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="/model/config.json")
    ap.add_argument("--vocab", type=int, default=0, help="override padded vocab (smaller = less memory)")
    ap.add_argument("--hot", type=int, default=32768)
    ap.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--kernels-only", action="store_true")
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("needs a GPU")
        return 2
    cfg = model_config(args.model_config)
    if args.vocab:
        cfg["padded_vocab_size"] = args.vocab
        cfg["unpadded_vocab_size"] = min(cfg["unpadded_vocab_size"], args.vocab)
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name()}, "
          f"cc {torch.cuda.get_device_capability()}")
    part1_kernels(args, cfg)
    if not args.kernels_only:
        try:
            part2_and_3(args, cfg)
        except Exception:  # noqa: BLE001
            traceback.print_exc()
            check("integration part ran", False)
    print(f"\n{'ALL GATED CHECKS PASSED' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
