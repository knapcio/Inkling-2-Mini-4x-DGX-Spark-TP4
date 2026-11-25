"""Scenarios run in a fresh interpreter (one per test) against the CPU stubs.
Usage: python _scenarios.py NAME  (prints one JSON line)."""

import json
import sys

import torch


def _setup(seed=0, padded=4096, org=4000, limit=3990, softcap=30.0):
    torch.manual_seed(seed)
    import inkling_draftbytes

    opts = inkling_draftbytes.install()
    from sglang.srt.models import inkling2_mtp as m

    model = m.Inkling2ForConditionalGenerationMTP(padded=padded, org=org, limit=limit, softcap=softcap)
    lm_head = m.LMHead(padded, org, 128)
    model.attach_shared_modules(embed_tokens=None, lm_head=lm_head)
    return opts, m, model


def _block(model, dd, bs=2, gamma=7, temps=None, seed=1):
    g = torch.Generator().manual_seed(seed)
    hidden = (torch.randn(bs * gamma, 128, generator=g) * 3).to(torch.bfloat16)
    draft_hidden = torch.randn(bs, gamma, 128, generator=g).to(torch.bfloat16)
    model.markov_head._f_rnn = torch.randn(bs, 128, generator=g).to(torch.bfloat16)
    anchor = torch.randint(0, 3000, (bs,), generator=g)
    base, _ = model.compute_base_logits(hidden)
    base = base.view(bs, gamma, -1)
    temps = torch.zeros(bs) if temps is None else temps
    noise = dd.DraftNoise(key=123, offset=torch.zeros(1, dtype=torch.int64))
    tokens, logits = dd.draft_markov_block(
        markov_head=model.markov_head, base_logits=base, anchor_tokens=anchor,
        draft_hidden=draft_hidden, step_temperatures=temps, noise=noise,
    )
    return tokens, logits


def noop():
    import sys as _s

    n_before = len(_s.meta_path)
    import inkling_draftbytes

    res = inkling_draftbytes.install()
    from sglang.srt.models import inkling2_mtp as m

    return {
        "installed": res is not None,
        "meta_path_grew": len(_s.meta_path) != n_before,
        "step_bias_stock": m.Inkling2MTPHead.step_bias.__module__ == m.__name__,
        "project_stock": m.project_through_lm_head.__module__ == m.__name__,
    }


def fold_compare():
    """Patched step_bias (fold8 or foldk per env) vs the stock step_bias."""
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    head = model.markov_head
    stock = patches.STATE.originals["Inkling2MTPHead.step_bias"]
    bs = 4
    g = torch.Generator().manual_seed(5)
    state = (torch.zeros(bs, 64, dtype=torch.bfloat16),
             torch.randn(bs, 128, generator=g).to(torch.bfloat16))
    prev = torch.randint(0, 3000, (bs,), generator=g)
    hid = torch.randn(bs, 128, generator=g).to(torch.bfloat16)
    ref, _ = stock(head, prev, hidden_states=hid, state=state)
    new, _ = head.step_bias(prev, hidden_states=hid, state=state)
    diff = (new.float() - ref.float())
    rel = (diff.norm() / ref.float().norm()).item()
    ulp1 = (diff.abs() <= ref.float().abs() * 2 ** -7 + 1e-30).float().mean().item()
    return {"shape_ok": list(new.shape) == list(ref.shape), "dtype": str(new.dtype),
            "rel_err": rel, "within_1ulp": ulp1, "patched": patches.STATE.patched}


def hot_compare():
    """hotvocab (+ any other env options) vs stock, through the draft loop."""
    from inkling_draftbytes import patches
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    opts, m, model = _setup()
    hot = model.markov_head._idb_hot
    ids = hot.ids64
    tokens, logits = _block(model, dd)
    # The stock run on the same model: drop the hot state, restore stock paths.
    model.markov_head._idb_hot = None
    model.markov_head._idb_fold = None
    model.lm_head._idb_head = None
    tokens_ref, logits_ref = _block(model, dd)
    in_hot = torch.zeros(logits.shape[-1], dtype=torch.bool)
    in_hot[ids] = True
    off = logits[..., ~in_hot]
    # Compare only steps whose earlier drafted tokens agree (same recurrent input).
    same_prefix = torch.ones_like(tokens, dtype=torch.bool)
    same_prefix[:, 1:] = (tokens == tokens_ref).cumprod(1)[:, :-1].bool()
    on = logits[..., in_hot][same_prefix].float()
    on_ref = logits_ref[..., in_hot][same_prefix].float()
    finite = torch.isfinite(on_ref)
    d = (on - on_ref)[finite].abs()
    tol = on_ref[finite].abs() * 2 ** -7
    ref_in_hot = in_hot[tokens_ref] & same_prefix
    return {
        "n_hot": int(ids.numel()),
        "compared_steps": int(same_prefix.sum()),
        "off_all_neginf": bool(torch.isneginf(off.float()).all()),
        "hot_exact_frac": (d == 0).float().mean().item(),
        "hot_within_1ulp": (d <= tol).float().mean().item(),
        "hot_neginf_match": bool((torch.isneginf(on) == torch.isneginf(on_ref)).all()),
        "tokens_all_hot": bool(in_hot[tokens].all()),
        "agree_when_ref_hot": (tokens[ref_in_hot] == tokens_ref[ref_in_hot]).float().mean().item(),
        "patched": patches.STATE.patched,
    }


def hot_sampled():
    """Sampled rows: the kept logits must be what the token was drawn from:
    every drawn token is hot and has a finite kept logit."""
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    opts, m, model = _setup()
    ids = model.markov_head._idb_hot.ids64
    in_hot = torch.zeros(4000, dtype=torch.bool)
    in_hot[ids] = True
    ok = True
    for seed in range(5):
        tokens, logits = _block(model, dd, bs=3, temps=torch.tensor([0.7, 1.0, 0.0]), seed=seed)
        kept = logits.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)
        ok &= bool(in_hot[tokens].all()) and bool(torch.isfinite(kept.float()).all())
    return {"ok": ok}


def rnnhoist_compare():
    """rnnhoist (+ any other env options) vs the same options without it, through the draft loop; plus the
    per-step rnn_proj output vs the stock Linear and the fallback for a call outside the prepared walk."""
    from inkling_draftbytes import patches, rnnhoist
    from sglang.srt.speculative.dspark_components import dspark_draft as dd

    opts, m, model = _setup()
    head = model.markov_head
    out = {"patched": patches.STATE.patched}
    temps = torch.tensor([0.0, 0.8, 1.0])
    tokens, logits = _block(model, dd, bs=3, temps=temps)
    calls = {"hoisted": 0}
    orig_hp = rnnhoist.hoisted_proj

    def counting(*a, **k):
        calls["hoisted"] += 1
        return orig_hp(*a, **k)

    rnnhoist.hoisted_proj = counting
    _block(model, dd, bs=3, temps=temps)
    rnnhoist.hoisted_proj = orig_hp
    out["hoisted_calls"] = calls["hoisted"]
    packs = head._idb_rnn
    head._idb_rnn = None  # every step runs stock
    tokens_ref, logits_ref = _block(model, dd, bs=3, temps=temps)
    head._idb_rnn = packs
    out["tokens_equal"] = bool(torch.equal(tokens, tokens_ref))
    fin = torch.isfinite(logits_ref.float())
    d = (logits.float() - logits_ref.float())[fin].abs()
    out["logits_exact_frac"] = (d == 0).float().mean().item()
    out["logits_within_1ulp"] = (d <= logits_ref.float()[fin].abs() * 2 ** -7).float().mean().item()
    out["neginf_match"] = bool((torch.isneginf(logits.float()) == torch.isneginf(logits_ref.float())).all())
    # per-step projection vs the stock BF16 Linear
    g = torch.Generator().manual_seed(9)
    bs, gamma = 4, 7
    dh = torch.randn(bs, gamma, 128, generator=g).to(torch.bfloat16)
    f_rnn = torch.randn(bs, 128, generator=g).to(torch.bfloat16)
    walk = rnnhoist.begin_walk(packs, dh, f_rnn, 16)
    state = torch.randn(bs, 64, generator=g).to(torch.bfloat16)
    prev = torch.randint(0, 3000, (bs,), generator=g)
    worst = 0.0
    idx = []
    for s in range(gamma):
        idx.append(rnnhoist.step_index(walk, dh[:, s], f_rnn))
        ref = head.rnn_proj(torch.cat([state, head.markov_w1(prev), dh[:, s], f_rnn], -1)).float()
        got = rnnhoist.hoisted_proj(packs, walk, s, state, prev, head.markov_w1.weight).float()
        worst = max(worst, float(((got - ref).abs() / (ref.abs() * 2 ** -7 + 1e-30)).max()))
    out["step_index"] = idx
    out["proj_worst_ulp"] = worst
    out["outside_walk"] = [
        rnnhoist.step_index(walk, dh[:, 0].clone(), f_rnn),  # another tensor
        rnnhoist.step_index(walk, dh[:2, 0], f_rnn),  # other rows
        rnnhoist.step_index(walk, dh[:, 0], f_rnn.clone()),  # other f_rnn
        rnnhoist.step_index(None, dh[:, 0], f_rnn),
    ]
    out["too_many_rows"] = rnnhoist.begin_walk(packs, dh, f_rnn, 3) is None
    return out


def head_compare():
    """gemv/head8 via project_through_lm_head vs stock matmul."""
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    stock = patches.STATE.originals["inkling2_mtp.project_through_lm_head"]
    hidden = (torch.randn(7, 128) * 3).to(torch.bfloat16)
    ref = stock(hidden, model.lm_head).float()
    new = m.project_through_lm_head(hidden, model.lm_head).float()
    rel = ((new - ref).norm() / ref.norm()).item()
    return {"rel_err": rel, "argmax_agree": (new.argmax(-1) == ref.argmax(-1)).float().mean().item()}


def draftmx8():
    from inkling_draftbytes import patches

    opts, m, model = _setup()
    lin = model.layers[0].qkv_proj
    x = torch.randn(8, 128).to(torch.bfloat16)
    out, _ = lin(x)  # CPU: falls back to the inner (stock) apply
    ref = torch.nn.functional.linear(x, lin.weight)
    pack = lin._idb_w8
    from inkling_draftbytes.quant import linear_ref

    q = linear_ref(x, pack).float()
    rel = ((q - ref.float()).norm() / ref.float().norm()).item()
    return {
        "wrapped": type(lin.quant_method).__name__,
        "isinstance_kept": isinstance(lin.quant_method, m.UnquantizedLinearMethod),
        "cpu_fallback_exact": bool(torch.equal(out, ref)),
        "pack_label": pack.label,
        "w8_rel_err": rel,
        "n_wrapped": sum(1 for l in model.layers for p in (l.qkv_proj, l.o_proj)
                         if type(p.quant_method).__name__.startswith("DraftBytesW8")),
    }


def dwin_paths():
    """dwin end to end on the CPU stand-ins: marks, passthroughs, identity below the window, windowed == stock
    attention over the reference key subset, metadata restored (also when the stock call raises)."""
    import os

    from inkling_draftbytes import dwin, patches

    opts, m, model = _setup()
    from sglang.srt.layers.attention import flashattention_backend as fb

    patched = hasattr(fb.FlashAttentionBackend.forward_extend, "__wrapped__")
    marks = [getattr(l.self_attn.attn, dwin.MARK, None) is not None for l in model.layers]
    P, H, D, QL = 4, 2, 8, 3
    g = torch.Generator().manual_seed(5)
    pages_total = 200
    kc = torch.randn(pages_total, P, H, D, generator=g)
    vc = torch.randn(pages_total, P, H, D, generator=g)
    be = fb.FlashAttentionBackend(kc, vc, P)
    sp, wp = dwin.DwinCfg(opts.dwin, opts.dwin_sink).pages(P)
    width = 60
    bs = 2
    table = torch.stack([torch.randperm(pages_total - 1, generator=g)[:width] + 1 for _ in range(bs)]).to(torch.int32)
    q = torch.randn(bs * QL, H, D, generator=g)
    glob, loc = model.layers[1].self_attn.attn, model.layers[0].self_attn.attn
    orig = patches.STATE.originals["FlashAttentionBackend.forward_extend"]
    out = {"patched": patched, "marks": marks, "sp": sp, "wp": wp, "cases": []}
    W = opts.dwin
    for lens in ([1, 5], [W + (sp + 1) * P - 1, 3], [W + (sp + 1) * P, 40], [150, 237], [239, 17]):
        cs = torch.tensor(lens, dtype=torch.int32)
        meta = fb.FlashAttentionMetadata(
            cache_seqlens_int32=cs, page_table=table,
            cu_seqlens_k=torch.cat([torch.zeros(1, dtype=torch.int32), cs.cumsum(0).to(torch.int32)]),
        )
        be.forward_metadata = meta
        fbatch = fb.ForwardBatch("TARGET_VERIFY")
        got = be.forward_extend(q, None, None, glob, fbatch)
        restored = be.forward_metadata is meta
        stock = orig(be, q, None, None, glob, fbatch)
        # reference: stock attention restricted to the reference key subset of each request
        ref = []
        for r in range(bs):
            keys = torch.tensor(dwin.reference_keys(lens[r], P, sp, W), dtype=torch.long)
            pg = table[r].long()[keys // P]
            kk, vv = kc[pg, keys % P], vc[pg, keys % P]
            s = torch.einsum("thd,nhd->thn", q[r * QL:(r + 1) * QL], kk) * glob.scaling
            ref.append(torch.einsum("thn,nhd->thd", torch.softmax(s, -1), vv))
        ref = torch.cat(ref)
        ident = [dwin.reference_keys(n, P, sp, W) == list(range(n)) for n in lens]
        out["cases"].append({
            "lens": lens,
            "restored": restored,
            "err_vs_ref": (got - ref).abs().max().item(),
            "equal_stock": bool(torch.equal(got, stock)),
            "identity": ident,
            "seen_lens": [int(x) for x in be.seen[-2].cache_seqlens_int32],
            "max_k": be.seen_max_k[-2:],
            "no_leftover_attr": "flash_attn_with_kvcache" not in be.__dict__,
        })
    # passthroughs: local layer, non-verify mode, topk > 1 -> exactly the stock call on the original metadata
    cs = torch.tensor([239, 200], dtype=torch.int32)
    meta = fb.FlashAttentionMetadata(cache_seqlens_int32=cs, page_table=table, cu_seqlens_k=None)
    be.forward_metadata = meta
    be.seen.clear()
    be.forward_extend(q, None, None, loc, fb.ForwardBatch("TARGET_VERIFY"))
    be.forward_extend(q, None, None, glob, fb.ForwardBatch("EXTEND"))
    be.topk = 2
    be.forward_extend(q, None, None, glob, fb.ForwardBatch("TARGET_VERIFY"))
    be.topk = 1
    out["passthrough_same_meta"] = [x is meta for x in be.seen]
    raised = False
    try:
        be.forward_extend(q, None, None, glob, fb.ForwardBatch("TARGET_VERIFY"), boom=True)
    except RuntimeError:
        raised = True
    out["raised_and_restored"] = raised and be.forward_metadata is meta
    out["stats"] = dict(dwin.STATS)
    return out


def import_fa_only():
    import inkling_draftbytes

    inkling_draftbytes.install()
    from sglang.srt.layers.attention import flashattention_backend as fb

    return {"patched": hasattr(fb.FlashAttentionBackend.forward_extend, "__wrapped__")}


def import_only():
    from sglang.srt.models import inkling2_mtp  # noqa: F401

    return {"imported": True}


if __name__ == "__main__":
    print(json.dumps(globals()[sys.argv[1]]()))
