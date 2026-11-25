"""CPU checks: (1) the pre-patch applies to the pinned patch and yields valid Python; (2) the per-shard clamp's
gathered top-k equals the TP1 top-k over the real vocabulary, for random logits, at TP2 and TP4."""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import tp4_dflash2_prepatch as pp

PINNED = os.environ.get("DFLASH2_PATCH")
if PINNED:
    src = open(PINNED).read()
    out = pp.prepatch(src)
    compile(out, "patched", "exec")
    assert "validated for TP1" not in out and "org_vocab_start_index" in out
    print("prepatch applies to the pinned file: OK")

VOCAB_PAD, VALID, K = 201024, 200058, 16
g = torch.Generator().manual_seed(0)
for trial in range(50):
    rows = 8
    logits = torch.randn(rows, VOCAB_PAD, generator=g)
    # make the padded tail attractive: a clamp bug would leak a padded id
    logits[:, VALID:] += 10.0
    ref_v, ref_i = torch.topk(logits[:, :VALID], K, dim=-1)
    for tp in (2, 4):
        per = VOCAB_PAD // tp
        vals, ids = [], []
        for r in range(tp):
            start = r * per
            num_org = max(0, min(per, VALID - start))
            v, i = torch.topk(logits[:, start:start + num_org], K, dim=-1)
            vals.append(v); ids.append(i + start)
        gv, gi = torch.cat(vals, -1), torch.cat(ids, -1)
        tv, sel = torch.topk(gv, K, dim=-1)
        ti = torch.gather(gi, -1, sel)
        assert torch.equal(tv, ref_v), (trial, tp)
        assert torch.equal(torch.sort(ti, -1).values, torch.sort(ref_i, -1).values), (trial, tp)
print("per-shard clamp == TP1 candidate set (50 trials, TP2/TP4): OK")
