#!/usr/bin/env python3
"""trace_buckets.py TRACE.json.gz [STEPS]: GPU timeline occupancy per kernel bucket from an SGLang /start_profile trace.

Kernels over 5 ms are dropped with their interval (start-up collectives). Intervals are merged per bucket (parallel
streams are not double counted), so buckets can overlap and need not sum to the window. STEPS divides the totals.
"""
import gzip, json, re, sys
from collections import defaultdict

BUCKETS = [
    ("roce", r"roce|b12x|oneshot|one_shot"),
    ("nccl", r"nccl"),
    ("moe_marlin", r"marlin|moe|expert|topk_softmax|route"),
    ("attention", r"fmha|flash|fa4|attn|attention|softmax_.*kv|split_kv|rel_bias"),
    ("gemm", r"gemm|gemv|cutlass|sm90|sm100|sm120|cublas|matmul|nvjet|xmma"),
    ("lm_head/sampling", r"topk|radix|sampl|argmax|logsumexp|softcap|reject|selector|candidate"),
    ("norm/elementwise", r"norm|elementwise|vectorized|reduce|copy|fill|cat|index|conv|silu|act|mul|add"),
]

def bucket(name):
    n = name.lower()
    for b, rx in BUCKETS:
        if re.search(rx, n):
            return b
    return "other"

def merged(iv):
    iv.sort(); tot = 0; cs = ce = None
    for s, e in iv:
        if cs is None or s > ce:
            if cs is not None: tot += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    if cs is not None: tot += ce - cs
    return tot

ev = json.load(gzip.open(sys.argv[1]))
ev = ev["traceEvents"] if isinstance(ev, dict) else ev
steps = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
k = [e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
k = [e for e in k if e.get("dur", 0) < 5000]
per = defaultdict(list); names = defaultdict(float); allv = []
for e in k:
    s, d = e["ts"], e["dur"]; b = bucket(e["name"])
    per[b].append((s, s + d)); allv.append((s, s + d)); names[(b, e["name"][:90])] += d
win = (max(e for _, e in allv) - min(s for s, _ in allv)) if allv else 0
busy = merged(allv)
print(f"window {win/1e3:.1f} ms, GPU busy {busy/1e3:.1f} ms ({100*busy/max(win,1):.0f} %), per step /{steps:g}: "
      f"window {win/1e3/steps:.2f} ms, busy {busy/1e3/steps:.2f} ms")
for b in sorted(per, key=lambda b: -merged(per[b])):
    m = merged(per[b]); print(f"  {b:18s} {m/1e3/steps:7.2f} ms/step  ({100*m/max(win,1):4.1f} % of window)  n={len(per[b])/steps:.0f}/step")
print("top kernels (sum of durations per step):")
for (b, n), d in sorted(names.items(), key=lambda x: -x[1])[:25]:
    print(f"  {d/1e3/steps:7.3f} ms  [{b}] {n}")
