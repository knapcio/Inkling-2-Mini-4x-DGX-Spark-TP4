#!/usr/bin/env python3
"""gen_logprobs.py dump OUT.json [--url U] [--n N] [--tokens T] [--conc C] | compare REF.json B.json

Greedy generation (T=0, ignore_eos) with output-token logprobs: these come from the target's verify / decode forwards,
so comparing two configurations on their COMMON token prefix tests the decode-side kernels (pico12 GEMV <= 16 rows,
wide GEMM 17-256 rows, verify attention) that prompt-logprob checks never reach. --conc C sends C prompts at once
(verify batches of 8*C rows). compare: common-prefix length (first divergence), mean / p99 / max |dlogprob| on the
common prefix, and the delta of the chosen token's logprob at the divergence point (a near-tie there is FP-order
noise; a large gap is a bug). Standard library only."""
import json, sys, os, urllib.request, concurrent.futures as cf

def post(url, path, body, timeout=900):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def one(url, p, tokens):
    t = post(url, "/tokenize", {"model": "inkling-2-mini", "messages": p["messages"]})
    ids = t["tokens"]
    r = post(url, "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": tokens, "temperature": 0,
                                "ignore_eos": True}, "return_logprob": True, "logprob_start_len": len(ids)})
    o = r["meta_info"]["output_token_logprobs"]
    return p["id"], {"tok": [x[1] for x in o], "lp": [x[0] for x in o]}

def dump(out, url, n, tokens, conc, prompts_path):
    ps = sorted(json.load(open(prompts_path))["prompts"], key=lambda p: p["id"])[:n]
    res = {}
    with cf.ThreadPoolExecutor(conc) as ex:
        for k, v in ex.map(lambda p: one(url, p, tokens), ps):
            res[k] = v
    json.dump(res, open(out, "w"))
    print(f"{len(res)} prompts x {tokens} tokens at c{conc}", file=sys.stderr)

def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    d, pref, div_gaps, full = [], [], [], 0
    for k in sorted(set(A) & set(B)):
        ta, tb, la, lb = A[k]["tok"], B[k]["tok"], A[k]["lp"], B[k]["lp"]
        m = 0
        while m < min(len(ta), len(tb)) and ta[m] == tb[m]:
            m += 1
        pref.append(m); full += m == min(len(ta), len(tb))
        d += [abs(x - y) for x, y in zip(la[:m], lb[:m])]
        if m < min(len(ta), len(tb)):
            div_gaps.append(round(abs(la[m] - lb[m]), 3))
    d.sort(); k = len(d)
    print(json.dumps({"prompts": len(pref), "identical": full, "median_common_prefix": sorted(pref)[len(pref) // 2],
                      "tokens_compared": k, "mean_abs_dlogprob": round(sum(d) / max(k, 1), 5),
                      "p99": round(d[int(0.99 * (k - 1))], 4) if k else None, "max": round(d[-1], 4) if k else None,
                      "divergence_chosen_lp_gaps": div_gaps}))

if __name__ == "__main__":
    a = sys.argv[2:]
    if sys.argv[1] == "dump":
        g = lambda f, d: a[a.index(f) + 1] if f in a else d
        dump(a[0], g("--url", "http://127.0.0.1:8000"), int(g("--n", 30)), int(g("--tokens", 256)), int(g("--conc", 1)),
             g("--prompts", os.path.expanduser("~/speedbench-6733653/prompts/speedbench-v1.json")))
    else:
        compare(a[0], a[1])
