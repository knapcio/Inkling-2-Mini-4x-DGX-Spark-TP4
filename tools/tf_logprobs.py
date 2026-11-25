#!/usr/bin/env python3
"""tf_logprobs.py dump OUT.json [--url U] [--n N] | compare A.json B.json

Teacher-forced prompt-token logprobs: each speedbench prompt (chat template applied by the server's tokenizer) is sent to
/generate with return_logprob, logprob_start_len=0, max_new_tokens=1, so every prompt token's logprob comes from the
prefill path. Comparing two engine configurations on the SAME token ids isolates prefill numerics (no generation drift).
compare: mean / p99 / max |dlogprob| over all tokens, NLL per token of each side, and argmax-agreement is not needed.
Standard library only."""
import json, sys, urllib.request, statistics, os

def post(url, path, body, timeout=600):
    req = urllib.request.Request(url + path, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=timeout))

def dump(out, url, n, prompts_path):
    ps = json.load(open(prompts_path))["prompts"]
    ps = sorted(ps, key=lambda p: p["id"])[:n]
    res = {}
    for p in ps:
        msgs = p.get("messages") or [{"role": "user", "content": p.get("prompt") or p.get("text")}]
        # tokenize through the server's own template (/tokenize accepts messages in this image's API)
        t = post(url, "/tokenize", {"model": "inkling-2-mini", "messages": msgs})
        ids = t.get("tokens") or t.get("input_ids") or t.get("ids")
        r = post(url, "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                    "return_logprob": True, "logprob_start_len": 0})
        lp = [x[0] for x in r["meta_info"]["input_token_logprobs"] if x[0] is not None]
        res[p["id"]] = {"n": len(ids), "lp": lp}
        print(p["id"], len(ids), f"nll/tok {-sum(lp)/max(len(lp),1):.4f}", file=sys.stderr)
    json.dump(res, open(out, "w"))

def dumpx(out, url, n, prompts_path, tail=512, min_len=700):
    """Extend-with-prefix: cache ids[:split] with a 1-token request, then ask for logprobs from `split` on, so the
    suffix runs as an extend over cached KV (the prefix-hit path speedbench and the DFlash2 prefix tail use)."""
    ps = sorted(json.load(open(prompts_path))["prompts"], key=lambda p: p["id"])[:n]
    res = {}
    for p in ps:
        ids = post(url, "/tokenize", {"model": "inkling-2-mini", "messages": p["messages"]})["tokens"]
        if len(ids) < min_len:
            continue
        split = len(ids) - tail
        post(url, "/generate", {"input_ids": ids[:split], "sampling_params": {"max_new_tokens": 1, "temperature": 0}})
        # the first hit re-prefills from the nearest hybrid-state checkpoint and plants one at `split`; the second hit
        # resumes there, so its suffix is a true extend over cached KV
        for _ in range(2):
            r = post(url, "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                        "return_logprob": True, "logprob_start_len": split})
        lp = [x[0] for x in r["meta_info"]["input_token_logprobs"] if x[0] is not None]
        res[p["id"]] = {"n": len(ids), "split": split, "lp": lp,
                        "cached": r["meta_info"].get("cached_tokens")}
        print(p["id"], len(ids), "cached", res[p["id"]]["cached"], file=sys.stderr)
    json.dump(res, open(out, "w"))

def dumpdocs(out, url, n_docs=12, chars=16000):
    """Teacher-forced logprobs over deterministic speedbench long documents (~4K tokens each: every prefill is one
    >= 512-row extend, the path the prefill levers change). Kept near 4K tokens: full-vocabulary prompt logits cost
    ~0.8 GB per 1K tokens of GPU memory."""
    sys.path.insert(0, os.path.expanduser("~/speedbench-6733653"))
    from speedbench import longdoc
    res = {}
    for i in range(n_docs):
        msgs = longdoc.messages(longdoc.document(chars, seed=100 + i), longdoc.QUESTIONS[i % len(longdoc.QUESTIONS)])
        ids = post(url, "/tokenize", {"model": "inkling-2-mini", "messages": msgs})["tokens"]
        r = post(url, "/generate", {"input_ids": ids, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                                    "return_logprob": True, "logprob_start_len": 0})
        lp = [x[0] for x in r["meta_info"]["input_token_logprobs"] if x[0] is not None]
        res[f"doc{i:02d}"] = {"n": len(ids), "lp": lp}
        print(f"doc{i:02d}", len(ids), f"nll/tok {-sum(lp)/len(lp):.4f}", file=sys.stderr)
    json.dump(res, open(out, "w"))

def compare(a, b):
    A, B = json.load(open(a)), json.load(open(b))
    d, na, nb, k = [], 0.0, 0.0, 0
    for key in sorted(set(A) & set(B)):
        la, lb = A[key]["lp"], B[key]["lp"]
        if len(la) != len(lb):  # an extend-with-prefix dump holds the tail only: compare the common tail
            m = min(len(la), len(lb), 256)
            la, lb = la[-m:], lb[-m:]
        d += [abs(x - y) for x, y in zip(la, lb)]; na -= sum(la); nb -= sum(lb); k += len(la)
    d.sort()
    per = []
    for key in sorted(set(A) & set(B)):
        if len(A[key]["lp"]) == len(B[key]["lp"]) and len(A[key]["lp"]) >= 511:
            per.append(round(-sum(B[key]["lp"]) / len(B[key]["lp"]) + sum(A[key]["lp"]) / len(A[key]["lp"]), 4))
    print(json.dumps({"tokens": k, "long_prompt_dNLL_B_minus_A": per, "mean_abs_dlogprob": round(sum(d) / k, 5), "p99": round(d[int(0.99 * (k - 1))], 4),
                      "max": round(d[-1], 4), "nll_per_tok_A": round(na / k, 5), "nll_per_tok_B": round(nb / k, 5)}))

if __name__ == "__main__":
    if sys.argv[1] == "dumpdocs":
        args = sys.argv[2:]
        dumpdocs(args[0], args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8000")
    elif sys.argv[1] == "dumpx":
        args = sys.argv[2:]
        url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8000"
        dumpx(args[0], url, 99, os.path.expanduser("~/speedbench-6733653/prompts/speedbench-v1.json"))
    elif sys.argv[1] == "dump":
        args = sys.argv[2:]; out = args[0]
        url = args[args.index("--url") + 1] if "--url" in args else "http://127.0.0.1:8000"
        n = int(args[args.index("--n") + 1]) if "--n" in args else 30
        pp = args[args.index("--prompts") + 1] if "--prompts" in args else os.path.expanduser("~/speedbench-6733653/prompts/speedbench-v1.json")
        dump(out, url, n, pp)
    else:
        compare(sys.argv[2], sys.argv[3])
