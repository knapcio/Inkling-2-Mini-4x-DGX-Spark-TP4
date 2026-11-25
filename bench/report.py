#!/usr/bin/env python3
"""report.py DIR [DIR ...]: README tables from the files bench/full.sh wrote (one DIR per run).

With several DIRs (repeats of one configuration) every cell is the median over the runs, and the per-run values
are printed under each table.
"""
import glob
import json
import os
import statistics
import sys


def one(d, pat):
    f = glob.glob(os.path.join(d, pat))
    return f[0] if f else None


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(statistics.median(xs), 1) if xs else None


def main():
    dirs = sys.argv[1:]
    runs = []
    for d in dirs:
        r = {"dir": d}
        f = one(d, "*-speedbench.json")
        if f:
            s = json.load(open(f))["summary"]
            r["sb"] = {k: v.get("geomean_tok_s") for k, v in s.items() if isinstance(v, dict) and "geomean_tok_s" in v}
            r["sb_ttft"] = s.get("ALL", {}).get("median_ttft_ms")
            r["sb_failed"] = s.get("failed")
            r["sb_tps"] = s.get("ALL", {}).get("median_tokens_per_step")
        f = one(d, "*-decode.json")
        if f:
            r["dec"] = json.load(open(f))["table"]
        f = one(d, "*-think.jsonl")
        if f:
            r["think"] = {json.loads(x)["concurrency"]: json.loads(x) for x in open(f) if x.strip().startswith("{")}
        f = one(d, "*-longctx.json")
        if f:
            r["lc"] = {row["context"]: row for row in json.load(open(f))["table"]}
        f = one(d, "*-mem.tsv")
        if f:
            v = [float(x.split("\t")[1]) for x in open(f) if "\t" in x]
            r["mem_min"] = min(v) if v else None
        runs.append(r)

    cats = ["prose", "code", "structured", "agentic", "multilingual", "longctx", "ALL"]
    print("### speedbench full (c1, T=0, effort low): decode tok/s geomean")
    print("| " + " | ".join(c if c != "ALL" else "all 73 prompts" for c in cats) + " |")
    print("|" + "---:|" * len(cats))
    print("| " + " | ".join(str(med([r.get("sb", {}).get(c) for r in runs])) for c in cats) + " |")
    for r in runs:
        print("  %s: %s ttft %s ms, failed %s, tok/step %s" % (r["dir"], {c: r.get("sb", {}).get(c) for c in cats},
                                                            r.get("sb_ttft"), r.get("sb_failed"), r.get("sb_tps")))

    print("\n### decode table (T=0, 256 tokens, effort none): c1 per-stream, c>1 aggregate (per stream)")
    concs = sorted({int(c) for r in runs for t in r.get("dec", {}).values() for c in t})
    print("| prompt type | " + " | ".join("c%d" % c for c in concs) + " |")
    print("|---|" + "---:|" * len(concs))
    for t in ["prose", "code", "structured", "json"]:
        cells = []
        for c in concs:
            agg = med([r["dec"][t][str(c)]["aggregate_tok_s"] for r in runs if t in r.get("dec", {})])
            ps = med([r["dec"][t][str(c)]["per_stream_tok_s"] for r in runs if t in r.get("dec", {})])
            cells.append(str(ps) if c == 1 else "%s (%s)" % (agg, ps))
        print("| %s | %s |" % (t, " | ".join(cells)))
    for r in runs:
        if "dec" in r:
            print("  %s: %s" % (r["dir"], {t: [r["dec"][t][str(c)]["aggregate_tok_s"] for c in concs]
                                         for t in r["dec"]}))

    print("\n### thinking load (effort high, T=0.8): aggregate tok/s (median per-request decode)")
    tc = sorted({c for r in runs for c in r.get("think", {})})
    print("| " + " | ".join("c%d" % c for c in tc) + " |")
    print("|" + "---:|" * len(tc))
    print("| " + " | ".join("%s (%s)" % (med([r["think"][c]["aggregate_tok_s"] for r in runs if c in r.get("think", {})]),
                                         med([r["think"][c]["per_request_decode_tok_s_median"] for r in runs
                                              if c in r.get("think", {})])) for c in tc) + " |")
    for r in runs:
        print("  %s: %s" % (r["dir"], {c: (r["think"][c]["aggregate_tok_s"], r["think"][c]["length_finishes"],
                                         r["think"][c]["failed"]) for c in r.get("think", {})}))

    print("\n### prefill / long context")
    ctxs = sorted({c for r in runs for c in r.get("lc", {})})
    print("| | " + " | ".join("%dK" % (c // 1024) for c in ctxs) + " |")
    print("|---|" + "---:|" * len(ctxs))
    for key, lab in (("prompt_tokens", "prompt tokens"), ("ttft_cold_s", "TTFT cold, s"),
                     ("prefill_tok_s", "prefill tok/s (cold)"), ("ttft_cached_s", "TTFT cached, s"),
                     ("decode_tok_s", "decode tok/s at that context")):
        print("| %s | %s |" % (lab, " | ".join(str(med([r["lc"][c].get(key) for r in runs if c in r.get("lc", {})]))
                                              for c in ctxs)))
    for r in runs:
        print("  %s: %s; MemAvailable min during run %s GiB" % (
            r["dir"], {c: r["lc"][c].get("prefill_tok_s") for c in r.get("lc", {})}, r.get("mem_min")))


if __name__ == "__main__":
    main()
