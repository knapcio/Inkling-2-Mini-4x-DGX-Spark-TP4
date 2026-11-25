#!/usr/bin/env python3
"""Decode table: prompt type x concurrency, aggregate tok/s and per-stream tok/s.

Python 3 standard library only. Talks to any OpenAI-compatible /v1/chat/completions endpoint (SGLang, vLLM).

Prompts are taken from speedbench's frozen prompt set (speedbench-v1, https://github.com/knapcio/speedbench), split
into four types:
  prose       the 18 prose prompts
  code        the 14 code prompts
  structured  the structured prompts that ask for markdown tables, YAML, TOML, SQL, CSV, XML, GraphQL or OpenAPI
  json        the prompts that ask for JSON output (structured-01/05/10, agentic-10/12)

For every type and concurrency level c, c streams run closed-loop; each stream sends ROUNDS requests one after the
other, starting at a different prompt of the type. When a level needs more requests than the type has prompts,
the extra ones carry a variant tag on the user message ("(Request tag: N)"), so no two requests of a level replay
the same greedy trajectory. Settings: temperature 0, max_tokens 256 (ignore_eos is never sent), streaming with
usage.

Reported per cell (the markdown table shows per_stream_tok_s at c1, aggregate (per stream) above c1):
  aggregate_tok_s   all completion tokens of the level / wall time of the level (first send to last finish;
                    includes each request's time to first token)
  per_stream_tok_s  median over requests of (completion_tokens - 1) / (last token event - first token event)
  ttft_median_s     median time to first streamed token
Before the table, WARMUP requests per type are sent and discarded.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import threading
import time
import urllib.request

JSON_IDS = ["structured-01", "structured-05", "structured-10", "agentic-10", "agentic-12"]


def load_types(path):
    ps = {p["id"]: p for p in json.load(open(path, encoding="utf-8"))["prompts"]}
    by = {"prose": [], "code": [], "structured": [], "json": []}
    for pid, p in sorted(ps.items()):
        if p.get("tools"):
            continue
        if pid in JSON_IDS:
            by["json"].append(p)
        elif p["category"] in by:
            by[p["category"]].append(p)
    return by


def post_stream(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    usage, finish, err = {}, None, None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[5:])
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning") or d.get("tool_calls"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                    finish = c.get("finish_reason") or finish
                usage = ch.get("usage") or usage
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    t1 = time.monotonic()
    n = usage.get("completion_tokens")
    ok = err is None and n and first is not None and finish is not None
    rec = {"ok": bool(ok), "t_send": t0, "t_end": t1, "completion_tokens": n,
           "prompt_tokens": usage.get("prompt_tokens"), "finish": finish, "error": err}
    if ok:
        rec["ttft_s"] = first - t0
        rec["decode_tok_s"] = (n - 1) / (last - first) if last > first and n > 1 else None
    return rec


def body_for(p, tag, a):
    msgs = [dict(m) for m in p["messages"]]
    if tag:
        msgs[-1]["content"] = msgs[-1]["content"] + "\n\n(Request tag: %d)" % tag
    b = {"model": a.model, "messages": msgs, "max_tokens": a.max_tokens, "temperature": 0.0, "stream": True,
         "stream_options": {"include_usage": True}}
    if a.effort:
        if a.effort_field in ("template", "both"):
            b["chat_template_kwargs"] = {"reasoning_effort": a.effort}
        if a.effort_field in ("top", "both"):
            b["reasoning_effort"] = a.effort
    return b


def level(url, prompts, c, rounds, a):
    jobs = []
    n = len(prompts)
    for s in range(c):
        for r in range(rounds):
            k = s * rounds + r                      # distinct index per request of this level
            jobs.append((s, prompts[k % n], k // n))
    recs, lock = [], threading.Lock()

    def worker(s):
        for (ws, p, tag) in jobs:
            if ws != s:
                continue
            rec = post_stream(url, body_for(p, tag, a), a.timeout)
            rec.update({"id": p["id"], "tag": tag, "stream": s})
            with lock:
                recs.append(rec)
    t0 = time.monotonic()
    th = [threading.Thread(target=worker, args=(s,)) for s in range(c)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.monotonic() - t0
    ok = [r for r in recs if r["ok"]]
    dec = [r["decode_tok_s"] for r in ok if r.get("decode_tok_s")]
    toks = sum(r["completion_tokens"] for r in ok)
    cell = {"conc": c, "requests": len(recs), "failed": len(recs) - len(ok), "wall_s": round(wall, 2),
            "completion_tokens": toks, "aggregate_tok_s": round(toks / wall, 1) if ok else None,
            "per_stream_tok_s": round(statistics.median(dec), 1) if dec else None,
            "ttft_median_s": round(statistics.median(r["ttft_s"] for r in ok), 3) if ok else None,
            "length_finishes": sum(r["finish"] == "length" for r in ok)}
    for r in recs:
        r["t_send"] = round(r["t_send"] - t0, 3)
        r["t_end"] = round(r["t_end"] - t0, 3)
    return cell, recs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(__file__), "..", "speedbench", "prompts",
                                                     "speedbench-v1.json"),
                    help="speedbench-v1.json from a speedbench checkout")
    ap.add_argument("--types", default="prose,code,structured,json")
    ap.add_argument("--conc", default="1,2,4,8")
    ap.add_argument("--rounds", default="1:3,*:2", help="requests per stream: 'c:n' pairs, '*' = default")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--effort", default="none", help="reasoning effort ('' sends none)")
    ap.add_argument("--effort-field", choices=["template", "top", "both"], default="template")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    types = load_types(a.prompts)
    rounds = {}
    for kv in a.rounds.split(","):
        k, v = kv.split(":")
        rounds[k] = int(v)
    concs = [int(x) for x in a.conc.split(",")]
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    table, allrecs = {}, []
    for t in a.types.split(","):
        ps = types[t]
        for i in range(a.warmup):
            post_stream(a.url, body_for(ps[i % len(ps)], 1000 + i, a), a.timeout)
        for c in concs:
            cell, recs = level(a.url, ps, c, rounds.get(str(c), rounds.get("*", 2)), a)
            table.setdefault(t, {})[str(c)] = cell
            for r in recs:
                r.update({"type": t, "conc": c})
            allrecs += recs
            print("%-10s c%-2d aggregate %7s tok/s  per-stream %6s  ttft %s s  failed %d" % (
                t, c, cell["aggregate_tok_s"], cell["per_stream_tok_s"], cell["ttft_median_s"], cell["failed"]),
                file=sys.stderr, flush=True)
    doc = {"schema": "inkling-decode-table/1", "label": a.label, "started": started,
           "settings": {"url": a.url, "model": a.model, "types": a.types, "conc": concs, "rounds": a.rounds,
                        "max_tokens": a.max_tokens, "temperature": 0.0, "effort": a.effort,
                        "effort_field": a.effort_field, "warmup": a.warmup,
                        "prompt_set": os.path.basename(a.prompts)},
           "table": table, "requests": allrecs}
    out = a.out or "decode-%s-%s.json" % (a.label or "run", time.strftime("%Y%m%d-%H%M%S"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=1)
    # markdown
    print("| prompt type | " + " | ".join("c%d" % c for c in concs) + " |")
    print("|---|" + "---:|" * len(concs))
    for t in table:
        cells = []
        for c in concs:
            x = table[t][str(c)]
            cells.append("%s" % x["per_stream_tok_s"] if c == 1 else "%s (%s)" % (x["aggregate_tok_s"],
                                                                                x["per_stream_tok_s"]))
        print("| %s | %s |" % (t, " | ".join(cells)))
    print("wrote", out, file=sys.stderr)


if __name__ == "__main__":
    main()
