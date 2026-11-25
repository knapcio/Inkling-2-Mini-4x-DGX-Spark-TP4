#!/usr/bin/env python3
"""Thinking-workload throughput: C concurrent agents that think for thousands of tokens.

Python 3 standard library only; OpenAI-compatible /v1/chat/completions (SGLang, vLLM).

Workload: seeded synthetic arithmetic puzzles (not from any evaluation set), reasoning effort high (the chat
template's default), temperature 0.8 with a per-request seed, max_tokens fixed (default 4096), streamed. No early
stop besides the model's own end. This is the shape of an agent swarm: long reasoning, many requests at once.

Reported: aggregate tok/s = all completion tokens (reasoning + answer) / wall time of the run, median per-request
decode tok/s, median TTFT, and how many requests hit max_tokens.
"""
import argparse
import datetime
import json
import random
import statistics
import sys
import threading
import time
import urllib.request

SYSTEM = "Solve the precise arithmetic task. Return only the requested JSON object. Do not explain or use tools."


def puzzle(rng):
    nums = [rng.randint(1, 13) for _ in range(5)]
    target = rng.randint(100, 999)
    return ("Numbers: %s. Target: %d. Using each number at most once and only + - * / with positive integer "
            "intermediate results, reach the target exactly. Return {\"steps\": [...], \"result\": n}." % (nums, target))


def one(a, prompt, seed, out, lock):
    body = {"model": a.model, "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "max_tokens": a.max_tokens, "temperature": a.temperature, "seed": seed, "stream": True,
            "stream_options": {"include_usage": True}}
    if a.effort:
        body["chat_template_kwargs"] = {"reasoning_effort": a.effort}
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    usage, finish, err = {}, None, None
    try:
        with urllib.request.urlopen(req, timeout=a.timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[5:])
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                    finish = c.get("finish_reason") or finish
                usage = ch.get("usage") or usage
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    n = usage.get("completion_tokens") or 0
    rec = {"tokens": n, "finish": finish, "error": err, "ttft_s": (first - t0) if first else None,
           "decode_tok_s": (n - 1) / (last - first) if first and last and last > first and n > 1 else None}
    with lock:
        out.append(rec)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--requests", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--effort", default="high")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    queue = [(puzzle(rng), 1000 * a.seed + i) for i in range(a.requests)]
    out, lock = [], threading.Lock()

    def worker():
        while True:
            with lock:
                if not queue:
                    return
                prompt, seed = queue.pop(0)
            one(a, prompt, seed, out, lock)
    started = datetime.datetime.now().astimezone().isoformat(timespec="seconds")
    t0 = time.monotonic()
    th = [threading.Thread(target=worker) for _ in range(a.concurrency)]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.monotonic() - t0
    ok = [r for r in out if not r["error"] and r["finish"]]
    dec = [r["decode_tok_s"] for r in ok if r["decode_tok_s"]]
    total = sum(r["tokens"] for r in ok)
    print(json.dumps({"label": a.label, "started": started, "concurrency": a.concurrency, "requests": len(out),
                      "failed": len(out) - len(ok), "max_tokens": a.max_tokens, "temperature": a.temperature,
                      "effort": a.effort, "wall_s": round(wall, 2), "completion_tokens": total,
                      "aggregate_tok_s": round(total / wall, 1),
                      "per_request_decode_tok_s_median": round(statistics.median(dec), 1) if dec else None,
                      "ttft_median_s": round(statistics.median(r["ttft_s"] for r in ok if r["ttft_s"]), 2) if ok else None,
                      "length_finishes": sum(r["finish"] == "length" for r in ok)}))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
