#!/usr/bin/env python3
"""ag64.py: N concurrent agents with long distinct contexts, then further turns on the same conversations (prefix-cache
hits expected), plus the same rounds with short contexts as the reference. Ported from the single-Spark recipe's
AG64 harness (r2/tracks/ag64/ag64.py, the test behind v2.9.2's prompt-end cache touch): same documents (speedbench
longdoc, same seeds and salts), same follow-up, same sampling, so runs compare with the 1x AG64 / AG292 results.

Thinking on (template default effort), T 0.8, max_tokens fixed, streamed. Turn k+1 sends turn k's conversation plus
the assistant's answer (the template drops the reasoning) and a follow-up question, so its prefix is turn k's prompt.

Needs the server's --enable-cache-report (otherwise the API returns no cached_tokens); a preflight checks it and
stops the run unless --allow-no-cache-report. At TP4: `scripts/launch-tp4.sh up --enable-cache-report`.

Cold turn 1: every request carries the API's cache_salt, fresh per run and agent and constant across that agent's
turns (ag64-<run id>-<phase>-<agent>), so a rerun never hits the cache an earlier run left; the documents themselves
(longdoc seeds and salt lines) stay those of the 1x harness. A stream that ends in an error (HTTP or an SSE error
frame) or without a finish reason counts as failed; a turn is classified only when it and the agent's previous turn
both completed.

Per phase (short-turn1, short-turn2, ..., long-turn1, long-turn2, ...):
  cache: per request cached_tokens / prompt_tokens from the API usage. For turn >= 2 each agent is classified
         against its previous turn's prompt_tokens (the reusable prefix): hit >= 90 %, miss < 10 % (a whole-context
         miss: the prompt is prefilled again), partial in between. Turn-1 requests are cold by construction (salted).
  aggregate decode tok/s = all completion tokens / (last token time - first token time over the phase)
  aggregate wall tok/s   = all completion tokens / phase wall time (prefill included)
  per-agent decode tok/s = (tokens - 1) / (last - first token) of that request
  TTFT min / p50 / p90 / max (a whole-context miss shows up as a turn-2 TTFT of minutes instead of seconds)
plus the end-to-end wall time of the whole run and the head node's MemAvailable minimum (this host only).

  python3 bench/ag64.py --out results/AG-tp4-1                  # 24 agents x ~72K tokens, short + long, 2 turns
  python3 bench/ag64.py --out results/AG-tp4-2 --phases long    # long contexts only
"""
import argparse
import json
import os
import secrets
import statistics
import sys
import threading
import time
import urllib.request

FOLLOW_UPS = [
    "Thanks. Now list the three facts from the document that mattered most for your answer, one line each.",
    "Good. Which part of the document would change your answer the most if it were wrong? One short paragraph.",
    "Finally, give a one-sentence summary of the document for a busy reader.",
]
HIT, MISS = 0.9, 0.1


def post(url, body, timeout):
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def stream(url, model, messages, max_tokens, temperature, seed, timeout, cache_salt=None):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
            "seed": seed, "stream": True, "stream_options": {"include_usage": True}}
    if cache_salt is not None:
        body["cache_salt"] = cache_salt
    req = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    usage, finish, err, text = {}, None, None, []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ch = json.loads(line[5:])
                if ch.get("error") is not None or ch.get("object") == "error":   # SSE error frame
                    e = ch.get("error", ch)
                    err = "stream error: " + (json.dumps(e) if not isinstance(e, str) else e)[:300]
                    break
                for c in ch.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        now = time.monotonic()
                        first = first or now
                        last = now
                    if d.get("content"):
                        text.append(d["content"])
                    finish = c.get("finish_reason") or finish
                usage = ch.get("usage") or usage
    except Exception as e:  # noqa: BLE001
        err = repr(e)
    n = usage.get("completion_tokens") or 0
    t_end = time.monotonic()
    return {"t_start": t0, "t_first": first, "t_last": last, "t_end": t_end, "tokens": n,
            "prompt_tokens": usage.get("prompt_tokens"),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "finish": finish, "error": err, "ttft_s": (first - t0) if first else None,
            "e2e_s": t_end - t0,
            "decode_tok_s": (n - 1) / (last - first) if first and last and last > first and n > 1 else None,
            "answer": "".join(text)}


def completed(r):
    """The request streamed to a regular finish (no HTTP / SSE error, a finish reason that is not an abort)."""
    return r is not None and not r["error"] and r["finish"] is not None and r["finish"] not in ("abort", "error")


def classify(cached, prev_prompt):
    if cached is None or not prev_prompt:
        return "unknown"
    f = cached / prev_prompt
    return "hit" if f >= HIT else ("miss" if f < MISS else "partial")


def phase(a, convs, tag, seed0, prev=None):
    """prev: the previous turn's records (same agent order), for the hit / miss classification."""
    out = [None] * len(convs)
    t0 = time.monotonic()

    def run(i):
        out[i] = stream(a.url, a.model, convs[i], a.max_tokens, a.temperature, seed0 + i, a.timeout,
                        "%s-%s-%d" % (a.run_id, tag.split("-")[0], i))
    th = [threading.Thread(target=run, args=(i,)) for i in range(len(convs))]
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.monotonic() - t0
    for i, r in enumerate(out):
        r["agent"] = i
        r["phase"] = tag
        if prev is not None:
            r["prev_prompt_tokens"] = prev[i]["prompt_tokens"]
            if not completed(r):
                r["cache"] = "failed"
            elif not completed(prev[i]):
                r["cache"] = "prev_failed"
            else:
                r["cache"] = classify(r["cached_tokens"], prev[i]["prompt_tokens"])
    ok = [r for r in out if completed(r)]
    firsts = [r["t_first"] for r in ok if r["t_first"]]
    lasts = [r["t_last"] for r in ok if r["t_last"]]
    tot = sum(r["tokens"] for r in ok)
    ttft = sorted(r["ttft_s"] for r in ok if r["ttft_s"] is not None)
    dec = sorted(r["decode_tok_s"] for r in ok if r["decode_tok_s"])

    def q(v, p):
        return round(v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))], 3) if v else None
    cached = [r["cached_tokens"] for r in ok]
    cache = {"report": any(c is not None for c in cached),
             "cached_tokens_sum": sum(c or 0 for c in cached),
             "prompt_tokens_sum": sum(r["prompt_tokens"] or 0 for r in ok)}
    cache["cached_frac"] = round(cache["cached_tokens_sum"] / cache["prompt_tokens_sum"], 4) \
        if cache["prompt_tokens_sum"] else None
    if prev is not None:
        for k, name in (("hit", "hits"), ("partial", "partials"), ("miss", "misses"), ("unknown", "unknown"),
                        ("failed", "failed"), ("prev_failed", "prev_failed")):
            cache[name] = sum(r.get("cache") == k for r in out)
        cache["miss_agents"] = [r["agent"] for r in out if r.get("cache") == "miss"]
        cache["partial_agents"] = [r["agent"] for r in out if r.get("cache") == "partial"]
    summ = {"phase": tag, "requests": len(out), "failed": len(out) - len(ok), "wall_s": round(wall, 1),
            "completion_tokens": tot,
            "prompt_tokens_median": statistics.median([r["prompt_tokens"] for r in ok if r["prompt_tokens"]] or [0]),
            "cached_tokens_median": statistics.median([c or 0 for c in cached] or [0]),
            "cache": cache,
            "aggregate_decode_tok_s": round(tot / (max(lasts) - min(firsts)), 1) if firsts and lasts else None,
            "aggregate_wall_tok_s": round(tot / wall, 1),
            "per_agent_decode_tok_s": {"min": q(dec, 0), "median": q(dec, 0.5), "max": q(dec, 1)},
            "ttft_s": {"min": q(ttft, 0), "p50": q(ttft, 0.5), "p90": q(ttft, 0.9), "max": q(ttft, 1)},
            "length_finishes": sum(r["finish"] == "length" for r in ok)}
    print(json.dumps(summ), flush=True)
    if prev is not None:
        print("  agent  prompt  cached  class    ttft_s", flush=True)
        for r in out:
            print("  %5d %7s %7s  %-7s %7s" % (r["agent"], r["prompt_tokens"], r["cached_tokens"], r.get("cache"),
                                              "%.1f" % r["ttft_s"] if r["ttft_s"] is not None else "-"), flush=True)
    return summ, out


def preflight(a, longdoc):
    """Two identical ~2K-token requests: the second must report cached_tokens (the server's --enable-cache-report;
    None = not reported, 0 = reported but no hit)."""
    doc = longdoc.document(6000, seed=7, salt="ag64-preflight-%s" % a.run_id)
    msgs = longdoc.messages(doc, "Reply with the word ok.")
    body = {"model": a.model, "messages": msgs, "max_tokens": 1, "temperature": 0,
            "cache_salt": "%s-preflight" % a.run_id}
    post(a.url, body, 300)
    u = post(a.url, body, 300).get("usage") or {}
    return (u.get("prompt_tokens_details") or {}).get("cached_tokens")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--agents", type=int, default=24)
    ap.add_argument("--turns", type=int, default=2, help="turns per conversation (2 = the AG64 test; max %d)"
                    % (len(FOLLOW_UPS) + 1))
    ap.add_argument("--phases", default="short,long", help="short, long or both (short first)")
    ap.add_argument("--long-tokens", type=int, default=64000, help="~72K served tokens at the default --cpt")
    ap.add_argument("--short-tokens", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--cpt", type=float, default=3.275)
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sb = os.environ.get("SPEEDBENCH") or next(
        (p for p in (os.path.join(here, "speedbench"), os.path.expanduser("~/speedbench-6733653")) if os.path.isdir(p)),
        os.path.join(here, "speedbench"))
    ap.add_argument("--speedbench", default=sb, help="speedbench checkout (longdoc); env SPEEDBENCH")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--allow-no-cache-report", action="store_true",
                    help="run even if the API reports no cached_tokens (hits / misses then 'unknown')")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    a.run_id = time.strftime("%Y%m%dT%H%M%S") + "-" + secrets.token_hex(4)   # cache_salt prefix: cold turn 1 per run
    if not 2 <= a.turns <= len(FOLLOW_UPS) + 1:
        ap.error("--turns must be 2..%d" % (len(FOLLOW_UPS) + 1))
    phases = [p.strip() for p in a.phases.split(",") if p.strip()]
    if not phases or any(p not in ("short", "long") for p in phases):
        ap.error("--phases: short, long or short,long")
    sys.path.insert(0, a.speedbench)
    from speedbench import longdoc  # noqa: E402
    os.makedirs(a.out, exist_ok=True)

    pre = preflight(a, longdoc)
    print(json.dumps({"preflight_cached_tokens": pre}), flush=True)
    if pre == 0:
        print("warning: the API reports cached_tokens but the repeated preflight prompt did not hit", flush=True)
    if pre is None and not a.allow_no_cache_report:
        sys.exit("the API reports no cached_tokens: start the server with --enable-cache-report "
                 "(or pass --allow-no-cache-report)")

    mem, stop = [], threading.Event()

    def sampler():
        while not stop.is_set():
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemAvailable:"):
                            mem.append((time.time(), int(line.split()[1]) / 1048576))
            except OSError:
                return
            stop.wait(1.0)
    st = threading.Thread(target=sampler, daemon=True)
    st.start()

    def convs_for(n_tokens, base_seed):
        cs = []
        for i in range(a.agents):
            q = longdoc.QUESTIONS[i % len(longdoc.QUESTIONS)]
            doc = longdoc.document(max(200, int(n_tokens * a.cpt) - len(q) - 16), seed=base_seed + i,
                                   salt="ag64-%d-%d" % (n_tokens, i))
            cs.append(longdoc.messages(doc, q))
        return cs

    res = {"agents": a.agents, "turns": a.turns, "long_tokens": a.long_tokens, "short_tokens": a.short_tokens,
           "max_tokens": a.max_tokens, "temperature": a.temperature, "effort": "template default (high)",
           "url": a.url, "run_id": a.run_id, "preflight_cached_tokens": pre, "hit_threshold": HIT, "miss_threshold": MISS,
           "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "phases": [], "records": []}
    t_run = time.monotonic()
    for tag in phases:
        size = a.short_tokens if tag == "short" else a.long_tokens
        convs = convs_for(size, 1000 if tag == "short" else 5000)
        prev = None
        t_tag = time.monotonic()
        for turn in range(1, a.turns + 1):
            seed0 = 100 * turn   # turn 1: 100 + i, turn 2: 200 + i (the 1x AG64 seeds)
            name = "%s-%s" % (tag, "cold" if turn == 1 else "turn%d" % turn)
            s, recs = phase(a, convs, name, seed0, prev)
            res["phases"].append(s)
            convs = [c + [{"role": "assistant", "content": r["answer"] or "(no answer)"},
                          {"role": "user", "content": FOLLOW_UPS[(turn - 1) % len(FOLLOW_UPS)]}]
                     for c, r in zip(convs, recs)]
            prev = recs
            for r in recs:
                res["records"].append({k: v for k, v in r.items() if k != "answer"})
        res.setdefault("phase_group_wall_s", {})[tag] = round(time.monotonic() - t_tag, 1)
    res["wall_s"] = round(time.monotonic() - t_run, 1)
    stop.set()
    res["mem_available_min_gib_head"] = round(min(v for _, v in mem), 2) if mem else None
    with open(os.path.join(a.out, "ag64.json"), "w") as f:
        json.dump(res, f, indent=1)
    turns2 = [p for p in res["phases"] if "turn" in p["phase"]]
    print(json.dumps({"wall_s": res["wall_s"], "phase_group_wall_s": res["phase_group_wall_s"],
                      "misses": {p["phase"]: p["cache"].get("misses") for p in turns2},
                      "mem_available_min_gib_head": res["mem_available_min_gib_head"]}), flush=True)


if __name__ == "__main__":
    main()
