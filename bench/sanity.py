#!/usr/bin/env python3
"""Quality and sanity checks for an Inkling-2-Mini endpoint (Python 3 standard library only).

  python3 bench/sanity.py chat      two short chats with known answers
  python3 bench/sanity.py effort    one puzzle at reasoning effort none/low/medium/high: reasoning length + answer
  python3 bench/sanity.py tools     a tool call, then the tool result back, then the final answer
  python3 bench/sanity.py json      a JSON-schema constrained answer (response_format json_schema)
  python3 bench/sanity.py needle --tokens 120000   passphrase at 50 % depth of varied filler text
  python3 bench/sanity.py all       everything above (needle at --tokens)

Every check prints PASS/FAIL with the evidence; --out FILE writes the results as JSON. Requests are sent one at a
time, temperature 0 unless stated.
"""
import argparse
import json
import random
import sys
import time
import urllib.request


def chat(a, messages, max_tokens=1024, effort="low", **extra):
    body = {"model": a.model, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0}
    if effort is not None:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    body.update(extra)
    req = urllib.request.Request(a.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=a.timeout) as r:
        d = json.load(r)
    dt = time.monotonic() - t0
    m = d["choices"][0]["message"]
    return {"content": m.get("content") or "", "reasoning": m.get("reasoning_content") or m.get("reasoning") or "",
            "tool_calls": m.get("tool_calls") or [], "finish": d["choices"][0].get("finish_reason"),
            "usage": d.get("usage") or {}, "seconds": round(dt, 2)}


def report(results, name, ok, **ev):
    results.append({"check": name, "pass": bool(ok), **ev})
    print("%-28s %s  %s" % (name, "PASS" if ok else "FAIL", json.dumps(ev, ensure_ascii=False)[:300]), flush=True)


def check_chat(a, res):
    r = chat(a, [{"role": "user", "content": "What is 17 * 23? Answer with just the number."}])
    report(res, "chat: 17*23", "391" in r["content"], answer=r["content"].strip()[:80],
           completion_tokens=r["usage"].get("completion_tokens"))
    r = chat(a, [{"role": "system", "content": "You are a concise assistant."},
                 {"role": "user", "content": "Name the capital of Poland and the river that flows through it. "
                                             "One sentence."}])
    c = r["content"].lower()
    report(res, "chat: capital + river", "warsaw" in c and "vistula" in c, answer=r["content"].strip()[:160])


PUZZLE = ("How many positive integers below 1000 are divisible by 7 or by 11, but not by both? "
          "End your answer with 'Answer: N'.")


def check_effort(a, res):
    lengths = {}
    for eff in ("none", "low", "medium", "high"):
        r = chat(a, [{"role": "user", "content": PUZZLE}], max_tokens=8192, effort=eff)
        ok = "answer: 208" in r["content"].lower().replace("**", "")
        lengths[eff] = r["usage"].get("completion_tokens")
        report(res, "effort %s" % eff, ok, completion_tokens=lengths[eff], reasoning_chars=len(r["reasoning"]),
               prompt_tokens=r["usage"].get("prompt_tokens"), answer=r["content"].strip()[-60:])
    # the template writes "Thinking effort level: X" into the prompt; prompt_tokens shows the field arrived
    report(res, "effort: none <= high", (lengths["none"] or 0) <= (lengths["high"] or 0), lengths=lengths)


TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                    "unit": {"type": "string", "enum": ["c", "f"]}},
                   "required": ["city"]}}}]


def check_tools(a, res):
    msgs = [{"role": "user", "content": "What is the weather in Krakow right now, in Celsius?"}]
    r = chat(a, msgs, tools=TOOLS, tool_choice="auto")
    tc = r["tool_calls"]
    args = {}
    if tc:
        try:
            args = json.loads(tc[0]["function"]["arguments"])
        except (ValueError, KeyError, TypeError):
            args = {}
    ok = bool(tc) and tc[0]["function"]["name"] == "get_weather" and "krak" in str(args.get("city", "")).lower()
    report(res, "tool call", ok, finish=r["finish"], calls=[t["function"] for t in tc][:2])
    if not ok:
        return
    msgs.append({"role": "assistant", "content": None, "tool_calls": [
        {"id": tc[0].get("id") or "call_0", "type": "function",
         "function": {"name": "get_weather", "arguments": tc[0]["function"]["arguments"]}}]})
    msgs.append({"role": "tool", "tool_call_id": tc[0].get("id") or "call_0",
                 "content": json.dumps({"city": "Krakow", "temp_c": 7, "conditions": "light rain"})})
    r = chat(a, msgs, tools=TOOLS)
    c = r["content"].lower()
    report(res, "tool result -> answer", "7" in c and "rain" in c, answer=r["content"].strip()[:160])


def check_json(a, res):
    schema = {"type": "object", "properties": {"city": {"type": "string"}, "population_millions": {"type": "number"},
                                               "landmarks": {"type": "array", "items": {"type": "string"},
                                                             "minItems": 2, "maxItems": 4}},
              "required": ["city", "population_millions", "landmarks"], "additionalProperties": False}
    r = chat(a, [{"role": "user", "content": "Give facts about Gdansk as JSON."}], max_tokens=2048,
             response_format={"type": "json_schema", "json_schema": {"name": "city", "schema": schema}})
    try:
        d = json.loads(r["content"])
        ok = set(d) == {"city", "population_millions", "landmarks"} and 2 <= len(d["landmarks"]) <= 4
    except ValueError:
        d, ok = r["content"][:120], False
    report(res, "json_schema output", ok, parsed=d)


WORDS = ("harbor lantern meadow copper violet signal orchard granite ember willow canyon falcon thistle quarry "
         "lattice saffron tundra beacon marble juniper cobalt prairie delta summit glacier fern atlas kestrel "
         "mosaic pylon").split()


def filler(rng, n_words):
    out, i = [], 0
    while i < n_words:
        k = rng.randint(8, 18)
        s = " ".join(rng.choice(WORDS) for _ in range(k))
        out.append(s.capitalize() + ". Record %d notes that the %s team logged %d items." % (
            rng.randint(1, 99999), rng.choice(WORDS), rng.randint(1, 999)))
        i += k + 9
    return " ".join(out)


def check_needle(a, res, tokens):
    rng = random.Random(tokens)
    code = "%s-%s-%04d" % (rng.choice(WORDS).upper(), rng.choice(WORDS).upper(), rng.randint(0, 9999))
    words = int(tokens * 0.553)         # ~1.81 tokens per filler word with this tokenizer; usage reports the real size
    half = filler(rng, words // 2)
    doc = half + "\n\nIMPORTANT: the secret passphrase is %s. Remember it.\n\n" % code + filler(rng, words // 2)
    msgs = [{"role": "user", "content": doc + "\n\nWhat is the secret passphrase mentioned in the text above? "
                                              "Reply with the passphrase only."}]
    t0 = time.monotonic()
    r = chat(a, msgs, max_tokens=512, effort="none")
    dt = time.monotonic() - t0
    pt = r["usage"].get("prompt_tokens")
    report(res, "needle %dK" % (tokens // 1000), code in r["content"], prompt_tokens=pt, seconds=round(dt, 1),
           expected=code, answer=r["content"].strip()[:80])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["chat", "effort", "tools", "json", "needle", "all"])
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--tokens", type=int, default=120000, help="needle prompt size (approximate)")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = []
    steps = {"chat": [check_chat], "effort": [check_effort], "tools": [check_tools], "json": [check_json],
             "needle": [lambda a, r: check_needle(a, r, a.tokens)]}
    todo = sum(steps.values(), []) if a.what == "all" else steps[a.what]
    for f in todo:
        try:
            f(a, res)
        except Exception as e:  # noqa: BLE001
            report(res, getattr(f, "__name__", "check"), False, error=repr(e))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"url": a.url, "results": res}, fh, indent=1, ensure_ascii=False)
    n_ok = sum(r["pass"] for r in res)
    print("%d / %d passed" % (n_ok, len(res)))
    sys.exit(0 if n_ok == len(res) else 1)


if __name__ == "__main__":
    main()
