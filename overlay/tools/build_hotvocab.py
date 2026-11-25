#!/usr/bin/env python3
"""Build the hot-vocabulary file for INKLING_DRAFT_BYTES=hotvocab:N from
generated-output traces, ranked by token frequency (Inkling token ids are NOT
frequency-ordered, so "ids < N" is not a shortlist).

  python3 build_hotvocab.py --tokenizer /model --out /opt/hot/hot.json \
      traces/*.jsonl [--top 65536] [--extra-ids 1,2]

Each JSONL line may be any of:
  * a JSON list of token ids
  * SGLang /generate output: {"text": ..., "output_ids": [...], ...}
    (token ids are preferred over text when both are present)
  * OpenAI chat/completions responses: {"choices": [{"message": {"content",
    "reasoning_content"}}]} or {"choices": [{"text": ...}]}; streaming chunks
    with "delta" also work (concatenated per line)
  * {"text": ...} / {"content": ...} / {"output": ...} / {"response": ...}
Text is tokenized with the model tokenizer (no BOS/EOS added; special-token
strings in the text are parsed as special tokens). Thinking text
("reasoning_content") counts: the draft proposes thinking tokens too.

Output: a JSON list of ids, most frequent first (the overlay takes the first N).
EOS ids from <tokenizer>/generation_config.json and config.json and any
--extra-ids are put at the front. A <out>.stats.json is written next to it.

Coverage is printed for N = 8K/16K/32K/64K: in-sample (fit and score on all
records) and held-out (fit on even records, score on odd ones). Held-out is the
honest estimate of the share of target tokens the hot set can propose; every
miss is a guaranteed rejection at that position (about 1 % of accepted drafts
per 1 % of misses).
"""

import argparse
import collections
import glob
import json
import os
import sys

SIZES = (8192, 16384, 32768, 65536)
ID_KEYS = ("output_ids", "token_ids", "output_token_ids", "completion_token_ids", "ids")
TEXT_KEYS = ("reasoning_content", "reasoning", "content", "text")


def load_tokenizer(path):
    tj = os.path.join(path, "tokenizer.json")
    if os.path.isfile(tj):
        try:
            from tokenizers import Tokenizer

            tok = Tokenizer.from_file(tj)
            return lambda s: tok.encode(s, add_special_tokens=False).ids
        except ImportError:
            pass
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    return lambda s: tok.encode(s, add_special_tokens=False)


def eos_ids(path):
    out = []
    for name in ("generation_config.json", "config.json"):
        try:
            with open(os.path.join(path, name)) as f:
                cfg = json.load(f)
        except (OSError, ValueError):
            continue
        for c in (cfg, cfg.get("text_config", {})):
            v = c.get("eos_token_id")
            if isinstance(v, int):
                out.append(v)
            elif isinstance(v, list):
                out.extend(int(x) for x in v)
    return list(dict.fromkeys(out))


def extract(obj, texts, ids):
    """Collect text pieces and id lists from one decoded JSON value."""
    if isinstance(obj, list):
        if obj and all(isinstance(x, int) for x in obj):
            ids.append(obj)
        else:
            for x in obj:
                extract(x, texts, ids)
        return
    if not isinstance(obj, dict):
        return
    for key in ID_KEYS:
        v = obj.get(key)
        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
            ids.append(v)
            return
    if "choices" in obj and isinstance(obj["choices"], list):
        for ch in obj["choices"]:
            if not isinstance(ch, dict):
                continue
            for part in (ch.get("message"), ch.get("delta"), ch):
                if isinstance(part, dict):
                    got = False
                    for key in ID_KEYS:
                        v = part.get(key)
                        if isinstance(v, list) and v and all(isinstance(x, int) for x in v):
                            ids.append(v)
                            got = True
                    if got:
                        break
                    for key in TEXT_KEYS:
                        if isinstance(part.get(key), str) and part[key]:
                            texts.append(part[key])
                            got = True
                    if got:
                        break
        return
    found = False
    for key in TEXT_KEYS:
        if isinstance(obj.get(key), str) and obj[key]:
            texts.append(obj[key])
            found = True
    if found:
        return
    for key in ("response", "output", "outputs", "result", "data"):
        if key in obj:
            extract(obj[key], texts, ids)


def read_records(paths, encode):
    """Yields one token-id list per JSONL line."""
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    texts, id_lists = [], []
                    extract(obj, texts, id_lists)
                    if id_lists:
                        yield [t for lst in id_lists for t in lst]
                    elif texts:
                        yield encode("".join(texts))


def coverage(counter_fit, counter_eval, sizes, front=()):
    front = list(front)
    ranked = front + [t for t, _ in counter_fit.most_common() if t not in set(front)]
    total = sum(counter_eval.values()) or 1
    out = {}
    for n in sizes:
        top = set(ranked[:n])
        out[n] = sum(c for t, c in counter_eval.items() if t in top) / total
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+", help="JSONL files or globs")
    ap.add_argument("--tokenizer", default="/model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=65536, help="ids to write (the overlay uses the first N)")
    ap.add_argument("--extra-ids", default="", help="comma list of ids always included first")
    ap.add_argument("--vocab-limit", type=int, default=0, help="drop ids >= this (0 = keep all)")
    args = ap.parse_args()

    encode = None
    needs_tok = True
    try:
        encode = load_tokenizer(args.tokenizer)
    except Exception as e:  # noqa: BLE001
        print(f"tokenizer unavailable ({e!r}); only token-id records will be used", file=sys.stderr)
        needs_tok = False

    def enc(text):
        return encode(text) if needs_tok else []

    counts_all, counts_even, counts_odd = (collections.Counter() for _ in range(3))
    n_records = 0
    for i, ids in enumerate(read_records(args.traces, enc)):
        if args.vocab_limit:
            ids = [t for t in ids if 0 <= t < args.vocab_limit]
        counts_all.update(ids)
        (counts_even if i % 2 == 0 else counts_odd).update(ids)
        n_records += 1
    if not counts_all:
        print("no tokens found in the traces", file=sys.stderr)
        return 1

    front = [int(x) for x in args.extra_ids.split(",") if x.strip()]
    front += [t for t in eos_ids(args.tokenizer) if t not in front]
    ranked = front + [t for t, _ in counts_all.most_common() if t not in set(front)]
    ranked = ranked[: args.top]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(ranked, f)

    total = sum(counts_all.values())
    sizes = [n for n in SIZES] + ([args.top] if args.top not in SIZES else [])
    insample = coverage(counts_all, counts_all, sizes, front)
    heldout = coverage(counts_even, counts_odd, sizes, front) if counts_odd else {}
    stats = {
        "records": n_records,
        "tokens": total,
        "distinct_ids": len(counts_all),
        "written_ids": len(ranked),
        "front_ids": front,
        "coverage_in_sample": {str(k): v for k, v in insample.items()},
        "coverage_held_out": {str(k): v for k, v in heldout.items()},
    }
    with open(args.out + ".stats.json", "w") as f:
        json.dump(stats, f, indent=1)
    print(f"records {n_records}, tokens {total}, distinct ids {len(counts_all)}, wrote {len(ranked)} ids -> {args.out}")
    print(f"{'N':>7} {'in-sample':>10} {'held-out':>10}")
    for n in sizes:
        ho = f"{100 * heldout[n]:9.2f}%" if heldout else "       n/a"
        print(f"{n:>7} {100 * insample[n]:9.2f}% {ho}")
    if len(counts_all) < max(SIZES):
        print(f"note: only {len(counts_all)} distinct ids seen; hotvocab:N above that uses "
              f"{len(ranked)} ids. More traces give a better tail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
