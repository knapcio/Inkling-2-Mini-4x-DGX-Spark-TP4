#!/usr/bin/env bash
# screen.sh LABEL [URL]: the ~10 min screening set used for every arm in docs/experiments.md.
#   1. speedbench quick (30 prompts x 160 tokens, c1, T=0, effort low)
#   2. thinking load c8: 16 puzzles x 2048 tokens, effort high, T=0.8 (bench/think_load.py)
#   3. speedbench longctx at 32K: cold/cached TTFT and decode at 32K context (2 documents)
# Results land in results/screen/LABEL-*.json. Needs a speedbench checkout in SPEEDBENCH (default ./speedbench).
set -euo pipefail
L=$1; URL=${2:-http://127.0.0.1:8000}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SB=${SPEEDBENCH:-$HERE/speedbench}
OUT=${OUT:-$HERE/results/screen}; mkdir -p "$OUT"
META=(--hardware "${HARDWARE:-4x DGX Spark (GB10), TP4}" --clock-policy "cap 2200 MHz" --notes "${NOTES:-screen}")
(cd "$SB" && python3 -m speedbench run --url "$URL" --mode quick --label "$L" --metrics --out "$OUT/$L-sbquick.json" \
   "${META[@]}" --quiet) 2>&1 | tail -3
python3 "$HERE/bench/think_load.py" --url "$URL" --concurrency 8 --requests 16 --max-tokens 2048 --label "$L-think-c8" \
  | tee "$OUT/$L-think-c8.json"
(cd "$SB" && python3 -m speedbench longctx --url "$URL" --label "$L" --contexts 32k --decode-contexts 32k --reps 2 \
   --out "$OUT/$L-longctx32k.json" "${META[@]}" --quiet) 2>&1 | tail -3
python3 - "$OUT" "$L" <<'PY'
import json, sys, os
o, l = sys.argv[1:3]
q = json.load(open(os.path.join(o, l + "-sbquick.json")))
s = q.get("summary") or {}
t = json.load(open(os.path.join(o, l + "-think-c8.json")))
c = json.load(open(os.path.join(o, l + "-longctx32k.json")))["table"][0]
A = s.get("ALL", {})
print(json.dumps({"label": l, "sb_quick_all": A.get("geomean_tok_s"), "sb_quick_prose": s.get("prose", {}).get("geomean_tok_s"),
                  "failed": s.get("failed"), "tok_per_step": A.get("median_tokens_per_step"), "ttft_ms": A.get("median_ttft_ms"),
                  "think_c8": t["aggregate_tok_s"],
                  "prefill32k": c.get("prefill_tok_s"), "ttft32k_cold": c.get("ttft_cold_s"),
                  "decode32k": c.get("decode_tok_s")}, default=str))
PY
