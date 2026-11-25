#!/usr/bin/env bash
# full.sh LABEL [URL]: every number in the README, from one boot, in this order (~75 min):
#   speedbench full (73 prompts x 256 tokens, c1, T=0, effort low)          -> LABEL-speedbench.json
#   decode table (prose/code/structured/json x c1..c16, effort none)         -> LABEL-decode.json
#   thinking load (effort high, T=0.8): c1 3x2048, c8 16x4096, c24 48x4096  -> LABEL-think.jsonl
#   speedbench longctx: cold/cached TTFT at 1K/8K/32K/128K, decode at 32K/128K -> LABEL-longctx.json
#   sanity: chat, effort ladder, tool call, JSON schema, needle at 120K and 240K -> LABEL-sanity*.json
#   speedbench quality guard (30 exact-answer items, T=0, c1)                 -> LABEL-guard.json
#   MemAvailable every 5 s during the whole run                              -> LABEL-mem.tsv
set -euo pipefail
L=$1; URL=${2:-http://127.0.0.1:8000}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SB=${SPEEDBENCH:-$HERE/speedbench}
OUT=${OUT:-$HERE/results/$L}; mkdir -p "$OUT"
META=(--hardware "${HARDWARE:-4x DGX Spark (GB10), TP4}" --clock-policy "cap 2200 MHz" --notes "${NOTES:-full}" --engine "${ENGINE_NAME:-unknown}")
( while true; do echo -e "$(date -Is)\t$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)"; sleep 5; done ) > "$OUT/$L-mem.tsv" &
MEMPID=$!; trap 'kill $MEMPID 2>/dev/null' EXIT
step() { echo "== $(date -Is) $*" >&2; }
step speedbench full
(cd "$SB" && python3 -m speedbench run --url "$URL" --mode full --label "$L" --metrics --out "$OUT/$L-speedbench.json" "${META[@]}" --quiet) 2>&1 | tail -2
step decode table
python3 "$HERE/bench/decode_table.py" --url "$URL" --prompts "$SB/prompts/speedbench-v1.json" --conc ${DECODE_CONC:-1,2,4,8,16} \
  --label "$L" --out "$OUT/$L-decode.json"
step thinking load
: > "$OUT/$L-think.jsonl"
python3 "$HERE/bench/think_load.py" --url "$URL" --concurrency 1 --requests 3 --max-tokens 2048 --label "$L-c1" >> "$OUT/$L-think.jsonl"
python3 "$HERE/bench/think_load.py" --url "$URL" --concurrency 8 --requests 16 --max-tokens 4096 --label "$L-c8" >> "$OUT/$L-think.jsonl"
python3 "$HERE/bench/think_load.py" --url "$URL" --concurrency 24 --requests 48 --max-tokens 4096 --label "$L-c24" >> "$OUT/$L-think.jsonl"
cat "$OUT/$L-think.jsonl"
step longctx
(cd "$SB" && python3 -m speedbench longctx --url "$URL" --label "$L" --contexts 1k,8k,32k,128k --decode-contexts 32k,128k --reps 2 \
   --out "$OUT/$L-longctx.json" "${META[@]}" --quiet) 2>&1 | tail -2
step quality guard
(cd "$SB" && python3 -m speedbench run --url "$URL" --mode quick --label "$L-guard" --quality-guard --out "$OUT/$L-guard.json" "${META[@]}" --quiet) 2>&1 | tail -3
step sanity
python3 "$HERE/bench/sanity.py" all --url "$URL" --tokens 120000 --out "$OUT/$L-sanity.json" || true
python3 "$HERE/bench/sanity.py" needle --url "$URL" --tokens 240000 --out "$OUT/$L-sanity-needle240k.json" || true
step done
