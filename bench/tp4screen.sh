#!/usr/bin/env bash
# tp4screen.sh LABEL [URL]: the TP4 boot screen, run on the head (rank 0).
#   0. warm-up: 4 short requests (the first run after a boot is discarded)
#   1. speedbench quick (30 prompts x 160 tokens, c1, T=0, effort low)
#   2. decode table: prose + code at c1 / c4 / c16 (T=0, 256 tokens, effort none)
#   3. thinking c8: 16 x 2048 tokens, effort high, T=0.8 (THINK=0 skips)
#   4. speedbench quality guard, 30 exact-answer items (GUARD=0 skips)
#   MemAvailable of all four ranks every 5 s during the run -> LABEL-mem.tsv (min per rank printed at the end)
set -euo pipefail
L=$1; URL=${2:-http://127.0.0.1:8000}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SB=${SPEEDBENCH:-$HOME/speedbench-6733653}
OUT=${OUT:-$HERE/results/$L}; mkdir -p "$OUT"
IPS=(${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4}); KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_nvsync_cluster_assistant}
META=(--hardware "${HARDWARE:-4x DGX Spark (GB10), TP4}" --clock-policy "cap 2200 MHz" --notes "${NOTES:-tp4screen}")
memloop() {
  while true; do
    line="$(date -Is)"
    for r in 0 1 2 3; do
      if [[ $r == 0 ]]; then m=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
      else m=$(ssh -o BatchMode=yes -i "$KEY" "${IPS[$r]}" "awk '/MemAvailable/ {printf \"%.2f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA); fi
      line+=$'\t'$m
    done
    echo "$line"; sleep 5
  done
}
memloop > "$OUT/$L-mem.tsv" & MEMPID=$!
trap 'kill $MEMPID 2>/dev/null' EXIT
step() { echo "== $(date -Is) $*" >&2; }
step warm-up
python3 "$HERE/bench/decode_table.py" --url "$URL" --prompts "$SB/prompts/speedbench-v1.json" --types prose --conc 1 --rounds '*:1' \
  --warmup 2 --label "$L-warm" --out "$OUT/$L-warm.json" >/dev/null
step speedbench quick
(cd "$SB" && python3 -m speedbench run --url "$URL" --mode quick --label "$L" --metrics --out "$OUT/$L-sbquick.json" \
   "${META[@]}" --quiet) 2>&1 | tail -3
step decode table
python3 "$HERE/bench/decode_table.py" --url "$URL" --prompts "$SB/prompts/speedbench-v1.json" --types "${TYPES:-prose,code}" \
  --conc "${CONC:-1,4,16}" --label "$L" --out "$OUT/$L-decode.json" | tail -8
if [[ ${THINK:-1} == 1 ]]; then
  step thinking c8
  python3 "$HERE/bench/think_load.py" --url "$URL" --concurrency 8 --requests 16 --max-tokens 2048 --label "$L-c8" \
    > "$OUT/$L-think-c8.json"; cat "$OUT/$L-think-c8.json"
fi
if [[ ${GUARD:-1} == 1 ]]; then
  step quality guard
  (cd "$SB" && python3 -m speedbench run --url "$URL" --mode quick --label "$L-guard" --quality-guard --out "$OUT/$L-guard.json" \
     "${META[@]}" --quiet) 2>&1 | grep -E "quality guard|failed" | tail -2
fi
if [[ ${PREFILL:-0} == 1 ]]; then
  step longctx 8K/32K
  (cd "$SB" && python3 -m speedbench longctx --url "$URL" --label "$L" --contexts 8k,32k --decode-contexts 32k --reps 2 \
     --out "$OUT/$L-longctx.json" "${META[@]}" --quiet) 2>&1 | tail -3
fi
kill $MEMPID 2>/dev/null || true
awk -F'\t' 'NR>0 {for (i=2;i<=5;i++) if ($i!="NA" && (min[i]=="" || $i+0<min[i])) min[i]=$i+0}
  END {printf "min MemAvailable GiB r0..r3: %s %s %s %s\n", min[2], min[3], min[4], min[5]}' "$OUT/$L-mem.tsv"
step done
