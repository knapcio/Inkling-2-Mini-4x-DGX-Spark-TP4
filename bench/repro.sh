#!/usr/bin/env bash
# repro.sh LABEL: wait for /health, then speedbench quick capped at 300 s (healthy boots finish in ~60 s).
# Prints PASS <summary> or HANG; exit 0 / 1.
set -u
L=$1; URL=${URL:-http://127.0.0.1:8000}; SB=${SPEEDBENCH:-$HOME/speedbench-6733653}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); OUT=$HERE/results/$L; mkdir -p "$OUT"
for _ in $(seq 120); do curl -fsS --max-time 3 "$URL/health" >/dev/null 2>&1 && break; sleep 10; done
curl -fsS --max-time 3 "$URL/health" >/dev/null 2>&1 || { echo "NOBOOT $L"; exit 2; }
cd "$SB" && timeout 300 python3 -m speedbench run --url "$URL" --mode quick --label "$L" --metrics --out "$OUT/$L-sbquick.json" \
  --hardware "4x DGX Spark (GB10), TP4" --clock-policy "cap 2200 MHz" --notes repro --quiet > "$OUT/$L-sbquick.log" 2>&1
rc=$?
if (( rc == 0 )); then echo "PASS $L $(grep -o '"geomean_tok_s": [0-9.]*' "$OUT/$L-sbquick.log" | head -1)"; exit 0; fi
echo "HANG/FAIL $L rc=$rc"; exit 1
