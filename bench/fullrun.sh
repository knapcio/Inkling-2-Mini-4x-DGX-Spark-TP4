#!/usr/bin/env bash
# fullrun.sh LABEL: bench/full.sh on the running TP4 engine with all-rank memory sampling; detached-safe.
L=$1; HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); OUT=$HERE/results/$L; mkdir -p "$OUT"
exec > "$HERE/full-$L.log" 2>&1
bash "$HERE/bench/memall.sh" "$OUT/$L-memall.tsv" & M=$!
SPEEDBENCH=${SPEEDBENCH:-$HOME/speedbench-6733653} ENGINE_NAME="SGLang TP4 ${ENGINE_NOTE:-}" bash "$HERE/bench/full.sh" "$L"
kill $M
awk -F'\t' '{for (i=2;i<=5;i++) if ($i!="NA" && (min[i]=="" || $i+0<min[i])) min[i]=$i+0}
  END {printf "worst-case MemAvailable GiB r0..r3: %s %s %s %s\n", min[2], min[3], min[4], min[5]}' "$OUT/$L-memall.tsv"
echo "FULL $L DONE $(date -u +%FT%TZ)"
