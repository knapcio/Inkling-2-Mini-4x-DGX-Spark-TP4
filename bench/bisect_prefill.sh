#!/usr/bin/env bash
# bisect_prefill.sh: B5h + ONE prefill lever per boot, teacher-forced prompt logprobs vs stock TP4 (results/tf/B3ref-a.json).
# Log: ~/inkling-tp4/bisect-prefill.log (one JSON line per lever). Detached-safe.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); LOG=$HERE/bisect-prefill.log
exec >> "$LOG" 2>&1
base=(INKLING_TP4_ROCE=1 SPEC=dflash2 OVERLAY=1 SCREEN=0 INKLING_RSFIX=1 INKLING_DFLASH_PREFIX_TAIL=2048
      INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT=4096 INKLING_DFLASH_TAIL_SNAPSHOTS=8 INKLING_PICO12=dense,lmhead,stack1
      INKLING_DFLASH_BYTES=mx8,mx8w,hot8
      INKLING_DRAFT_HOTVOCAB_FILE=/opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json
      INKLING_VATTN=1 INKLING_WIDE=dense,lmhead INKLING_WIDE_SCOPE=verify)
declare -A arms=(
  [bl]="INKLING_PREFILL=bias_layout"
  [qkvr]="INKLING_PREFILL=qkvr_split INKLING_PREFILL_QKVR_OUT=strided"
  [pfuse]="INKLING_PFUSE=all"
  [prologue]="INKLING_PROLOGUE=1"
  [attn]="INKLING_ATTN=prefill_swa,prefill_global INKLING_ATTN_GATE=1")
for a in ${ARMS:-bl qkvr pfuse prologue attn}; do
  echo "== $(date -u +%FT%TZ) lever $a: ${arms[$a]}"
  env "${base[@]}" ${arms[$a]} bash "$HERE/bench/arm.sh" "BX$a" --speculative-use-block-verification
  grep -E "PASS|HANG|UPFAIL|REPROFAIL" "$HERE/arm-BX$a.log" | head -2
  python3 "$HERE/tools/tf_logprobs.py" dump "$HERE/results/tf/BX$a-a.json" --n 30 2>/dev/null &&
    echo "RESULT $a $(python3 "$HERE/tools/tf_logprobs.py" compare "$HERE/results/tf/B3ref-a.json" "$HERE/results/tf/BX$a-a.json")"
done
echo "BISECT DONE $(date -u +%FT%TZ)"
