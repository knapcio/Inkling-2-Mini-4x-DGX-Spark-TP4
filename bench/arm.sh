#!/usr/bin/env bash
# arm.sh LABEL [engine flags...]: one A/B arm on the head, detached-safe. Settings come from the environment
# (SPEC, OVERLAY, INKLING_*, ...). down -> up -> watch -> repro (5-min hang cap) -> tp4screen. Log: ~/inkling-tp4/arm-LABEL.log
# Run as: setsid nohup bash bench/arm.sh LABEL [flags] > /dev/null 2>&1 < /dev/null &
set -u
L=$1; shift
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); LOG=$HERE/arm-$L.log
exec > "$LOG" 2>&1
echo "ARM $L start $(date -u +%FT%TZ) env: $(env | grep -E '^(SPEC|OVERLAY|MEM_FRACTION|INKLING_[A-Z0-9_]*)=' | tr '\n' ' ') flags: $*"
bash "$HERE/scripts/launch-tp4.sh" down
timeout 900 bash "$HERE/scripts/launch-tp4.sh" up "$@" 2>&1 | grep -vE '^\+ docker|^ {10,}|sha256|config.json|model.safetensors'
[[ ${PIPESTATUS[0]} == 0 ]] || { echo "ARM $L UPFAIL"; exit 1; }
setsid nohup bash "$HERE/scripts/launch-tp4.sh" watch > "$HERE/watch.log" 2>&1 < /dev/null &
bash "$HERE/bench/repro.sh" "$L-repro" || { echo "ARM $L REPROFAIL"; bash "$HERE/scripts/launch-tp4.sh" down; exit 1; }
[[ ${SCREEN:-1} == 1 ]] && bash "$HERE/bench/tp4screen.sh" "$L" 2>&1 | grep -v '^wrote'
echo "ARM $L DONE $(date -u +%FT%TZ)"
