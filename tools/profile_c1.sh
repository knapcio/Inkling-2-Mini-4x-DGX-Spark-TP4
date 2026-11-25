#!/usr/bin/env bash
# profile_c1.sh LABEL [STEPS]: one live GPU trace of c1 decode on the running TP4 engine (rank 0's trace lands in
# ~/.cache/inkling-tp4/prof/LABEL on the head), then bucket it with tools/trace_buckets.py. Warm engine only.
set -euo pipefail
L=$1; N=${2:-60}; URL=${URL:-http://127.0.0.1:8000}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
req() { curl -fsS -m 120 "$URL/v1/chat/completions" -H 'content-type: application/json' -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"Write a long, detailed essay about the history of the printing press in Europe."}],"max_tokens":600,"temperature":0,"chat_template_kwargs":{"reasoning_effort":"none"}}' >/dev/null; }
req   # warm
KEY=$(cat "$HOME/.cache/inkling-tp4/admin-key" 2>/dev/null) || { echo "no admin key: boot with PROFILE=1" >&2; exit 1; }
curl -fsS -m 30 "$URL/start_profile" -H 'content-type: application/json' -H "Authorization: Bearer $KEY" \
  -d "{\"output_dir\":\"/root/.cache/prof/$L\",\"num_steps\":$N,\"activities\":[\"GPU\"],\"profile_by_stage\":false}" ; echo
req
sleep 5
f=$(ls -t "$HOME/.cache/inkling-tp4/prof/$L"/*TP-0*.trace.json.gz 2>/dev/null | head -1)
[[ -n $f ]] || { echo "no trace in ~/.cache/inkling-tp4/prof/$L" >&2; exit 1; }
python3 "$HERE/tools/trace_buckets.py" "$f" "$N"
