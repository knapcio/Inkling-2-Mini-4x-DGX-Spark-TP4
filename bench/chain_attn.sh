#!/usr/bin/env bash
# chain_attn.sh: (P) c1 profile of the release profile; then attn dissection arms (repro tok/step + teacher-forced NLL).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$HERE"; exec >> chain-attn.log 2>&1
echo "== $(date -u +%FT%TZ) P: release + PROFILE=1"
PROFILE=1 SCREEN=0 bash bench/arm.sh Prof
bash tools/profile_c1.sh relc1 60 2>&1 | tail -45
tps() { python3 - "$1" <<'PY'
import json, statistics as st, sys
d = json.load(open(sys.argv[1])); rq = [r for r in d["requests"] if r.get("valid_decode") and r.get("kind") != "sentinel"]
s = [r["tokens_per_step"] for r in rq if r["prompt_tokens"] < 512]
print("tok/s %.2f tps %.3f short-tps %.3f step %.2f" % (d["summary"]["ALL"]["geomean_tok_s"], d["summary"]["ALL"]["median_tokens_per_step"], st.median(s), d["summary"]["ALL"]["median_step_ms"]))
PY
}
arm() { local L=$1; shift
  echo "== $(date -u +%FT%TZ) $L: $*"
  env SCREEN=0 ALLOW_TP4_UNVERIFIED=1 "$@" bash bench/arm.sh "$L"
  echo "RESULT $L $(tps results/$L-repro/$L-repro-sbquick.json)"
  python3 tools/tf_logprobs.py dump results/tf/$L-a.json --n 30 2>/dev/null &&
    echo "RESULT $L tf $(python3 tools/tf_logprobs.py compare results/tf/B3ref-a.json results/tf/$L-a.json)"
}
arm AnoGate INKLING_ATTN=prefill_swa,prefill_global INKLING_ATTN_GATE=0
arm AswaNoGate INKLING_ATTN=prefill_swa INKLING_ATTN_GATE=0
arm AgateAll INKLING_ATTN=prefill_swa,prefill_global INKLING_ATTN_GATE=all
echo "CHAIN DONE $(date -u +%FT%TZ)"
