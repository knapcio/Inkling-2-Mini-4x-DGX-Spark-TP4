#!/usr/bin/env bash
# chain_dfcap.sh: GPU checks of the verify-width cap at TP4 on top of the release profile.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$HERE"; exec >> chain-dfcap.log 2>&1
set -o pipefail
die() { echo "CHAIN FAIL: $*"; echo "CHAIN DONE $(date -u +%FT%TZ)"; exit 1; }
POL=/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json
cap() { echo "ALLOW_TP4_UNVERIFIED=1 INKLING_VERIFY_CAP_REMAP=${2:-1} INKLING_VERIFY_CAP_BV=mask INKLING_VERIFY_CAP_LOG_EVERY=0 INKLING_VERIFY_CAP=$1"; }
ranklogs() { for ip in 10.100.96.1 10.100.96.3 10.100.96.4; do
    ssh -i ~/.ssh/id_ed25519_nvsync_cluster_assistant $ip "docker logs \$(docker ps --filter name=inkling-tp4 --format {{.Names}}) 2>&1" ; done
  docker logs inkling-tp4-r0 2>&1; }
echo "== $(date -u +%FT%TZ) C1: cap dfpolicy rowscale 1.0, TP_SYNC=check"
env $(cap "dfpolicy:$POL,bsmax=8") INKLING_VERIFY_CAP_TP_SYNC=check bash bench/arm.sh C1cap || die C1
echo "RESULT C1 attached: $(ranklogs | grep -c 'DFLASH attached: tp=4') differ: $(ranklogs | grep -ci 'live lengths differ') errors: $(ranklogs | grep -ciE 'Traceback|refus')"
ranklogs | grep -E "DFLASH attached|capping" | sort -u | head -4
echo "== $(date -u +%FT%TZ) C2: cap rowscale 0.5"
env $(cap "dfpolicy:$POL,bsmax=8,rowscale=0.5") bash bench/arm.sh C2cap05 || die C2
echo "== $(date -u +%FT%TZ) X1: fixed:4 REMAP=1 (exactness)"
env $(cap "fixed:4" 1) SCREEN=0 bash bench/arm.sh X1fix4r1 || die X1
rm -f results/tf/gen-X1-*.json results/tf/gen-X0-*.json
python3 tools/gen_logprobs.py dump results/tf/gen-X1-c1.json --n 30 --tokens 256 --conc 1 || die dump
python3 tools/gen_logprobs.py dump results/tf/gen-X1-c4.json --n 30 --tokens 256 --conc 4 || die dump
echo "== $(date -u +%FT%TZ) X0: fixed:4 REMAP=0"
env $(cap "fixed:4" 0) SCREEN=0 bash bench/arm.sh X0fix4r0 || die X0
python3 tools/gen_logprobs.py dump results/tf/gen-X0-c1.json --n 30 --tokens 256 --conc 1 || die dump
python3 tools/gen_logprobs.py dump results/tf/gen-X0-c4.json --n 30 --tokens 256 --conc 4 || die dump
echo "RESULT remap c1 $(python3 tools/gen_logprobs.py compare results/tf/gen-X0-c1.json results/tf/gen-X1-c1.json)"
echo "RESULT remap c4 $(python3 tools/gen_logprobs.py compare results/tf/gen-X0-c4.json results/tf/gen-X1-c4.json)"
echo "CHAIN DONE $(date -u +%FT%TZ)"
