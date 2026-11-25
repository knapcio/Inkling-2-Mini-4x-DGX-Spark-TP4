#!/usr/bin/env bash
# chain_cert.sh: GPU check of the sharded certified head (check mode: certified result compared with the full gather).
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd); cd "$HERE"; exec >> chain-cert.log 2>&1
die() { echo "CHAIN FAIL: $*"; echo "== $(date -u +%FT%TZ) K2: cert production + cap (candidate final)"
env $CERT INKLING_CERT_HEAD=1 EXTRA_ENV="$XE" bash bench/arm.sh K2cert || die K2
echo "RESULT K2 active: $(ranklogs | grep -c 'active (mode 1, TP4') errors: $(ranklogs | grep -ciE 'Traceback|CertFatal|exit 78')"
echo "CHAIN DONE $(date -u +%FT%TZ)"; exit 1; }
ranklogs() { for ip in 10.100.96.1 10.100.96.3 10.100.96.4; do
    ssh -i ~/.ssh/id_ed25519_nvsync_cluster_assistant $ip "docker logs \$(docker ps --filter name=inkling-tp4 --format {{.Names}}) 2>&1" ; done
  docker logs inkling-tp4-r0 2>&1; }
CERT="ALLOW_TP4_UNVERIFIED=1 INKLING_CERT_TP=1 INKLING_VERIFY_CAP=dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8,rowscale=0.5 INKLING_VERIFY_CAP_REMAP=1 INKLING_VERIFY_CAP_BV=mask INKLING_VERIFY_CAP_LOG_EVERY=0"
XE="-e SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1"
echo "== $(date -u +%FT%TZ) K1: cert check mode"
env $CERT INKLING_CERT_HEAD=check EXTRA_ENV="$XE" bash bench/arm.sh K1certchk || die K1
echo "RESULT K1 active: $(ranklogs | grep -c 'active (mode check, TP4') errors: $(ranklogs | grep -ciE 'Traceback|CertFatal|exit 78')"
ranklogs | grep -iE "inkling_cert.*(mismatch|certified|disarm|active)" | sort | uniq -c | sort -rn | head -12
python3 tools/gen_logprobs.py dump results/tf/gen-K1-c1.json --n 30 --tokens 256 --conc 1 || die dump
echo "RESULT K1 gen-vs-ref $(python3 tools/gen_logprobs.py compare results/tf/gen-ref-c1.json results/tf/gen-K1-c1.json)"
ranklogs | grep -iE "inkling_cert.*(mismatch|certified)" | tail -8
echo "== $(date -u +%FT%TZ) K2: cert production + cap (candidate final)"
env $CERT INKLING_CERT_HEAD=1 EXTRA_ENV="$XE" bash bench/arm.sh K2cert || die K2
echo "RESULT K2 active: $(ranklogs | grep -c 'active (mode 1, TP4') errors: $(ranklogs | grep -ciE 'Traceback|CertFatal|exit 78')"
echo "CHAIN DONE $(date -u +%FT%TZ)"
