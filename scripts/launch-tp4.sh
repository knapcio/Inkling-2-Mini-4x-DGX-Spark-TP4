#!/usr/bin/env bash
# launch-tp4.sh up|down|status|watch: Inkling-2-Mini on four DGX Sparks (GB10), SGLang TP4 / EP4 over the
# ConnectX-7 RoCE fabric, one container per node. Run it on the head node (rank 0); it reaches the three workers
# over SSH on their fabric addresses.
#
#   scripts/launch-tp4.sh up                 # default profile (see README)
#   SPEC=dspark OVERLAY=0 INKLING_TP4_ROCE=0 scripts/launch-tp4.sh up   # the stock engine (cookbook cell) at TP4, NCCL
#   scripts/launch-tp4.sh up --some-flag x   # extra engine flags are appended to every rank's command line
#   scripts/launch-tp4.sh status | down
#   scripts/launch-tp4.sh watch              # stays in the foreground: takes all ranks down if one of them dies
#
# The API binds to 127.0.0.1 on the head by default; reach it from another machine with
# `ssh -L 8000:127.0.0.1:8000 <head>`. Never expose it through the router.
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
NAME=${NAME:-inkling-tp4}
IPS=(${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4})   # rank 0 (head) first: fabric addresses
SSH_KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_nvsync_cluster_assistant}
SSH_USER=${SSH_USER:-$USER}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-8000}
DIST_PORT=${DIST_PORT:-5757}
MODEL_DIR=${MODEL_DIR:-$HOME/models/Inkling-2-Mini}
DFLASH2_DIR=${DFLASH2_DIR:-$HOME/models/Inkling-2-Mini-DFlash2}
CACHE_DIR=${CACHE_DIR:-$HOME/.cache/inkling-tp4}
NCCL_DIR=${NCCL_DIR:-$HOME/nccl-2.30.7}                     # NCCL 2.30.7 (as the DeepSeek / MiniMax TP4 recipes); NCCL_DIR= keeps the image's 2.28.3
IFACE=${FABRIC_IFACE:-enp1s0f0np0}                          # bootstrap interface (same name on every node)
HCA=${IB_HCA:-rocep1s0f0,roceP2p1s0f0}                      # both PCIe halves of the one 200G port
SERVED=${SERVED:-inkling-2-mini}
CTX=${CTX:-262144}
MEM_FRACTION=${MEM_FRACTION:-0.80}
SWA_RATIO=${SWA_RATIO:-0.05}
MAX_RUNNING=${MAX_RUNNING:-48}
MIN_AVAIL_GIB=${MIN_AVAIL_GIB:-3.5}                         # memory guard per node (stops that rank, `watch` stops the rest)
LOAD_NOMMAP=${LOAD_NOMMAP:-1}
SPEC=${SPEC:-dflash2}                                       # dflash2 (default: incoai DFlash2 drafter) | dspark | none
NUM_DRAFT=${NUM_DRAFT:-8}
OVERLAY=${OVERLAY:-1}                                       # 0: the image's own bytes (no INKLING_* lever)
OVERLAY_DIR=${OVERLAY_DIR:-$HOME/inkling-tp4/overlay}        # same path on every node (synced by `up`)
TP4_DIR=${TP4_DIR:-$HOME/inkling-tp4/tp4}                    # TP helpers (DFlash2 pre-patch, RoCE hooks), same path on every node
INKLING_TP4_ROCE=${INKLING_TP4_ROCE-1}; export INKLING_TP4_ROCE   # 0: NCCL for every collective
ROCE=$INKLING_TP4_ROCE                                       # 1: RoCEnante one-shot RDMA all-reduce / all-gather on the TP group (roce/, tp4/)
ROCE_DIR=${ROCE_DIR:-$HOME/inkling-tp4/roce}
SGLANG_IMAGE=${SGLANG_IMAGE:-lmsysorg/sglang@sha256:efac853b46811ccb760545b7b29e7119bdf907537ccf6e863fac3011fe8b98de}
MODEL_REVISION=ed40e7da625f594fc9c4990f5f66be24873c22d7
DFLASH2_REVISION=4f9088ea66c21f2b99ed657d1941c306b03e84d2
DFLASH2_SHA256="14223ae2f3aa1d661da67cf0143ec33070d952a6e28a119e6cec64fcc0bebe50 sglang_inkling_dflash2.py
d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa config.json
217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee model.safetensors"
SGLANG_ARGS=${SGLANG_ARGS:-}
# Release profile (validated 2026-11-25, README "Levers"): with SPEC=dflash2 and OVERLAY=1 each lever below defaults
# on; an explicitly set variable wins, and an empty value (e.g. INKLING_WIDE=) switches that lever off.
# Not in the profile (see README): INKLING_ATTN / INKLING_ATTN_GATE (decode acceptance -5..-7 % at TP4),
# INKLING_PREFILL=qkvr_split (not bit-exact at TP4 shapes: runs stock at TP > 1). Refused or inert at TP4: KV12S, DFCAP / verify cap, CERT_HEAD, DFLASH_FUSED_ACCEPT,
# NOEXP, PADX, MOE2, SCODES, KERNELS=moe_tiles.
if [[ $SPEC == dflash2 && $OVERLAY == 1 ]]; then
  _def() { [[ -n ${!1+set} ]] || export "$1=$2"; }
  _def INKLING_RSFIX 1
  _def INKLING_DFLASH_PREFIX_TAIL 2048
  _def INKLING_DFLASH_PREFIX_TAIL_MAX_INPUT 4096
  _def INKLING_DFLASH_TAIL_SNAPSHOTS 8
  _def INKLING_PICO12 dense,lmhead,stack1
  _def INKLING_DFLASH_BYTES mx8,mx8w,hot8
  _def INKLING_DRAFT_HOTVOCAB_FILE /opt/inkling-overlay/inkling_draftbytes/hotvocab/inkling2-mini-hot-r2.json
  _def INKLING_VATTN 1
  _def INKLING_WIDE dense,lmhead
  _def INKLING_WIDE_SCOPE verify
  _def INKLING_PREFILL bias_layout
  _def INKLING_PROLOGUE 1
  _def INKLING_PFUSE all                                     # prefill glue fusion: FP32-order class, prefill +11.6 %
  # verify-width cap (TP port: live lengths synced over the TP group, settings digest checked at attach); TP1 prices
  # halved (rowscale=0.5: per-row price at TP4 ~half of TP1's); block verification with the sparse-q mask
  _def INKLING_VERIFY_CAP dfpolicy:/opt/inkling-overlay/inkling_vcap/policies/df-SLDF2x.json,bsmax=8,rowscale=0.5
  _def INKLING_VERIFY_CAP_REMAP 1
  _def INKLING_VERIFY_CAP_BV mask
  _def INKLING_VERIFY_CAP_LOG_EVERY 0
  # sharded certified greedy LM head (global screen bounds by one small all-gather; deferred argmax replaces the
  # logits gather on greedy verify steps)
  _def INKLING_CERT_HEAD 1
  _def INKLING_CERT_TP 1
  # v2.9.2: prefill loop-bank hoist (FP32-order class; start-up self-test agreed over the TP group) and prompt-end
  # cache touch (LRU order only; arming agreed over the TP group, fail-stop at runtime)
  _def INKLING_PFUSE_HOIST 1
  _def INKLING_PROMPT_TOUCH 1
  BLOCK_VERIFY=${BLOCK_VERIFY:-1}                            # block verification on sampled steps (exact with RSFIX)
fi
case ",${INKLING_PREFILL:-}," in *,qkvr_split,*) [[ ${ALLOW_TP4_UNVERIFIED:-0} == 1 ]] || {
  echo "INKLING_PREFILL=qkvr_split is numerically wrong at TP4 (README); ALLOW_TP4_UNVERIFIED=1 to run it anyway" >&2; exit 2; };; esac
case ",${INKLING_PFUSE:-}," in *,all,*|*,loopmix,*) ;; *) export INKLING_PFUSE_HOIST=0 ;; esac   # the hoist wraps pfuse loopmix
case ${INKLING_PFUSE_HOIST:-} in exact) echo "INKLING_PFUSE_HOIST=exact is TP1 only" >&2; exit 2 ;; esac
case ${INKLING_CERT_HEAD:-} in ''|0|off|none) ;; *) [[ ${INKLING_CERT_TP:-} == 1 || ${ALLOW_TP4_UNVERIFIED:-0} == 1 ]] || {
  echo "INKLING_CERT_HEAD at TP4 needs INKLING_CERT_TP=1 (the sharded head)" >&2; exit 2; } ;; esac
[[ ${INKLING_CERT_TP:-} == 1 ]] && EXTRA_ENV="${EXTRA_ENV:-} -e SGLANG_OPT_DFLASH_DEFER_PLAIN_GATHER=1"
for v in INKLING_ATTN INKLING_KV12S; do
  case ${!v:-} in ''|0|off|none) ;; *) [[ ${ALLOW_TP4_UNVERIFIED:-0} == 1 ]] || {
    echo "$v is wrong, refused or untested at TP4 (README); ALLOW_TP4_UNVERIFIED=1 to run it anyway" >&2; exit 2; } ;; esac
done
EXTRA_ENV=${EXTRA_ENV:-}

on() { local r=$1; shift
  if [[ $r == 0 ]]; then bash -c "$*"
  else ssh -o BatchMode=yes -o ConnectTimeout=10 -i "$SSH_KEY" "$SSH_USER@${IPS[$r]}" "$*"; fi; }

cmd=${1:-}; shift || true
case $cmd in
  down)
    # a running `watch` would take the next boot down: stop it (and the head's bench clients) first
    pkill -f "launch-tp4[.]sh watch" 2>/dev/null || true
    pkill -f "bench/(repro|tp4screen)[.]sh" 2>/dev/null || true
    for r in 3 2 1 0; do
      on $r "docker stop -t 10 $NAME-r$r >/dev/null 2>&1; docker rm -f $NAME-r$r >/dev/null 2>&1; pkill -f 'memguard[.]sh $NAME-r$r ' 2>/dev/null; true" || true
    done
    echo "stopped $NAME on all ranks"; exit 0 ;;
  status)
    for r in 0 1 2 3; do
      on $r "printf 'rank $r %s  ' \$(hostname); docker ps -a --filter name=^$NAME-r$r\$ --format '{{.Status}}' | tr '\n' ' ';
             awk '/MemAvailable/ {printf \"MemAvailable %.1f GiB\n\", \$2/1048576}' /proc/meminfo" || echo "rank $r unreachable"
    done
    curl -fsS --max-time 5 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && echo "API: ready on :$PORT" || echo "API: not ready"
    exit 0 ;;
  watch)
    # Every 10 s: if any rank's container is not running, take every rank down (a lone rank holds its GPU memory
    # and the survivors wait in NCCL forever).
    while true; do
      for r in 0 1 2 3; do
        st=$(on $r "docker inspect -f '{{.State.Running}}' $NAME-r$r 2>/dev/null || echo gone" 2>/dev/null || echo unreachable)
        if [[ $st != true ]]; then
          echo "$(date -Is) rank $r: $st -> taking all ranks down"
          "$0" down; exit 1
        fi
      done
      sleep 10
    done ;;
  up) ;;
  *) echo "usage: $0 up|down|status|watch [extra engine flags]" >&2; exit 2 ;;
esac

# ---- preflight on every node ----------------------------------------------------------------------------------
for r in 0 1 2 3; do
  on $r "set -e
    [[ -f $MODEL_DIR/config.json && -f $MODEL_DIR/mtp.safetensors ]] || { echo \"rank $r: no model in $MODEL_DIR\"; exit 1; }
    [[ \$(cat $MODEL_DIR/.revision 2>/dev/null) == $MODEL_REVISION ]] || { echo \"rank $r: $MODEL_DIR is not revision $MODEL_REVISION\"; exit 1; }
    if docker ps --format '{{.Names}}' | grep -qx $NAME-r$r; then echo \"rank $r: $NAME-r$r already running\"; exit 1; fi
    others=\$(docker ps --format '{{.Names}} {{.Image}}' | grep -Ei 'sglang|vllm|tensorrt|llama|ollama' || true)
    [[ -z \$others ]] || { echo \"rank $r: another model server is running: \$others\"; exit 1; }
    avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo)
    (( avail > 100 )) || { echo \"rank $r: MemAvailable \$avail GiB (want > 100 before a boot)\"; exit 1; }
    [[ -z '$NCCL_DIR' || -f $NCCL_DIR/libnccl.so.2.30.7 ]] || { echo \"rank $r: no $NCCL_DIR/libnccl.so.2.30.7\"; exit 1; }
    for d in ${HCA//,/ }; do [[ \$(cat /sys/class/infiniband/\$d/ports/1/state) == *ACTIVE* ]] || { echo \"rank $r: \$d not ACTIVE\"; exit 1; }; done
    mkdir -p $CACHE_DIR" || exit 1
done

if [[ $SPEC == dflash2 ]]; then
  for r in 0 1 2 3; do
    on $r "cd $DFLASH2_DIR 2>/dev/null && [[ \$(cat .revision) == $DFLASH2_REVISION ]] &&
      [[ -z \$(ls -A | grep -vxE 'config\.json|model\.safetensors|sglang_inkling_dflash2\.py|README\.md|\.revision') ]] &&
      printf '%s\n' '$DFLASH2_SHA256' | sed 's/ /  /' | sha256sum -c --quiet -" ||
      { echo "rank $r: DFlash2 drafter in $DFLASH2_DIR missing, not revision $DFLASH2_REVISION or not the pinned sha256" >&2; exit 1; }
  done
fi

# Same overlay bytes on every node.
if [[ $OVERLAY == 1 ]]; then
  [[ $(basename "$OVERLAY_DIR") == overlay* && -f $HERE/overlay/sitecustomize.py ]] || { echo "OVERLAY=1 needs $HERE/overlay" >&2; exit 2; }
  sums=()
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then [[ $HERE/overlay -ef $OVERLAY_DIR ]] || rsync -a --delete --exclude __pycache__ "$HERE/overlay/" "$OVERLAY_DIR/"
    else on $r "mkdir -p $OVERLAY_DIR"; rsync -a --delete --exclude __pycache__ -e "ssh -i $SSH_KEY" "$HERE/overlay/" "$SSH_USER@${IPS[$r]}:$OVERLAY_DIR/"; fi
    sums+=("$(on $r "cd $OVERLAY_DIR && find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
  done
  (( $(printf "%s\n" "${sums[@]}" | sort -u | wc -l) == 1 )) || { echo "overlay differs across nodes: ${sums[*]}" >&2; exit 1; }
  echo "overlay ${sums[0]} on all nodes" >&2
fi
# TP helpers: same bytes on every node
if [[ $SPEC == dflash2 || $ROCE == 1 ]]; then
  sums=()
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then [[ $HERE/tp4 -ef $TP4_DIR ]] || rsync -a --delete --exclude __pycache__ "$HERE/tp4/" "$TP4_DIR/"
    else on $r "mkdir -p $TP4_DIR"; rsync -a --delete --exclude __pycache__ -e "ssh -i $SSH_KEY" "$HERE/tp4/" "$SSH_USER@${IPS[$r]}:$TP4_DIR/"; fi
    sums+=("$(on $r "cd $TP4_DIR && find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
  done
  (( $(printf "%s\n" "${sums[@]}" | sort -u | wc -l) == 1 )) || { echo "tp4/ differs across nodes: ${sums[*]}" >&2; exit 1; }
fi
# RoCEnante: same roce/ bytes on every node, the RDMA proxy .so built into each node's cache by a throwaway CPU
# container (no GPU, no network), then the in-image SGLang preflight
if [[ $ROCE == 1 ]]; then
  rsums=()
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then [[ $HERE/roce -ef $ROCE_DIR ]] || rsync -a --delete --exclude __pycache__ "$HERE/roce/" "$ROCE_DIR/"
    else on $r "mkdir -p $ROCE_DIR"; rsync -a --delete --exclude __pycache__ -e "ssh -i $SSH_KEY" "$HERE/roce/" "$SSH_USER@${IPS[$r]}:$ROCE_DIR/"; fi
    rsums+=("$(on $r "cd $ROCE_DIR && find . -type f -not -path '*/__pycache__/*' | LC_ALL=C sort | xargs sha256sum | sha256sum | cut -c1-16")")
    on $r "mkdir -p $CACHE_DIR/b12x-roce && docker run --rm --network none --memory 2g --entrypoint bash \
      -e B12X_ROCE_CACHE_DIR=/root/.cache/b12x-roce -e INKLING_TP4_ROCE_PATH=/opt/inkling-roce \
      -v $ROCE_DIR:/opt/inkling-roce:ro -v $TP4_DIR:/opt/inkling-tp4:ro -v $CACHE_DIR:/root/.cache \
      $SGLANG_IMAGE /opt/inkling-roce/build_proxy.sh" >&2 || { echo "rank $r: RoCE proxy build / preflight failed" >&2; exit 1; }
  done
  (( $(printf "%s\n" "${rsums[@]}" | sort -u | wc -l) == 1 )) || { echo "roce/ differs across nodes: ${rsums[*]}" >&2; exit 1; }
  echo "roce ${rsums[0]} on all nodes" >&2
fi
# memguard on every node (same script everywhere)
for r in 1 2 3; do scp -q -i "$SSH_KEY" "$HERE/scripts/memguard.sh" "$SSH_USER@${IPS[$r]}:$CACHE_DIR/memguard.sh"; done
cp "$HERE/scripts/memguard.sh" "$CACHE_DIR/memguard.sh"

# ---- engine command -------------------------------------------------------------------------------------------
spec=(); entry=(--entrypoint python3 "$SGLANG_IMAGE"); entry_env=()
case $SPEC in
  dspark) spec=(--speculative-algorithm DSPARK --speculative-num-draft-tokens "$NUM_DRAFT") ;;
  dflash2)
    spec=(--speculative-algorithm DFLASH --speculative-draft-model-path /draft --speculative-num-draft-tokens "$NUM_DRAFT"
          --speculative-draft-attention-backend fa4)
    # block verification is exact only with RSFIX's reject fix
    [[ ${BLOCK_VERIFY:-0} == 1 && $OVERLAY == 1 && ${INKLING_RSFIX:-0} =~ ^(1|.*reject.*)$ ]] && spec+=(--speculative-use-block-verification)
    # The drafter's SGLang compatibility patch rewrites installed sglang files, so it runs inside the container only
    # (it never outlives the run). Our TP pre-patch (overlay/tp4_dflash2_prepatch.py) first replaces the patch's
    # TP1-only guard with the per-shard vocabulary clamp, in a copy; the pinned file on the mount is never edited.
    entry_env=(-e SGLANG_ENABLE_DFLASH_DRAFT_KV_RING=1 -e DFLASH2_SHA256="$DFLASH2_SHA256" -v "$DFLASH2_DIR":/draft:ro)
    entry=(--entrypoint bash "$SGLANG_IMAGE"
           -c '(cd /draft && printf "%s\n" "$DFLASH2_SHA256" | sed "s/ /  /" | sha256sum -c --quiet -) || {
                 echo "DFlash2 drafter files do not match the pinned sha256" >&2; exit 1; }
               python3 -I /opt/inkling-tp4/tp4_dflash2_prepatch.py /draft/sglang_inkling_dflash2.py /tmp/dflash2_patch_tp.py &&
               python3 -I /tmp/dflash2_patch_tp.py && exec python3 "$@"' dflash2) ;;
  none) ;;
  *) echo "SPEC must be dspark, dflash2 or none" >&2; exit 2 ;;
esac
nommap_arg=(); nommap_env=()
[[ $LOAD_NOMMAP == 1 ]] && { nommap_arg=(--weight-loader-disable-mmap); nommap_env=(-e MALLOC_MMAP_THRESHOLD_=1048576); }
nccl_env=()
[[ -n $NCCL_DIR ]] && nccl_env=(-v "$NCCL_DIR":/opt/nccl:ro -e LD_PRELOAD=/opt/nccl/libnccl.so.2.30.7
                                -e SGLANG_NCCL_SO_PATH=/opt/nccl/libnccl.so.2.30.7)
ov=()
if [[ $OVERLAY == 1 ]]; then
  ov=(-v "$OVERLAY_DIR":/opt/inkling-overlay:ro)
  # files the levers read must come from the overlay (identical bytes on every node, checked above)
  if [[ -n ${INKLING_DRAFT_HOTVOCAB_FILE:-} && $INKLING_DRAFT_HOTVOCAB_FILE != /opt/inkling-overlay/* ]]; then
    echo "INKLING_DRAFT_HOTVOCAB_FILE must be under /opt/inkling-overlay (same bytes on every rank)" >&2; exit 2
  fi
  # every INKLING_* variable of the caller's environment reaches every rank's container WITH its value (a bare
  # `-e NAME` would be read from the remote shell's environment, where it is unset: rank-divergent levers deadlock)
  while IFS= read -r kv; do ov+=(-e "$kv"); done < <(env | grep -E '^INKLING_[A-Z0-9_]*=' || true)
fi
tp_tools=(); [[ $SPEC == dflash2 || $ROCE == 1 ]] && tp_tools=(-v "$TP4_DIR":/opt/inkling-tp4:ro)
# one PYTHONPATH: tp4/ first (its sitecustomize installs the RoCE hooks, then chains to the overlay's)
pp=(); [[ $ROCE == 1 ]] && pp+=(/opt/inkling-tp4); [[ $OVERLAY == 1 ]] && pp+=(/opt/inkling-overlay)
roce_env=()
if [[ $ROCE == 1 ]]; then
  tp_tools+=(-v "$ROCE_DIR":/opt/inkling-roce:ro)
  roce_env=(-e INKLING_TP4_ROCE=1 -e INKLING_TP4_ROCE_PATH=/opt/inkling-roce -e B12X_ROCE_HCA="$HCA"
            -e B12X_ROCE_CACHE_DIR=/root/.cache/b12x-roce -e B12X_ROCE_SPIN_LIMIT="${B12X_ROCE_SPIN_LIMIT:-300000000}")
  for k in INKLING_TP4_ROCE_MAX_SIZE INKLING_TP4_ROCE_GATHER_MAX_SIZE INKLING_TP4_ROCE_REQUIRE INKLING_TP4_ROCE_SELFTEST; do
    [[ -n ${!k:-} ]] && roce_env+=(-e "$k=${!k}")
  done
fi
(( ${#pp[@]} )) && roce_env+=(-e "PYTHONPATH=$(IFS=:; echo "${pp[*]}")")

# PROFILE=1: an admin key for /start_profile (ADMIN_FORCE endpoint), random per boot, kept only in $CACHE_DIR/admin-key
# on the head (mode 600; tools/profile_c1.sh reads it). Not for benchmark arms: admin-optional endpoints then need it.
admin_args=()
if [[ ${PROFILE:-0} == 1 ]]; then
  ( umask 077; head -c 24 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CACHE_DIR/admin-key" )
  admin_args=(--admin-api-key "$(cat "$CACHE_DIR/admin-key")")
fi
RUN_ID=inkling-tp4-$(date +%Y%m%d%H%M%S)-$$
started=()
cleanup_failed_start() {
  local rc=$?; trap - EXIT
  if (( rc != 0 )); then
    echo "launch failed (rc=$rc): removing containers labelled inkling.launch=$RUN_ID" >&2
    for r in ${started[@]+"${started[@]}"}; do
      on $r "docker ps -aq --filter label=inkling.launch=$RUN_ID | xargs -r docker rm -f >/dev/null 2>&1 || true" || true
    done
  fi
  exit $rc
}
trap cleanup_failed_start EXIT

for r in 0 1 2 3; do
  env=(-e NCCL_SOCKET_IFNAME="$IFACE" -e GLOO_SOCKET_IFNAME="$IFACE"
       -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="$HCA" -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_ROCE_VERSION_NUM=2
       -e NCCL_NET_PLUGIN=none -e NCCL_P2P_DISABLE=1 -e NCCL_SHM_DISABLE=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0
       -e NCCL_BUFFSIZE=1048576 -e NCCL_LL128_BUFFSIZE=262144 -e NCCL_PROTO=^LL128 -e NCCL_MAX_NCHANNELS=8
       -e NCCL_DEBUG=WARN -e OMP_NUM_THREADS=1
       -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
       -e MAX_JOBS=4 -e SGLANG_OPT_INKLING2_VERIFY_REL_BIAS_KERNEL=1
       -e SGLANG_OPT_DSV41_DSPARK_MID_FOLD=1 -e SGLANG_OPT_DSV41_DSPARK_PREPARE_INPUTS_ONCE=1
       -e SGLANG_OPT_DSV41_DSPARK_SKIP_PREFILLED_INPUTS=1)
  args=(-m sglang.launch_server --model-path /model --served-model-name "$SERVED"
        --tp 4 --nnodes 4 --node-rank $r --dist-init-addr "${IPS[0]}:$DIST_PORT"
        --attention-backend fa4 --dtype bfloat16 --page-size 128 --disable-custom-all-reduce
        ${spec[@]+"${spec[@]}"}
        --mem-fraction-static "$MEM_FRACTION" --context-length "$CTX" --max-running-requests "$MAX_RUNNING"
        --grammar-backend xgrammar --reasoning-parser inkling --tool-call-parser inkling2
        --enable-metrics --host "$HOST" --port "$PORT" ${nommap_arg[@]+"${nommap_arg[@]}"}
        --swa-full-tokens-ratio "$SWA_RATIO" --max-mamba-cache-size 512 --max-prefill-tokens 8192
        ${admin_args[@]+"${admin_args[@]}"} $SGLANG_ARGS "$@")
  run=(docker run -d --name "$NAME-r$r" --label inkling.launch=$RUN_ID --gpus all --network host --ipc host
       --device /dev/infiniband --cap-add IPC_LOCK --ulimit memlock=-1 --ulimit stack=67108864 --restart no
       -v "$MODEL_DIR":/model:ro -v "$CACHE_DIR":/root/.cache
       ${nccl_env[@]+"${nccl_env[@]}"} ${ov[@]+"${ov[@]}"} ${tp_tools[@]+"${tp_tools[@]}"} ${roce_env[@]+"${roce_env[@]}"} "${env[@]}" $EXTRA_ENV
       ${nommap_env[@]+"${nommap_env[@]}"} ${entry_env[@]+"${entry_env[@]}"} "${entry[@]}" "${args[@]}")
  started+=($r)
  [[ $r == 0 ]] && echo "+ ${run[*]}" | sed 's/--admin-api-key [0-9a-f]*/--admin-api-key <redacted>/' >&2
  on $r "$(printf '%q ' "${run[@]}") >/dev/null && { nohup $CACHE_DIR/memguard.sh $NAME-r$r $MIN_AVAIL_GIB 172800 \
         > $CACHE_DIR/memguard-$NAME-r$r.log 2>&1 < /dev/null & }"
done
trap - EXIT
# every rank must see the same INKLING_* settings (levers that differ per rank deadlock the collectives)
esums=()
for r in 0 1 2 3; do
  esums+=("$(on $r "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $NAME-r$r | grep -E '^(INKLING_|B12X_|NCCL_|PYTHONPATH=)' | LC_ALL=C sort | sha256sum | cut -c1-16")")
done
if (( $(printf "%s\n" "${esums[@]}" | sort -u | wc -l) != 1 )); then
  echo "rank environments differ (${esums[*]}): taking all ranks down" >&2; "$0" down; exit 1
fi
echo "rank environments identical (${esums[0]})" >&2
echo "started $NAME on 4 ranks (run $RUN_ID); API on $HOST:$PORT of rank 0 once ready (docker logs -f $NAME-r0)"
