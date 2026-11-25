#!/usr/bin/env bash
# download.sh [DIR] | --dflash2 [DIR]: thinkingmachines/Inkling-2-Mini at the pinned revision into DIR (default ~/models/Inkling-2-Mini),
# with plain curl (public repository, no token), resumable, then sha256 checks of the large files; then the DFlash2
# drafter the default profile uses (below; NO_DFLASH2=1 skips it). About 21 GB. On a Spark the download fills the
# page cache: do not start a model server while it runs.
set -euo pipefail
SELF=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/download.sh
# verify "sha256 file" lines on stdin; a failing file is removed so the next run fetches it again (curl -C - would
# otherwise resume a complete but corrupt file and never fix it)
verify() {
  local h f bad=0
  while read -r h f; do
    [[ -n $h ]] || continue
    if echo "$h  $f" | sha256sum -c --quiet - 2>/dev/null; then echo "$f: OK"
    else echo "$f: checksum mismatch, removed" >&2; rm -f "$f"; bad=1; fi
  done
  (( bad == 0 )) || { echo "run the download again" >&2; exit 1; }
}
if [[ ${1:-} == --dflash2 ]]; then
  # download.sh --dflash2 [DIR]: the third-party DFlash2 drafter (incoai/Inkling-2-Mini-DFlash2, Apache-2.0) for the
  # default SPEC=dflash2 profile, at the pinned revision into DIR (default ~/models/Inkling-2-Mini-DFlash2), about
  # 1 GB. Only the files the profile reads are fetched; scripts/launch.sh checks the same sha256 on every start.
  REV=4f9088ea66c21f2b99ed657d1941c306b03e84d2
  DST=${2:-${DFLASH2_DIR:-$HOME/models/Inkling-2-Mini-DFlash2}}
  BASE=https://huggingface.co/incoai/Inkling-2-Mini-DFlash2/resolve/$REV
  mkdir -p "$DST"
  cd "$DST"
  for f in README.md config.json sglang_inkling_dflash2.py model.safetensors; do
    echo "fetching $f" >&2
    curl -fSL -C - --retry 5 --progress-bar -o "$f" "$BASE/$f"
  done
  verify <<SUMS
14223ae2f3aa1d661da67cf0143ec33070d952a6e28a119e6cec64fcc0bebe50  sglang_inkling_dflash2.py
d094e435eb939e01fca6297f637c5af35e3da4b552f8e1ab0a181cd278a1cefa  config.json
217980ab00d116693fa5946df7bbfd967caf5fac1f529afd541fbd470ddb54ee  model.safetensors
SUMS
  echo "$REV" > .revision
  echo "Inkling-2-Mini-DFlash2 $REV in $DST"
  exit 0
fi
REV=ed40e7da625f594fc9c4990f5f66be24873c22d7
DST=${1:-${MODEL_DIR:-$HOME/models/Inkling-2-Mini}}
BASE=https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/$REV
mkdir -p "$DST/tiktoken"
cd "$DST"
for f in .gitattributes README.md chat_template.jinja config.json generation_config.json hf_quant_config.json \
         model.safetensors.index.json processor_config.json special_tokens_map.json tokenizer.json \
         tokenizer_config.json tiktoken/tokenizer.model mtp.safetensors model-00001-of-00001.safetensors; do
  echo "fetching $f" >&2
  curl -fSL -C - --retry 5 --progress-bar -o "$f" "$BASE/$f"
done
verify <<SUMS
65e60f51aa1284d4fb022584908c315270334d991828bb774af60584523539d1  model-00001-of-00001.safetensors
7c3b651317f01f73bcf4e1dceba66ef8b7e1a1eac6bc171e882e690a4b10a1d2  mtp.safetensors
7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6  tokenizer.json
SUMS
echo "$REV" > .revision
echo "Inkling-2-Mini $REV in $DST"
# the DSpark (SPEC=dspark) and vLLM profiles do not read the drafter
[[ ${NO_DFLASH2:-0} == 1 || ${SPEC:-dflash2} != dflash2 || ${ENGINE:-sglang} == vllm ]] || exec "$SELF" --dflash2
