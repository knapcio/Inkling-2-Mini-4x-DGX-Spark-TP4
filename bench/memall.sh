#!/usr/bin/env bash
# memall.sh OUT.tsv: MemAvailable (GiB) of the four ranks every 5 s until killed: date, r0..r3.
IPS=(${IPS:-10.100.96.2 10.100.96.1 10.100.96.3 10.100.96.4}); KEY=${SSH_KEY:-$HOME/.ssh/id_ed25519_nvsync_cluster_assistant}
while true; do
  line="$(date -Is)"
  for r in 0 1 2 3; do
    if [[ $r == 0 ]]; then m=$(awk '/MemAvailable/ {printf "%.2f", $2/1048576}' /proc/meminfo)
    else m=$(ssh -o BatchMode=yes -i "$KEY" "${IPS[$r]}" "awk '/MemAvailable/ {printf \"%.2f\", \$2/1048576}' /proc/meminfo" 2>/dev/null || echo NA); fi
    line+=$'\t'$m
  done
  echo "$line"; sleep 5
done > "$1"
