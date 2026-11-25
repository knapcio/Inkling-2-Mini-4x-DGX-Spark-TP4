#!/usr/bin/env bash
# memguard.sh NAME MIN_AVAIL_GIB SECONDS (decimal GiB allowed): stop container NAME if MemAvailable < MIN_AVAIL_GIB for two samples 2 s apart.
# Started in the background by launch.sh up. GB10 shares one memory pool between GPU and OS; running it out can
# hang the box until a power cycle, so a stopped container is the cheaper failure.
set -u
name=$1; min_gib=$2; secs=$3
min_kb=$(awk -v g="$min_gib" 'BEGIN { printf "%d", g * 1048576 }')  # decimal GiB allowed (e.g. 3.5)
n=0; low=0; end=$((SECONDS + secs)); lowest=999999
while (( SECONDS < end )) && docker ps --format '{{.Names}}' | grep -qx "$name"; do
  kb=$(awk '/MemAvailable/ {print $2}' /proc/meminfo)
  (( kb / 1024 < lowest )) && lowest=$((kb / 1024))
  if (( kb < min_kb )); then low=$((low + 1)); else low=0; fi
  if (( low >= 2 )); then
    echo "$(date -Is) MemAvailable $((kb / 1024)) MiB < $min_gib GiB: stopping $name"
    docker stop -t 5 "$name"; exit 1
  fi
  sleep 2
  n=$((n + 1)); (( n % 30 == 0 )) && echo "$(date -Is) MemAvailable $((kb / 1024)) MiB, lowest $lowest MiB"
done
echo "$(date -Is) guard done; lowest MemAvailable seen: $lowest MiB"
