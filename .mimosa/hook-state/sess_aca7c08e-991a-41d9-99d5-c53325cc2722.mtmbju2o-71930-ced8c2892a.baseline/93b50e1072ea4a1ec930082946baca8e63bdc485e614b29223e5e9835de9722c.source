#!/usr/bin/env bash
# usage: k3_autoslice.sh STAGE_PACK OUT_PREFIX
set -u
PACK="$1"; PREFIX="$2"
while :; do
  if [ -f "$PACK" ]; then
    s1=$(stat -c %s "$PACK" 2>/dev/null || echo 0)
    sleep 10
    s2=$(stat -c %s "$PACK" 2>/dev/null || echo 0)
    if [ "$s1" = "$s2" ] && [ ! -e "$PACK.payload" ]; then break; fi
  else
    sleep 30
  fi
done
PYTHONDONTWRITEBYTECODE=1 python3 /tmp/k3_shard.py "$PACK" "$PREFIX" 4
rc=$?
echo "shard rc=$rc $(date)" > "$PREFIX.autoslice.log"
for rank in 0 1 2 3; do
  out="${PREFIX}.rank0${rank}.pack"
  [ -s "$out" ] && du -sh "$out" >> "$PREFIX.autoslice.log"
done
echo "autoslice done $(date)" >> "$PREFIX.autoslice.log"
