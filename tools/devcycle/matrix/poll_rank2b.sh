#!/usr/bin/env bash
for k in 1 2 3 4 5 6 7 8; do
  sleep 20
  c=$(ssh -o BatchMode=yes spark6 'ss -ltn | grep -c 18480' 2>/dev/null)
  if [ -n "$c" ] && [ "$c" -ge 1 ]; then echo "rank2-up-at-t=$((k*20))s"; exit 0; fi
  echo "t=$((k*20))s down"
done
ssh -o BatchMode=yes spark6 'tail -1 /tmp/dsv4flash-serving-rank2.restore.log'
