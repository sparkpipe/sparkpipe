#!/usr/bin/env bash
for k in 1 2 3 4 5 6 7 8 9 10 11 12; do
  sleep 25
  c=$(ssh -o BatchMode=yes spark6 'ss -ltn | grep -c 18480' 2>/dev/null)
  if [ -n "$c" ] && [ "$c" -ge 1 ]; then echo "rank2-up-at-t=$((k*25))s"; exit 0; fi
done
echo rank2-still-down-after-300s
ssh -o BatchMode=yes spark6 'tail -2 /tmp/dsv4flash-serving-rank2.restore.log'
