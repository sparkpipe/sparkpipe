#!/usr/bin/env bash
# wait_ready.sh — poll until all four ranks listen on 18480 (or fail).
for k in 1 2 3 4 5 6 7 8 9 10 11 12 13 14; do
  sleep 20
  up=0
  for h in spark4 spark5 spark6 spark7; do
    if ssh -o BatchMode=yes "$h" 'ss -ltn' 2>/dev/null | grep -q 18480; then
      up=$((up+1))
    fi
  done
  echo "t=$((k*20))s ready=$up/4"
  if [ "$up" -eq 4 ]; then exit 0; fi
done
echo NOT-READY
exit 1
