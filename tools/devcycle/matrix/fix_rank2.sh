#!/usr/bin/env bash
ssh -o BatchMode=yes spark6 'pkill -9 -f b8_residentd'
ssh -o BatchMode=yes spark6 'bash /tmp/start_renamed.sh 2'
sleep 45
up=0
for h in spark4 spark5 spark6 spark7; do
  if ssh -o BatchMode=yes "$h" 'ss -ltn' 2>/dev/null | grep -q 18480; then
    up=$((up+1))
  fi
done
echo "ready=$up/4"
