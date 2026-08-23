#!/usr/bin/env bash
# Session watcher: poll fleet reachability every 3 min; EXIT (wakes the agent)
# as soon as all 16 spark hosts are ssh-reachable -> Phase 3 window trigger.
LOG=/Users/mac/dsh.sparkpipe/tmp/k3_fleet_watch.log
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
for i in $(seq 1 240); do
  ts=$(date -u +%FT%TZ)
  rm -rf /tmp/k3fw.$$ ; mkdir -p /tmp/k3fw.$$
  for h in $HOSTS; do
    ( timeout 6 ssh -o ConnectTimeout=4 -o BatchMode=yes "$h" true 2>/dev/null && touch "/tmp/k3fw.$$/$h" ) &
  done
  wait
  up=$(ls /tmp/k3fw.$$ | wc -l | tr -d ' ')
  uplist=$(ls /tmp/k3fw.$$ | tr '\n' ',' )
  rm -rf /tmp/k3fw.$$
  echo "$ts fleet_up=$up/16 [$uplist]" >> "$LOG"
  if [ "$up" -ge 16 ]; then echo "$ts TRIGGER: fleet 16/16 -> run tools/devcycle/fleet_status.sh probe + fleet_swap.sh k3" >> "$LOG"; exit 0; fi
  sleep 180
done
echo "$(date -u +%FT%TZ) EXPIRED after 240 polls without 16/16" >> "$LOG"
exit 3
