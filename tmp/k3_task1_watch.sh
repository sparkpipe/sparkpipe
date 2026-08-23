#!/bin/bash
# Watcher: polls spark6 pack completion + fleet reachability every 5 min.
LOG=/Users/mac/dsh.sparkpipe/tmp/k3_task1_watch.log
HOSTS="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
for i in $(seq 1 120); do
  ts=$(date -u +%FT%TZ)
  sz=$(timeout 10 ssh -o ConnectTimeout=6 -o BatchMode=yes spark6 'stat -c %s /home/spark6/k3tp16prod/k3.full.tilek32.pack.payload 2>/dev/null; ls /home/spark6/k3tp16prod/SHA256SUMS.tp16 >/dev/null 2>&1 && echo SUMS-LANDED; pgrep -c -f k3_pack.py' 2>/dev/null | tr '\n' ' ')
  rm -rf /tmp/k3w.$$ ; mkdir -p /tmp/k3w.$$
  for h in $HOSTS; do
    ( timeout 6 ssh -o ConnectTimeout=4 -o BatchMode=yes "$h" true 2>/dev/null && touch "/tmp/k3w.$$/$h" ) &
  done
  wait
  up=$(ls /tmp/k3w.$$ | wc -l | tr -d ' ')
  rm -rf /tmp/k3w.$$
  echo "$ts payload=$sz fleet_up=$up/16" >> "$LOG"
  if echo "$sz" | grep -q SUMS-LANDED && [ "$up" -ge 16 ]; then echo "$ts READY: sums landed + 16/16 fleet" >> "$LOG"; break; fi
  sleep 300
done
