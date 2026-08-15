#!/usr/bin/env bash
# k3_wait_and_deploy.sh - poll the four stage nodes until all 16 TP4 rank
# packs exist (slicing complete), then deploy them to the rank hosts with
# sha256 verification. Run from the workstation.
set -u
declare -A STAGE_HOST=( [0]=spark1 [1]=spark4 [2]=spark8 [3]=sparkc )
declare -A STAGE_PREFIX=( [0]=/tmp/k3_stage_0_24 [1]=/tmp/k3_stage_24_23 [2]=/tmp/k3_stage_47_23 [3]=/tmp/k3_stage_70_23 )
while :; do
  ready=1
  for s in 0 1 2 3; do
    for t in 0 1 2 3; do
      f="${STAGE_PREFIX[$s]}.rank0${t}.pack"
      info=$(ssh "${STAGE_HOST[$s]}" "stat -c '%s' $f 2>/dev/null; pgrep -f 'k3_shard.py' >/dev/null && echo RUNNING || echo DONE" 2>/dev/null)
      size=$(echo "$info" | head -1)
      state=$(echo "$info" | tail -1)
      [ "${size:-0}" -gt 90000000000 ] && [ "$state" = "DONE" ] || ready=0
    done
  done
  if [ $ready = 1 ]; then echo "all 16 rank packs present $(date)"; break; fi
  sleep 60
done
bash tools/k3_deploy_ranks.sh
