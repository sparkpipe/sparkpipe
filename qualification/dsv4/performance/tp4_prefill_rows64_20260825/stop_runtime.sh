#!/usr/bin/env bash
# stop_runtime.sh - stop any residentd serving config/model_resident.json
# on spark4-7 (prefill-rows64 experiment cleanup / config switch).
set -u

for h in spark4 spark5 spark6 spark7; do
  pids=$(ssh -o BatchMode=yes "$h" "pgrep -f 'bin/sparkpipe_model_residentd --deployment config/model_resident.json'" 2>/dev/null || true)
  if [ -n "$pids" ]; then
    ssh -o BatchMode=yes "$h" "kill $pids 2>/dev/null; sleep 1; kill -9 $pids 2>/dev/null" || true
  fi
  echo "stop $h done"
done
