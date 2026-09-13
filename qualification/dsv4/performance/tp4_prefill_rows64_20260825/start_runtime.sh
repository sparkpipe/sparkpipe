#!/usr/bin/env bash
# start_runtime.sh NAME ROOT [GIT_COMMIT] - launch residentd on spark4-7
# for the given runtime root (rank i = 0..3), wait for control readiness.
set -u

NAME="${1:?usage: start_runtime.sh NAME ROOT [GIT_COMMIT]}"
ROOT="${2:?usage: start_runtime.sh NAME ROOT [GIT_COMMIT]}"
GIT_COMMIT="${3:-14024cf7895930cd92f07871d38876ea1e944fae}"
RANKS=(spark4 spark5 spark6 spark7)

i=0
for h in "${RANKS[@]}"; do
  if ssh -o BatchMode=yes "$h" "ss -ltnH | grep -q :18480" 2>/dev/null; then
    echo "start $h rank$i SKIP port-bound"
    i=$((i+1))
    continue
  fi
  ssh -o BatchMode=yes "$h" "cd $ROOT && LD_LIBRARY_PATH=$ROOT/lib:\$LD_LIBRARY_PATH SPARKPIPE_RELEASE_GENERATION=20260815000000 SPARKPIPE_RELEASE_GIT_COMMIT=$GIT_COMMIT SPARKPIPE_RELEASE_ID=prefill-rows64-$NAME-rank$i setsid -f bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $i >/tmp/prefill-rows64-$NAME-rank$i.log 2>&1 </dev/null" || { echo "launch $h FAILED"; exit 1; }
  echo "launched $h rank$i"
  i=$((i+1))
done

ready=0
for attempt in $(seq 1 30); do
  ready=0
  for h in "${RANKS[@]}"; do
    if ssh -o BatchMode=yes "$h" "ss -ltnH | grep -q :18480"; then ready=$((ready+1)); fi
  done
  if [ "$ready" = 4 ]; then echo "ALL-READY ($NAME)"; exit 0; fi
  sleep 10
done
echo "READY-TIMEOUT $ready/4 - log tails:"
i=0
for h in "${RANKS[@]}"; do
  echo "--- rank$i ($h) ---"
  ssh -o BatchMode=yes "$h" "tail -n 5 /tmp/prefill-rows64-$NAME-rank$i.log" 2>/dev/null
  i=$((i+1))
done
exit 1
