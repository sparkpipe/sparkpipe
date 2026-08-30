#!/bin/bash
# DSV4 Flash M1 task B (queued, GPU): launch TP4 residentd on spark4-7,
# wait ready, run the O128 B1 cell 3x with line timestamps, verify the
# exact token hash BEFORE any timing claim (mismatch = RED stop), then
# TERM the daemons cwd-filtered. Runs from spark5.
set -uo pipefail
BASE=/home/spark5/lane-dsv4flash-m1
RT=$BASE/rt; RES=$BASE/results; STAGE=$BASE/staging
EXPECTED=211462f2525f73b76137ee1ce9bd4e015ad8a3fd825a7c45d38fff0488598083
HOSTS="spark4 spark5 spark6 spark7"
declare -A RANKOF=( [spark4]=0 [spark5]=1 [spark6]=2 [spark7]=3 )

term_all() {
  for h in $HOSTS; do
    ssh -o BatchMode=yes $h 'for p in $(pgrep -f "bin/sparkpipe_model_[r]esidentd" 2>/dev/null); do [ "$(readlink /proc/$p/cwd)" = "/home/'$h'/lane-dsv4flash-m1/rt" ] && kill -TERM $p && echo "TERM '$h' $p"; done; true'
  done
}
trap term_all EXIT

echo "== [1/4] TERM any prior lane daemons + drop caches + launch"
for h in $HOSTS; do
  ssh -o BatchMode=yes $h 'for p in $(pgrep -f "bin/sparkpipe_model_[r]esidentd" 2>/dev/null); do [ "$(readlink /proc/$p/cwd)" = "/home/'$h'/lane-dsv4flash-m1/rt" ] && kill -TERM $p; done; true'
done
sleep 2
for h in $HOSTS; do
  r=${RANKOF[$h]}
  ssh -o BatchMode=yes $h "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null; cd /home/$h/lane-dsv4flash-m1/rt && rm -f residentd.log && nohup ./bin/sparkpipe_model_residentd --deployment config/model_resident.json --rank-index $r > residentd.log 2>&1 < /dev/null & echo launched-$h-rank$r"
done

echo "== [2/4] ready-wait (max 20 min)"
for h in $HOSTS; do
  ok=""
  for i in $(seq 1 240); do
    if ssh -o BatchMode=yes $h "grep -q 'model_residentd ready' /home/$h/lane-dsv4flash-m1/rt/residentd.log 2>/dev/null"; then ok=1; echo "$h READY"; break; fi
    if ! ssh -o BatchMode=yes $h "pgrep -f 'bin/sparkpipe_model_[r]esidentd' >/dev/null"; then
      echo "$h DIED"; ssh -o BatchMode=yes $h "tail -20 /home/$h/lane-dsv4flash-m1/rt/residentd.log"; exit 1
    fi
    sleep 5
  done
  [ -n "$ok" ] || { echo "$h TIMEOUT-READY"; exit 1; }
done

echo "== [3/4] bench x3 (exactness gate first)"
RED=0
for i in 1 2 3; do
  ssh -o BatchMode=yes spark4 "cd /home/spark4/lane-dsv4flash-m1/rt && ./bin/sparkpipe_model_batch --deployment config/model_resident.json --runtime-root /home/spark4/lane-dsv4flash-m1/rt --batch $STAGE/devcycle-o128-batch.json" \
    | python3 -u "$STAGE/m1_timestamp.py" > "$RES/bench_m1_run$i.jsonl" 2> "$RES/bench_m1_run$i.stderr"
  v=$(python3 "$STAGE/m1_parse.py" "$RES/bench_m1_run$i.jsonl")
  echo "run$i: $v"
  echo "run$i: $v" >> "$RES/m1_bench_summary.txt"
  case "$v" in *HASH-OK*) ;; *) echo "RED-STOP exactness mismatch run$i"; RED=1; break;; esac
done

echo "== [4/4] TERM daemons"
term_all
trap - EXIT
if [ "$RED" = "1" ]; then echo "BENCH-TASK-RED"; exit 1; fi
echo "BENCH-TASK-DONE"
