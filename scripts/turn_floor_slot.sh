#!/usr/bin/env bash
set -u
[ -d "$HOME/bin/sparkpipe-kit-ssh" ] && PATH=$HOME/bin/sparkpipe-kit-ssh:$PATH
KIT=$(cd "$(dirname "$0")" && pwd)
. "$KIT/rollover_lib.sh"
[ $# -eq 1 ] || { echo "usage: SLOT_FW=<local node files> SLOT_API_BUILD=<hub x86 build> SLOT_TOKENS=<hub token file> SLOT_OUT=<local receipts> $0 RUN" >&2; exit 2; }
run=$1
case "$run" in ''|*[!A-Za-z0-9._-]*) echo "the run id '$run' may hold letters, digits, '.', '_' and '-'" >&2; exit 2;; esac
SLOT_FW=${SLOT_FW:?SLOT_FW is the local folder with the node files of the build under test}
SLOT_API_BUILD=${SLOT_API_BUILD:?SLOT_API_BUILD is the hub folder with the x86 build of the same commit}
SLOT_TOKENS=${SLOT_TOKENS:?SLOT_TOKENS is the hub file of token ids from the tokenizer of the model}
SLOT_OUT=${SLOT_OUT:?SLOT_OUT is the local folder the receipts are copied to}
SLOT_MODEL=${SLOT_MODEL:-dsv41_flash_tp4}
SLOT_LABEL=${SLOT_LABEL:-deepseek-v4.1-flash-tp4}
SLOT_FIRST=${SLOT_FIRST:-4}
SLOT_LANE=${SLOT_LANE:-1}
SLOT_PORT=${SLOT_PORT:-$((8491 + SLOT_FIRST / 4))}
SLOT_CONTEXTS=${SLOT_CONTEXTS:-8000,50000}
SLOT_WAVE=${SLOT_WAVE:-60000}
SLOT_MIN_FREE_GIB=${SLOT_MIN_FREE_GIB:-100}
SLOT_DIR=${SLOT_DIR:-$HUB_WORK/turn-floor-slot}
SLOT_DRY=${SLOT_DRY:-0}
export STAGE_MODEL=$SLOT_MODEL STAGE_LANE=$SLOT_LANE STAGE_FIRST_NODE=$SLOT_FIRST STAGE_FIRMWARE=$SLOT_FW
export STAGE_SEQUENCES=${STAGE_SEQUENCES:-16} STAGE_ROWS=${STAGE_ROWS:-16384} STAGE_KV_PAGES=${STAGE_KV_PAGES:-1024} STAGE_KV_PHYSICAL_PAGES_CAP=${STAGE_KV_PHYSICAL_PAGES_CAP:-8192}
export STAGE_KV_LOGICAL_PAGES=${STAGE_KV_LOGICAL_PAGES:-32768} STAGE_STATE_BUDGET_BYTES=${STAGE_STATE_BUDGET_BYTES:-$((16 << 30))}
export STAGE_WEIGHTD_SOCKET=${STAGE_WEIGHTD_SOCKET:-/tmp/spark_weightd.sock} STAGE_MEMORY_MAX=${STAGE_MEMORY_MAX:-40G}
LANE=$KIT/../tools/stage_lane.sh
ranks=$(python3 -c "import sys; sys.path.insert(0, '$KIT/../tools'); import stage_lane; print(len(stage_lane.lane_hosts(stage_lane.MODELS['$SLOT_MODEL'], $SLOT_FIRST)))") ||
  { echo "SLOT-REFUSED: '$SLOT_MODEL' is not a tools/stage_lane.py model" >&2; exit 2; }
adapter=$(python3 -c "import sys; sys.path.insert(0, '$KIT/../tools'); import stage_lane; print(stage_lane.MODELS['$SLOT_MODEL']['adapter'])")
hosts=""; for r in $(seq "$SLOT_FIRST" $((SLOT_FIRST + ranks - 1))); do hosts="$hosts spark${HEX:$r:1}"; done
unit=tf-slot-api-$SLOT_FIRST
dir=$SLOT_DIR/$run
node_files="sparkpipe_model_residentd weightd_warm $adapter libhidden_transport_spark_host_rdma_verbs.so runtime_libs/libsparkpipe_device_cuda.so"
api_files="sparkpipe_model_api sparkpipe_tokenize_prompt $adapter runtime_libs/libsparkpipe_device_cuda.so"
tf="python3 ../turn_floor.py run --endpoint http://127.0.0.1:$SLOT_PORT --tokens $SLOT_TOKENS --api-log $dir/api.log --label $SLOT_LABEL --output . --contexts $SLOT_CONTEXTS --tails 1,40,500,5000 --repeats 3 --spec-repeats 0 --once-from 2000 --decode-tokens 96 --served-tokens 128 --wave-tokens $SLOT_WAVE"
copy() { if command -v sparkcp > /dev/null 2>&1; then sparkcp "$@"; else python3 "$KIT/../tools/sparkcp.py" "$@"; fi; }
root_of() { echo "/home/$1/stage-lanes/$SLOT_MODEL/lane$SLOT_LANE/root"; }
on() { ssh -o BatchMode=yes -o ConnectTimeout=8 "$1" "$2" < /dev/null; }
free_gib() { on "$1" "awk '/MemAvailable/ {printf \"%.0f\", \$2/1048576}' /proc/meminfo"; }

if [ "$SLOT_DRY" = 1 ]; then
  cat <<PLAN
slot $run: $SLOT_MODEL on$hosts, mesh lane $SLOT_LANE, API unit $unit on port $SLOT_PORT, hub folder $dir
refuse unless: the controller reads "idle None None"; $SLOT_FW has: $node_files; $SLOT_API_BUILD has: $api_files; both SOURCE_COMMIT files are equal; each node has $SLOT_MIN_FREE_GIB GiB available
lane: reclaim an old root; stage_lane.sh setup (one retry); STAGE_RUN_ID=$run stage_lane.sh start; wait for READY on $ranks nodes
api: stage_lane.sh api-setup from $SLOT_API_BUILD; copy to ~/$unit; start $unit; wait for $ranks connected ranks
hub: cd $dir && $tf --arms turns
hub: cd $dir && $tf --arms truth --event after-the-turns
hub: cd $dir && $tf --arms wave
restart: stop $unit; stage_lane.sh stop; STAGE_RUN_ID=${run}r stage_lane.sh start; start $unit again with its prefix index kept
hub: cd $dir && $tf --arms truth --event before-the-resume
hub: cd $dir && $tf --arms resume --event restart
hub: cd $dir && $tf --arms truth --event after-the-resume
hub: cd $dir && python3 ../turn_floor.py parts . && python3 ../turn_floor.py excerpt . --output engine.log
copy: hub:$dir/{requests.jsonl,parts.jsonl,summary.json,engine.log,run.log} to $SLOT_OUT/
always: stop $unit; stage_lane.sh stop; stage_lane.sh reclaim; MemAvailable on each node
PLAN
  exit 0
fi

state=$(controller 2>/dev/null)
[ "$state" = "idle None None" ] || { echo "SLOT-REFUSED: the controller reads '${state:-nothing}', not 'idle None None'; a lane never starts beside an active model"; exit 1; }
for f in $node_files; do [ -s "$SLOT_FW/$f" ] || { echo "SLOT-REFUSED: $SLOT_FW/$f is missing"; exit 1; }; done
hub "for f in $api_files; do test -s $SLOT_API_BUILD/\$f || exit 1; done; test -s $SLOT_TOKENS" || { echo "SLOT-REFUSED: the hub build $SLOT_API_BUILD or the token file $SLOT_TOKENS is incomplete"; exit 1; }
node_commit=$(cat "$SLOT_FW/SOURCE_COMMIT" 2>/dev/null); api_commit=$(hub "cat $SLOT_API_BUILD/../SOURCE_COMMIT 2>/dev/null")
[ -n "$node_commit" ] && [ "$node_commit" = "$api_commit" ] || { echo "SLOT-REFUSED: node files are of '${node_commit:-no commit}' and the hub build of '${api_commit:-no commit}'"; exit 1; }
for h in $hosts; do
  g=$(free_gib "$h")
  echo "$h MemAvailable ${g:-?} GiB"
  [ "${g:-0}" -ge "$SLOT_MIN_FREE_GIB" ] || { echo "SLOT-REFUSED: $h has under $SLOT_MIN_FREE_GIB GiB available"; exit 1; }
done
work=$(mktemp -d) && cd "$work" || exit 1
mkdir -p "$SLOT_OUT" || exit 1

down() {
  stamp "slot $run teardown"
  hub "systemctl --user stop $unit 2>/dev/null; systemctl --user reset-failed $unit 2>/dev/null; true"
  "$LANE" stop 2>&1 | tail -1
  "$LANE" reclaim 2>&1 | tail -2 | cut -c1-200
  for h in $hosts; do echo "$h MemAvailable $(free_gib "$h") GiB"; done
  rm -rf "$work"
}
trap down EXIT

wait_ready() {
  local id=$1 n=0 ready=0 dead=0 h r
  while [ $n -lt 90 ]; do
    ready=0; dead=0
    for h in $hosts; do
      r=$(on "$h" "$(root_of "$h")/lane.sh ready $id" 2>/dev/null | tail -1)
      [ "$r" = READY ] && ready=$((ready + 1))
      [ "$r" = DEAD ] && dead=$((dead + 1))
    done
    [ $ready -eq "$ranks" ] && break
    [ $dead -ne 0 ] && break
    sleep 10; n=$((n + 1))
  done
  stamp "lane $id ready=$ready of $ranks dead=$dead after $((n * 10)) s"
  [ $ready -eq "$ranks" ] && return 0
  for h in $hosts; do on "$h" "grep -a -E 'budget|refused|ERRSITE|error|failed' $(root_of "$h")/logs/residentd-$id.log | tail -3" 2>/dev/null | sed "s/^/$h: /" | cut -c1-260; done
  return 1
}

api_start() {
  local keep=$1
  hub "A=\$HOME/$unit; systemctl --user stop $unit 2>/dev/null; systemctl --user reset-failed $unit 2>/dev/null; [ $keep = 1 ] || rm -f \$A/runtime/prefix_index.spi \$A/runtime/api_submission.seq; cd \$A && systemd-run --user --quiet --unit=$unit --working-directory=\$A bash -c \"exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$A/runtime --port $SLOT_PORT >> $dir/api.log 2>&1\"; n=0; until curl -s -m 3 localhost:$SLOT_PORT/health | grep -q '\"connected_ranks\":$ranks' || [ \$n -ge 40 ]; do sleep 3; n=\$((n+1)); done; curl -s -m 3 localhost:$SLOT_PORT/health | grep -q '\"connected_ranks\":$ranks'"
}

remote() {
  local mark=$1 command=$2 rc last=""
  stamp "slot $run $mark"
  hub "cd $dir && rm -f $mark.rc && (setsid nohup sh -c '$command >> run.log 2>&1; echo \$? > $mark.rc' > /dev/null 2>&1 &)" || return 1
  while true; do
    sleep 10
    rc=$(hub "test -s $dir/$mark.rc && echo rc=\$(cat $dir/$mark.rc); tail -n 1 $dir/run.log 2>/dev/null") || continue
    case "$rc" in rc=0*) return 0;; rc=*) echo "$rc" | tail -n 1 | cut -c1-240; return 1;; esac
    [ "$rc" = "$last" ] || echo "  $rc" | cut -c1-240
    last=$rc
  done
}

stamp "slot $run start"
for h in $hosts; do on "$h" "r=$(root_of "$h"); [ -x \$r/lane.sh ] && [ -s \$r/packs/pack.sha256 ] && \$r/lane.sh reclaim > /dev/null 2>&1; true"; done
"$LANE" setup > setup.out 2>&1 || { echo "setup retry: $(tail -1 setup.out)"; sleep 5; "$LANE" setup > setup.out 2>&1; } || { echo "SLOT-FAILED setup: $(tail -3 setup.out)"; exit 1; }
STAGE_RUN_ID=$run "$LANE" start > start.out 2>&1
wait_ready "$run" || { echo "SLOT-FAILED: the lane did not come up"; exit 1; }
STAGE_API_FIRMWARE=$SLOT_API_BUILD "$LANE" api-setup 2>&1 | tail -1 | cut -c1-160
hub "mkdir -p $dir && rm -rf ~/$unit && cp -a ~/$SLOT_MODEL-kv-api ~/$unit && sed -i 's|/$SLOT_MODEL-kv-api|/$unit|g' ~/$unit/model_resident.json && grep -q '$unit' ~/$unit/model_resident.json" || { echo "SLOT-FAILED: the API folder was not prepared"; exit 1; }
copy "$KIT/../tools/turn_floor.py" "hub:$SLOT_DIR/" > /dev/null || { echo "SLOT-FAILED: the hub did not take the tool"; exit 1; }
api_start 0 || { echo "SLOT-FAILED: the API did not connect to $ranks ranks"; exit 1; }
remote turns "$tf --arms turns" || { echo "SLOT-FAILED turns"; exit 1; }
remote truth-turns "$tf --arms truth --event after-the-turns" || { echo "SLOT-FAILED truth"; exit 1; }
remote wave "$tf --arms wave" || { echo "SLOT-FAILED wave"; exit 1; }
stamp "slot $run restart"
hub "systemctl --user stop $unit 2>/dev/null; systemctl --user reset-failed $unit 2>/dev/null; true"
"$LANE" stop 2>&1 | tail -1
STAGE_RUN_ID=${run}r "$LANE" start > restart.out 2>&1
wait_ready "${run}r" || { echo "SLOT-FAILED: the lane did not come back"; exit 1; }
api_start 1 || { echo "SLOT-FAILED: the API did not connect to $ranks ranks after the restart"; exit 1; }
remote truth-before "$tf --arms truth --event before-the-resume" || { echo "SLOT-FAILED truth"; exit 1; }
remote resume "$tf --arms resume --event restart" || { echo "SLOT-FAILED resume"; exit 1; }
remote truth-after "$tf --arms truth --event after-the-resume" || { echo "SLOT-FAILED truth"; exit 1; }
remote parts "python3 ../turn_floor.py parts ." || { echo "SLOT-FAILED parts"; exit 1; }
remote excerpt "python3 ../turn_floor.py excerpt . --output engine.log" || { echo "SLOT-FAILED excerpt"; exit 1; }
copy "hub:$dir/requests.jsonl" "hub:$dir/parts.jsonl" "hub:$dir/summary.json" "hub:$dir/engine.log" "hub:$dir/run.log" "$SLOT_OUT/" > /dev/null &&
  echo "SLOT-DONE $run: receipts in $SLOT_OUT" || echo "SLOT-DONE $run: the receipts did not copy; they are on the hub in $dir"
