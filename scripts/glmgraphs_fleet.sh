#!/usr/bin/env bash
set -u
REPO=/Users/mac/wf/drv-glmgraphs
HERE=$REPO/tools
HUB=rtx5090
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
HEX=0123456789abcdef
OUTDIR="${1:?local output directory}"
GRAPH_SHA="${2:?graph build sha}"
CYCLES="${3:-graph:smoke,capture,speed eager:speed}"
source /Users/mac/sparkpipe-coord/lanes/driver-sprint/env/glmfull_env.sh
FW_eager=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-2aa246576
API_eager=/home/spec/api-build-2aa246576/build
FW_graph=/Users/mac/sparkpipe-coord/lanes/driver-sprint/fw-glmgraphs-$GRAPH_SHA
API_graph=/home/spec/api-build-glmgraphs-$GRAPH_SHA/build
API_PORT=8475
API_DIR="${STAGE_MODEL}-kv-api"
UNIT_API="kv-${STAGE_MODEL//_/-}-api"
WORK='~/glmgraphs-port'
mkdir -p "$OUTDIR"
log() { echo "$(date -u +%FT%TZ) $*"; }
root_of() { echo "/home/$1/stage-lanes/$STAGE_MODEL/lane$STAGE_LANE/root"; }

activate() {
  local model="$1" deadline state
  $SSH $HUB "curl -s -X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" < /dev/null > /dev/null || return 1
  deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    state="$($SSH $HUB "curl -s localhost:8440/status" < /dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])")"
    case "$state" in
      "idle $model None") log "activated $model"; return 0 ;;
      idle*|failed*) log "activate $model failed: $state"; return 1 ;;
    esac
    sleep 5
  done
  log "activate $model timed out"
  return 1
}

each_rank() {
  local rank host pids=() failed=""
  for rank in $(seq 0 15); do
    host="spark${HEX:$rank:1}"
    $SSH "$host" "$(root_of "$host")/lane.sh $*" < /dev/null > /dev/null 2>&1 &
    pids[$rank]=$!
  done
  for rank in $(seq 0 15); do
    wait "${pids[$rank]}" || failed="$failed spark${HEX:$rank:1}"
  done
  [ -z "$failed" ] || { log "lane.sh $* failed on:$failed"; return 1; }
}

wait_ready() {
  local deadline=$((SECONDS + 1500)) rank host state ready dead
  while [ $SECONDS -lt $deadline ]; do
    ready=0; dead=""
    for rank in $(seq 0 15); do
      host="spark${HEX:$rank:1}"
      state="$($SSH "$host" "$(root_of "$host")/lane.sh ready $RUN_ID" < /dev/null 2>/dev/null)"
      [ "$state" = READY ] && ready=$((ready + 1))
      [ "$state" = DEAD ] && dead="$dead $host"
    done
    [ -n "$dead" ] && { log "ranks dead:$dead"; return 1; }
    [ $ready = 16 ] && { log "16/16 ready"; return 0; }
    sleep 10
  done
  log "lane not ready in time"
  return 1
}

collect() {
  local out="$1" rank host log_path
  for rank in 0 1 8 15; do
    host="spark${HEX:$rank:1}"
    log_path="$(root_of "$host")/logs/residentd-$RUN_ID.log"
    $SSH "$host" "grep -a -E 'STAGE-GRAPH|STAGE-STEP-TIMING|STAGE-RESIDENT|chain refused|step refused|KV access error|collective failed|copy failed|glm53_full_tp16_stage rank' $log_path | tail -400" < /dev/null > "$out/rank$rank.events.log" 2>/dev/null
    $SSH "$host" "tail -c 40000 $log_path" < /dev/null > "$out/rank$rank.tail.log" 2>/dev/null
  done
  $SSH $HUB "tail -c 40000 ~/$API_DIR/api.log" < /dev/null > "$out/api.tail.log" 2>/dev/null
}

run() {
  $SSH $HUB "cd $WORK/repo && python3 tools/glm53_full_golden.py $*" < /dev/null
}

cycle() {
  local kind="$1" phases="$2" out="$OUTDIR/$1" status=0 fw api
  eval fw="\$FW_$kind"; eval api="\$API_$kind"
  mkdir -p "$out"
  RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
  log "cycle $kind run $RUN_ID firmware $fw api $api phases $phases"
  export STAGE_FIRMWARE="$fw" STAGE_API_FIRMWARE="$api"
  if ! bash "$HERE/stage_lane.sh" setup > "$out/setup.log" 2>&1 || ! bash "$HERE/stage_lane.sh" api-setup > "$out/api_setup.log" 2>&1; then
    log "setup failed"; tail -5 "$out/setup.log" "$out/api_setup.log"; return 1
  fi
  STAGE_RUN_ID="$RUN_ID" bash "$HERE/stage_lane.sh" start > "$out/start.log" 2>&1 || status=1
  if [ $status = 0 ] && wait_ready; then
    $SSH $HUB "systemctl --user reset-failed $UNIT_API 2>/dev/null; rm -f ~/$API_DIR/runtime/prefix_index.spi; cd ~/$API_DIR && systemd-run --user --unit=$UNIT_API -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$API_DIR bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$API_DIR/runtime --port $API_PORT >> api.log 2>&1' > /dev/null" < /dev/null
    for attempt in $(seq 1 60); do
      $SSH $HUB "curl -sf localhost:$API_PORT/health" < /dev/null > /dev/null 2>&1 && break
      sleep 5
    done
    $SSH $HUB "mkdir -p $WORK/out-$RUN_ID-$kind" < /dev/null
    case ",$phases," in *,smoke,*)
      run capture --prompts ../golden_prompts.json --names capital --port $API_PORT --max-tokens 16 --logprobs 4 --label "$kind-smoke" --out "$WORK/out-$RUN_ID-$kind/smoke.json" || status=1 ;;
    esac
    case ",$phases," in *,capture,*)
      if [ $status = 0 ]; then
        run capture --prompts ../golden_prompts.json --port $API_PORT --max-tokens 128 --logprobs 4 --label "$kind-$RUN_ID" --out "$WORK/out-$RUN_ID-$kind/lp.json" || status=1
        run batch --prompts ../golden_prompts.json --names capital,arith,code,story --port $API_PORT --max-tokens 64 --label "$kind-$RUN_ID" --out "$WORK/out-$RUN_ID-$kind/batch.json" || status=1
      fi ;;
    esac
    case ",$phases," in *,speed,*)
      if [ $status = 0 ]; then
        run speed --prompts ../golden_prompts.json --names long_2k --port $API_PORT --max-tokens 256 --batches 1,4 --label "$kind-$RUN_ID" --out "$WORK/out-$RUN_ID-$kind/speed.json" || status=1
      fi ;;
    esac
    scp -q "$HUB:glmgraphs-port/out-$RUN_ID-$kind/*.json" "$out/" 2>/dev/null
  else
    status=1
  fi
  $SSH $HUB "systemctl --user stop $UNIT_API 2>/dev/null; systemctl --user reset-failed $UNIT_API 2>/dev/null; true" < /dev/null
  collect "$out"
  each_rank stop
  each_rank reclaim
  each_rank drop-kv
  log "cycle $kind done status=$status"
  return $status
}

overall=0
state="$($SSH $HUB "curl -s localhost:8440/status" < /dev/null | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])")"
log "controller before: $state"
activate qwen3.8-flash || { activate qwen3.8-max; exit 1; }
for entry in $CYCLES; do
  cycle "${entry%%:*}" "${entry#*:}" || overall=1
done
activate qwen3.8-max || { sleep 30; activate qwen3.8-max; } || overall=1
log "done status=$overall"
exit $overall
