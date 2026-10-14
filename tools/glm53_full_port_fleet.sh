#!/usr/bin/env bash
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
HUB=rtx5090
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
HEX=0123456789abcdef
LABEL="${1:?label}"
PHASES="${2:-smoke,capture}"
OUTDIR="${3:?local output directory}"
: "${STAGE_MODEL:?source the lane env}"
API_PORT=8475
API_DIR="${STAGE_MODEL}-kv-api"
UNIT_API="kv-${STAGE_MODEL//_/-}-api"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
WORK='~/glmfull-port'
TOKENIZER=/mnt/cold-raid6/models/glm-5.3-fp8/tokenizer.json
mkdir -p "$OUTDIR"
log() { echo "$(date -u +%FT%TZ) $*"; }
root_of() { echo "/home/$1/stage-lanes/$STAGE_MODEL/lane$STAGE_LANE/root"; }

activate() {
  local model="$1" deadline state
  $SSH $HUB "curl -s -X POST localhost:8440/activate -d '{\"model\":\"$model\"}'" > /dev/null || return 1
  deadline=$((SECONDS + 900))
  while [ $SECONDS -lt $deadline ]; do
    state="$($SSH $HUB "curl -s localhost:8440/status" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['state'], d['active'], d['error'])")"
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
  local rank host
  for rank in 0 1 8 15; do
    host="spark${HEX:$rank:1}"
    $SSH "$host" "tail -c 60000 $(root_of "$host")/logs/residentd-$RUN_ID.log" < /dev/null > "$OUTDIR/rank$rank.tail.log" 2>/dev/null
  done
  $SSH $HUB "tail -c 60000 /srv/workspace/sparkpipe/logs/$API_DIR.log" < /dev/null > "$OUTDIR/api.tail.log" 2>/dev/null
}

teardown() {
  $SSH $HUB "systemctl --user stop $UNIT_API 2>/dev/null; systemctl --user reset-failed $UNIT_API 2>/dev/null; true" < /dev/null
  collect
  each_rank stop
  each_rank reclaim
  each_rank drop-kv
  activate qwen3.8-max || { sleep 30; activate qwen3.8-max; }
}

run() {
  $SSH $HUB "cd $WORK/repo && python3 tools/glm53_full_golden.py $*"
}

status=0
log "run $RUN_ID label $LABEL phases $PHASES"
activate qwen3.8-flash || { activate qwen3.8-max; exit 1; }
if ! bash "$HERE/stage_lane.sh" setup > "$OUTDIR/setup.log" 2>&1 || ! bash "$HERE/stage_lane.sh" api-setup > "$OUTDIR/api_setup.log" 2>&1; then
  log "setup failed"; tail -5 "$OUTDIR/setup.log" "$OUTDIR/api_setup.log"; activate qwen3.8-max; exit 1
fi
STAGE_RUN_ID="$RUN_ID" bash "$HERE/stage_lane.sh" start > "$OUTDIR/start.log" 2>&1 || status=1
if [ $status = 0 ] && wait_ready; then
  $SSH $HUB "systemctl --user reset-failed $UNIT_API 2>/dev/null; rm -f ~/$API_DIR/runtime/prefix_index.spi; cd ~/$API_DIR && systemd-run --user --unit=$UNIT_API -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$API_DIR bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$API_DIR/runtime --port $API_PORT >> api.log 2>&1' > /dev/null" < /dev/null
  for attempt in $(seq 1 60); do
    $SSH $HUB "curl -sf localhost:$API_PORT/health" > /dev/null 2>&1 && break
    sleep 5
  done
  $SSH $HUB "mkdir -p $WORK/golden-$LABEL" < /dev/null
  case ",$PHASES," in *,smoke,*)
    run capture --prompts ../golden_prompts.json --names capital --port $API_PORT --max-tokens 16 --logprobs 4 --label "$LABEL-smoke" --out "$WORK/golden-$LABEL/smoke.json" || status=1 ;;
  esac
  case ",$PHASES," in *,capture,*)
    if [ $status = 0 ]; then
      run capture --prompts ../golden_prompts.json --port $API_PORT --max-tokens 128 --logprobs 4 --label "$LABEL" --out "$WORK/golden-$LABEL/lp.json" || status=1
      run batch --prompts ../golden_prompts.json --names capital,arith,code,story --port $API_PORT --max-tokens 64 --label "$LABEL" --out "$WORK/golden-$LABEL/batch.json" || status=1
    fi ;;
  esac
  case ",$PHASES," in *,compsec,*)
    if [ $status = 0 ]; then
      run compsec --prompts qualification/glm53full/compsec17_prompts_off.json --port $API_PORT --max-tokens 512 --concurrency 2 --tokenizer "$TOKENIZER" --label "$LABEL" --out "$WORK/golden-$LABEL/compsec17_off.json" || status=1
    fi ;;
  esac
  scp -q "$HUB:glmfull-port/golden-$LABEL/*.json" "$OUTDIR/" 2>/dev/null
else
  status=1
fi
teardown
log "done status=$status"
exit $status
