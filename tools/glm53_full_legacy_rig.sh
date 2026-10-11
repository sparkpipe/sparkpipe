#!/usr/bin/env bash
set -u
HUB=rtx5090
SSH="ssh -o BatchMode=yes -o ConnectTimeout=8"
HEX=0123456789abcdef
PROD=glmfull-lane6/kvproof
PROD_API=glmfull-kv-api
RIG=stage-lanes/glm52_legacy/lane6/root
RIG_API=glm52-legacy-rig-api
RIG_PORT=8477
UNIT=sp-glm52-legacy-rig
UNIT_API=kv-glm52-legacy-rig-api
log() { echo "$(date -u +%FT%TZ) $*"; }

each() {
  local rank host pids=() failed=""
  local command
  for rank in $(seq 0 15); do
    host="spark${HEX:$rank:1}"
    command="${1//@RANK2@/$(printf %02d "$rank")}"
    command="${command//@RANK@/$rank}"
    $SSH "$host" "$command" < /dev/null > /dev/null 2>&1 &
    pids[$rank]=$!
  done
  for rank in $(seq 0 15); do
    wait "${pids[$rank]}" || failed="$failed spark${HEX:$rank:1}"
  done
  [ -z "$failed" ] || { log "$2 failed on:$failed"; return 1; }
}

setup() {
  each "set -e; P=\$HOME/$PROD; R=\$HOME/$RIG; rm -rf \$R; mkdir -p \$R/bin \$R/lib \$R/config \$R/packs \$R/kvcache \$R/kvsnapshot \$R/logs
cp \$P/bin/sparkpipe_model_residentd \$P/bin/weightd_warm \$R/bin/; cp \$P/lib/*.so \$R/lib/; cp \$P/config/stage_@RANK2@.json \$R/config/; cp \$P/MODULE_IDENTITY \$P/SOURCE_COMMIT \$R/
for f in \$P/packs/*; do case \$f in *.glm52sp|*.experts) ln -s \$(readlink -f \$f) \$R/packs/;; *) cp \$f \$R/packs/;; esac; done
sed 's#/$PROD#/$RIG#g' \$P/model_resident.json > \$R/model_resident.json" "setup" || return 1
  $SSH $HUB "set -e; rm -rf ~/$RIG_API; mkdir -p ~/$RIG_API/runtime; cp -r ~/$PROD_API/bin ~/$RIG_API/; cp -r ~/$PROD_API/runtime/lib ~/$PROD_API/runtime/tokenizer ~/$RIG_API/runtime/; cp ~/$PROD_API/model_resident.json ~/$PROD_API/SOURCE_COMMIT ~/$RIG_API/" < /dev/null
}

clean() {
  each "R=\$HOME/$RIG; find \$R/kvcache \$R/kvsnapshot -mindepth 1 -delete" "clean" || return 1
  $SSH $HUB "rm -f ~/$RIG_API/runtime/prefix_index.spi ~/$RIG_API/runtime/prefix_index.spi.tmp-* ~/$RIG_API/runtime/api_submission.seq" < /dev/null
}

start() {
  local run="$1" chain="$2"
  each "R=\$HOME/$RIG; cd \$R && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=20G -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=\$R -E LD_LIBRARY_PATH=\$R/lib -E SPARK_GLM52_SERVING_FLAT_RANKS=16 -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_ATTACH_LAZY=1 -E SPARK_WEIGHTD_SOCKET=/tmp/spark_weightd.sock -E SPARK_WEIGHTD_LANE=6 -E SPARK_TP_MESH_RANKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=55834574848 -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=8589934592 -E SPARK_GLM52_PIN_EXPERTS=1 -E SPARK_TP_WAIT_MODE=hardware -E SPARK_GLM52_CHAIN_MODE=$chain -E SPARK_GLM52_PROJECTION_SPLIT=1 -E SPARK_GLM52_PREFILL_WAVE_ROWS=1024 bash -c 'SPARK_WEIGHTD_PACK_SHA256=\$(cat packs/pack.sha256) exec ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index @RANK@ > logs/residentd-$run.log 2>&1' > /dev/null" "start"
}

ready() {
  local run="$1" deadline=$((SECONDS + 1200)) rank host count
  while [ $SECONDS -lt $deadline ]; do
    count=0
    for rank in $(seq 0 15); do
      host="spark${HEX:$rank:1}"
      $SSH "$host" "grep -q 'model_residentd ready' \$HOME/$RIG/logs/residentd-$run.log && systemctl --user is-active -q $UNIT" < /dev/null
      case $? in
        0) count=$((count + 1)) ;;
        255) log "rig rank $rank unreachable"; return 1 ;;
      esac
    done
    [ $count = 16 ] && { log "rig 16/16 ready"; return 0; }
    for rank in $(seq 0 15); do
      host="spark${HEX:$rank:1}"
      $SSH "$host" "systemctl --user is-failed -q $UNIT" < /dev/null && { log "rig rank $rank failed"; return 1; }
    done
    sleep 10
  done
  return 1
}

api_start() {
  $SSH $HUB "systemctl --user reset-failed $UNIT_API 2>/dev/null; systemd-run --user --unit=$UNIT_API -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$RIG_API -E SPARK_GLM52_SERVING_FLAT_RANKS=16 bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$RIG_API/runtime --port $RIG_PORT >> api.log 2>&1' > /dev/null" < /dev/null
  for attempt in $(seq 1 60); do
    $SSH $HUB "curl -sf localhost:$RIG_PORT/health" > /dev/null 2>&1 && return 0
    sleep 5
  done
  return 1
}

api_stop() {
  $SSH $HUB "systemctl --user stop $UNIT_API 2>/dev/null; systemctl --user reset-failed $UNIT_API 2>/dev/null; true" < /dev/null
}

stop() {
  each "systemctl --user stop $UNIT 2>/dev/null; systemctl --user reset-failed $UNIT 2>/dev/null; ! systemctl --user is-active -q $UNIT" "stop"
}

reclaim() {
  each "R=\$HOME/$RIG; \$R/bin/weightd_warm /tmp/spark_weightd.sock --reclaim-pack \$(cut -c1-64 \$R/packs/pack.sha256)" "reclaim"
}

drop_kv() {
  each "\$HOME/sparkdata/swap/weightd_warm /tmp/spark_weightd.sock --drop-kv glm52_stage" "drop-kv"
}

case "${1:-}" in
  setup) setup ;;
  clean) clean ;;
  start) start "${2:?run id}" "${3:?chain mode}" ;;
  ready) ready "${2:?run id}" ;;
  api-start) api_start ;;
  api-stop) api_stop ;;
  stop) stop ;;
  reclaim) reclaim ;;
  drop-kv) drop_kv ;;
  *) echo "usage: $0 setup|clean|start RUN CHAIN|ready RUN|api-start|api-stop|stop|reclaim|drop-kv" >&2; exit 2 ;;
esac
