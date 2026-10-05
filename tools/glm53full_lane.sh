#!/usr/bin/env bash
set -euo pipefail
: "${GLMFULL_LANE:?GLMFULL_LANE is the weightd lane id (0..15)}"
: "${GLMFULL_CODEC:?GLMFULL_CODEC is fp8 or bf16}"
: "${GLMFULL_FIRMWARE:?GLMFULL_FIRMWARE is the local directory holding sparkpipe_model_residentd, sparkpipe_model_api, model_driver.so, model_serving_adapter.so and hidden_transport.so}"
: "${GLMFULL_WEIGHTD_SOCKET:?GLMFULL_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${GLMFULL_EXPERT_POOL_BYTES:?GLMFULL_EXPERT_POOL_BYTES is the per-node routed expert pool}"
: "${GLMFULL_SPINE_BUDGET_BYTES:?GLMFULL_SPINE_BUDGET_BYTES is the per-node non-expert spine budget}"
: "${GLMFULL_MEMORY_MAX:?GLMFULL_MEMORY_MAX is the residentd unit MemoryMax, e.g. 16G}"
: "${GLMFULL_SEQUENCES:?GLMFULL_SEQUENCES is the resident sequence capacity (1..16)}"
: "${GLMFULL_ROWS:?GLMFULL_ROWS is the execution row capacity, equal to the firmware bucket}"
: "${GLMFULL_POSITIONS:?GLMFULL_POSITIONS is max_sequence_positions}"
: "${GLMFULL_INFLIGHT:?GLMFULL_INFLIGHT is max_inflight_submissions (1..4)}"
GLMFULL_ARM="${GLMFULL_ARM:-$GLMFULL_CODEC}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LANE_TOOL=glm53full_lane
LANE_RANKS=16
. "$HERE/lane_run_log.sh"
HEX=0123456789abcdef
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-glmfull-rd$GLMFULL_LANE"
API_HOST=rtx5090
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
host_of() { echo "spark${HEX:$1:1}"; }
root_of() { echo "/home/$1/glmfull-lane$GLMFULL_LANE/root"; }
pack_of() { echo "/home/$1/sparkdata/glm53full.$GLMFULL_ARM.tp16/packs/glm53full.$GLMFULL_ARM.tp16-rank$2.glm52sp"; }
PIDS=()

api_host_only() {
  if [ -n "${GLMFULL_API_HOST:-}" ] && [ "$GLMFULL_API_HOST" != "$API_HOST" ]; then
    echo "glm53full_lane: refused: model APIs run only on $API_HOST, never on a Spark; GLMFULL_API_HOST=$GLMFULL_API_HOST" >&2
    exit 2
  fi
}

join_ranks() {
  local label="$1" rank failed=""
  for rank in $(seq 0 15); do
    if ! wait "${PIDS[$rank]}"; then
      failed="$failed $(host_of "$rank")"
    fi
  done
  PIDS=()
  if [ -n "$failed" ]; then
    echo "glm53full_lane $label failed on:$failed" >&2
    return 1
  fi
}

render() {
  python3 "$HERE/glm53full_lane.py" --lane "$GLMFULL_LANE" --codec "$GLMFULL_CODEC" --arm "$GLMFULL_ARM" --socket "$GLMFULL_WEIGHTD_SOCKET" \
    --kv-backing-bytes 4294967296 --max-sequence-positions "$GLMFULL_POSITIONS" --execution-row-capacity "$GLMFULL_ROWS" \
    --sequences "$GLMFULL_SEQUENCES" --inflight "$GLMFULL_INFLIGHT" --output "$1"
}

setup() {
  local generated rank host root pack
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    pack="$(pack_of "$host" "$rank")"
    (
      $SSH "$host" "test -f $pack -a -f $pack.experts -a -f $pack.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache && find $root/packs -mindepth 1 -delete && (grep -q '^/home/$host/glmfull-lane$GLMFULL_LANE	' ~/KEEP || printf '/home/$host/glmfull-lane$GLMFULL_LANE\tglmfull\tKEEP\tGLM-5.3 Full lane $GLMFULL_LANE runtime root (binaries, configs, pack symlinks, kvcache); DELETE-OK when the lane stops\n' >> ~/KEEP)"
      scp -q "$GLMFULL_FIRMWARE/sparkpipe_model_residentd" "$host:$root/bin/"
      scp -q "$GLMFULL_FIRMWARE/model_driver.so" "$GLMFULL_FIRMWARE/model_serving_adapter.so" "$GLMFULL_FIRMWARE/hidden_transport.so" "$host:$root/lib/"
      scp -q "$generated/model_resident.json" "$host:$root/"
      scp -q "$generated/config/stage_$(printf %02d "$rank").json" "$host:$root/config/"
      $SSH "$host" "ln -sfn $pack $root/packs/ && ln -sfn $pack.experts $root/packs/ && head -c 64 $pack.sha256 > $root/packs/pack.sha256"
      echo "$host ready"
    ) &
    PIDS[$rank]=$!
  done
  if ! join_ranks setup; then
    rm -rf "$generated"
    exit 1
  fi
  rm -rf "$generated"
}

start() {
  local rank host root run_id
  run_id="$(lane_run_id "${GLMFULL_RUN_ID:-}")" || exit 2
  echo "glm53full_lane: run $run_id; rank logs are logs/residentd-$run_id.log under each lane root"
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    $SSH "$host" "cd $root && $(lane_log_prelude "$host" "$run_id") && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=$GLMFULL_MEMORY_MAX -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_GLM52_SERVING_FLAT_RANKS=16 -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_ATTACH_LAZY=1 -E SPARK_WEIGHTD_SOCKET=$GLMFULL_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$GLMFULL_LANE -E SPARK_TP_MESH_RANKS=$MESH_RANKS -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=$GLMFULL_EXPERT_POOL_BYTES -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=$GLMFULL_SPINE_BUDGET_BYTES ${GLMFULL_EXTRA_ENV:-} bash -c 'SPARK_WEIGHTD_PACK_SHA256=\$(cat packs/pack.sha256) exec ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $rank > logs/residentd-$run_id.log 2>&1'" &
    PIDS[$rank]=$!
  done
  join_ranks start
}

status() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    printf '%s %s\n' "$host" "$($SSH "$host" "systemctl --user show $UNIT -p ActiveState --value; systemctl --user show $UNIT -p MemoryCurrent --value; awk '/MemAvailable/{print int(\$2/1048576)\"G\"}' /proc/meminfo; tail -1 $(root_of "$host")/residentd.log 2>/dev/null | cut -c1-200" | tr '\n' ' ')"
  done
}

stop() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    $SSH "$host" "systemctl --user stop $UNIT 2>/dev/null; systemctl --user reset-failed $UNIT 2>/dev/null; true" &
    PIDS[$rank]=$!
  done
  join_ranks stop
}

api() {
  api_host_only
  : "${GLMFULL_API_PORT:?GLMFULL_API_PORT is the API listen port}"
  : "${GLMFULL_API_BUILD:?GLMFULL_API_BUILD is the rtx5090 directory holding the x86 sparkpipe_model_api and model_serving_adapter.so}"
  : "${GLMFULL_API_TOKENIZER:?GLMFULL_API_TOKENIZER is the GLM tokenizer.json on the rtx5090}"
  local generated root unit
  unit="glmfull-api$GLMFULL_LANE"
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  root="glmfull-lane$GLMFULL_LANE-api"
  $SSH "$API_HOST" "mkdir -p $root/bin $root/runtime/lib $root/runtime/tokenizer && cp $GLMFULL_API_BUILD/sparkpipe_model_api $root/bin/ && cp $GLMFULL_API_BUILD/model_serving_adapter.so $root/runtime/lib/ && cp $GLMFULL_API_TOKENIZER $root/runtime/tokenizer/tokenizer.json && echo '$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokenizer"]["sha256"])' "$generated/model_resident.json")  $root/runtime/tokenizer/tokenizer.json' | sha256sum -c --quiet"
  scp -q "$generated/model_resident.json" "$API_HOST:$root/"
  rm -rf "$generated"
  $SSH "$API_HOST" "cd $root && systemctl --user stop $unit 2>/dev/null; systemctl --user reset-failed $unit 2>/dev/null; systemd-run --user --unit=$unit -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$root -E SPARK_GLM52_SERVING_FLAT_RANKS=16 bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$root/runtime --port $GLMFULL_API_PORT >> api.log 2>&1'"
}

api_stop() {
  api_host_only
  $SSH "$API_HOST" "systemctl --user stop glmfull-api$GLMFULL_LANE 2>/dev/null; systemctl --user reset-failed glmfull-api$GLMFULL_LANE 2>/dev/null; true"
}

decode() {
  api_host_only
  : "${GLMFULL_API_PORT:?GLMFULL_API_PORT is the API listen port}"
  $SSH "$API_HOST" "curl -s --max-time 900 -X POST http://localhost:$GLMFULL_API_PORT/v1/completions -H 'Content-Type: application/json' -d '{\"prompt_token_ids\":[$1],\"max_tokens\":$2}'"
  echo
}

case "${1:-}" in
  render) render "${2:?render OUTPUT_DIRECTORY}" ;;
  setup) setup ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  api) api ;;
  api-stop) api_stop ;;
  decode) decode "${2:?decode TOKEN_IDS_CSV NEW_TOKENS}" "${3:?decode TOKEN_IDS_CSV NEW_TOKENS}" ;;
  archive) lane_archive "${2:?archive RUN_ID DESTINATION}" "${3:?archive RUN_ID DESTINATION}" ;;
  *) echo "usage: $0 render DIR|setup|start|status|stop|api|api-stop|decode IDS NEW|archive RUN_ID DEST" >&2; exit 2 ;;
esac
