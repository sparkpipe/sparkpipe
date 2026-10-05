#!/usr/bin/env bash
set -euo pipefail
: "${LING_LANE:?LING_LANE is the weightd lane id (0..15)}"
: "${LING_CODEC:?LING_CODEC is bf16 or fp8}"
: "${LING_FIRMWARE:?LING_FIRMWARE is the local directory holding sparkpipe_model_residentd, model_driver.so, model_serving_adapter.so and hidden_transport.so}"
: "${LING_EXPERT_POOL_BYTES:?LING_EXPERT_POOL_BYTES is the per-node routed expert pool}"
: "${LING_WEIGHTD_SOCKET:?LING_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${LING_MEMORY_MAX:?LING_MEMORY_MAX is the residentd unit MemoryMax, e.g. 12G}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LANE_TOOL=ling_lane
LANE_RANKS=16
. "$HERE/lane_run_log.sh"
HEX=0123456789abcdef
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-ling-rd$LING_LANE"
MESH_RANKS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"
host_of() { echo "spark${HEX:$1:1}"; }
root_of() { echo "/home/$1/ling-lane$LING_LANE/root"; }
PIDS=()

join_ranks() {
  local label="$1" rank failed=""
  for rank in $(seq 0 15); do
    if ! wait "${PIDS[$rank]}"; then
      failed="$failed $(host_of "$rank")"
    fi
  done
  PIDS=()
  if [ -n "$failed" ]; then
    echo "ling_lane $label failed on:$failed" >&2
    return 1
  fi
}

render() {
  local out="$1"
  python3 "$HERE/ling_lane.py" --lane "$LING_LANE" --codec "$LING_CODEC" --socket "$LING_WEIGHTD_SOCKET" \
    --kv-backing-bytes 8589934592 --max-sequence-positions 32768 --execution-row-capacity 128 --output "$out"
}

setup() {
  local generated rank host root pack budget
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    pack="/home/$host/sparkdata/ling.$LING_CODEC.tp16/packs/ling.$LING_CODEC.tp16.rank${HEX:$rank:1}.sp"
    (
      $SSH "$host" "test -f $pack -a -f $pack.experts -a -f $pack.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache && find $root/packs -mindepth 1 -delete"
      scp -q "$LING_FIRMWARE/sparkpipe_model_residentd" "$host:$root/bin/"
      scp -q "$LING_FIRMWARE/model_driver.so" "$LING_FIRMWARE/model_serving_adapter.so" "$LING_FIRMWARE/hidden_transport.so" "$host:$root/lib/"
      scp -q "$generated/model_resident.json" "$host:$root/"
      scp -q "$generated/config/stage_$(printf %02d "$rank").json" "$host:$root/config/"
      scp -q "$HERE/ling_multidev_lane.py" "$host:$root/"
      $SSH "$host" "ln -sfn $pack $root/packs/ && ln -sfn $pack.experts $root/packs/ && head -c 64 $pack.sha256 > $root/packs/pack.sha256 && python3 $root/ling_multidev_lane.py --spine-budget $pack > $root/spine_budget"
      budget="$($SSH "$host" cat "$root/spine_budget")"
      echo "$host ready spine_budget=$budget"
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
  run_id="$(lane_run_id "${LING_RUN_ID:-}")" || exit 2
  echo "ling_lane: run $run_id; rank logs are $(lane_log_file "$run_id") under each lane root"
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    $SSH "$host" "cd $root && $(lane_log_prelude "$host" "$run_id") && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=$LING_MEMORY_MAX -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=$LING_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$LING_LANE -E SPARK_TP_MESH_RANKS=$MESH_RANKS -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=$LING_EXPERT_POOL_BYTES -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=\$(cat spine_budget) bash -c 'exec ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $rank > $(lane_log_file "$run_id") 2>&1'" &
    PIDS[$rank]=$!
  done
  join_ranks start
}

status() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    printf '%s %s\n' "$host" "$($SSH "$host" "systemctl --user show $UNIT -p ActiveState --value; tail -1 $(root_of "$host")/residentd.log 2>/dev/null | cut -c1-160" | tr '\n' ' ')"
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
  : "${LING_API_HOST:?LING_API_HOST is the API host (rtx5090)}"
  : "${LING_API_PORT:?LING_API_PORT is the API listen port}"
  : "${LING_API_BUILD:?LING_API_BUILD is the API host directory holding sparkpipe_model_api and model_serving_adapter.so built for that host}"
  : "${LING_API_UNIT:?LING_API_UNIT is the API systemd user unit name}"
  : "${LING_API_TOKENIZER:?LING_API_TOKENIZER is the Ling tokenizer.json on the API host}"
  local generated root sha
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["tokenizer"]["sha256"])' "$generated/model_resident.json")"
  root="ling-lane$LING_LANE-api"
  $SSH "$LING_API_HOST" "mkdir -p $root/bin $root/runtime/lib $root/runtime/tokenizer && cp $LING_API_BUILD/sparkpipe_model_api $root/bin/ && cp $LING_API_BUILD/model_serving_adapter.so $root/runtime/lib/ && cp $LING_API_TOKENIZER $root/runtime/tokenizer/tokenizer.json && echo '$sha  $root/runtime/tokenizer/tokenizer.json' | sha256sum -c --quiet"
  scp -q "$generated/model_resident.json" "$LING_API_HOST:$root/"
  rm -rf "$generated"
  $SSH "$LING_API_HOST" "cd $root && systemctl --user stop $LING_API_UNIT 2>/dev/null; systemctl --user reset-failed $LING_API_UNIT 2>/dev/null; systemd-run --user --unit=$LING_API_UNIT -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$root bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$root/runtime --port $LING_API_PORT >> api.log 2>&1'"
}

decode() {
  : "${LING_API_HOST:?LING_API_HOST is the API host (rtx5090)}"
  : "${LING_API_PORT:?LING_API_PORT is the API listen port}"
  $SSH "$LING_API_HOST" "curl -s --max-time 600 -X POST http://localhost:$LING_API_PORT/v1/completions -H 'Content-Type: application/json' -d '{\"prompt_token_ids\":[$1],\"max_tokens\":$2}'"
  echo
}

case "${1:-}" in
  render) render "${2:?render OUTPUT_DIRECTORY}" ;;
  setup) setup ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  api) api ;;
  decode) decode "${2:?decode TOKEN_IDS_CSV NEW_TOKENS}" "${3:?decode TOKEN_IDS_CSV NEW_TOKENS}" ;;
  archive) lane_archive "${2:?archive RUN_ID DESTINATION}" "${3:?archive RUN_ID DESTINATION}" ;;
  *) echo "usage: $0 render DIR|setup|start|status|stop|api|decode IDS NEW|archive RUN_ID DEST" >&2; exit 2 ;;
esac
