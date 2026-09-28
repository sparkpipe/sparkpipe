#!/usr/bin/env bash
set -euo pipefail
: "${MIMO_HOSTS:?MIMO_HOSTS is four comma-separated hosts in TP rank order}"
: "${MIMO_MESH:?MIMO_MESH is the four mesh ranks of those hosts}"
: "${MIMO_LANE:?MIMO_LANE is the weightd lane id}"
: "${MIMO_FIRMWARE:?MIMO_FIRMWARE is the local directory holding sparkpipe_model_residentd, model_driver.so, model_serving_adapter.so and hidden_transport.so}"
: "${MIMO_WEIGHTD_SOCKET:?MIMO_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${MIMO_PACKS:?MIMO_PACKS is the per-node directory holding rankN.sp, .experts and .sha256}"
MIMO_LANES="${MIMO_LANES:?MIMO_LANES is the resident sequence count}"
MIMO_POSITIONS="${MIMO_POSITIONS:?MIMO_POSITIONS is the per-sequence position capacity}"
MIMO_GRAPH="${MIMO_GRAPH:?MIMO_GRAPH is 1 for the graph-replayed step, 0 for eager}"
HERE="$(cd "$(dirname "$0")" && pwd)"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-mimo-rd$MIMO_LANE"
IFS=, read -r -a HOSTS <<< "$MIMO_HOSTS"
root_of() { echo "/home/$1/mimo-lane$MIMO_LANE/root"; }

render() {
  python3 "$HERE/mimo26_lane.py" --hosts "$MIMO_HOSTS" --lane "$MIMO_LANE" --socket "$MIMO_WEIGHTD_SOCKET" --lanes "$MIMO_LANES" --max-sequence-positions "$MIMO_POSITIONS" --output "$1"
}

setup() {
  local generated rank host root
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  for rank in 0 1 2 3; do
    host="${HOSTS[$rank]}"
    root="$(root_of "$host")"
    $SSH "$host" "test -f $MIMO_PACKS/rank$rank.sp -a -f $MIMO_PACKS/rank$rank.sp.experts -a -f $MIMO_PACKS/rank$rank.sp.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache && find $root/packs -mindepth 1 -delete"
    scp -q "$MIMO_FIRMWARE/sparkpipe_model_residentd" "$host:$root/bin/"
    scp -q "$MIMO_FIRMWARE/model_driver.so" "$MIMO_FIRMWARE/model_serving_adapter.so" "$MIMO_FIRMWARE/hidden_transport.so" "$host:$root/lib/"
    scp -q "$generated/model_resident.json" "$host:$root/"
    scp -q "$generated/config/stage_0$rank.json" "$host:$root/config/"
    $SSH "$host" "cd $root/packs && ln -sfn $MIMO_PACKS/rank$rank.sp && ln -sfn $MIMO_PACKS/rank$rank.sp.experts && head -c 64 $MIMO_PACKS/rank$rank.sp.sha256 > pack.sha256"
    echo "$host ready rank=$rank"
  done
  rm -rf "$generated"
}

start() {
  local rank host root
  for rank in 0 1 2 3; do
    host="${HOSTS[$rank]}"
    root="$(root_of "$host")"
    $SSH "$host" "cd $root && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=16G -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=$MIMO_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$MIMO_LANE -E SPARK_TP_MESH_RANKS=$MIMO_MESH -E SPARK_TP_WAIT_MODE=hardware -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=42000000000 -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=2600000000 -E SPARK_MIMO26_TP_RANK=$rank -E SPARK_MIMO26_STAGE_MAX_POSITIONS=$MIMO_POSITIONS -E SPARK_MIMO26_STAGE_GRAPH=$MIMO_GRAPH bash -c 'exec ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $rank > residentd.log 2>&1'" &
  done
  wait
}

status() {
  local host
  for host in "${HOSTS[@]}"; do
    printf '%s %s\n' "$host" "$($SSH "$host" "systemctl --user show $UNIT -p ActiveState --value; tail -1 $(root_of "$host")/residentd.log 2>/dev/null | cut -c1-200" | tr '\n' ' ')"
  done
}

stop() {
  local host
  for host in "${HOSTS[@]}"; do
    $SSH "$host" "systemctl --user stop $UNIT 2>/dev/null; systemctl --user reset-failed $UNIT 2>/dev/null; true" &
  done
  wait
}

api() {
  : "${MIMO_API_HOST:?MIMO_API_HOST is the API host (rtx5090)}"
  : "${MIMO_API_PORT:?MIMO_API_PORT is the API listen port}"
  : "${MIMO_API_BUILD:?MIMO_API_BUILD is the API host directory holding sparkpipe_model_api and model_serving_adapter.so built for that host}"
  : "${MIMO_API_UNIT:?MIMO_API_UNIT is the API systemd user unit name}"
  : "${MIMO_API_TOKENIZER:?MIMO_API_TOKENIZER is the MiMo tokenizer.json on the API host}"
  local generated root
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  root="mimo-lane$MIMO_LANE-api"
  $SSH "$MIMO_API_HOST" "mkdir -p $root/bin $root/runtime/lib $root/runtime/tokenizer && cp $MIMO_API_BUILD/sparkpipe_model_api $root/bin/ && cp $MIMO_API_BUILD/model_serving_adapter.so $root/runtime/lib/ && cp $MIMO_API_TOKENIZER $root/runtime/tokenizer/tokenizer.json"
  scp -q "$generated/model_resident.json" "$MIMO_API_HOST:$root/"
  rm -rf "$generated"
  $SSH "$MIMO_API_HOST" "cd $root && systemctl --user stop $MIMO_API_UNIT 2>/dev/null; systemctl --user reset-failed $MIMO_API_UNIT 2>/dev/null; systemd-run --user --unit=$MIMO_API_UNIT -p MemoryMax=2G -p MemorySwapMax=0 --working-directory=\$HOME/$root bash -c 'exec ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root \$HOME/$root/runtime --port $MIMO_API_PORT >> api.log 2>&1'"
}

case "${1:-}" in
  render) render "${2:?render OUTPUT_DIRECTORY}" ;;
  setup) setup ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  api) api ;;
  *) echo "usage: $0 render DIR|setup|start|status|stop|api" >&2; exit 2 ;;
esac
