#!/usr/bin/env bash
set -euo pipefail
: "${STAGE_MODEL:?STAGE_MODEL names a tools/stage_lane.py MODELS entry}"
: "${STAGE_LANE:?STAGE_LANE is the weightd mesh lane (1..15)}"
: "${STAGE_FIRMWARE:?STAGE_FIRMWARE holds sparkpipe_model_residentd, the model adapter library, libhidden_transport_spark_host_rdma_verbs.so and weightd_warm}"
: "${STAGE_WEIGHTD_SOCKET:?STAGE_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${STAGE_EXPERT_POOL_BYTES:?STAGE_EXPERT_POOL_BYTES is the per-rank routed expert pool}"
: "${STAGE_MEMORY_MAX:?STAGE_MEMORY_MAX is the residentd unit MemoryMax, e.g. 16G}"
HERE="$(cd "$(dirname "$0")" && pwd)"
LANE_TOOL=stage_lane
LANE_RANKS=16
. "$HERE/lane_run_log.sh"
HEX=0123456789abcdef
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-${STAGE_MODEL//_/-}-rd$STAGE_LANE"
model_field() { python3 -c "import sys; sys.path.insert(0, '$HERE'); import stage_lane; print(stage_lane.MODELS['$STAGE_MODEL']['$1'])"; }
ADAPTER_FILE="$(model_field adapter)"
PACK_TEMPLATE="$(model_field pack)"
host_of() { echo "spark${HEX:$1:1}"; }
root_of() { echo "/home/$1/stage-lanes/$STAGE_MODEL/lane$STAGE_LANE/root"; }
pack_of() { echo "/home/$(host_of "$1")/$(python3 -c "print('$PACK_TEMPLATE'.format(rank=$1))")"; }
PIDS=()

join_ranks() {
  local label="$1" rank failed=""
  for rank in $(seq 0 15); do
    wait "${PIDS[$rank]}" || failed="$failed $(host_of "$rank")"
  done
  PIDS=()
  [ -z "$failed" ] || { echo "stage_lane $label failed on:$failed" >&2; return 1; }
}

render() {
  python3 "$HERE/stage_lane.py" --model "$STAGE_MODEL" --lane "$STAGE_LANE" \
    --sequences "${STAGE_SEQUENCES:-8}" --rows "${STAGE_ROWS:-1024}" --kv-pages "${STAGE_KV_PAGES:-64}" \
    ${STAGE_KV_PHYSICAL_PAGES:+--kv-physical-pages "$STAGE_KV_PHYSICAL_PAGES"} \
    ${STAGE_KV_LOGICAL_PAGES:+--kv-logical-pages "$STAGE_KV_LOGICAL_PAGES"} \
    ${STAGE_KV_BACKING_BYTES:+--kv-backing-bytes "$STAGE_KV_BACKING_BYTES"} \
    ${STAGE_STATE_BUDGET_BYTES:+--state-budget-bytes "$STAGE_STATE_BUDGET_BYTES"} \
    ${STAGE_TOKENIZER_SHA256:+--tokenizer-sha256 "$STAGE_TOKENIZER_SHA256"} \
    --runtime-root "/home/{host}/stage-lanes/$STAGE_MODEL/lane$STAGE_LANE/root" \
    --weightd-socket "$STAGE_WEIGHTD_SOCKET" --output-dir "$1"
}

setup() {
  local generated rank host root pack
  generated="$(mktemp -d)"
  render "$generated" > /dev/null
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    pack="$(pack_of "$rank")"
    (
      $SSH "$host" "test -f $pack -a -f $pack.experts -a -f $pack.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache $root/kvsnapshot && find $root/packs $root/kvcache $root/kvsnapshot -mindepth 1 -delete"
      scp -q "$STAGE_FIRMWARE/sparkpipe_model_residentd" "$STAGE_FIRMWARE/weightd_warm" "$HERE/weightd_spine_budget.py" "$host:$root/bin/"
      scp -q "$STAGE_FIRMWARE/$ADAPTER_FILE" "$host:$root/lib/$ADAPTER_FILE"
      scp -q "$STAGE_FIRMWARE/libhidden_transport_spark_host_rdma_verbs.so" "$host:$root/lib/hidden_transport.so"
      scp -q "$generated/deployment.json" "$host:$root/"
      scp -q "$generated/adapter.$host.json" "$host:$root/config/adapter.json"
      $SSH "$host" "ln -sfn $pack $root/packs/ && ln -sfn $pack.experts $root/packs/ && head -c 64 $pack.sha256 > $root/packs/pack.sha256 && python3 $root/bin/weightd_spine_budget.py $pack > $root/spine_budget"
      echo "$host ready spine_budget=$($SSH "$host" cat "$root/spine_budget")"
    ) &
    PIDS[$rank]=$!
  done
  join_ranks setup || { rm -rf "$generated"; exit 1; }
  rm -rf "$generated"
}

start() {
  local rank host root run_id
  run_id="$(lane_run_id "${STAGE_RUN_ID:-}")" || exit 2
  echo "stage_lane: $STAGE_MODEL run $run_id; rank logs are $(lane_log_file "$run_id") under each lane root"
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    $SSH "$host" "cd $root && $(lane_log_prelude "$host" "$run_id") && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=$STAGE_MEMORY_MAX -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=$STAGE_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$STAGE_LANE -E SPARK_TP_MESH_RANKS=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=$STAGE_EXPERT_POOL_BYTES -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=\$(cat spine_budget) -E CUDA_MODULE_LOADING=LAZY -E CUDA_DEVICE_MAX_CONNECTIONS=32 bash -c 'exec ./bin/sparkpipe_model_residentd --deployment deployment.json --rank-index $rank > $(lane_log_file "$run_id") 2>&1'" &
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

case "${1:-}" in
  render) render "${2:?render OUTPUT_DIRECTORY}" ;;
  setup) setup ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  *) echo "usage: $0 render DIR|setup|start|status|stop" >&2; exit 2 ;;
esac
