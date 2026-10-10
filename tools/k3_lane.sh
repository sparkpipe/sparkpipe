#!/usr/bin/env bash
set -euo pipefail
: "${K3_LANE:?K3_LANE is the weightd mesh lane (1..15, never 0)}"
: "${K3_FIRMWARE:?K3_FIRMWARE is a local directory holding sparkpipe_model_residentd, libk3_serving_adapter.so (TP4xPP4) or libk3_tp16_serving_adapter.so (TP16), libhidden_transport_spark_host_rdma_verbs.so, weightd_warm and sparkpipe_model_batch}"
: "${K3_WEIGHTD_SOCKET:?K3_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${K3_EXPERT_POOL_BYTES:?K3_EXPERT_POOL_BYTES is the per-rank routed expert pool}"
: "${K3_MEMORY_MAX:?K3_MEMORY_MAX is the residentd unit MemoryMax, e.g. 10G}"
: "${K3_STATE_BUDGET_BYTES:?K3_STATE_BUDGET_BYTES is the per-rank KDA state + windows + MLA KV + scratch budget the runner enforces}"
K3_TOPOLOGY="${K3_TOPOLOGY:-tp4pp4}"
case "$K3_TOPOLOGY" in tp4pp4|tp16) ;; *) echo "k3_lane: K3_TOPOLOGY $K3_TOPOLOGY is not tp4pp4 or tp16" >&2; exit 2 ;; esac
[ "$K3_LANE" -ge 1 ] && [ "$K3_LANE" -le 15 ] || { echo "k3_lane: lane $K3_LANE outside 1..15" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
LANE_TOOL=k3_lane
LANE_RANKS=16
. "$HERE/lane_run_log.sh"
CHECKOUT="$(cd "$HERE/.." && pwd)"
HEX=0123456789abcdef
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-k3-rd$K3_LANE"
ADAPTER_FILE=libk3_serving_adapter.so
[ "$K3_TOPOLOGY" = tp16 ] && ADAPTER_FILE=libk3_tp16_serving_adapter.so
PACK_DIR=sparkdata/k3.mxfp4.$K3_TOPOLOGY/packs
host_of() { echo "spark${HEX:$1:1}"; }
root_of() { echo "/home/$1/k3-lanes/lane$K3_LANE/root"; }
pack_of() {
  if [ "$K3_TOPOLOGY" = tp16 ]; then
    printf '/home/%s/%s/k3.stage0.rank%02d.pack\n' "$(host_of "$1")" "$PACK_DIR" "$1"
  else
    echo "/home/$(host_of "$1")/$PACK_DIR/k3.stage$(($1 / 4)).rank0$(($1 % 4)).pack"
  fi
}
mesh_ranks_of() {
  local stage=$(($1 / 4))
  if [ "$K3_TOPOLOGY" = tp16 ]; then
    echo 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
  else
    echo $((stage * 4)),$((stage * 4 + 1)),$((stage * 4 + 2)),$((stage * 4 + 3))
  fi
}
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
    echo "k3_lane $label failed on:$failed" >&2
    return 1
  fi
}

render() {
  python3 "$HERE/k3_multidev_lane.py" --lane "$K3_LANE" \
    --sequences "${K3_SEQUENCES:-16}" --kv-pages "${K3_KV_PAGES:-64}" \
    ${K3_ROWS:+--rows "$K3_ROWS"} \
    ${K3_KV_PHYSICAL_PAGES:+--kv-physical-pages "$K3_KV_PHYSICAL_PAGES"} \
    ${K3_KV_LOGICAL_PAGES:+--kv-logical-pages "$K3_KV_LOGICAL_PAGES"} \
    --topology "$K3_TOPOLOGY" \
    --pipeline-transport "${K3_PIPELINE_TRANSPORT:-host-rdma}" \
    ${K3_KV_BACKING_BYTES:+--kv-backing-bytes "$K3_KV_BACKING_BYTES"} \
    --kv-snapshot-bytes "${K3_KV_SNAPSHOT_BYTES:-8589934592}" \
    --runtime-root "/home/{host}/k3-lanes/lane$K3_LANE/root" \
    --state-budget-bytes "$K3_STATE_BUDGET_BYTES" \
    --weightd-socket "$K3_WEIGHTD_SOCKET" --output-dir "$1"
}

setup() {
  local generated rank host root pack
  generated="$(mktemp -d)"
  render "$generated" >/dev/null
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    pack="$(pack_of "$rank")"
    (
      $SSH "$host" "test -f $pack -a -f $pack.experts -a -f $pack.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache $root/kvsnapshot && find $root/packs $root/kvcache $root/kvsnapshot -mindepth 1 -delete"
      scp -q "$K3_FIRMWARE/sparkpipe_model_residentd" "$K3_FIRMWARE/weightd_warm" "$HERE/weightd_spine_budget.py" "$host:$root/bin/"
      scp -q "$K3_FIRMWARE/$ADAPTER_FILE" "$host:$root/lib/libk3_serving_adapter.so"
      scp -q "$K3_FIRMWARE/libhidden_transport_spark_host_rdma_verbs.so" "$host:$root/lib/hidden_transport.so"
      if [ "${K3_PIPELINE_TRANSPORT:-host-rdma}" = host-staged ]; then
        scp -q "$K3_FIRMWARE/libhidden_transport_host_staged_tcp.so" "$host:$root/lib/hidden_pipeline.so"
      fi
      scp -q "$generated/deployment.json" "$host:$root/"
      scp -q "$generated/adapter.$host.json" "$host:$root/config/adapter.json"
      $SSH "$host" "ln -sfn $pack $root/packs/ && ln -sfn $pack.experts $root/packs/ && head -c 64 $pack.sha256 > $root/packs/pack.sha256 && python3 $root/bin/weightd_spine_budget.py $pack > $root/spine_budget"
      echo "$host ready spine_budget=$($SSH "$host" cat "$root/spine_budget")"
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
  local rank host root stage run_id
  run_id="$(lane_run_id "${K3_RUN_ID:-}")" || exit 2
  echo "k3_lane: run $run_id; rank logs are $(lane_log_file "$run_id") under each lane root"
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    stage=$((rank / 4))
    wrapper=""
    case " ${K3_WRAP_RANKS:-} " in *" $rank "*) wrapper="${K3_RANK_WRAPPER:-} " ;; esac
    $SSH "$host" "cd $root && $(lane_log_prelude "$host" "$run_id") && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=$K3_MEMORY_MAX -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=$K3_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$K3_LANE -E SPARK_TP_MESH_RANKS=$(mesh_ranks_of "$rank") -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=$K3_EXPERT_POOL_BYTES -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=\$(cat spine_budget) -E CUDA_MODULE_LOADING=LAZY -E CUDA_DEVICE_MAX_CONNECTIONS=${K3_DEVICE_MAX_CONNECTIONS:-32} bash -c 'exec ${wrapper}./bin/sparkpipe_model_residentd --deployment deployment.json --rank-index $rank > $(lane_log_file "$run_id") 2>&1'" &
    PIDS[$rank]=$!
  done
  join_ranks start
}

status() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    printf '%s %s\n' "$host" "$($SSH "$host" "systemctl --user show $UNIT -p ActiveState --value; awk '/MemAvailable/{printf \"avail=%dG \", \$2/1048576}' /proc/meminfo; tail -1 $(root_of "$host")/residentd.log 2>/dev/null | cut -c1-160" | tr '\n' ' ')"
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

reclaim() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    $SSH "$host" "systemctl --user is-active -q $UNIT && { echo $host: $UNIT still active; exit 1; }; $(root_of "$host")/bin/weightd_warm $K3_WEIGHTD_SOCKET --reclaim-pack $(pack_of "$rank") 2>&1 | sed 's/^/$host: /'" &
    PIDS[$rank]=$!
  done
  join_ranks reclaim
}

clean() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    $SSH "$host" "systemctl --user is-active -q $UNIT && { echo $host: $UNIT still active; exit 1; }; rm -rf /home/$host/k3-lanes/lane$K3_LANE" &
    PIDS[$rank]=$!
  done
  join_ranks clean
}

batch() {
  local host root
  host="$(host_of 0)"
  root="$(root_of "$host")"
  scp -q "$K3_FIRMWARE/sparkpipe_model_batch" "$host:$root/bin/"
  scp -q "$1" "$host:$root/batch.json"
  $SSH "$host" "cd $root && timeout 900 ./bin/sparkpipe_model_batch --deployment deployment.json --runtime-root $root --batch batch.json"
}

case "${1:-}" in
  render) render "${2:?render OUTPUT_DIRECTORY}" ;;
  setup) setup ;;
  start) start ;;
  status) status ;;
  stop) stop ;;
  reclaim) reclaim ;;
  clean) clean ;;
  batch) batch "${2:?batch BATCH_JSON}" ;;
  archive) lane_archive "${2:?archive RUN_ID DESTINATION}" "${3:?archive RUN_ID DESTINATION}" ;;
  *) echo "usage: $0 render DIR|setup|start|status|stop|reclaim|clean|batch FILE|archive RUN_ID DEST" >&2; exit 2 ;;
esac
