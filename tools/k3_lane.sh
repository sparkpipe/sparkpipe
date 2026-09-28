#!/usr/bin/env bash
set -euo pipefail
: "${K3_LANE:?K3_LANE is the weightd mesh lane (1..15, never 0)}"
: "${K3_FIRMWARE:?K3_FIRMWARE is a local directory holding sparkpipe_model_residentd, libk3_serving_adapter.so, libhidden_transport_spark_host_rdma_verbs.so, weightd_warm and sparkpipe_model_batch}"
: "${K3_WEIGHTD_SOCKET:?K3_WEIGHTD_SOCKET is the running weightd socket on every node}"
: "${K3_EXPERT_POOL_BYTES:?K3_EXPERT_POOL_BYTES is the per-rank routed expert pool}"
: "${K3_MEMORY_MAX:?K3_MEMORY_MAX is the residentd unit MemoryMax, e.g. 10G}"
[ "$K3_LANE" -ge 1 ] && [ "$K3_LANE" -le 15 ] || { echo "k3_lane: lane $K3_LANE outside 1..15" >&2; exit 2; }
HERE="$(cd "$(dirname "$0")" && pwd)"
CHECKOUT="$(cd "$HERE/.." && pwd)"
HEX=0123456789abcdef
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
UNIT="sp-k3-rd$K3_LANE"
PACK_DIR=sparkdata/k3.mxfp4.tp4pp4/packs
host_of() { echo "spark${HEX:$1:1}"; }
root_of() { echo "/dev/shm/k3-lane$K3_LANE-$1/root"; }
pack_of() { echo "/home/$(host_of "$1")/$PACK_DIR/k3.stage$(($1 / 4)).rank0$(($1 % 4)).pack"; }
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

spine_budget_of() {
  python3 - "$CHECKOUT/model-families/k3/smoke_experts.json" "$1" <<'PY'
import json, sys
manifest = json.load(open(sys.argv[1]))
rank = int(sys.argv[2])
name = f"k3.stage{rank // 4}.rank0{rank % 4}.pack"
for entry in manifest["provenance"]["ranks"]:
    if entry["pack"] == name:
        print(entry["spine_bytes"])
        raise SystemExit(0)
raise SystemExit(f"no spine budget for {name}")
PY
}

render() {
  python3 "$HERE/k3_multidev_lane.py" --lane "$K3_LANE" \
    --runtime-root "/dev/shm/k3-lane$K3_LANE-{host}/root" \
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
      $SSH "$host" "test -f $pack -a -f $pack.experts -a -f $pack.sha256 && mkdir -p $root/bin $root/lib $root/config $root/packs $root/kvcache && find $root/packs $root/kvcache -mindepth 1 -delete"
      scp -q "$K3_FIRMWARE/sparkpipe_model_residentd" "$K3_FIRMWARE/weightd_warm" "$host:$root/bin/"
      scp -q "$K3_FIRMWARE/libk3_serving_adapter.so" "$host:$root/lib/"
      scp -q "$K3_FIRMWARE/libhidden_transport_spark_host_rdma_verbs.so" "$host:$root/lib/hidden_transport.so"
      scp -q "$generated/deployment.json" "$host:$root/"
      scp -q "$generated/adapter.$host.json" "$host:$root/config/adapter.json"
      $SSH "$host" "ln -sfn $pack $root/packs/ && ln -sfn $pack.experts $root/packs/ && head -c 64 $pack.sha256 > $root/packs/pack.sha256 && echo $(spine_budget_of "$rank") > $root/spine_budget"
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
  local rank host root stage
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    root="$(root_of "$host")"
    stage=$((rank / 4))
    $SSH "$host" "cd $root && systemctl --user reset-failed $UNIT 2>/dev/null; systemd-run --user --unit=$UNIT -p MemoryMax=$K3_MEMORY_MAX -p MemorySwapMax=0 -p LimitMEMLOCK=infinity --working-directory=$root -E LD_LIBRARY_PATH=$root/lib -E SPARK_WEIGHTD_ATTACH=1 -E SPARK_WEIGHTD_SOCKET=$K3_WEIGHTD_SOCKET -E SPARK_WEIGHTD_LANE=$K3_LANE -E SPARK_TP_MESH_RANKS=$((stage * 4)),$((stage * 4 + 1)),$((stage * 4 + 2)),$((stage * 4 + 3)) -E SPARK_WEIGHTD_EXPERT_POOL_BYTES=$K3_EXPERT_POOL_BYTES -E SPARK_WEIGHTD_SPINE_BUDGET_BYTES=\$(cat spine_budget) -E CUDA_MODULE_LOADING=LAZY -E CUDA_DEVICE_MAX_CONNECTIONS=32 bash -c 'exec ./bin/sparkpipe_model_residentd --deployment deployment.json --rank-index $rank > residentd.log 2>&1'" &
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
    $SSH "$host" "systemctl --user is-active -q $UNIT && { echo $host: $UNIT still active; exit 1; }; $(root_of "$host")/bin/weightd_warm $K3_WEIGHTD_SOCKET --reclaim 2>&1 | sed 's/^/$host: /'" &
    PIDS[$rank]=$!
  done
  join_ranks reclaim
}

clean() {
  local rank host
  for rank in $(seq 0 15); do
    host="$(host_of "$rank")"
    $SSH "$host" "systemctl --user is-active -q $UNIT && { echo $host: $UNIT still active; exit 1; }; rm -rf /dev/shm/k3-lane$K3_LANE-$host" &
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
  *) echo "usage: $0 render DIR|setup|start|status|stop|reclaim|clean|batch FILE" >&2; exit 2 ;;
esac
