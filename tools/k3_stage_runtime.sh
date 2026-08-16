#!/usr/bin/env bash
# k3_stage_runtime.sh — pre-stage the K3 runtime trees on all 16 sparks:
#   <runtime_root>/bin/sparkpipe_model_residentd
#   <runtime_root>/lib/libk3_serving_adapter.so
#   <runtime_root>/lib/hidden_transport.so
#   <runtime_root>/config/model_resident.json
#   <runtime_root>/config/adapter.json   (per-host adapter config)
# No residentd is STARTED - this is file placement only, safe without the
# ring reservation.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_HOST="${1:-sparka}"
BUILD_DIR="${2:-/home/sparka/sparkpipe-k3/build}"
for i in $(seq 0 15); do
  hex=$(printf '%x' "$i")
  host="spark$hex"
  runtime="/home/$host/sparkdata/k3.mxfp4.tp4pp4"
  ssh "$host" "mkdir -p $runtime/bin $runtime/lib $runtime/config $runtime/kvcache"
  scp -q "$BUILD_HOST:$BUILD_DIR/sparkpipe_model_residentd" "$host:$runtime/bin/"
  scp -q "$BUILD_HOST:$BUILD_DIR/libk3_serving_adapter.so" "$host:$runtime/lib/"
  scp -q "$BUILD_HOST:$BUILD_DIR/libhidden_transport_spark_host_rdma_verbs.so" "$host:$runtime/lib/hidden_transport.so"
  ssh "$host" "chmod +x $runtime/bin/sparkpipe_model_residentd"
  scp -q "$ROOT/modules/k3_resident_decode_stage/configs/model_resident.json" "$host:$runtime/config/"
  scp -q "$ROOT/modules/k3_resident_decode_stage/configs/$host.json" "$host:$runtime/config/adapter.json"
  echo "staged $host"
done
echo "k3 runtime staged on 16 sparks"
