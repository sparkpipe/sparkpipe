#!/usr/bin/env bash
# k3_stage_runtime.sh — pre-stage the K3 runtime trees on all 16 sparks:
#   <runtime_root>/bin/sparkpipe_model_residentd
#   <runtime_root>/lib/libk3_serving_adapter.so
#   <runtime_root>/lib/hidden_transport.so
#   <runtime_root>/config/model_resident.json
#   <runtime_root>/config/adapter.json   (per-host adapter config)
# No residentd is STARTED - this is file placement only, safe without the
# ring reservation.
# TP_DEGREE (arg 3, default 4): 16 stages the TP16 PP1 tree
# (k3.mxfp4.tp16) and REQUIRES CONFIG_DIR (arg 4) - a directory of
# generated per-host TP16 adapter configs (k3_gen_adapter_configs.sh DIR 16)
# plus the TP16 deployment JSON it stages as model_resident.json
# (k3_gen_deployment.sh FILE 16).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_HOST="${1:-sparka}"
BUILD_DIR="${2:-/home/sparka/sparkpipe-k3/build}"
TP="${3:-4}"
case "$TP" in
  4) RT="tp4pp4" ;;
  16)
    RT="tp16"
    CONFIG_DIR="${4:?TP16 needs CONFIG_DIR: k3_gen_adapter_configs.sh DIR 16 output}"
    ;;
  *) echo "unsupported tp degree $TP (4 or 16)" >&2; exit 1 ;;
esac
for i in $(seq 0 15); do
  hex=$(printf '%x' "$i")
  host="spark$hex"
  runtime="/home/$host/sparkdata/k3.mxfp4.$RT"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "mkdir -p $runtime/bin $runtime/lib $runtime/config $runtime/kvcache"
  scp -q "$BUILD_HOST:$BUILD_DIR/sparkpipe_model_residentd" "$host:$runtime/bin/"
  scp -q "$BUILD_HOST:$BUILD_DIR/libk3_serving_adapter.so" "$host:$runtime/lib/"
  scp -q "$BUILD_HOST:$BUILD_DIR/libhidden_transport_spark_host_rdma_verbs.so" "$host:$runtime/lib/hidden_transport.so"
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "chmod +x $runtime/bin/sparkpipe_model_residentd"
  if [ "$TP" = "16" ]; then
    gen=$(mktemp)
    bash "$ROOT/tools/k3_gen_deployment.sh" "$gen" 16 > /dev/null
    scp -q "$gen" "$host:$runtime/config/model_resident.json"
    rm -f "$gen"
    scp -q "$CONFIG_DIR/$host.json" "$host:$runtime/config/adapter.json"
  else
    scp -q "$ROOT/modules/k3_resident_decode_stage/configs/model_resident.json" "$host:$runtime/config/"
    scp -q "$ROOT/modules/k3_resident_decode_stage/configs/$host.json" "$host:$runtime/config/adapter.json"
  fi
  echo "staged $host ($RT)"
done
echo "k3 runtime staged on 16 sparks ($RT)"
