#!/usr/bin/env bash
# k3_smoke_rank0.sh SPARK_HOST [SRC_DIR] — P4 pack smoke on the rank-0 node.
#
# Assembles the K3 runtime dir (bin/lib/config) from a source tree that
# already holds the lane builds, then proves the deployed rank pack loads
# two ways, WITHOUT needing the rest of the fleet:
#   1. the serving-adapter smoke (tp_degree 1, no collective, no port):
#      config JSON -> runner -> SparkK3PackOpen -> bind -> dispatch init.
#   2. the real residentd with a single-node deployment: waits for the
#      "model_residentd ready ... model=k3" line, then TERM-stops it.
#
# Reuses hidden_transport.so + libnccl.so.2 from an existing deployment on
# the node when present (model-agnostic ring transport); everything else is
# the lane's own build.
#
# Script parameterization rule: the spark host is argument 1, never
# hardcoded. GPU hygiene: no daemon is left running (TERM + wait).
set -euo pipefail
HOST="${1:?usage: k3_smoke_rank0.sh SPARK_HOST [SRC_DIR]}"
SRC="${2:-/tmp/k3_lane_src}"
RT="/home/${HOST}/sparkdata/k3.mxfp4.tp4pp4"
PACK="${RT}/packs/k3.stage3.rank00.pack"

ssh -o BatchMode=yes "$HOST" "test -s '$SRC/build/k3_adapter_smoke' &&
  test -s '$SRC/build/libk3_serving_adapter.so' &&
  test -s '$SRC/build/sparkpipe_model_residentd' &&
  test -s '$PACK'" || { echo "smoke prerequisites missing on $HOST" >&2; exit 1; }

ssh -o BatchMode=yes "$HOST" "set -e
  mkdir -p $RT/bin $RT/lib/runtime_libs $RT/config $RT/kvcache-smoke
  cp $SRC/build/sparkpipe_model_residentd $RT/bin/
  cp $SRC/build/libk3_serving_adapter.so $RT/lib/
  if [ ! -s $RT/lib/hidden_transport.so ]; then
    cp /home/$HOST/sparkdata/dsv4_flash.fp8.tp4_pp4.b1/lib/hidden_transport.so $RT/lib/
  fi
  cp /home/$HOST/sparkdata/dsv4_flash.fp8.tp4_pp4.b1/lib/runtime_libs/libnccl.so.2 $RT/lib/runtime_libs/
  cp $SRC/build/k3_adapter_smoke $RT/bin/
  cat > $RT/config/adapter_smoke.json <<JSON
{
  \"stage_pack_path\": \"$PACK\",
  \"tp_degree\": 1,
  \"tp_rank\": 0,
  \"world_size\": 1,
  \"max_sequences\": 16,
  \"max_rows\": 16,
  \"resident_capacity\": 16,
  \"kv_pages\": 2,
  \"hidden\": 7168
}
JSON
  cat > $RT/config/model_resident_smoke.json <<JSON
{
  \"schema_version\": 2,
  \"coordinator_rank_index\": 0,
  \"adapter\": {\"shared_object_path\": \"lib/libk3_serving_adapter.so\"},
  \"driver\": {\"shared_object_path\": \"lib/libk3_serving_adapter.so\",
              \"program_name\": \"k3\"},
  \"transport\": {\"shared_object_path\": \"lib/hidden_transport.so\",
                 \"mode\": \"host-rdma\", \"control_port_base\": 62700},
  \"runtime_limits\": {
    \"max_inflight_submissions\": 16,
    \"max_active_sequences\": 16,
    \"max_input_rows\": 16,
    \"resident_sequence_capacity\": 16,
    \"kv_logical_page_capacity\": 4096,
    \"kv_physical_page_capacity\": 4096
  },
  \"nodes\": [
    {
      \"rank_index\": 0,
      \"stage_index\": 0,
      \"runtime_root\": \"$RT\",
      \"node_target\": \"cuda.sm121.k3.resident_decode_stage.linear_bf16.expert_mxfp4.kv_bf16\",
      \"transport_host\": \"$HOST\",
      \"adapter_configuration_path\": \"config/adapter_smoke.json\",
      \"kv_backing_directory\": \"$RT/kvcache-smoke\",
      \"kv_backing_maximum_bytes\": 137438953472,
      \"control_endpoint\": {\"kind\": \"tcp\", \"host\": \"$HOST\", \"port\": 21480}
    }
  ]
}
JSON
  echo '--- adapter smoke (tp_degree 1, real rank pack) ---'
  cd $RT
  LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH timeout 120 bin/k3_adapter_smoke config/adapter_smoke.json
"

echo "--- residentd single-node ready line ---"
ssh -o BatchMode=yes "$HOST" "set -e
  cd $RT
  pkill -f 'sparkpipe_model_residentd --deployment config/model_resident_smoke.json' 2>/dev/null || true
  sleep 2
  rm -f /tmp/k3-smoke-rank0.log
  LD_LIBRARY_PATH=\$PWD/lib:\$LD_LIBRARY_PATH setsid -f \
    bin/sparkpipe_model_residentd --deployment config/model_resident_smoke.json \
    --rank-index 0 >/tmp/k3-smoke-rank0.log 2>&1 </dev/null
  for i in \$(seq 1 60); do
    if grep -q 'model_residentd ready' /tmp/k3-smoke-rank0.log 2>/dev/null; then break; fi
    if grep -qiE 'status [-1-9][0-9]*|FAIL|error' /tmp/k3-smoke-rank0.log 2>/dev/null; then break; fi
    sleep 2
  done
  cat /tmp/k3-smoke-rank0.log
  pkill -f 'sparkpipe_model_residentd --deployment config/model_resident_smoke.json' 2>/dev/null || true
  sleep 2
  pgrep -f sparkpipe_model >/dev/null && echo 'WARNING: daemon still up' || echo 'daemon stopped cleanly'
"
