#!/usr/bin/env bash
# k3_gen_deployment.sh OUT_PATH [TP_DEGREE] — emit the K3 model_resident.json
# deployment (schema 2) for the 16 ranks. TP_DEGREE 4 (default) = TP4xPP4;
# 16 = TP16 PP1 (runtime root k3.mxfp4.tp16). rank_index = world rank;
# stage_index = the LINEAR pipe position (unique per node: the deployment
# schema treats every rank as a stage; the K3 adapter derives its PP stage
# as stage_index / tp_degree itself). The adapter embeds the driver so the
# driver path is never loaded, but the residentd's schema requires the
# fields and the program-name match ("k3"). Transport = the host-RDMA
# hidden stream at the registry's transport control base 62700.
set -euo pipefail
OUT="${1:?usage: k3_gen_deployment.sh OUT_PATH [TP_DEGREE(4|16)]}"
TP="${2:-4}"
case "$TP" in
  4) RT="tp4pp4" ;;
  16) RT="tp16" ;;
  *) echo "unsupported tp degree $TP (4 or 16)" >&2; exit 1 ;;
esac
{
  echo '{'
  echo '  "schema_version": 2,'
  echo '  "coordinator_rank_index": 0,'
  echo '  "adapter": {'
  echo '    "shared_object_path": "lib/libk3_serving_adapter.so"'
  echo '  },'
  echo '  "driver": {'
  echo '    "shared_object_path": "lib/libk3_serving_adapter.so",'
  echo '    "program_name": "k3"'
  echo '  },'
  echo '  "transport": {'
  echo '    "shared_object_path": "lib/hidden_transport.so",'
  echo '    "mode": "host-rdma",'
  echo '    "control_port_base": 62700'
  echo '  },'
  echo '  "runtime_limits": {'
  echo '    "max_inflight_submissions": 16,'
  echo '    "max_active_sequences": 16,'
  echo '    "max_input_rows": 16,'
  echo '    "resident_sequence_capacity": 16,'
  echo '    "kv_logical_page_capacity": 0,'
  echo '    "kv_physical_page_capacity": 0'
  echo '  },'
  echo '  "nodes": ['
  for i in $(seq 0 15); do
    hex=$(printf '%x' "$i")
    host="spark$hex"
    # stage_index is the LINEAR pipe position (must be unique per node:
    # the deployment schema treats every rank as a stage). The K3 adapter
    # derives its PP stage as stage_index / tp_degree itself.
    stage=$i
    comma=","
    [ "$i" = "15" ] && comma=""
    echo '    {'
    echo '      "rank_index": '$i','
    echo '      "stage_index": '$stage','
    echo '      "runtime_root": "/home/'$host'/sparkdata/k3.mxfp4.'$RT'",'
    echo '      "node_target": "cuda.sm121.k3.resident_decode_stage.linear_bf16.expert_mxfp4.kv_bf16",'
    echo '      "transport_host": "'$host'",'
    echo '      "adapter_configuration_path": "config/adapter.json",'
    echo '      "kv_backing_directory": "/home/'$host'/sparkdata/k3.mxfp4.'$RT'/kvcache",'
    echo '      "kv_backing_maximum_bytes": 137438953472,'
    echo '      "control_endpoint": {'
    echo '        "kind": "tcp",'
    echo '        "host": "'$host'",'
    echo '        "port": 21480'
    echo '      }'
    echo '    }'$comma
  done
  echo '  ]'
  echo '}'
} > "$OUT"
echo "wrote $OUT"
