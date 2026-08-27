#!/usr/bin/env bash
# Lay down the DSV4 Pro TP4xPP4 runtime scaffold (bin/lib/config) on one
# target spark, from the controller mac. Packs arrive separately via
# dsv4pro_rank_deploy.sh. Parameterized: --target HOST --rank N required.
#
# usage: dsv4pro_scaffold_deploy.sh --target HOST --rank N [--binary-source HOST]
#   --binary-source   node holding the Aug-17 Pro binaries (default spark3)
set -euo pipefail

target=""
rank=""
binary_source="spark3"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) target="$2"; shift 2 ;;
        --rank) rank="$2"; shift 2 ;;
        --binary-source) binary_source="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ -n "${target}" && -n "${rank}" ]] || { echo "--target HOST --rank N required" >&2; exit 2; }
[[ "${rank}" =~ ^(1[0-5]|[0-9])$ ]] || { echo "--rank must be 0..15" >&2; exit 2; }

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
deploy_root="/home/${target}/sparkdata/dsv4_pro.tp4pp4"
src_root="/home/${binary_source}/sparkdata/dsv4_pro.tp4pp4"

ssh -o BatchMode=yes "${target}" "mkdir -p '${deploy_root}/bin' '${deploy_root}/lib' '${deploy_root}/config' '${deploy_root}/packs' '${deploy_root}/kv'"

# Binaries + shared objects relay via the controller (sparks do not ssh peers).
for f in bin/sparkpipe_model_residentd bin/sparkpipe_model_batch \
         lib/model_driver.so lib/libdsv4_pro_tp4_pp4_serving_adapter.so \
         lib/libhidden_transport_spark_host_rdma_verbs.so; do
    rsync -a "${binary_source}:${src_root}/${f}" "${target}:${deploy_root}/${f}"
done

# Adapter/firmware config: current repo descriptor (PR 721, 33024 ceiling).
rsync -a "${repo}/examples/deployments/dsv4_pro_tp4_pp4_stage.json" \
    "${target}:${deploy_root}/config/dsv4_pro_tp4_pp4_stage.json"

# model_resident.json: same schema as the original deployment, runtime_root
# corrected to dsv4_pro.tp4pp4 on every node.
ssh -o BatchMode=yes "${target}" "rank='${rank}' host='${target}' python3 - <<'PYEOF'
import json, os
rank = int(os.environ['rank'])
host = os.environ['host']
root = f'/home/{host}/sparkdata/dsv4_pro.tp4pp4'
document = {
  'schema_version': 2,
  'coordinator_rank_index': 0,
  'adapter': {'shared_object_path': 'lib/libdsv4_pro_tp4_pp4_serving_adapter.so'},
  'driver': {'shared_object_path': 'lib/model_driver.so', 'program_name': 'resident_decode'},
  'transport': {
    'shared_object_path': 'lib/libhidden_transport_spark_host_rdma_verbs.so',
    'mode': 'host-rdma', 'control_port_base': 61700,
  },
  'runtime_limits': {
    'max_inflight_submissions': 4, 'max_active_sequences': 1024,
    'max_input_rows': 1024, 'resident_sequence_capacity': 4096,
    'kv_logical_page_capacity': 1048576, 'kv_physical_page_capacity': 16384,
  },
  'nodes': [
    {
      'rank_index': r,
      'stage_index': r,
      'runtime_root': f'/home/spark{format(r, \"x\")}/sparkdata/dsv4_pro.tp4pp4',
      'node_target': 'cuda.sm121.dsv4.pro.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16',
      'transport_host': f'spark{format(r, \"x\")}-fabric',
      'adapter_configuration_path': 'config/dsv4_pro_tp4_pp4_stage.json',
      'kv_backing_directory': f'/home/spark{format(r, \"x\")}/kvcache/dsv4_pro/tp4_pp4.bf16',
      'kv_backing_maximum_bytes': 4398046511104,
      'control_endpoint': {'kind': 'tcp', 'host': f'spark{format(r, \"x\")}', 'port': 20480},
    } for r in range(16)
  ],
}
path = os.path.join(root, 'config', 'model_resident.json')
with open(path, 'w') as handle:
    json.dump(document, handle, indent=2)
print('wrote', path, 'for rank', rank)
PYEOF"
echo "scaffold done target=${target} rank=${rank}"
