#!/usr/bin/env bash
#
# deploy_pro.sh [--skip-packs] — deploy the DSV4 Pro TP4xPP4 runtime to all
# 16 hosts. Run from the pro worktree on the MacBook.
#
#   - runtime artifacts from /tmp/devcycle-pro-build-pro-base on sparkb
#   - rank packs from /home/sparkb/sparkdata/dsv4_pro.tp4_pp4.ranks on sparkb
#   - deployment configs generated from the pro spec in this checkout
#
#   --skip-packs  deploy bins/libs/configs only (use while the split runs)
set -euo pipefail

BUILD_HOST="sparkb"
RUNTIME_DIR_NAME="dsv4_pro.tp4pp4"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SKIP_PACKS=0
[[ "${1:-}" == "--skip-packs" ]] && SKIP_PACKS=1

SC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SC/../.." && pwd)"

rm -rf /tmp/pro-deploy
mkdir -p /tmp/pro-deploy

for artifact in sparkpipe_model_residentd sparkpipe_model_batch model_driver.so \
                model_serving_adapter.so hidden_transport.so; do
    scp -q -o BatchMode=yes "${BUILD_HOST}:/tmp/devcycle-pro-build-pro-base/${artifact}" /tmp/pro-deploy/ \
        || { echo "missing artifact ${artifact} on ${BUILD_HOST}"; exit 1; }
done

python3 "${ROOT}/tools/generate_model_resident_deployment.py" \
    --specification "${ROOT}/examples/deployments/dsv4_pro_tp4_pp4_host_rdma.spec.json" \
    --output /tmp/pro-deploy/model_resident.json
cp "${ROOT}/examples/deployments/dsv4_pro_tp4_pp4_stage.json" /tmp/pro-deploy/dsv4_pro_tp4_pp4_stage.json

rank=0
for host in "${HOSTS[@]}"; do
    root="/home/${host}/sparkdata/${RUNTIME_DIR_NAME}"
    ssh -o BatchMode=yes "${host}" "mkdir -p ${root}/bin ${root}/lib ${root}/config ${root}/kv ${root}/packs" || exit 1
    scp -q -o BatchMode=yes /tmp/pro-deploy/sparkpipe_model_residentd /tmp/pro-deploy/sparkpipe_model_batch "${host}:${root}/bin/" || exit 1
    scp -q -o BatchMode=yes /tmp/pro-deploy/model_driver.so "${host}:${root}/lib/" || exit 1
    scp -q -o BatchMode=yes /tmp/pro-deploy/model_serving_adapter.so "${host}:${root}/lib/libdsv4_pro_tp4_pp4_serving_adapter.so" || exit 1
    scp -q -o BatchMode=yes /tmp/pro-deploy/hidden_transport.so "${host}:${root}/lib/libhidden_transport_spark_host_rdma_verbs.so" || exit 1
    ssh -o BatchMode=yes "${host}" "mkdir -p /home/${host}/kvcache/dsv4_pro/tp4pp4.bf16" || exit 1
    scp -q -o BatchMode=yes /tmp/pro-deploy/model_resident.json /tmp/pro-deploy/dsv4_pro_tp4_pp4_stage.json "${host}:${root}/config/" || exit 1
    if [[ $SKIP_PACKS == 0 ]]; then
        scp -q -o BatchMode=yes "${BUILD_HOST}:/home/sparkb/sparkdata/dsv4_pro.tp4_pp4.ranks/dsv4_pro.tp4_pp4.rank$(printf '%02d' $rank).spstage" \
            "${host}:${root}/packs/dsv4_pro_tp4_pp4_stage.spstage" || exit 1
    fi
    echo "deployed ${host} rank=${rank}$([[ $SKIP_PACKS == 1 ]] && echo ' (no pack)')"
    rank=$((rank + 1))
done

echo "deploy_pro done; designate with: tools/fleet_swap.sh dsv4-pro"
