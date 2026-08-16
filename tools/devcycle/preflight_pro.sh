#!/usr/bin/env bash
# preflight_pro.sh — verify the DSV4 Pro deployment is ring-test-ready on all
# 16 hosts: driver sha, adapter + transport libs, pack presence+size, configs
# (with the regenerated transport_hosts variant), KV backing dirs.
set -euo pipefail

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
ROOT_NAME="dsv4_pro.tp4pp4"
EXPECTED_SHA="5d6ee5bf53e3dfd5"  # GA 0813 driver
EXPECTED_CONFIG_SHA="0c7103acc7ab59b872bef10db450711c4ca4fb94d81d6017efcccab36c38780b"
EXPECTED_STAGE_SHA="7bdc343786885f1a5b3f2d78acfe6791105ca97335e0662ab95ce1084a34e57a"

pass=0
fail=0
for host in "${HOSTS[@]}"; do
    root="/home/${host}/sparkdata/${ROOT_NAME}"
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "
        d=\$(sha256sum ${root}/lib/model_driver.so 2>/dev/null | cut -c1-16)
        a=\$(test -s ${root}/lib/libdsv4_pro_tp4_pp4_serving_adapter.so && echo yes || echo no)
        t=\$(test -s ${root}/lib/libhidden_transport_spark_host_rdma_verbs.so && echo yes || echo no)
        k=\$(test -d /home/${host}/kvcache/dsv4_pro/tp4pp4.bf16 && echo yes || echo no)
        p=\$(stat -c %s ${root}/packs/dsv4_pro_tp4_pp4_stage.spstage 2>/dev/null || echo 0)
        c=\$(test -f ${root}/config/model_resident.json && test -f ${root}/config/dsv4_pro_tp4_pp4_stage.json && echo yes || echo no)
        r=\$(sha256sum ${root}/config/model_resident.json 2>/dev/null | cut -c1-64)
        s=\$(sha256sum ${root}/config/dsv4_pro_tp4_pp4_stage.json 2>/dev/null | cut -c1-64)
        echo "\${d} \${p} \${c} \${a} \${t} \${k} \${r} \${s}"
    " 2>/dev/null)
    driver=$(echo "$out" | cut -d' ' -f1)
    pack=$(echo "$out" | cut -d' ' -f2)
    config=$(echo "$out" | cut -d' ' -f3)
    adapter=$(echo "$out" | cut -d' ' -f4)
    transport=$(echo "$out" | cut -d' ' -f5)
    kv=$(echo "$out" | cut -d' ' -f6)
    resident_sha=$(echo "$out" | cut -d' ' -f7)
    stage_sha=$(echo "$out" | cut -d' ' -f8)
    if [[ "$driver" == "$EXPECTED_SHA" && "$config" == "yes" && "$adapter" == "yes" && "$transport" == "yes" && "$kv" == "yes" && "$pack" -gt 50000000000 && "$resident_sha" == "$EXPECTED_CONFIG_SHA" && "$stage_sha" == "$EXPECTED_STAGE_SHA" ]]; then
        echo "$host OK driver=$driver pack=$pack config=$resident_sha"
        pass=$((pass + 1))
    else
        echo "$host INCOMPLETE driver=$driver pack=$pack config=${resident_sha:0:8} stage=${stage_sha:0:8} adapter=$adapter transport=$transport kv=$kv"
        fail=$((fail + 1))
    fi
done
echo "preflight: ${pass}/16 ready, ${fail} incomplete"
