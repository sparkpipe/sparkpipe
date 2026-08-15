#!/usr/bin/env bash
# preflight_pro.sh — verify the DSV4 Pro deployment is ring-test-ready on all
# 16 hosts: driver sha, pack presence+size, configs present.
set -euo pipefail

HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
ROOT_NAME="dsv4_pro.tp4pp4"
DRIVER_SHA="43914327517f3a73"
EXPECTED_SHA="43914327517f3a73"

pass=0
fail=0
for host in "${HOSTS[@]}"; do
    root="/home/${host}/sparkdata/${ROOT_NAME}"
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "${host}" "
        d=\$(sha256sum ${root}/lib/model_driver.so 2>/dev/null | cut -c1-16)
        p=\$(stat -c %s ${root}/packs/dsv4_pro_tp4_pp4_stage.spstage 2>/dev/null || echo 0)
        c=\$(test -f ${root}/config/model_resident.json && test -f ${root}/config/dsv4_pro_tp4_pp4_stage.json && echo yes || echo no)
        echo "\${d} \${p} \${c}"
    " 2>/dev/null)
    driver=$(echo "$out" | cut -d' ' -f1)
    pack=$(echo "$out" | cut -d' ' -f2)
    config=$(echo "$out" | cut -d' ' -f3)
    if [[ "$driver" == "$EXPECTED_SHA" && "$config" == "yes" && "$pack" -gt 50000000000 ]]; then
        echo "$host OK driver=$driver pack=$pack config=$config"
        pass=$((pass + 1))
    else
        echo "$host INCOMPLETE driver=$driver pack=$pack config=$config"
        fail=$((fail + 1))
    fi
done
echo "preflight: ${pass}/16 ready, ${fail} incomplete"
