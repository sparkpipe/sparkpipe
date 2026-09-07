#!/usr/bin/env bash
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
start=$(date +%s)
while true; do
    now=$(date +%s)
    if [ $((now - start)) -gt 90 ]; then
        echo "TIMEOUT 90s"
        exit 1
    fi
    rec=$(ssh -o BatchMode=yes -o ConnectTimeout=4 rtx5090 \
        "f=\$(ls release/glm53flash.fp8.tp16/UPDATE 2>/dev/null); [ -n \"\$f\" ] && echo UPDATE-PENDING || echo no-update" 2>/dev/null)
    n=0
    for h in "${HOSTS[@]}"; do
        r=$(ssh -o BatchMode=yes -o ConnectTimeout=3 "$h" \
            "tail -1 ~/sparkdata/glm53flash.fp8.tp16/residentd.log 2>/dev/null | grep -c 'model_residentd ready'" 2>/dev/null)
        n=$((n + r))
    done
    echo "t=$((now - start))s $rec ready=$n/16"
    if [ "$rec" = no-update ] && [ "$n" -ge 16 ]; then
        echo "FLEET-READY t=$((now - start))s"
        exit 0
    fi
    sleep 5
done
