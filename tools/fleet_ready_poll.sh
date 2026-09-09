#!/usr/bin/env bash
start=$(date +%s)
while true; do
    now=$(date +%s)
    if [ $((now - start)) -gt 90 ]; then
        echo "TIMEOUT 90s"
        exit 1
    fi
    view=$(ssh -o BatchMode=yes -o ConnectTimeout=4 sparkf \
        "grep -o '\"state\":\"[a-z:]*\"' current/*.json 2>/dev/null | sort | uniq -c" 2>/dev/null)
    ready=$(printf '%s\n' "$view" | grep '"state":"ready"' | awk '{print $1}')
    pending=$(ssh -o BatchMode=yes -o ConnectTimeout=4 rtx5090 \
        "test -f release/glm53flash.fp8.tp16/UPDATE && echo UPDATE-PENDING || echo no-update" 2>/dev/null)
    echo "t=$((now - start))s $pending ready=${ready:-0}/16"
    if [ "$pending" = no-update ] && [ "${ready:-0}" -ge 16 ]; then
        echo "FLEET-READY t=$((now - start))s"
        exit 0
    fi
    sleep 5
done
