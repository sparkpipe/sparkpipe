#!/usr/bin/env bash
ROOT="${1:-glm53flash.fp8.tp16}"
VIEW_HOST="${FLEET_VIEW_HOST:-rtx5090}"
start=$(date +%s)
while true; do
    now=$(date +%s)
    if [ $((now - start)) -gt 90 ]; then
        echo "TIMEOUT 90s"
        exit 1
    fi
    ready=$(ssh -o BatchMode=yes -o ConnectTimeout=4 "$VIEW_HOST" "python3 - '$ROOT'" 2>/dev/null <<'PY'
import glob, json, sys
count = 0
for path in glob.glob("current/*.json"):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        continue
    count += data.get("roots", {}).get(sys.argv[1], {}).get("state") == "ready"
print(count)
PY
)
    echo "t=$((now - start))s $ROOT ready=${ready:-0}/16"
    if [ "${ready:-0}" -ge 16 ]; then
        echo "FLEET-READY t=$((now - start))s"
        exit 0
    fi
    sleep 5
done
