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
    view=$(ssh -o BatchMode=yes -o ConnectTimeout=4 "$VIEW_HOST" "python3 - '$ROOT'" 2>/dev/null <<'PY'
import glob, json, sys
count = 0
degraded = []
ports = {}
for path in glob.glob("current/*.json"):
    try:
        with open(path) as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        continue
    count += data.get("roots", {}).get(sys.argv[1], {}).get("state") == "ready"
    mask = data.get("mesh_degraded", "none")
    if mask not in ("none", "0x0000"):
        degraded.append("%s:%s" % (data.get("host", path), mask))
    layout = data.get("mesh_ports", "none")
    ports[layout] = ports.get(layout, 0) + 1
print("%d\t%s\t%s" % (count, ",".join(sorted(degraded)) or "-",
                      ", ".join("%d on %s" % (number, layout) for layout, number in sorted(ports.items())) or "-"))
PY
)
    IFS=$'\t' read -r ready degraded layout <<< "$view"
    echo "t=$((now - start))s $ROOT ready=${ready:-0}/16"
    if [ "${ready:-0}" -ge 16 ]; then
        if [ -n "$degraded" ] && [ "$degraded" != "-" ]; then
            echo "FLEET-READY-DEGRADED t=$((now - start))s mesh ports: $layout; degraded links (host:ranks): $degraded"
            exit 4
        fi
        echo "FLEET-READY t=$((now - start))s mesh ports: ${layout:--}"
        exit 0
    fi
    sleep 5
done
