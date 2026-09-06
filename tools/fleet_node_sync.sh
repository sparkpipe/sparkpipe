#!/usr/bin/env bash
# fleet_node_sync.sh — runs ON a spark: pull-loop the runtime root from a
# reference tree (ceph or any reachable path), restart the local residentd
# when a pull changed anything. Restart is self-triggered per node, so
# reloads are parallel and need no coordinator. Rank derives from hostname.
#
# usage: fleet_node_sync.sh REFERENCE ROOT_NAME   (run under systemd/tmux)
set -uo pipefail
REF="${1:?reference tree (readable from this node)}"
NAME="${2:?runtime root name (under ~/sparkdata/)}"
HOST=$(hostname)
RANK=$((16#${HOST#spark}))
ROOT="$HOME/sparkdata/$NAME"
PARTS=(lib bin stages config model_resident.json model_package.json)
PID_FILE="$HOME/.fleet_sync_$NAME.pid"

restart_self() {
    local p rr="$ROOT"
    for p in $(pgrep -f "bin/sparkpipe_model_(residentd|api)"); do
        [ "$(readlink /proc/$p/cwd 2>/dev/null)" = "$rr" ] && kill -TERM "$p"
    done
    sleep 1
    cd "$rr" || return 1
    ln -sf "stage_$(printf %02d "$RANK").json" config/stage.json
    mv residentd.log residentd.log.prev 2>/dev/null
    LD_LIBRARY_PATH="$rr/lib" nohup ./bin/sparkpipe_model_residentd \
        --deployment model_resident.json --rank-index "$RANK" \
        > residentd.log 2>&1 < /dev/null &
    echo $! > "$PID_FILE"
}

echo "$$" > "$PID_FILE"
echo "node-sync: rank=$RANK root=$ROOT ref=$REF"
while true; do
    CHANGED=""
    for p in "${PARTS[@]}"; do
        [ -e "$REF/$p" ] || continue
        OUT=$(rsync -a --out-format=%n "$REF/$p" "$ROOT/" 2>/dev/null) \
            && [ -n "$OUT" ] && CHANGED=1
    done
    [ -n "$CHANGED" ] && { echo "$(date +%T) changed; restarting"; restart_self; }
    sleep 5
done
