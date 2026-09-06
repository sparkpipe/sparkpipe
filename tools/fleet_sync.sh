#!/usr/bin/env bash
# fleet_sync.sh — install/start/stop the per-node pull loops on every spark.
# Each node pulls the reference tree itself (parallel, no fan-out) and
# restarts its own daemons when a pull changes anything. The reference is
# the manifest: build into it, the fleet converges within one sleep.
#
# usage: tools/fleet_sync.sh REFERENCE ROOT_NAME [start|stop|status]
set -uo pipefail
REF="${1:?reference tree}"
NAME="${2:?runtime root name}"
CMD="${3:-start}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

case "$CMD" in
start)
    for h in "${HOSTS[@]}"; do
        scp -q "$HERE/fleet_node_sync.sh" "$h:~/fleet_node_sync.sh" &
    done
    wait
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "chmod +x ~/fleet_node_sync.sh; pkill -f fleet_node_sync.sh.*$NAME 2>/dev/null; setsid nohup ~/fleet_node_sync.sh '$REF' '$NAME' > ~/fleet_node_sync.log 2>&1 < /dev/null &" &
    done
    wait
    echo "pull loops running on ${#HOSTS[@]} hosts (ref=$REF)"
    ;;
stop)
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "pkill -f fleet_node_sync.sh.*$NAME; true" &
    done
    wait
    echo "pull loops stopped"
    ;;
status)
    for h in "${HOSTS[@]}"; do
        n=$($SSH "$h" "pgrep -fc 'fleet_node_sync.sh $REF $NAME' 2>/dev/null" || true)
        echo "$h: ${n:-0}"
    done
    ;;
esac
