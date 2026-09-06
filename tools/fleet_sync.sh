#!/usr/bin/env bash
# fleet_sync.sh — install/start/stop the per-node release agents on the
# fleet. Each node pulls its runtime roots from the reference base itself
# (parallel, no fan-out), restarts a root only when its files changed
# (TERM -> start new), and reports running versions to hub:current/.
# The reference base is the manifest: build into it, the fleet converges.
#
# usage: tools/fleet_sync.sh REFERENCE_BASE ROOTS_CSV [start|stop|status]
set -uo pipefail
REF="${1:?reference base dir}"
ROOTS="${2:?comma-separated runtime root names}"
CMD="${3:-start}"
HUB="${FLEET_HUB:-sparkf}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

case "$CMD" in
start)
    for h in "${HOSTS[@]}"; do
        scp -q "$HERE/fleet_node_agent.sh" "$h:~/fleet_node_agent.sh" &
    done
    wait
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "mkdir -p current; chmod +x ~/fleet_node_agent.sh; pkill -f fleet_node_agent.sh 2>/dev/null; setsid nohup ~/fleet_node_agent.sh '$REF' '$ROOTS' '$HUB' > ~/fleet_agent.log 2>&1 < /dev/null &" &
    done
    wait
    echo "release agents running on ${#HOSTS[@]} hosts -> $REF (view: $HUB:current/)"
    ;;
stop)
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "pkill -f fleet_node_agent.sh; true" &
    done
    wait
    echo "agents stopped"
    ;;
status)
    $SSH "$HUB" 'for f in current/*.json; do echo "== $f"; cat "$f"; done' 2>/dev/null \
        || echo "no view on $HUB yet"
    ;;
esac
