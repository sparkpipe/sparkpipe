#!/usr/bin/env bash
# fleet_sync.sh — install/start/stop the per-node release agents on the
# fleet. Each node pulls its runtime roots from the HTTP hub itself
# (parallel, no fan-out), restarts a root only on a MANIFEST diff, and
# reports running versions to the hub's current/ view. The MANIFEST is
# the version: build into the hub release base, and the fleet converges.
#
# usage: tools/fleet_sync.sh ROOTS_CSV [start|stop|status] [HUB]
#   ROOTS_CSV  comma-separated runtime root names the agent manages,
#              e.g. glm53flash.fp8.tp16
#   HUB        defaults to $FLEET_HUB, then spec@100.123.97.61 (the
#              rtx5090 release hub; release files serve over :8802)
#
# The unit runs the agent from ~/sparkdata/core/bin/fleet_node_agent.sh —
# the path sync_core updates and self_update execs. Installing or running
# it from any other path defeats self-update.
set -uo pipefail
ROOTS="${1:?comma-separated runtime root names}"
CMD="${2:-start}"
HUB="${3:-${FLEET_HUB:-spec@100.123.97.61}}"
HERE="$(cd "$(dirname "$0")" && pwd)"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

case "$CMD" in
start)
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "mkdir -p ~/sparkdata/core/bin" && \
        scp -q "$HERE/fleet_node_agent.sh" \
            "$h:sparkdata/core/bin/fleet_node_agent.sh" &
    done
    wait
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "chmod 755 ~/sparkdata/core/bin/fleet_node_agent.sh; \
            mkdir -p ~/.config/systemd/user; \
            printf '[Unit]\nDescription=fleet release agent\nAfter=network-online.target\n\n[Service]\nExecStart=%%h/sparkdata/core/bin/fleet_node_agent.sh %s %s\nRestart=always\nRestartSec=3\n\n[Install]\nWantedBy=default.target\n' \
              '$ROOTS' '$HUB' > ~/.config/systemd/user/fleet-agent.service; \
            systemctl --user daemon-reload; \
            systemctl --user enable fleet-agent.service; \
            systemctl --user restart fleet-agent.service" &
    done
    wait
    echo "release agents running on ${#HOSTS[@]} hosts, roots=$ROOTS, hub=$HUB"
    ;;
stop)
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "systemctl --user stop fleet-agent.service; rm -f ~/.fleet_agent.pid; true" &
    done
    wait
    echo "agents stopped"
    ;;
status)
    $SSH "$HUB" 'for f in current/*.json; do
        h=${f##*/}; h=${h%.json}
        printf "%-8s wd=%s agent=%s %s\n" "$h" \
          "$(grep -o "\"weightd\":\"[^\"]*\"" $f | cut -d\" -f4)" \
          "$(grep -o "\"agent\":\"[^\"]*\"" $f | cut -d\" -f4)" \
          "$(grep -o "\"roots\":{[^}]*}" $f | head -c 200)"
      done' 2>/dev/null || echo "no view on $HUB yet"
    ;;
*)
    echo "unknown command: $CMD (start|stop|status)" >&2
    exit 2
    ;;
esac
