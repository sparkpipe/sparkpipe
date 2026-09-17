#!/usr/bin/env bash
# fleet_sync.sh — install/start/stop the per-node release agents on the
# fleet. Each node pulls its runtime roots from the reference base itself
# (parallel, no fan-out), restarts a root only on the UPDATE sentinel
# (down:/up: ledger -> TERM -> start), and reports running versions to
# hub:current/. The reference base is the manifest: build into it, the
# fleet converges.
#
# usage: tools/fleet_sync.sh REFERENCE_BASE ROOTS_CSV [start|stop|status|retire-update]
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
        $SSH "$h" "mkdir -p current ~/.config/systemd/user; chmod +x ~/fleet_node_agent.sh; printf '[Unit]\nDescription=fleet release agent\nAfter=network-online.target\n\n[Service]\nExecStart=%s/fleet_node_agent.sh %s %s %s\nRestart=always\nRestartSec=3\n\n[Install]\nWantedBy=default.target\n' \"\$HOME\" '$REF' '$ROOTS' '$HUB' > ~/.config/systemd/user/fleet-agent.service; systemctl --user daemon-reload; systemctl --user enable fleet-agent.service; systemctl --user restart fleet-agent.service" &
    done
    wait
    echo "release agents running on ${#HOSTS[@]} hosts -> $REF (view: $HUB:current/)"
    ;;
stop)
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "systemctl --user stop fleet-agent.service; rm -f ~/.fleet_agent.pid; true" &
    done
    wait
    echo "agents stopped"
    ;;
status)
    $SSH "$HUB" 'for f in current/*.json; do echo "== $f"; cat "$f"; done' 2>/dev/null \
        || echo "no view on $HUB yet"
    ;;
retire-update)
    # weightd UPDATE sentinel holds the target sha16; retire it from the
    # hub only once every node's current/<host>.json reports that sha.
    refhost="${REF%%:*}" refdir="${REF#*:}"
    target=$($SSH "$refhost" "cat '$refdir/weightd/UPDATE'" 2>/dev/null | tr -d '[:space:]')
    [ -n "$target" ] || { echo "no weightd UPDATE on $REF (nothing to retire)"; exit 1; }
    missing=""
    for h in "${HOSTS[@]}"; do
        sha=$($SSH "$HUB" "grep -o '\"weightd\":\"[^\"]*\"' 'current/$h.json' 2>/dev/null | cut -d'\"' -f4" 2>/dev/null)
        [ "$sha" = "$target" ] || missing="$missing $h(${sha:-no-report})"
    done
    [ -z "$missing" ] || { echo "NOT retiring:$missing not on $target" >&2; exit 1; }
    $SSH "$refhost" "rm '$refdir/weightd/UPDATE'" \
        && echo "retired weightd UPDATE ($target): all ${#HOSTS[@]} nodes report it"
    ;;
esac
