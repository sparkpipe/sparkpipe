#!/usr/bin/env bash
#
# fleet_swap.sh MODEL — designate the current big model on the fleet.
#
# The registry (tools/devcycle/fleet_registry.json) assigns every model a
# tier ("always-on" | "big"), a scope ("band" = its own hosts,
# "fleet" = all 16 sparks), hosts, ports, and its runtime root.
#
# Semantics:
#   - "band" models are mutually exclusive among themselves: swapping in
#     a band model stops the current band model and starts the new one.
#     Always-on models (qwen27b on spark0-3, dsv4-flash on spark4-7) are
#     NOT touched unless the swap is fleet-scoped.
#   - "fleet" models (dsv4-pro, k3) evict everything: the script snapshots
#     which models were running, stops all of them, and starts the fleet
#     model. Swapping a fleet model OUT restores the snapshot.
#   - State lives at /tmp/sparkpipe_fleet_state.json on every spark; the
#     authoritative copy is on the first host (spark0) and is broadcast.
#
# usage: tools/fleet_swap.sh <model_id>
#        tools/fleet_swap.sh status
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY="$SCRIPT_DIR/devcycle/fleet_registry.json"
STATE_REMOTE="/tmp/sparkpipe_fleet_state.json"
ALL_HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
           spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
STATE_HOST="spark0"

die() { echo "fleet_swap: ERROR: $*" >&2; exit 1; }

registry_hosts() { jq -r ".models[\"$1\"].hosts[]" "$REGISTRY"; }
registry_field() { jq -r ".models[\"$1\"].$2" "$REGISTRY"; }

remote_state() {
    ssh -o BatchMode=yes "$STATE_HOST" "cat '$STATE_REMOTE' 2>/dev/null" 2>/dev/null         || echo '{"schema_version":1,"current_big":null,"running":{}}'
}

broadcast_state() {
    local state="$1" host
    for host in "${ALL_HOSTS[@]}"; do
        printf '%s' "$state" | ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "cat > '$STATE_REMOTE'"             || echo "fleet_swap: $host unreachable, state broadcast skipped"
    done
}

stop_model() {
    local model="$1"
    echo "fleet_swap: stopping $model"
    registry_hosts "$model" | while read -r host; do
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$host"             "sudo systemctl stop sparkpipe_model_residentd 2>/dev/null || true"             || echo "fleet_swap: $host unreachable, stop skipped"
    done
    sleep 2
}

start_model() {
    local model="$1" runtime rank=0
    echo "fleet_swap: starting $model"
    registry_hosts "$model" | while read -r host; do
        runtime="$(registry_field "$model" runtime_root | sed "s/{host}/$host/")"
        ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" "
            test -x '$runtime/bin/sparkpipe_model_residentd' \
                || { echo "missing residentd for $model on $host: $runtime" >&2; exit 1; }
            sudo mkdir -p /etc/sparkpipe /etc/systemd/system/sparkpipe_model_residentd.service.d
            printf 'RUNTIME_ROOT=%s\nRANK_INDEX=%s\nMODEL=%s\nHOST=%s\nLD_LIBRARY_PATH=%s/lib\n' '$runtime' '$rank' '$model' '$host' '$runtime' \
                | sudo tee /etc/sparkpipe/residentd.env >/dev/null
            printf '[Service]\nUser=%s\n' \"\$(whoami)\" \
                | sudo tee /etc/systemd/system/sparkpipe_model_residentd.service.d/10-user.conf >/dev/null
            sudo systemctl daemon-reload
            sudo systemctl start sparkpipe_model_residentd
        " || die "start $model on $host failed"
        rank=$((rank + 1))
    done
}

cmd_status() {
    local state
    state="$(remote_state)"
    echo "current big model: $(printf '%s' "$state" | jq -r '.current_big')"
    echo "running: $(printf '%s' "$state" | jq -c '.running')"
}

cmd_swap() {
    local model="$1" state current scope
    jq -e ".models[\"$model\"]" "$REGISTRY" >/dev/null || die "unknown model '$model'"
    state="$(remote_state)"
    current="$(printf '%s' "$state" | jq -r '.current_big')"
    scope="$(registry_field "$model" scope)"

    if [[ "$current" == "$model" ]]; then
        echo "fleet_swap: $model already current; no-op"
        return 0
    fi

    if [[ "$scope" == "fleet" ]]; then
        # snapshot everything currently running for later restore
        printf '%s' "$state" | jq --arg m "$model" \
            '{schema_version:.schema_version,current_big:$m,
              running:{prev:(.current_big//"none"),prev_state:.running}}'             > /tmp/fleet-new-state.json
        # evict all residentds on every host
        for host in "${ALL_HOSTS[@]}"; do
            ssh -o BatchMode=yes -o ConnectTimeout=8 "$host"                 "sudo systemctl stop sparkpipe_model_residentd 2>/dev/null || true"                 || echo "fleet_swap: $host unreachable, evict skipped"
        done
        sleep 2
        start_model "$model"
    else
        # band swap: stop current big only, start the new one
        if [[ "$current" != "null" && "$current" != "" ]]; then
            stop_model "$current"
        fi
        start_model "$model"
        printf '%s' "$state" | jq --arg m "$model" \
            '{schema_version:.schema_version,current_big:$m,running:.running}'             > /tmp/fleet-new-state.json
    fi

    broadcast_state "$(cat /tmp/fleet-new-state.json)"
    echo "fleet_swap: $model is now the current big model"
}

case "${1:-}" in
    status) cmd_status ;;
    *)      [[ -n "${1:-}" ]] || { echo "usage: $0 MODEL|status" >&2; exit 2; }
            cmd_swap "$1" ;;
esac
