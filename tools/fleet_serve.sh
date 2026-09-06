#!/usr/bin/env bash
# fleet_serve.sh — relaunch a resident deployment on the fleet. The daemons
# are load-order-independent (background accepts, retrying connects) and
# listeners set SO_REUSEADDR: TERM in parallel -> same-second launch ->
# ready-or-error poll that fails in seconds -> api.
#
# usage: tools/fleet_serve.sh RUNTIME_ROOT_NAME [stop|start|api|full]
#        (default full; api host spark0 port 8433 unless G5_API_HOST/PORT)
set -uo pipefail
NAME="${1:?runtime root name (under ~/sparkdata/)}"
CMD="${2:-full}"
API_HOST="${G5_API_HOST:-spark0}"
API_PORT="${G5_API_PORT:-8433}"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

rr() { echo "/home/$1/sparkdata/$NAME"; }

stop() {
    local h
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "rr='$(rr "$h")'; for p in \$(pgrep -f 'bin/sparkpipe_model_(residentd|api)'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; true" &
    done
    wait
    echo "stopped"
}

start() {
    local h i=0 ready err line deadline
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "cd '$(rr "$h")' && mv residentd.log residentd.log.prev 2>/dev/null; LD_LIBRARY_PATH='$(rr "$h")'/lib nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $i > residentd.log 2>&1 < /dev/null &" &
        i=$((i+1))
    done
    wait
    deadline=$((SECONDS + 120))
    while (( SECONDS < deadline )); do
        ready=0
        err=""
        for h in "${HOSTS[@]}"; do
            line=$($SSH "$h" "tail -1 '$(rr "$h")'/residentd.log 2>/dev/null" 2>/dev/null || true)
            case "$line" in
                *"model_residentd ready"*) ready=$((ready+1)) ;;
                *usage*|*error*|*failed*|*Mismatch*) err="$err [$h] $line" ;;
            esac
        done
        if [ -n "$err" ]; then
            echo "FAIL-FAST:${err}"
            return 1
        fi
        if [ "$ready" -eq "${#HOSTS[@]}" ]; then
            echo "ready ${#HOSTS[@]}/${#HOSTS[@]} in ${SECONDS}s"
            return 0
        fi
        sleep 1
    done
    echo "TIMEOUT not-ready (see residentd.log per host)"
    return 1
}

api() {
    $SSH "$API_HOST" "cd '$(rr "$API_HOST")' && setsid nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$(rr "$API_HOST")' --port $API_PORT > api.log 2>&1 < /dev/null &"
    sleep 1
    curl -s --max-time 3 "http://$API_HOST:$API_PORT/health" || true
    echo
}

case "$CMD" in
    stop) stop ;;
    start) start ;;
    api) api ;;
    full|*) stop && start && api ;;
esac
