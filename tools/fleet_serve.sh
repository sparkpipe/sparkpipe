#!/usr/bin/env bash
# fleet_serve.sh — relaunch a resident deployment on the fleet. The daemons
# are load-order-independent (background accepts, retrying connects) and
# listeners set SO_REUSEADDR: TERM in parallel -> same-second launch ->
# ready-or-error poll that fails in seconds -> api.
# One instance per root, ever: full stops first, start refuses if running.
#
# usage: tools/fleet_serve.sh RUNTIME_ROOT_NAME [stop|start|api|full] [--force-kill]
#        (default full; G5_API_HOST is required, the API never runs on a Spark by default)
set -uo pipefail
NAME="${1:?runtime root name (under ~/sparkdata/)}"
CMD="${2:-full}"
FORCE_KILL=0
[ "${3:-}" = "--force-kill" ] && FORCE_KILL=1
API_HOST="${G5_API_HOST:?G5_API_HOST is required: the API host is not a Spark by default}"
API_PORT="${G5_API_PORT:-8433}"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"
LANE_TOOL=fleet_serve
LANE_RANKS=${#HOSTS[@]}
. "$(cd "$(dirname "$0")" && pwd)/lane_run_log.sh"

rr() { echo "/home/$1/sparkdata/$NAME"; }
host_of() { echo "${HOSTS[$1]}"; }
root_of() { rr "$1"; }

stop() {
    local h
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "rr='$(rr "$h")'; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$l; done; true" &
    done
    wait
    if [ "$FORCE_KILL" = 1 ]; then
        sleep 1
        for h in "${HOSTS[@]}"; do
            $SSH "$h" "rr='$(rr "$h")'; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -KILL \$l; done; true" &
        done
        wait
    fi
    local t n busy
    for t in $(seq 1 30); do
        busy=""
        for h in "${HOSTS[@]}"; do
            n=$($SSH "$h" "rr='$(rr "$h")'; n=0; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && n=\$((n+1)); done; echo \$n" 2>/dev/null)
            if [ -z "$n" ]; then busy="$busy $h(unreachable)"
            elif [ "$n" -gt 0 ]; then busy="$busy $h"; fi
        done
        [ -z "$busy" ] && break
        sleep 1
    done
    if [ -n "$busy" ]; then
        echo "TIMEOUT: $NAME daemons still running on:$busy after 30s — investigate, do not start" >&2
        return 1
    fi
    echo "stopped (drained ${t}s)"
}

start() {
    local h i=0 ready err line deadline n busy="" run_id pid refused=""
    local -a pids=()
    run_id="$(lane_run_id "${FLEET_RUN_ID:-}")" || return 2
    for h in "${HOSTS[@]}"; do
        n=$($SSH "$h" "rr='$(rr "$h")'; n=0; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && n=1; done; echo \$n" 2>/dev/null)
        if [ -z "$n" ]; then busy="$busy $h(unreachable)"
        elif [ "$n" = 1 ]; then busy="$busy $h"; fi
    done
    if [ -n "$busy" ]; then
        echo "REFUSE: $NAME daemons already running on:$busy — stop first (one instance per root)" >&2
        return 1
    fi
    echo "fleet_serve: run $run_id; rank logs are $(lane_log_file "$run_id") under each runtime root"
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "cd '$(rr "$h")' && ln -sf stage_$(printf %02d $i).json config/stage.json && $(lane_log_prelude "$h" "$run_id") && LD_LIBRARY_PATH='$(rr "$h")'/lib ${SPARK_GLM5_NEXT_MTP:+SPARK_GLM5_NEXT_MTP=$SPARK_GLM5_NEXT_MTP} SPARK_TP_D2A_TIMING='${SPARK_TP_D2A_TIMING:-0}' SPARK_GLM52_SERVING_FLAT_RANKS='${SPARK_GLM52_SERVING_FLAT_RANKS-}' nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $i > $(lane_log_file "$run_id") 2>&1 < /dev/null &" &
        pids+=("$!")
        i=$((i+1))
    done
    i=0
    for pid in "${pids[@]}"; do
        wait "$pid" || refused="$refused ${HOSTS[$i]}"
        i=$((i+1))
    done
    if [ -n "$refused" ]; then
        echo "FAIL: run $run_id did not start on:$refused" >&2
        return 1
    fi
    deadline=$((SECONDS + 300))
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
    $SSH "$API_HOST" "cd '$(rr "$API_HOST")' && ${SPARK_GLM5_NEXT_MTP:+SPARK_GLM5_NEXT_MTP=$SPARK_GLM5_NEXT_MTP} setsid nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$(rr "$API_HOST")' --port $API_PORT > api.log 2>&1 < /dev/null &"
    sleep 1
    curl -s --max-time 3 "http://$API_HOST:$API_PORT/health" || true
    echo
}

sync() {
    local h p REF="${FLEET_REF:-rtx5090:release}"
    local HUBHOST="${REF%%:*}" HUBPATH="${REF#*:}"
    for h in "${HOSTS[@]}"; do
        if ! { ssh -o BatchMode=yes "$HUBHOST" "tar -C '$HUBPATH/$NAME' --exclude=stage.json -cf - lib bin stages config model_resident.json" \
              | ssh -o BatchMode=yes "$h" "tar -C ~/sparkdata/$NAME -xf -"; }; then
            echo "SYNC-FAILED on $h" >&2
            return 1
        fi
    done
    echo "synced from $REF"
}

case "$CMD" in
    stop) stop ;;
    start) start ;;
    sync) sync ;;
    api) api ;;
    archive) lane_archive "${3:?archive RUN_ID DESTINATION}" "${4:?archive RUN_ID DESTINATION}" ;;
    full|*) stop && sync && "$0" "$NAME" start && api ;;
esac
