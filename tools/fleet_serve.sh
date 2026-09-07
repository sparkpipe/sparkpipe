#!/usr/bin/env bash
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
        $SSH "$h" "rr='$(rr "$h")'; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$l; done; sleep 1; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -KILL \$l; done; true" &
    done
    wait
    local t busy
    for t in $(seq 1 30); do
        busy=0
        for h in "${HOSTS[@]}"; do
            n=$($SSH "$h" "ls -l /proc/[0-9]*/exe 2>/dev/null | grep -c sparkpipe_model" 2>/dev/null)
            [ "${n:-0}" -gt 0 ] && busy=1
        done
        [ "$busy" = 0 ] && break
        sleep 2
    done
    echo "stopped (drained ${t}x2s)"
}

start() {
    local h i=0 ready err line deadline
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "cd '$(rr "$h")' && ln -sf stage_$(printf %02d $i).json config/stage.json && mv residentd.log residentd.log.prev 2>/dev/null; LD_LIBRARY_PATH='$(rr "$h")'/lib nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $i > residentd.log 2>&1 < /dev/null &" &
        i=$((i+1))
    done
    wait
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
    $SSH "$API_HOST" "cd '$(rr "$API_HOST")' && setsid nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$(rr "$API_HOST")' --port $API_PORT > api.log 2>&1 < /dev/null &"
    sleep 1
    curl -s --max-time 3 "http://$API_HOST:$API_PORT/health" || true
    echo
}

sync() {
    local h p REF="${FLEET_REF:-rtx5090:release}"
    local HUBHOST="${REF%%:*}" HUBPATH="${REF#*:}"
    for h in "${HOSTS[@]}"; do
        for p in lib bin stages config model_resident.json; do
            { ssh -o BatchMode=yes "$HUBHOST" "tar -C '$HUBPATH/$NAME' -cf - '$p' --exclude=stage.json" \
              | ssh -o BatchMode=yes "$h" "tar -C ~/sparkdata/$NAME -xf -"; } &
        done
    done
    wait
    echo "synced from $REF"
}

case "$CMD" in
    stop) stop ;;
    start) start ;;
    sync) sync ;;
    api) api ;;
    full|*) sync && "$0" "$NAME" start && api ;;
esac
