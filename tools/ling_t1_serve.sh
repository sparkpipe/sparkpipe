#!/usr/bin/env bash
# ling_t1_serve.sh — lane-scoped serve lifecycle for the LING-T1 window.
# Kills ONLY processes with cwd == this lane's runtime root; syncs the lane
# release root from the build hub; starts 16 residentds with SPARK_LING_T1=1;
# starts the lane api; decodes a prompt. Never touches other lanes' roots,
# production roots, or weightd/weightsd.
# usage: ling_t1_serve.sh [stop|sync|start|api|decode NAME TOKENS NEW|full]
set -uo pipefail
NAME="ling.bf16.tp16.t1ling"
CMD="${1:-full}"
API_HOST="spark0"
API_PORT="18477"
HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
       spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
SSH="ssh -o BatchMode=yes -o ConnectTimeout=5"

rr() { echo "/home/$1/sparkdata/$NAME"; }

stop() {
    local h
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "rr='$(rr "$h")'; for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$l; done; true" &
    done
    wait
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
        echo "TIMEOUT: $NAME daemons still running on:$busy after 30s - investigate, do not start" >&2
        return 1
    fi
    echo "stopped (drained ${t}s)"
}

sync() {
    local h
    for h in "${HOSTS[@]}"; do
        if ! { ssh -o BatchMode=yes "rtx5090" "tar -C ~/release/$NAME -cf - lib bin stages config model_resident.json" \
              | ssh -o BatchMode=yes "$h" "mkdir -p ~/sparkdata/$NAME && tar -C ~/sparkdata/$NAME -xf -"; }; then
            echo "SYNC-FAILED on $h" >&2
            return 1
        fi
    done
    echo "synced to ${#HOSTS[@]} nodes from rtx5090:release/$NAME"
}

start() {
    local h i=0 ready err line deadline n
    for h in "${HOSTS[@]}"; do
        $SSH "$h" "cd '$(rr "$h")' && ln -sf stage_$(printf %02d $i).json config/stage.json && mkdir -p /home/$h/kvcache/ling.bf16.tp16 && mv residentd.log residentd.log.prev 2>/dev/null; LD_LIBRARY_PATH='$(rr "$h")'/lib SPARK_LING_T1='${SPARK_LING_T1:-1}' nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $i > residentd.log 2>&1 < /dev/null &" &
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
                *usage*|*error*|*failed*|*Mismatch*|*status=*) err="$err [$h] $line" ;;
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
    $SSH "$API_HOST" "for l in \$(ls -l /proc/[0-9]*/exe 2>/dev/null | grep sparkpipe_model_api | sed 's|.*/proc/\\([0-9]*\\)/exe.*|\\1|'); do c=\$(readlink /proc/\$l/cwd 2>/dev/null); [ \"\$c\" = \"$(rr "$API_HOST")\" ] && kill -TERM \$l; done; sleep 2; cd '$(rr "$API_HOST")' && LD_LIBRARY_PATH='$(rr "$API_HOST")'/lib setsid nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$(rr "$API_HOST")' --port $API_PORT > api.log 2>&1 < /dev/null &"
    local i line
    for i in $(seq 1 60); do
        line=$($SSH "$API_HOST" "tail -1 '$(rr "$API_HOST")'/api.log 2>/dev/null" 2>/dev/null || true)
        case "$line" in
            *"model_api ready"*) break ;;
            *"api_exit"*) echo "API FAILED:"; $SSH "$API_HOST" "tail -3 '$(rr "$API_HOST")'/api.log"; return 1 ;;
        esac
        sleep 2
    done
    curl -s --max-time 5 "http://$API_HOST:$API_PORT/health" || true
    echo
}

decode() {
    local name="${2:?decode NAME TOKENS NEW}"
    local tokens="${3:?prompt token ids csv}"
    local new_tokens="${4:-4}"
    $SSH "$API_HOST" "curl -s --max-time 300 -X POST http://localhost:$API_PORT/v1/completions -H 'Content-Type: application/json' -d '{\"prompt_token_ids\":[$tokens],\"max_tokens\":$new_tokens}'"
    echo
}

case "$CMD" in
    stop) stop ;;
    sync) sync ;;
    start) start ;;
    api) api ;;
    decode) decode "$@" ;;
    full|*) stop && sync && "$0" start && api ;;
esac
