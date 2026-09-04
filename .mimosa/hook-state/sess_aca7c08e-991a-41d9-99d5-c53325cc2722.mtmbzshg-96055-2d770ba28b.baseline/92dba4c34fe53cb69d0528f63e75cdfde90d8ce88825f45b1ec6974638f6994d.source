#!/usr/bin/env bash
# glm5_next closeout C3: one-owner fleet wave — stop, settle, relaunch, api.
#
# Implements the LAUNCH-STATE.md patterns verbatim (the working form from
# the kda-lane waves):
#   STOP  = cwd-scoped TERM: TERM only pids whose /proc/<pid>/cwd is the
#           glm5_next.tp16 runtime root — never -f, never other lanes'
#           processes (spark5 runs dry-template2 under /tmp/dry2-*;
#           sparke runs the K3 build).
#   START = all 16 residentds in the same second, setsid + nohup + session
#           hold (DECIMAL --rank-index), then the api on spark0:8433.
#
# usage: glm5_next_closeout_wave.sh stop|start|ready|api|full
set -u
CMD="${1:-full}"
API_HOST=spark0
API_PORT=8433

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }
rr_for() { echo "/home/$1/sparkdata/glm5_next.tp16"; }

stop_wave() {
    echo "== cwd-scoped TERM of glm5_next residentds (16 hosts) =="
    for i in $(seq 0 15); do
        h=$(printf "spark%x" "$i")
        ssh_run "$h" "rr=$(rr_for "$h"); for p in \$(pgrep -x sparkpipe_model); do \
            c=\$(readlink /proc/\$p/cwd 2>/dev/null); \
            [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; exit 0"
    done
    echo "== waiting for zero glm5 glm5_next procs (collective teardown) =="
    for try in $(seq 1 60); do
        alive=0
        for i in $(seq 0 15); do
            h=$(printf "spark%x" "$i")
            n=$(ssh_run "$h" "rr=$(rr_for "$h"); a=0; for p in \$(pgrep -x sparkpipe_model); do \
                c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && a=\$((a+1)); done; echo \$a" \
                2>/dev/null || echo 1)
            alive=$((alive + n))
        done
        echo "  try $try: $alive still alive"
        [[ $alive -eq 0 ]] && break
        sleep 3
    done
    if [[ $alive -ne 0 ]]; then
        echo "TERM-IGNORED: $alive glm5_next procs survived TERM after 180s." >&2
        echo "NO-KILL PROTOCOL: capture /proc/<pid>/stack + status + top -H," >&2
        echo "leave running, report the node. Do NOT escalate." >&2
        return 1
    fi
    echo "== 45s TIME_WAIT settle =="
    sleep 45
}

start_wave() {
    echo "== simultaneous residentd launch (16 ranks, one wave) =="
    pids=()
    for i in $(seq 0 15); do
        h=$(printf "spark%x" "$i")
        rr=$(rr_for "$h")
        ssh "$h" "cd $rr && rm -f residentd.log && setsid nohup env \
            LD_LIBRARY_PATH=$rr/lib ./bin/sparkpipe_model_residentd --deployment model_resident.json \
            --rank-index $i > residentd.log 2>&1 < /dev/null & sleep 1" &
        pids+=($!)
    done
    wait "${pids[@]}" || true
    echo "launch ssh fan-out returned (may hold past 2 min); polling ready"
}

ready_wave() {
    local deadline=$(( $(date +%s) + 240 ))
    while [[ $(date +%s) -lt $deadline ]]; do
        ready=0
        for i in $(seq 0 15); do
            h=$(printf "spark%x" "$i")
            rr=$(rr_for "$h")
            if ssh_run "$h" "grep -q 'model_residentd ready' '$rr/residentd.log' 2>/dev/null"; then
                ready=$((ready + 1))
            fi
        done
        echo "ready: $ready/16"
        [[ $ready -eq 16 ]] && return 0
        sleep 5
    done
    echo "READY TIMEOUT — per-host tails:" >&2
    for i in $(seq 0 15); do
        h=$(printf "spark%x" "$i")
        rr=$(rr_for "$h")
        echo "-- $h: $(ssh_run "$h" "tail -1 '$rr/residentd.log' 2>/dev/null")" >&2
    done
    return 1
}

start_api() {
    rr=$(rr_for "$API_HOST")
    echo "== api on $API_HOST:$API_PORT =="
    ssh_run "$API_HOST" "cd $rr && setsid nohup ./bin/sparkpipe_model_api \
        --deployment model_resident.json --runtime-root . --port $API_PORT \
        > api.log 2>&1 < /dev/null & sleep 1"
    sleep 10
    ssh_run "$API_HOST" "tail -3 '$rr/api.log'"
}

case "$CMD" in
    stop)  stop_wave ;;
    start) start_wave; ready_wave ;;
    ready) ready_wave ;;
    api)   start_api ;;
    full)  stop_wave && start_wave && ready_wave && start_api ;;
    *) echo "usage: $0 stop|start|ready|api|full" >&2; exit 2 ;;
esac
