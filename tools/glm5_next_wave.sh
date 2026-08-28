#!/usr/bin/env bash
# glm5_next_wave.sh — simultaneous all-16 fleet wave for the glm5_next TP16
# deployment. Re-creation of the launch pattern the lane used for waves 1-18
# (the prose LAUNCH-STATE.md was lost to a harness crash; this is the same
# sequence, committed).
#
# Sequence (binding, learned the hard way):
#   1. TERM-kill model daemons on every spark (pkill -x; -f kills the ssh
#      wrapper of this very script), wait for zero cuda processes.
#   2. 45s TIME_WAIT sleep (EADDRINUSE killed wave 2).
#   3. Launch residentd on ALL 16 hosts within the same second — the hidden
#      transport connect window is 180s and late joiners are REJECTED
#      (staggered launch killed every early rank at wave 1).
#   4. Wait for "model_residentd ready" on every host before the api.
#   5. api on the coordinator host (default spark0:8433).
#
# usage: glm5_next_wave.sh [--spark HEX ...] [--api-host spark0] [--api-port 8433]
#                          [--debug-rdma] [--skip-kill] [stop|start|ready|api|full]
set -euo pipefail

ALL_HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
           spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
HOSTS=()
API_HOST=spark0
API_PORT=8433
DEBUG_RDMA=0
SKIP_KILL=0
CMD=full

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spark)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do HOSTS+=("$1"); shift; done ;;
        --api-host)   API_HOST="$2"; shift 2 ;;
        --api-port)   API_PORT="$2"; shift 2 ;;
        --debug-rdma) DEBUG_RDMA=1; shift ;;
        --skip-kill)  SKIP_KILL=1; shift ;;
        stop|start|ready|api|full) CMD="$1"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#HOSTS[@]} -gt 0 ]] || HOSTS=("${ALL_HOSTS[@]}")

runtime_root() { echo "/home/$1/sparkdata/glm5_next.tp16"; }

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }

stop_wave() {
    echo "== TERM model daemons on ${#HOSTS[@]} hosts =="
    for h in "${HOSTS[@]}"; do
        ssh_run "$h" "pkill -x sparkpipe_model 2>/dev/null; true" || true
    done
    for i in $(seq 1 10); do
        alive=0
        for h in "${HOSTS[@]}"; do
            n=$(ssh_run "$h" "pgrep -x sparkpipe_model | wc -l" 2>/dev/null || echo 1)
            alive=$((alive + n))
        done
        [[ $alive -eq 0 ]] && break
        sleep 2
    done
    echo "== verifying zero model processes (comm match, not -f) =="
    for h in "${HOSTS[@]}"; do
        n=$(ssh_run "$h" "pgrep -x sparkpipe_model | wc -l" 2>/dev/null || echo 0)
        [[ "$n" -gt 0 ]] && echo "WARNING: $h still has $n model procs" >&2
    done
    echo "== 45s TIME_WAIT settle =="
    sleep 45
}

start_wave() {
    echo "== simultaneous residentd launch on ${#HOSTS[@]} hosts (debug-rdma=$DEBUG_RDMA) =="
    pids=()
    idx=0
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        env_prefix=""
        [[ $DEBUG_RDMA -eq 1 ]] && env_prefix="SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DEBUG=1 "
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
            "cd '$rr' && rm -f residentd.log && env $env_prefix LD_LIBRARY_PATH='$rr/lib' nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $idx > residentd.log 2>&1 < /dev/null &" </dev/null &
        pids+=($!)
        idx=$((idx + 1))
    done
    wait "${pids[@]}" || true
    echo "launch commands dispatched; polling ready lines (180s transport window)"
}

ready_wave() {
    deadline=$(( $(date +%s) + 180 ))
    while [[ $(date +%s) -lt $deadline ]]; do
        ready=0
        for h in "${HOSTS[@]}"; do
            rr="$(runtime_root "$h")"
            if ssh_run "$h" "grep -q 'model_residentd ready' '$rr/residentd.log' 2>/dev/null"; then
                ready=$((ready + 1))
            fi
        done
        echo "ready: $ready/${#HOSTS[@]}"
        [[ $ready -eq ${#HOSTS[@]} ]] && return 0
        sleep 5
    done
    echo "READY TIMEOUT — per-host state:" >&2
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        echo "-- $h: $(ssh_run "$h" "tail -1 '$rr/residentd.log' 2>/dev/null")" >&2
    done
    return 1
}

start_api() {
    rr="$(runtime_root "$API_HOST")"
    echo "== api on $API_HOST:$API_PORT =="
    ssh_run "$API_HOST" "cd '$rr' && nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$rr' --port $API_PORT > api.log 2>&1 < /dev/null &"
    sleep 3
    ssh_run "$API_HOST" "tail -3 '$rr/api.log'"
}

case "$CMD" in
    stop)  stop_wave ;;
    start) [[ $SKIP_KILL -eq 1 ]] || stop_wave; start_wave; ready_wave ;;
    ready) ready_wave ;;
    api)   start_api ;;
    full)  stop_wave; start_wave; ready_wave && start_api ;;
esac
