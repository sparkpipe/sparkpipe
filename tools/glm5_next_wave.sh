#!/usr/bin/env bash
# glm5_next_wave.sh — simultaneous all-16 fleet wave for the glm5_next TP16
# deployment. Re-creation of the launch pattern the lane used for waves 1-18
# (the prose LAUNCH-STATE.md was lost to a harness crash; this is the same
# sequence, committed).
#
# Sequence (binding, learned the hard way):
#   0. REGISTRAR PHASE (docs/FLEET_STARTUP_PROTOCOL.md phases 1+1b): launch
#      the featherweight sparkpipe_registrar on ALL hosts simultaneously.
#      They announce/merge until the three-level ready condition holds
#      (all see all + every previous daemon TERMed-gone by deployment cwd),
#      then GO. The registrars themselves TERM stale daemons (TERM only,
#      never KILL, exact cwd+exe match); a TERM-immune daemon fails the
#      phase LOUDLY with STALE-IMMUNE: rank->pid. Skip: --skip-registrar.
#   1. TERM-kill model daemons on every spark (pkill -x; -f kills the ssh
#      wrapper of this very script), wait for zero cuda processes. Post-GO
#      this verifies the cleanslate and catches the api.
#   2. 75s TIME_WAIT sleep (EADDRINUSE killed wave 2; 45s was the classic
#      number, +30s headroom since cleanslate TERMs land mid-registrar-phase).
#   3. Launch residentd on ALL 16 hosts within the same second — the hidden
#      transport connect window is 180s and late joiners are REJECTED
#      (staggered launch killed every early rank at wave 1).
#   4. Wait for "model_residentd ready" on every host before the api.
#   5. api on the coordinator host (default spark0:8433).
#
# Stage the registrar binary first (compile once on spark0, copy to all):
#   tools/registrar_stage.sh
#
# usage: glm5_next_wave.sh [--spark HEX ...] [--api-host spark0] [--api-port 8433]
#                          [--debug-rdma] [--probe] [--skip-kill] [--skip-registrar]
#                          [stop|start|ready|api|full|registrars]
set -euo pipefail

ALL_HOSTS=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7
           spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
HOSTS=()
API_HOST=spark0
API_PORT=8433
DEBUG_RDMA=0
PROBE=0
PROBE_VEC=0
SKIP_KILL=0
SKIP_REGISTRAR=0
REGISTRAR_PORT_BASE=22480
REGISTRAR_TIMEOUT_MS=120000
CMD=full

while [[ $# -gt 0 ]]; do
    case "$1" in
        --spark)      shift; while [[ $# -gt 0 && "$1" != --* ]]; do HOSTS+=("$1"); shift; done ;;
        --api-host)   API_HOST="$2"; shift 2 ;;
        --api-port)   API_PORT="$2"; shift 2 ;;
        --api-port)   API_PORT="$2"; shift 2 ;;
        --debug-rdma) DEBUG_RDMA=1; shift ;;
        --probe)      PROBE=1; shift ;;
        --probe-vec)  PROBE=1; PROBE_VEC=1; shift ;;
        --skip-kill)  SKIP_KILL=1; shift ;;
        --skip-registrar) SKIP_REGISTRAR=1; shift ;;
        stop|start|ready|api|full|registrars) CMD="$1"; shift ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#HOSTS[@]} -gt 0 ]] || HOSTS=("${ALL_HOSTS[@]}")

runtime_root() { echo "/home/$1/sparkdata/glm5_next.tp16"; }

ssh_run() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$1" "$2"; }

registrar_phase() {
    echo "== registrar phase on ${#HOSTS[@]} hosts (GO = three-level ready + cleanslate) =="
    local hosts_csv=""
    for h in "${HOSTS[@]}"; do hosts_csv="$hosts_csv$h,"; done
    hosts_csv="${hosts_csv%,}"
    local pids=()
    local idx=0
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
            "cd '$rr' && test -x bin/sparkpipe_registrar || { echo 'MISSING bin/sparkpipe_registrar — run tools/registrar_stage.sh' >&2; exit 87; }; rm -f registrar.log; nohup ./bin/sparkpipe_registrar --rank $idx --hosts '$hosts_csv' --port-base $REGISTRAR_PORT_BASE --timeout-ms $REGISTRAR_TIMEOUT_MS --deployment-cwd '$rr' > registrar.log 2>&1 < /dev/null &" </dev/null &
        pids+=($!)
        idx=$((idx + 1))
    done
    wait "${pids[@]}" || true
    echo "registrars dispatched; waiting for GO or the loud failure (timeout ${REGISTRAR_TIMEOUT_MS}ms + grace)"
    local deadline=$(( $(date +%s) + REGISTRAR_TIMEOUT_MS / 1000 + 30 ))
    local running
    while [[ $(date +%s) -lt $deadline ]]; do
        running=0
        for h in "${HOSTS[@]}"; do
            local n
            n=$(ssh_run "$h" "pgrep -c -f 'sparkpipe_registrar --rank' 2>/dev/null || echo 0" || echo 0)
            running=$((running + n))
        done
        [[ $running -eq 0 ]] && break
        sleep 2
    done
    local rc=0 status
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        status=$(ssh_run "$h" "grep -h 'registrar exit' '$rr/registrar.log' 2>/dev/null | tail -1" || echo "registrar exit status=unknown")
        echo "$h: $status"
        echo "$status" | grep -q "status=0" || rc=1
    done
    if [[ $rc -ne 0 ]]; then
        echo "REGISTRAR PHASE FAILED — per-host diffs:" >&2
        for h in "${HOSTS[@]}"; do
            rr="$(runtime_root "$h")"
            echo "-- $h:" >&2
            ssh_run "$h" "grep -h 'REGISTRAR' '$rr/registrar.log' 2>/dev/null | tail -4" >&2 || true
        done
        return 1
    fi
    echo "== GO: membership exists and the fleet is cleanslate =="
}

stop_wave() {
    echo "== TERM model daemons on ${#HOSTS[@]} hosts (cwd-scoped) =="
    # CWD-SCOPED TERM ONLY: pkill -x sparkpipe_model also hits OTHER lanes'
    # daemons sharing a node (the spark5 dry2 / sparke K3 incidents). Match
    # the residentd/api by exe path + exact deployment cwd, TERM, never KILL.
    # The [r]/[a] bracket keeps pgrep -f from matching this very shell.
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        ssh_run "$h" "rr='$rr'; for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; for p in \$(pgrep -f 'bin/sparkpipe_model_[a]pi'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; true" || true
    done
    for i in $(seq 1 10); do
        alive=0
        for h in "${HOSTS[@]}"; do
            rr="$(runtime_root "$h")"
            n=$(ssh_run "$h" "rr='$rr'; a=0; for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && a=\$((a+1)); done; for p in \$(pgrep -f 'bin/sparkpipe_model_[a]pi'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && a=\$((a+1)); done; echo \$a" 2>/dev/null || echo 1)
            alive=$((alive + n))
        done
        [[ $alive -eq 0 ]] && break
        sleep 2
    done
    echo "== verifying zero model processes (cwd-scoped, not -x) =="
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        n=$(ssh_run "$h" "rr='$rr'; a=0; for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && a=\$((a+1)); done; echo \$a" 2>/dev/null || echo 0)
        [[ "$n" -gt 0 ]] && echo "WARNING: $h still has $n model procs" >&2
    done
    sleep 75
    sleep 45
}

start_wave() {
    echo "== simultaneous residentd launch on ${#HOSTS[@]} hosts (debug-rdma=$DEBUG_RDMA probe=$PROBE) =="
    pids=()
    idx=0
    for h in "${HOSTS[@]}"; do
        rr="$(runtime_root "$h")"
        env_prefix=""
        [[ $DEBUG_RDMA -eq 1 ]] && env_prefix="SPARKPIPE_HIDDEN_SPARK_HOST_RDMA_DEBUG=1 "
        # NCCL fabric pins (required when the collective backend is nccl):
        # unpinned NCCL picks the wrong RoCE port (10.10.200.x) and dies
        # post-bootstrap - receipts NCCL_16WIDE_RECEIPTS.md
        env_prefix="NCCL_SOCKET_IFNAME=enp1s0f1np1 NCCL_IB_HCA=rocep1s0f1 NCCL_IB_GID_INDEX=3 $env_prefix"
        # probe=$PROBE arms the G5N-PROBE diag ladder fleet-wide; the probe
        # build scales the TP connect window itself (SPARK_GLM5_NEXT_PROBE,
        # lane/probe-fix) so the open deadline no longer expires while the
        # ladder-armed ranks are still coming up.
        [[ $PROBE -eq 1 ]] && env_prefix="${env_prefix}SPARK_GLM5_NEXT_PROBE=1 "
        # --probe-vec implies --probe and arms the G5N-VEC full-vector dumps
        # (glm5-attractor lane, layer-0 KDA stages + head, diag only); the
        # pass cap is overridable (G5N_VEC_PASSES) - a 176-token prompt needs
        # >= prompt_len+decode passes for the oracle's zero-state tracking.
        # G5N_VEC_DSA=1 additionally arms the glm5-dsa lane's DSA-site dumps
        # (layer G5N_VEC_LAYER, default 3: HC site stages + the whole MLA
        # chain) alongside the L0 KDA dumps.
        [[ $PROBE_VEC -eq 1 ]] && env_prefix="${env_prefix}SPARK_GLM5_NEXT_PROBE_VEC=1 SPARK_GLM5_NEXT_PROBE_VEC_PASSES=${G5N_VEC_PASSES:-30} SPARK_GLM5_NEXT_PROBE_VEC_DSA=${G5N_VEC_DSA:-0} SPARK_GLM5_NEXT_PROBE_VEC_LAYER=${G5N_VEC_LAYER:-3} SPARK_GLM5_NEXT_PROBE_VEC_KDA_LAYER=${G5N_VEC_KDA_LAYER:-0} "
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$h" \
            "cd '$rr' && mv residentd.log residentd.log.prev-\$(date +%s) 2>/dev/null || true; env $env_prefix LD_LIBRARY_PATH='$rr/lib' nohup ./bin/sparkpipe_model_residentd --deployment model_resident.json --rank-index $idx > residentd.log 2>&1 < /dev/null &" </dev/null &
        pids+=($!)
        idx=$((idx + 1))
    done
    wait "${pids[@]}" || true
    echo "launch commands dispatched; polling ready lines (180s transport window)"
}

ready_wave() {
    # Under the probe the ladder-armed ranks may legitimately take longer to
    # become ready (rank 0 loads the embedding+head pack), and the probe build
    # scales the connect window to 720s: a 180s poll would call a healthy
    # probe wave dead.
    local wait_s=180
    [[ $PROBE -eq 1 ]] && wait_s=780
    deadline=$(( $(date +%s) + wait_s ))
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
    # setsid + full fd detach: the plain `nohup ... &` form left the launch
    # ssh holding the session open for 31 minutes (the api inherits the
    # sshd-side pipe unless every fd is redirected BEFORE backgrounding),
    # which set -e then turned into WAVE-FAIL (glm5-dsa lane, wave3).
    ssh_run "$API_HOST" "cd '$rr' && setsid nohup ./bin/sparkpipe_model_api --deployment model_resident.json --runtime-root '$rr' --port $API_PORT > api.log 2>&1 < /dev/null &" </dev/null
    sleep 3
    ssh_run "$API_HOST" "tail -3 '$rr/api.log'"
}

case "$CMD" in
    stop)  stop_wave ;;
    registrars) registrar_phase ;;
    start)
        [[ $SKIP_REGISTRAR -eq 1 ]] || registrar_phase || exit 1
        [[ $SKIP_KILL -eq 1 ]] || stop_wave
        start_wave; ready_wave ;;
    ready) ready_wave ;;
    api)   start_api ;;
    full)
        [[ $SKIP_REGISTRAR -eq 1 ]] || registrar_phase || exit 1
        [[ $SKIP_KILL -eq 1 ]] || stop_wave
        start_wave && ready_wave && start_api ;;
esac
