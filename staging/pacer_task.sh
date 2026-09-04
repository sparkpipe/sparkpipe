#!/bin/bash
# glm53 PACER probe v2 — queue-owned serving experiment (runs on nodes[0]).
# Three waves, same 3-row probe (FORCE_WAVE_ROWS=3 = the divergence case):
#   A: no fence      (expected WRONG per handoff — negative control)
#   B: PACER_KB=96   (the untested rank-0 D2H fence — the question)
#   C: LAYERDUMP=1   (known-CORRECT oracle per handoff)
# Verdict: B==C (and B!=A) => small D2H fence suffices => stopgap viable.
# weightds are PRE-WARMED by the prewarm task; setup only verifies sockets.
# --skip-registrar: the two 09-01 zombies are TERM-immune and would fail the
# registrar's cleanslate gate; they hold no ports after reboot cleanup.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
NODES="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
WAVE=/home/spark5/glm5_next_wave.sh
WAVE_ARGS="--spark spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf --api-host spark0"
ROWS_ENV="SPARK_GLM5_NEXT_FORCE_WAVE_ROWS=3"
PROBE='{"prompt_token_ids":[151644,872,198],"max_tokens":12,"temperature":0}'
LOG=/tmp/pacer_task.log
: > "$LOG"

rr() { echo "/home/$1/sparkdata/glm5_next.tp16"; }

cleanup() {
    for h in $NODES; do
        timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "rr=/home/$h/sparkdata/glm5_next.tp16; for p in \$(pgrep -f 'bin/sparkpipe_model_[r]esidentd'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; for p in \$(pgrep -f 'bin/sparkpipe_model_[a]pi'); do c=\$(readlink /proc/\$p/cwd 2>/dev/null); [ \"\$c\" = \"\$rr\" ] && kill -TERM \$p; done; true" &
    done
    wait
}

setup() {
    local missing=0
    for h in $NODES; do
        timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" \
            "[ -S /tmp/spark_weightd.sock ]" || { echo "MISSING weightd socket: $h" >> "$LOG"; missing=$((missing+1)); }
    done
    [ "$missing" -eq 0 ] && echo "warm-check: 16/16 weightd sockets present" >> "$LOG"
    return "$missing"
}

wait_ready() {
    local deadline=$(( $(date +%s) + ${PHASE_WAIT:-500} ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
        local n=0
        for h in $NODES; do
            timeout 12 ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" \
                "grep -q 'model_residentd ready' $(rr "$h")/residentd.log 2>/dev/null" && n=$((n+1))
        done
        echo "ready: $n/16" >> "$LOG"
        [ "$n" -eq 16 ] && return 0
        sleep 15
    done
    return 1
}

probe() {
    local out=/tmp/probe_$1.json
    for i in $(seq 1 60); do
        ssh -o BatchMode=yes -o ConnectTimeout=6 spark0 \
            "curl -s --max-time 5 http://localhost:8433/health" 2>/dev/null | grep -q '"ok"' && break
        sleep 3
    done
    ssh -o BatchMode=yes -o ConnectTimeout=8 spark0 \
        "curl -s --max-time 300 -X POST http://localhost:8433/v1/completions -H 'Content-Type: application/json' -d '$PROBE'" \
        > "$out" 2>/dev/null
    echo "PROBE $1: $(head -c 400 "$out")" >> "$LOG"
}

wave() {
    cleanup
    sleep 10
    G5N_EXTRA_ENV="$2" bash "$WAVE" $WAVE_ARGS --skip-registrar full >> "$LOG" 2>&1
    if wait_ready; then
        probe "$1"
    else
        echo "WAVE $1: READY TIMEOUT" >> "$LOG"
        for h in spark0 spark5 spark8; do
            echo "-- $h: $(ssh -o BatchMode=yes -o ConnectTimeout=6 "$h" "grep -vE 'tp completion' $(rr "$h")/residentd.log 2>/dev/null | tail -2" 2>/dev/null)" >> "$LOG"
        done
    fi
}

PHASE=${1:-A}
case "$PHASE" in
    A) setup || echo "setup: proceeding despite missing sockets (cold fallback)" >> "$LOG"
       wave A "$ROWS_ENV" ;;
    B) wave B "$ROWS_ENV SPARK_GLM5_NEXT_PACER_KB=96" ;;
    C) wave C "$ROWS_ENV SPARK_GLM5_NEXT_LAYERDUMP=1"
       echo "=== VERDICT ===" >> "$LOG"
       for L in A B C; do echo "$L: $(python3 -c "import json;print(json.load(open('/tmp/probe_$L.json')).get('tokens','PARSE-FAIL'))" 2>/dev/null)" >> "$LOG"; done
       cleanup ;;
esac
echo "PHASE-$PHASE-DONE" >> "$LOG"
