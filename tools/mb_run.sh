#!/usr/bin/env bash
# Orchestrate the mb_doorbell allreduce bench across all 16 sparks.
# usage: mb_run.sh <mode:0 async|1 sync> <iters> <rows> [credits]
set -uo pipefail
MODE="${1:?mode 0 async 1 sync}"
ITERS="${2:?iters}"
ROWS="${3:?rows}"
CREDITS="${4:-8}"
TRANSPORT='$HOME/sparkdata/glm53flash.fp8.tp16/lib/hidden_transport.so'
declare -a NODES=(spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf)
TAG="m${MODE}_i${ITERS}_r${ROWS}_c${CREDITS}"

for rank in "${!NODES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 "${NODES[$rank]}" \
        "BENCH_CREDITS=${CREDITS} nohup \$HOME/mb_doorbell ${rank} 16 ${ITERS} ${ROWS} ${MODE} ${TRANSPORT} > /tmp/mb_${TAG}_r${rank}.log 2>&1 < /dev/null &" &
done
wait

deadline=$(( $(date +%s) + 300 ))
while :; do
    done_count=0
    for rank in "${!NODES[@]}"; do
        ssh -o BatchMode=yes -o ConnectTimeout=5 "${NODES[$rank]}" \
            "grep -q '^doorbell rank=' /tmp/mb_${TAG}_r${rank}.log 2>/dev/null" && done_count=$((done_count+1))
    done
    [ "${done_count}" -eq 16 ] && break
    [ "$(date +%s)" -ge "${deadline}" ] && { echo "TIMEOUT done=${done_count}/16"; break; }
    sleep 5
done

for rank in "${!NODES[@]}"; do
    ssh -o BatchMode=yes -o ConnectTimeout=5 "${NODES[$rank]}" \
        "grep -E '^(doorbell|latency|BENCH-BUILD)' /tmp/mb_${TAG}_r${rank}.log | tr '\n' ' '; echo" 2>/dev/null
done
