#!/usr/bin/env bash
# Orchestrate the mb_doorbell allreduce bench across all 16 sparks.
# usage: mb_run.sh <mode:0 async|1 sync> <iters> <rows> [credits] [d2a_max_bytes]
set -euo pipefail
MODE="${1:?mode 0 async 1 sync}"
ITERS="${2:?iters}"
ROWS="${3:?rows}"
CREDITS="${4:-8}"
D2A="${5:-0}"
PORT_BASE="${BENCH_PORT_BASE:-61000}"
SKEW="${BENCH_SKEW_US:-0}"
JUMP="${BENCH_ORDINAL_BASE_JUMP:-0}"
: "${BENCH_ID:?set a unique shared-queue job ID}"
: "${BENCH_CWD:?set the cwd template printed by spark_queue.py sync}"
for value in "$MODE" "$ITERS" "$ROWS" "$CREDITS" "$D2A" "$PORT_BASE" "$SKEW" "$JUMP"; do
    [[ "$value" =~ ^[0-9]+$ ]] || { echo "benchmark arguments must be nonnegative integers" >&2; exit 2; }
done
printf -v command 'set -eu; git rev-parse HEAD; sha256sum ./build/mb_doorbell ./build/libhidden_transport_spark_host_rdma_verbs.so; export BENCH_CREDITS=%q BENCH_D2A_MAX_BYTES=%q BENCH_PORT_BASE=%q BENCH_SKEW_US=%q BENCH_ORDINAL_BASE_JUMP=%q; exec ./build/mb_doorbell "$SPARK_QUEUE_RANK" "$SPARK_QUEUE_SIZE" %q %q %q ./build/libhidden_transport_spark_host_rdma_verbs.so' "$CREDITS" "$D2A" "$PORT_BASE" "$SKEW" "$JUMP" "$ITERS" "$ROWS" "$MODE"
exec python3 "$(dirname "$0")/spark_queue.py" add --id "$BENCH_ID" \
    --nodes spark0,spark1,spark2,spark3,spark4,spark5,spark6,spark7,spark8,spark9,sparka,sparkb,sparkc,sparkd,sparke,sparkf \
    --per-node --resources exclusive --memory-mib 8192 --ttl-min 5 \
    --cwd "$BENCH_CWD" --cmd "$command"
