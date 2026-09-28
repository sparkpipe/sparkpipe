#!/bin/bash
# qwen38_max timed warm receipt (lane 2 M3): the three cold-launch numbers
# on the shared weightd plus BOTH working-set byte bases (fleet standard
# after the 2026-09-22 chunk-basis correction — this family's measured
# factor is 1.4032..1.4444 over the exact 2 MiB chunk union; receipts
# carry the numbers on both bases instead of a factor).
#
#   1. daemon cold-warm    - first warm after a daemon restart: lazy-arena
#                            create + pool-chunk materialization dominate
#                            (the once-per-daemon cost)
#   2. warm-daemon preload - a cold-launching instance's batch preload when
#                            the daemon already holds the set resident:
#                            WSET-WARM elapsed_ms. THE < 5 s claim.
#   3. steady-state        - repeated preloads against the warm daemon
#
# Bases: smoke_set_raw_bytes_per_node (exact sidecar span bytes of this
# rank's smoke experts) and smoke_set_chunked_bytes_per_node (exact 2 MiB
# chunk union — the committed smoke_experts_pools.json basis), reported
# by tools/qwen38max_verify_pools.py from THIS node's placed pack.
# Only the lazy/partial pool tier pays chunking; the pin-all whole-arena
# tier does not (runtime/spark_weightd.c: pool chunk = max(gpu allocation
# granularity, 2 MiB); 2 MiB measured on sm_121a).
#
# Identity: the default positional path (socket pack sha revision world),
# the same invocation tools/qwen38max_multidev_run_family.sh's warm leg
# uses — weightd_warm's --family derivation is dsv4_pro-only.
#
# Queue invocation (bare repo-resident path):
#   --cmd 'bash tools/devcycle/qwen38max_warm_receipt.sh'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${SPARK_WEIGHTD_SOCKET:-/tmp/spark_weightd.sock}"
RUNS="${QMAX_WARM_RUNS:-5}"
POOL="${QMAX_EXPERT_POOL_BYTES:-0}"   # 0 = this rank's chunked default

HOST="$(hostname)"
case "$HOST" in
spark[0-9a-f]) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
RANK="$((16#${HOST#spark}))"
PACK="/home/$HOST/sparkdata/qwenmax.nvfp4.tp16/packs/qwenmax.nvfp4.tp16.rank${HOST#spark}.sp"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
# The pinned serving revision (the adapter matches it EXACTLY): lane.py's
# MODEL_REVISION constant, the same value the deployment render carries in
# adapter.json (the firmware example has no model_revision member).
REVISION="$(sed -n 's/^MODEL_REVISION = "\(.*\)"$/\1/p' \
  "$REPO/tools/qwen38max_multidev_lane.py")"
[ "${#REVISION}" -eq 40 ] || { echo "bad MODEL_REVISION parse" >&2; exit 2; }
WSET="$(mktemp -q /tmp/qmax-wset.XXXXXX)"
trap 'rm -f "$WSET"' EXIT

for required in "$SOCKET" "$PACK" "$PACK.sha256" "$PACK.experts"; do
  [ -e "$required" ] || { echo "missing: $required" >&2; exit 2; }
done

cd "$REPO"
python3 tools/qwen38max_multidev_lane.py --emit-wset "$WSET"
if [ "$POOL" = 0 ]; then
  POOL="$(python3 tools/qwen38max_multidev_lane.py \
    --budgets model-families/qwen38_max/smoke_experts.json --rank "$RANK")"
  POOL="${POOL%% *}"   # first field = chunk-basis pool bytes for this rank
fi
make -s build/weightd_warm

echo "== qwen38_max warm receipt (rank $RANK, $PACK)"
echo "== pool=$POOL runs=$RUNS"
echo "== bases (measured on this node's placed pack):"
python3 tools/qwen38max_verify_pools.py --rank "$RANK" \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("smoke_set_raw_bytes_per_node=%d" % d["raw"]); print("smoke_set_chunked_bytes_per_node=%d" % d["chunked"]); print("smoke_experts=%d spans=%d" % (d["experts"], d["spans"]))'

echo "== timed warm (run 1 = daemon-cold-warm, run 2 = warm-daemon preload"
echo "   [the < 5 s claim], runs 3+ = steady-state)"
echo "   RUN-WALL = whole invocation (connect + attach/arena-create +"
echo "   acquire + release + close); WSET-WARM elapsed_ms = the resident"
echo "   keys acquire+release; attach leg = RUN-WALL - WSET-WARM."
index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
    build/weightd_warm "$SOCKET" "$PACK" "$SHA" "$REVISION" 16 \
    --wset "$WSET" 300
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))"
  index=$((index + 1))
done
echo "WARM-RECEIPT-DONE rank=$RANK pool=$POOL runs=$RUNS"
