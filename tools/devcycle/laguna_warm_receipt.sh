#!/bin/bash
# laguna timed warm receipt (lane 8 M3): the three cold-launch numbers on
# the shared weightd plus BOTH working-set byte bases (fleet standard
# after the 2026-09-22 chunk-basis correction; laguna's per-expert spans
# are small - w1 1.5 MiB, w2 0.75 MiB - so the 2 MiB lazy-pool chunk
# rounding is material; receipts carry the numbers on both bases, never
# a factor).
#
#   1. daemon cold-warm    - first warm of THIS pack's arena after a
#                            daemon restart: lazy-arena create + pool
#                            chunk materialization dominate (the
#                            once-per-daemon cost)
#   2. warm-daemon preload - a cold-launching instance's batch preload
#                            when the daemon already holds the set
#                            resident: WSET-WARM elapsed_ms. THE < 5 s
#                            claim.
#   3. steady-state        - repeated preloads against the warm daemon
#
# Laguna deltas vs the qwen38_max lane-2 receipt: the placed packs carry
# NO .experts sidecars, so this receipt generates the v2 sidecar first
# (tools/laguna_multidev_experts_manifest.sh - directory parse + ck128
# digests over the expert spans, idempotent) and the pack path follows
# the GLOBAL-rank filename convention (stage{rank/8}.rank{rank}.lgsp).
#
# Identity: the default positional path (socket pack sha revision world),
# the same invocation tools/laguna_multidev_run_family.sh's warm leg
# uses - weightd_warm's --family derivation is dsv4_pro-only.
#
# Queue invocation (bare repo-resident path):
#   --cmd 'bash tools/devcycle/laguna_warm_receipt.sh'
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${LAGUNA_WEIGHTD_SOCKET:-/run/sparkpipe-weightd-shared/weightd.sock}"
RUNS="${LAGUNA_WARM_RUNS:-5}"
POOL="${LAGUNA_EXPERT_POOL_BYTES:-0}"   # 0 = this rank's chunked default

HOST="$(hostname)"
case "$HOST" in
spark[0-9a-f]) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
RANK="$((16#${HOST#spark}))"
STAGE="$((RANK / 8))"
PACK="/home/$HOST/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/laguna_stage.tp8.pp2.stage${STAGE}.rank${RANK}.lgsp"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
REVISION="$(python3 -c 'import json;print(json.load(open("'"$REPO"'/examples/model_descriptions/laguna_resident_decode_stage_firmware.json"))["model"]["revision"])')"
WSET="$(mktemp -q /tmp/laguna-wset.XXXXXX)"
trap 'rm -f "$WSET" "$WSET.chunk."*' EXIT

for required in "$SOCKET" "$PACK" "$PACK.sha256"; do
  [ -e "$required" ] || { echo "missing: $required" >&2; exit 2; }
done

cd "$REPO"
# The placed packs ship without .experts sidecars; generate (idempotent -
# an existing v2 sidecar is left untouched). First run digests the pack's
# expert spans (~13.9 GiB NVMe read, one time per node).
bash tools/laguna_multidev_experts_manifest.sh "$PACK"

# Per-rank filtered wset (weightd_warm rejects pairs outside this pack's
# manifest), split into <=512-pair chunks (SPARK_WEIGHTD_LEASE_GROUPS_MAX;
# 512 pairs x 8 B = 4096 B exactly) warmed sequentially - the GLM
# pin-experts chunked-lease precedent. A "run" below = the full chunk
# sequence, so RUN-WALL stays the honest whole-invocation number.
python3 tools/laguna_multidev_lane.py --emit-wset "$WSET" --rank "$RANK"
rm -f "$WSET.chunk."*
split -b 4096 -d "$WSET" "$WSET.chunk."
CHUNKS="$(ls "$WSET.chunk."* | sort)"
if [ "$POOL" = 0 ]; then
  # The daemon charges the WHOLE pack file in 2 MiB chunks against the
  # declared pool (ACQUIRE-LOAD-STAGE stage=budget; lane-4 rank-3 found
  # the same law) - the pool default is the pack chunk basis, not the
  # smoke subset. A stale arena created with an under-sized pool blocks
  # the correctly-sized one until reclaimed:
  #   build/weightd_warm SOCKET --reclaim
  BUDGETS="$(python3 tools/laguna_multidev_lane.py --budgets "$PACK")"
  POOL="${BUDGETS%% *}"   # first field = whole-pack chunk-basis pool bytes
fi
make -s build/weightd_warm

echo "== laguna warm receipt (rank $RANK, $PACK)"
echo "== pool=$POOL runs=$RUNS (pool = whole-pack 2 MiB chunk basis;"
echo "   the daemon's acquire charges the full pack footprint)"
echo "== working-set bases (this rank's smoke head pairs, both bases):"
python3 tools/laguna_multidev_lane.py \
  --smoke-budgets model-families/laguna/smoke_experts.json "$RANK" \
  | python3 -c 'import sys; raw, chunked = sys.stdin.read().split(); print("smoke_set_raw_bytes_per_node=%s" % raw); print("smoke_set_chunked_bytes_per_node=%s" % chunked)'

echo "== timed warm (run 1 = daemon-cold-warm for this pack, run 2 ="
echo "   warm-daemon preload [the < 5 s claim], runs 3+ = steady-state)"
echo "   RUN-WALL = whole invocation (connect + attach/arena-create +"
echo "   acquire + release + close); WSET-WARM elapsed_ms = the resident"
echo "   keys acquire+release; attach leg = RUN-WALL - WSET-WARM."
index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  for chunk in $CHUNKS; do
    SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
      build/weightd_warm "$SOCKET" "$PACK" "$SHA" "$REVISION" 16 \
      --wset "$chunk" 300
  done
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))"
  index=$((index + 1))
done
echo "WARM-RECEIPT-DONE rank=$RANK pool=$POOL runs=$RUNS"
