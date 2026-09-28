#!/bin/bash
set -euo pipefail

fail() { echo "laguna_warm_receipt: $*" >&2; exit 2; }
decimal() { [[ "$1" =~ ^[0-9]+$ ]]; }

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

SOCKET="${LAGUNA_WEIGHTD_SOCKET:-}"
case "$SOCKET" in
  /*) ;;
  *) fail "LAGUNA_WEIGHTD_SOCKET must name the running weightd socket by absolute path (got '$SOCKET')" ;;
esac
[ -S "$SOCKET" ] || fail "weightd socket $SOCKET is not a live socket; this receipt never starts a weightd"

LANE="${LAGUNA_LANE:-}"
decimal "$LANE" && [ "$LANE" -le 15 ] || fail "LAGUNA_LANE must be a weightd mesh lane 0..15 (got '$LANE')"

RUNS="${LAGUNA_WARM_RUNS:-5}"
decimal "$RUNS" && [ "$RUNS" -gt 0 ] || fail "LAGUNA_WARM_RUNS must be a positive decimal"
POOL="${LAGUNA_EXPERT_POOL_BYTES:-}"
[ -z "$POOL" ] || { decimal "$POOL" && [ "$POOL" -gt 0 ]; } || fail "LAGUNA_EXPERT_POOL_BYTES must be a positive decimal"

HOST="$(hostname)"
case "$HOST" in
  spark[0-9a-f]) ;;
  *) fail "unexpected host: $HOST" ;;
esac
RANK="$((16#${HOST#spark}))"
STAGE="$((RANK / 8))"
PACK="/home/$HOST/sparkdata/laguna-s-2.1.bf16.tp8pp2/packs/laguna_stage.tp8.pp2.stage${STAGE}.rank${RANK}.lgsp"
for required in "$PACK" "$PACK.sha256"; do
  [ -e "$required" ] || fail "missing: $required"
done
SHA="$(cut -d' ' -f1 "$PACK.sha256")"
REVISION="$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["model"]["revision"])' "$REPO/examples/model_descriptions/laguna_resident_decode_stage_bf16_firmware.json")"
WSET="$(mktemp -q /tmp/laguna-wset.XXXXXX)"
trap 'rm -f "$WSET" "$WSET.chunk."*' EXIT

cd "$REPO"
bash tools/laguna_multidev_experts_manifest.sh "$PACK"

python3 tools/laguna_multidev_lane.py --emit-wset "$WSET" --rank "$RANK"
rm -f "$WSET.chunk."*
split -b 4096 -d "$WSET" "$WSET.chunk."
CHUNKS="$(ls "$WSET.chunk."* | sort)"
if [ -z "$POOL" ]; then
  BUDGETS="$(python3 tools/laguna_multidev_lane.py --budgets "$PACK")"
  POOL="${BUDGETS%% *}"
fi
make -s build/weightd_warm

echo "== laguna warm receipt (rank $RANK, lane $LANE, socket $SOCKET, $PACK)"
echo "== pool=$POOL runs=$RUNS"
python3 tools/laguna_multidev_lane.py \
  --smoke-budgets model-families/laguna/smoke_experts.json "$RANK" \
  | python3 -c 'import sys; raw, chunked = sys.stdin.read().split(); print("smoke_set_raw_bytes_per_node=%s" % raw); print("smoke_set_chunked_bytes_per_node=%s" % chunked)'

index=1
while [ "$index" -le "$RUNS" ]; do
  echo "-- warm run $index/$RUNS"
  began_ns=$(date +%s%N)
  for chunk in $CHUNKS; do
    SPARK_WEIGHTD_LANE="$LANE" SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
      build/weightd_warm "$SOCKET" "$PACK" "$SHA" "$REVISION" 16 \
      --wset "$chunk" 300
  done
  ended_ns=$(date +%s%N)
  printf 'RUN-WALL run=%s elapsed_ms=%s\n' "$index" \
    "$(( (ended_ns - began_ns) / 1000000 ))"
  index=$((index + 1))
done
echo "WARM-RECEIPT-DONE rank=$RANK lane=$LANE pool=$POOL runs=$RUNS"
