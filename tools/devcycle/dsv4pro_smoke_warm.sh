#!/bin/bash
# dsv4_pro smoke working-set warm (lane 5 M3): derive this node's exact
# module attach identity, print it, then warm the committed smoke expert
# set through the shared weightd socket. Queue job (repo-resident, bare
# invocation - the dispatcher pre-expands $ patterns in inline cmd text).
#
# Env:
#   SPARK_WEIGHTD_SOCKET   the shared daemon socket (required)
#   FAMILY_WSET            .wset path (default: the committed smoke set)
#   FAMILY_POOL_BYTES      expert pool bound (default 4 GiB: the 179-key
#                          smoke set is 1497.7 MiB of expert bytes, but the
#                          daemon materializes at 2 MiB chunk granularity -
#                          payload spans round 2.75 MiB -> 4 MiB and each
#                          168 KiB scale span holds a full chunk - ~18 MiB
#                          chunked per expert per rank, ~3.2 GiB total)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOCKET="${SPARK_WEIGHTD_SOCKET:?shared weightd socket required}"
WSET="${FAMILY_WSET:-$REPO/model-families/dsv4/smoke-standard-v1.wset}"
POOL="${FAMILY_POOL_BYTES:-4294967296}"

HOST="$(hostname)"
case "$HOST" in
spark*) RANK=$((16#${HOST#spark})) ;;
*) echo "unexpected host: $HOST" >&2; exit 2 ;;
esac
ROOT="/home/$HOST/sparkdata/dsv4_pro.tp4pp4"
PACK="$ROOT/packs/dsv4_pro_tp4_pp4_stage.spstage"
SHA="$(cut -d' ' -f1 "$PACK.sha256")"

cd "$REPO"
make -s build/weightd_warm

echo "== identity (rank $RANK, $PACK)"
build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
  --family dsv4_pro --world-rank "$RANK" --identity-print

echo "== warm $(basename "$WSET") pool=$POOL"
SPARK_WEIGHTD_EXPERT_POOL_BYTES="$POOL" \
build/weightd_warm "$SOCKET" "$PACK" "$SHA" x 16 \
  --family dsv4_pro --world-rank "$RANK" --wset "$WSET" 300
