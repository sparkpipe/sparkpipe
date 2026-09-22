#!/bin/bash
# One-time per-pack .ck128 sidecar stamp (glm5_next lane 0).
#
# The release shared weightd requires a whole-pack ck128 sidecar next to the
# pack (runtime/spark_weightd.c SparkWeightdSidecarCk128 -> NOT_FOUND without
# it; observed as lazy-attach status=3/4 in glm-m3-cold-a5). The qualified
# campaign ran a private daemon that predated the requirement, so the NVMe
# family packs carry .sha256/.experts but no .ck128. Stamp each node's own
# rank pack once; idempotent (existing sidecar -> print and exit 0).
#
# Queue job: --per-node --resources cpu --memory-mib 512 --ttl-min 5
# Runs from a synced lane checkout (repo-owned command, no shell syntax in
# the queue cmd beyond invoking this script).

set -euo pipefail

HOST="$(hostname)"
suffix="${HOST#spark}"
FAMILY_ROOT="/home/$HOST/sparkdata/glm53flash.fp8.tp16"
CHECKOUT="$(pwd)"
PACK="$FAMILY_ROOT/packs/glm53flash.fp8.tp16.rank$suffix.sp"

[ -f "$PACK" ] || { echo "pack missing: $PACK" >&2; exit 2; }
if [ -f "$PACK.ck128" ]; then
  echo "already stamped rank=$suffix ck128=$(cat "$PACK.ck128")"
  exit 0
fi
cc -O2 -I "$CHECKOUT/include" "$CHECKOUT/tools/ck128_stamp.c" "$CHECKOUT/src/spark_ck128.c" -o /tmp/ck128_stamp.$$
trap 'rm -f /tmp/ck128_stamp.$$' EXIT
/tmp/ck128_stamp.$$ "$PACK"
echo "stamped rank=$suffix bytes=$(stat -c %s "$PACK") ck128=$(cat "$PACK.ck128")"
