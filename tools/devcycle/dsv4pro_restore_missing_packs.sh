#!/bin/bash
# dsv4_pro slice-direct restore driver (lane 5): rebuild the missing rank
# packs from the Ceph GA generation through the manager-booked window,
# reading ONLY the needed layer slices (no 865G full-pack intermediate).
#
# Approved plan: ranks 1/2/3 (stage 0, layers [0,16)) + rank 6 (stage 1,
# layers [16,15)). FAIL-CLOSED: every produced pack is sha256-verified
# against the round-10 placement table before anything moves; a mismatch
# leaves the workdir untouched and exits nonzero (bisect the toolchain to
# the b8211ae1 era sharder before retrying - the current sharder has
# drifted since round 10).
#
# Runs as a per-node queue CPU job from a synced checkout. The Ceph read
# happens in step 1 (the slice pack build) - inside the booked window
# only. NVMe-only: WORKDIR must be on the node's sparkdata NVMe.
#
# Queue invocation (bare repo-resident path):
#   --cmd 'bash tools/devcycle/dsv4pro_restore_missing_packs.sh'
# Env:
#   RESTORE_STAGE    0 (ranks 1,2,3) or 1 (rank 6)          (required)
#   RESTORE_MODEL    GA model dir on Ceph                    (required)
#   RESTORE_WORKDIR  NVMe workdir for slice + rank packs     (required)
#   RESTORE_RANKS    tp ranks to shard this run, overriding the stage
#                    default (disk-pipelined windows: shard one rank,
#                    ship + verify + delete, repeat; the slice pack is
#                    reused across runs)
#   RESTORE_PLACE    "1" to place verified packs into the
#                    runtime packs/ dir (default: verify only)
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STAGE="${RESTORE_STAGE:?set RESTORE_STAGE to 0 or 1}"
MODEL="${RESTORE_MODEL:?set RESTORE_MODEL to the Ceph GA model dir}"
WORK="${RESTORE_WORKDIR:?set RESTORE_WORKDIR to an NVMe workdir}"
PLACE="${RESTORE_PLACE:-0}"

HOST="$(hostname)"
ROOT="/home/$HOST/sparkdata/dsv4_pro.tp4pp4"
PACKS="$ROOT/packs"

# Round-10 placement table (tools/devcycle/dsv4_pro_single_spark_receipts.md;
# also pinned in tools/devcycle/dsv4pro_multidev_fleet_prep.sh).
expected_sha() {
  case "$1" in
  1) echo 0e9f015877bde1400c2e36b352e646b0792c645e09f3014597f9cbbaefb75512 ;;
  2) echo 24821d5736da788cc1177a9265f9463f86a6e434491c90ba3b3b512138122418 ;;
  3) echo 197348a90316e1b3652e07a3330d818c3440204d93e2fa57855bd3ff787e0f7 ;;
  6) echo a50eb96b3a40decb24454fceab9414492da78c01bf1ab7fd5cd6e4e82bbc4332 ;;
  *) return 1 ;;
  esac
}

case "$STAGE" in
0) FIRST=0; COUNT=16; RANKS="${RESTORE_RANKS:-1,2,3}" ;;
1) FIRST=16; COUNT=15; RANKS="${RESTORE_RANKS:-2}" ;;
*) echo "RESTORE_STAGE must be 0 or 1" >&2; exit 2 ;;
esac
WORLD_RANKS=""
for R in ${RANKS//,/ }; do
  case "$STAGE.$R" in
  0.1|0.2|0.3|1.2) WORLD_RANKS="$WORLD_RANKS $((STAGE * 4 + R))" ;;
  *) echo "tp rank $R does not belong to stage $STAGE" >&2; exit 2 ;;
  esac
done
WORLD_RANKS="${WORLD_RANKS# }"

mkdir -p "$WORK"
SLICE="$WORK/dsv4_pro.slice${FIRST}.spstage"

echo "== restore stage $STAGE: layers $FIRST+$COUNT, world ranks $WORLD_RANKS"
echo "== model: $MODEL"
echo "== work: $WORK"

[ -d "$MODEL" ] || { echo "model dir missing: $MODEL" >&2; exit 2; }
case "$WORK" in
/home/*/sparkdata/*) : ;;
*) echo "WORKDIR must live under the node's sparkdata NVMe: $WORK" >&2; exit 2 ;;
esac

echo "== step 1/3: slice pack (the Ceph generation read - window only)"
if [ -f "$SLICE" ]; then
  echo "slice pack already present, keeping: $SLICE"
else
  python3 "$REPO/tools/dsv4_pro_stagepack.py" \
    --model-dir "$MODEL" \
    --first-layer "$FIRST" --layer-count "$COUNT" \
    --output "$SLICE"
fi

echo "== step 2/3: shard rank packs from the slice"
python3 "$REPO/tools/devcycle/dsv4pro_restore_ranks.py" \
  --input "$SLICE" --stage "$STAGE" --ranks "$RANKS" \
  --output-root "$WORK"

echo "== step 3/3: digest-compare vs the round-10 table (fail-closed)"
FAILED=0
for WORLD in $WORLD_RANKS; do
  PACK="$WORK/dsv4_pro.tp4_pp4.rank$(printf '%02d' "$WORLD").spstage"
  WANT="$(expected_sha "$WORLD")"
  [ -f "$PACK" ] || { echo "MISSING rank $WORLD pack: $PACK" >&2; FAILED=1; continue; }
  GOT="$(sha256sum "$PACK" | cut -d' ' -f1)"
  if [ "$GOT" = "$WANT" ]; then
    echo "rank $WORLD: DIGEST-MATCH $GOT"
  else
    echo "rank $WORLD: DIGEST-MISMATCH" >&2
    echo "  expected $WANT" >&2
    echo "  produced $GOT" >&2
    FAILED=1
  fi
done
[ "$FAILED" -eq 0 ] || {
  echo "RESTORE-VERIFY-FAILED: no pack placed; toolchain bisect to the" >&2
  echo "b8211ae1 era sharder required before retrying" >&2
  exit 5; }

echo "RESTORE-VERIFY-PASS stage=$STAGE ranks=$WORLD_RANKS"

if [ "$PLACE" = "1" ]; then
  for WORLD in $WORLD_RANKS; do
    PACK="$WORK/dsv4_pro.tp4_pp4.rank$(printf '%02d' "$WORLD").spstage"
    TARGET_HOST="spark$(printf '%x' "$WORLD")"
    if [ "$TARGET_HOST" = "$HOST" ]; then
      mkdir -p "$PACKS"
      [ -e "$PACKS/$(basename "$PACK")" ] && {
        echo "refusing to overwrite existing pack: $PACKS/$(basename "$PACK")" >&2
        exit 6; }
      mv "$PACK" "$PACKS/"
      echo "placed rank $WORLD: $PACKS/$(basename "$PACK")"
    else
      echo "rank $WORLD belongs on $TARGET_HOST (this is $HOST);" \
           "ship or re-run there with RESTORE_PLACE=1" >&2
      exit 7
    fi
  done
  echo "placed packs still need sidecars + uniform link: run"
  echo "tools/devcycle/dsv4pro_multidev_fleet_prep.sh on each restored node"
else
  echo "verify-only mode (RESTORE_PLACE=0): packs remain in $WORK"
fi
