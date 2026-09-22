#!/usr/bin/env bash
# qwen38max_multidev_pack_emit.sh — emit + verify + place ONE qwen38_max
# TP16 rank pack for the shared-socket lane (multidev M2/M3).
#
# Runs INSIDE one per-node queue job on the synced checkout (CPU
# resources; dollar-free queue cmd — systemd $-expansion, the lane-6
# finding). With --nodes spark0,..,sparkf --per-node, rank i runs on
# spark{hex(i)} and emits its own rank pack from the warm checkpoint
# through that host's warm client into that host's local NVMe:
#
#   /home/<host>/sparkdata/qwen38max.tp16/packs/qwen38max.tp16-rank<i>.qwen38sp
#
# Chain (the spark3 precedent of tools/qwen38max_rebuild_rank.sh, adapted
# to any host with no sparkcap — the queue cgroup already bounds us):
#   1. tools/qwen38_stagepack.py      nvfp4 TP16 shard from the warm
#                                     checkpoint (config/index sha
#                                     fail-closed against the T1 fixture
#                                     MANIFEST identity)
#   2. tools/qwen38max_tp16_rank_verify.py  byte-exact rank verification
#                                     with --recompute-file-hash receipt
#   3. tools/qwen38max_multidev_experts_manifest.sh  the .experts sidecar
#                                     the shared weightd lazy attach
#                                     fails closed without
#   4. sha256sum sidecar (exactly one digest; weightd resolves pack
#                                     identity from it)
# All artifacts land under a .partial.$$ namespace and are renamed into
# place atomically at the end; a completed rank is verified-and-skipped
# (idempotent re-runs are safe).
#
# Required environment: the standard SPARK_QUEUE_* contract only - the
# queue cmd stays BARE (no env prefixes; systemd-run pre-expands $VAR).
# The emitted rank derives from the hostname (identity mesh), so the
# fleet form (--nodes spark0,..,sparkf --per-node) and a targeted
# single-node run on sparkN both emit world rank N without any env.
# Optional: QMAX_EMIT_CHECKPOINT (default the recorded warm checkpoint).
set -euo pipefail

# ----------------------------- FAMILY PARAMETERS -----------------------------

WORLD=16
CHECKPOINT_DEFAULT="/mnt/model-warm/qwen3.8-max-nvfp4-radixark-bf16-spine"
DEST_REL="sparkdata/qwenmax.nvfp4.tp16/packs"
LAYER_COUNT=92

fail() { echo "qwen38max-pack-emit: $*" >&2; exit 1; }

# ------------------------------ QUEUE CONTRACT -------------------------------

ATTEMPT="${SPARK_QUEUE_ATTEMPT:?run through the authoritative spark queue}"
case "$ATTEMPT" in
  *[!0-9a-f]*|""|?????????????????????????????????*) fail "bad attempt id" ;;
esac
[ "${#ATTEMPT}" -eq 32 ] || fail "bad attempt id length"
# Rank derives from the HOSTNAME (identity mesh: rank i lives on
# spark{hex(i)}), so the queue cmd stays BARE - no env prefixes or other
# shell syntax (systemd-run pre-expands $VAR; lane-5/lane-6 findings).
# The fleet form cross-checks the queue rank against it.
HOST="$(hostname)"
case "$HOST" in
  spark[0-9a-f]) ;;
  *) fail "cannot derive rank: unexpected hostname '$HOST'" ;;
esac
HEXDigit="${HOST#spark}"
RANK="$((16#$HEXDigit))"
[ "$RANK" -ge 0 ] && [ "$RANK" -lt "$WORLD" ] ||
  fail "rank out of range from hostname '$HOST'"
if [ "${SPARK_QUEUE_SIZE:-}" = "$WORLD" ]; then
  [ "${SPARK_QUEUE_RANK:-}" = "$RANK" ] ||
    fail "queue rank '${SPARK_QUEUE_RANK:-}' disagrees with host rank $RANK ($HOST)"
elif [ "${SPARK_QUEUE_SIZE:-}" = "1" ]; then
  : # targeted single-node emission of this host's own rank
else
  fail "SPARK_QUEUE_SIZE must be $WORLD or 1 (got '${SPARK_QUEUE_SIZE:-}')"
fi

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
CHECKPOINT="${QMAX_EMIT_CHECKPOINT:-$CHECKPOINT_DEFAULT}"
DEST="/home/$HOST/$DEST_REL"
PACK="$DEST/qwenmax.nvfp4.tp16.rank$RANK.sp"
MANIFEST="$CHECKOUT/qualification/t1_reference/qwen38_max/MANIFEST.json"

# --------------------------- FAIL-CLOSED IDENTITY ----------------------------

[ -f "$CHECKPOINT/config.json" ] || fail "checkpoint config missing"
[ -f "$CHECKPOINT/model.safetensors.index.json" ] || fail "checkpoint index missing"
CONFIG_SHA="$(sha256sum "$CHECKPOINT/config.json" | awk '{print $1}')"
INDEX_SHA="$(sha256sum "$CHECKPOINT/model.safetensors.index.json" | awk '{print $1}')"
WANT_CONFIG="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["config_sha256"])' "$MANIFEST")"
WANT_INDEX="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"]["index_sha256"])' "$MANIFEST")"
[ "$CONFIG_SHA" = "$WANT_CONFIG" ] || fail "checkpoint config sha drift"
[ "$INDEX_SHA" = "$WANT_INDEX" ] || fail "checkpoint index sha drift"

# ------------------------------ IDEMPOTENCE ----------------------------------

mkdir -p "$DEST"
if [ -f "$PACK" ] && [ -f "$PACK.experts" ] && [ -f "$PACK.sha256" ] \
   && [ -f "$PACK.receipt.json" ]; then
  echo "qwen38max-pack-emit: rank $RANK already complete ($PACK)"
  sha256sum -c "$PACK.sha256" >/dev/null 2>&1 ||
    fail "existing pack digest mismatch: $PACK"
  echo "EMIT-DONE rank=$RANK state=cached"
  exit 0
fi

# -------------------------------- EMIT ---------------------------------------

START="$(date +%s)"
PARTIAL="$DEST/partial-$ATTEMPT.sp"
# A ttl kill bypasses the EXIT trap; sweep partials of dead attempts
# first (queue conflict rules keep this lane alone on the node).
rm -f "$DEST"/partial-*.sp "$DEST"/partial-*.sp.* \
      "$DEST"/.partial-*.qwen38sp.* "$DEST"/.partial-*.sp.* 2>/dev/null || true
trap 'rm -f "$PARTIAL" "$PARTIAL.experts" "$PARTIAL.receipt.json"' EXIT

python3 "$CHECKOUT/tools/qwen38_stagepack.py" \
  --checkpoint "$CHECKPOINT" \
  --output "$PARTIAL" \
  --first-layer 0 --layer-count "$LAYER_COUNT" \
  --strip-mtp --expert-codec nvfp4 \
  --tp-degree "$WORLD" --tp-rank "$RANK"
MID="$(date +%s)"

python3 "$CHECKOUT/tools/qwen38max_tp16_rank_verify.py" \
  --pack "$PARTIAL" --tp-degree "$WORLD" --tp-rank "$RANK" \
  --receipt "$PARTIAL.receipt.json" --recompute-file-hash

bash "$CHECKOUT/tools/qwen38max_multidev_experts_manifest.sh" "$PARTIAL"

sha256sum "$PARTIAL" | awk '{print $1}' > "$PARTIAL.sha256"
END="$(date +%s)"

# --------------------------- ATOMIC PLACEMENT --------------------------------

install -m 0644 "$PARTIAL" "$PACK"
install -m 0644 "$PARTIAL.experts" "$PACK.experts"
install -m 0644 "$PARTIAL.sha256" "$PACK.sha256"
install -m 0644 "$PARTIAL.receipt.json" "$PACK.receipt.json"
BYTES="$(wc -c < "$PACK")"
echo "qwen38max-pack-emit: rank $RANK placed $PACK ($BYTES bytes); \
pack+verify=$((MID - START))s verify+manifest=$((END - MID))s"
echo "EMIT-DONE rank=$RANK state=emitted bytes=$BYTES \
pack_seconds=$((MID - START)) total_seconds=$((END - START))"
