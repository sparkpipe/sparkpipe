#!/usr/bin/env bash
# ling_multidev_build.sh — the lane-9 ling family firmware build (M3
# build arm). Runs INSIDE one per-node queue job on the synced checkout;
# produces the five artifacts tools/ling_multidev_run_family.sh consumes
# through LING_PREBUILT_DIR, so serving attaches never pay the nvcc bill
# (the qwen38_27b lane-1 M3 flow, PR #1133):
#
#   FIRMWARE/sparkpipe_model_residentd   the common resident launcher
#   FIRMWARE/sparkpipe_model_api         the lane API (decode receipts)
#   FIRMWARE/weightd_warm                the shared-socket warmer
#   FIRMWARE/model_serving_adapter.so    libling_serving_adapter_<codec>.so
#   FIRMWARE/model_driver.so             the linked ling stage driver
#   FIRMWARE/hidden_transport.so         libhidden_transport_...verbs.so
#   FIRMWARE/SOURCE_COMMIT + SHA256SUMS  the coherence receipt
#
# BUCKETED VARIANT BUILD (the queue-window law): the default module
# archive's unflagged CUDA TU compiles the b1024-class tuning and
# EXCEEDS the queue's hard 15-minute task window by itself (measured
# 2026-09-23: r1-r4 TTL-terminated mid-TU; nvcc checkpoints nothing, so
# retries never converge). The sanctioned mechanism is the module's
# batch-variant ladder: this build targets the b128 VARIANT archive
# (-DSPARK_BATCH_BUCKET=128, measured 8.5 s wall for the whole variant
# archive) and publishes it under the documented bucketed identity
# <prefix>.b128.<suffix>; a lane-local firmware json (generated here
# from the committed one, module id rewritten) links it. The b128 bucket
# serves 1..128 rows - exactly the LING-T1 serving shape this lane's
# stage configs pin (execution_row_capacity 128). Re-extend the ladder
# (and rebuild the heavier TU outside a 15-minute window, manager
# authorization required) before any capacity bump above 128 rows.
#
# Numerical validation is NOT this script's job: the record-only publish
# (no --validator) links the driver; exactness gates against the
# committed T1R token streams run at decode time.
#
# Env:
#   LING_EXPERT_CODEC   bf16 (default) | fp8 - which placed arm to build
#   LING_FIRMWARE_ROOT  output dir (default
#                       /home/<host>/sparkdata/ling-lane9-firmware)
set -euo pipefail

FAMILY="ling"
LANE=9
BUCKET=128
EXPERT_CODEC="${LING_EXPERT_CODEC:-bf16}"
MODEL_REVISION="e0dfe7cd0f6e3b572bbbc0a8a84947469e428cc3"
CONTRACT="model_contracts/ling_authoritative.json"
FIRMWARE="examples/model_descriptions/ling_resident_decode_stage_bf16_firmware.json"
MODULE="modules/ling_resident_decode_stage"
MODULE_ID_PREFIX="spark.ling.resident_decode_stage.bf16.expert_${EXPERT_CODEC}.h2560.l42.kda35.e512.k8"
MODULE_ID_SUFFIX="v2"
MODULE_ID_DEFAULT="${MODULE_ID_PREFIX}.${MODULE_ID_SUFFIX}"
MODULE_ID_BUCKETED="${MODULE_ID_PREFIX}.b${BUCKET}.${MODULE_ID_SUFFIX}"
MODULE_TARGET="cuda.sm121.ling.resident_decode_stage.bf16.expert_${EXPERT_CODEC}"
ENTRY_PREFIX="SparkLingResidentDecodeStage"

case "$EXPERT_CODEC" in
  bf16|fp8) ;;
  *) echo "ling-$FAMILY-lane$LANE build: LING_EXPERT_CODEC must be bf16 or fp8" >&2; exit 2 ;;
esac

HOST="$(hostname)"
case "$HOST" in
  spark*) ;;
  *) echo "ling-$FAMILY-lane$LANE build: not on a spark node: $HOST" >&2; exit 2 ;;
esac
FIRMWARE_ROOT="${LING_FIRMWARE_ROOT:-/home/$HOST/sparkdata/ling-lane9-firmware}"

CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
COMMIT="$(git -C "$CHECKOUT" rev-parse HEAD)"

PATH="/usr/local/cuda/bin:$PATH"
export PATH

CONTRACT_SHA256="$(sha256sum "$CHECKOUT/$CONTRACT" | awk '{print $1}')"
make -C "$CHECKOUT" -j4 \
  build/sparkpipe_model_residentd \
  build/sparkpipe_model_api \
  build/sparkpipe_model_compile \
  build/sparkpipe_module_publish \
  build/weightd_warm \
  hidden_transport_spark_host_rdma_verbs
# The variant archive (NOT the default `archive` target): the b128 TU
# carries -DSPARK_BATCH_BUCKET=128; the adapter takes the ladder's last
# bucket (128) for the same reason. publish_variants then publishes the
# bucket under its documented bucketed identity THROUGH the module's
# retained-receipt GPU validator (synthesized weights; the release
# publish path for the unbucketed archive stays as-is elsewhere).
make -C "$CHECKOUT/$MODULE" -j4 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  MODULE_BATCH_VARIANT_BUCKETS="$BUCKET" \
  "../../build/modules/ling_resident_decode_stage/$EXPERT_CODEC/libling_resident_decode_stage_${EXPERT_CODEC}_b${BUCKET}.a"
make -C "$CHECKOUT/$MODULE" -j1 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  MODULE_BATCH_VARIANT_BUCKETS="$BUCKET" \
  publish_variants
VARIANT_ARCHIVE="$CHECKOUT/build/modules/ling_resident_decode_stage/$EXPERT_CODEC/libling_resident_decode_stage_${EXPERT_CODEC}_b${BUCKET}.a"
ADAPTER="$CHECKOUT/build/modules/ling_resident_decode_stage/$EXPERT_CODEC/libling_serving_adapter_$EXPERT_CODEC.so"
for artifact in "$VARIANT_ARCHIVE"; do
  [ -f "$artifact" ] || { echo "not built: $artifact" >&2; exit 1; }
done
grep -q "$MODULE_ID_BUCKETED" "$CHECKOUT"/build/module_library/active/*.json 2>/dev/null \
  || { echo "bucketed module record missing from the library" >&2; exit 1; }

# Lane-local firmware: the committed firmware with the module id
# rewritten to the bucketed variant (generated at build time; no
# shared-file edit).
BUILD_DRIVER="$CHECKOUT/build/ling-driver.$$"
LANE_FIRMWARE="$CHECKOUT/build/ling-firmware-b${BUCKET}.$$.json"
rm -rf "$BUILD_DRIVER"
python3 - "$CHECKOUT/$FIRMWARE" "$LANE_FIRMWARE" "$MODULE_ID_DEFAULT" "$MODULE_ID_BUCKETED" <<'PYFW'
import json, sys
source, output, default_id, bucketed_id = sys.argv[1:5]
document = json.load(open(source, encoding="utf-8"))
rewritten = 0
for stage in document.get("stages", []):
    for program in stage.get("programs", []):
        for operation in program.get("operations", []):
            if operation.get("module") == default_id:
                operation["module"] = bucketed_id
                rewritten += 1
if rewritten != 1:
    raise SystemExit(f"firmware rewrite: expected exactly 1 module id, rewrote {rewritten}")
with open(output, "w", encoding="utf-8") as handle:
    json.dump(document, handle, indent=1)
    handle.write("\n")
print(f"firmware variant {output}: module id -> {bucketed_id}")
PYFW
make -C "$CHECKOUT/$MODULE" -j4 \
  CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  EXPERT_CODEC="$EXPERT_CODEC" \
  MODEL_REVISION="$MODEL_REVISION" \
  CONTRACT_SHA256="$CONTRACT_SHA256" \
  MODULE_BATCH_VARIANT_BUCKETS="$BUCKET" \
  MODEL_DESCRIPTION="$LANE_FIRMWARE" \
  adapter
[ -f "$ADAPTER" ] || { echo "not built: $ADAPTER" >&2; exit 1; }
"$CHECKOUT/build/sparkpipe_model_compile" \
  --model "$LANE_FIRMWARE" \
  --library "$CHECKOUT/build/module_library" \
  --output "$BUILD_DRIVER" \
  --cc /usr/bin/cc \
  --include "$CHECKOUT/include" \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
  --cc-arg -lcuda \
  --cc-arg -lcudart \
  --cc-arg -lstdc++ \
  --cc-arg -lm \
  --cc-arg -ldl \
  --cc-arg -pthread
ldd -r "$BUILD_DRIVER/stages/stage_000/model_driver.so" 2>&1 \
  | grep -E 'not found|undefined symbol' \
  && { echo "driver link has unresolved symbols" >&2; exit 1; } || true

STAGE="$FIRMWARE_ROOT.new.$$"
rm -rf "$FIRMWARE_ROOT".new.* "$STAGE"
mkdir -p "$STAGE"
install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$STAGE/"
install -m 0755 "$CHECKOUT/build/sparkpipe_model_api" "$STAGE/"
install -m 0755 "$CHECKOUT/build/weightd_warm" "$STAGE/"
install -m 0644 "$ADAPTER" "$STAGE/model_serving_adapter.so"
install -m 0644 "$BUILD_DRIVER/stages/stage_000/model_driver.so" "$STAGE/model_driver.so"
install -m 0644 \
  "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" \
  "$STAGE/hidden_transport.so"
printf '%s\n' "$COMMIT" > "$STAGE/SOURCE_COMMIT"
printf 'module=%s\nbucket=%s\n' "$MODULE_ID_BUCKETED" "$BUCKET" \
  > "$STAGE/MODULE_IDENTITY"
( cd "$STAGE" && sha256sum sparkpipe_model_residentd \
    sparkpipe_model_api weightd_warm \
    model_serving_adapter.so model_driver.so hidden_transport.so \
    > SHA256SUMS )
rm -rf "$BUILD_DRIVER" "$LANE_FIRMWARE" "$FIRMWARE_ROOT.old"
[ -d "$FIRMWARE_ROOT" ] && mv "$FIRMWARE_ROOT" "$FIRMWARE_ROOT.old"
mv "$STAGE" "$FIRMWARE_ROOT"
[ "$(cat "$FIRMWARE_ROOT/SOURCE_COMMIT")" = "$COMMIT" ] || {
  echo "publish verification failed: $FIRMWARE_ROOT/SOURCE_COMMIT" >&2
  exit 1
}
echo "ling-$FAMILY-lane$LANE build: firmware at $FIRMWARE_ROOT commit $COMMIT codec $EXPERT_CODEC bucket $BUCKET"
