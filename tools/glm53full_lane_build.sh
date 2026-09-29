#!/usr/bin/env bash
set -euo pipefail
: "${GLMFULL_CODEC:?GLMFULL_CODEC is fp8, bf16 or nvfp4}"
: "${GLMFULL_BUCKET:?GLMFULL_BUCKET is the batch variant bucket the lane serves, e.g. 16}"
: "${GLMFULL_FIRMWARE_ROOT:?GLMFULL_FIRMWARE_ROOT is the output directory for the staged firmware}"
: "${GLMFULL_STAGE_PACK:?GLMFULL_STAGE_PACK is a readable placed pack the publish receipt binds}"
case "$GLMFULL_CODEC" in
  fp8|bf16|nvfp4) ;;
  *) echo "glm53full build: GLMFULL_CODEC must be fp8, bf16 or nvfp4" >&2; exit 2 ;;
esac
GLMFULL_ARM="${GLMFULL_ARM:-$GLMFULL_CODEC}"
GLMFULL_SCORE_DUMP="${GLMFULL_SCORE_DUMP:-0}"
case "$GLMFULL_SCORE_DUMP" in
  0) SUFFIX=v2; BUILD_SUBDIR="$GLMFULL_CODEC" ;;
  1) SUFFIX=v2.scoredump; BUILD_SUBDIR="$GLMFULL_CODEC-scoredump" ;;
  *) echo "glm53full build: GLMFULL_SCORE_DUMP must be 0 or 1" >&2; exit 2 ;;
esac
CHECKOUT="$(cd "$(dirname "$0")/.." && pwd)"
MODULE="modules/glm52_resident_decode_stage"
FIRMWARE="examples/model_descriptions/glm52_resident_decode_stage_${GLMFULL_ARM}_firmware.json"
PREFIX="spark.glm52.resident_decode_stage.bf16.expert_${GLMFULL_CODEC}.h6144.l78.e256.k8"
DEFAULT_ID="$PREFIX.v2"
BUCKETED_ID="$PREFIX.b$GLMFULL_BUCKET.$SUFFIX"
read -r REVISION CONTRACT_SHA256 < <(python3 "$CHECKOUT/tools/glm52_model_contract.py" --print-build-identity "$GLMFULL_ARM" --expert-codec "$GLMFULL_CODEC")
: "${CONTRACT_SHA256:?glm53full build: no build identity for arm $GLMFULL_ARM with $GLMFULL_CODEC experts}"
COMMIT="$(cat "$CHECKOUT/SOURCE_COMMIT")"
export PATH="/usr/local/cuda/bin:$PATH"
MAKE_IDENTITY=(CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a EXPERT_CODEC="$GLMFULL_CODEC" MODEL_ARM="$GLMFULL_ARM" MODEL_REVISION="$REVISION" CONTRACT_SHA256="$CONTRACT_SHA256" MODULE_BATCH_VARIANT_BUCKETS="$GLMFULL_BUCKET" STAGE_PACK_PATH="$GLMFULL_STAGE_PACK" EXECUTION_ROW_CAPACITY="$GLMFULL_BUCKET" SCORE_DUMP="$GLMFULL_SCORE_DUMP")
make -C "$CHECKOUT" -j8 CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_121a \
  build/sparkpipe_model_residentd build/sparkpipe_model_api build/sparkpipe_model_compile \
  build/sparkpipe_module_publish build/sparkpipe_driver_inspect build/weightd_warm build/sparkpipe_model_batch \
  hidden_transport_spark_host_rdma_verbs
make -C "$CHECKOUT/$MODULE" -j8 "${MAKE_IDENTITY[@]}" variants
make -C "$CHECKOUT/$MODULE" -j1 "${MAKE_IDENTITY[@]}" publish_variants
LANE_FIRMWARE="$CHECKOUT/build/glm53full-firmware-b$GLMFULL_BUCKET.json"
python3 - "$CHECKOUT/$FIRMWARE" "$LANE_FIRMWARE" "$DEFAULT_ID" "$BUCKETED_ID" <<'PYFW'
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
PYFW
make -C "$CHECKOUT/$MODULE" -j8 "${MAKE_IDENTITY[@]}" MODEL_DESCRIPTION="$LANE_FIRMWARE" adapter
ADAPTER="$CHECKOUT/build/modules/glm52_resident_decode_stage/$BUILD_SUBDIR/libglm52_serving_adapter_$GLMFULL_CODEC.so"
DRIVER="$CHECKOUT/build/glm53full-driver-b$GLMFULL_BUCKET"
rm -rf "$DRIVER"
"$CHECKOUT/build/sparkpipe_model_compile" --model "$LANE_FIRMWARE" --library "$CHECKOUT/build/module_library" \
  --output "$DRIVER" --cc /usr/bin/cc --include "$CHECKOUT/include" \
  --cc-arg "$CHECKOUT/build/libsparkpipe_runtime.a" --cc-arg "$CHECKOUT/build/libsparkpipe_core.a" \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda --cc-arg -lcudart \
  --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl --cc-arg -pthread
TARGET="cuda.sm121.glm52.resident_decode_stage.bf16.expert_$GLMFULL_CODEC"
"$CHECKOUT/build/sparkpipe_driver_inspect" "$DRIVER/stages/stage_000/model_driver.so" "$TARGET" > "$DRIVER/driver-inspect.log"
STAGE="$GLMFULL_FIRMWARE_ROOT.new"
rm -rf "$STAGE"
mkdir -p "$STAGE"
install -m 0755 "$CHECKOUT/build/sparkpipe_model_residentd" "$CHECKOUT/build/sparkpipe_model_api" "$CHECKOUT/build/sparkpipe_model_batch" "$CHECKOUT/build/weightd_warm" "$STAGE/"
install -m 0644 "$ADAPTER" "$STAGE/model_serving_adapter.so"
install -m 0644 "$DRIVER/stages/stage_000/model_driver.so" "$STAGE/model_driver.so"
install -m 0644 "$CHECKOUT/build/libhidden_transport_spark_host_rdma_verbs.so" "$STAGE/hidden_transport.so"
printf '%s\n' "$COMMIT" > "$STAGE/SOURCE_COMMIT"
printf 'module=%s\nbucket=%s\nrevision=%s\ncontract=%s\narm=%s\nscore_dump=%s\n' "$BUCKETED_ID" "$GLMFULL_BUCKET" "$REVISION" "$CONTRACT_SHA256" "$GLMFULL_ARM" "$GLMFULL_SCORE_DUMP" > "$STAGE/MODULE_IDENTITY"
( cd "$STAGE" && sha256sum sparkpipe_model_residentd sparkpipe_model_api sparkpipe_model_batch weightd_warm model_serving_adapter.so model_driver.so hidden_transport.so > SHA256SUMS )
rm -rf "$GLMFULL_FIRMWARE_ROOT.old"
if [ -d "$GLMFULL_FIRMWARE_ROOT" ]; then mv "$GLMFULL_FIRMWARE_ROOT" "$GLMFULL_FIRMWARE_ROOT.old"; fi
mv "$STAGE" "$GLMFULL_FIRMWARE_ROOT"
echo "glm53full build: firmware at $GLMFULL_FIRMWARE_ROOT commit $COMMIT codec $GLMFULL_CODEC arm $GLMFULL_ARM module $BUCKETED_ID"
