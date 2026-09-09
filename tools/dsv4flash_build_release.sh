#!/usr/bin/env bash
# Build a complete local dsv4_flash release inside a clean queue-synced
# checkout. Submit with spark_queue.py add --resources gpu, memory >=32 GiB.
# No branch checkout, process stopping, hub writes or fleet UPDATE side
# effects. COHERENT GENERATION: residentd + api + transport + adapter +
# driver + config ship together with ARTIFACT_SHA256SUMS + SOURCE_COMMIT;
# replacing only model_driver.so is not a release.
#
# PAIRING LAW: the tp16 serving adapter embeds the UNBUCKETED contract sha,
# so the driver compiles from the DEFAULT firmware description against the
# DEFAULT archive publish (a firmware_b1 driver fails target_mismatch).
set -euo pipefail
if (( $# != 0 )); then
    echo "usage: queue this script with no arguments from a clean synced checkout" >&2
    exit 2
fi
: "${SPARK_QUEUE_ID:?run through spark_queue.py with GPU ownership}"
cd "$(dirname "$0")/.."
git diff --quiet HEAD --
git diff --cached --quiet --
export PATH="/usr/local/cuda/bin:$PATH"
read -r contract_sha contract_path < <(sha256sum model_contracts/dsv4_flash_authoritative.json)
output=build/dsv4flash_release
rm -f "$output/ARTIFACT_SHA256SUMS" "$output/SOURCE_COMMIT"
echo "queue=$SPARK_QUEUE_ID source=$(git rev-parse HEAD) contract=$contract_sha"
make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd \
    build/sparkpipe_model_api hidden_transport_spark_host_rdma_verbs \
    NVCC=/usr/local/cuda/bin/nvcc
make -j4 -C modules/dsv4_resident_decode_stage publish adapter \
    STAGE_PACK_PATH="${DSV4_VALIDATION_PACK:?set DSV4_VALIDATION_PACK to a readable validation pack}" \
    STAGE_COUNT="${DSV4_PUBLISH_STAGE_COUNT:-16}" \
    STAGE_INDEX="${DSV4_PUBLISH_STAGE_INDEX:-0}" \
    STAGE_FIRST_LAYER="${DSV4_PUBLISH_STAGE_FIRST_LAYER:-0}" \
    STAGE_LAYER_COUNT="${DSV4_PUBLISH_STAGE_LAYER_COUNT:-43}" \
    MAX_ACTIVE_SEQUENCES=8 PIPELINE_SLOT_COUNT=1 \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
build/sparkpipe_model_compile \
    --model examples/model_descriptions/dsv4_resident_decode_stage_firmware.json \
    --library build/module_library --output "$output" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm \
    --cc-arg -ldl --cc-arg -pthread
mkdir -p "$output/bin" "$output/lib" "$output/config"
install -m755 build/sparkpipe_model_residentd build/sparkpipe_model_api "$output/bin/"
install -m755 build/libhidden_transport_spark_host_rdma_verbs.so "$output/lib/hidden_transport.so"
install -m755 build/libdsv4_tp16_serving_adapter.so "$output/lib/model_serving_adapter.so"
install -m644 examples/deployments/dsv4_flash_tp16_b1_host_rdma.spec.json "$output/config/"
git rev-parse HEAD > "$output/SOURCE_COMMIT"
(cd "$output" && sha256sum bin/sparkpipe_model_residentd bin/sparkpipe_model_api \
    lib/hidden_transport.so lib/model_serving_adapter.so config/*.json \
    stages/stage_000/model_driver.so) > "$output/ARTIFACT_SHA256SUMS.partial"
mv "$output/ARTIFACT_SHA256SUMS.partial" "$output/ARTIFACT_SHA256SUMS"
echo "BUILD-PASS $output (local artifact; not deployed)"
