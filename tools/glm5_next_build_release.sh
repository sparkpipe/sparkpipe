#!/usr/bin/env bash
# Build a complete local GLM release inside a clean queue-synced checkout.
# Submit with spark_queue.py add --resources gpu (or exclusive), memory >=32 GiB.
# No branch checkout, process stopping, hub writes or fleet UPDATE side effects.
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
read -r contract_sha contract_path < <(sha256sum model_contracts/glm53_flash_authoritative.json)
module_args=(EXPERT_CODEC=fp8 MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033
    "CONTRACT_SHA256=$contract_sha" NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a)
output=build/glm53_release
rm -f "$output/ARTIFACT_SHA256SUMS" "$output/SOURCE_COMMIT"
echo "queue=$SPARK_QUEUE_ID source=$(git rev-parse HEAD) contract=$contract_sha"
make -j8 build/sparkpipe_model_compile build/sparkpipe_model_residentd build/sparkpipe_model_api \
    hidden_transport_spark_host_rdma_verbs NVCC=/usr/local/cuda/bin/nvcc
make -j4 -C modules/glm5_next_resident_decode_stage publish adapter "${module_args[@]}"
build/sparkpipe_model_compile \
    --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
    --library build/module_library --output "$output" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm \
    --cc-arg -ldl --cc-arg -pthread
mkdir -p "$output/bin" "$output/lib"
install -m755 build/sparkpipe_model_residentd build/sparkpipe_model_api "$output/bin/"
install -m755 build/libhidden_transport_spark_host_rdma_verbs.so "$output/lib/hidden_transport.so"
install -m755 build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so "$output/lib/model_serving_adapter.so"
git rev-parse HEAD > "$output/SOURCE_COMMIT"
(cd "$output" && sha256sum bin/sparkpipe_model_residentd bin/sparkpipe_model_api \
    lib/hidden_transport.so lib/model_serving_adapter.so stages/stage_000/model_driver.so) \
    > "$output/ARTIFACT_SHA256SUMS.partial"
mv "$output/ARTIFACT_SHA256SUMS.partial" "$output/ARTIFACT_SHA256SUMS"
echo "BUILD-PASS $output (local artifact; not deployed)"
