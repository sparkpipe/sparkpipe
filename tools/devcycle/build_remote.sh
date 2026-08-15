#!/usr/bin/env bash
#
# build_remote.sh NAME — run ON the build spark, inside the sparkpipe checkout.
# Compiles the DSV4 TP4 B1 variant ladder (bucket 1) and produces the model
# driver + runtime artifacts in /tmp/devcycle-build-NAME/.
#
# Usage (on spark):  tools/devcycle/build_remote.sh NAME
set -euo pipefail

NAME="${1:?usage: build_remote.sh NAME [LOCAL_SHA] [BUCKET]}"
SOURCE_SHA="${2:-}"
BUCKET="${3:-1}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="/tmp/devcycle-build-${NAME}"
MODULE_LIB="/tmp/devcycle-build-${NAME}-module-library"
FIRMWARE="/tmp/devcycle-firmware-b${BUCKET}.json"
VALIDATOR="/tmp/sparkpipe_dsv4_devcycle_validator.sh"
ARCHIVE="build/modules/dsv4_resident_decode_stage/libdsv4_resident_decode_stage_b${BUCKET}.a"
if [[ -z "${SOURCE_SHA}" ]]; then
    SOURCE_SHA="$(git -C "${ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
fi

cd "${ROOT}"
export NVCC=/usr/local/cuda-13.0/bin/nvcc
export CUDA_HOME=/usr/local/cuda-13.0
[[ -x "${NVCC}" ]] || { echo "missing nvcc at ${NVCC}" >&2; exit 2; }
[[ -x "${VALIDATOR}" ]] || { echo "missing validator ${VALIDATOR}" >&2; exit 2; }
[[ -z "$(git status --porcelain -- modules/ inference/ include/ model-families/ ring/ runtime/ src/ 2>/dev/null)" ]] || \
    echo "WARNING: dirty source (expected for candidate patches)"

rm -rf "${OUT}" "${MODULE_LIB}"
mkdir -p "${OUT}"

make clean >/dev/null
make -C modules/dsv4_resident_decode_stage -j4 variants \
    MODULE_BATCH_VARIANT_BUCKETS=${BUCKET} \
    NVCC=/usr/local/cuda-13.0/bin/nvcc \
    CUDA_ARCH=sm_121a
# firmware for this bucket: the generated bucket description (committed)
[[ "${BUCKET}" == "1" ]] && cp examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json "${FIRMWARE}"
[[ "${BUCKET}" != "1" ]] && cp examples/model_descriptions/dsv4_resident_decode_stage_firmware_b${BUCKET}.json "${FIRMWARE}"
make -j4 \
    build/sparkpipe_module_publish \
    build/sparkpipe_model_compile \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_batch \
    build/libdsv4_tp4_b1_serving_adapter.so \
    build/libhidden_transport_spark_host_rdma_verbs.so
# non-B1 buckets use the base TP4 adapter (bucket-1024 ceiling; mid-bucket
# adapters need generated batch descriptions, a known ladder gap)
if [[ "${BUCKET}" != "1" ]]; then
    make build/libdsv4_tp4_serving_adapter.so
    ADAPTER_SO=build/libdsv4_tp4_serving_adapter.so
else
    ADAPTER_SO=build/libdsv4_tp4_b1_serving_adapter.so
fi

ARCHIVE_SHA="$(sha256sum "${ARCHIVE}" | cut -d' ' -f1)"
build/sparkpipe_module_publish \
    --library "${MODULE_LIB}" \
    --module spark.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16.h4096.l43.e256.k6.ga0731.b${BUCKET}.v4 \
    --target cuda.sm121.dsv4.flash.resident_decode_stage.linear_fp8.expert_mxfp4.kv_bf16 \
    --link-unit "${ARCHIVE}" \
    --recipe merged-main.dsv4.tp4.b1.static-sm121.live-pending.v1 \
    --initialize SparkDsv4ResidentDecodeStageInitialize \
    --execute SparkDsv4ResidentDecodeStageExecute \
    --admit SparkDsv4ResidentDecodeStageAdmit \
    --snapshot SparkDsv4ResidentDecodeStageSnapshot \
    --destroy SparkDsv4ResidentDecodeStageDestroy \
    --validator "${VALIDATOR}" \
    --validator-arg "${ROOT}" \
    --validator-arg "${SOURCE_SHA}" \
    --validator-arg "${ARCHIVE_SHA}"

build/sparkpipe_model_compile \
    --model "${FIRMWARE}" \
    --stage dsv4_resident_decode_stage \
    --library "${MODULE_LIB}" \
    --output "${OUT}" \
    --cc /usr/bin/cc \
    --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda \
    --cc-arg -lcudart \
    --cc-arg -lstdc++ \
    --cc-arg -lm \
    --cc-arg -ldl \
    --cc-arg -pthread

# fail fast on bucket plumbing bugs: the driver embeds the bucket firmware's
# raw-file sha256 as its model description sha (a silent b1 fallback would
# embed the b1 firmware sha instead).
FIRMWARE_SHA="$(sha256sum "${FIRMWARE}" | cut -d' ' -f1)"
strings "${OUT}/model_driver.so" | grep -q "${FIRMWARE_SHA}" || { \
    echo "build_remote: driver firmware mismatch (expected ${FIRMWARE_SHA:0:16}...)" >&2; exit 9; }
cp build/sparkpipe_model_residentd "${OUT}/sparkpipe_model_residentd"
cp build/sparkpipe_model_batch "${OUT}/sparkpipe_model_batch"
cp "${ADAPTER_SO}" "${OUT}/model_serving_adapter.so"
cp build/libhidden_transport_spark_host_rdma_verbs.so "${OUT}/hidden_transport.so"
printf '%s\n' "${SOURCE_SHA}" >"${OUT}/SOURCE_COMMIT"

echo "build_remote ${NAME} done"
sha256sum "${OUT}/model_driver.so" "${OUT}/sparkpipe_model_residentd"
