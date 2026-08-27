#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
    exit 2
fi

configuration_hash="$1"
module_archive="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
cuda_validator="${script_directory}/spark_qwen4_flash_resident_decode_stage_cuda_validation.cu"
cpu_oracle="${script_directory}/spark_qwen4_flash_reference.c"
trap 'rm -rf "${validation_directory}"' EXIT

require_source_digest() {
    local expected="$1"
    local path="$2"
    local label="$3"
    local actual remainder
    if [[ ! "${expected}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "${label} expected SHA-256 is invalid" >&2
        exit 2
    fi
    read -r actual remainder < <(sha256sum "${path}")
    if [[ "${actual}" != "${expected}" ]]; then
        echo "${label} SHA-256 mismatch" >&2
        exit 2
    fi
}

require_configuration_value() {
    local name="$1"
    local expected="$2"
    local actual="${!name:-}"
    if [[ "${actual}" != "${expected}" ]]; then
        echo "qwen4_flash hardware validation requires ${name}=${expected}, got '${actual}'" >&2
        exit 2
    fi
}

if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ -z "${SPARK_QWEN4_FLASH_STAGE_PACK_PATH:-}" || ! -s "${SPARK_QWEN4_FLASH_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_QWEN4_FLASH_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
    exit 2
fi
require_source_digest "${SPARK_QWEN4_FLASH_CUDA_VALIDATOR_SHA256:-}" "${cuda_validator}" "Qwen38_27b CUDA validator"
require_source_digest "${SPARK_QWEN4_FLASH_CPU_ORACLE_SHA256:-}" "${cpu_oracle}" "Qwen38_27b CPU oracle"

# The KV block table must span max_active_sequence_count lanes and the
# validator drives eight lanes of one block each. Degree 1 validates the
# stage-0 mid-pipeline tier (hidden output transport mandatory); a TP degree
# validates the whole-stack tier instead: rank 0 in standalone collective
# mode (the build host has no peer group), so consistency and determinism
# gate here while cross-rank numerics gate at the band E2E run.
require_configuration_value SPARK_QWEN4_FLASH_ALLOW_UNQUALIFIED_EXECUTION 1
require_configuration_value SPARK_QWEN4_FLASH_STAGE_INDEX 0
require_configuration_value SPARK_QWEN4_FLASH_STAGE_FIRST_LAYER 0
# The harness block table spans MAX_ACTIVE_SEQUENCES lanes (the -D plumbing
# from PR #714); admit the qualified lane-ladder values.
case "${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-8}" in
    8|16|64) ;;
    *)
        echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES in {8,16,64}, got '${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-}'" >&2
        exit 2
        ;;
esac
# Whole-stack tier: TP-sharded (TP_DEGREE != 1) OR TP1 full-width
# standalone (TP_DEGREE == 1 with TP_STANDALONE == 1). The TP1 topology
# knob postdates the original >1 restriction.
if [[ "${SPARK_QWEN4_FLASH_TP_DEGREE:-1}" != "1" ]] || [[ "${SPARK_QWEN4_FLASH_TP_STANDALONE:-0}" == "1" ]]; then
    require_configuration_value SPARK_QWEN4_FLASH_TP_RANK 0
    require_configuration_value SPARK_QWEN4_FLASH_TP_STANDALONE 1
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_COUNT 1
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT 48
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_MTP 1
else
    if (( ${SPARK_QWEN4_FLASH_STAGE_COUNT:-0} < 2 )); then
        echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_COUNT >= 2 (mid-pipeline stage 0)" >&2
        exit 2
    fi
    if (( ${SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT:-0} < 4 || ${SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT:-48} >= 48 )); then
        echo "qwen4_flash hardware validation requires 4 <= SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT < 48 (GDN and full attention, no final head)" >&2
        exit 2
    fi
fi
if (( ${SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS:-0} < 8 )); then
    echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS >= 8" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "Qwen38_27b hardware validation admits only CUDA_ARCH=sm_121a" >&2
    exit 2
fi
if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for Qwen38_27b hardware validation" >&2
    exit 2
fi

make -C "${repository_root}" \
    build/libsparkpipe_core.a \
    build/libsparkpipe_runtime.a

"${nvcc_path}" \
    -std=c++17 \
    -DSPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES=${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-8} \
    -O3 \
    --expt-relaxed-constexpr \
    -gencode arch=compute_121a,code=sm_121a \
    -I"${repository_root}/include" \
    -I"${repository_root}/model-families/qwen4_flash/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${cuda_validator}" \
    "${module_archive}" \
    "${repository_root}/build/libsparkpipe_runtime.a" \
    "${repository_root}/build/libsparkpipe_core.a" \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" \
	-lcuda \
    -lcudart \
    -ldl \
    -lm \
    -Xcompiler -pthread \
    -o "${validation_directory}/qwen4_flash_resident_decode_stage_validator"

"${validation_directory}/qwen4_flash_resident_decode_stage_validator" "${configuration_hash}"
