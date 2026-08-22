#!/usr/bin/env bash
set -euo pipefail

# GLM 5.2 resident decode stage, retained-receipt GPU validation.
#
# Called by resident_decode_stage_rules.mk as:
#   GPU_VALIDATOR VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE
# with the RUNTIME_CONFIGURATION environment (SPARK_GLM52_*) exported, so the
# configuration hash names the exact configuration this run validated and the
# publish recipe retains it. The validator source digest is pinned through
# SPARK_GLM52_CUDA_VALIDATOR_SHA256 in the same configuration, so a published
# artifact carries which validator text approved it.

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
cuda_validator="${script_directory}/spark_glm52_resident_decode_stage_cuda_validation.cu"
trap 'rm -rf "${validation_directory}"' EXIT

glm52_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)
glm52_codec_ids=(2 3 4 5 6 7)

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

if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM52_STAGE_PACK_PATH:-}" || ! -s "${SPARK_GLM52_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_GLM52_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
    exit 2
fi
require_source_digest "${SPARK_GLM52_CUDA_VALIDATOR_SHA256:-}" "${cuda_validator}" "Glm52 CUDA validator"

# The v1 validator exercises one dense layer's forward on synthetic weights;
# it does not load the pack. The pack still gates here (non-empty, named by
# the pinned configuration) because the receipt binds the configuration the
# archive publishes under; the loader's own schema chain pins its content.
codec_index=-1
for index in "${!glm52_codecs[@]}"; do
    if [[ "${SPARK_GLM52_EXPERT_CODEC:-}" == "${glm52_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "glm52 hardware validation requires SPARK_GLM52_EXPERT_CODEC to name one of: ${glm52_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "glm52 hardware validation requires SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "glm52 hardware validation admits only CUDA_ARCH=sm_121a" >&2
    exit 2
fi
if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for glm52 hardware validation" >&2
    exit 2
fi

# Single-source the build identity exactly as the compile gate does, so the
# validator translation unit sees the same package defines the archive did.
read -r model_revision contract_sha256 < <(
    python3 "${repository_root}/tools/glm52_model_contract.py" \
        --print-build-identity "${SPARK_GLM52_EXPERT_CODEC}"
)

make -C "${repository_root}" \
    build/libsparkpipe_core.a \
    build/libsparkpipe_runtime.a

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --expt-relaxed-constexpr \
    -gencode arch=compute_121a,code=sm_121a \
    -I"${repository_root}/include" \
    -I"${repository_root}/model-families/glm52/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    -DGLM52_EXPERT_WEIGHT_CODEC="${glm52_codec_ids[${codec_index}]}" \
    -DGLM52_EXPERT_CODEC_NAME="\"${SPARK_GLM52_EXPERT_CODEC}\"" \
    -DGLM52_MODEL_REVISION="\"${model_revision}\"" \
    -DGLM52_CONTRACT_SHA256="\"${contract_sha256}\"" \
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
    -o "${validation_directory}/glm52_resident_decode_stage_validator"

"${validation_directory}/glm52_resident_decode_stage_validator" "${configuration_hash}"
