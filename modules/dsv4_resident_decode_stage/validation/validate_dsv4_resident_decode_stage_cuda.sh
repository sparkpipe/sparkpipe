#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
    exit 2
fi

configuration_hash="$1"
module_archive="$2"
batch_bucket="${SPARK_MODULE_BATCH_BUCKET:-}"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
reference_fixture_directory="${repository_root}/qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128"
reference_verifier="${repository_root}/tools/verify_dsv4_ga_reference_fixture.py"
cuda_validator="${script_directory}/spark_dsv4_resident_decode_stage_cuda_validation.cu"
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

if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
case "${batch_bucket}" in
    1|2|4|8|16|32|64|128|256|512|1024)
        ;;
    *)
        echo "SPARK_MODULE_BATCH_BUCKET must name the archive's built variant" >&2
        exit 2
        ;;
esac
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ -z "${SPARK_DSV4_STAGE_PACK_PATH:-}" || ! -s "${SPARK_DSV4_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_DSV4_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
    exit 2
fi
require_source_digest "${SPARK_DSV4_CUDA_VALIDATOR_SHA256:-}" "${cuda_validator}" "DSV4 CUDA validator"
require_source_digest "${SPARK_DSV4_REFERENCE_VERIFIER_SHA256:-}" "${reference_verifier}" "DSV4 reference verifier"

unset SPARK_DSV4_REFERENCE_TOKEN_PATH
unset SPARK_DSV4_REFERENCE_OUTPUT_PATH
if [[ "${SPARK_DSV4_STAGE_INDEX:-}" == "0" &&
      "${SPARK_DSV4_STAGE_FIRST_LAYER:-}" == "0" &&
      "${SPARK_DSV4_STAGE_LAYER_COUNT:-}" == "3" ]]; then
    if [[ ! "${SPARK_DSV4_REFERENCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "stage-0 validation requires its retained reference manifest SHA-256" >&2
        exit 2
    fi
    python3 "${reference_verifier}" \
        "${reference_fixture_directory}" \
        "${SPARK_DSV4_REFERENCE_MANIFEST_SHA256}"
    export SPARK_DSV4_REFERENCE_TOKEN_PATH="${reference_fixture_directory}/prompt_tokens.u32le"
    export SPARK_DSV4_REFERENCE_OUTPUT_PATH="${reference_fixture_directory}/after_layer_2.bf16le"
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
validation_defines="${SPARK_DSV4_VALIDATION_DEFINES:-}"
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "DSV4 hardware validation admits only CUDA_ARCH=sm_121a" >&2
    exit 2
fi
if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for DSV4 hardware validation" >&2
    exit 2
fi

make -C "${repository_root}" \
    build/libsparkpipe_core.a \
    build/libsparkpipe_runtime.a

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --expt-relaxed-constexpr \
    -gencode arch=compute_121a,code=sm_121a \
    "-DSPARK_BATCH_BUCKET=${batch_bucket}" \
    ${validation_defines:+"${validation_defines}"} \
    -I"${repository_root}/include" \
    -I"${repository_root}/model-families/dsv4/include" \
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
    -o "${validation_directory}/dsv4_resident_decode_stage_validator"

"${validation_directory}/dsv4_resident_decode_stage_validator" "${configuration_hash}"
