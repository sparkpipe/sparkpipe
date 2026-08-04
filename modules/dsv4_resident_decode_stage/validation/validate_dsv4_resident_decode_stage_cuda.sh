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
trap 'rm -rf "${validation_directory}"' EXIT

if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ "${SPARK_DSV4_ALLOW_UNQUALIFIED_EXECUTION:-0}" != "1" ]]; then
    echo "set SPARK_DSV4_ALLOW_UNQUALIFIED_EXECUTION=1 for the explicit DSV4 bring-up gate" >&2
    exit 2
fi
if [[ -z "${SPARK_DSV4_STAGE_PACK_PATH:-}" || ! -s "${SPARK_DSV4_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_DSV4_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
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
    -I"${repository_root}/include" \
    -I"${repository_root}/model-families/dsv4/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${script_directory}/spark_dsv4_resident_decode_stage_cuda_validation.cu" \
    "${module_archive}" \
    "${repository_root}/build/libsparkpipe_runtime.a" \
    "${repository_root}/build/libsparkpipe_core.a" \
    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" \
    -lcudart \
    -ldl \
    -lm \
    -Xcompiler -pthread \
    -o "${validation_directory}/dsv4_resident_decode_stage_validator"

"${validation_directory}/dsv4_resident_decode_stage_validator" "${configuration_hash}"
