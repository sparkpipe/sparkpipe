#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 MAX_STAGE_MICROSECONDS MODULE_ARCHIVE" >&2
    exit 2
fi

maximum_stage_microseconds="$1"
module_archive="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
trap 'rm -rf "${validation_directory}"' EXIT
common_archive="${repository_root}/build/libsparkpipe_common.a"

if [[ "${maximum_stage_microseconds}" == "0" ]]; then
    echo "MAX_STAGE_MICROSECONDS must be nonzero" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121}"

if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for hardware validation" >&2
    exit 2
fi
if [[ "${cuda_architecture}" != "sm_121" ]]; then
    echo "this validator admits only sm_121 artifacts" >&2
    exit 2
fi

make -C "${repository_root}" "${common_archive}"

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --use_fast_math \
    -arch="${cuda_architecture}" \
    -I"${repository_root}/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${script_directory}/spark_glm52_resident_decode_stage_cuda_validation.cu" \
    "${module_archive}" \
    "${common_archive}" \
    -o "${validation_directory}/glm52_resident_decode_stage_validator"

"${validation_directory}/glm52_resident_decode_stage_validator" \
    "${maximum_stage_microseconds}"
