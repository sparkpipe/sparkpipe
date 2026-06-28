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

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121}"

make -C "${repository_root}" "${common_archive}"

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --use_fast_math \
    -arch="${cuda_architecture}" \
    -I"${repository_root}/include" \
    -I"${module_directory}/include" \
    "${script_directory}/spark_glm52_resident_sparse_mla_cuda_validation.cu" \
    "${module_archive}" \
    "${common_archive}" \
    -o "${validation_directory}/glm52_resident_sparse_mla_validator"

"${validation_directory}/glm52_resident_sparse_mla_validator" \
    "${maximum_stage_microseconds}"
