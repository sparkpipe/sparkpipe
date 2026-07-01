#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 && $# -ne 3 ]]; then
    echo "usage: $0 MAX_STAGE_MICROSECONDS MODULE_ARCHIVE [DRIVER_SO]" >&2
    exit 2
fi

maximum_stage_microseconds="$1"
module_archive="$2"
driver_path="${3:-}"
model_directory="${GLM52_MODEL_DIR:-}"
allow_remote_model_directory="${GLM52_ALLOW_REMOTE_MODEL_DIR:-0}"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
validation_directory="$(mktemp -d)"
trap 'rm -rf "${validation_directory}"' EXIT
common_target="build/libsparkpipe_common.a"
runtime_target="build/libsparkpipe_runtime.a"
compiler_target="build/libsparkpipe_compiler.a"
common_archive="${repository_root}/build/libsparkpipe_common.a"
runtime_archive="${repository_root}/build/libsparkpipe_runtime.a"
compiler_archive="${repository_root}/build/libsparkpipe_compiler.a"

if [[ "${maximum_stage_microseconds}" == "0" ]]; then
    echo "MAX_STAGE_MICROSECONDS must be nonzero" >&2
    exit 2
fi
if [[ -z "${model_directory}" ]]; then
    echo "set GLM52_MODEL_DIR to the local Spark NVMe GLM artifact directory" >&2
    exit 2
fi
if [[ "${allow_remote_model_directory}" != "1" ]]; then
    case "${model_directory}" in
        /mnt/mac/*|/Volumes/*)
            echo "GLM52 hardware validation requires local Spark NVMe artifacts, not remote model source: ${model_directory}" >&2
            echo "set GLM52_ALLOW_REMOTE_MODEL_DIR=1 only for non-performance debugging" >&2
            exit 2
            ;;
    esac
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "module archive is missing or empty" >&2
    exit 2
fi
if [[ -n "${driver_path}" && ! -s "${driver_path}" ]]; then
    echo "driver shared object is missing or empty" >&2
    exit 2
fi

nvcc_path="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
active_sequence_count="${GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT:-1}"

if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
    echo "nvcc unavailable for hardware validation" >&2
    exit 2
fi
if [[ "${cuda_architecture}" != "sm_121a" ]]; then
    echo "this validator admits only sm_121a required-CUDA artifacts" >&2
    exit 2
fi
if ! [[ "${active_sequence_count}" =~ ^[0-9]+$ ]] || [[ "${active_sequence_count}" == "0" ]]; then
    echo "GLM52_VALIDATION_ACTIVE_SEQUENCE_COUNT must be a positive integer" >&2
    exit 2
fi
required_cuda_link_args=()
if [[ -n "${GLM52_REQUIRED_CUDA_LINK_ARGS:-}" ]]; then
    read -r -a required_cuda_link_args <<< "${GLM52_REQUIRED_CUDA_LINK_ARGS}"
fi
required_cuda_library_path=""
for required_cuda_link_arg in "${required_cuda_link_args[@]}"; do
    if [[ "${required_cuda_link_arg}" == *.so ]]; then
        required_cuda_library_directory="$(cd "$(dirname "${required_cuda_link_arg}")" && pwd)"
        if [[ -z "${required_cuda_library_path}" ]]; then
            required_cuda_library_path="${required_cuda_library_directory}"
        else
            required_cuda_library_path="${required_cuda_library_path}:${required_cuda_library_directory}"
        fi
    fi
done
if [[ -n "${required_cuda_library_path}" ]]; then
    export LD_LIBRARY_PATH="${required_cuda_library_path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

make -C "${repository_root}" "${common_target}" "${runtime_target}" "${compiler_target}"

"${nvcc_path}" \
    -std=c++17 \
    -O3 \
    --use_fast_math \
    -arch="${cuda_architecture}" \
    -DSPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT="${active_sequence_count}u" \
    -I"${repository_root}/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${script_directory}/spark_glm52_resident_decode_stage_cuda_validation.cu" \
    "${module_archive}" \
    "${runtime_archive}" \
    "${compiler_archive}" \
    "${common_archive}" \
    "${required_cuda_link_args[@]}" \
    -lcublasLt \
    -lcublas \
    -ldl \
    -o "${validation_directory}/glm52_resident_decode_stage_validator"

if [[ -n "${driver_path}" ]]; then
    "${validation_directory}/glm52_resident_decode_stage_validator" \
        "${maximum_stage_microseconds}" \
        "${driver_path}"
else
    "${validation_directory}/glm52_resident_decode_stage_validator" \
        "${maximum_stage_microseconds}"
fi
