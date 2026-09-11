#!/usr/bin/env bash
set -euo pipefail

CONFIGURATION_SHA="${1:?usage: wrapper CONFIGURATION_SHA ARCHIVE}"
ARCHIVE="${2:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
export CUDA_ARCH="${CUDA_ARCH:-sm_121a}"
export SPARK_MUSE_GLIMMER_CUDA_VALIDATOR_SHA256="$(sha256sum "${ROOT}/modules/muse_glimmer_resident_decode_stage/validation/spark_muse_glimmer_resident_decode_stage_cuda_validation.cu" | cut -d' ' -f1)"

exec "${ROOT}/modules/muse_glimmer_resident_decode_stage/validation/validate_muse_glimmer_resident_decode_stage_cuda.sh" \
    "${CONFIGURATION_SHA}" "${ARCHIVE}"
