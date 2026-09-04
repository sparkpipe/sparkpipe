#!/usr/bin/env bash
# Module-publish validator wrapper for qwen38_max (the dsv4 pattern).
# sparkpipe_module_publish calls: wrapper CONFIGURATION_SHA ARCHIVE.
# The pack and tier come from the environment (the publish recipe exports
# the module's RUNTIME_CONFIGURATION), so one wrapper serves every tier.
set -euo pipefail

CONFIGURATION_SHA="${1:?usage: wrapper CONFIGURATION_SHA ARCHIVE}"
ARCHIVE="${2:?}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"
export CUDA_ARCH="${CUDA_ARCH:-sm_121a}"
export SPARK_QWEN38_MAX_CUDA_VALIDATOR_SHA256="$(sha256sum "${ROOT}/modules/qwen38_max_resident_decode_stage/validation/spark_qwen38_max_resident_decode_stage_cuda_validation.cu" | cut -d' ' -f1)"

exec "${ROOT}/modules/qwen38_max_resident_decode_stage/validation/validate_qwen38_max_resident_decode_stage_cuda.sh" \
    "${CONFIGURATION_SHA}" "${ARCHIVE}"
