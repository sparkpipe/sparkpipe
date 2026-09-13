#!/usr/bin/env bash
#
# W7 end-to-end parity on the REAL DFlash2 weights: the reference's full forward
# (5 conv-wrapped layers on the real drafter checkpoint + the target's shared
# lm_head) into the module's own selector emit sequence, drafts compared.
#
# usage: validate_qwen36_dflash2_selector_real_cuda.sh [OUTPUT_DIRECTORY]
# needs: SPARK_QWEN36_DFLASH2_DRAFTER and SPARK_QWEN36_TARGET pointing at the
#        checkpoints on this host, CUDA 13 nvcc, one sm_121a device, ~3 GB disk.
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
output_directory="${1:-${repository_root}/build/dflash2/real}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"

mkdir -p "${output_directory}"
echo "== reference full forward + selector expectations (real weights)"
python3 "${repository_root}/tools/qwen36_dflash2_selector_real_case.py" "${output_directory}"

echo "== compiling for ${cuda_architecture}"
"${nvcc_binary}" -std=c++17 -gencode "arch=compute_${cuda_architecture#sm_},code=${cuda_architecture}" \
    --expt-relaxed-constexpr -O3 \
    -I"${repository_root}" -I"${repository_root}/include" -I"${repository_root}/deployment/include" \
    -I"${repository_root}/model-families/common/include" -I"${repository_root}/model-families/qwen36/include" \
    -I"${module_directory}/include" -I"${module_directory}/source" \
    "${script_directory}/spark_qwen36_dflash2_selector_real_validation.cu" \
    "${module_directory}/source/spark_qwen36_resident_decode_stage_cuda.cu" \
    -o "${output_directory}/real_validation"

echo "== running"
"${output_directory}/real_validation" "${output_directory}"
