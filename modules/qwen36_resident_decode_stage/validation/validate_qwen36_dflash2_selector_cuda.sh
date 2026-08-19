#!/usr/bin/env bash
#
# DFlash2 candidate-selector parity: numpy oracle vs the CUDA kernels.
#
# usage: validate_qwen36_dflash2_selector_cuda.sh [smoke|mid|full|tie] [OUTPUT_DIRECTORY]
#
# The "tie" scale is the rule discriminator: its logits are large enough that
# the BF16 truncation collapses distinct values, its head rows are paired so
# every logit ties exactly, and its codebook span is zero so each lattice row
# collapses onto those tied unary logits. It is run with
# --require-discriminating, so it FAILS if it ever stops separating
# truncate-then-select from select-then-truncate, or strict-greater first-max
# from a >= walk.
#
# Builds the case with tools/qwen36_dflash2_selector_case.py (which runs the
# oracles in tools/qwen36_dspark_reference.py), compiles the module's CUDA
# translation unit plus this validator for sm_121a, and runs the comparison.
# Requires CUDA 13 nvcc and one sm_121a device; nothing else.
set -euo pipefail

scale="${1:-smoke}"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
output_directory="${2:-${repository_root}/build/dflash2}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"

case "${scale}" in
smoke|mid|full|tie) ;;
*) echo "scale must be smoke, mid, full or tie" >&2; exit 2 ;;
esac
validator_flags=()
if [[ "${scale}" == "tie" ]]; then
    validator_flags+=(--require-discriminating)
fi
if ! command -v "${nvcc_binary}" >/dev/null 2>&1; then
    echo "DFlash2 selector validation requires nvcc from CUDA 13" >&2
    exit 2
fi
if ! "${nvcc_binary}" --version | grep -Eq 'release 13\.'; then
    echo "DFlash2 selector validation requires CUDA 13.x" >&2
    exit 2
fi

mkdir -p "${output_directory}"
case_file="${output_directory}/dflash2_selector_case_${scale}.bin"
binary="${output_directory}/spark_qwen36_dflash2_selector_validation"

echo "== generating the oracle case (${scale})"
python3 "${repository_root}/tools/qwen36_dflash2_selector_case.py" --scale "${scale}" --output "${case_file}"

echo "== compiling for ${cuda_architecture}"
"${nvcc_binary}" -std=c++17 -gencode "arch=compute_${cuda_architecture#sm_},code=${cuda_architecture}" \
    --expt-relaxed-constexpr -lineinfo -O3 \
    -I"${repository_root}" \
    -I"${repository_root}/include" \
    -I"${repository_root}/deployment/include" \
    -I"${repository_root}/model-families/common/include" \
    -I"${repository_root}/model-families/qwen36/include" \
    -I"${module_directory}/include" \
    -I"${module_directory}/source" \
    "${script_directory}/spark_qwen36_dflash2_selector_validation.cu" \
    "${module_directory}/source/spark_qwen36_resident_decode_stage_cuda.cu" \
    -o "${binary}"

echo "== running"
"${binary}" "${case_file}" "${validator_flags[@]+${validator_flags[@]}}"
