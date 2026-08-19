#!/usr/bin/env bash
#
# DFlash2 selector HOST-PATH parity (adoption item W7): the module's own inline
# emit sequence vs the numpy oracle's walk.
#
# usage: validate_qwen36_dflash2_selector_host_cuda.sh [smoke|mid|full|tie] [OUTPUT_DIRECTORY]
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
if ! "${nvcc_binary}" --version | grep -Eq 'release 13\.'; then
    echo "DFlash2 host-path validation requires CUDA 13.x nvcc" >&2
    exit 2
fi

mkdir -p "${output_directory}"
case_file="${output_directory}/dflash2_selector_case_${scale}.bin"
binary="${output_directory}/spark_qwen36_dflash2_selector_host_validation"

echo "== generating the oracle case (${scale})"
python3 "${repository_root}/tools/qwen36_dflash2_selector_case.py" --scale "${scale}" --output "${case_file}" >/dev/null

echo "== compiling the host-path validator for ${cuda_architecture}"
"${nvcc_binary}" -std=c++17 -gencode "arch=compute_${cuda_architecture#sm_},code=${cuda_architecture}" \
    --expt-relaxed-constexpr -O3 \
    -I"${repository_root}" -I"${repository_root}/include" -I"${repository_root}/deployment/include" \
    -I"${repository_root}/model-families/common/include" -I"${repository_root}/model-families/qwen36/include" \
    -I"${module_directory}/include" -I"${module_directory}/source" \
    "${script_directory}/spark_qwen36_dflash2_selector_host_validation.cu" \
    "${module_directory}/source/spark_qwen36_resident_decode_stage_cuda.cu" \
    -o "${binary}"

echo "== running"
"${binary}" "${case_file}"
