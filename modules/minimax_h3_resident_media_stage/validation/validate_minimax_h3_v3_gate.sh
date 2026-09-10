#!/usr/bin/env bash
set -euo pipefail

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
module_directory="$(cd "${script_directory}/.." && pwd)"
repository_root="$(cd "${module_directory}/../.." && pwd)"
work="${MINIMAX_H3_V3_WORK:-${TMPDIR:-/tmp}/minimax_h3_v3_gate}"
warm="${MINIMAX_H3_V3_WARM:-/mnt/model-warm/minimax-h3}"
fixtures="${module_directory}/validation/fixtures/real/v3"
nvcc_bin="${MINIMAX_H3_NVCC:-$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)}"
arch="${MINIMAX_H3_CUDA_ARCH:-sm_121a}"

rm -rf "${work}"
mkdir -p "${work}/weights"

names=""
for block in 0 1; do
	for tensor in "attn.to_q.weight" "attn.to_k.weight" "attn.to_v.weight" \
		"attn.to_out.0.weight" "attn.norm_q.weight" "attn.norm_k.weight" \
		"ff.net.0.proj.weight" "ff.net.2.weight" "norm1.weight" "norm2.weight"; do
		names="${names}transformer_blocks.${block}.${tensor},"
	done
done
names="${names%,}"

python3 "${repository_root}/tools/minimax_h3_extract_tensors.py" \
	--component-dir "${warm}/transformer" --names "${names}" --out "${work}/weights"

"${nvcc_bin}" -O2 -std=c++17 \
	-gencode arch="${MINIMAX_H3_CUDA_COMPUTE_ARCH:-compute_121}",code="${arch}" \
	-I"${repository_root}/include" \
	-I"${repository_root}/model-families/minimax_h3/include" \
	-I"${repository_root}/model-families/common/include" \
	-I"${repository_root}/src" \
	-I"${repository_root}" \
	-o "${work}/minimax_h3_v3_gate" \
	"${module_directory}/source/spark_minimax_h3_resident_media_stage_cuda.cu" \
	"${script_directory}/spark_minimax_h3_v3_gate.cu"

"${work}/minimax_h3_v3_gate" "${fixtures}" "${work}/weights" "${work}/weights/manifest.txt"
