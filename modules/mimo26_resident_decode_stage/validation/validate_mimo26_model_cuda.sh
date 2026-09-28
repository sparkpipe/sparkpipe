#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
checkpoint="${1:?checkpoint directory}"
work="${2:?work directory on local NVMe}"
shift 2
nvcc_binary="${NVCC:-nvcc}"
binary="${work}/mimo26_model_cuda"
mkdir -p "${work}"
"${nvcc_binary}" -O3 -std=c++17 -gencode arch=compute_121a,code=sm_121a \
	-I"${repository_root}" -I"${repository_root}/include" \
	-I"${repository_root}/model-families/common/include" -I"${repository_root}/model-families/mimo26/include" \
	"${repository_root}/modules/mimo26_resident_decode_stage/validation/mimo26_model_cuda.cu" -o "${binary}"
weights=""
for prompt in "$@"; do
	fixture="${repository_root}/qualification/t1_reference/mimo26/${prompt}.t1r"
	if [[ -z "${weights}" ]]; then
		python3 "${repository_root}/tools/mimo26_model_inputs.py" --checkpoint "${checkpoint}" --fixture "${fixture}" --out "${work}/${prompt}"
		weights="${work}/${prompt}"
	else
		python3 "${repository_root}/tools/mimo26_model_inputs.py" --checkpoint "${checkpoint}" --fixture "${fixture}" --out "${work}/${prompt}" --weights-from "${weights}"
	fi
	"${binary}" "${work}/${prompt}"
done
