#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="${SPARK_CUDA_GATE_OUTPUT_DIRECTORY:-${repository_root}/build/cuda13_sm121a_gate}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"

if [[ "${cuda_architecture}" != "sm_121a" ]]; then
	echo "CUDA gate requires CUDA_ARCH=sm_121a, got ${cuda_architecture}" >&2
	exit 2
fi
if ! command -v "${nvcc_binary}" >/dev/null 2>&1; then
	echo "CUDA gate requires nvcc from CUDA 13" >&2
	exit 2
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
	echo "CUDA gate requires cuobjdump for exact-architecture validation" >&2
	exit 2
fi
if [[ ! -f /usr/include/infiniband/verbs.h ]]; then
	echo "CUDA gate requires libibverbs development headers" >&2
	exit 2
fi

nvcc_version="$(${nvcc_binary} --version)"
if ! grep -Eq 'release 13\.' <<<"${nvcc_version}"; then
	echo "CUDA gate requires CUDA 13.x" >&2
	printf '%s\n' "${nvcc_version}" >&2
	exit 2
fi

python3 "${repository_root}/tools/glm52_model_contract.py" --check
python3 "${repository_root}/tools/generate_dsv4_contracts.py" --check

rm -rf "${output_directory}"
mkdir -p "${output_directory}/objects" "${output_directory}/ptx" "${output_directory}/logs"
printf '%s\n' "${nvcc_version}" > "${output_directory}/nvcc-version.txt"
printf 'CUDA_ARCH=%s\n' "${cuda_architecture}" > "${output_directory}/configuration.txt"

cat > "${output_directory}/probe.cu" <<'PROBE'
#include <cuda_runtime.h>
#include <cstdint>

__global__ void SparkSm121aProbe(float *output, const float *input)
{
	uint32_t index;

	index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = input[index] * 2.0f;
}
PROBE

include_flags=(
	-I"${repository_root}"
	-I"${repository_root}/include"
	-I"${repository_root}/deployment/include"
	-I"${repository_root}/model-families/common/include"
	-I"${repository_root}/model-families/glm52/include"
	-I"${repository_root}/model-families/qwen36/include"
	-I"${repository_root}/model-families/dsv4/include"
	-I"${repository_root}/model-families/k3/include"
	-I"${repository_root}/model-families/mimo25/include"
	-I"${repository_root}/modules/glm52_resident_decode_stage/include"
	-I"${repository_root}/modules/glm52_resident_decode_stage/source"
	-I"${repository_root}/modules/glm52_dspark_draft_backend/include"
	-I"${repository_root}/modules/dsv4_resident_decode_stage/include"
	-I"${repository_root}/modules/dsv4_resident_decode_stage/source"
)
object_flags=(
	-std=c++17
	-gencode
	arch=compute_121a,code=sm_121a
	--expt-relaxed-constexpr
	-lineinfo
	-Xptxas=-v
)
ptx_flags=(
	-std=c++17
	-arch=compute_121a
	--expt-relaxed-constexpr
)

compile_cuda()
{
	local relative_source="$1"
	local artifact_name="$2"
	shift 2
	local source_path
	local object_path="${output_directory}/objects/${artifact_name}.o"
	local ptx_path="${output_directory}/ptx/${artifact_name}.compute_121a.ptx"

	if [[ "${relative_source}" = /* ]]; then
		source_path="${relative_source}"
	else
		source_path="${repository_root}/${relative_source}"
	fi
	if [[ ! -f "${source_path}" ]]; then
		echo "required CUDA translation unit missing: ${relative_source}" >&2
		exit 3
	fi
	"${nvcc_binary}" "${object_flags[@]}" "${include_flags[@]}" "$@" \
		-c "${source_path}" -o "${object_path}" \
		2> "${output_directory}/logs/${artifact_name}.ptxas.txt"
	"${nvcc_binary}" "${ptx_flags[@]}" "${include_flags[@]}" "$@" \
		-ptx "${source_path}" -o "${ptx_path}"
	if ! grep -Eq '^\.target[[:space:]]+sm_121a' "${ptx_path}"; then
		echo "architecture-specific PTX target missing: ${relative_source}" >&2
		exit 4
	fi
}

compile_cuda "${output_directory}/probe.cu" probe

translation_units=(
	tools/hardware/spark_cuda_characterize.cu
	tools/hardware/spark_nvme_characterize.cu
	inference/llms/kimi_k3/bind.cu
	inference/llms/kimi_k3/unity.cu
	inference/llms/mimo_2_5/bind.cu
	inference/llms/mimo_2_5/unity.cu
	inference/llms/qwen_3_6/bind.cu
	inference/llms/qwen_3_6/unity.cu
	modules/glm52_dspark_draft_backend/source/spark_glm52_dspark_draft_backend.cu
	modules/glm52_dspark_draft_backend/validation/validate_glm52_dspark_epoch3_cuda.cu
)
for relative_source in "${translation_units[@]}"; do
	artifact_name="${relative_source//\//__}"
	artifact_name="${artifact_name%.cu}"
	compile_cuda "${relative_source}" "${artifact_name}"
done

dsv4_model_header="${repository_root}/model-families/dsv4/include/sparkpipe/spark_dsv4_model.h"
compile_cuda \
	modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu \
	dsv4_resident_decode_stage \
	-include "${dsv4_model_header}" \
	-DSPARK_DSV4_MODULE_BUILD=1 \
	-DSPARK_BATCH_BUCKET=1024u
compile_cuda \
	modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu \
	dsv4_resident_decode_stage_validation \
	-include "${dsv4_model_header}" \
	-DSPARK_DSV4_MODULE_BUILD=1 \
	-DSPARK_BATCH_BUCKET=1024u

glm_model_header="${repository_root}/model-families/glm52/include/sparkpipe/spark_glm52_model.h"
glm_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)
glm_codec_ids=(2 3 4 5 6 7)
for codec_index in "${!glm_codecs[@]}"; do
	codec="${glm_codecs[${codec_index}]}"
	codec_id="${glm_codec_ids[${codec_index}]}"
	read -r model_revision contract_sha256 < <(
		python3 "${repository_root}/tools/glm52_model_contract.py" \
			--print-build-identity "${codec}"
	)
	compile_cuda \
		modules/glm52_resident_decode_stage/source/spark_glm52_resident_decode_stage_cuda.cu \
		"glm52_resident_decode_stage_${codec}" \
		-include "${glm_model_header}" \
		-DGLM52_EXPERT_WEIGHT_CODEC="${codec_id}" \
		-DGLM52_EXPERT_CODEC_NAME=\""${codec}"\" \
		-DGLM52_MODEL_REVISION=\""${model_revision}"\" \
		-DGLM52_CONTRACT_SHA256=\""${contract_sha256}"\"
done

for direct_mode in 0 1; do
	compile_cuda ring/transport/rdma.cu "rdma_mode_${direct_mode}" \
		-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT="${direct_mode}"
done

for codec in "${glm_codecs[@]}"; do
	read -r model_revision contract_sha256 < <(
		python3 "${repository_root}/tools/glm52_model_contract.py" \
			--print-build-identity "${codec}"
	)
	make -C "${repository_root}/modules/glm52_resident_decode_stage" clean \
		EXPERT_CODEC="${codec}" \
		MODEL_REVISION="${model_revision}" \
		CONTRACT_SHA256="${contract_sha256}" \
		NVCC="${nvcc_binary}" \
		CUDA_ARCH=sm_121a \
		> "${output_directory}/logs/glm52-${codec}-archive.txt" 2>&1
	make -C "${repository_root}/modules/glm52_resident_decode_stage" \
		-j2 archive \
		EXPERT_CODEC="${codec}" \
		MODEL_REVISION="${model_revision}" \
		CONTRACT_SHA256="${contract_sha256}" \
		NVCC="${nvcc_binary}" \
		CUDA_ARCH=sm_121a \
		>> "${output_directory}/logs/glm52-${codec}-archive.txt" 2>&1
done

make -C "${repository_root}/modules/dsv4_resident_decode_stage" clean \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	> "${output_directory}/logs/dsv4-archive.txt" 2>&1
make -C "${repository_root}/modules/dsv4_resident_decode_stage" \
	-j2 archive \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	>> "${output_directory}/logs/dsv4-archive.txt" 2>&1

while IFS= read -r -d '' object_file; do
	elf_listing="$(cuobjdump --list-elf "${object_file}" 2>/dev/null || true)"
	if [[ -n "${elf_listing}" ]] && ! grep -q 'sm_121a' <<<"${elf_listing}"; then
		echo "CUDA object missing sm_121a target: ${object_file}" >&2
		exit 4
	fi
done < <(find \
	"${output_directory}/objects" \
	"${repository_root}/build/modules/glm52_resident_decode_stage" \
	"${repository_root}/build/modules/dsv4_resident_decode_stage" \
	-type f -name '*.o' -print0)

for object_file in "${output_directory}"/objects/*.o; do
	object_name="$(basename "${object_file}")"
	cuobjdump --list-elf "${object_file}" > \
		"${output_directory}/logs/${object_name}.elf.txt" 2>&1 || true
	cuobjdump --dump-resource-usage "${object_file}" > \
		"${output_directory}/logs/${object_name}.resources.txt" 2>&1 || true
done

(
	cd "${output_directory}"
	find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "PASS CUDA 13 exact sm_121a compile gate"
