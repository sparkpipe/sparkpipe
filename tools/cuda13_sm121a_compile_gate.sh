#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="${SPARK_CUDA_GATE_OUTPUT_DIRECTORY:-${repository_root}/build/cuda13_sm121a_gate}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"

"${repository_root}/tools/source_package_gate.sh"

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
	-I"${repository_root}/model-families/qwen38_27b/include"
	-I"${repository_root}/model-families/dsv4/include"
	-I"${repository_root}/model-families/k3/include"
	-I"${repository_root}/model-families/mimo25/include"
	-I"${repository_root}/modules/glm52_resident_decode_stage/include"
	-I"${repository_root}/modules/glm52_resident_decode_stage/source"
	-I"${repository_root}/modules/glm52_dspark_draft_backend/include"
	-I"${repository_root}/modules/dsv4_resident_decode_stage/include"
	-I"${repository_root}/modules/dsv4_resident_decode_stage/source"
	-I"${repository_root}/modules/qwen38_27b_resident_decode_stage/include"
	-I"${repository_root}/modules/qwen38_27b_resident_decode_stage/source"
	-I"${repository_root}/modules/minimax_resident_decode_stage/include"
	-I"${repository_root}/modules/minimax_resident_decode_stage/source"
	-I"${repository_root}/model-families/minimax/include"
	-I"${repository_root}/model-families/minimax/include/sparkpipe"
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
	tools/mb_doorbell.cu
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

qwen38_27b_model_header="${repository_root}/model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_model.h"
compile_cuda \
	modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu \
	qwen38_27b_resident_decode_stage \
	-include "${qwen38_27b_model_header}" \
	-DSPARK_QWEN38_27B_MODULE_BUILD=1

minimax_model_header="${repository_root}/modules/minimax_resident_decode_stage/include/sparkpipe/spark_minimax_model.h"
compile_cuda \
	modules/minimax_resident_decode_stage/source/spark_minimax_resident_decode_stage_cuda.cu \
	minimax_resident_decode_stage \
	-include "${minimax_model_header}" \
	-DSPARK_MINIMAX_MODULE_BUILD=1

# GLM 5.3 Flash uses the glm5_next implementation, separate from glm52.
make -C "${repository_root}" -j2 build/glm5_next_driver_probe \
	CUDA_HOME="$(dirname "$(dirname "$(command -v "${nvcc_binary}")")")" \
	> "${output_directory}/logs/glm5-next-driver-probe.txt" 2>&1
python3 "${repository_root}/tests/test_glm5_next_driver_probe.py"
glm5_next_gpu_tests=(
	build/test_glm5_next_head_offset
	build/test_glm5_next_mtp_join
	build/test_glm5_next_hc_mix
	build/test_glm5_next_l2_prefetch
	build/test_glm5_next_index_cp
	build/test_glm5_next_rows_kernels
)
if ! make -C "${repository_root}" -j2 "${glm5_next_gpu_tests[@]}" \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="$(dirname "$(dirname "$(command -v "${nvcc_binary}")")")" \
	> "${output_directory}/logs/glm5-next-gpu-tests.txt" 2>&1; then
	echo "glm5_next GPU tests do not build for sm_121a; see logs/glm5-next-gpu-tests.txt" >&2
	exit 5
fi
for gpu_test in "${glm5_next_gpu_tests[@]}"; do
	elf_listing="$(cuobjdump --list-elf "${repository_root}/${gpu_test}")"
	if ! grep -q 'sm_121a' <<<"${elf_listing}"; then
		echo "CUDA test binary missing sm_121a target: ${gpu_test}" >&2
		exit 4
	fi
done
compile_cuda \
	modules/glm5_next_resident_decode_stage/source/spark_glm5_next_resident_decode_stage_cuda.cu \
	glm5_next_resident_decode_stage_fp8 \
	-I"${repository_root}/model-families/glm5_next/include" \
	-I"${repository_root}/modules/glm5_next_resident_decode_stage/include" \
	-I"${repository_root}/modules/glm5_next_resident_decode_stage/source" \
	-include "${repository_root}/model-families/glm5_next/include/sparkpipe/spark_glm5_next_model.h" \
	-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5 \
	'-DGLM5_NEXT_EXPERT_CODEC_NAME="fp8"' \
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
		-DGLM_EXPERT_WEIGHT_CODEC="${codec_id}" \
		-DGLM_EXPERT_CODEC_NAME=\""${codec}"\" \
		-DGLM_MODEL_REVISION=\""${model_revision}"\" \
		-DGLM_CONTRACT_SHA256=\""${contract_sha256}"\" \
		-DSPARK_BATCH_BUCKET=1024u
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

make -C "${repository_root}/modules/minimax_resident_decode_stage" clean \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	> "${output_directory}/logs/minimax-archive.txt" 2>&1
make -C "${repository_root}/modules/minimax_resident_decode_stage" \
	-j2 archive \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	>> "${output_directory}/logs/minimax-archive.txt" 2>&1

cuda_home="$(dirname "$(dirname "$(command -v "${nvcc_binary}")")")"
make -C "${repository_root}" -j2 \
	build/libsparkpipe_core.a build/libsparkpipe_runtime.a build/libsparkpipe_model_common.a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/runtime-libraries.txt" 2>&1

module_codec()
{
	case "$1" in
	gemma4_resident_decode_stage|hy4_resident_decode_stage|minimax_resident_decode_stage|muse_glimmer_resident_decode_stage)
		echo bf16
		;;
	*)
		echo fp8
		;;
	esac
}

link_module()
{
	local module_name="$1"
	local archive_path="$2"
	local entry_prefix entry_source
	entry_prefix="$(sed -n 's/^MODULE_ENTRY_PREFIX[[:space:]]*:=[[:space:]]*//p' "${repository_root}/modules/${module_name}/Makefile")"
	if [[ -z "${entry_prefix}" ]]; then
		echo "module ${module_name} names no MODULE_ENTRY_PREFIX" >&2
		exit 5
	fi
	entry_source="${output_directory}/modules/${module_name}/link_entries.c"
	{
		for entry in Initialize Execute Admit Snapshot Destroy; do
			printf 'extern int %s%s();\n' "${entry_prefix}" "${entry}"
		done
		printf 'void *SparkLinkGateEntries[] = {(void *)%sInitialize,(void *)%sExecute,(void *)%sAdmit,(void *)%sSnapshot,(void *)%sDestroy};\n' \
			"${entry_prefix}" "${entry_prefix}" "${entry_prefix}" "${entry_prefix}" "${entry_prefix}"
	} > "${entry_source}"
	if ! cc -shared -fPIC -Wl,-z,defs "${entry_source}" "${archive_path}" \
		"${repository_root}/build/libsparkpipe_model_common.a" \
		"${repository_root}/build/libsparkpipe_runtime.a" \
		"${repository_root}/build/libsparkpipe_core.a" \
		-L"${cuda_home}/lib64" -lcudart -lcuda -lstdc++ -ldl -lpthread -lm \
		-o "${output_directory}/modules/${module_name}/driver_link_check.so" \
		> "${output_directory}/logs/${module_name}-link.txt" 2>&1; then
		echo "module ${module_name} does not link as a driver; see logs/${module_name}-link.txt" >&2
		exit 5
	fi
}

for module_makefile in "${repository_root}"/modules/*_resident_decode_stage/Makefile; do
	module_name="$(basename "$(dirname "${module_makefile}")")"
	module_build="${output_directory}/modules/${module_name}"
	mkdir -p "${module_build}"
	if ! make -C "${repository_root}/modules/${module_name}" -j2 archive \
		EXPERT_CODEC="$(module_codec "${module_name}")" \
		MODEL_REVISION=sm121a-gate \
		CONTRACT_SHA256=0000000000000000000000000000000000000000000000000000000000000000 \
		NVCC="${nvcc_binary}" \
		CUDA_ARCH=sm_121a \
		BUILD_DIRECTORY="${module_build}" \
		> "${output_directory}/logs/${module_name}-archive.txt" 2>&1; then
		echo "module ${module_name} does not build for sm_121a; see logs/${module_name}-archive.txt" >&2
		exit 5
	fi
	link_module "${module_name}" "$(find "${module_build}" -maxdepth 1 -name '*.a' -print -quit)"
done
make -C "${repository_root}/modules/glm52_dspark_draft_backend" archive \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	BUILD_DIRECTORY="${output_directory}/modules/glm52_dspark_draft_backend" \
	> "${output_directory}/logs/glm52_dspark_draft_backend-archive.txt" 2>&1

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
	"${repository_root}/build/modules/minimax_resident_decode_stage" \
	"${output_directory}/modules" \
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
