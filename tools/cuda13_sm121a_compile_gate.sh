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
cuda_home="$(dirname "$(dirname "$(command -v "${nvcc_binary}")")")"
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
	-I"${repository_root}/model-families/k3/include"
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
)
for relative_source in "${translation_units[@]}"; do
	artifact_name="${relative_source//\//__}"
	artifact_name="${artifact_name%.cu}"
	compile_cuda "${relative_source}" "${artifact_name}"
done

for direct_mode in 0 1; do
	compile_cuda ring/transport/rdma.cu "rdma_mode_${direct_mode}" \
		-DSPARK_HIDDEN_SPARK_RDMA_DEVICE_DIRECT="${direct_mode}"
done

make_variable()
{
	make -s --no-print-directory -C "${repository_root}" -f Makefile \
		-f <(printf 'print-%%:\n\t@echo $(%s)\n' '$*') "print-$1" \
		NVCC="${nvcc_binary}" CUDA_ARCH=sm_121a CUDA_HOME="${cuda_home}"
}

adapters=()
while IFS= read -r variable; do
	adapters+=("$(make_variable "${variable}")")
done < <(sed -n 's/^\([A-Z0-9_]*_SERVING_ADAPTER\)[[:space:]]*:=.*/\1/p' "${repository_root}/Makefile")
if [[ "${#adapters[@]}" -eq 0 ]]; then
	echo "the Makefile names no serving adapters" >&2
	exit 5
fi
cuda_programs=(
	build/sparkpipe_model_residentd
	build/sparkpipe_weightd
	build/k3_tp_context_cuda
)
if ! make -C "${repository_root}" -j"$(nproc)" "${adapters[@]}" "${cuda_programs[@]}" \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/adapters.txt" 2>&1; then
	echo "the serving adapters or CUDA programs do not build for sm_121a; see logs/adapters.txt" >&2
	exit 5
fi
for artifact in "${adapters[@]}" build/k3_tp_context_cuda; do
	if [[ ! -f "${repository_root}/${artifact}" ]]; then
		echo "build produced no ${artifact}" >&2
		exit 5
	fi
	elf_listing="$(cuobjdump --list-elf "${repository_root}/${artifact}" 2>/dev/null || true)"
	if ! grep -q 'sm_121a' <<<"${elf_listing}"; then
		echo "CUDA artifact missing sm_121a target: ${artifact}" >&2
		exit 4
	fi
	printf '%s\n' "${elf_listing}" > "${output_directory}/logs/$(basename "${artifact}").elf.txt"
done
printf '%s\n' "${adapters[@]}" > "${output_directory}/adapters.txt"

for object_file in "${output_directory}"/objects/*.o; do
	object_name="$(basename "${object_file}")"
	elf_listing="$(cuobjdump --list-elf "${object_file}" 2>/dev/null || true)"
	if [[ -n "${elf_listing}" ]] && ! grep -q 'sm_121a' <<<"${elf_listing}"; then
		echo "CUDA object missing sm_121a target: ${object_file}" >&2
		exit 4
	fi
	printf '%s\n' "${elf_listing}" > "${output_directory}/logs/${object_name}.elf.txt"
	cuobjdump --dump-resource-usage "${object_file}" > \
		"${output_directory}/logs/${object_name}.resources.txt" 2>&1 || true
done

(
	cd "${output_directory}"
	find . -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > SHA256SUMS
)

echo "PASS CUDA 13 exact sm_121a compile gate (${#adapters[@]} serving adapters)"
