#!/usr/bin/env bash
set -euo pipefail
repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_directory="${SPARK_CUDA_GATE_OUTPUT_DIRECTORY:-${repository_root}/build/cuda13_sm121a_gate}"
nvcc_binary="${NVCC:-nvcc}"
cuda_architecture="${CUDA_ARCH:-sm_121a}"
phase_wall="${SECONDS}"
phase_cpu=0
gate_phase()
{
	local stat children
	local -a fields
	read -r stat < "/proc/$$/stat"
	read -r -a fields <<<"${stat##*) }"
	children=$(( (fields[13] + fields[14]) / $(getconf CLK_TCK) ))
	echo "gate phase: $1: $(( SECONDS - phase_wall )) s wall, $(( children - phase_cpu )) s cpu"
	phase_wall="${SECONDS}"
	phase_cpu="${children}"
}
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
release_toolkit="13.0"
nvcc_version="$(${nvcc_binary} --version)"
if ! grep -Fq "release ${release_toolkit}," <<<"${nvcc_version}"; then
	echo "CUDA gate requires CUDA ${release_toolkit}, the release build hosts' toolkit, so that the spill audit checks what ships" >&2
	printf '%s\n' "${nvcc_version}" >&2
	exit 2
fi
python3 "${repository_root}/tools/glm52_model_contract.py" --check
cuda_home="$(dirname "$(dirname "$(command -v "${nvcc_binary}")")")"
gate_phase "source package, toolkit and model contracts"
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
	-I"${repository_root}/build"
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

make -C "${repository_root}" build/skl2_allreduce_policy.h build/skl2_k3_embedded.h build/skl2_glm5next_embedded.h NVCC="${nvcc_binary}" CUDA_ARCH="${cuda_architecture}" CUDA_HOME="${cuda_home}" >/dev/null
compile_pids=()
compile_cuda "${output_directory}/probe.cu" probe &
compile_pids+=("$!")

translation_units=(
	tools/hardware/spark_cuda_characterize.cu
	tools/hardware/spark_nvme_characterize.cu
	model-families/k3/source/bind.cu
	model-families/k3/source/unity.cu
)
for relative_source in "${translation_units[@]}"; do
	artifact_name="${relative_source//\//__}"
	artifact_name="${artifact_name%.cu}"
	compile_cuda "${relative_source}" "${artifact_name}" &
	compile_pids+=("$!")
done
for compile_pid in "${compile_pids[@]}"; do
	wait "${compile_pid}" || exit "$?"
done
gate_phase "exact-target translation units"

make_variable()
{
	make -s --no-print-directory -C "${repository_root}" -f Makefile \
		-f <(printf 'print-%%:\n\t@echo $(%s)\n' '$*') "print-$1" \
		NVCC="${nvcc_binary}" CUDA_ARCH=sm_121a CUDA_HOME="${cuda_home}"
}

adapters=()
while IFS= read -r variable; do
	adapters+=("$(make_variable "${variable}")")
done < <(sed -n 's/^\([A-Z0-9_]*_SERVING_ADAPTER\)[[:space:]]*:=.*/\1/p' "${repository_root}/Makefile" | grep -v '_METAL_SERVING_ADAPTER$')
if [[ "${#adapters[@]}" -eq 0 ]]; then
	echo "the Makefile names no serving adapters" >&2
	exit 5
fi
cuda_programs=(
	build/sparkpipe_model_residentd
	build/sparkpipe_weightd
	build/k3_tp_context_cuda
	build/weightd_warm
	build/runtime_libs/libsparkpipe_device_cuda.so
	"$(make_variable HIDDEN_TRANSPORT_SPARK_HOST_RDMA)"
)
if ! make -C "${repository_root}" -j"$(nproc)" "${adapters[@]}" "${cuda_programs[@]}" \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/adapters.txt" 2>&1; then
	echo "the serving adapters or CUDA programs do not build for sm_121a; see logs/adapters.txt" >&2
	exit 5
fi
gate_phase "adapters and CUDA programs"
read -r -a stage_runner_binaries <<<"$(make_variable STAGE_RUNNER_BINARIES)"
if ! make -C "${repository_root}" -j"$(nproc)" "${stage_runner_binaries[@]}" \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/stage_runner_binaries.txt" 2>&1; then
	grep -E 'undefined reference|(fatal )?error:|nvcc fatal|ptxas error|ld returned [1-9]' "${output_directory}/logs/stage_runner_binaries.txt" | head -n 20 >&2 || true
	echo "a stage-runner binary in STAGE_RUNNER_BINARIES does not build or link for sm_121a; see logs/stage_runner_binaries.txt" >&2
	exit 11
fi
gate_phase "stage-runner binaries"
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
audited=()
for artifact in "${adapters[@]}" "${cuda_programs[@]}"; do
	if [[ -f "${repository_root}/${artifact}" ]]; then
		audited+=("${repository_root}/${artifact}")
	fi
done
if ! python3 -I "${repository_root}/skl2/tools/dependent_wait_audit.py" "${audited[@]}" > "${output_directory}/logs/dependent_wait_audit.txt" 2>&1; then
	grep -E '^(REFUSED|dependent)' "${output_directory}/logs/dependent_wait_audit.txt" >&2 || tail -n 20 "${output_directory}/logs/dependent_wait_audit.txt" >&2
	echo "a kernel stores to global memory before it waits for the kernel it depends on; see logs/dependent_wait_audit.txt" >&2
	exit 9
fi

gate_phase "target and dependent-wait audits"
if ! make -C "${repository_root}" test-nvrm-launch \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/nvrm_launch_arguments.txt" 2>&1; then
	tail -n 40 "${output_directory}/logs/nvrm_launch_arguments.txt" >&2
	echo "the nvrm flavour's LM_LAUNCH does not convert arguments to their kernel parameter types; see logs/nvrm_launch_arguments.txt" >&2
	exit 8
fi

gate_phase "nvrm launch arguments"
hosted_release_root="${output_directory}/hosted_release_root"
if ! make -C "${repository_root}" -j"$(nproc)" tools hosted-release-gate \
	HOSTED_RELEASE_GATE_FLAGS=--complete \
	HOSTED_RELEASE_ROOT="${hosted_release_root}" \
	NVCC="${nvcc_binary}" \
	CUDA_ARCH=sm_121a \
	CUDA_HOME="${cuda_home}" \
	> "${output_directory}/logs/hosted_release_gate.txt" 2>&1; then
	grep -E '^(FAIL|hosted-release-gate)' "${output_directory}/logs/hosted_release_gate.txt" >&2 || tail -n 40 "${output_directory}/logs/hosted_release_gate.txt" >&2
	if [[ -d "${hosted_release_root}" ]]; then
		python3 "${repository_root}/tools/hosted_release_gate.py" findings "${hosted_release_root}" > "${output_directory}/hosted_release_findings.json"
	fi
	echo "the hosted release root does not build or does not match tools/hosted_release_debt.json; see logs/hosted_release_gate.txt and hosted_release_findings.json" >&2
	exit 6
fi
python3 "${repository_root}/tools/hosted_release_gate.py" findings "${hosted_release_root}" > "${output_directory}/hosted_release_findings.json"
grep -E '^hosted-release-gate' "${output_directory}/logs/hosted_release_gate.txt"

gate_phase "hosted release root"
spill_audited=()
for artifact in "${adapters[@]}" "${cuda_programs[@]}"; do
	if [[ -f "${repository_root}/${artifact}" ]]; then
		spill_audited+=("${repository_root}/${artifact}")
	fi
done
if ! python3 -I "${repository_root}/skl2/tools/kernel_spill_audit.py" "${spill_audited[@]}" \
	> "${output_directory}/logs/kernel_spill_audit.txt" 2>&1; then
	grep -E '^(REFUSED|kernel)' "${output_directory}/logs/kernel_spill_audit.txt" >&2 || tail -n 20 "${output_directory}/logs/kernel_spill_audit.txt" >&2
	echo "a skinny, cell, stream GEMM or attention kernel keeps a stack frame or local memory, or the attention PENDING list in skl2/tools/kernel_spill_audit.py does not match; see logs/kernel_spill_audit.txt" >&2
	exit 10
fi
grep -E '^kernel spill audit' "${output_directory}/logs/kernel_spill_audit.txt"

gate_phase "spill audit"
staged_roots="$(mktemp -d)"
trap 'rm -rf "${staged_roots}"' EXIT
while IFS=' ' read -r model adapter; do
	[[ -f "${repository_root}/build/${adapter}" ]] || continue
	STAGE_MODEL="${model}" STAGE_LANE=0 STAGE_WEIGHTD_SOCKET=/unused STAGE_MEMORY_MAX=1G STAGE_FIRMWARE="${repository_root}/build" \
		"${repository_root}/tools/stage_lane.sh" stage-local "${staged_roots}/${model}"
done < <(python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); import stage_lane; [print(name, entry["adapter"]) for name, entry in sorted(stage_lane.MODELS.items())]' "${repository_root}/tools")
K3_LANE=1 K3_FIRMWARE="${repository_root}/build" K3_WEIGHTD_SOCKET=/unused K3_MEMORY_MAX=1G K3_STATE_BUDGET_BYTES=1 \
	"${repository_root}/tools/k3_lane.sh" stage-local "${staged_roots}/k3_lane"
staged_failures=0
for root in "${staged_roots}"/*; do
	if ! python3 -I "${repository_root}/tools/staged_kernels_audit.py" "${root}" > "${output_directory}/logs/staged.$(basename "${root}").txt"; then
		cat "${output_directory}/logs/staged.$(basename "${root}").txt" >&2
		staged_failures=$((staged_failures + 1))
	fi
done
staged_root_count="$(ls "${staged_roots}" | wc -l)"
rm -rf "${staged_roots}"
if [[ "${staged_failures}" -ne 0 ]]; then
	echo "${staged_failures} staged lane roots ship a module with device code but without its kernels under its staged name" >&2
	exit 7
fi

gate_phase "staged roots"
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

gate_phase "object targets and receipts"
echo "PASS CUDA 13 exact sm_121a compile gate (${#adapters[@]} serving adapters; ${staged_root_count} staged lane roots carry every module's kernels)"
