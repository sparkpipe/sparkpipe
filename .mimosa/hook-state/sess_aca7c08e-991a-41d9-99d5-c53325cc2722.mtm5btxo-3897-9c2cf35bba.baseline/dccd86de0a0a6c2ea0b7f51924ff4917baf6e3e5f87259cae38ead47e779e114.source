# Shared driver for the per-family resident-decode-stage CUDA validation
# scripts (DRY wave 1).
#
# The six validate_<family>_resident_decode_stage_cuda.sh scripts were one
# pasted skeleton: usage/argument check, directory derivation, the retained
# validation directory, the source-digest pin, the archive/pack checks, the
# sm_121a/nvcc toolchain gate, the core-library make, the nvcc link, and the
# validator run. This file is that skeleton. The family script keeps its own
# admission gates and their ORDER — those differ per family on purpose (tier
# ladders, codec ladders, reference fixtures) and stay visible family-side.
#
# Family configuration (set before sourcing):
#   validation_label           toolchain message label, exactly as pasted
#   validation_digest_label    source-digest message label (empty = no digests)
#   validation_gate_label      lowercase label in require_configuration_value
#   validation_env_prefix      e.g. SPARK_QWEN38_27B (pack path + digest envs)
#   validation_validator_file  validator .cu under the validation directory
#   validation_oracle_file     CPU oracle .c, or empty when the family has none
#   validation_include_dirs()  emits the family include dirs (repo-relative)
#   validation_output_name     validator binary name
#   validation_hash_format_check  1 unless the family skips the digest check
#   validation_nvcc_splice     where the family's extra nvcc args sit: std
#                              (after -std=c++17), mid (after -gencode), or
#                              late (after the -I list)
#   validation_nvcc_extra_args()  emits the family's extra nvcc arguments
#
# Stages — call in the family's original order:
#   spark_cuda_validation_begin "$@"
#   spark_cuda_validation_check_hash_format
#   spark_cuda_validation_check_archive
#   spark_cuda_validation_check_pack
#   spark_cuda_validation_check_source_digests
#   spark_cuda_validation_check_toolchain
#   spark_cuda_validation_build_and_run
#
# Behavior law: this consolidation must not move a byte of any family's
# validator invocation. The proof is a shim replay (stub nvcc/make logging
# their exact argv) diffed against each original script — see
# docs/AGENT_LANE_BRIEFS/reports/dry-wave1-*.md.

spark_cuda_validation_begin() {
	if [[ $# -ne 2 ]]; then
	    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
	    exit 2
	fi

	configuration_hash="$1"
	module_archive="$2"
	script_directory="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
	module_directory="$(cd "${script_directory}/.." && pwd)"
	repository_root="$(cd "${module_directory}/../.." && pwd)"
	validation_directory="$(mktemp -d)"
	cuda_validator="${script_directory}/${validation_validator_file}"
	if [[ -n "${validation_oracle_file}" ]]; then
		cpu_oracle="${script_directory}/${validation_oracle_file}"
	fi
	trap 'rm -rf "${validation_directory}"' EXIT

	require_source_digest() {
	    local expected="$1"
	    local path="$2"
	    local label="$3"
	    local actual remainder
	    if [[ ! "${expected}" =~ ^[0-9a-f]{64}$ ]]; then
	        echo "${label} expected SHA-256 is invalid" >&2
	        exit 2
	    fi
	    read -r actual remainder < <(sha256sum "${path}")
	    if [[ "${actual}" != "${expected}" ]]; then
	        echo "${label} SHA-256 mismatch" >&2
	        exit 2
	    fi
	}

	require_configuration_value() {
	    local name="$1"
	    local expected="$2"
	    local actual="${!name:-}"
	    if [[ "${actual}" != "${expected}" ]]; then
	        echo "${validation_gate_label} hardware validation requires ${name}=${expected}, got '${actual}'" >&2
	        exit 2
	    fi
	}
}

spark_cuda_validation_check_hash_format() {
	if [[ "${validation_hash_format_check:-0}" != "1" ]]; then
		return 0
	fi
	if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
	    echo "validation configuration must be a lowercase SHA-256 digest" >&2
	    exit 2
	fi
}

spark_cuda_validation_check_archive() {
	if [[ ! -s "${module_archive}" ]]; then
	    echo "module archive is missing or empty: ${module_archive}" >&2
	    exit 2
	fi
}

spark_cuda_validation_check_pack() {
	local pack_path="${validation_env_prefix}_STAGE_PACK_PATH"
	if [[ -z "${!pack_path:-}" || ! -s "${!pack_path}" ]]; then
	    echo "${validation_env_prefix}_STAGE_PACK_PATH must name a readable non-empty stage pack" >&2
	    exit 2
	fi
}

spark_cuda_validation_check_source_digests() {
	local validator_env="${validation_env_prefix}_CUDA_VALIDATOR_SHA256"
	require_source_digest "${!validator_env:-}" "${cuda_validator}" "${validation_digest_label} CUDA validator"
	if [[ -n "${validation_oracle_file}" ]]; then
		local oracle_env="${validation_env_prefix}_CPU_ORACLE_SHA256"
		require_source_digest "${!oracle_env:-}" "${cpu_oracle}" "${validation_digest_label} CPU oracle"
	fi
}

spark_cuda_validation_check_toolchain() {
	nvcc_path="${NVCC:-nvcc}"
	cuda_architecture="${CUDA_ARCH:-sm_121a}"
	if [[ "${cuda_architecture}" != "sm_121a" ]]; then
	    echo "${validation_label} hardware validation admits only CUDA_ARCH=sm_121a" >&2
	    exit 2
	fi
	if ! command -v "${nvcc_path}" >/dev/null 2>&1; then
	    echo "nvcc unavailable for ${validation_label} hardware validation" >&2
	    exit 2
	fi
}

spark_cuda_validation_build_and_run() {
	make -C "${repository_root}" \
	    build/libsparkpipe_core.a \
	    build/libsparkpipe_runtime.a

	local nvcc_extra_args=()
	local emitted
	while IFS= read -r emitted; do
		nvcc_extra_args+=("${emitted}")
	done < <(validation_nvcc_extra_args)

	local nvcc_args=("-std=c++17")
	case "${validation_nvcc_splice}" in
	std)
		nvcc_args+=("${nvcc_extra_args[@]}"
			"-O3"
			"--expt-relaxed-constexpr"
			"-gencode" "arch=compute_121a,code=sm_121a")
		;;
	mid)
		nvcc_args+=("-O3"
			"--expt-relaxed-constexpr"
			"-gencode" "arch=compute_121a,code=sm_121a"
			"${nvcc_extra_args[@]}")
		;;
	late)
		nvcc_args+=("-O3"
			"--expt-relaxed-constexpr"
			"-gencode" "arch=compute_121a,code=sm_121a")
		;;
	esac
	nvcc_args+=("-I${repository_root}/include")
	local include_dir
	while IFS= read -r include_dir; do
		nvcc_args+=("-I${repository_root}/${include_dir}")
	done < <(validation_include_dirs)
	nvcc_args+=("-I${module_directory}/include"
		"-I${module_directory}/source")
	if [[ "${validation_nvcc_splice}" == "late" ]]; then
		nvcc_args+=("${nvcc_extra_args[@]}")
	fi

	"${nvcc_path}" \
	    "${nvcc_args[@]}" \
	    "${cuda_validator}" \
	    "${module_archive}" \
	    "${repository_root}/build/libsparkpipe_runtime.a" \
	    "${repository_root}/build/libsparkpipe_core.a" \
	    -L"${CUDA_HOME:-/usr/local/cuda}/lib64" \
	    -lcuda \
	    -lcudart \
	    -ldl \
	    -lm \
	    -Xcompiler -pthread \
	    -o "${validation_directory}/${validation_output_name}"

	"${validation_directory}/${validation_output_name}" "${configuration_hash}"
}
