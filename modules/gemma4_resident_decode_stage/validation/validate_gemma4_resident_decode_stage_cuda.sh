#!/usr/bin/env bash
set -euo pipefail

# gemma4 resident decode stage, retained-receipt GPU validation. The
# mechanical skeleton is the shared validation driver; the tier policy below
# is gemma4's own: the kernel-dataflow tier validates the carved kernel set
# (gqa triple + norm/rope + family-local gelu/scaled-gather) against a
# host-side mirror of the anchor oracle math, per arm, with a bit-exact
# determinism rerun. No stage pack is consumed at this tier.

validation_label="Gemma4"
validation_digest_label="Gemma4"
validation_gate_label="gemma4"
validation_env_prefix="SPARK_GEMMA4"
validation_validator_file="spark_gemma4_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="gemma4_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/gemma4/include"
    printf '%s\n' "model-families/common/include"
    printf '%s\n' "."
}

validation_nvcc_extra_args() {
    if [[ "${SPARK_GEMMA4_VALIDATION_MOE_BUILD:-0}" == "1" ]]; then
        printf '%s\n' "-DSPARK_GEMMA4_MOE_BUILD=1"
        printf '%s\n' "-DSPARK_GEMMA4_MODEL_MOE_BLOCK=1"
    fi
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_source_digests

require_configuration_value SPARK_GEMMA4_CUDA_VALIDATOR_SHA256 "$(sha256sum "${script_directory}/${validation_validator_file}" | awk '{print $1}')"

case "${SPARK_GEMMA4_VALIDATION_MOE_BUILD:-0}" in
    0|1) ;;
    *)
        echo "gemma4 hardware validation requires SPARK_GEMMA4_VALIDATION_MOE_BUILD in {0,1}, got '${SPARK_GEMMA4_VALIDATION_MOE_BUILD:-}'" >&2
        exit 2
        ;;
esac

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
