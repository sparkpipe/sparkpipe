#!/usr/bin/env bash
set -euo pipefail

validation_label="MinimaxH3"
validation_digest_label="MinimaxH3"
validation_gate_label="minimax_h3"
validation_env_prefix="MINIMAX_H3"
validation_validator_file="spark_minimax_h3_resident_media_stage_cuda_validation.cu"
validation_oracle_file="spark_minimax_h3_reference.c"
validation_output_name="minimax_h3_resident_media_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/minimax_h3/include" "model-families/common/include" "."
}

validation_nvcc_extra_args() {
    printf '%s\n' "-DSPARK_MINIMAX_H3_MODULE_BUILD=1"
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_source_digests
spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
