#!/usr/bin/env bash
set -euo pipefail

# minimax text-tower resident decode stage, component-tier GPU validation
# driver (sm_121a).
#
# Compiles the validator translation unit against the MODULE ARCHIVE and
# runs its component fixture with the supplied configuration identity:
# embedding gather, RMSNorm (determinism rerun), residual add, per-head
# norm + rope at the TP4 local geometry, SwiGLU, vocabulary argmax through
# the sortable-u64 packing with the u64 max combine, and the two TP combine
# kernels - each against a host mirror of the same math.
#
# It does NOT consume a stage pack and does not run the linear projections,
# attention dataflow, KV write path, multi-row decode, TP collectives over
# real transports, or the t1 fixture streams. A component PASS is not full
# minimax numerical or driver acceptance; the fixture-level ladder runs
# through validation/minimax_cpu_validate.c (t1 reference streams) and the
# lane receipts.
#
# The mechanical skeleton is the shared validation driver; scope and
# tolerances below are minimax's own (laguna component-tier precedent).

validation_label="minimax"
validation_digest_label="minimax"
validation_gate_label="minimax"
validation_env_prefix="SPARK_MINIMAX"
validation_validator_file="spark_minimax_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="minimax_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/minimax/include"
    printf '%s\n' "model-families/minimax/include/sparkpipe"
    printf '%s\n' "model-families/common/include"
    printf '%s\n' "."
}

validation_nvcc_extra_args() {
    :
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_source_digests

require_configuration_value SPARK_MINIMAX_CUDA_VALIDATOR_SHA256 "$(sha256sum "${script_directory}/${validation_validator_file}" | awk '{print $1}')"

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
