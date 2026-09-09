#!/usr/bin/env bash
set -euo pipefail

# laguna (GLM 5.3 Flash) hardware validation driver (sm_121a).
#
# Compiles the validator translation unit against the MODULE ARCHIVE and
# runs its synthetic fixture with the supplied configuration identity. The binary runs
# the host oracle selftest first (bounded decay, expert-major codec
# addressing, e4m3, mHC sinkhorn, kpool expansion, and end-to-end
# KDA/MLA/router oracle executions at real geometry), then a GPU numerical
# check of KDA+dense MLP+mHC and repeatability checks of KDA and DSA attention.
# It currently executes TP1/B1. It does not compare GPU DSA against its host
# oracle or run a routed MLP, multiple rows, distributed ranks or real weights.
# A component PASS is not full GLM numerical or driver acceptance.
# The mechanical skeleton is the shared validation driver; the codec ladder
# and the build-identity defaults below are laguna's own.

validation_label="laguna"
validation_digest_label=""
validation_gate_label="laguna"
validation_env_prefix="SPARK_LAGUNA"
validation_validator_file="spark_laguna_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="laguna_resident_decode_stage_validator"
validation_hash_format_check=0
validation_nvcc_splice=late

validation_include_dirs() {
    printf '%s\n' "model-families/laguna/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' \
        "-DLAGUNA_EXPERT_WEIGHT_CODEC=${laguna_codec_ids[${codec_index}]}" \
        "-DLAGUNA_EXPERT_CODEC_NAME=\"${SPARK_LAGUNA_EXPERT_CODEC}\"" \
        "-DLAGUNA_MODEL_REVISION=\"${model_revision}\"" \
        "-DLAGUNA_CONTRACT_SHA256=\"${contract_sha256}\""
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

laguna_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)
laguna_codec_ids=(2 3 4 5 6 7)

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_archive

codec_index=-1
for index in "${!laguna_codecs[@]}"; do
    if [[ "${SPARK_LAGUNA_EXPERT_CODEC:-}" == "${laguna_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "laguna hardware validation requires SPARK_LAGUNA_EXPERT_CODEC to name one of: ${laguna_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_LAGUNA_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_LAGUNA_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "laguna hardware validation requires SPARK_LAGUNA_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

model_revision="${SPARK_LAGUNA_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_LAGUNA_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

spark_cuda_validation_build_and_run
