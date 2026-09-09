#!/usr/bin/env bash
set -euo pipefail

# Compiles the validator translation unit against the MODULE ARCHIVE and
# runs it against the pinned configuration. The binary runs the host
# oracle selftest first (bounded decay, bf16 round trip, rope identity,
# group-limited router, codec slab addressing, and end-to-end KDA/MLA
# oracle executions at real geometry), then the GPU tiers on synthesized
# weights: tier 1 the KDA layer + dense MLP (layers 0,1), tier 2a the
# rope-64 absorbed-MLA layer + routed experts through the group router
# (layers 5,6), tier 3 a multi-position prefill run through the KDA
# recurrence followed by a cached decode step, each against the same fp32
# oracle with a bit-exact determinism re-walk and a clean KV access-error
# lane. The binary FAILS (nonzero) until every tier passes; a failing
# tier is a hard failure, never a silent pass.
# The mechanical skeleton is the shared validation driver; the codec
# ladder and the build-identity defaults below are ling's own.

validation_label="ling"
validation_digest_label=""
validation_gate_label="ling"
validation_env_prefix="SPARK_LING"
validation_validator_file="spark_ling_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="ling_resident_decode_stage_validator"
validation_hash_format_check=0
validation_nvcc_splice=late

validation_include_dirs() {
    printf '%s\n' "model-families/ling/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' \
        "-DLING_EXPERT_WEIGHT_CODEC=${ling_codec_ids[${codec_index}]}" \
        "-DLING_EXPERT_CODEC_NAME=\"${SPARK_LING_EXPERT_CODEC}\"" \
        "-DLING_MODEL_REVISION=\"${model_revision}\"" \
        "-DLING_CONTRACT_SHA256=\"${contract_sha256}\""
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

ling_codecs=(bf16 int6 int7 int8 fp8 nvfp4 mxfp4)
ling_codec_ids=(1 2 3 4 5 6 7)

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_archive

codec_index=-1
for index in "${!ling_codecs[@]}"; do
    if [[ "${SPARK_LING_EXPERT_CODEC:-}" == "${ling_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "ling hardware validation requires SPARK_LING_EXPERT_CODEC to name one of: ${ling_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_LING_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_LING_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "ling hardware validation requires SPARK_LING_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

model_revision="${SPARK_LING_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_LING_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

spark_cuda_validation_build_and_run
