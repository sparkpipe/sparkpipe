#!/usr/bin/env bash
set -euo pipefail

# glm5_next (GLM 5.3 Flash) hardware validation driver (sm_121a).
#
# Compiles the validator translation unit against the MODULE ARCHIVE and
# runs it on the pack named by the pinned configuration. The binary runs
# the host oracle selftest first (bounded decay, expert-major codec
# addressing, e4m3, mHC sinkhorn, kpool expansion, and end-to-end
# KDA/MLA/router oracle executions at real geometry), then the GPU tier
# drivers: tier 1 the KDA layer with its dense MLP through both mHC sites,
# tier 2a the DSA (rope-0 MLA) layer with routed experts, each against the
# same fp32 oracle with a bit-exact determinism re-walk. The binary FAILS
# (nonzero) until every wired tier passes; a not-yet-wired tier is a hard
# failure, never a silent pass.
# The mechanical skeleton is the shared validation driver; the codec ladder
# and the build-identity defaults below are glm5_next's own.

validation_label="glm5_next"
validation_digest_label=""
validation_gate_label="glm5_next"
validation_env_prefix="SPARK_GLM5_NEXT"
validation_validator_file="spark_glm5_next_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="glm5_next_resident_decode_stage_validator"
validation_hash_format_check=0
validation_nvcc_splice=late

validation_include_dirs() {
    printf '%s\n' "model-families/glm5_next/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' \
        "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=${glm5_next_codec_ids[${codec_index}]}" \
        "-DGLM5_NEXT_EXPERT_CODEC_NAME=\"${SPARK_GLM5_NEXT_EXPERT_CODEC}\"" \
        "-DGLM5_NEXT_MODEL_REVISION=\"${model_revision}\"" \
        "-DGLM5_NEXT_CONTRACT_SHA256=\"${contract_sha256}\""
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

glm5_next_codecs=(int6 int7 int8 fp8 nvfp4 mxfp4)
glm5_next_codec_ids=(2 3 4 5 6 7)

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_archive
spark_cuda_validation_check_pack

codec_index=-1
for index in "${!glm5_next_codecs[@]}"; do
    if [[ "${SPARK_GLM5_NEXT_EXPERT_CODEC:-}" == "${glm5_next_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "glm5_next hardware validation requires SPARK_GLM5_NEXT_EXPERT_CODEC to name one of: ${glm5_next_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "glm5_next hardware validation requires SPARK_GLM5_NEXT_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

model_revision="${SPARK_GLM5_NEXT_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_GLM5_NEXT_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

spark_cuda_validation_build_and_run
