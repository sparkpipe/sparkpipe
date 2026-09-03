#!/usr/bin/env bash
set -euo pipefail

# GLM 5.2 resident decode stage, retained-receipt GPU validation.
#
# Called by resident_decode_stage_rules.mk as:
#   GPU_VALIDATOR VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE
# with the RUNTIME_CONFIGURATION environment (SPARK_GLM52_*) exported, so the
# configuration hash names the exact configuration this run validated and the
# publish recipe retains it. The validator source digest is pinned through
# SPARK_GLM52_CUDA_VALIDATOR_SHA256 in the same configuration, so a published
# artifact carries which validator text approved it.
# The mechanical skeleton is the shared validation driver; the codec ladder
# and the build-identity pin below are glm52's own.

validation_label="glm52"
validation_digest_label="Glm52"
validation_gate_label="glm52"
validation_env_prefix="SPARK_GLM52"
validation_validator_file="spark_glm52_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="glm52_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=late

validation_include_dirs() {
    printf '%s\n' "model-families/glm52/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' \
        "-DGLM52_EXPERT_WEIGHT_CODEC=${glm52_codec_ids[${codec_index}]}" \
        "-DGLM52_EXPERT_CODEC_NAME=\"${SPARK_GLM52_EXPERT_CODEC}\"" \
        "-DGLM52_MODEL_REVISION=\"${model_revision}\"" \
        "-DGLM52_CONTRACT_SHA256=\"${contract_sha256}\""
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

glm52_codecs=(bf16 int6 int7 int8 fp8 nvfp4 mxfp4)
glm52_codec_ids=(1 2 3 4 5 6 7)

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_pack
spark_cuda_validation_check_source_digests

# The validator walks three tiers on synthetic weights; it does not load the
# pack: tier 1 is one dense layer's forward plus a bit-exact determinism
# re-walk, tier 2a is the first routed-expert layer (router/top-k selection,
# package-codec expert forwards, shared expert) and tier 2b is the DSA
# indexer at context beyond the selection width. The pack still gates here
# (non-empty, named by the pinned configuration) because the receipt binds
# the configuration the archive publishes under; the loader's own schema
# chain pins its content.
codec_index=-1
for index in "${!glm52_codecs[@]}"; do
    if [[ "${SPARK_GLM52_EXPERT_CODEC:-}" == "${glm52_codecs[${index}]}" ]]; then
        codec_index="${index}"
    fi
done
if (( codec_index < 0 )); then
    echo "glm52 hardware validation requires SPARK_GLM52_EXPERT_CODEC to name one of: ${glm52_codecs[*]}" >&2
    exit 2
fi
if [[ -z "${SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES:-}" ]] || (( SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES < 1 )); then
    echo "glm52 hardware validation requires SPARK_GLM52_STAGE_MAX_ACTIVE_SEQUENCES >= 1" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

# Single-source the build identity exactly as the compile gate does, so the
# validator translation unit sees the same package defines the archive did.
read -r model_revision contract_sha256 < <(
    python3 "${repository_root}/tools/glm52_model_contract.py" \
        --print-build-identity "${SPARK_GLM52_EXPERT_CODEC}"
)

spark_cuda_validation_build_and_run
