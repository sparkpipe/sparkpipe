#!/usr/bin/env bash
set -euo pipefail

# glm5_next MTP chain speculation parity gate (sm_121a).
#
# Compiles the MTP parity harness against the MODULE ARCHIVE and runs it on
# this node. The harness builds a small synthetic stack in-device (three KDA
# layers, one DSA+MoE layer, the layer-45 MTP draft weights, embedding and
# lm_head at full model geometry), decodes a fixed prompt serially
# (commit=1), then re-runs the same prompt through draft -> verify
# (commit=0, replay recording) -> resolve -> fold, cycling forced
# full-reject / mid-chain / full-accept drafts against organic ones. It
# FAILS (nonzero) unless the emitted token stream is identical to baseline
# AND the KDA state, conv windows, KV cache and index cache bytes match the
# serial decode at every burst boundary. No stage pack is needed: the
# weights are synthesized in-fixture (parity is spec-vs-baseline, not
# vs-checkpoint). The expert payload filler is fp8-exact, so the archive
# must be the EXPERT_CODEC=fp8 build.

validation_label="glm5_next"
validation_digest_label=""
validation_gate_label="glm5_next"
validation_env_prefix="SPARK_GLM5_NEXT"
validation_validator_file="spark_glm5_next_resident_decode_stage_mtp_parity.cu"
validation_oracle_file=""
validation_output_name="glm5_next_resident_decode_stage_mtp_parity"
validation_hash_format_check=0
validation_nvcc_splice=late

validation_include_dirs() {
    printf '%s\n' "model-families/glm5_next/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' \
        "-DGLM5_NEXT_EXPERT_WEIGHT_CODEC=5" \
        "-DGLM5_NEXT_EXPERT_CODEC_NAME=\"fp8\"" \
        "-DGLM5_NEXT_MODEL_REVISION=\"${model_revision}\"" \
        "-DGLM5_NEXT_CONTRACT_SHA256=\"${contract_sha256}\""
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_archive

if [[ "${SPARK_GLM5_NEXT_EXPERT_CODEC:-}" != "fp8" ]]; then
    echo "glm5_next MTP parity validation requires SPARK_GLM5_NEXT_EXPERT_CODEC=fp8 (the harness synthesizes fp8 expert payloads)" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

model_revision="${SPARK_GLM5_NEXT_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_GLM5_NEXT_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

spark_cuda_validation_build_and_run
