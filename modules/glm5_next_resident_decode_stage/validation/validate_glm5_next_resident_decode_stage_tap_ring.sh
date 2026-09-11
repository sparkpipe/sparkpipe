#!/usr/bin/env bash
set -euo pipefail

# glm5_next hidden-tap ring gate (sm_121a).
#
# Compiles the tap ring harness against the MODULE ARCHIVE and runs it on
# this node. The harness builds a 43-layer synthetic stack in-device (KDA
# and DSA+MoE layers at real model geometry, first_layer_index=0 so global
# layers 5/14/24/33/42 are the tap sites), walks a fixed prompt through
# prefill plus serial decode, and at every tap layer's post-MlpPost point
# runs the same capture the module chain runs: HC-mean of the residual into
# a device stage, then the tap ring's async D2H on the ring's dedicated
# stream. It FAILS (nonzero) unless every in-window ring entry is byte-exact
# against the wave residual captured directly from the device stage, the
# last position's tap is byte-exact against an independent host-computed HC
# mean of the post-MlpPost hidden, wrapped positions evict the stale entry,
# and reads of uncaptured, uncommitted (beyond-anchor) or other-lane
# positions are rejected. It prints the measured per-capture mean-kernel and
# ring D2H cost. No stage pack is needed: weights are synthesized
# in-fixture. The expert payload filler is fp8-exact, so the archive must
# be the EXPERT_CODEC=fp8 build.

validation_label="glm5_next"
validation_digest_label=""
validation_gate_label="glm5_next"
validation_env_prefix="SPARK_GLM5_NEXT"
validation_validator_file="spark_glm5_next_resident_decode_stage_tap_ring.cu"
validation_oracle_file=""
validation_output_name="glm5_next_resident_decode_stage_tap_ring"
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
    echo "glm5_next tap ring validation requires SPARK_GLM5_NEXT_EXPERT_CODEC=fp8 (the harness synthesizes fp8 expert payloads)" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain

model_revision="${SPARK_GLM5_NEXT_MODEL_REVISION:-synthesized}"
contract_sha256="${SPARK_GLM5_NEXT_CONTRACT_SHA256:-0000000000000000000000000000000000000000000000000000000000000000}"

spark_cuda_validation_build_and_run
