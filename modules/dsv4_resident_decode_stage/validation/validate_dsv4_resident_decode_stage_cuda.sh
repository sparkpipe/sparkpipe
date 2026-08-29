#!/usr/bin/env bash
set -euo pipefail

# dsv4 resident decode stage, retained-receipt GPU validation. The
# mechanical skeleton is the shared validation driver; the batch-bucket
# gate, the stage-0 reference-fixture gate, and the pass-through validation
# defines below are dsv4's own.

validation_label="DSV4"
validation_digest_label="DSV4"
validation_gate_label="dsv4"
validation_env_prefix="SPARK_DSV4"
validation_validator_file="spark_dsv4_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="dsv4_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=mid

validation_include_dirs() {
    printf '%s\n' "model-families/dsv4/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' "-DSPARK_BATCH_BUCKET=${batch_bucket}"
    if [[ -n "${validation_defines:-}" ]]; then
        printf '%s\n' ${validation_defines:+"${validation_defines}"}
    fi
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

batch_bucket="${SPARK_MODULE_BATCH_BUCKET:-}"
validation_defines="${SPARK_DSV4_VALIDATION_DEFINES:-}"

spark_cuda_validation_begin "$@"
reference_fixture_directory="${repository_root}/qualification/dsv4/reference_vectors/ga_stage0_compsec076_p128"
reference_verifier="${repository_root}/tools/verify_dsv4_ga_reference_fixture.py"
spark_cuda_validation_check_hash_format
case "${batch_bucket}" in
    1|2|4|8|16|32|64|128|256|512|1024)
        ;;
    *)
        echo "SPARK_MODULE_BATCH_BUCKET must name the archive's built variant" >&2
        exit 2
        ;;
esac
spark_cuda_validation_check_archive
spark_cuda_validation_check_pack
spark_cuda_validation_check_source_digests
require_source_digest "${SPARK_DSV4_REFERENCE_VERIFIER_SHA256:-}" "${reference_verifier}" "DSV4 reference verifier"

unset SPARK_DSV4_REFERENCE_TOKEN_PATH
unset SPARK_DSV4_REFERENCE_OUTPUT_PATH
if [[ "${SPARK_DSV4_STAGE_INDEX:-}" == "0" &&
      "${SPARK_DSV4_STAGE_FIRST_LAYER:-}" == "0" &&
      "${SPARK_DSV4_STAGE_LAYER_COUNT:-}" == "3" ]]; then
    if [[ ! "${SPARK_DSV4_REFERENCE_MANIFEST_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "stage-0 validation requires its retained reference manifest SHA-256" >&2
        exit 2
    fi
    python3 "${reference_verifier}" \
        "${reference_fixture_directory}" \
        "${SPARK_DSV4_REFERENCE_MANIFEST_SHA256}"
    export SPARK_DSV4_REFERENCE_TOKEN_PATH="${reference_fixture_directory}/prompt_tokens.u32le"
    export SPARK_DSV4_REFERENCE_OUTPUT_PATH="${reference_fixture_directory}/after_layer_2.bf16le"
fi

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
