#!/usr/bin/env bash
set -euo pipefail

# qwen4_flash resident decode stage, retained-receipt GPU validation. The
# mechanical skeleton is the shared validation driver; the admission gates
# below are qwen4_flash's own tier policy.

validation_label="Qwen38_27b"
validation_digest_label="Qwen38_27b"
validation_gate_label="qwen4_flash"
validation_env_prefix="SPARK_QWEN4_FLASH"
validation_validator_file="spark_qwen4_flash_resident_decode_stage_cuda_validation.cu"
validation_oracle_file="spark_qwen4_flash_reference.c"
validation_output_name="qwen4_flash_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/qwen4_flash/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' "-DSPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES=${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-8}"
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_pack
spark_cuda_validation_check_source_digests

# The KV block table must span max_active_sequence_count lanes and the
# validator drives eight lanes of one block each. Degree 1 validates the
# stage-0 mid-pipeline tier (hidden output transport mandatory); a TP degree
# validates the whole-stack tier instead: rank 0 in standalone collective
# mode (the build host has no peer group), so consistency and determinism
# gate here while cross-rank numerics gate at the band E2E run.
require_configuration_value SPARK_QWEN4_FLASH_ALLOW_UNQUALIFIED_EXECUTION 1
require_configuration_value SPARK_QWEN4_FLASH_STAGE_INDEX 0
require_configuration_value SPARK_QWEN4_FLASH_STAGE_FIRST_LAYER 0
# The harness block table spans MAX_ACTIVE_SEQUENCES lanes (the -D plumbing
# from PR #714); admit the qualified lane-ladder values.
case "${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-8}" in
    8|16|64) ;;
    *)
        echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES in {8,16,64}, got '${SPARK_QWEN4_FLASH_STAGE_MAX_ACTIVE_SEQUENCES:-}'" >&2
        exit 2
        ;;
esac
# Whole-stack tier: TP-sharded (TP_DEGREE != 1) OR TP1 full-width
# standalone (TP_DEGREE == 1 with TP_STANDALONE == 1). The TP1 topology
# knob postdates the original >1 restriction.
if [[ "${SPARK_QWEN4_FLASH_TP_DEGREE:-1}" != "1" ]] || [[ "${SPARK_QWEN4_FLASH_TP_STANDALONE:-0}" == "1" ]]; then
    require_configuration_value SPARK_QWEN4_FLASH_TP_RANK 0
    require_configuration_value SPARK_QWEN4_FLASH_TP_STANDALONE 1
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_COUNT 1
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT 48
    require_configuration_value SPARK_QWEN4_FLASH_STAGE_MTP 1
else
    if (( ${SPARK_QWEN4_FLASH_STAGE_COUNT:-0} < 2 )); then
        echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_COUNT >= 2 (mid-pipeline stage 0)" >&2
        exit 2
    fi
    if (( ${SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT:-0} < 4 || ${SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT:-48} >= 48 )); then
        echo "qwen4_flash hardware validation requires 4 <= SPARK_QWEN4_FLASH_STAGE_LAYER_COUNT < 48 (GDN and full attention, no final head)" >&2
        exit 2
    fi
fi
if (( ${SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS:-0} < 8 )); then
    echo "qwen4_flash hardware validation requires SPARK_QWEN4_FLASH_STAGE_KV_BLOCKS >= 8" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
