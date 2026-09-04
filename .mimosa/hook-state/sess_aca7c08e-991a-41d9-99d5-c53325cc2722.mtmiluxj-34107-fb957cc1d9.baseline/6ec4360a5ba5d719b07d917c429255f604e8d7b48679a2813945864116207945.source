#!/usr/bin/env bash
set -euo pipefail

# qwen38_max resident decode stage, retained-receipt GPU validation. The
# mechanical skeleton is the shared validation driver; the admission gates
# below are qwen38_max's own tier policy.

validation_label="Qwen38_max"
validation_digest_label="Qwen38_max"
validation_gate_label="qwen38_max"
validation_env_prefix="SPARK_QWEN38_MAX"
validation_validator_file="spark_qwen38_max_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="qwen38_max_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/common/include" "model-families/qwen38_max/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' "-DSPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES=${SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES:-8}"
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_pack
spark_cuda_validation_check_source_digests

# The module tier drives decode frames through the module's own unqualified
# smoke path (the serving adapter owns the qualified one), so the gate must
# be open and the stage must be the slice-0 stage of its configuration.
require_configuration_value SPARK_QWEN38_MAX_ALLOW_UNQUALIFIED_EXECUTION 1
require_configuration_value SPARK_QWEN38_MAX_STAGE_INDEX 0
require_configuration_value SPARK_QWEN38_MAX_STAGE_FIRST_LAYER 0
require_configuration_value SPARK_QWEN38_MAX_STAGE_MTP 0
case "${SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES:-8}" in
    8|16|64) ;;
    *)
        echo "qwen38_max hardware validation requires SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES in {8,16,64}, got '${SPARK_QWEN38_MAX_STAGE_MAX_ACTIVE_SEQUENCES:-}'" >&2
        exit 2
        ;;
esac
# Two admitted module tiers: the whole stack (STAGE_COUNT 1, all 92 layers,
# embedding through head, TP1) or a mid-pipeline stage 0 (STAGE_COUNT >= 2,
# 4..88 layers, embedding in, hidden out). A TP degree beyond 1 needs the
# device collective (four live ranks) and gates at the fleet window, not
# here - the kernel tier covers the rank-local geometry.
if [[ "${SPARK_QWEN38_MAX_TP_DEGREE:-1}" != "1" ]]; then
    echo "qwen38_max hardware validation requires SPARK_QWEN38_MAX_TP_DEGREE=1 (tp>1 needs the live collective; see the lane report)" >&2
    exit 2
fi
if [[ "${SPARK_QWEN38_MAX_STAGE_COUNT:-0}" == "1" ]]; then
    require_configuration_value SPARK_QWEN38_MAX_STAGE_LAYER_COUNT 92
else
    if (( ${SPARK_QWEN38_MAX_STAGE_COUNT:-0} < 2 )); then
        echo "qwen38_max hardware validation requires SPARK_QWEN38_MAX_STAGE_COUNT >= 2 (mid-pipeline stage 0) or 1 (whole stack)" >&2
        exit 2
    fi
    if (( ${SPARK_QWEN38_MAX_STAGE_LAYER_COUNT:-0} < 4 || ${SPARK_QWEN38_MAX_STAGE_LAYER_COUNT:-92} >= 92 )); then
        echo "qwen38_max hardware validation requires 4 <= SPARK_QWEN38_MAX_STAGE_LAYER_COUNT < 92 (GDN and full attention, no final head)" >&2
        exit 2
    fi
fi
if (( ${SPARK_QWEN38_MAX_STAGE_KV_BLOCKS:-0} < 8 )); then
    echo "qwen38_max hardware validation requires SPARK_QWEN38_MAX_STAGE_KV_BLOCKS >= 8" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
