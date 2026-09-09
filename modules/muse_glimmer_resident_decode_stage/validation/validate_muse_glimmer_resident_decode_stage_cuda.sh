#!/usr/bin/env bash
set -euo pipefail

# muse_glimmer resident decode stage, retained-receipt GPU validation. The
# mechanical skeleton is the shared validation driver; the admission gates
# below are muse_glimmer's own tier policy.

validation_label="MuseGlimmer"
validation_digest_label="MuseGlimmer"
validation_gate_label="muse_glimmer"
validation_env_prefix="SPARK_MUSE_GLIMMER"
validation_validator_file="spark_muse_glimmer_resident_decode_stage_cuda_validation.cu"
validation_oracle_file=""
validation_output_name="muse_glimmer_resident_decode_stage_validator"
validation_hash_format_check=1
validation_nvcc_splice=std

validation_include_dirs() {
    printf '%s\n' "model-families/common/include" "model-families/muse_glimmer/include"
}

validation_nvcc_extra_args() {
    printf '%s\n' "-DSPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES=${SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES:-8}"
}

script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_directory}/../../spark_resident_decode_stage_cuda_validation_common.sh"

spark_cuda_validation_begin "$@"
spark_cuda_validation_check_hash_format
spark_cuda_validation_check_archive
spark_cuda_validation_check_source_digests

# V0 synthesize-pack policy: the module tier loads the pack this script
# builds from the same enum/shape table the real packer implements
# (muse_glimmer_pack_synthesize), so a pre-download V0 needs no warm copy.
# An externally supplied pack (the future real rank-0 shard) wins.
if [[ -z "${SPARK_MUSE_GLIMMER_STAGE_PACK_PATH:-}" ]]; then
    SPARK_MUSE_GLIMMER_STAGE_PACK_PATH="${validation_directory}/muse_v0.gsmu"
    export SPARK_MUSE_GLIMMER_STAGE_PACK_PATH
    cc -std=c11 -O2 -Wall -Wextra -Werror \
        -I"${repository_root}/include" \
        -I"${repository_root}/src" \
        -I"${repository_root}/model-families/common/include" \
        -I"${repository_root}/model-families/muse_glimmer/include" \
        -I"${repository_root}/modules/muse_glimmer_resident_decode_stage/include" \
        -I"${repository_root}/modules/muse_glimmer_resident_decode_stage/source" \
        -I"${repository_root}" \
        "${module_directory}/tools/muse_glimmer_pack_synthesize.c" \
        "${repository_root}/runtime/stagepack_format.c" \
        "${repository_root}/src/spark_status.c" \
        -o "${validation_directory}/muse_glimmer_pack_synthesize"
    "${validation_directory}/muse_glimmer_pack_synthesize" \
        --output "${SPARK_MUSE_GLIMMER_STAGE_PACK_PATH}" --tp 16
fi

# The module tier drives decode frames through the module's own unqualified
# smoke path (the serving adapter owns the qualified one), so the gate must
# be open and the stage must be the slice-0 stage of its configuration.
require_configuration_value SPARK_MUSE_GLIMMER_ALLOW_UNQUALIFIED_EXECUTION 1
require_configuration_value SPARK_MUSE_GLIMMER_STAGE_INDEX 0
require_configuration_value SPARK_MUSE_GLIMMER_STAGE_FIRST_LAYER 0
case "${SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES:-8}" in
    8|16|64) ;;
    *)
        echo "muse_glimmer hardware validation requires SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES in {8,16,64}, got '${SPARK_MUSE_GLIMMER_STAGE_MAX_ACTIVE_SEQUENCES:-}'" >&2
        exit 2
        ;;
esac
# A TP degree beyond 1 needs the live device collective and gates at the
# fleet window, not here - the kernel tier covers the rank-local geometry.
if [[ "${SPARK_MUSE_GLIMMER_TP_DEGREE:-1}" != "1" ]]; then
    echo "muse_glimmer hardware validation requires SPARK_MUSE_GLIMMER_TP_DEGREE=1 (tp>1 needs the live collective; see the lane report)" >&2
    exit 2
fi
# The whole stack in one stage: 52 layers, embedding through head, TP1.
require_configuration_value SPARK_MUSE_GLIMMER_STAGE_COUNT 1
require_configuration_value SPARK_MUSE_GLIMMER_STAGE_LAYER_COUNT 52
if (( ${SPARK_MUSE_GLIMMER_STAGE_KV_BLOCKS:-0} < 8 )); then
    echo "muse_glimmer hardware validation requires SPARK_MUSE_GLIMMER_STAGE_KV_BLOCKS >= 8" >&2
    exit 2
fi

spark_cuda_validation_check_toolchain
spark_cuda_validation_build_and_run
