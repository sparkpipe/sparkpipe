#!/usr/bin/env bash
# Module-publish validator wrapper (reconstructed for the spark1 build host).
# Called by sparkpipe_module_publish as: wrapper ROOT SOURCE_SHA ARCHIVE_SHA.
# Validates the just-published module archive against the GA val4 pack using
# the same single-spark GPU harness as validate_ga_pro.sh.
set -euo pipefail
ROOT="${1:?usage: wrapper ROOT SOURCE_SHA ARCHIVE_SHA}"
SOURCE_SHA="${2:?}"
ARCHIVE_SHA="${3:?}"

export NVCC="${NVCC:-/usr/local/cuda-13.0/bin/nvcc}"
export CUDA_ARCH="${CUDA_ARCH:-sm_121a}"
export SPARK_DSV4_VALIDATION_DEFINES="-DSPARK_DSV4_PRO_BUILD=1"
export SPARK_MODULE_BATCH_BUCKET="${SPARK_DSV4_PUBLISH_VALIDATION_BUCKET:-1024}"
export SPARK_DSV4_STAGE_PACK_PATH="${SPARK_DSV4_PUBLISH_VALIDATION_PACK:-/home/spark1/sparkdata/dsv4_pro_ga.val0p4.spstage}"
export SPARK_DSV4_STAGE_COUNT=16
export SPARK_DSV4_STAGE_INDEX=0
export SPARK_DSV4_STAGE_FIRST_LAYER=0
export SPARK_DSV4_STAGE_LAYER_COUNT=4
export SPARK_DSV4_STAGE_MAX_ACTIVE_SEQUENCES=8
export SPARK_DSV4_STAGE_PIPELINE_SLOTS=1
export SPARK_DSV4_STAGE_MAX_SEQ=4096
export SPARK_DSV4_STAGE_LOGICAL_PAGES=1024
export SPARK_DSV4_STAGE_PHYSICAL_PAGES=1024
export SPARK_DSV4_STAGE_MTP=0
export SPARK_DSV4_STAGE_GRAPHS=0
export SPARK_DSV4_CUDA_VALIDATOR_SHA256="$(sha256sum "${ROOT}/modules/dsv4_resident_decode_stage/validation/spark_dsv4_resident_decode_stage_cuda_validation.cu" | cut -d' ' -f1)"
export SPARK_DSV4_REFERENCE_VERIFIER_SHA256="$(sha256sum "${ROOT}/tools/verify_dsv4_ga_reference_fixture.py" | cut -d' ' -f1)"

ARCHIVE="${ROOT}/build/modules/dsv4_pro_resident_decode_stage/libdsv4_resident_decode_stage_b${SPARK_MODULE_BATCH_BUCKET}.a"
CONFIG_SHA="$(printf '%s' "dsv4_pro ga publish-validation stage=0/16 slice=0+4 rows=1 max_seq=4096 logical_pages=1024 physical_pages=1024 mtp=0 graphs=0 source=${SOURCE_SHA} archive=${ARCHIVE_SHA}" | sha256sum | cut -d' ' -f1)"
exec "${ROOT}/modules/dsv4_resident_decode_stage/validation/validate_dsv4_resident_decode_stage_cuda.sh" "${CONFIG_SHA}" "${ARCHIVE}"
