#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
    exit 2
fi

configuration_hash="$1"
module_archive="$2"
script_directory="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! "${configuration_hash}" =~ ^[0-9a-f]{64}$ ]]; then
    echo "TP16 validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "TP16 module archive is missing or empty: ${module_archive}" >&2
    exit 2
fi
if [[ -z "${SPARK_DSV4_STAGE_PACK_PATH:-}" || ! -s "${SPARK_DSV4_STAGE_PACK_PATH}" ]]; then
    echo "SPARK_DSV4_STAGE_PACK_PATH must name a readable rank pack" >&2
    exit 2
fi
if [[ -z "${SPARK_DSV4_TP16_SOURCE_PACK_PATH:-}" || ! -s "${SPARK_DSV4_TP16_SOURCE_PACK_PATH}" ]]; then
    echo "SPARK_DSV4_TP16_SOURCE_PACK_PATH must name the full source pack" >&2
    exit 2
fi
if [[ ! "${SPARK_DSV4_TP16_RANK:-}" =~ ^([0-9]|1[0-5])$ ]]; then
    echo "SPARK_DSV4_TP16_RANK must be a rank in [0,15]" >&2
    exit 2
fi

ar t "${module_archive}" | grep -q 'spark_dsv4_resident_decode_stage_cuda.o'
python3 "${script_directory}/dsv4_tp16_stagepack.py" \
    --input-pack "${SPARK_DSV4_TP16_SOURCE_PACK_PATH}" \
    --output "${SPARK_DSV4_STAGE_PACK_PATH}" \
    --rank "${SPARK_DSV4_TP16_RANK}" \
    --verify-output >/dev/null

echo "dsv4 TP16 pack and CUDA module archive validation pass rank=${SPARK_DSV4_TP16_RANK}"
