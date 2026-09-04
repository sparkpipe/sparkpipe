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
    echo "validation configuration must be a lowercase SHA-256 digest" >&2
    exit 2
fi
if [[ ! -s "${module_archive}" ]]; then
    echo "TP module archive is missing or empty: ${module_archive}" >&2
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
tp_degree="${SPARK_DSV4_TP_DEGREE:-16}"
if [[ ! "${tp_degree}" =~ ^(1|2|4|8|16)$ ]]; then
    echo "SPARK_DSV4_TP_DEGREE must be one of 1,2,4,8,16" >&2
    exit 2
fi
rank="${SPARK_DSV4_TP_RANK:-}"
if [[ ! "${rank}" =~ ^[0-9]+$ || "${rank}" -ge "${tp_degree}" ]]; then
    echo "SPARK_DSV4_TP_RANK must be in [0,$((tp_degree - 1))]" >&2
    exit 2
fi

ar t "${module_archive}" | grep -q 'spark_dsv4_resident_decode_stage_cuda.o'
python3 "${script_directory}/dsv4_tp16_stagepack.py" \
    --input-pack "${SPARK_DSV4_TP16_SOURCE_PACK_PATH}" \
    --output "${SPARK_DSV4_STAGE_PACK_PATH}" \
    --rank "${rank}" \
    --tp-degree "${tp_degree}" \
    --verify-output >/dev/null

echo "dsv4 TP${tp_degree} pack and CUDA module archive validation pass rank=${rank}"
