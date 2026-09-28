#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
    echo "usage: $0 VALIDATION_CONFIGURATION_SHA256 MODULE_ARCHIVE" >&2
    exit 2
fi
[[ "$1" =~ ^[0-9a-f]{64}$ ]] || { echo "mimo26: configuration hash must be 64 hex characters" >&2; exit 2; }
archive="$2"
test -r "${archive}" || { echo "mimo26: archive not readable: ${archive}" >&2; exit 2; }
test -r "${SPARK_MIMO26_STAGE_PACK_PATH:?}" || { echo "mimo26: stage pack not readable: ${SPARK_MIMO26_STAGE_PACK_PATH}" >&2; exit 2; }
symbols="$(nm -g --defined-only "${archive}")"
for entry in Initialize Execute Admit Snapshot Destroy; do
    grep -q " T SparkMimo26ResidentDecodeStage${entry}$" <<<"${symbols}" || { echo "mimo26: archive lacks SparkMimo26ResidentDecodeStage${entry}" >&2; exit 1; }
done
for engine in Create Step ReadStats Destroy; do
    grep -q " T SparkMimo26RankEngine${engine}$" <<<"${symbols}" || { echo "mimo26: archive lacks SparkMimo26RankEngine${engine}" >&2; exit 1; }
done
printf 'mimo26 archive tier PASS: entry points and rank engine present; numerical acceptance is the TP4 fleet receipt (qualification/mimo26/runs)\n'
