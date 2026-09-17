#!/usr/bin/env bash
# k3_tp4_slice.sh STAGE_PACK OUT_PREFIX — slice one PP stage pack into four
# TP4 rank packs with the K3 sharder, then verify each output opens.
# Run on the stage node that holds the pack. The rank packs (~87 GB each,
# 1/16 of the model) are the deployment units copied to the stage's four
# ranks under /home/<user>/sparkdata/k3.mxfp4.tp4pp4/packs/.
set -euo pipefail
PACK="${1:?usage: k3_tp4_slice.sh STAGE_PACK OUT_PREFIX}"
PREFIX="${2:?usage: k3_tp4_slice.sh STAGE_PACK OUT_PREFIX}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHONDONTWRITEBYTECODE=1 python3 "${SCRIPT_DIR}/k3_shard.py" "${PACK}" "${PREFIX}" 4
for rank in 0 1 2 3; do
    out="${PREFIX}.rank0${rank}.pack"
    test -s "$out" || { echo "missing $out" >&2; exit 1; }
    du -sh "$out"
done
echo "k3_tp4_slice done: 4 rank packs from $PACK"
