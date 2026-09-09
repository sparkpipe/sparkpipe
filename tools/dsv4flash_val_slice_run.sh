#!/usr/bin/env bash
# dsv4flash: regenerate the 3-layer validation slice (pp13 shape) from the
# warm source — the publish-validation pack shape every prior dsv4 module
# publish used (TP1/PP13 slice semantics; the family validators on main are
# TP1/B1-only by construction). CPU job; streams only the needed tensors.
set -euo pipefail
OUT=/home/spark2/lane-dsv4flash-build/build/dsv4_val3
mkdir -p "$OUT"
python3 tools/dsv4_stagepack.py \
    --model-dir /mnt/model-warm/deepseek-v4-flash-0731 \
    --first-layer 3 --layer-count 3 \
    --output "$OUT/dsv4_flash_val3.spstage"
python3 tools/dsv4_stagepack.py --verify-pack "$OUT/dsv4_flash_val3.spstage"
sha256sum "$OUT/dsv4_flash_val3.spstage"
echo "VAL-SLICE-DONE"
