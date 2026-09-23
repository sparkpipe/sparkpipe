#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/build-stagepack-gaps/lingfin.bf16.tp16"
mkdir -p "$OUT"
python3 tools/ling_stagepack.py \
  --source /mnt/model-warm/ling-3.0-flash-fin \
  --output-dir "$OUT" \
  --model lingfin --expert-codec bf16 --tp-degree 16 --tp-rank 7 --resume
