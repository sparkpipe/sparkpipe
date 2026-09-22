#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/build-stagepack-gaps/qwen27b.fp8.tp4pp4"
mkdir -p "$OUT"
python3 tools/qwen38_27b_stagepack.py \
  --checkpoint /mnt/model-warm/qwen3.8-27b-fp8 \
  --output "$OUT/qwen27b.fp8.tp4pp4.rank07.spstage" \
  --tp-degree 4 --tp-rank 3 --first-layer 16 --layer-count 16 \
  --ffn-format bf16
sha256sum "$OUT/qwen27b.fp8.tp4pp4.rank07.spstage" \
  > "$OUT/qwen27b.fp8.tp4pp4.rank07.spstage.sha256"
