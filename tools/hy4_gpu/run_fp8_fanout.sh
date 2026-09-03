#!/bin/bash
# hy4 FP8 fanout packer runner: internal redirects, immune to launcher drops.
OUT=/home/sparkc/hy4-fp8-packs
SRC=/mnt/model-warm/hy4-preview-fp8-official
cd "$OUT" || exit 1
rm -f "$OUT"/model-fp8-tp16-rank-*.safetensors "$OUT"/manifest-rank-*.json
exec python3 /home/sparkc/hy4-fp8-packs/hy4_fp8_stagepack_fanout.py \
  --checkpoint "$SRC" --output-directory "$OUT" \
  >> "$OUT/fanout.log" 2>&1 < /dev/null
