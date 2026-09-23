#!/usr/bin/env bash
set -euo pipefail
LO="$1"; HI="$2"
WARM=/mnt/model-warm/deepseek-v4.1-flash
OUT="$HOME/build-stagepack-gaps/dsv41flash.mxfp4.tp4"
MARK=$(printf 'copy_%02d_%02d.done' "$LO" "$HI")
[ -f "$OUT/$MARK" ] && { echo "$MARK already complete"; exit 0; }
python3 tools/dsv41_flash_stagepack.py copy "$OUT/headers.json" \
  "$OUT/plan.json" "$OUT" "$WARM" "$LO" "$HI"
