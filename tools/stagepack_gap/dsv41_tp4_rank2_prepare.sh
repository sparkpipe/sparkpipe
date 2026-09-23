#!/usr/bin/env bash
set -euo pipefail
WARM=/mnt/model-warm/deepseek-v4.1-flash
OUT="$HOME/build-stagepack-gaps/dsv41flash.mxfp4.tp4"
mkdir -p "$OUT"
python3 tools/dsv41_flash_stagepack.py headers "$WARM" \
  "$WARM/model.safetensors.index.json" "$OUT/headers.json"
python3 tools/dsv41_flash_stagepack.py plan "$OUT/headers.json" "$OUT" 4 \
  dba1be0a40aa45a94ad051997016db3960a90277 \
  44fcba0b411b23db03d3b5c3807a76024f81df2af299030d768e26e840aadb30 \
  8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879 \
  d58a0258b67ba0208794e3a6ea3fdf802e32703532f480b3c0b26ce2368a6f66
