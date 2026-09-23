#!/usr/bin/env bash
set -euo pipefail
OUT="$HOME/build-stagepack-gaps/dsv41flash.mxfp4.tp4"
for LO_HI in "1 6" "7 12" "13 18" "19 24" "25 30" "31 36" "37 42" "43 48"; do
  set -- $LO_HI
  MARK=$(printf 'copy_%02d_%02d.done' "$1" "$2")
  [ -f "$OUT/$MARK" ] || { echo "missing $MARK" >&2; exit 1; }
done
cc -O2 \
  -Iinclude -Imodel-families/common/include \
  -Imodel-families/dsv41_flash/include \
  -Imodules/dsv41_flash_resident_decode_stage/include \
  -Imodules/dsv41_flash_resident_decode_stage/source \
  modules/dsv41_flash_resident_decode_stage/tools/dsv41_flash_experts_manifest.c \
  src/spark_ck128.c -o "$OUT/dsv41_flash_experts_manifest"
"$OUT/dsv41_flash_experts_manifest" "$OUT/rank2.spstage" \
  "$OUT/rank2.spstage.experts"
python3 tools/dsv41_verify_pack.py --pack "$OUT/rank2.spstage" --emit-receipt
