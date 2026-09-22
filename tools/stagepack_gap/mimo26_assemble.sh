#!/usr/bin/env bash
set -euo pipefail
ARM="$1"; TP="$2"; RANK="$3"
case "$ARM" in
  pro) CKPT=/mnt/model-warm/mimo-v2.6-pro-rl; SET=mimo26pro.mxfp4.tp8 ;;
  flash) CKPT=/mnt/model-warm/mimo-v2.6-flash-rl; SET=mimo26flash.mxfp4.tp4 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
ROOT="$HOME/sparkdata/$SET/emit/rank$RANK"
python3 tools/mimo26_stagepack.py \
  --arm "$ARM" --checkpoint "$CKPT" --tp "$TP" --rank "$RANK" \
  --out "$ROOT/rank$RANK.sp" --stage-dir "$ROOT/stage" --assemble
python3 tools/mimo26_stagepack.py \
  --arm "$ARM" --checkpoint "$CKPT" --tp "$TP" --rank "$RANK" \
  --out "$ROOT/rank$RANK.sp" --stage-dir "$ROOT/stage" --verify
cd "$ROOT"
sha256sum "rank$RANK.sp" > "rank$RANK.sp.sha256"
