#!/usr/bin/env bash
set -euo pipefail
ARM="$1"; TP="$2"; RANK="$3"; SRC_HOST="$4"; SRCSUB="${5:-emit}"
case "$ARM" in
  pro) SET=mimo26pro.mxfp4.tp8 ;;
  flash) SET=mimo26flash.mxfp4.tp4 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
SRC="$SRC_HOST:sparkdata/$SET/$SRCSUB/rank$RANK"
DST="$HOME/sparkdata/$SET/packs"
mkdir -p "$DST"
rsync -av "$SRC/rank$RANK.sp" "$SRC/rank$RANK.sp.receipt.json" "$SRC/rank$RANK.sp.sha256" "$DST/"
cd "$DST"
sha256sum -c "rank$RANK.sp.sha256"
chmod 664 "rank$RANK.sp"
