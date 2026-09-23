#!/usr/bin/env bash
set -euo pipefail
ARM="$1"; TP="$2"; RANK="$3"; SRC_HOST="$4"; SRCSUB="${5:-emit}"
case "$ARM" in
  pro) SET=mimo26pro.mxfp4.tp8 ;;
  flash) SET=mimo26flash.mxfp4.tp4 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac
if [ "$SRCSUB" = "packs" ]; then
  SRC="$SRC_HOST:sparkdata/$SET/packs"
else
  SRC="$SRC_HOST:sparkdata/$SET/$SRCSUB/rank$RANK"
fi
DST="$HOME/sparkdata/$SET/packs"
mkdir -p "$DST"
rsync -a --partial --append-verify --bwlimit="${RSYNC_BWLIMIT:-150000}" "$SRC/rank$RANK.sp" "$SRC/rank$RANK.sp.receipt.json" "$SRC/rank$RANK.sp.sha256" "$DST/"
cd "$DST"
want=$(cut -d' ' -f1 "rank$RANK.sp.sha256")
got=$(sha256sum "rank$RANK.sp" | cut -d' ' -f1)
[ "$want" = "$got" ]
printf '%s  %s\n' "$want" "rank$RANK.sp" > "rank$RANK.sp.sha256"
chmod 664 "rank$RANK.sp"
