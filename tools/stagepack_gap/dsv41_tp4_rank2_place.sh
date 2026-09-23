#!/usr/bin/env bash
set -euo pipefail
SRC="spark8:build-stagepack-gaps/dsv41flash.mxfp4.tp4"
DST="$HOME/sparkdata/dsv41flash.mxfp4.tp4/packs"
mkdir -p "$DST"
rsync -a "$SRC/rank2.spstage" "$SRC/rank2.spstage.experts" \
  "$SRC/rank2.spstage.receipt.json" "$SRC/rank2.spstage.sha256" "$DST/"
cd "$DST"
want=$(cut -d' ' -f1 rank2.spstage.sha256)
got=$(sha256sum rank2.spstage | cut -d' ' -f1)
[ "$want" = "$got" ]
printf '%s  %s\n' "$want" rank2.spstage > rank2.spstage.sha256
chmod 664 rank2.spstage
date -u +%Y-%m-%dT%H:%M:%SZ > rank2.placed.stamp
