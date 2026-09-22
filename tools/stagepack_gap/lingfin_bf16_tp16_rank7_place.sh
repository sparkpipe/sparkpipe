#!/usr/bin/env bash
set -euo pipefail
SRC="spark8:build-stagepack-gaps/lingfin.bf16.tp16"
DST="$HOME/sparkdata/lingfin.bf16.tp16/packs"
mkdir -p "$DST/receipts"
rsync -av \
  "$SRC/lingfin.bf16.tp16.rank7.sp" \
  "$SRC/lingfin.bf16.tp16.rank7.sp.sha256" \
  "$SRC/lingfin.bf16.tp16.rank7.sp.experts" \
  "$DST/"
rsync -av "$SRC/receipts/rank7.json" "$DST/receipts/"
cd "$DST"
sha256sum -c lingfin.bf16.tp16.rank7.sp.sha256
chmod 664 lingfin.bf16.tp16.rank7.sp lingfin.bf16.tp16.rank7.sp.experts
