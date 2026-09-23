#!/usr/bin/env bash
set -euo pipefail
SRC="spark8:build-stagepack-gaps/lingfin.bf16.tp16"
DST="$HOME/sparkdata/lingfin.bf16.tp16/packs"
mkdir -p "$DST/receipts"
rsync -a --partial --append-verify \
  "$SRC/lingfin.bf16.tp16.rank7.sp" \
  "$SRC/lingfin.bf16.tp16.rank7.sp.sha256" \
  "$SRC/lingfin.bf16.tp16.rank7.sp.experts" \
  "$DST/"
rsync -av "$SRC/receipts/rank7.json" "$DST/receipts/"
cd "$DST"
want=$(cut -d' ' -f1 lingfin.bf16.tp16.rank7.sp.sha256)
got=$(sha256sum lingfin.bf16.tp16.rank7.sp | cut -d' ' -f1)
[ "$want" = "$got" ]
printf '%s  %s\n' "$want" lingfin.bf16.tp16.rank7.sp > lingfin.bf16.tp16.rank7.sp.sha256
chmod 664 lingfin.bf16.tp16.rank7.sp lingfin.bf16.tp16.rank7.sp.experts
