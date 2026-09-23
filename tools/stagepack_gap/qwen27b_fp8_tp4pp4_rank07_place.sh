#!/usr/bin/env bash
set -euo pipefail
SRC="spark8:build-stagepack-gaps/qwen27b.fp8.tp4pp4"
DST="$HOME/sparkdata/qwen27b.fp8.tp4pp4/packs"
mkdir -p "$DST"
rsync -av \
  "$SRC/qwen27b.fp8.tp4pp4.rank07.spstage" \
  "$SRC/qwen27b.fp8.tp4pp4.rank07.spstage.sha256" \
  "$SRC/qwen27b.fp8.tp4pp4.rank07.spstage.receipt.json" \
  "$DST/"
cd "$DST"
want=$(cut -d' ' -f1 qwen27b.fp8.tp4pp4.rank07.spstage.sha256)
got=$(sha256sum qwen27b.fp8.tp4pp4.rank07.spstage | cut -d' ' -f1)
[ "$want" = "$got" ]
printf '%s  %s\n' "$want" qwen27b.fp8.tp4pp4.rank07.spstage > qwen27b.fp8.tp4pp4.rank07.spstage.sha256
chmod 664 qwen27b.fp8.tp4pp4.rank07.spstage
