#!/bin/bash
set -u
RANK_HEX="$1"
RANK_DEC=$((16#$RANK_HEX))
SRC="$HOME/lingbuild/packs"
DST="$HOME/sparkdata/ling.bf16.tp16/packs"
PACK="ling.bf16.tp16.rank${RANK_HEX}.sp"
EXPECT_BYTES=15725069824
[ -f "$SRC/$PACK" ] || { echo "SPARK_FAIL 10 missing $SRC/$PACK"; exit 10; }
[ -f "$SRC/$PACK.sha256" ] || { echo "SPARK_FAIL 11 missing sidecar"; exit 11; }
[ -f "$SRC/$PACK.experts" ] || { echo "SPARK_FAIL 12 missing experts"; exit 12; }
[ -f "$SRC/receipts/rank${RANK_DEC}.json" ] || { echo "SPARK_FAIL 13 missing receipt"; exit 13; }
BYTES=$(stat -c%s "$SRC/$PACK")
[ "$BYTES" -eq "$EXPECT_BYTES" ] || { echo "SPARK_FAIL 14 size $BYTES"; exit 14; }
cd "$SRC" && sudo -n sha256sum -c "$PACK.sha256" --quiet || { echo "SPARK_FAIL 15 sidecar"; exit 15; }
RECEIPT_SHA=$(python3 -c "import json;print(json.load(open('$SRC/receipts/rank${RANK_DEC}.json'))['sha256'])")
ACTUAL=$(sudo -n sha256sum "$SRC/$PACK" | cut -d' ' -f1)
[ "$RECEIPT_SHA" = "$ACTUAL" ] || { echo "SPARK_FAIL 16 receipt $RECEIPT_SHA != $ACTUAL"; exit 16; }
if [ -f "$DST/$PACK" ]; then
  lsof "$DST/$PACK" 2>/dev/null | grep -q "^COMMAND" && { echo "SPARK_FAIL 17 open handles"; exit 17; }
  ATTRS=$(lsattr "$DST/$PACK" 2>/dev/null | cut -d' ' -f1)
  case "$ATTRS" in *i*) echo "SPARK_FAIL 18 immutable"; exit 18;; esac
fi
sudo -n cp "$SRC/$PACK" "$DST/.sp5stage" || { echo "SPARK_FAIL 19 cp"; exit 19; }
STAGE_SHA=$(sudo -n sha256sum "$DST/.sp5stage" | cut -d' ' -f1)
[ "$STAGE_SHA" = "$ACTUAL" ] || { sudo -n rm -f "$DST/.sp5stage"; echo "SPARK_FAIL 20 stage verify"; exit 20; }
sudo -n mv "$DST/.sp5stage" "$DST/$PACK" || { echo "SPARK_FAIL 21 mv"; exit 21; }
OWNER=$(stat -c '%U:%G' "$DST")
sudo -n chown "$OWNER" "$DST/$PACK" || { echo "SPARK_FAIL 26 chown"; exit 26; }
sudo -n cp "$SRC/$PACK.experts" "$DST/$PACK.experts" || { echo "SPARK_FAIL 22 experts"; exit 22; }
sudo -n cp "$SRC/$PACK.sha256" "$DST/$PACK.sha256" || { echo "SPARK_FAIL 23 sidecar"; exit 23; }
sudo -n mkdir -p "$DST/receipts"
sudo -n cp "$SRC/receipts/rank${RANK_DEC}.json" "$DST/receipts/rank${RANK_DEC}.json" || { echo "SPARK_FAIL 24 receipt"; exit 24; }
(cd "$DST" && sudo -n sha256sum -c "$PACK.sha256" --quiet) || { echo "SPARK_FAIL 25 dest verify"; exit 25; }
sudo -n rm -f "$SRC/$PACK" "$SRC/$PACK.sha256" "$SRC/$PACK.experts"
echo "SPARK_OK rank$RANK_HEX placed $ACTUAL"
