#!/usr/bin/env bash
# glm53_m5_stage2.sh — resume of glm53-m5-stage after its path bugs. The
# expensive legs are DONE on spark0 (receipts in /tmp/sparkqueue-glm53-m5-stage.log):
# module publish validator PASS (artifact 8b1b94f5...), driver compiled into
# the runtime root (model_sha256 a40e9ec5...), adapter built at
# build/modules/glm5_next_resident_decode_stage/fp8/. Remaining: install the
# adapter, deploy driver+adapter to all 16 runtime roots (backup first),
# registrar re-stage, sha-verify 16/16.
set -uo pipefail
SRC="$HOME/g5m5-src"
RR="$HOME/sparkdata/glm5_next.tp16"
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
cd "$SRC" || { echo NO-SRC; exit 1; }

AD="$SRC/build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so"
[ -f "$AD" ] || { echo "MISSING adapter $AD"; exit 1; }
[ -f "$RR/lib/model_driver.so" ] || { echo "MISSING $RR/lib/model_driver.so"; exit 1; }

SUF="pre-m5-$(date +%s)"
for h in $ALL; do
  ssh -o BatchMode=yes "$h" "rr=\$HOME/sparkdata/glm5_next.tp16; mkdir -p \$rr/lib; cd \$rr/lib && for f in model_driver.so model_serving_adapter.so; do [ -f \$f ] && cp \$f \$f.$SUF; done" || true
done
for h in $ALL; do
  scp -o BatchMode=yes -q "$RR/lib/model_driver.so" "/tmp/g5m5_driver.so"
  scp -o BatchMode=yes -q "$AD" "/tmp/g5m5_adapter.so"
  scp -o BatchMode=yes -q /tmp/g5m5_driver.so /tmp/g5m5_adapter.so "$h:/tmp/" || { echo "SCP-FAIL $h"; exit 1; }
  ssh -o BatchMode=yes "$h" "mv /tmp/g5m5_driver.so \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so && mv /tmp/g5m5_adapter.so \$HOME/sparkdata/glm5_next.tp16/lib/model_serving_adapter.so" || { echo "MV-FAIL $h"; exit 1; }
done

echo "== registrar re-stage =="
bash tools/registrar_stage.sh || { echo REGISTRAR-FAIL; exit 1; }

echo "== sha-verify 16/16 =="
DREF=$(sha256sum < "$RR/lib/model_driver.so" | cut -d' ' -f1)
AREF=$(sha256sum < "$AD" | cut -d' ' -f1)
dn=0; an=0
for h in $ALL; do
  d=$(ssh -o BatchMode=yes "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" | cut -d' ' -f1)
  a=$(ssh -o BatchMode=yes "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_serving_adapter.so" | cut -d' ' -f1)
  [ "$d" = "$DREF" ] && dn=$((dn+1)) || echo "DRIVER-SHA-MISMATCH $h"
  [ "$a" = "$AREF" ] && an=$((an+1)) || echo "ADAPTER-SHA-MISMATCH $h"
done
echo "driver  $DREF verified $dn/16"
echo "adapter $AREF verified $an/16"
[ "$dn" -eq 16 ] && [ "$an" -eq 16 ] || { echo DEPLOY-INCOMPLETE; exit 1; }
echo STAGE2-DONE
