#!/bin/bash
# Driver rebuild with the mesh-robust transport: clone the lane branch,
# host-build, compile the driver against the DEPLOYED module library (module
# itself unchanged), verify the new transport strings landed in the .so,
# deploy 16/16 atomically, sha-verify. Runs detached; journal past TTL.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
RR="$HOME/sparkdata/glm5_next.tp16"
SRC="$HOME/g5mesh-src"
LOG=/tmp/mesh_task8.log
exec >> "$LOG" 2>&1
echo "=== START $(date -u +%H:%M:%S) ==="

rm -rf "$SRC"
git clone -q -b lane/glm5next-mtp-accept-takeover https://github.com/sparkpipe/sparkpipe.git "$SRC" || { echo CLONE-FAIL; exit 1; }
cd "$SRC" || { echo NO-SRC; exit 1; }
git log --oneline -1

echo "== host build =="
export PATH=/usr/local/cuda/bin:$PATH
make -j8 build/sparkpipe_model_compile || { echo HOST-BUILD-FAIL; exit 1; }

echo "== driver compile =="
build/sparkpipe_model_compile \
  --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
  --library "$RR/build/module_library" --output "$RR" \
  --cc /usr/bin/cc --include include \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
  --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
  --cc-arg -pthread || { echo COMPILE-FAIL; exit 1; }

echo "== verify transport strings in driver =="
n_new=$(strings "$RR/lib/model_driver.so" | grep -c 'tp-nccl rank0 listening')
n_err=$(strings "$RR/lib/model_driver.so" | grep -c 'sparkpipe_tp_nccl_error')
echo "driver strings: listening=$n_new tp_nccl_error=$n_err"
[ "$n_new" -ge 1 ] || { echo TRANSPORT-NOT-IN-DRIVER; exit 1; }

echo "== deploy 16/16 =="
cp "$RR/lib/model_driver.so" /tmp/g5mesh_driver.so
REF=$(sha256sum < /tmp/g5mesh_driver.so | cut -d' ' -f1)
ok=0
for h in $ALL; do
    timeout 120 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5mesh_driver.so "$h:/tmp/g5mesh_driver.so" && \
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mv /tmp/g5mesh_driver.so \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" && ok=$((ok+1))
done
echo "deployed=$ok/16"
[ "$ok" -eq 16 ] || { echo DEPLOY-FAILED; exit 1; }

echo "== sha-verify =="
n=0
for h in $ALL; do
    s=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" | cut -d' ' -f1)
    [ "$s" = "$REF" ] && n=$((n+1)) || echo "SHA-MISMATCH $h"
done
echo "driver sha $REF verified $n/16"
[ "$n" -eq 16 ] || { echo VERIFY-FAILED; exit 1; }
echo MESH-TASK8-DONE
