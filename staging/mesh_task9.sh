#!/bin/bash
# Full driver rebuild: module publish (re-stages the link unit with the new
# host archives carrying the mesh-robust transport), driver compile against
# the FRESH source-tree library, transport-string verify, atomic deploy.
set -u
export PATH=/usr/bin:/bin:/usr/sbin:/sbin
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
RR="$HOME/sparkdata/glm5_next.tp16"
SRC="$HOME/g5mesh-src"
LOG=/tmp/mesh_task9.log
exec >> "$LOG" 2>&1
echo "=== START $(date -u +%H:%M:%S) ==="
rm -rf "$SRC"
git clone -q -b lane/glm5next-mtp-accept-takeover https://github.com/sparkpipe/sparkpipe.git "$SRC" || { echo CLONE-FAIL; exit 1; }
cd "$SRC" || { echo NO-SRC; exit 1; }
git log --oneline -1
export PATH=/usr/local/cuda/bin:$PATH

echo "== host build =="
make -j8 build/sparkpipe_model_compile || { echo HOST-BUILD-FAIL; exit 1; }

REV=84c6a6aa9497188e15a635ba793b0f95a79b1033
SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
echo "contract_sha=$SHA"

echo "== module publish =="
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo cache-drop-failed-continuing
make -C modules/glm5_next_resident_decode_stage publish \
  EXPERT_CODEC=fp8 MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH="$RR/packs/glm5_next_stage.tp16.rank0.g5nsp" \
  || { echo PUBLISH-FAIL; exit 1; }

echo "== driver compile (fresh library) =="
build/sparkpipe_model_compile \
  --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
  --library build/module_library --output "$RR" \
  --cc /usr/bin/cc --include include \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
  --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
  --cc-arg -pthread || { echo COMPILE-FAIL; exit 1; }

n_new=$(strings "$RR/stages/stage_000/model_driver.so" | grep -c 'tp-nccl rank0 listening')
echo "driver strings: listening=$n_new"
[ "$n_new" -ge 1 ] || { echo TRANSPORT-NOT-IN-DRIVER; exit 1; }

echo "== deploy 16/16 =="
cp "$RR/stages/stage_000/model_driver.so" /tmp/g5mesh_driver.so
REF=$(sha256sum < /tmp/g5mesh_driver.so | cut -d' ' -f1)
ok=0
for h in $ALL; do
    timeout 120 scp -q -o BatchMode=yes -o ConnectTimeout=8 /tmp/g5mesh_driver.so "$h:/tmp/g5mesh_driver.so" && \
    timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mv /tmp/g5mesh_driver.so \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" && ok=$((ok+1))
done
echo "deployed=$ok/16"
[ "$ok" -eq 16 ] || { echo DEPLOY-FAILED; exit 1; }

n=0
for h in $ALL; do
    s=$(timeout 20 ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" | cut -d' ' -f1)
    [ "$s" = "$REF" ] && n=$((n+1)) || echo "SHA-MISMATCH $h"
done
echo "driver sha $REF verified $n/16"
[ "$n" -eq 16 ] || { echo VERIFY-FAILED; exit 1; }
echo MESH-TASK9-DONE
