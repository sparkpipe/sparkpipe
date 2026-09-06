#!/usr/bin/env bash
# m5_tree_stage.sh — bring up the glm53flash serving stack on the TREE
# allreduce (local g5mesh-src, root glm53flash.bf16.tp16). Steps follow
# tools/glm53_m5_stage.sh but run on spark5 from the already-built tree.
set -uo pipefail
SRC="$HOME/g5mesh-src"
RR="$HOME/sparkdata/glm53flash.bf16.tp16"
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"
cd "$SRC" || { echo NO-SRC; exit 1; }
git log --oneline -1 2>/dev/null || true
echo "== cache-drop =="
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || echo CACHE-DROP-SKIPPED
free -g | head -2

echo "== host build =="
export PATH=$PATH:/usr/local/cuda/bin
make -j8 build/sparkpipe_model_residentd build/sparkpipe_model_batch \
  build/sparkpipe_module_publish build/sparkpipe_model_compile \
  build/sparkpipe_model_api || { echo HOST-BUILD-FAIL; exit 1; }

REV=84c6a6aa9497188e15a635ba793b0f95a79b1033
SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)
PACK="$RR/packs/glm53flash.bf16-official.tp16.rank5.sp"

echo "== module publish (GPU validator, rank5 pack) =="
make -C modules/glm5_next_resident_decode_stage publish \
  EXPERT_CODEC=fp8 MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH="$PACK" \
  || { echo PUBLISH-FAIL; exit 1; }

echo "== driver compile =="
build/sparkpipe_model_compile \
  --model examples/model_descriptions/glm5_next_resident_decode_stage_fp8_firmware.json \
  --library build/module_library --output "$RR" \
  --cc /usr/bin/cc --include include \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
  --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
  --cc-arg -pthread \
  || { echo COMPILE-FAIL; exit 1; }

echo "== adapter =="
make -C modules/glm5_next_resident_decode_stage adapter EXPERT_CODEC=fp8 \
  MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  || { echo ADAPTER-FAIL; exit 1; }
AD="$SRC/build/modules/glm5_next_resident_decode_stage/fp8/libglm5_next_serving_adapter_fp8.so"
mkdir -p "$RR/lib" "$RR/bin"
cp "$AD" "$RR/lib/model_serving_adapter.so"
cp build/sparkpipe_model_residentd build/sparkpipe_model_api build/sparkpipe_model_batch "$RR/bin/"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$RR/lib/hidden_transport.so"

echo "== deploy configs (tree) + lib/bin fleet-wide =="
GLM5_NEXT_BACKEND=hidden_transport python3 tools/glm5_next_gen_deployment.py --output "$RR"
for f in config lib bin; do
  tar -C "$RR" -cf /tmp/g5tree_$f.tar $f || { echo TAR-FAIL $f; exit 1; }
done
for h in $ALL; do
  [ "$h" = "spark5" ] && continue
  ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "mkdir -p \$HOME/sparkdata/glm53flash.bf16.tp16" || true
  for f in config lib bin; do
    scp -o BatchMode=yes -o ConnectTimeout=8 -q /tmp/g5tree_$f.tar "$h:/tmp/g5tree_$f.tar" || { echo SCP-FAIL $h $f; exit 1; }
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "tar -C \$HOME/sparkdata/glm53flash.bf16.tp16 -xf /tmp/g5tree_$f.tar && rm /tmp/g5tree_$f.tar" || { echo UNTAR-FAIL $h; exit 1; }
  done
done
echo "== sha-verify lib 16/16 =="
REF=$(sha256sum < "$RR/lib/model_driver.so" | cut -d' ' -f1)
n=0
for h in $ALL; do
  s=$(ssh -o BatchMode=yes -o ConnectTimeout=8 "$h" "sha256sum < \$HOME/sparkdata/glm53flash.bf16.tp16/lib/model_driver.so" | cut -d' ' -f1)
  [ "$s" = "$REF" ] && n=$((n+1)) || echo "SHA-MISMATCH $h"
done
echo "driver sha verified on $n/16"
[ "$n" -eq 16 ] || { echo DEPLOY-INCOMPLETE; exit 1; }
echo TREE-STAGE-DONE
