#!/usr/bin/env bash
# glm53_m5_stage.sh — M5 prerequisite, queue-staged on spark0 only (GPU use:
# the module publish validator). Rebuilds the serving stack from current
# lane/glm53 (= origin/main + lane commits) and redeploys it to all 16
# runtime roots: the deployed set predates the completion-emit fix and the
# fixed2 pack repack. Backup-with-suffix then install, sha-verify 16/16.
#
# After this: tools/glm53_m5_cell_task.sh (all-16, exclusive window).
set -uo pipefail
SRC="$HOME/g5m5-src"
RR="$HOME/sparkdata/glm5_next.tp16"
ALL="spark0 spark1 spark2 spark3 spark4 spark5 spark6 spark7 spark8 spark9 sparka sparkb sparkc sparkd sparke sparkf"

echo "== clone lane/glm53 =="
rm -rf "$SRC"
git clone -q -b lane/glm53 https://github.com/sparkpipe/sparkpipe.git "$SRC" \
  || { echo CLONE-FAIL; exit 1; }
cd "$SRC" || { echo NO-SRC; exit 1; }
git log --oneline -1

echo "== cache-drop (GB10 page-cache trap; validator loads 21.7G pack) =="
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || { echo CACHE-DROP-FAIL; exit 1; }
free -g | head -2

echo "== host build =="
make -j8 build/sparkpipe_model_residentd build/sparkpipe_model_batch \
  build/sparkpipe_module_publish build/sparkpipe_model_compile \
  build/sparkpipe_model_api \
  || { echo HOST-BUILD-FAIL; exit 1; }

REV=84c6a6aa9497188e15a635ba793b0f95a79b1033
SHA=$(shasum -a 256 model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1)

echo "== module publish (GPU validator, real rank0 pack) =="
make -C modules/glm5_next_resident_decode_stage publish \
  EXPERT_CODEC=fp8 MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH="$RR/packs/glm5_next_stage.tp16.rank0.g5nsp" \
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

echo "== adapter + binaries into runtime root =="
make -C modules/glm5_next_resident_decode_stage adapter EXPERT_CODEC=fp8 \
  MODEL_REVISION=$REV CONTRACT_SHA256=$SHA \
  || { echo ADAPTER-FAIL; exit 1; }
cp build/libglm5_next_serving_adapter_fp8.so "$RR/lib/model_serving_adapter.so.new"
cp build/sparkpipe_model_residentd build/sparkpipe_model_api build/sparkpipe_model_batch "$RR/bin/"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$RR/lib/hidden_transport.so.new" 2>/dev/null \
  || echo "note: no freshly built hidden transport, keeping incumbent"

echo "== deploy lib/*.so + registrar fleet-wide (backup, install, sha) =="
SUF="pre-m5-$(date +%s)"
for f in model_driver.so model_serving_adapter.so hidden_transport.so; do
  [ -f "$RR/lib/$f.new" ] || { echo "MISSING $RR/lib/$f.new"; exit 1; }
  cp "$RR/lib/$f.new" /tmp/g5m5_$f
  rm -f "$RR/lib/$f.new"
done
bash tools/registrar_stage.sh || { echo REGISTRAR-FAIL; exit 1; }
for h in $ALL; do
  ssh -o BatchMode=yes "$h" "rr=\$HOME/sparkdata/glm5_next.tp16; cd \$rr/lib && for f in model_driver.so model_serving_adapter.so hidden_transport.so; do [ -f \$f ] && cp \$f \$f.$SUF; done" || true
  for f in model_driver.so model_serving_adapter.so hidden_transport.so; do
    scp -o BatchMode=yes -q "/tmp/g5m5_$f" "$h:/tmp/g5m5_$f" || { echo "SCP-FAIL $h $f"; exit 1; }
    ssh -o BatchMode=yes "$h" "mv /tmp/g5m5_$f \$HOME/sparkdata/glm5_next.tp16/lib/$f" || { echo "MV-FAIL $h"; exit 1; }
  done
done
echo "== sha-verify 16/16 =="
REF=$(sha256sum < /tmp/g5m5_model_driver.so | cut -d' ' -f1)
n=0
for h in $ALL; do
  s=$(ssh -o BatchMode=yes "$h" "sha256sum < \$HOME/sparkdata/glm5_next.tp16/lib/model_driver.so" | cut -d' ' -f1)
  [ "$s" = "$REF" ] && n=$((n+1)) || echo "SHA-MISMATCH $h $s"
done
echo "driver sha $REF verified on $n/16"
[ "$n" -eq 16 ] || { echo DEPLOY-INCOMPLETE; exit 1; }
echo STAGE-DONE
