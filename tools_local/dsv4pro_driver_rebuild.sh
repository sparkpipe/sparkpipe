#!/bin/bash
# DSV4 Pro driver rebuild from the staged checkout (single node).
# Host build -> module publish (GPU validator vs the deployed rank-0
# TP4PP4 pack, geometry mirrored from the deployed deployment config)
# -> sparkpipe_model_compile -> adapter staged as .new. Deployment to
# the fleet is a separate, later task. Exit non-zero names the stage.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
[[ -f Makefile ]] || { echo "NO-REPO-MAKEFILE at $REPO"; exit 1; }
RR=/home/$(hostname)/sparkdata/dsv4_pro.tp4pp4
PACK="$RR/packs/dsv4_pro_tp4_pp4_stage.spstage"
[[ -s "$PACK" ]] || { echo "MISSING-RANK0-PACK $PACK"; exit 1; }
export PATH=/usr/local/cuda/bin:$PATH
command -v nvcc >/dev/null || { echo "NO-NVCC"; exit 1; }
sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || true

echo "== host build =="
make -j8 build/sparkpipe_model_residentd build/sparkpipe_model_batch \
  build/sparkpipe_module_publish build/sparkpipe_model_compile \
  build/sparkpipe_model_api || { echo HOST-BUILD-FAIL; exit 1; }

echo "== module publish (GPU validator, real rank0 pack; deployed geometry) =="
make -C modules/dsv4_resident_decode_stage -f Makefile.pro publish \
  PRO_EXPERT_CODEC=mxfp4 PRO_KV_CODEC=bf16 \
  STAGE_PACK_PATH="$PACK" \
  STAGE_COUNT=4 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=16 \
  MAX_ACTIVE_SEQUENCES=1024 MAX_SEQUENCE_POSITIONS=33024 \
  PIPELINE_SLOT_COUNT=13 PHYSICAL_PAGE_CAPACITY=1024 \
  LOGICAL_PAGE_CAPACITY=16384 MTP_LAYER_COUNT=3 CUDA_GRAPH_COUNT=0 \
  || { echo PUBLISH-FAIL; exit 1; }

echo "== driver compile =="
build/sparkpipe_model_compile \
  --model examples/model_descriptions/dsv4_pro_resident_decode_stage_firmware.json \
  --library build/module_library --output "$RR" \
  --cc /usr/bin/cc --include include \
  --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
  --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
  --cc-arg -pthread \
  || { echo COMPILE-FAIL; exit 1; }

echo "== adapter + binaries staged as .new (deploy is a separate task) =="
AD=$(ls build/modules/dsv4_pro_resident_decode_stage/*/lib*adapter*.so 2>/dev/null | head -1)
[[ -n "$AD" ]] || { echo ADAPTER-NOT-FOUND; ls build/modules/dsv4_pro_resident_decode_stage/; exit 1; }
cp "$AD" "$RR/lib/model_serving_adapter.so.new"
cp build/sparkpipe_model_residentd build/sparkpipe_model_api \
  build/sparkpipe_model_batch "$RR/bin/" 2>/dev/null || true
sha256sum "$RR/lib/model_driver.so" "$RR/lib/model_serving_adapter.so.new" \
  "$RR/bin/sparkpipe_model_residentd" | tee "$RR/driver_rebuild_receipt.txt"
echo DSV4PRO-DRIVER-REBUILD-OK
