#!/bin/bash
# DSV4 Flash M1 task A (queued, GPU): publish b1 module (GPU validator on
# the 3-layer v4 validation slice), compile the driver, deploy runtime +
# v4 rank packs to the TP4 ranks. Runs ON spark5; home dirs only.
set -euo pipefail
BASE=/home/spark5/lane-dsv4flash-m1
SRC=$BASE/src; RT=$BASE/rt; BIS=/home/spark5/lane-dsv4bisect
cd "$SRC"

echo "== [1/3] publish b1 module (GPU validator, 3-layer v4 slice)"
make -C modules/dsv4_resident_decode_stage publish_variants \
    MODULE_BATCH_VARIANT_BUCKETS=1 CUDA_ARCH=sm_121a \
    STAGE_PACK_PATH=$BIS/packs/dsv4_flash_v4_val3.spstage \
    STAGE_COUNT=13 STAGE_INDEX=1 STAGE_FIRST_LAYER=3 STAGE_LAYER_COUNT=3 \
    MAX_ACTIVE_SEQUENCES=1 PIPELINE_SLOT_COUNT=1

echo "== [2/3] compile model driver"
rm -rf "$BASE/driver-out"
build/sparkpipe_model_compile \
    --model examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json \
    --stage dsv4_resident_decode_stage \
    --library build/module_library \
    --output "$BASE/driver-out" \
    --include include \
    --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcuda --cc-arg -lcudart \
    --cc-arg -lstdc++ --cc-arg -ldl --cc-arg -lm --cc-arg -pthread
cp "$BASE/driver-out/model_driver.so" "$RT/lib/model_driver.so"
sha256sum "$RT/lib/model_driver.so" | tee -a "$BASE/results/m1_cpu_build_receipt.txt"

echo "== [3/3] deploy runtime + rank packs"
declare -A RANKOF=( [spark4]=0 [spark5]=1 [spark6]=2 [spark7]=3 )
for h in spark4 spark6 spark7; do
  r=${RANKOF[$h]}
  echo "-- $h rank$r"
  ssh -o BatchMode=yes $h "mkdir -p /home/$h/lane-dsv4flash-m1/rt/packs /home/$h/lane-dsv4flash-m1/rt/kv"
  rsync -a "$RT/bin" "$RT/lib" "$RT/config" "$h:/home/$h/lane-dsv4flash-m1/rt/"
  if ! ssh -o BatchMode=yes $h "test -s /home/$h/lane-dsv4flash-m1/rt/packs/dsv4_flash_stage.spstage"; then
    echo "   copying v4 rank$r pack (50G)"
    ssh -o BatchMode=yes $h "nc -l -p 4040 > /home/$h/lane-dsv4flash-m1/rt/packs/.pack.tmp && mv /home/$h/lane-dsv4flash-m1/rt/packs/.pack.tmp /home/$h/lane-dsv4flash-m1/rt/packs/dsv4_flash_stage.spstage" &
    sleep 1
    nc -q1 $h 4040 < "$BIS/packs/dsv4_flash_v4_tp4rank${r}.spstage"
    wait
  fi
done
[ -s "$RT/packs/dsv4_flash_stage.spstage" ] || cp "$BIS/packs/dsv4_flash_v4_tp4rank1.spstage" "$RT/packs/dsv4_flash_stage.spstage"

echo "== pack size verification (expected r0/r1 50996758576, r2/r3 50995710000)"
for h in spark4 spark5 spark6 spark7; do
  r=${RANKOF[$h]}
  s=$(ssh -o BatchMode=yes $h "stat -c '%s' /home/$h/lane-dsv4flash-m1/rt/packs/dsv4_flash_stage.spstage" 2>/dev/null || stat -c '%s' "$RT/packs/dsv4_flash_stage.spstage")
  echo "$h rank$r pack_bytes=$s"
done
echo "BUILD-TASK-DONE"
