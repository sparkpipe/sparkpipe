#!/bin/bash
# dsv4flash derisk: publish b1 module (GPU validator, val3 slice) + compile
# the TP16 driver on ONE free node, so the 16-node cell only does launch+bench.
set -uo pipefail
SRC=/home/spark5/lane-dsv4flash-m1/src2
BASE=/home/spark5/lane-dsv4flash-m1
VALSLICE=/home/spark5/lane-dsv4bisect/packs/dsv4_flash_v4_val3.spstage

echo "== publish b1 module (GPU validator)"
cd $SRC
make -C modules/dsv4_resident_decode_stage publish_variants MODULE_BATCH_VARIANT_BUCKETS=1 CUDA_ARCH=sm_121a \
  STAGE_PACK_PATH=$VALSLICE STAGE_COUNT=13 STAGE_INDEX=1 STAGE_FIRST_LAYER=3 STAGE_LAYER_COUNT=3 \
  MAX_ACTIVE_SEQUENCES=1 PIPELINE_SLOT_COUNT=1 2>&1 | grep -E "validation|module=|error|FAIL" | tail -4
rc=${PIPESTATUS[0]}
[ $rc -eq 0 ] || { echo PUBLISH-FAIL rc=$rc; exit 1; }

echo "== compile driver"
if [ -d $BASE/driver-tp16 ]; then rm -r $BASE/driver-tp16; fi
build/sparkpipe_model_compile --model examples/model_descriptions/dsv4_resident_decode_stage_firmware_b1.json \
  --stage dsv4_resident_decode_stage --library build/module_library --output $BASE/driver-tp16 --include include \
  --cc-arg -L/usr/local/cuda/lib64 --cc-arg -lcuda --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -ldl --cc-arg -lm --cc-arg -pthread \
  2>&1 | tail -1
[ -f $BASE/driver-tp16/model_driver.so ] || { echo COMPILE-FAIL; exit 1; }
sha256sum $BASE/driver-tp16/model_driver.so
echo DERISK-DONE
