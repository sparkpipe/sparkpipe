#!/bin/bash
# DSV4 Pro publish, staged for the 15-minute queue windows.
# usage: dsv4pro_publish_stage.sh <archive|validate|publish>
#   archive  - compile module objects + link unit (no GPU)
#   validate - run the one-time GPU validator by hand (acceptance gate)
#   publish  - module publish (receipt reuse) + driver + adapter
set -uo pipefail
cd $HOME/dsv4pro_checkout
mkdir -p /home/spark6/sparkdata/dsv4_pro.tp4pp4/packs /mnt/model-warm/packbuild/dsv4pro
LOG=/mnt/model-warm/packbuild/dsv4pro/publish_${1}_$(date +%s).log
exec > "$LOG" 2>&1
export PATH=/usr/local/cuda/bin:$PATH
PACK=$HOME/sparkdata/dsv4_pro.tp4pp4/packs/dsv4_pro.tp4_pp4.rank00.spstage
[[ -s "$PACK" ]] || { echo "MISSING-PACK $PACK"; exit 1; }
[[ -s "$PACK.sha256" ]] || { echo "MISSING-SIDECAR $PACK.sha256"; exit 1; }
STAGE="$1"
PRO_MAKE_FLAGS=(PRO_EXPERT_CODEC=mxfp4 PRO_KV_CODEC=bf16
  STAGE_PACK_PATH="$PACK"
  STAGE_COUNT=4 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=16
  MAX_ACTIVE_SEQUENCES=1024 MAX_SEQUENCE_POSITIONS=33024
  PIPELINE_SLOT_COUNT=13 PHYSICAL_PAGE_CAPACITY=1024
  LOGICAL_PAGE_CAPACITY=16384 MTP_LAYER_COUNT=0 CUDA_GRAPH_COUNT=0)

case "$STAGE" in
archive)
  make -C modules/dsv4_resident_decode_stage -f Makefile.pro archive \
    "${PRO_MAKE_FLAGS[@]}"
  rc=$?
  sha256sum build/modules/dsv4_pro_resident_decode_stage/libdsv4_resident_decode_stage.a
  echo "STAGE-$STAGE-RC=$rc"
  exit $rc
  ;;
validate)
  make -C modules/dsv4_resident_decode_stage -f Makefile.pro validate \
    "${PRO_MAKE_FLAGS[@]}"
  rc=$?
  echo "STAGE-$STAGE-RC=$rc"
  exit $rc
  ;;
publish)
  make build/sparkpipe_module_compile build/sparkpipe_model_residentd \
    build/sparkpipe_model_api build/sparkpipe_model_batch
  make -C modules/dsv4_resident_decode_stage -f Makefile.pro publish \
    "${PRO_MAKE_FLAGS[@]}" GPU_VALIDATOR=validation/dsv4_validator_wrapper.sh \
    || { echo "STAGE-$STAGE-FAIL"; exit 1; }
  RR=$HOME/sparkdata/dsv4_pro.tp4pp4
  build/sparkpipe_model_compile \
    --model examples/model_descriptions/dsv4_pro_resident_decode_stage_firmware.json \
    --library build/module_library --output "$RR" \
    --cc /usr/bin/cc --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib --cc-arg -lcuda \
    --cc-arg -lcudart --cc-arg -lstdc++ --cc-arg -lm --cc-arg -ldl \
    --cc-arg -pthread || { echo "COMPILE-FAIL"; exit 1; }
  AD=$(ls build/modules/dsv4_pro_resident_decode_stage/*/lib*adapter*.so 2>/dev/null | head -1)
  [[ -n "$AD" ]] || { echo "ADAPTER-NOT-FOUND"; exit 1; }
  cp "$AD" "$RR/lib/model_serving_adapter.so.new"
  cp build/sparkpipe_model_residentd build/sparkpipe_model_api \
    build/sparkpipe_model_batch "$RR/bin/"
  make build/weightd_expert_segments
  ./build/weightd_expert_segments "$PACK"
  sha256sum "$RR/lib/model_driver.so" "$RR/lib/model_serving_adapter.so.new" \
    | tee "$RR/driver_rebuild_receipt.txt"
  echo "STAGE-$STAGE-RC=0"
  exit 0
  ;;
*)
  echo "unknown stage: $STAGE"; exit 2 ;;
esac
