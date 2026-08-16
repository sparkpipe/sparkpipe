#!/usr/bin/env bash
# qwen36 TP4 whole-stack build: module archive + TP4 GPU validation + driver
set -euo pipefail
ROOT=/home/sparkd/sparkpipe-qwen
PACKS=/home/sparkd/sparkdata/qwen38.bf16.tp4/packs
OUT=/home/sparkd/sparkdata/qwen38.bf16.tp4/build
export NVCC=/usr/local/cuda/bin/nvcc
cd "$ROOT"
mkdir -p "$OUT"
echo "=== module archive ==="
make -C modules/qwen36_resident_decode_stage -j20 archive \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a
echo "=== main binaries ==="
make -j20 \
    build/sparkpipe_module_publish \
    build/sparkpipe_model_compile \
    build/sparkpipe_model_residentd \
    build/sparkpipe_model_batch \
    build/libqwen36_serving_adapter.so \
    build/libhidden_transport_spark_host_rdma_verbs.so
echo "=== module publish (GPU validator, TP4 rank0 standalone) ==="
make -C modules/qwen36_resident_decode_stage publish \
    NVCC=/usr/local/cuda/bin/nvcc CUDA_ARCH=sm_121a \
    STAGE_PACK_PATH="$PACKS/tp4-rank0.qwen36sp" \
    STAGE_COUNT=1 STAGE_INDEX=0 STAGE_FIRST_LAYER=0 STAGE_LAYER_COUNT=64 \
    TP_DEGREE=4 TP_RANK=0 TP_STANDALONE=1 \
    MTP_LAYER_COUNT=0 GDN_SNAPSHOT_SLOT_COUNT=0 \
    MAX_ACTIVE_SEQUENCES=8 KV_BLOCK_COUNT=8 ALLOW_UNQUALIFIED_EXECUTION=1
echo "=== model driver ==="
build/sparkpipe_model_compile \
    --model examples/model_descriptions/qwen36_resident_decode_stage_firmware.json \
    --stage qwen36_resident_decode_stage \
    --library build/module_library \
    --output "$OUT/driver" \
    --cc /usr/bin/cc \
    --include include \
    --cc-arg -L/usr/local/cuda/targets/sbsa-linux/lib \
    --cc-arg -lcuda \
    --cc-arg -lcudart \
    --cc-arg -lstdc++ \
    --cc-arg -lm \
    --cc-arg -ldl \
    --cc-arg -pthread
cp build/sparkpipe_model_residentd "$OUT/"
cp build/sparkpipe_model_batch "$OUT/"
cp build/libqwen36_serving_adapter.so "$OUT/model_serving_adapter.so"
cp build/libhidden_transport_spark_host_rdma_verbs.so "$OUT/hidden_transport.so"
rm -rf "$OUT/link_units"; cp -a "$OUT/driver"/. "$OUT/"
python3 tools/generate_model_resident_deployment.py \
    --specification examples/deployments/qwen36_tp4_host_rdma.spec.json \
    --output "$OUT/model_resident.json"
cp examples/deployments/qwen36_tp4_rank0.json "$OUT/qwen36_tp4_rank0.json"
cp examples/deployments/qwen36_tp4_rank1.json "$OUT/qwen36_tp4_rank1.json"
cp examples/deployments/qwen36_tp4_rank2.json "$OUT/qwen36_tp4_rank2.json"
cp examples/deployments/qwen36_tp4_rank3.json "$OUT/qwen36_tp4_rank3.json"
echo "=== build done ==="
sha256sum "$OUT/model_driver.so"
