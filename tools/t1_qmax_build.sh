#!/bin/sh
set -eu
: "${T1_STAGE:=/tmp/t1qmax_stage}"
: "${T1_RUNTIME:=/tmp/t1qmax}"
cd "$T1_STAGE/tree"
REPOSITORY_ROOT="$T1_STAGE/tree" make -C modules/qwen38_max_resident_decode_stage archive > "$T1_STAGE/make_q38.log" 2>&1
REPOSITORY_ROOT="$T1_STAGE/tree" EXPERT_CODEC=nvfp4 \
	MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
	CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1) \
	make -C modules/glm5_next_resident_decode_stage archive > "$T1_STAGE/make_g5n.log" 2>&1
make build/libsparkpipe_core.a build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so > "$T1_STAGE/make_rest.log" 2>&1
nvcc -std=c++17 -O3 -arch=sm_121a \
	-I. -Iinclude -Imodel-families/common/include -Imodel-families/qwen38_max/include \
	-Imodules/qwen38_max_resident_decode_stage/include \
	-Imodules/qwen38_max_resident_decode_stage/source \
	-DSPARK_QWEN38_MAX_MODULE_BUILD=1 \
	-DSPARK_QWEN38_MAX_MODEL_MTP_LAYER_COUNT=0u \
	-DQWEN38_MODEL_REVISION="d2dc35658bcf77e66643428cb52e774cc3b5bd29" \
	tools/t1_qmax_harness.c \
	build/modules/qwen38_resident_decode_stage/libqwen38_resident_decode_stage.a \
	build/modules/glm5_next_resident_decode_stage/nvfp4/libglm5_next_resident_decode_stage_nvfp4.a \
	build/libsparkpipe_core.a \
	-L/usr/local/cuda/lib64 -lcudart -lcuda -o "$T1_RUNTIME/t1_qmax_harness"
cp build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so "$T1_RUNTIME/"
cp "$T1_RUNTIME/t1_qmax_harness" build/sparkpipe_weightd build/libhidden_transport_spark_host_rdma_verbs.so "$T1_STAGE/"
chmod +x "$T1_RUNTIME/t1_qmax_harness" "$T1_RUNTIME/sparkpipe_weightd"
ls -la "$T1_STAGE/t1_qmax_harness" "$T1_STAGE/libhidden_transport_spark_host_rdma_verbs.so" "$T1_STAGE/sparkpipe_weightd"
