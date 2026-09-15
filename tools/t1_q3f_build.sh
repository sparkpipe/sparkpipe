#!/bin/sh
set -eu
: "${T1_STAGE:=/tmp/t1q3f_stage}"
: "${T1_RUNTIME:=/tmp/t1q3f}"
cd "$T1_STAGE/tree"
REPOSITORY_ROOT="$T1_STAGE/tree" MTP_LAYER_COUNT=0 make -C modules/qwen4_flash_resident_decode_stage archive > "$T1_STAGE/make_q4f.log" 2>&1
REPOSITORY_ROOT="$T1_STAGE/tree" EXPERT_CODEC=nvfp4 \
	MODEL_REVISION=84c6a6aa9497188e15a635ba793b0f95a79b1033 \
	CONTRACT_SHA256=$(sha256sum model_contracts/glm53_flash_authoritative.json | cut -d' ' -f1) \
	make -C modules/glm5_next_resident_decode_stage archive > "$T1_STAGE/make_g5n.log" 2>&1
make build/libsparkpipe_core.a build/libsparkpipe_runtime.a build/libhidden_transport_spark_host_rdma_verbs.so > "$T1_STAGE/make_rest.log" 2>&1
ARCHIVE=$(find build -name "libqwen4_flash_resident_decode_stage.a" | head -1)
G5N_ARCHIVE=$(find build -path "*glm5_next*" -name "*.a" | head -1)
[ -n "$ARCHIVE" ] || { echo "module archive missing" >&2; exit 1; }
[ -n "$G5N_ARCHIVE" ] || { echo "glm5_next mesh-kernel archive missing" >&2; exit 1; }
nvcc -std=c++17 -O3 -arch=sm_121a \
	-I. -Iinclude -Imodel-families/common/include -Imodel-families/qwen4_flash/include \
	-Imodules/qwen4_flash_resident_decode_stage/include \
	-Imodules/qwen4_flash_resident_decode_stage/source \
	-DSPARK_QWEN4_FLASH_MODULE_BUILD=1 \
	-DSPARK_QWEN4_FLASH_MODEL_MTP_LAYER_COUNT=0u \
	tools/t1_q3f_harness.c \
	"$ARCHIVE" \
	"$G5N_ARCHIVE" \
	build/libsparkpipe_runtime.a \
	build/libsparkpipe_core.a \
	-L/usr/local/cuda/lib64 -lcudart -lcuda -o "$T1_RUNTIME/t1_q3f_harness" 2> "$T1_STAGE/make_harness.log" || {
		tail -20 "$T1_STAGE/make_harness.log" >&2
		exit 1
	}
cp build/libhidden_transport_spark_host_rdma_verbs.so "$T1_RUNTIME/"
ls -la "$T1_RUNTIME/t1_q3f_harness" "$T1_RUNTIME/libhidden_transport_spark_host_rdma_verbs.so"
