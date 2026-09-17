#!/bin/sh
# Build the fail-frame probe on an sm_121a node (spark5). One binary links
# the three resident-decode-stage module .cu files whose shipping launchers
# the probe exercises, each with its own module build contract (the same
# flags modules/resident_decode_stage_rules.mk and the module Makefiles
# use), plus the probe main.
set -e
ROOT="${ROOT:-$HOME/sparkpipe}"
BUILD=/tmp/kernel_crew_probe
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
ARCH_FLAGS="-gencode arch=compute_121a,code=sm_121a"
COMMON="-std=c++17 -O3 --expt-relaxed-constexpr -lineinfo"

mkdir -p "$BUILD"

INCLUDES="-I$ROOT/include -I$ROOT/model-families/common/include"

# dsv4 module object (its own contract)
$NVCC $COMMON $ARCH_FLAGS $INCLUDES \
  -I"$ROOT/model-families/dsv4/include" \
  -I"$ROOT/modules/dsv4_resident_decode_stage/include" \
  -I"$ROOT/modules/dsv4_resident_decode_stage/source" \
  -DSPARK_DSV4_MODULE_BUILD=1 -DSPARK_BATCH_BUCKET=1024u \
  -include "$ROOT/model-families/dsv4/include/sparkpipe/spark_dsv4_model.h" \
  -c "$ROOT/modules/dsv4_resident_decode_stage/source/spark_dsv4_resident_decode_stage_cuda.cu" \
  -o "$BUILD/dsv4.o"

# qwen4_flash module object
$NVCC $COMMON $ARCH_FLAGS $INCLUDES \
  -I"$ROOT/model-families/qwen4_flash/include" \
  -I"$ROOT/modules/qwen4_flash_resident_decode_stage/include" \
  -I"$ROOT/modules/qwen4_flash_resident_decode_stage/source" \
  -include "$ROOT/model-families/qwen4_flash/include/sparkpipe/spark_qwen4_flash_model.h" \
  -c "$ROOT/modules/qwen4_flash_resident_decode_stage/source/spark_qwen4_flash_resident_decode_stage_cuda.cu" \
  -o "$BUILD/qwen4.o"

# qwen38_27b module object
$NVCC $COMMON $ARCH_FLAGS $INCLUDES \
  -I"$ROOT/model-families/qwen38_27b/include" \
  -I"$ROOT/modules/qwen38_27b_resident_decode_stage/include" \
  -I"$ROOT/modules/qwen38_27b_resident_decode_stage/source" \
  -DSPARK_QWEN38_27B_MODULE_BUILD=1 \
  -include "$ROOT/model-families/qwen38_27b/include/sparkpipe/spark_qwen38_27b_model.h" \
  -c "$ROOT/modules/qwen38_27b_resident_decode_stage/source/spark_qwen38_27b_resident_decode_stage_cuda.cu" \
  -o "$BUILD/qwen38.o"

# the probe main
$NVCC $COMMON $ARCH_FLAGS $INCLUDES \
  -I"$ROOT/model-families/dsv4/include" \
  -I"$ROOT/modules/dsv4_resident_decode_stage/include" \
  -I"$ROOT/modules/qwen4_flash_resident_decode_stage/include" \
  -I"$ROOT/modules/qwen38_27b_resident_decode_stage/include" \
  "$ROOT/tools/kernel_crew/spark_frame_error_probe.cu" \
  "$BUILD/dsv4.o" "$BUILD/qwen4.o" "$BUILD/qwen38.o" \
  -o "$BUILD/spark_frame_error_probe"

echo "built $BUILD/spark_frame_error_probe"
"$BUILD/spark_frame_error_probe"
