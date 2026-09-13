#!/usr/bin/env bash
# Single-spark K3 decode step gate: builds the runner test against a real
# rank pack and runs it on the local spark. Compiles the runner, dispatch,
# driver TUs and the CUDA-free module sources into one binary.
# Usage: bash tools/k3_single_spark_step.sh <rank.pack> [NVCC]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PACK="${1:?usage: k3_single_spark_step.sh RANK_PACK [NVCC]}"
NVCC="${2:-nvcc}"
cd "$ROOT"
INC="-I. -Iinclude -Isrc -Imodules/k3_resident_decode_stage/include -Imodel-families/common/include -Imodel-families/k3/include"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a $INC -Xcompiler -fPIC \
    tests/test_k3_runner_step.cu \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu \
    inference/llms/kimi_k3/bind.cu \
    inference/llms/kimi_k3/unity.cu \
    modules/k3_resident_decode_stage/source/spark_k3_pack_load.c \
    modules/k3_resident_decode_stage/source/spark_k3_bind.c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_module.c \
    runtime/json.c runtime/filesystem.c src/spark_status.c \
    build/libsparkpipe_model_common.a \
    -Xcompiler -pthread -ldl -lcuda -lcudart -o /tmp/k3_single_step
/tmp/k3_single_step "$PACK"
