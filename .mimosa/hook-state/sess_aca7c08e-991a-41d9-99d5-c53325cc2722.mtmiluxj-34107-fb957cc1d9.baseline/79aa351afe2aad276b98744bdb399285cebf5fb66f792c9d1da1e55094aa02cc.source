#!/usr/bin/env bash
# K3 serving stage runner sm_121a compile gate (compile-only; K3StageSlice
# resolves at the module .so link).
# Usage: bash tools/k3_runner_compile_gate.sh [NVCC]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVCC="${1:-nvcc}"
cd "$ROOT"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodules/k3_resident_decode_stage/include \
    -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_runner.cu \
    -o /tmp/k3_runner_gate.o
echo "k3 runner sm_121a compile gate PASS"
