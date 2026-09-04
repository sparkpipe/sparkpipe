#!/usr/bin/env bash
# K3 serving dispatch sm_121a compile gate (single-spark verification; no run).
# Compiles the dispatch TU against bind.cu's ABI and the module headers; the
# K3StageSlice symbol stays unresolved because this is a compile-only gate.
# Usage: sh tools/k3_dispatch_compile_gate.sh [NVCC]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVCC="${1:-nvcc}"
cd "$ROOT"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodules/k3_resident_decode_stage/include \
    -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c \
    modules/k3_resident_decode_stage/source/spark_k3_resident_decode_stage_cuda.cu \
    -o /tmp/k3_dispatch_gate.o
echo "k3 dispatch sm_121a compile gate PASS"
