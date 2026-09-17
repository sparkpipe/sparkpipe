#!/usr/bin/env bash
# K3 driver family sm_121a compile gate (single-spark verification; no run).
# Usage: sh tools/k3_sm121a_compile_gate.sh [NVCC]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVCC="${1:-nvcc}"
cd "$ROOT"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c inference/llms/kimi_k3/bind.cu -o /tmp/k3_bind_gate.o
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c inference/llms/kimi_k3/unity.cu -o /tmp/k3_unity_gate.o
echo "k3 sm_121a compile gate PASS"
