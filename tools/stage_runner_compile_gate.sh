#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NVCC="${1:-nvcc}"
cd "$ROOT"
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodules/k3_resident_decode_stage/include \
    -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c inference/runner/stage_runner.cu -o /tmp/stage_runner_gate.o
"$NVCC" -std=c++17 -O3 --expt-relaxed-constexpr -lineinfo \
    -gencode arch=compute_121a,code=sm_121a \
    -I. -Iinclude -Imodules/k3_resident_decode_stage/include \
    -Imodel-families/common/include -Imodel-families/k3/include \
    -Xcompiler -Wall,-Wextra,-fPIC -c modules/k3_resident_decode_stage/source/spark_k3_stage_model.cu -o /tmp/k3_stage_model_gate.o
echo "stage runner and k3 stage model sm_121a compile gate PASS"
