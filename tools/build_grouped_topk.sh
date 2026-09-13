#!/bin/sh
# The grouped expert-selection path has no model in this tree: Kimi K3 sets
# num_expert_group to 1 and nothing else uses groups. So nothing instantiates
# LmTopkSmallKernel with GROUPS > 1, and it could stop compiling without any
# gate noticing. This instantiates it.
#
# tests/test_expert_grouping.py checks that the arithmetic matches
# KimiMoEGate; this checks that the arithmetic still builds for a GPU.
CUDA=${CUDA_HOME:-/opt/cuda}
ARCH="-gencode arch=compute_121a,code=sm_121a"
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA; run tools/get_cuda.sh"; exit 2; }
cat > /tmp/lm_grouped_topk.cu <<'EOF'
#include "inference/kernels/topk.cuh"
template __global__ void LmTopkSmallKernel<256u, 6u, true, 8u, 3u, LM_TOPK_SCORE_IDENTITY>(
	const float *, uint32_t, uint32_t *, float *, const float *, const uint16_t *, float);
EOF
cd "$(dirname "$0")/.." || exit 1
"$CUDA/bin/nvcc" -std=c++17 $ARCH -O1 -I. -c /tmp/lm_grouped_topk.cu \
	-o /tmp/lm_grouped_topk.o 2> /tmp/lm_grouped_topk.log || {
	grep -E "error" /tmp/lm_grouped_topk.log | head -5
	exit 1
}
