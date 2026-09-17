#!/bin/sh
# The head's chunked top-k has no caller in this tree yet: the serving path
# is greedy argmax and the sampler arrives with the driver wave. Nothing
# instantiates LmHeadTopkCandidateKernel or LmHeadTopkCommitKernel for a
# device target, and an uninstantiated template can stop compiling without
# any gate noticing. This instantiates both at the decode sizes.
#
# tests/test_head_host.py checks the selection matches a float32 reference;
# this checks the kernels still build for a GPU.
CUDA=${CUDA_HOME:-/opt/cuda}
ARCH="-gencode arch=compute_121a,code=sm_121a"
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA; run tools/get_cuda.sh"; exit 2; }
cat > /tmp/lm_head_topk.cu <<'EOF'
#include "inference/kernels/head.cuh"
template __global__ void LmHeadTopkCandidateKernel<256u, 1024u, 8u>(
	const uint16_t *, const uint16_t *, const uint32_t *, float *, uint32_t *, uint32_t, uint32_t, uint32_t);
template __global__ void LmHeadTopkCommitKernel<256u, 8u>(
	const float *, const uint32_t *, uint32_t, uint32_t *, float *, uint32_t);
EOF
cd "$(dirname "$0")/.." || exit 1
"$CUDA/bin/nvcc" -std=c++17 $ARCH -O1 -I. -c /tmp/lm_head_topk.cu \
	-o /tmp/lm_head_topk.o 2> /tmp/lm_head_topk.log || {
	grep -E "error" /tmp/lm_head_topk.log | head -5
	exit 1
}
