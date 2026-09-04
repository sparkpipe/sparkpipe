#!/bin/sh
# LmReplayFoldKernel has no model driving it yet - speculation is not wired - so
# nothing instantiates it for a device and it could stop compiling silently.
# tests/test_kda_host.py checks that it reproduces the decode state; this checks
# it still builds for a GPU at K3's real head shape. The uint16_t (bf16-state)
# instantiations of the fold and the delta rule are here for the same reason:
# no driver may select them until the kda_state_bf16 flag lifts, so without
# these lines the admission-time option could rot before it is admitted.
CUDA=${CUDA_HOME:-/opt/cuda}
[ -x "$CUDA/bin/nvcc" ] || { echo "no nvcc at $CUDA; run tools/get_cuda.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
cat > /tmp/lm_replay_fold.cu <<'EOF'
#include "inference/kernels/linear_attn.cuh"
template __global__ void LmReplayFoldKernel<256u, 128u, 128u>(
	uint8_t *, uint32_t, const uint32_t *, const LmReplayStep *, const uint32_t *,
	uint32_t, uint32_t, uint32_t);
template __global__ void LmReplayFoldKernel<256u, 128u, 128u, uint16_t>(
	uint8_t *, uint32_t, const uint32_t *, const LmReplayStep *, const uint32_t *,
	uint32_t, uint32_t, uint32_t);
template __global__ void LmDeltaRuleKernel<256u, 128u, 128u, uint16_t>(
	uint8_t *, uint32_t, const uint32_t *, const uint32_t *, const uint32_t *,
	const uint16_t *, const uint16_t *, const uint16_t *, const float *,
	const float *, uint16_t *, uint32_t, uint32_t, uint32_t, uint32_t);
EOF
"$CUDA/bin/nvcc" -std=c++17 -gencode arch=compute_121a,code=sm_121a -O1 -I. \
	-c /tmp/lm_replay_fold.cu -o /tmp/lm_replay_fold.o 2> /tmp/lm_replay_fold.log || {
	grep -E "error" /tmp/lm_replay_fold.log | head -5; exit 1; }
