
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/norm.cuh"
#include "inference/kernels/topk.cuh"

#define EXPERTS 32u
#define TOP_K 6u
#define ROWS 8u
#define THREADS 1u

static float NextRandom(uint32_t *state)
{
	*state = (*state * 1664525u) + 1013904223u;
	return (float)((*state >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

int main(void)
{
	static uint16_t logits[ROWS * EXPERTS];
	static float scores[ROWS * EXPERTS];
	static float bias[EXPERTS];
	static uint32_t chosen[ROWS * TOP_K];
	static float weights[ROWS * TOP_K];
	uint32_t seed = 987654321u, row, expert;

	for (expert = 0u; expert < EXPERTS; ++expert)
		bias[expert] = NextRandom(&seed) * 0.4f;
	for (row = 0u; row < ROWS; ++row)
		for (expert = 0u; expert < EXPERTS; ++expert)
			logits[(row * EXPERTS) + expert] = LmFloatToBf16(NextRandom(&seed) * 3.0f);

	printf("EXPERTS %u TOP_K %u ROWS %u\n", EXPERTS, TOP_K, ROWS);
	for (expert = 0u; expert < EXPERTS; ++expert)
		printf("bias %.9g\n", (double)bias[expert]);
	for (row = 0u; row < ROWS; ++row)
		for (expert = 0u; expert < EXPERTS; ++expert)
			printf("logit %.9g\n",
				(double)LmBf16ToFloat(logits[(row * EXPERTS) + expert]));

	(void)scores;
	LM_HOST_LAUNCH(dim3(ROWS),
		(LmTopkSmallKernel<THREADS, TOP_K, true, 1u, 1u, LM_TOPK_SCORE_SIGMOID>(
			0, EXPERTS, chosen, weights, bias, logits, 1.0f)));

	for (row = 0u; row < ROWS; ++row)
		for (expert = 0u; expert < TOP_K; ++expert)
			printf("pick %u\nweight %.9g\n",
				chosen[(row * TOP_K) + expert],
				(double)weights[(row * TOP_K) + expert]);
	return 0;
}
