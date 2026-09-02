
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"

#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/head.cuh"

#define THREADS 1u
#define TILE 8u
#define TOPK 3u
#define ROWS 2u
#define HIDDEN 16u
#define VOCAB 23u
#define TILES 3u
#define RESTRICTED 9u

static uint32_t seed = 13579u;
static float NextRandom(void)
{
	seed = (seed * 1664525u) + 1013904223u;
	return (float)((seed >> 8) & 0xffffu) / 32768.0f - 1.0f;
}

static void Emit(const char *tag, const uint16_t *values, uint32_t count)
{
	uint32_t index;
	for (index = 0u; index < count; ++index)
		printf("%s %.9g\n", tag, (double)LmBf16ToFloat(values[index]));
}

int main(void)
{
	static uint16_t normed[ROWS * HIDDEN];
	static uint16_t weight[VOCAB * HIDDEN];
	static float candidate_score[ROWS * TILES * TOPK];
	static uint32_t candidate_token[ROWS * TILES * TOPK];
	static uint32_t token_out[ROWS * TOPK];
	static float score_out[ROWS * TOPK];
	static const uint32_t restricted[RESTRICTED] =
		{ 20u, 3u, 14u, 1u, 22u, 7u, 9u, 0u, 17u };
	uint32_t row, index;
	int32_t status;

	for (index = 0u; index < ROWS * HIDDEN; ++index)
		normed[index] = LmFloatToBf16(NextRandom());
	for (index = 0u; index < VOCAB * HIDDEN; ++index)
		weight[index] = LmFloatToBf16(NextRandom() * 0.5f);
	for (index = 0u; index < HIDDEN; ++index)
	{
		weight[4u * HIDDEN + index] = weight[2u * HIDDEN + index];
		weight[13u * HIDDEN + index] = weight[2u * HIDDEN + index];
	}

	printf("VOCAB %u HIDDEN %u TOPK %u ROWS %u\n", VOCAB, HIDDEN, TOPK, ROWS);
	Emit("normed", normed, ROWS * HIDDEN);
	Emit("weight", weight, VOCAB * HIDDEN);

	if ( LmHeadTopkCandidatePairs<TILE,TOPK>(ROWS,VOCAB) != ROWS * TILES * TOPK )
	{
		printf("FAIL candidate pair count\n");
		return 1;
	}
	status = LmHeadTopk<THREADS,TILE,TOPK>(
		normed,weight,0,candidate_score,candidate_token,token_out,score_out,
		ROWS,HIDDEN,VOCAB,0);
	if ( status != LM_LAUNCH_OK )
	{
		printf("FAIL full topk status %d\n", (int)status);
		return 1;
	}
	for (row = 0u; row < ROWS; ++row)
		for (index = 0u; index < TOPK; ++index)
			printf("full %u %u %.9g\n",
				row,
				token_out[(row * TOPK) + index],
				(double)score_out[(row * TOPK) + index]);

	status = LmHeadTopk<THREADS,TILE,TOPK>(
		normed,weight,restricted,candidate_score,candidate_token,
		token_out,score_out,ROWS,HIDDEN,RESTRICTED,0);
	if ( status != LM_LAUNCH_OK )
	{
		printf("FAIL restricted topk status %d\n", (int)status);
		return 1;
	}
	for (row = 0u; row < ROWS; ++row)
		for (index = 0u; index < TOPK; ++index)
			printf("restricted %u %u %.9g\n",
				row,
				token_out[(row * TOPK) + index],
				(double)score_out[(row * TOPK) + index]);
	return 0;
}
