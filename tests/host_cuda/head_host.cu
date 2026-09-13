// Run the head's chunked top-k on a CPU.
//
// The argmax pair already executes in every layer harness; the top-k path is
// new and its failure modes are exactly the kind that compile: a suppression
// that forgets a taken token emits it twice, a ragged final tile reads past
// the vocabulary, a tie broken by arrival order is not the tie the argmax
// commit breaks by token id. This runs the kernels at THREADS == 1 - the
// sequential schedule a correct kernel must also be valid under - and the
// python driver scores them against a float32 reference, ties included.
//
// head.cuh is included unmodified.

#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

#include "inference/kernels/dtype.cuh"

// One lane per warp, as in every harness: the block reductions must degenerate
// correctly at one thread, which 32 lanes per warp does not.
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "inference/kernels/head.cuh"

#define THREADS 1u
#define TILE 8u
#define TOPK 3u
#define ROWS 2u
#define HIDDEN 16u
// 23 tokens over an 8-wide tile: the last tile is ragged, which is where a
// boundary slip in the candidate kernel lives.
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
	// A restricted set that is NOT sorted and does not cover the vocabulary's
	// best: the selection must follow the ids, not the row order.
	static const uint32_t restricted[RESTRICTED] =
		{ 20u, 3u, 14u, 1u, 22u, 7u, 9u, 0u, 17u };
	uint32_t row, index;
	int32_t status;

	for (index = 0u; index < ROWS * HIDDEN; ++index)
		normed[index] = LmFloatToBf16(NextRandom());
	for (index = 0u; index < VOCAB * HIDDEN; ++index)
		weight[index] = LmFloatToBf16(NextRandom() * 0.5f);
	// Two engineered ties AT THE TOP, in different tiles: tokens 4 and 13
	// share token 2's weight row, so the shortlist starts with three equal
	// scores and the order must be 2, 4, 13 - the lower token id, not
	// whichever tile emitted first.
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
