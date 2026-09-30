#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <vector>

LmHostDim3 blockIdx, threadIdx, blockDim, gridDim;

uint32_t lm_topk_shared[LM_HOST_SHARED_BYTES / sizeof(uint32_t)];
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float state_s[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_fused_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
float lm_quant_shared[LM_HOST_SHARED_BYTES / sizeof(float)];

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/tile.cuh"
#include "inference/kernels/mma.cuh"
#undef LM_WARP_LANES
#define LM_WARP_LANES LM_HOST_WARP_LANES

#include "runtime/gemm.cuh"
std::vector<LmRecordedGemm> lm_recorded_gemms;

#define __CUDACC__ 1
#include "inference/kernels/kv.cuh"
#undef __CUDACC__

struct LmHostRecorderFormat
{
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 15));
	}
};
#include "inference/llms/kimi_k3/layer.cuh"

#define ROWS 3u
#define SLICE 16u

static uint16_t hidden[ROWS * K3_HIDDEN], normed[ROWS * K3_HIDDEN];
static uint16_t norm_weight[K3_HIDDEN], head_weight[SLICE * K3_HIDDEN];
static float candidate_score[ROWS], output_score[ROWS];
static uint32_t candidate_token[ROWS], output_token[ROWS];

int main(void)
{
	static K3LayerBuffers b;
	static const uint32_t winner[ROWS] = { 3u, 11u, 0u };
	static const uint32_t offsets[] = { 0u, 10240u, 15u * 10240u };
	uint32_t row, index, failures = 0u;
	memset(&b, 0, sizeof(b));
	for (index = 0u; index < K3_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (row = 0u; row < ROWS; ++row)
		hidden[row * K3_HIDDEN + row] = LmFloatToBf16(1.0f);
	for (index = 0u; index < SLICE; ++index)
		for (row = 0u; row < ROWS; ++row)
			head_weight[index * K3_HIDDEN + row] =
				LmFloatToBf16(index == winner[row] ? 2.0f : 0.5f);
	b.hidden_bf16 = hidden; b.normed_bf16 = normed;
	b.head_candidate_score = candidate_score; b.head_candidate_token = candidate_token;
	b.output_token = output_token; b.output_score = output_score;
	for (index = 0u; index < sizeof(offsets) / sizeof(offsets[0]); ++index)
	{
		int32_t status = K3HeadRankSlice(&b, norm_weight, head_weight,
			offsets[index], SLICE, ROWS, 0);
		if ( status != LM_LAUNCH_OK )
		{
			printf("FAIL rank offset %u: launch %d\n", offsets[index], status);
			failures++;
			continue;
		}
		for (row = 0u; row < ROWS; ++row)
		{
			printf("rank offset %u row %u token %u\n", offsets[index], row, output_token[row]);
			if ( output_token[row] != offsets[index] + winner[row] )
			{
				printf("FAIL rank offset %u row %u: token %u, want %u\n", offsets[index], row,
					output_token[row], offsets[index] + winner[row]);
				failures++;
			}
		}
	}
	if ( failures != 0u )
		return 1;
	printf("every row of a vocabulary-sliced head returns the global token id\n");
	return 0;
}
