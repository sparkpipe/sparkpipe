
#include "tests/host_cuda/lm_host_cuda.cuh"

#include <stdio.h>
#include <string.h>
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

struct LmHostRecorderFormat
{
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr float kMax = 6.0f;
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		float clamped = value > kMax ? kMax : (value < -kMax ? -kMax : value);
		return((uint8_t)(((int)clamped) & 15));
	}
};

#define MIMO25_LAYER_THREADS 1u
#include "inference/llms/mimo_2_5/layer.cuh"

#define ROWS 2u

static uint16_t hidden[ROWS * MIMO25_HIDDEN];
static uint16_t residual[ROWS * MIMO25_HIDDEN];
static uint16_t normed[ROWS * MIMO25_HIDDEN];
static uint16_t fused_qkv[ROWS * MIMO25_SWA_QKV_DIM];
static uint16_t query[ROWS * MIMO25_Q_DIM];
static uint16_t key[ROWS * MIMO25_SWA_KV_HEADS * MIMO25_HEAD_DIM];
static uint16_t value[ROWS * MIMO25_SWA_KV_HEADS * MIMO25_VALUE_DIM];
static uint16_t attention_out[ROWS * MIMO25_Q_DIM];
static uint16_t gate_up[ROWS * MIMO25_DENSE_INTERMEDIATE * 2u];
static uint16_t intermediate[ROWS * MIMO25_DENSE_INTERMEDIATE];
static uint8_t packed_activation[ROWS * MIMO25_DENSE_INTERMEDIATE];
static uint8_t packed_scale[ROWS * (MIMO25_DENSE_INTERMEDIATE / 32u)];
static uint16_t norm_weight[MIMO25_HIDDEN];
static uint32_t sequence_of_row[ROWS];
static uint32_t context_length[ROWS];
static uint32_t positions[ROWS];
static uint32_t window_positions[ROWS * MIMO25_SLIDING_WINDOW];
static uint32_t dense_offsets[ROWS + 1u];
static uint32_t dense_tiles[2];
static uint8_t full_pool[2u * Mimo25FullKv::kPageBytes];
static uint8_t swa_pool[2u * Mimo25SwaKv::kPageBytes];
static uint32_t page_table[ROWS];
static uint32_t swa_page_table[ROWS * 2u];
static LmKvAccessError kv_access_error;

int main(void)
{
	static Mimo25LayerBuffers b;
	uint32_t index;
	float expected,maxdiff;
	memset(&b, 0, sizeof(b));
	LmKvAccessErrorReset(&kv_access_error);
	for (index = 0u; index < MIMO25_HIDDEN; ++index)
		norm_weight[index] = LmFloatToBf16(1.0f);
	for (index = 0u; index < ROWS * MIMO25_HIDDEN; ++index)
		hidden[index] = LmFloatToBf16(0.01f * (float)(index % 17u));
	memset(full_pool, 0xff, sizeof(full_pool));
	memset(swa_pool, 0xff, sizeof(swa_pool));
	sequence_of_row[0] = 0u; sequence_of_row[1] = 1u;
	page_table[0] = 1u; page_table[1] = 0u;
	dense_offsets[0] = 0u; dense_offsets[1] = ROWS; dense_offsets[2] = ROWS * 2u;

	b.hidden_bf16 = hidden; b.residual_bf16 = residual;
	b.normed_bf16 = normed; b.fused_qkv_bf16 = fused_qkv;
	b.query_bf16 = query; b.key_bf16 = key; b.value_bf16 = value;
	b.attention_out_bf16 = attention_out;
	b.gate_up_bf16 = gate_up; b.intermediate_bf16 = intermediate;
	b.packed_activation = packed_activation; b.packed_scale = packed_scale;
	b.attn_norm_weight = norm_weight; b.mlp_norm_weight = norm_weight;
	b.sequence_of_row = sequence_of_row;
	b.context_length = context_length; b.positions = positions;
	b.window_positions = window_positions;
	b.dense_row_offset = dense_offsets; b.dense_tile_prefix = dense_tiles;

	expected = LmBf16ToFloat(LmFloatToBf16(
		LmBf16ToFloat(LmFloatToBf16(0.125f)) * MIMO25_VALUE_SCALE));

	positions[0] = 0u; positions[1] = 0u;
	context_length[0] = 1u; context_length[1] = 1u;
	b.cache.pool = full_pool; b.cache.page_table = page_table;
	b.cache.page_table_stride = 1u; b.cache.sequence_count = ROWS;
	b.cache.pool_page_count = 2u; b.cache.access_error = &kv_access_error;
	lm_recorded_gemms.clear();
	if ( Mimo25LayerAttention<LmHostRecorderFormat,Mimo25FullKv,MIMO25_FULL_KV_HEADS,MIMO25_FULL_QKV_DIM>(
		&b,ROWS,1u,0u,MIMO25_FULL_ROPE_THETA,1u,0) != LM_LAUNCH_OK )
	{
		printf("full_layer_status FAIL\n");
		return(1);
	}
	printf("full_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("full_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	{
		const uint16_t *slot = (const uint16_t *)(full_pool
			+ ((uint64_t)page_table[0] * Mimo25FullKv::kPageBytes));
		uint32_t kv = MIMO25_FULL_KV_HEADS * (MIMO25_HEAD_DIM + MIMO25_VALUE_DIM);
		maxdiff = 0.0f;
		for (index = 0u; index < kv; ++index)
		{
			float want = index < MIMO25_FULL_KV_HEADS * MIMO25_HEAD_DIM
				? LmBf16ToFloat(LmFloatToBf16(0.125f)) : expected;
			float diff = LmBf16ToFloat(slot[index]) - want;
			if ( diff < 0.0f ) diff = -diff;
			if ( diff > maxdiff ) maxdiff = diff;
		}
		printf("full_slot_maxdiff %.9g\n", (double)maxdiff);
	}
	{
		static uint16_t direct_out[ROWS * MIMO25_O_INPUT_DIM];
		LM_HOST_LAUNCH(dim3(ROWS,MIMO25_ATTN_HEADS),
			(LmGqaAttentionDecodeKernel<Mimo25FullKv,1u,MIMO25_FULL_KV_HEADS,MIMO25_HEAD_DIM,MIMO25_VALUE_DIM>(
				query, b.cache, sequence_of_row, context_length,
				0, 0u, MIMO25_ATTN_HEADS, rsqrtf((float)MIMO25_HEAD_DIM), direct_out, 0)));
		maxdiff = 0.0f;
		for (index = 0u; index < ROWS * MIMO25_O_INPUT_DIM; ++index)
		{
			float diff = LmBf16ToFloat(direct_out[index]) - expected;
			if ( diff < 0.0f ) diff = -diff;
			if ( diff > maxdiff ) maxdiff = diff;
		}
		printf("full_attn_maxdiff %.9g\n", (double)maxdiff);
	}

	positions[0] = 1u; positions[1] = 1u;
	context_length[0] = 2u; context_length[1] = 2u;
	for (index = 0u; index < ROWS * MIMO25_SLIDING_WINDOW; ++index)
		window_positions[index] = (index % MIMO25_SLIDING_WINDOW) == 0u
			? 1u : MIMO25_KV_PAGE_SLOTS;
	swa_page_table[0] = 1u; swa_page_table[1] = LM_KV_PAGE_UNMAPPED;
	swa_page_table[2] = 0u; swa_page_table[3] = LM_KV_PAGE_UNMAPPED;
	b.cache.pool = swa_pool; b.cache.page_table = swa_page_table;
	b.cache.page_table_stride = 2u; b.cache.sequence_count = ROWS;
	b.cache.pool_page_count = 2u; b.cache.access_error = &kv_access_error;
	lm_recorded_gemms.clear();
	if ( Mimo25LayerAttention<LmHostRecorderFormat,Mimo25SwaKv,MIMO25_SWA_KV_HEADS,MIMO25_SWA_QKV_DIM>(
		&b,ROWS,2u,MIMO25_SLIDING_WINDOW,MIMO25_SWA_ROPE_THETA,1u,0) != LM_LAUNCH_OK )
	{
		printf("swa_layer_status FAIL\n");
		return(1);
	}
	printf("swa_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("swa_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	{
		const uint16_t *slot = (const uint16_t *)(swa_pool
			+ ((uint64_t)swa_page_table[0] * Mimo25SwaKv::kPageBytes)
			+ Mimo25SwaKv::kSlotBytes);
		uint32_t kv = MIMO25_SWA_KV_HEADS * (MIMO25_HEAD_DIM + MIMO25_VALUE_DIM);
		uint32_t kwidth = MIMO25_SWA_KV_HEADS * MIMO25_HEAD_DIM;
		uint32_t poisoned = 0u;
		maxdiff = 0.0f;
		for (index = 0u; index < kv; ++index)
		{
			float got = LmBf16ToFloat(slot[index]);
			if ( got != got )
			{
				++poisoned;
				continue;
			}
			if ( index >= kwidth )
			{
				float diff = got - expected;
				if ( diff < 0.0f ) diff = -diff;
				if ( diff > maxdiff ) maxdiff = diff;
			}
		}
		printf("swa_slot_maxdiff %.9g\nswa_slot_poisoned %u\n", (double)maxdiff, poisoned);
	}
	{
		static uint16_t direct_out[ROWS * MIMO25_O_INPUT_DIM];
		LM_HOST_LAUNCH(dim3(ROWS,MIMO25_ATTN_HEADS),
			(LmGqaAttentionDecodeKernel<Mimo25SwaKv,1u,MIMO25_SWA_KV_HEADS,MIMO25_HEAD_DIM,MIMO25_VALUE_DIM>(
				query, b.cache, sequence_of_row, context_length,
				window_positions, MIMO25_SLIDING_WINDOW, MIMO25_ATTN_HEADS, rsqrtf((float)MIMO25_HEAD_DIM), direct_out, 0)));
		maxdiff = 0.0f;
		for (index = 0u; index < ROWS * MIMO25_O_INPUT_DIM; ++index)
		{
			float diff = LmBf16ToFloat(direct_out[index]) - expected;
			if ( diff < 0.0f ) diff = -diff;
			if ( diff > maxdiff ) maxdiff = diff;
		}
		printf("swa_attn_maxdiff %.9g\n", (double)maxdiff);
	}

	lm_recorded_gemms.clear();
	if ( Mimo25LayerDenseMlp<LmHostRecorderFormat>(&b,ROWS,1u,0) != LM_LAUNCH_OK )
	{
		printf("mlp_layer_status FAIL\n");
		return(1);
	}
	printf("mlp_gemms %u\n", (unsigned)lm_recorded_gemms.size());
	for (index = 0u; index < lm_recorded_gemms.size(); ++index)
		printf("mlp_gemm in %u out %u\n",
			lm_recorded_gemms[index].input_dimension,
			lm_recorded_gemms[index].output_dimension);
	{
		float peak = 0.0f;
		for (index = 0u; index < ROWS * MIMO25_DENSE_INTERMEDIATE; ++index)
		{
			float v = LmBf16ToFloat(intermediate[index]);
			if ( v < 0.0f ) v = -v;
			if ( v > peak ) peak = v;
		}
		printf("mlp_intermediate_max %.9g\n", (double)peak);
	}
	return(0);
}
