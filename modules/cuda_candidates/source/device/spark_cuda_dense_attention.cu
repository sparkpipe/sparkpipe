#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_dense_attention.h"

#define SPARK_CUDA_DENSE_ATTENTION_SCORE_THREADS 512u
#define SPARK_CUDA_DENSE_ATTENTION_VALUE_THREADS 512u
#define SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT 2048u
#define SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE 128u
#define SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_LANES 4u
#define SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_SHIFT 7u
#define SPARK_CUDA_DENSE_ATTENTION_FUSED_HEAD_SIZE 128u
#define SPARK_CUDA_DENSE_ATTENTION_FUSED_THREADS 512u
#define SPARK_CUDA_DENSE_ATTENTION_FUSED_PAIR_LANES 64u
#define SPARK_CUDA_DENSE_ATTENTION_FUSED_TOKEN_LANES 8u
#define SPARK_CUDA_DENSE_ATTENTION_FUSED_TOKEN_SHIFT 6u
#define SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT 4u

static __device__ float SparkCudaDenseAttentionBf16ToFloat(uint16_t value)
{
	union
	{
		uint32_t u;
		float f;
	} bits;

	bits.u = ((uint32_t)value) << 16u;
	return bits.f;
}

static __device__ uint16_t SparkCudaDenseAttentionFloatToBf16(float value)
{
	union
	{
		uint32_t u;
		float f;
	} bits;
	uint32_t rounding_bias;

	bits.f = value;
	rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
	return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static uint64_t SparkCudaDenseAttentionQueryElementCountHost(const SparkCudaDenseAttentionRequest *request)
{
	return (uint64_t)request->query_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaDenseAttentionKvElementCountHost(const SparkCudaDenseAttentionRequest *request)
{
	return (uint64_t)request->kv_block_count * (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaDenseAttentionScoreCountHost(const SparkCudaDenseAttentionRequest *request)
{
	return (uint64_t)request->query_count * (uint64_t)request->head_count * (uint64_t)request->max_context_tokens;
}

static void SparkCudaDenseAttentionFillReportShape(const SparkCudaDenseAttentionRequest *request, SparkCudaDenseAttentionReport *report)
{
	report->query_element_count = SparkCudaDenseAttentionQueryElementCountHost(request);
	report->kv_cache_element_count = SparkCudaDenseAttentionKvElementCountHost(request);
	report->score_workspace_count = SparkCudaDenseAttentionScoreCountHost(request);
	report->output_element_count = report->query_element_count;
	report->device_counter_count = SPARKPIPE_CUDA_DENSE_ATTENTION_DEVICE_COUNTERS;
}

static __device__ uint64_t SparkCudaDenseAttentionCacheBaseOffset(const SparkCudaDenseAttentionRequest request, uint32_t block_id, uint32_t token_offset, uint32_t head_index)
{
	uint64_t slot_index;
	uint64_t head_stride;

	slot_index = ((uint64_t)block_id * (uint64_t)request.block_size) + (uint64_t)token_offset;
	head_stride = (uint64_t)request.head_count * (uint64_t)request.head_size;
	return (slot_index * head_stride) + ((uint64_t)head_index * (uint64_t)request.head_size);
}

static __device__ uint32_t SparkCudaDenseAttentionBlockForToken(const SparkCudaDenseAttentionRequest request, const uint32_t *block_table, uint32_t row_index, uint32_t token_index, uint32_t *token_offset)
{
	uint32_t logical_block;

	logical_block = token_index / request.block_size;
	*token_offset = token_index - (logical_block * request.block_size);
	if (logical_block >= request.max_blocks_per_query)
		return UINT32_MAX;
	return block_table[((uint64_t)row_index * (uint64_t)request.max_blocks_per_query) + (uint64_t)logical_block];
}

static __device__ uint32_t SparkCudaDenseAttentionWarpBlockForToken(const SparkCudaDenseAttentionRequest request, const uint32_t *block_table, uint32_t row_index, uint32_t token_index, uint32_t lane_index, uint32_t *token_offset, uint32_t *valid)
{
	uint32_t block_id;
	uint32_t local_offset;

	block_id = UINT32_MAX;
	local_offset = 0u;
	if (lane_index == 0u)
		block_id = SparkCudaDenseAttentionBlockForToken(request, block_table, row_index, token_index, &local_offset);
	block_id = __shfl_sync(0xffffffffu, block_id, 0);
	local_offset = __shfl_sync(0xffffffffu, local_offset, 0);
	*token_offset = local_offset;
	*valid = block_id < request.kv_block_count ? 1u : 0u;
	return block_id;
}

static __device__ float SparkCudaDenseAttentionWarpReduceMax(float value)
{
	float other;

	other = __shfl_down_sync(0xffffffffu, value, 16);
	value = other > value ? other : value;
	other = __shfl_down_sync(0xffffffffu, value, 8);
	value = other > value ? other : value;
	other = __shfl_down_sync(0xffffffffu, value, 4);
	value = other > value ? other : value;
	other = __shfl_down_sync(0xffffffffu, value, 2);
	value = other > value ? other : value;
	other = __shfl_down_sync(0xffffffffu, value, 1);
	value = other > value ? other : value;
	return value;
}

static __device__ float SparkCudaDenseAttentionWarpReduceSum(float value)
{
	value += __shfl_down_sync(0xffffffffu, value, 16);
	value += __shfl_down_sync(0xffffffffu, value, 8);
	value += __shfl_down_sync(0xffffffffu, value, 4);
	value += __shfl_down_sync(0xffffffffu, value, 2);
	value += __shfl_down_sync(0xffffffffu, value, 1);
	return value;
}

static __device__ float SparkCudaDenseAttentionBlockReduceMax(float value, float *shared_values)
{
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;

	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	value = SparkCudaDenseAttentionWarpReduceMax(value);
	if (lane_index == 0u)
		shared_values[warp_index] = value;
	__syncthreads();
	value = threadIdx.x < warp_count ? shared_values[lane_index] : -FLT_MAX;
	if (warp_index == 0u)
		value = SparkCudaDenseAttentionWarpReduceMax(value);
	if (threadIdx.x == 0u)
		shared_values[0] = value;
	__syncthreads();
	return shared_values[0];
}

static __device__ float SparkCudaDenseAttentionBlockReduceSum(float value, float *shared_values)
{
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;

	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	value = SparkCudaDenseAttentionWarpReduceSum(value);
	if (lane_index == 0u)
		shared_values[warp_index] = value;
	__syncthreads();
	value = threadIdx.x < warp_count ? shared_values[lane_index] : 0.0f;
	if (warp_index == 0u)
		value = SparkCudaDenseAttentionWarpReduceSum(value);
	if (threadIdx.x == 0u)
		shared_values[0] = value;
	__syncthreads();
	return shared_values[0];
}

static __device__ void SparkCudaDenseAttentionLoadQuery(const SparkCudaDenseAttentionRequest request, const uint16_t *query, uint32_t row_index, uint32_t head_index, float *shared_query)
{
	uint32_t dim_index;
	uint64_t query_offset;

	query_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.head_size;
	for (dim_index = threadIdx.x; dim_index < request.head_size; dim_index += blockDim.x)
		shared_query[dim_index] = SparkCudaDenseAttentionBf16ToFloat(query[query_offset + dim_index]);
	__syncthreads();
}

static __device__ float SparkCudaDenseAttentionWarpDotShared(const SparkCudaDenseAttentionRequest request, const float *shared_query, const uint16_t *key_cache, uint64_t cache_base_offset, uint32_t lane_index)
{
	uint32_t dim_index;
	float accum;

	accum = 0.0f;
	for (dim_index = lane_index; dim_index < request.head_size; dim_index += 32u)
		accum += shared_query[dim_index] * SparkCudaDenseAttentionBf16ToFloat(key_cache[cache_base_offset + dim_index]);
	return SparkCudaDenseAttentionWarpReduceSum(accum) * request.qk_scale;
}

static __device__ float SparkCudaDenseAttentionWarpDotSharedPair(const SparkCudaDenseAttentionRequest request, const float *shared_query, const uint16_t *key_cache, uint64_t cache_base_offset, uint32_t lane_index)
{
	const __nv_bfloat162 *key_pairs;
	uint32_t pair_index;
	uint32_t pair_count;
	uint32_t dim_index;
	float2 key_pair;
	float accum;

	if ((request.head_size & 1u) != 0u || (cache_base_offset & 1u) != 0u)
		return SparkCudaDenseAttentionWarpDotShared(request, shared_query, key_cache, cache_base_offset, lane_index);
	key_pairs = (const __nv_bfloat162 *)&key_cache[cache_base_offset];
	pair_count = request.head_size >> 1u;
	accum = 0.0f;
	for (pair_index = lane_index; pair_index < pair_count; pair_index += 32u)
	{
		dim_index = pair_index << 1u;
		key_pair = __bfloat1622float2(key_pairs[pair_index]);
		accum += shared_query[dim_index] * key_pair.x;
		accum += shared_query[dim_index + 1u] * key_pair.y;
	}
	return SparkCudaDenseAttentionWarpReduceSum(accum) * request.qk_scale;
}

static __global__ void SparkCudaDensePagedAttentionScoreKernel(SparkCudaDenseAttentionRequest request, const uint16_t *query, const uint16_t *key_cache, const uint32_t *block_table, const uint32_t *context_lengths, float *score_workspace, uint32_t *error_counters)
{
	__shared__ float shared_values[SPARK_CUDA_DENSE_ATTENTION_SCORE_THREADS];
	__shared__ float shared_query[SPARKPIPE_CUDA_DENSE_ATTENTION_MAX_HEAD_SIZE];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t token_index;
	uint32_t token_offset;
	uint32_t block_id;
	uint32_t valid;
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;
	uint64_t score_offset;
	uint64_t cache_base_offset;
	float *score_values;
	float local_max;
	float row_max;
	float score;
	float local_sum;
	float row_sum;
	float probability;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
	{
		if (threadIdx.x == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_CONTEXT_COUNTER], 1u);
		context_length = request.max_context_tokens;
	}
	SparkCudaDenseAttentionLoadQuery(request, query, row_index, head_index, shared_query);
	score_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.max_context_tokens;
	score_values = &score_workspace[score_offset];
	local_max = -FLT_MAX;
	for (token_index = warp_index; token_index < context_length; token_index += warp_count)
	{
		block_id = SparkCudaDenseAttentionWarpBlockForToken(request, block_table, row_index, token_index, lane_index, &token_offset, &valid);
		cache_base_offset = valid != 0u ? SparkCudaDenseAttentionCacheBaseOffset(request, block_id, token_offset, head_index) : 0u;
		score = valid != 0u ? SparkCudaDenseAttentionWarpDotSharedPair(request, shared_query, key_cache, cache_base_offset, lane_index) : -FLT_MAX;
		if (lane_index == 0u && valid == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_BLOCK_COUNTER], 1u);
		if (lane_index == 0u)
			score_values[token_index] = score;
		if (lane_index == 0u && score > local_max)
			local_max = score;
	}
	row_max = SparkCudaDenseAttentionBlockReduceMax(local_max, shared_values);
	local_sum = 0.0f;
	for (token_index = threadIdx.x; token_index < context_length; token_index += blockDim.x)
	{
		score = score_values[token_index];
		probability = __expf(score - row_max);
		score_values[token_index] = probability;
		local_sum += probability;
	}
	row_sum = SparkCudaDenseAttentionBlockReduceSum(local_sum, shared_values);
	for (token_index = threadIdx.x; token_index < context_length; token_index += blockDim.x)
		score_values[token_index] = row_sum > 0.0f ? score_values[token_index] / row_sum : 0.0f;
}

static __global__ void SparkCudaDensePagedAttentionValueKernel(SparkCudaDenseAttentionRequest request, const uint16_t *value_cache, const uint32_t *block_table, const uint32_t *context_lengths, const float *score_workspace, uint16_t *output)
{
	__shared__ float shared_scores[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	__shared__ float shared_values[SPARK_CUDA_DENSE_ATTENTION_VALUE_THREADS];
	__shared__ uint64_t shared_cache_bases[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t chunk_base;
	uint32_t chunk_count;
	uint32_t token_local;
	uint32_t token_index;
	uint32_t token_offset;
	uint32_t block_id;
	uint64_t cache_base_offset;
	uint32_t dim_lane;
	uint32_t token_lane;
	uint32_t dim_index;
	uint32_t valid;
	uint64_t score_offset;
	uint64_t output_offset;
	float accum;
	float value;
	uint32_t lane_index;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	dim_lane = threadIdx.x & (SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE - 1u);
	token_lane = threadIdx.x >> SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_SHIFT;
	dim_index = (blockIdx.z * SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE) + dim_lane;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
		context_length = request.max_context_tokens;
	score_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.max_context_tokens;
	output_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.head_size;
	accum = 0.0f;
	for (chunk_base = 0u; chunk_base < context_length; chunk_base += SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT)
	{
		chunk_count = context_length - chunk_base;
		if (chunk_count > SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT)
			chunk_count = SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT;
		for (token_local = threadIdx.x; token_local < chunk_count; token_local += blockDim.x)
		{
			token_index = chunk_base + token_local;
			block_id = SparkCudaDenseAttentionBlockForToken(request, block_table, row_index, token_index, &token_offset);
			shared_scores[token_local] = score_workspace[score_offset + token_index];
			shared_cache_bases[token_local] = block_id < request.kv_block_count ? SparkCudaDenseAttentionCacheBaseOffset(request, block_id, token_offset, head_index) : UINT64_MAX;
		}
		__syncthreads();
		if (dim_index < request.head_size)
		{
			for (token_local = token_lane; token_local < chunk_count; token_local += SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_LANES)
			{
				cache_base_offset = shared_cache_bases[token_local];
				valid = cache_base_offset != UINT64_MAX ? 1u : 0u;
				value = valid != 0u ? SparkCudaDenseAttentionBf16ToFloat(value_cache[cache_base_offset + dim_index]) : 0.0f;
				accum += shared_scores[token_local] * value;
			}
		}
		__syncthreads();
	}
	shared_values[threadIdx.x] = accum;
	__syncthreads();
	if (token_lane == 0u && dim_index < request.head_size)
	{
		for (lane_index = 1u; lane_index < SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_LANES; ++lane_index)
			accum += shared_values[(lane_index * SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE) + dim_lane];
		output[output_offset + dim_index] = SparkCudaDenseAttentionFloatToBf16(accum);
	}
}

static __global__ void SparkCudaDensePagedAttentionFused128Kernel(SparkCudaDenseAttentionRequest request, const uint16_t *query, const uint16_t *key_cache, const uint16_t *value_cache, const uint32_t *block_table, const uint32_t *context_lengths, uint32_t *error_counters, uint16_t *output)
{
	__shared__ float shared_values[SPARK_CUDA_DENSE_ATTENTION_FUSED_THREADS];
	__shared__ float shared_accum_lo[SPARK_CUDA_DENSE_ATTENTION_FUSED_THREADS];
	__shared__ float shared_accum_hi[SPARK_CUDA_DENSE_ATTENTION_FUSED_THREADS];
	__shared__ float shared_query[SPARK_CUDA_DENSE_ATTENTION_FUSED_HEAD_SIZE];
	__shared__ float shared_scores[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	__shared__ uint64_t shared_cache_bases[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	const __nv_bfloat162 *value_pairs;
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t token_index;
	uint32_t token_offset;
	uint32_t block_id;
	uint32_t valid;
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;
	uint32_t dim_lane;
	uint32_t dim_index;
	uint32_t token_lane;
	float2 value_pair;
	uint64_t cache_base_offset;
	uint64_t output_offset;
	float local_max;
	float row_max;
	float score;
	float local_sum;
	float row_sum;
	float inv_row_sum;
	float probability;
	float accum_lo;
	float accum_hi;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
	{
		if (threadIdx.x == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_CONTEXT_COUNTER], 1u);
		context_length = request.max_context_tokens;
	}
	SparkCudaDenseAttentionLoadQuery(request, query, row_index, head_index, shared_query);
	local_max = -FLT_MAX;
	for (token_index = warp_index; token_index < context_length; token_index += warp_count)
	{
		block_id = SparkCudaDenseAttentionWarpBlockForToken(request, block_table, row_index, token_index, lane_index, &token_offset, &valid);
		cache_base_offset = valid != 0u ? SparkCudaDenseAttentionCacheBaseOffset(request, block_id, token_offset, head_index) : UINT64_MAX;
		score = valid != 0u ? SparkCudaDenseAttentionWarpDotSharedPair(request, shared_query, key_cache, cache_base_offset, lane_index) : -FLT_MAX;
		if (lane_index == 0u)
		{
			if (valid == 0u)
				atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_BLOCK_COUNTER], 1u);
			shared_scores[token_index] = score;
			shared_cache_bases[token_index] = cache_base_offset;
			if (score > local_max)
				local_max = score;
		}
	}
	row_max = SparkCudaDenseAttentionBlockReduceMax(local_max, shared_values);
	local_sum = 0.0f;
	for (token_index = threadIdx.x; token_index < context_length; token_index += blockDim.x)
	{
		score = shared_scores[token_index];
		score = row_max <= (-FLT_MAX * 0.5f) ? 0.0f : __expf(score - row_max);
		shared_scores[token_index] = score;
		local_sum += score;
	}
	row_sum = SparkCudaDenseAttentionBlockReduceSum(local_sum, shared_values);
	inv_row_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
	__syncthreads();
	dim_lane = threadIdx.x & (SPARK_CUDA_DENSE_ATTENTION_FUSED_PAIR_LANES - 1u);
	dim_index = dim_lane << 1u;
	token_lane = threadIdx.x >> SPARK_CUDA_DENSE_ATTENTION_FUSED_TOKEN_SHIFT;
	accum_lo = 0.0f;
	accum_hi = 0.0f;
	for (token_index = token_lane; token_index < context_length; token_index += SPARK_CUDA_DENSE_ATTENTION_FUSED_TOKEN_LANES)
	{
		cache_base_offset = shared_cache_bases[token_index];
		valid = cache_base_offset != UINT64_MAX ? 1u : 0u;
		probability = shared_scores[token_index] * inv_row_sum;
		value_pairs = valid != 0u ? (const __nv_bfloat162 *)&value_cache[cache_base_offset + dim_index] : 0;
		value_pair = valid != 0u ? __bfloat1622float2(value_pairs[0]) : make_float2(0.0f, 0.0f);
		accum_lo += probability * value_pair.x;
		accum_hi += probability * value_pair.y;
	}
	shared_accum_lo[threadIdx.x] = accum_lo;
	shared_accum_hi[threadIdx.x] = accum_hi;
	__syncthreads();
	if (token_lane == 0u)
	{
		for (token_index = 1u; token_index < SPARK_CUDA_DENSE_ATTENTION_FUSED_TOKEN_LANES; ++token_index)
		{
			accum_lo += shared_accum_lo[(token_index * SPARK_CUDA_DENSE_ATTENTION_FUSED_PAIR_LANES) + dim_lane];
			accum_hi += shared_accum_hi[(token_index * SPARK_CUDA_DENSE_ATTENTION_FUSED_PAIR_LANES) + dim_lane];
		}
		output_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.head_size;
		output[output_offset + dim_index] = SparkCudaDenseAttentionFloatToBf16(accum_lo);
		output[output_offset + dim_index + 1u] = SparkCudaDenseAttentionFloatToBf16(accum_hi);
	}
}

static int32_t SparkCudaDenseAttentionCanUseFused128(const SparkCudaDenseAttentionRequest *request)
{
	if (request->head_size != SPARK_CUDA_DENSE_ATTENTION_FUSED_HEAD_SIZE)
		return 0;
	if (request->max_context_tokens > SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT)
		return 0;
	return 1;
}

static __device__ uint64_t SparkCudaDenseAttentionSplitWorkspaceBase(const SparkCudaDenseAttentionRequest request, uint32_t row_index, uint32_t head_index, uint32_t split_index)
{
	uint64_t row_head_index;
	uint64_t split_stride;

	row_head_index = ((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index;
	split_stride = (uint64_t)request.head_size + 2u;
	return ((row_head_index * (uint64_t)SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT) + (uint64_t)split_index) * split_stride;
}

static uint64_t SparkCudaDenseAttentionSplitWorkspaceCountHost(const SparkCudaDenseAttentionRequest *request)
{
	uint64_t row_head_count;
	uint64_t split_stride;

	row_head_count = (uint64_t)request->query_count * (uint64_t)request->head_count;
	split_stride = (uint64_t)request->head_size + 2u;
	return row_head_count * (uint64_t)SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT * split_stride;
}

static __global__ void SparkCudaDensePagedAttentionSplitKernel(SparkCudaDenseAttentionRequest request, const uint16_t *query, const uint16_t *key_cache, const uint16_t *value_cache, const uint32_t *block_table, const uint32_t *context_lengths, uint32_t *error_counters, float *split_workspace)
{
	__shared__ float shared_scores[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	__shared__ float shared_query[SPARK_CUDA_DENSE_ATTENTION_FUSED_HEAD_SIZE];
	__shared__ float shared_values[SPARK_CUDA_DENSE_ATTENTION_VALUE_THREADS];
	__shared__ uint64_t shared_cache_bases[SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t split_index;
	uint32_t context_length;
	uint32_t split_token_count;
	uint32_t token_begin;
	uint32_t token_end;
	uint32_t chunk_count;
	uint32_t token_local;
	uint32_t token_index;
	uint32_t token_offset;
	uint32_t block_id;
	uint32_t valid;
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;
	uint32_t dim_lane;
	uint32_t token_lane;
	uint64_t cache_base_offset;
	uint64_t workspace_base;
	float local_max;
	float row_max;
	float score;
	float local_sum;
	float row_sum;
	float accum;
	float value;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	split_index = blockIdx.z;
	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
	{
		if (threadIdx.x == 0u && split_index == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_CONTEXT_COUNTER], 1u);
		context_length = request.max_context_tokens;
	}
	split_token_count = (request.max_context_tokens + SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT - 1u) / SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT;
	token_begin = split_index * split_token_count;
	token_end = token_begin + split_token_count;
	if (token_end > context_length)
		token_end = context_length;
	chunk_count = token_end > token_begin ? token_end - token_begin : 0u;
	workspace_base = SparkCudaDenseAttentionSplitWorkspaceBase(request, row_index, head_index, split_index);
	SparkCudaDenseAttentionLoadQuery(request, query, row_index, head_index, shared_query);
	local_max = -FLT_MAX;
	for (token_local = warp_index; token_local < chunk_count; token_local += warp_count)
	{
		token_index = token_begin + token_local;
		block_id = SparkCudaDenseAttentionWarpBlockForToken(request, block_table, row_index, token_index, lane_index, &token_offset, &valid);
		cache_base_offset = valid != 0u ? SparkCudaDenseAttentionCacheBaseOffset(request, block_id, token_offset, head_index) : UINT64_MAX;
		score = valid != 0u ? SparkCudaDenseAttentionWarpDotSharedPair(request, shared_query, key_cache, cache_base_offset, lane_index) : -FLT_MAX;
		if (lane_index == 0u)
		{
			if (valid == 0u)
				atomicAdd(&error_counters[SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_BLOCK_COUNTER], 1u);
			shared_scores[token_local] = score;
			shared_cache_bases[token_local] = cache_base_offset;
			if (score > local_max)
				local_max = score;
		}
	}
	row_max = SparkCudaDenseAttentionBlockReduceMax(local_max, shared_values);
	local_sum = 0.0f;
	for (token_local = threadIdx.x; token_local < chunk_count; token_local += blockDim.x)
	{
		score = row_max <= (-FLT_MAX * 0.5f) ? 0.0f : __expf(shared_scores[token_local] - row_max);
		shared_scores[token_local] = score;
		local_sum += score;
	}
	row_sum = SparkCudaDenseAttentionBlockReduceSum(local_sum, shared_values);
	dim_lane = threadIdx.x & (SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE - 1u);
	token_lane = threadIdx.x >> SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_SHIFT;
	accum = 0.0f;
	for (token_local = token_lane; token_local < chunk_count; token_local += SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_LANES)
	{
		cache_base_offset = shared_cache_bases[token_local];
		valid = cache_base_offset != UINT64_MAX ? 1u : 0u;
		value = valid != 0u ? SparkCudaDenseAttentionBf16ToFloat(value_cache[cache_base_offset + dim_lane]) : 0.0f;
		accum += shared_scores[token_local] * value;
	}
	shared_values[threadIdx.x] = accum;
	__syncthreads();
	if (token_lane == 0u)
	{
		for (token_index = 1u; token_index < SPARK_CUDA_DENSE_ATTENTION_VALUE_TOKEN_LANES; ++token_index)
			accum += shared_values[(token_index * SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE) + dim_lane];
		split_workspace[workspace_base + 2u + dim_lane] = accum;
	}
	if (threadIdx.x == 0u)
	{
		split_workspace[workspace_base] = chunk_count != 0u ? row_max : -FLT_MAX;
		split_workspace[workspace_base + 1u] = chunk_count != 0u ? row_sum : 0.0f;
	}
}

static __global__ void SparkCudaDensePagedAttentionSplitReduceKernel(SparkCudaDenseAttentionRequest request, const float *split_workspace, uint16_t *output)
{
	uint32_t row_index;
	uint32_t head_index;
	uint32_t dim_index;
	uint32_t split_index;
	uint64_t workspace_base;
	uint64_t output_offset;
	float row_max;
	float row_sum;
	float split_max;
	float split_sum;
	float scale;
	float accum;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	dim_index = threadIdx.x;
	if (dim_index >= request.head_size)
		return;
	row_max = -FLT_MAX;
	for (split_index = 0u; split_index < SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT; ++split_index)
	{
		workspace_base = SparkCudaDenseAttentionSplitWorkspaceBase(request, row_index, head_index, split_index);
		split_max = split_workspace[workspace_base];
		if (split_max > row_max)
			row_max = split_max;
	}
	row_sum = 0.0f;
	accum = 0.0f;
	for (split_index = 0u; split_index < SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT; ++split_index)
	{
		workspace_base = SparkCudaDenseAttentionSplitWorkspaceBase(request, row_index, head_index, split_index);
		split_max = split_workspace[workspace_base];
		split_sum = split_workspace[workspace_base + 1u];
		scale = split_sum > 0.0f ? __expf(split_max - row_max) : 0.0f;
		row_sum += scale * split_sum;
		accum += scale * split_workspace[workspace_base + 2u + dim_index];
	}
	output_offset = (((uint64_t)row_index * (uint64_t)request.head_count) + (uint64_t)head_index) * (uint64_t)request.head_size;
	output[output_offset + dim_index] = SparkCudaDenseAttentionFloatToBf16(row_sum > 0.0f ? accum / row_sum : 0.0f);
}

static int32_t SparkCudaDenseAttentionCanUseSplit128(const SparkCudaDenseAttentionRequest *request, const float *device_score_workspace)
{
	if (device_score_workspace == 0)
		return 0;
	if (request->head_size != SPARK_CUDA_DENSE_ATTENTION_FUSED_HEAD_SIZE)
		return 0;
	if (request->max_context_tokens <= 1024u || request->max_context_tokens > SPARK_CUDA_DENSE_ATTENTION_SHARED_CONTEXT)
		return 0;
	return 1;
}

extern "C" SparkStatus SparkRunCudaDensePagedAttentionBf16(const SparkCudaDenseAttentionRequest *request, const void *device_query_bf16, const void *device_key_cache_bf16, const void *device_value_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_bf16, SparkCudaDenseAttentionReport *report)
{
	cudaError_t cuda_status;
	SparkStatus status;
	dim3 grid_shape;
	dim3 value_grid_shape;
	uint32_t dim_tile_count;
	uint64_t split_workspace_count;
	int32_t use_fused;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaDenseAttentionRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_query_bf16 == 0 || device_key_cache_bf16 == 0 || device_value_cache_bf16 == 0 || device_block_table == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (device_context_lengths == 0 || device_error_counters == 0 || device_output_bf16 == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkCudaDenseAttentionFillReportShape(request, report);
	if (SparkCudaDenseAttentionCanUseSplit128(request, device_score_workspace) != 0)
	{
		value_grid_shape = dim3(request->query_count, request->head_count, SPARK_CUDA_DENSE_ATTENTION_SPLIT_COUNT);
		SparkCudaDensePagedAttentionSplitKernel<<<value_grid_shape, SPARK_CUDA_DENSE_ATTENTION_VALUE_THREADS>>>(*request, (const uint16_t *)device_query_bf16, (const uint16_t *)device_key_cache_bf16, (const uint16_t *)device_value_cache_bf16, device_block_table, device_context_lengths, device_error_counters, device_score_workspace);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess)
			return SPARK_STATUS_INTERNAL_ERROR;
		grid_shape = dim3(request->query_count, request->head_count, 1u);
		SparkCudaDensePagedAttentionSplitReduceKernel<<<grid_shape, SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE>>>(*request, device_score_workspace, (uint16_t *)device_output_bf16);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess)
			return SPARK_STATUS_INTERNAL_ERROR;
		split_workspace_count = SparkCudaDenseAttentionSplitWorkspaceCountHost(request);
		report->score_workspace_count = split_workspace_count;
		report->attention_kernel_count = 2u;
		report->hot_path_allocation_count = 0u;
		return SPARK_STATUS_OK;
	}
	use_fused = SparkCudaDenseAttentionCanUseFused128(request);
	grid_shape = dim3(request->query_count, request->head_count, 1u);
	if (use_fused != 0)
	{
		SparkCudaDensePagedAttentionFused128Kernel<<<grid_shape, SPARK_CUDA_DENSE_ATTENTION_FUSED_THREADS>>>(*request, (const uint16_t *)device_query_bf16, (const uint16_t *)device_key_cache_bf16, (const uint16_t *)device_value_cache_bf16, device_block_table, device_context_lengths, device_error_counters, (uint16_t *)device_output_bf16);
		cuda_status = cudaGetLastError();
		if (cuda_status != cudaSuccess)
			return SPARK_STATUS_INTERNAL_ERROR;
		report->score_workspace_count = 0u;
		report->attention_kernel_count = 1u;
		report->hot_path_allocation_count = 0u;
		return SPARK_STATUS_OK;
	}
	if (device_score_workspace == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkCudaDensePagedAttentionScoreKernel<<<grid_shape, SPARK_CUDA_DENSE_ATTENTION_SCORE_THREADS>>>(*request, (const uint16_t *)device_query_bf16, (const uint16_t *)device_key_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, device_error_counters);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	dim_tile_count = (request->head_size + SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE - 1u) / SPARK_CUDA_DENSE_ATTENTION_VALUE_DIM_TILE;
	value_grid_shape = dim3(request->query_count, request->head_count, dim_tile_count);
	SparkCudaDensePagedAttentionValueKernel<<<value_grid_shape, SPARK_CUDA_DENSE_ATTENTION_VALUE_THREADS>>>(*request, (const uint16_t *)device_value_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, (uint16_t *)device_output_bf16);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	report->attention_kernel_count = 2u;
	report->hot_path_allocation_count = 0u;
	return SPARK_STATUS_OK;
}
