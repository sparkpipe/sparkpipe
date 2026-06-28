#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_mla_attention.h"

#define SPARK_CUDA_MLA_ATTENTION_THREADS 256u
#define SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES 2048u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE 64u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_TOKEN_LANES 4u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_TOKEN_SHIFT 6u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_LANES 32u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_TOKEN_LANES 8u
#define SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_TOKEN_SHIFT 5u

static __device__ float SparkCudaMlaAttentionBf16ToFloat(uint16_t value)
{
	union
	{
		uint32_t u;
		float f;
	} bits;

	bits.u = ((uint32_t)value) << 16u;
	return bits.f;
}

static __device__ uint16_t SparkCudaMlaAttentionFloatToBf16(float value)
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

static uint64_t SparkCudaMlaAttentionQueryLatentCountHost(const SparkCudaMlaAttentionRequest *request)
{
	return (uint64_t)request->query_count * (uint64_t)request->query_head_count * (uint64_t)request->latent_dim;
}

static uint64_t SparkCudaMlaAttentionQueryRopeCountHost(const SparkCudaMlaAttentionRequest *request)
{
	return (uint64_t)request->query_count * (uint64_t)request->query_head_count * (uint64_t)request->rope_dim;
}

static uint64_t SparkCudaMlaAttentionCacheCountHost(const SparkCudaMlaAttentionRequest *request)
{
	return SparkCudaMlaAttentionRequiredCacheElements(request);
}

static uint64_t SparkCudaMlaAttentionScoreCountHost(const SparkCudaMlaAttentionRequest *request, uint32_t sparse_mode)
{
	uint32_t score_columns;

	score_columns = sparse_mode != 0u ? request->sparse_top_k : request->max_context_tokens;
	return (uint64_t)request->query_count * (uint64_t)request->query_head_count * (uint64_t)score_columns;
}

static void SparkCudaMlaAttentionFillReportShape(const SparkCudaMlaAttentionRequest *request, SparkCudaMlaAttentionReport *report, uint32_t sparse_mode, uint64_t cuda_stream)
{
	report->query_latent_element_count = SparkCudaMlaAttentionQueryLatentCountHost(request);
	report->query_rope_element_count = SparkCudaMlaAttentionQueryRopeCountHost(request);
	report->cache_element_count = SparkCudaMlaAttentionCacheCountHost(request);
	report->cache_token_stride_elements = request->cache_token_stride_elements;
	report->score_workspace_count = SparkCudaMlaAttentionScoreCountHost(request, sparse_mode);
	report->sparse_index_count = sparse_mode != 0u ? (uint64_t)request->query_count * (uint64_t)request->query_head_count * (uint64_t)request->sparse_top_k : 0u;
	report->output_element_count = report->query_latent_element_count;
	report->cache_token_capacity = request->cache_token_capacity;
	report->first_block_token_offset = request->first_block_token_offset;
	report->device_counter_count = SPARKPIPE_CUDA_MLA_ATTENTION_DEVICE_COUNTERS;
	report->explicit_cache_stride_count = 1u;
	report->partial_first_block_count = request->first_block_token_offset != 0u ? 1u : 0u;
	report->explicit_stream_count = cuda_stream != 0u ? 1u : 0u;
	report->default_stream_count = cuda_stream == 0u ? 1u : 0u;
}

static __device__ uint64_t SparkCudaMlaAttentionCacheOffset(const SparkCudaMlaAttentionRequest request, uint32_t slot_index, uint32_t dim_index)
{
	return ((uint64_t)slot_index * (uint64_t)request.cache_token_stride_elements) + (uint64_t)dim_index;
}

static __device__ uint32_t SparkCudaMlaAttentionSlotForToken(const SparkCudaMlaAttentionRequest request, const uint32_t *block_table, uint32_t row_index, uint32_t token_index, uint32_t *valid)
{
	uint32_t addressed_token_index;
	uint32_t logical_block;
	uint32_t token_offset;
	uint32_t block_id;
	uint32_t slot_index;

	addressed_token_index = request.first_block_token_offset + token_index;
	logical_block = addressed_token_index / request.block_size;
	token_offset = addressed_token_index - (logical_block * request.block_size);
	if (logical_block >= request.max_blocks_per_query)
	{
		*valid = 0u;
		return SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
	}
	block_id = block_table[((uint64_t)row_index * (uint64_t)request.max_blocks_per_query) + (uint64_t)logical_block];
	if (block_id >= request.kv_block_count)
	{
		*valid = 0u;
		return SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
	}
	slot_index = (block_id * request.block_size) + token_offset;
	if (slot_index >= request.cache_token_capacity)
	{
		*valid = 0u;
		return SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
	}
	*valid = 1u;
	return slot_index;
}

static __device__ uint32_t SparkCudaMlaAttentionWarpSlotForToken(const SparkCudaMlaAttentionRequest request, const uint32_t *block_table, uint32_t row_index, uint32_t token_index, uint32_t lane_index, uint32_t *valid)
{
	uint32_t slot_index;
	uint32_t local_valid;

	slot_index = SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
	local_valid = 0u;
	if (lane_index == 0u)
		slot_index = SparkCudaMlaAttentionSlotForToken(request, block_table, row_index, token_index, &local_valid);
	slot_index = __shfl_sync(0xffffffffu, slot_index, 0);
	local_valid = __shfl_sync(0xffffffffu, local_valid, 0);
	*valid = local_valid;
	return slot_index;
}

static __device__ float SparkCudaMlaAttentionWarpReduceSum(float value)
{
	value += __shfl_down_sync(0xffffffffu, value, 16);
	value += __shfl_down_sync(0xffffffffu, value, 8);
	value += __shfl_down_sync(0xffffffffu, value, 4);
	value += __shfl_down_sync(0xffffffffu, value, 2);
	value += __shfl_down_sync(0xffffffffu, value, 1);
	return value;
}

static __device__ float SparkCudaMlaAttentionBlockReduceMax(float value, float *shared_values)
{
	uint32_t stride;

	shared_values[threadIdx.x] = value;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride && shared_values[threadIdx.x + stride] > shared_values[threadIdx.x])
			shared_values[threadIdx.x] = shared_values[threadIdx.x + stride];
		__syncthreads();
	}
	return shared_values[0];
}

static __device__ float SparkCudaMlaAttentionBlockReduceSum(float value, float *shared_values)
{
	uint32_t stride;

	shared_values[threadIdx.x] = value;
	__syncthreads();
	for (stride = blockDim.x >> 1u; stride > 0u; stride >>= 1u)
	{
		if (threadIdx.x < stride)
			shared_values[threadIdx.x] += shared_values[threadIdx.x + stride];
		__syncthreads();
	}
	return shared_values[0];
}

static __device__ void SparkCudaMlaAttentionLoadQuery(const SparkCudaMlaAttentionRequest request, const uint16_t *query_latent, const uint16_t *query_rope, uint32_t row_index, uint32_t head_index, float *shared_query)
{
	uint64_t latent_query_offset;
	uint64_t rope_query_offset;
	uint32_t dim_index;

	latent_query_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.latent_dim;
	rope_query_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.rope_dim;
	for (dim_index = threadIdx.x; dim_index < request.latent_dim; dim_index += blockDim.x)
		shared_query[dim_index] = SparkCudaMlaAttentionBf16ToFloat(query_latent[latent_query_offset + dim_index]);
	for (dim_index = threadIdx.x; dim_index < request.rope_dim; dim_index += blockDim.x)
		shared_query[request.latent_dim + dim_index] = SparkCudaMlaAttentionBf16ToFloat(query_rope[rope_query_offset + dim_index]);
	__syncthreads();
}

static __device__ float SparkCudaMlaAttentionWarpDotShared(const SparkCudaMlaAttentionRequest request, const float *shared_query, const uint16_t *mla_cache, uint32_t slot_index, uint32_t lane_index)
{
	float accum;
	uint32_t dim_index;

	accum = 0.0f;
	for (dim_index = lane_index; dim_index < request.latent_dim; dim_index += 32u)
		accum += shared_query[dim_index] * SparkCudaMlaAttentionBf16ToFloat(mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, dim_index)]);
	for (dim_index = lane_index; dim_index < request.rope_dim; dim_index += 32u)
		accum += shared_query[request.latent_dim + dim_index] * SparkCudaMlaAttentionBf16ToFloat(mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, request.latent_dim + dim_index)]);
	return SparkCudaMlaAttentionWarpReduceSum(accum) * request.qk_scale;
}

static __device__ float SparkCudaMlaAttentionWarpDotSharedPair(const SparkCudaMlaAttentionRequest request, const float *shared_query, const uint16_t *mla_cache, uint32_t slot_index, uint32_t lane_index)
{
	const uint32_t *cache_pairs;
	uint32_t pair_index;
	uint32_t pair_count;
	uint32_t dim_index;
	uint32_t packed_value;
	float accum;

	if ((request.latent_dim & 1u) != 0u || (request.rope_dim & 1u) != 0u || (request.cache_token_stride_elements & 1u) != 0u)
		return SparkCudaMlaAttentionWarpDotShared(request, shared_query, mla_cache, slot_index, lane_index);
	accum = 0.0f;
	cache_pairs = (const uint32_t *)&mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, 0u)];
	pair_count = request.latent_dim >> 1u;
	for (pair_index = lane_index; pair_index < pair_count; pair_index += 32u)
	{
		dim_index = pair_index << 1u;
		packed_value = cache_pairs[pair_index];
		accum += shared_query[dim_index] * SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value & 0xffffu));
		accum += shared_query[dim_index + 1u] * SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value >> 16u));
	}
	cache_pairs = (const uint32_t *)&mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, request.latent_dim)];
	pair_count = request.rope_dim >> 1u;
	for (pair_index = lane_index; pair_index < pair_count; pair_index += 32u)
	{
		dim_index = pair_index << 1u;
		packed_value = cache_pairs[pair_index];
		accum += shared_query[request.latent_dim + dim_index] * SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value & 0xffffu));
		accum += shared_query[request.latent_dim + dim_index + 1u] * SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value >> 16u));
	}
	return SparkCudaMlaAttentionWarpReduceSum(accum) * request.qk_scale;
}

static __global__ void SparkCudaMlaDenseScoreKernel(SparkCudaMlaAttentionRequest request, const uint16_t *query_latent, const uint16_t *query_rope, const uint16_t *mla_cache, const uint32_t *block_table, const uint32_t *context_lengths, float *score_workspace, uint32_t *error_counters)
{
	__shared__ float shared_values[SPARK_CUDA_MLA_ATTENTION_THREADS];
	__shared__ float shared_query[SPARKPIPE_CUDA_MLA_ATTENTION_MAX_LATENT_DIM + SPARKPIPE_CUDA_MLA_ATTENTION_MAX_ROPE_DIM];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t token_index;
	uint32_t valid;
	uint32_t slot_index;
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;
	uint64_t score_offset;
	float *score_values;
	float local_max;
	float row_max;
	float score;
	float local_sum;
	float row_sum;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
	{
		if (threadIdx.x == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_CONTEXT_COUNTER], 1u);
		context_length = request.max_context_tokens;
	}
	SparkCudaMlaAttentionLoadQuery(request, query_latent, query_rope, row_index, head_index, shared_query);
	score_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.max_context_tokens;
	score_values = &score_workspace[score_offset];
	local_max = -FLT_MAX;
	for (token_index = warp_index; token_index < context_length; token_index += warp_count)
	{
		slot_index = SparkCudaMlaAttentionWarpSlotForToken(request, block_table, row_index, token_index, lane_index, &valid);
		score = valid != 0u ? SparkCudaMlaAttentionWarpDotSharedPair(request, shared_query, mla_cache, slot_index, lane_index) : -FLT_MAX;
		if (lane_index == 0u && valid == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_BLOCK_COUNTER], 1u);
		if (lane_index == 0u)
			score_values[token_index] = score;
		if (lane_index == 0u && score > local_max)
			local_max = score;
	}
	row_max = SparkCudaMlaAttentionBlockReduceMax(local_max, shared_values);
	local_sum = 0.0f;
	for (token_index = threadIdx.x; token_index < context_length; token_index += blockDim.x)
	{
		score = row_max <= (-FLT_MAX * 0.5f) ? 0.0f : expf(score_values[token_index] - row_max);
		score_values[token_index] = score;
		local_sum += score;
	}
	row_sum = SparkCudaMlaAttentionBlockReduceSum(local_sum, shared_values);
	for (token_index = threadIdx.x; token_index < context_length; token_index += blockDim.x)
		score_values[token_index] = row_sum > 0.0f ? score_values[token_index] / row_sum : 0.0f;
}

static __global__ void SparkCudaMlaDenseValueKernel(SparkCudaMlaAttentionRequest request, const uint16_t *mla_cache, const uint32_t *block_table, const uint32_t *context_lengths, const float *score_workspace, uint16_t *output_latent)
{
	__shared__ float shared_scores[SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES];
	__shared__ float shared_values[SPARK_CUDA_MLA_ATTENTION_THREADS];
	__shared__ uint32_t shared_slots[SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t chunk_base;
	uint32_t chunk_count;
	uint32_t token_local;
	uint32_t token_index;
	uint32_t dim_lane;
	uint32_t token_lane;
	uint32_t dim_index;
	uint32_t valid;
	uint32_t slot_index;
	uint64_t score_offset;
	uint64_t output_offset;
	float accum;
	float value;
	uint32_t lane_index;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	dim_lane = threadIdx.x & (SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE - 1u);
	token_lane = threadIdx.x >> SPARK_CUDA_MLA_ATTENTION_VALUE_TOKEN_SHIFT;
	dim_index = (blockIdx.z * SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE) + dim_lane;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
		context_length = request.max_context_tokens;
	score_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.max_context_tokens;
	output_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.latent_dim;
	accum = 0.0f;
	for (chunk_base = 0u; chunk_base < context_length; chunk_base += SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES)
	{
		chunk_count = context_length - chunk_base;
		if (chunk_count > SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES)
			chunk_count = SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES;
		for (token_local = threadIdx.x; token_local < chunk_count; token_local += blockDim.x)
		{
			shared_scores[token_local] = score_workspace[score_offset + chunk_base + token_local];
			token_index = chunk_base + token_local;
			shared_slots[token_local] = SparkCudaMlaAttentionSlotForToken(request, block_table, row_index, token_index, &valid);
		}
		__syncthreads();
		if (dim_index < request.latent_dim)
		{
			for (token_local = token_lane; token_local < chunk_count; token_local += SPARK_CUDA_MLA_ATTENTION_VALUE_TOKEN_LANES)
			{
				slot_index = shared_slots[token_local];
				value = slot_index != SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT ? SparkCudaMlaAttentionBf16ToFloat(mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, dim_index)]) : 0.0f;
				accum += shared_scores[token_local] * value;
			}
		}
		__syncthreads();
	}
	shared_values[threadIdx.x] = accum;
	__syncthreads();
	if (token_lane == 0u && dim_index < request.latent_dim)
	{
		for (lane_index = 1u; lane_index < SPARK_CUDA_MLA_ATTENTION_VALUE_TOKEN_LANES; ++lane_index)
			accum += shared_values[(lane_index * SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE) + dim_lane];
		output_latent[output_offset + dim_index] = SparkCudaMlaAttentionFloatToBf16(accum);
	}
}

static __global__ void SparkCudaMlaDenseValuePairKernel(SparkCudaMlaAttentionRequest request, const uint16_t *mla_cache, const uint32_t *block_table, const uint32_t *context_lengths, const float *score_workspace, uint16_t *output_latent)
{
	__shared__ float shared_scores[SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES];
	__shared__ float shared_accum_lo[SPARK_CUDA_MLA_ATTENTION_THREADS];
	__shared__ float shared_accum_hi[SPARK_CUDA_MLA_ATTENTION_THREADS];
	__shared__ uint32_t shared_slots[SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES];
	const uint32_t *value_pairs;
	uint32_t row_index;
	uint32_t head_index;
	uint32_t context_length;
	uint32_t chunk_base;
	uint32_t chunk_count;
	uint32_t token_local;
	uint32_t token_index;
	uint32_t dim_pair_lane;
	uint32_t token_lane;
	uint32_t dim_index;
	uint32_t valid;
	uint32_t slot_index;
	uint32_t packed_value;
	uint64_t score_offset;
	uint64_t output_offset;
	float accum_lo;
	float accum_hi;
	float value;
	uint32_t lane_index;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	dim_pair_lane = threadIdx.x & (SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_LANES - 1u);
	token_lane = threadIdx.x >> SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_TOKEN_SHIFT;
	dim_index = (blockIdx.z * SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE) + (dim_pair_lane << 1u);
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
		context_length = request.max_context_tokens;
	score_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.max_context_tokens;
	output_offset = (((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index) * (uint64_t)request.latent_dim;
	accum_lo = 0.0f;
	accum_hi = 0.0f;
	for (chunk_base = 0u; chunk_base < context_length; chunk_base += SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES)
	{
		chunk_count = context_length - chunk_base;
		if (chunk_count > SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES)
			chunk_count = SPARK_CUDA_MLA_ATTENTION_SHARED_SCORES;
		for (token_local = threadIdx.x; token_local < chunk_count; token_local += blockDim.x)
		{
			shared_scores[token_local] = score_workspace[score_offset + chunk_base + token_local];
			token_index = chunk_base + token_local;
			shared_slots[token_local] = SparkCudaMlaAttentionSlotForToken(request, block_table, row_index, token_index, &valid);
		}
		__syncthreads();
		if (dim_index < request.latent_dim)
		{
			for (token_local = token_lane; token_local < chunk_count; token_local += SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_TOKEN_LANES)
			{
				slot_index = shared_slots[token_local];
				value_pairs = slot_index != SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT ? (const uint32_t *)&mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, dim_index)] : 0;
				packed_value = value_pairs != 0 ? value_pairs[0] : 0u;
				value = SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value & 0xffffu));
				accum_lo += shared_scores[token_local] * value;
				value = dim_index + 1u < request.latent_dim ? SparkCudaMlaAttentionBf16ToFloat((uint16_t)(packed_value >> 16u)) : 0.0f;
				accum_hi += shared_scores[token_local] * value;
			}
		}
		__syncthreads();
	}
	shared_accum_lo[threadIdx.x] = accum_lo;
	shared_accum_hi[threadIdx.x] = accum_hi;
	__syncthreads();
	if (token_lane == 0u && dim_index < request.latent_dim)
	{
		for (lane_index = 1u; lane_index < SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_TOKEN_LANES; ++lane_index)
		{
			accum_lo += shared_accum_lo[(lane_index * SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_LANES) + dim_pair_lane];
			accum_hi += shared_accum_hi[(lane_index * SPARK_CUDA_MLA_ATTENTION_VALUE_PAIR_LANES) + dim_pair_lane];
		}
		output_latent[output_offset + dim_index] = SparkCudaMlaAttentionFloatToBf16(accum_lo);
		if (dim_index + 1u < request.latent_dim)
			output_latent[output_offset + dim_index + 1u] = SparkCudaMlaAttentionFloatToBf16(accum_hi);
	}
}

static __global__ void SparkCudaMlaSparseAttentionKernel(SparkCudaMlaAttentionRequest request, const uint16_t *query_latent, const uint16_t *query_rope, const uint16_t *mla_cache, const uint32_t *block_table, const uint32_t *context_lengths, const uint32_t *sparse_token_indices, float *score_workspace, uint32_t *error_counters, uint32_t *mapped_sparse_slots, uint16_t *output_latent)
{
	__shared__ float shared_values[SPARK_CUDA_MLA_ATTENTION_THREADS];
	__shared__ float shared_scores[SPARKPIPE_CUDA_MLA_ATTENTION_MAX_SPARSE_TOP_K];
	__shared__ float shared_query[SPARKPIPE_CUDA_MLA_ATTENTION_MAX_LATENT_DIM + SPARKPIPE_CUDA_MLA_ATTENTION_MAX_ROPE_DIM];
	uint32_t row_index;
	uint32_t head_index;
	uint32_t candidate_index;
	uint32_t token_index;
	uint32_t context_length;
	uint32_t dim_index;
	uint32_t valid;
	uint32_t slot_index;
	uint32_t lane_index;
	uint32_t warp_index;
	uint32_t warp_count;
	uint64_t sparse_row_index;
	uint64_t sparse_offset;
	uint64_t output_offset;
	float *score_values;
	float local_max;
	float row_max;
	float score;
	float local_sum;
	float row_sum;
	float accum;

	row_index = blockIdx.x;
	head_index = blockIdx.y;
	lane_index = threadIdx.x & 31u;
	warp_index = threadIdx.x >> 5u;
	warp_count = blockDim.x >> 5u;
	context_length = context_lengths[row_index];
	if (context_length > request.max_context_tokens)
	{
		if (threadIdx.x == 0u)
			atomicAdd(&error_counters[SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_CONTEXT_COUNTER], 1u);
		context_length = request.max_context_tokens;
	}
	SparkCudaMlaAttentionLoadQuery(request, query_latent, query_rope, row_index, head_index, shared_query);
	sparse_row_index = ((uint64_t)row_index * (uint64_t)request.query_head_count) + (uint64_t)head_index;
	sparse_offset = sparse_row_index * (uint64_t)request.sparse_top_k;
	score_values = request.sparse_top_k <= SPARKPIPE_CUDA_MLA_ATTENTION_MAX_SPARSE_TOP_K ? shared_scores : &score_workspace[sparse_offset];
	local_max = -FLT_MAX;
	for (candidate_index = warp_index; candidate_index < request.sparse_top_k; candidate_index += warp_count)
	{
		token_index = sparse_token_indices[sparse_offset + candidate_index];
		valid = token_index < context_length ? 1u : 0u;
		slot_index = valid != 0u ? SparkCudaMlaAttentionSlotForToken(request, block_table, row_index, token_index, &valid) : SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
		score = valid != 0u ? SparkCudaMlaAttentionWarpDotSharedPair(request, shared_query, mla_cache, slot_index, lane_index) : -FLT_MAX;
		if (lane_index == 0u)
		{
			score_values[candidate_index] = score;
			mapped_sparse_slots[sparse_offset + candidate_index] = valid != 0u ? slot_index : SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT;
			if (valid == 0u)
				atomicAdd(&error_counters[SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SPARSE_COUNTER], 1u);
			if (score > local_max)
				local_max = score;
		}
	}
	row_max = SparkCudaMlaAttentionBlockReduceMax(local_max, shared_values);
	local_sum = 0.0f;
	for (candidate_index = threadIdx.x; candidate_index < request.sparse_top_k; candidate_index += blockDim.x)
	{
		score = row_max <= (-FLT_MAX * 0.5f) ? 0.0f : expf(score_values[candidate_index] - row_max);
		score_values[candidate_index] = score;
		local_sum += score;
	}
	row_sum = SparkCudaMlaAttentionBlockReduceSum(local_sum, shared_values);
	output_offset = sparse_row_index * (uint64_t)request.latent_dim;
	for (dim_index = threadIdx.x; dim_index < request.latent_dim; dim_index += blockDim.x)
	{
		accum = 0.0f;
		for (candidate_index = 0u; candidate_index < request.sparse_top_k; ++candidate_index)
		{
			slot_index = mapped_sparse_slots[sparse_offset + candidate_index];
			if (slot_index != SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT && row_sum > 0.0f)
				accum += (score_values[candidate_index] / row_sum) * SparkCudaMlaAttentionBf16ToFloat(mla_cache[SparkCudaMlaAttentionCacheOffset(request, slot_index, dim_index)]);
		}
		output_latent[output_offset + dim_index] = SparkCudaMlaAttentionFloatToBf16(accum);
	}
}

extern "C" SparkStatus SparkRunCudaMlaDenseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, uint64_t cuda_stream_handle, SparkCudaMlaAttentionReport *report)
{
	cudaError_t cuda_status;
	cudaStream_t cuda_stream;
	SparkStatus status;
	dim3 grid_shape;
	dim3 value_grid_shape;
	uint32_t dim_tile_count;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaMlaAttentionRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_query_latent_bf16 == 0 || device_query_rope_bf16 == 0 || device_mla_cache_bf16 == 0 || device_block_table == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (device_context_lengths == 0 || device_score_workspace == 0 || device_error_counters == 0 || device_output_latent_bf16 == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	cuda_stream = (cudaStream_t)(uintptr_t)cuda_stream_handle;
	SparkCudaMlaAttentionFillReportShape(request, report, 0u, cuda_stream_handle);
	grid_shape = dim3(request->query_count, request->query_head_count, 1u);
	SparkCudaMlaDenseScoreKernel<<<grid_shape, SPARK_CUDA_MLA_ATTENTION_THREADS, 0u, cuda_stream>>>(*request, (const uint16_t *)device_query_latent_bf16, (const uint16_t *)device_query_rope_bf16, (const uint16_t *)device_mla_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, device_error_counters);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	dim_tile_count = (request->latent_dim + SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE - 1u) / SPARK_CUDA_MLA_ATTENTION_VALUE_DIM_TILE;
	value_grid_shape = dim3(request->query_count, request->query_head_count, dim_tile_count);
	if ((request->latent_dim & 1u) == 0u && (request->cache_token_stride_elements & 1u) == 0u)
		SparkCudaMlaDenseValuePairKernel<<<value_grid_shape, SPARK_CUDA_MLA_ATTENTION_THREADS, 0u, cuda_stream>>>(*request, (const uint16_t *)device_mla_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, (uint16_t *)device_output_latent_bf16);
	else
		SparkCudaMlaDenseValueKernel<<<value_grid_shape, SPARK_CUDA_MLA_ATTENTION_THREADS, 0u, cuda_stream>>>(*request, (const uint16_t *)device_mla_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, (uint16_t *)device_output_latent_bf16);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	report->dense_kernel_count = 2u;
	report->hot_path_allocation_count = 0u;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMlaSparseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, uint64_t cuda_stream_handle, SparkCudaMlaAttentionReport *report)
{
	cudaError_t cuda_status;
	cudaStream_t cuda_stream;
	SparkStatus status;
	dim3 grid_shape;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaMlaAttentionRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_query_latent_bf16 == 0 || device_query_rope_bf16 == 0 || device_mla_cache_bf16 == 0 || device_block_table == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (device_context_lengths == 0 || device_sparse_token_indices == 0 || device_score_workspace == 0 || device_error_counters == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (device_mapped_sparse_slots == 0 || device_output_latent_bf16 == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	cuda_stream = (cudaStream_t)(uintptr_t)cuda_stream_handle;
	SparkCudaMlaAttentionFillReportShape(request, report, 1u, cuda_stream_handle);
	grid_shape = dim3(request->query_count, request->query_head_count, 1u);
	SparkCudaMlaSparseAttentionKernel<<<grid_shape, SPARK_CUDA_MLA_ATTENTION_THREADS, 0u, cuda_stream>>>(*request, (const uint16_t *)device_query_latent_bf16, (const uint16_t *)device_query_rope_bf16, (const uint16_t *)device_mla_cache_bf16, device_block_table, device_context_lengths, device_sparse_token_indices, device_score_workspace, device_error_counters, device_mapped_sparse_slots, (uint16_t *)device_output_latent_bf16);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess)
		return SPARK_STATUS_INTERNAL_ERROR;
	report->sparse_kernel_count = 1u;
	report->hot_path_allocation_count = 0u;
	return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaMlaDenseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report)
{
	return SparkRunCudaMlaDenseAttentionBf16OnStream(request, device_query_latent_bf16, device_query_rope_bf16, device_mla_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, device_error_counters, device_output_latent_bf16, 0u, report);
}

extern "C" SparkStatus SparkRunCudaMlaSparseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report)
{
	return SparkRunCudaMlaSparseAttentionBf16OnStream(request, device_query_latent_bf16, device_query_rope_bf16, device_mla_cache_bf16, device_block_table, device_context_lengths, device_sparse_token_indices, device_score_workspace, device_error_counters, device_mapped_sparse_slots, device_output_latent_bf16, 0u, report);
}
