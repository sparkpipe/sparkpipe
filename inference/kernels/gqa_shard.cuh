#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/gqa.cuh"
#include "inference/kernels/kv_shard.cuh"
#include "runtime/launch.h"
#include <math.h>
#include <stdint.h>

#define LM_GQA_SHARD_MAX_GROUP 32u

static __host__ __device__ __forceinline__ uint32_t LmGqaShardWidth(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(heads * value_dim / degree);
}

static __host__ __device__ __forceinline__ uint32_t LmGqaShardHeadSlots(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return((LmGqaShardWidth(heads,value_dim,degree) + value_dim - 1u) / value_dim + 1u);
}

static __host__ __device__ __forceinline__ uint32_t LmGqaShardRecordFloats(uint32_t heads, uint32_t value_dim, uint32_t degree)
{
	return(LmGqaShardWidth(heads,value_dim,degree) + 2u * LmGqaShardHeadSlots(heads,value_dim,degree));
}

static __host__ __forceinline__ uint32_t LmGqaShardGeometryValid(uint32_t heads, uint32_t kv_heads, uint32_t value_dim, uint32_t degree)
{
	return(degree != 0u && degree <= SPARK_KV_SHARD_MAX_DEGREE && kv_heads != 0u && heads >= kv_heads &&
		(heads % kv_heads) == 0u && heads / kv_heads <= LM_GQA_SHARD_MAX_GROUP &&
		((uint64_t)heads * value_dim) % degree == 0u ? 1u : 0u);
}

template<uint32_t COUNT>
static __device__ __forceinline__ void LmGqaShardLoad(const uint16_t *__restrict__ source, float *values)
{
	uint32_t index;
	#pragma unroll
	for (index = 0u; index < COUNT; index += 2u)
	{
		const uint32_t word = *(const uint32_t *)(source + index);
		values[index] = __uint_as_float(word << 16u);
		values[index + 1u] = __uint_as_float(word & 0xffff0000u);
	}
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(LM_GQA_SHARD_MAX_GROUP * 32u, 1)
void LmGqaShardPartialKernel(
	Pages cache,
	const uint16_t *__restrict__ query_bf16,
	uint32_t heads,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ row_position,
	float qk_scale,
	float *__restrict__ send,
	uint64_t destination_stride,
	uint32_t rows)
{
	constexpr uint32_t KEYS = HEAD_DIM / 32u;
	constexpr uint32_t VALUES = VALUE_DIM / 32u;
	static_assert(HEAD_DIM % 64u == 0u && VALUE_DIM % 64u == 0u, "each lane reads whole bf16 pairs of a head");
	static_assert(Geometry::kSlotBytes == KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u,
		"a slot is [K: kv heads x head_dim][V: kv heads x value_dim] bf16 and nothing else");
	const uint32_t row = blockIdx.x, kv_head = blockIdx.y, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
	const uint32_t group = heads / KV_HEADS, head = kv_head * group + warp;
	const uint32_t degree = cache.shard.degree;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,degree), slots = LmGqaShardHeadSlots(heads,VALUE_DIM,degree);
	const uint32_t record = width + 2u * slots;
	float query[KEYS], key[KEYS], value[VALUES], accumulator[VALUES];
	float running_max = -INFINITY, running_sum = 0.0f, score, previous, rescale, weight;
	uint32_t sequence, limit, count, local, position, index, destination, first, last, element;
	const uint8_t *slot;

	if ( row >= rows || warp >= group )
		return;
	sequence = sequence_of_row[row];
	if ( !LmKvViewIsConfigured(cache.pages) || sequence >= cache.pages.sequence_count )
	{
		LmKvReportRequiredAccessFailure(cache.pages,
			!LmKvViewIsConfigured(cache.pages) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,row,sequence,0xffffffffu,0xffffffffu);
		return;
	}
	if ( (heads % KV_HEADS) != 0u || group > LM_GQA_SHARD_MAX_GROUP || ((uint64_t)heads * VALUE_DIM) % degree != 0u )
	{
		LmKvReportRequiredAccessFailure(cache.pages,LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY,LM_KV_ACCESS_READ,row,sequence,head,heads);
		return;
	}
	limit = context_length[sequence];
	if ( row_position != 0 && row_position[row] != LM_KV_POSITION_UNUSED && row_position[row] + 1u < limit )
		limit = row_position[row] + 1u;
	LmGqaShardLoad<KEYS>(query_bf16 + ((uint64_t)row * heads + head) * HEAD_DIM + lane * KEYS,query);
	#pragma unroll
	for (index = 0u; index < VALUES; ++index)
		accumulator[index] = 0.0f;
	count = SparkKvShardLocalKeys(cache.shard,limit);
	for (local = 0u; local < count; ++local)
	{
		position = SparkKvShardLocalPosition(cache.shard,local);
		slot = LmKvShardSlotRequired<Geometry>(cache,sequence,position,row,LM_KV_ACCESS_READ);
		if ( slot == 0 )
			return;
		LmGqaShardLoad<KEYS>((const uint16_t *)slot + kv_head * HEAD_DIM + lane * KEYS,key);
		LmGqaShardLoad<VALUES>((const uint16_t *)slot + KV_HEADS * HEAD_DIM + kv_head * VALUE_DIM + lane * VALUES,value);
		score = 0.0f;
		#pragma unroll
		for (index = 0u; index < KEYS; ++index)
			score = fmaf(query[index],key[index],score);
		#pragma unroll
		for (index = 16u; index > 0u; index >>= 1u)
			score += __shfl_xor_sync(0xffffffffu,score,index);
		score *= qk_scale;
		previous = running_max;
		running_max = fmaxf(previous,score);
		rescale = __expf(previous - running_max);
		weight = __expf(score - running_max);
		running_sum = fmaf(running_sum,rescale,weight);
		#pragma unroll
		for (index = 0u; index < VALUES; ++index)
			accumulator[index] = fmaf(accumulator[index],rescale,weight * value[index]);
	}
	first = head * VALUE_DIM;
	last = first + VALUE_DIM;
	for (destination = first / width; destination < degree && destination * width < last; ++destination)
	{
		float *base = send + (uint64_t)destination * destination_stride + (uint64_t)row * record;
		const uint32_t slot_index = head - (destination * width) / VALUE_DIM;
		if ( lane == 0u )
		{
			base[width + 2u * slot_index] = running_max;
			base[width + 2u * slot_index + 1u] = running_sum;
		}
		#pragma unroll
		for (index = 0u; index < VALUES; ++index)
		{
			element = first + lane * VALUES + index;
			if ( element >= destination * width && element < (destination + 1u) * width )
				base[element - destination * width] = accumulator[index];
		}
	}
}

template<uint32_t VALUE_DIM>
__global__ __launch_bounds__(256, 1)
void LmGqaShardMergeKernel(
	const float *__restrict__ received,
	uint64_t source_stride,
	uint32_t sources,
	uint32_t heads,
	uint32_t rank,
	uint16_t *__restrict__ output_bf16,
	uint32_t rows)
{
	const uint32_t row = blockIdx.x;
	const uint32_t width = LmGqaShardWidth(heads,VALUE_DIM,sources), slots = LmGqaShardHeadSlots(heads,VALUE_DIM,sources);
	const uint32_t record = width + 2u * slots, origin = (rank * width) / VALUE_DIM;
	uint32_t element, source, slot_index;
	float global_max, source_max, scale, denominator, merged;
	const float *base;

	if ( row >= rows )
		return;
	for (element = threadIdx.x; element < width; element += blockDim.x)
	{
		slot_index = (rank * width + element) / VALUE_DIM - origin;
		global_max = -INFINITY;
		for (source = 0u; source < sources; ++source)
		{
			source_max = received[(uint64_t)source * source_stride + (uint64_t)row * record + width + 2u * slot_index];
			global_max = source_max > global_max ? source_max : global_max;
		}
		denominator = 0.0f;
		merged = 0.0f;
		for (source = 0u; source < sources; ++source)
		{
			base = received + (uint64_t)source * source_stride + (uint64_t)row * record;
			source_max = base[width + 2u * slot_index];
			scale = global_max == -INFINITY || source_max == -INFINITY ? 0.0f : __expf(source_max - global_max);
			denominator = fmaf(base[width + 2u * slot_index + 1u],scale,denominator);
			merged = fmaf(base[element],scale,merged);
		}
		output_bf16[(uint64_t)row * width + element] = LmFloatToBf16(merged / fmaxf(denominator,1.0e-20f));
	}
}

template<class Geometry, class Pages, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardPartialLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint32_t heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *send,
	uint64_t destination_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	const uint32_t degree = cache.shard.degree;
	if ( rows == 0u || query_bf16 == 0 || send == 0 || sequence_of_row == 0 || context_length == 0 ||
		LmGqaShardGeometryValid(heads,KV_HEADS,VALUE_DIM,degree) == 0u ||
		SparkKvShardValid(cache.shard,Geometry::kPageSlots) == 0u ||
		destination_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,degree) )
		return(cudaErrorInvalidValue);
	LmGqaShardPartialKernel<Geometry,Pages,KV_HEADS,HEAD_DIM,VALUE_DIM><<<dim3(rows,KV_HEADS),(heads / KV_HEADS) * 32u,0,stream>>>(
		cache,query_bf16,heads,sequence_of_row,context_length,row_position,qk_scale,send,destination_stride,rows);
	return(cudaPeekAtLastError());
}

template<uint32_t VALUE_DIM>
static inline cudaError_t LmGqaShardMergeLaunch(
	const float *received,
	uint64_t source_stride,
	uint32_t sources,
	uint32_t heads,
	uint32_t rank,
	uint16_t *output_bf16,
	uint32_t rows,
	cudaStream_t stream)
{
	if ( rows == 0u || received == 0 || output_bf16 == 0 || sources == 0u || sources > SPARK_KV_SHARD_MAX_DEGREE ||
		rank >= sources || ((uint64_t)heads * VALUE_DIM) % sources != 0u ||
		source_stride < (uint64_t)rows * LmGqaShardRecordFloats(heads,VALUE_DIM,sources) )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmGqaShardMergeKernel<VALUE_DIM>),rows,256u,0,stream,received,source_stride,sources,heads,rank,output_bf16,rows);
	return(cudaPeekAtLastError());
}
