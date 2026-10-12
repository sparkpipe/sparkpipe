#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/topk_exact.cuh"

#define GLM_INDEX_SELECT_THREADS 512u
#define GLM_INDEX_SELECT_WARPS (GLM_INDEX_SELECT_THREADS / LM_TOPK_WARP_LANES)
#define GLM_INDEX_SELECT_KEEP 2048u
#define GLM_INDEX_SELECT_NONE 0xffffffffu

static __host__ __device__ __forceinline__ uint32_t GlmIndexBitmapWords(uint32_t positions)
{
	return (positions + 31u) / 32u;
}

static __device__ __forceinline__ uint32_t GlmIndexRowBound(const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position,
	uint32_t row)
{
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( row_position != 0 && row_position[row] + 1u < bound )
		bound = row_position[row] + 1u;
	return bound;
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint32_t GlmIndexExclusiveScan(uint32_t value, uint32_t *warp_totals, uint32_t *total)
{
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES, warp = threadIdx.x / LM_TOPK_WARP_LANES, warps = THREADS / LM_TOPK_WARP_LANES;
	uint32_t inclusive = value, other, offset, prefix = 0u, sum = 0u;
	for ( offset = 1u; offset < LM_TOPK_WARP_LANES; offset <<= 1u )
	{
		other = __shfl_up_sync(0xffffffffu, inclusive, offset);
		if ( lane >= offset )
			inclusive += other;
	}
	if ( lane == LM_TOPK_WARP_LANES - 1u )
		warp_totals[warp] = inclusive;
	__syncthreads();
	for ( offset = 0u; offset < warps; offset++ )
	{
		if ( offset < warp )
			prefix += warp_totals[offset];
		sum += warp_totals[offset];
	}
	__syncthreads();
	*total = sum;
	return prefix + inclusive - value;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GlmIndexLocalSelectKernel(SparkKvShard shard, const float *__restrict__ scores, uint32_t score_stride, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t keep, uint2 *__restrict__ candidates)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t scan[THREADS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t chosen[GLM_INDEX_SELECT_KEEP];
	const uint32_t row = blockIdx.x, bound = GlmIndexRowBound(sequence_of_row, context_length, row_position, row);
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	uint32_t local, entry, index;
	float score;
	if ( bound <= GLM_INDEX_SELECT_KEEP )
		return;
	local = SparkKvShardLocalKeys(shard, bound);
	if ( local > score_stride )
		local = score_stride;
	if ( local > keep )
	{
		LmTopkExactSelect<THREADS>(row_scores, local, keep, chosen, histogram, scan, state);
		__syncthreads();
	}
	for ( entry = threadIdx.x; entry < keep; entry += THREADS )
	{
		index = local > keep ? chosen[entry] : entry;
		score = index < local ? row_scores[index] : -INFINITY;
		out[entry] = make_uint2(__float_as_uint(score), index < local && score > -INFINITY ? SparkKvShardLocalPosition(shard, index) : LM_INDEX_SHARD_NO_POSITION);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GlmIndexMergeSelectKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t row_words, uint32_t *__restrict__ bitmaps, uint32_t *__restrict__ selected)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	const uint32_t row = blockIdx.x, bound = GlmIndexRowBound(sequence_of_row, context_length, row_position, row);
	const uint32_t total = degree * keep, lane = threadIdx.x % LM_TOPK_WARP_LANES;
	uint32_t *greater = bitmaps + (uint64_t)row * 2u * row_words, *equal = greater + row_words, *out = selected + (uint64_t)row * GLM_INDEX_SELECT_KEEP;
	uint32_t words, pass, shift, index, key, bucket, running, peers, threshold, need, per, first, last, word, bits, mask, count_greater, count_equal;
	uint32_t equal_before, take, chosen_before, sum, emitted, taken, bit, base;
	uint2 candidate;
	if ( bound <= GLM_INDEX_SELECT_KEEP )
		return;
	words = GlmIndexBitmapWords(bound);
	if ( words > row_words )
		words = row_words;
	for ( index = threadIdx.x; index < words; index += THREADS )
	{
		greater[index] = 0u;
		equal[index] = 0u;
	}
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = GLM_INDEX_SELECT_KEEP;
	}
	for ( pass = 0u; pass < LM_TOPK_EXACT_PASSES; pass++ )
	{
		shift = 32u - LM_TOPK_RADIX_BITS * (pass + 1u);
		for ( index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS )
			histogram[index] = 0u;
		__syncthreads();
		for ( base = 0u; base < total; base += THREADS )
		{
			index = base + threadIdx.x;
			candidate = index < total ? gathered[(uint64_t)(index / keep) * rank_stride_candidates + (uint64_t)row * keep + index % keep] : make_uint2(0u, bound);
			key = LmTopkKey(__uint_as_float(candidate.x));
			bucket = candidate.y < bound && (key & state[LM_TOPK_EXACT_MASK]) == state[LM_TOPK_EXACT_PREFIX] ? (key >> shift) & (LM_TOPK_BUCKETS - 1u) : LM_TOPK_BUCKETS;
			peers = __match_any_sync(0xffffffffu, bucket);
			if ( bucket < LM_TOPK_BUCKETS && lane == (uint32_t)(__ffs(peers) - 1) )
				atomicAdd(&histogram[bucket], (uint32_t)__popc(peers));
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			running = 0u;
			for ( bucket = LM_TOPK_BUCKETS - 1u; bucket > 0u && running + histogram[bucket] < state[LM_TOPK_EXACT_NEED]; --bucket )
				running += histogram[bucket];
			state[LM_TOPK_EXACT_PREFIX] |= bucket << shift;
			state[LM_TOPK_EXACT_MASK] |= (LM_TOPK_BUCKETS - 1u) << shift;
			state[LM_TOPK_EXACT_NEED] -= running;
		}
		__syncthreads();
	}
	threshold = state[LM_TOPK_EXACT_PREFIX];
	need = state[LM_TOPK_EXACT_NEED];
	for ( index = threadIdx.x; index < total; index += THREADS )
	{
		candidate = gathered[(uint64_t)(index / keep) * rank_stride_candidates + (uint64_t)row * keep + index % keep];
		if ( candidate.y >= bound )
			continue;
		key = LmTopkKey(__uint_as_float(candidate.x));
		if ( key > threshold )
			atomicOr(&greater[candidate.y / 32u], 1u << (candidate.y % 32u));
		else if ( key == threshold )
			atomicOr(&equal[candidate.y / 32u], 1u << (candidate.y % 32u));
	}
	__syncthreads();
	per = (words + THREADS - 1u) / THREADS;
	first = threadIdx.x * per < words ? threadIdx.x * per : words;
	last = first + per < words ? first + per : words;
	count_greater = 0u;
	count_equal = 0u;
	for ( word = first; word < last; word++ )
	{
		count_greater += (uint32_t)__popc(greater[word]);
		count_equal += (uint32_t)__popc(equal[word]);
	}
	equal_before = GlmIndexExclusiveScan<THREADS>(count_equal, warp_totals, &sum);
	take = need > equal_before ? need - equal_before : 0u;
	take = take < count_equal ? take : count_equal;
	chosen_before = GlmIndexExclusiveScan<THREADS>(count_greater + take, warp_totals, &sum);
	emitted = chosen_before;
	taken = 0u;
	for ( word = first; word < last; word++ )
	{
		bits = greater[word] | equal[word];
		while ( bits != 0u )
		{
			bit = (uint32_t)(__ffs(bits) - 1);
			mask = 1u << bit;
			bits &= ~mask;
			if ( (greater[word] & mask) == 0u )
			{
				if ( taken >= take )
					continue;
				taken++;
			}
			if ( emitted < GLM_INDEX_SELECT_KEEP )
				out[emitted] = word * 32u + bit;
			emitted++;
		}
	}
	for ( index = sum + threadIdx.x; index < GLM_INDEX_SELECT_KEEP; index += THREADS )
		out[index] = GLM_INDEX_SELECT_NONE;
}

template<uint32_t THREADS>
static inline cudaError_t GlmIndexLocalSelectLaunch(SparkKvShard shard, const float *scores, uint32_t score_stride, const uint32_t *sequence_of_row,
	const uint32_t *context_length, const uint32_t *row_position, uint32_t keep, uint32_t rows, uint2 *candidates, cudaStream_t stream)
{
	if ( scores == 0 || sequence_of_row == 0 || context_length == 0 || candidates == 0 || rows == 0u || keep == 0u || keep > GLM_INDEX_SELECT_KEEP ||
		score_stride == 0u || shard.degree == 0u || shard.grain == 0u )
		return cudaErrorInvalidValue;
	LM_LAUNCH((GlmIndexLocalSelectKernel<THREADS>), rows, THREADS, 0, stream, shard, scores, score_stride, sequence_of_row, context_length, row_position,
		keep, candidates);
	return cudaPeekAtLastError();
}

template<uint32_t THREADS>
static inline cudaError_t GlmIndexMergeSelectLaunch(const uint2 *gathered, uint64_t rank_stride_candidates, uint32_t degree, uint32_t keep,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t rows, uint32_t row_words, uint32_t *bitmaps,
	uint32_t *selected, cudaStream_t stream)
{
	if ( gathered == 0 || sequence_of_row == 0 || context_length == 0 || bitmaps == 0 || selected == 0 || rows == 0u || keep == 0u || degree == 0u ||
		degree > SPARK_KV_SHARD_MAX_DEGREE || row_words == 0u || rank_stride_candidates < (uint64_t)rows * keep )
		return cudaErrorInvalidValue;
	LM_LAUNCH((GlmIndexMergeSelectKernel<THREADS>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, degree, keep, sequence_of_row, context_length,
		row_position, row_words, bitmaps, selected);
	return cudaPeekAtLastError();
}
