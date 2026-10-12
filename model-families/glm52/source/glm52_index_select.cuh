#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/topk_exact.cuh"

#define GLM_INDEX_SELECT_THREADS 512u
#define GLM_INDEX_SELECT_KEEP 2048u
#define GLM_INDEX_SELECT_NONE 0xffffffffu
#define GLM_INDEX_RADIX_DONE LM_TOPK_EXACT_EMITTED
#define GLM_INDEX_RADIX_UNROLL 4u

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

struct GlmIndexScoreSource
{
	const float *scores;
	uint32_t count;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		if ( index >= count )
			return 0u;
		*key = LmTopkKey(scores[index]);
		return 1u;
	}
};

struct GlmIndexCandidateSource
{
	const uint2 *gathered;
	uint64_t rank_stride;
	uint32_t span;
	uint32_t degree;
	uint32_t bound;
	__device__ __forceinline__ uint32_t Total(void) const { return degree * span; }
	__device__ __forceinline__ uint2 Load(uint32_t index) const
	{
		return gathered[(uint64_t)(index / span) * rank_stride + index % span];
	}
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		uint2 candidate;
		if ( index >= Total() )
			return 0u;
		candidate = Load(index);
		*key = LmTopkKey(__uint_as_float(candidate.x));
		return candidate.y < bound ? 1u : 0u;
	}
};

template<uint32_t THREADS, class Source>
static __device__ void GlmIndexRadix(const Source &source, uint32_t k, uint32_t *histogram, uint32_t *state)
{
	const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES, warp = threadIdx.x / LM_TOPK_WARP_LANES;
	const uint32_t lane_buckets = LM_TOPK_BUCKETS / LM_TOPK_WARP_LANES, total = source.Total();
	uint32_t pass, shift, base, index, key, bucket, peers, need, own, suffix, other, offset, above, chosen, any;
	static_assert(LM_TOPK_BUCKETS % LM_TOPK_WARP_LANES == 0u, "every lane owns whole buckets");
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = k;
		state[GLM_INDEX_RADIX_DONE] = 0u;
	}
	for ( pass = 0u; pass < LM_TOPK_EXACT_PASSES; pass++ )
	{
		shift = 32u - LM_TOPK_RADIX_BITS * (pass + 1u);
		for ( index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS )
			histogram[index] = 0u;
		__syncthreads();
		if ( state[GLM_INDEX_RADIX_DONE] != 0u )
			break;
		for ( base = 0u; base < total; base += THREADS * GLM_INDEX_RADIX_UNROLL )
		{
			uint32_t keys[GLM_INDEX_RADIX_UNROLL], live[GLM_INDEX_RADIX_UNROLL];
			for ( index = 0u; index < GLM_INDEX_RADIX_UNROLL; index++ )
			{
				keys[index] = 0u;
				live[index] = source.Key(base + index * THREADS + threadIdx.x, &keys[index]);
			}
			for ( index = 0u; index < GLM_INDEX_RADIX_UNROLL; index++ )
			{
				key = keys[index];
				bucket = live[index] != 0u && (key & state[LM_TOPK_EXACT_MASK]) == state[LM_TOPK_EXACT_PREFIX] ? (key >> shift) & (LM_TOPK_BUCKETS - 1u) :
					LM_TOPK_BUCKETS;
				peers = __match_any_sync(0xffffffffu, bucket);
				if ( bucket < LM_TOPK_BUCKETS && lane == (uint32_t)(__ffs(peers) - 1) )
					atomicAdd(&histogram[bucket], (uint32_t)__popc(peers));
			}
		}
		__syncthreads();
		if ( warp == 0u )
		{
			need = state[LM_TOPK_EXACT_NEED];
			own = 0u;
			for ( index = 0u; index < lane_buckets; index++ )
				own += histogram[lane * lane_buckets + index];
			suffix = own;
			for ( offset = 1u; offset < LM_TOPK_WARP_LANES; offset <<= 1u )
			{
				other = __shfl_down_sync(0xffffffffu, suffix, offset);
				if ( lane + offset < LM_TOPK_WARP_LANES )
					suffix += other;
			}
			above = suffix - own;
			chosen = 0u;
			any = 0u;
			for ( index = lane_buckets; index > 0u; index-- )
			{
				bucket = lane * lane_buckets + index - 1u;
				if ( any == 0u && above + histogram[bucket] >= need )
				{
					chosen = bucket;
					any = 1u;
				}
				else if ( any == 0u )
					above += histogram[bucket];
			}
			peers = __ballot_sync(0xffffffffu, any != 0u);
			if ( lane == (peers != 0u ? 31u - (uint32_t)__clz(peers) : 0u) )
			{
				if ( any == 0u )
					above -= histogram[0];
				state[LM_TOPK_EXACT_PREFIX] |= chosen << shift;
				state[LM_TOPK_EXACT_MASK] |= (LM_TOPK_BUCKETS - 1u) << shift;
				state[LM_TOPK_EXACT_NEED] = need - above;
				state[GLM_INDEX_RADIX_DONE] = above + histogram[chosen] == need ? 1u : 0u;
			}
		}
		__syncthreads();
	}
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
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	const uint32_t row = blockIdx.x, bound = GlmIndexRowBound(sequence_of_row, context_length, row_position, row);
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	GlmIndexScoreSource source;
	uint32_t local, entry, base, key, greater, equal, threshold, mask, need, equal_before, greater_before, sum_equal, sum_greater, emitted, seen, take;
	float score;
	if ( bound <= GLM_INDEX_SELECT_KEEP )
		return;
	local = SparkKvShardLocalKeys(shard, bound);
	if ( local > score_stride )
		local = score_stride;
	if ( local <= keep )
	{
		for ( entry = threadIdx.x; entry < keep; entry += THREADS )
		{
			score = entry < local ? row_scores[entry] : -INFINITY;
			out[entry] = make_uint2(__float_as_uint(score), entry < local && score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
		}
		return;
	}
	source.scores = row_scores;
	source.count = local;
	GlmIndexRadix<THREADS>(source, keep, histogram, state);
	threshold = state[LM_TOPK_EXACT_PREFIX];
	mask = state[LM_TOPK_EXACT_MASK];
	need = state[LM_TOPK_EXACT_NEED];
	emitted = 0u;
	seen = 0u;
	for ( base = 0u; base < local; base += THREADS )
	{
		entry = base + threadIdx.x;
		key = entry < local ? LmTopkKey(row_scores[entry]) & mask : 0u;
		greater = entry < local && key > threshold ? 1u : 0u;
		equal = entry < local && key == threshold ? 1u : 0u;
		greater_before = GlmIndexExclusiveScan<THREADS>(greater, warp_totals, &sum_greater);
		equal_before = GlmIndexExclusiveScan<THREADS>(equal, warp_totals, &sum_equal);
		take = need > seen ? need - seen : 0u;
		if ( greater != 0u || (equal != 0u && equal_before < take) )
		{
			score = row_scores[entry];
			out[emitted + greater_before + (equal_before < take ? equal_before : take)] = make_uint2(__float_as_uint(score),
				score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
		}
		emitted += sum_greater + (sum_equal < take ? sum_equal : take);
		seen += sum_equal;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void GlmIndexMergeSelectKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t row_words, uint32_t *__restrict__ bitmaps, uint32_t *__restrict__ selected)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	const uint32_t row = blockIdx.x, bound = GlmIndexRowBound(sequence_of_row, context_length, row_position, row);
	uint32_t *greater = bitmaps + (uint64_t)row * 2u * row_words, *equal = greater + row_words, *out = selected + (uint64_t)row * GLM_INDEX_SELECT_KEEP;
	uint32_t words, index, key, threshold, mask, need, per, first, last, word, bits, bit, count_greater, count_equal, equal_before, take, sum, emitted, taken;
	GlmIndexCandidateSource source;
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
	source.gathered = gathered + (uint64_t)row * keep;
	source.rank_stride = rank_stride_candidates;
	source.span = SparkKvShardGatherKeys(shard, bound);
	source.span = source.span < keep ? source.span : keep;
	source.degree = shard.degree;
	source.bound = bound;
	GlmIndexRadix<THREADS>(source, GLM_INDEX_SELECT_KEEP, histogram, state);
	threshold = state[LM_TOPK_EXACT_PREFIX];
	mask = state[LM_TOPK_EXACT_MASK];
	need = state[LM_TOPK_EXACT_NEED];
	for ( index = threadIdx.x; index < source.Total(); index += THREADS * GLM_INDEX_RADIX_UNROLL )
	{
		uint2 loaded[GLM_INDEX_RADIX_UNROLL];
		for ( bit = 0u; bit < GLM_INDEX_RADIX_UNROLL; bit++ )
			loaded[bit] = index + bit * THREADS < source.Total() ? source.Load(index + bit * THREADS) : make_uint2(0u, bound);
		for ( bit = 0u; bit < GLM_INDEX_RADIX_UNROLL; bit++ )
		{
			candidate = loaded[bit];
			if ( candidate.y >= bound )
				continue;
			key = LmTopkKey(__uint_as_float(candidate.x)) & mask;
			if ( key > threshold )
				atomicOr(&greater[candidate.y / 32u], 1u << (candidate.y % 32u));
			else if ( key == threshold )
				atomicOr(&equal[candidate.y / 32u], 1u << (candidate.y % 32u));
		}
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
	emitted = GlmIndexExclusiveScan<THREADS>(count_greater + take, warp_totals, &sum);
	taken = 0u;
	for ( word = first; word < last; word++ )
	{
		bits = greater[word] | equal[word];
		while ( bits != 0u )
		{
			bit = (uint32_t)(__ffs(bits) - 1);
			bits &= bits - 1u;
			if ( (greater[word] >> bit & 1u) == 0u )
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
static inline cudaError_t GlmIndexMergeSelectLaunch(const uint2 *gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t rows, uint32_t row_words, uint32_t *bitmaps,
	uint32_t *selected, cudaStream_t stream)
{
	if ( gathered == 0 || sequence_of_row == 0 || context_length == 0 || bitmaps == 0 || selected == 0 || rows == 0u || keep == 0u || shard.degree == 0u ||
		shard.degree > SPARK_KV_SHARD_MAX_DEGREE || shard.grain == 0u || row_words == 0u || rank_stride_candidates < (uint64_t)rows * keep )
		return cudaErrorInvalidValue;
	LM_LAUNCH((GlmIndexMergeSelectKernel<THREADS>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, shard, keep, sequence_of_row, context_length,
		row_position, row_words, bitmaps, selected);
	return cudaPeekAtLastError();
}
