#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_INDEX_SELECT_THREADS 512u
#define LM_INDEX_SELECT_NONE 0xffffffffu
#define LM_INDEX_SELECT_RADIX_DONE LM_TOPK_EXACT_EMITTED
#define LM_INDEX_SELECT_UNROLL 4u
#define LM_INDEX_SELECT_CACHED_KEYS 8192u
#define LM_INDEX_SELECT_SAMPLED_MIN_KEYS 16384u
#define LM_INDEX_SELECT_SAMPLED_SPACING 64u
#define LM_INDEX_SELECT_SAMPLED_SAMPLES 16384u
#define LM_INDEX_SELECT_SAMPLED_LIST 8192u
#define LM_INDEX_SELECT_SAMPLED_BUCKETS 2048u
#define LM_INDEX_SELECT_SAMPLED_MAX_KEEP 2048u
#define LM_INDEX_SELECT_SAMPLED_MARGIN 5.0f
#define LM_INDEX_SELECT_SAMPLED_THREADS 1024u
#define LM_INDEX_SELECT_SAMPLED_UNROLL 8u
#define LM_INDEX_SELECT_MERGE_MAX_K 2048u
#define LM_INDEX_SELECT_MERGE_LIST 4096u
#define LM_INDEX_SELECT_MERGE_SAMPLES 1024u
#define LM_INDEX_SELECT_MERGE_MARGIN 3.5f

static __host__ __device__ __forceinline__ uint32_t LmIndexSelectBitmapWords(uint32_t positions)
{
	return (positions + 31u) / 32u;
}

static __device__ __forceinline__ uint32_t LmIndexSelectRowBound(const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position,
	uint32_t row)
{
	uint32_t bound = __ldg(context_length + __ldg(sequence_of_row + row));
	if ( row_position != 0 )
	{
		const uint32_t pos = __ldg(row_position + row);
		if ( pos + 1u < bound )
			bound = pos + 1u;
	}
	return bound;
}

struct LmIndexSelectScoreSource
{
	const float *scores;
	uint32_t count;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		if ( index >= count )
			return 0u;
		*key = LmTopkKey(__ldg(scores + index));
		return 1u;
	}
};

struct LmIndexSelectSampleSource
{
	const uint32_t *samples;
	uint32_t count;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		if ( index >= count )
			return 0u;
		*key = samples[index];
		return 1u;
	}
};

struct LmIndexSelectListSource
{
	const unsigned long long *list;
	uint32_t count;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		if ( index >= count )
			return 0u;
		*key = (uint32_t)(list[index] >> 32u);
		return 1u;
	}
};

struct LmIndexSelectTiedSource
{
	const unsigned long long *list;
	uint32_t count;
	uint32_t threshold;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		unsigned long long entry;
		if ( index >= count )
			return 0u;
		entry = list[index];
		*key = (uint32_t)entry;
		return (uint32_t)(entry >> 32u) == threshold ? 1u : 0u;
	}
};

struct LmIndexSelectCandidateSource
{
	const uint2 *gathered;
	uint64_t rank_stride;
	uint32_t span;
	uint32_t degree;
	uint32_t bound;
	__device__ __forceinline__ uint32_t Total(void) const { return degree * span; }
	__device__ __forceinline__ uint2 Load(uint32_t index) const
	{
		return __ldg(gathered + (uint64_t)(index / span) * rank_stride + index % span);
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

template<uint32_t THREADS>
static __device__ __forceinline__ uint32_t LmIndexSelectExclusiveScan(uint32_t value, uint32_t *warp_totals, uint32_t *total)
{
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u, warps = THREADS / 32u;
	uint32_t inclusive = value, other, offset, prefix = 0u, sum = 0u;
	#pragma unroll
	for ( offset = 1u; offset < 32u; offset <<= 1u )
	{
		other = __shfl_up_sync(0xffffffffu, inclusive, offset);
		if ( lane >= offset )
			inclusive += other;
	}
	if ( lane == 31u )
		warp_totals[warp] = inclusive;
	__syncthreads();
	#pragma unroll
	for ( offset = 0u; offset < warps; offset++ )
	{
		const uint32_t w = warp_totals[offset];
		if ( offset < warp )
			prefix += w;
		sum += w;
	}
	__syncthreads();
	*total = sum;
	return prefix + inclusive - value;
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmIndexSelectRadix11Pick(
	const uint32_t *histogram, uint32_t shift, uint32_t field, uint32_t *warp_totals, uint32_t *state)
{
	constexpr uint32_t BUCKETS = LM_INDEX_SELECT_SAMPLED_BUCKETS;
	constexpr uint32_t PER = BUCKETS / THREADS;
	static_assert(BUCKETS % THREADS == 0u, "every thread owns whole 11-bit buckets");
	const uint32_t need = state[LM_TOPK_EXACT_NEED];
	uint32_t own = 0u, total = 0u, local_h[PER];
	#pragma unroll
	for ( uint32_t i = 0u; i < PER; i++ )
	{
		local_h[i] = histogram[BUCKETS - 1u - (threadIdx.x * PER + i)];
		own += local_h[i];
	}
	uint32_t above = LmIndexSelectExclusiveScan<THREADS>(own, warp_totals, &total);
	if ( total <= need )
	{
		if ( threadIdx.x == 0u )
			state[LM_INDEX_SELECT_RADIX_DONE] = 1u;
		__syncthreads();
		return;
	}
	#pragma unroll
	for ( uint32_t i = 0u; i < PER; i++ )
	{
		const uint32_t b = BUCKETS - 1u - (threadIdx.x * PER + i);
		const uint32_t c = local_h[i];
		if ( above < need && above + c >= need )
		{
			state[LM_TOPK_EXACT_PREFIX] |= b << shift;
			state[LM_TOPK_EXACT_MASK] |= field << shift;
			state[LM_TOPK_EXACT_NEED] = need - above;
			state[LM_INDEX_SELECT_RADIX_DONE] = above + c == need ? 1u : 0u;
		}
		above += c;
	}
	__syncthreads();
}

template<uint32_t THREADS, class Source>
static __device__ void LmIndexSelectRadix11(const Source &source, uint32_t k, uint32_t *histogram, uint32_t *state, uint32_t *warp_totals)
{
	constexpr uint32_t BUCKETS = LM_INDEX_SELECT_SAMPLED_BUCKETS;
	const uint32_t shifts[3] = { 21u, 10u, 0u };
	const uint32_t fields[3] = { 2047u, 2047u, 1023u };
	const uint32_t total = source.Total();
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = k;
		state[LM_INDEX_SELECT_RADIX_DONE] = 0u;
	}
	__syncthreads();
	for ( uint32_t pass = 0u; pass < 3u; pass++ )
	{
		if ( state[LM_INDEX_SELECT_RADIX_DONE] != 0u )
			break;
		const uint32_t shift = shifts[pass], field = fields[pass];
		const uint32_t prefix = state[LM_TOPK_EXACT_PREFIX], mask = state[LM_TOPK_EXACT_MASK];
		for ( uint32_t i = threadIdx.x; i < BUCKETS; i += THREADS )
			histogram[i] = 0u;
		__syncthreads();
		for ( uint32_t base = 0u; base < total; base += THREADS * 2u )
		{
			uint32_t k0 = 0u, k1 = 0u;
			const uint32_t l0 = source.Key(base + threadIdx.x, &k0);
			const uint32_t l1 = source.Key(base + THREADS + threadIdx.x, &k1);
			if ( l0 != 0u && (k0 & mask) == prefix )
				atomicAdd(&histogram[(k0 >> shift) & field], 1u);
			if ( l1 != 0u && (k1 & mask) == prefix )
				atomicAdd(&histogram[(k1 >> shift) & field], 1u);
		}
		__syncthreads();
		LmIndexSelectRadix11Pick<THREADS>(histogram, shift, field, warp_totals, state);
	}
}

template<uint32_t THREADS, class Source>
static __device__ void LmIndexSelectRadix(const Source &source, uint32_t k, uint32_t *histogram, uint32_t *state)
{
	const uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	const uint32_t lane_buckets = LM_TOPK_BUCKETS / LM_TOPK_WARP_LANES, total = source.Total();
	uint32_t pass, shift, base, index, key, bucket, peers, need, own, suffix, other, offset, above, chosen, any;
	static_assert(LM_TOPK_BUCKETS % LM_TOPK_WARP_LANES == 0u, "every lane owns whole buckets");
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = k;
		state[LM_INDEX_SELECT_RADIX_DONE] = 0u;
	}
	for ( pass = 0u; pass < LM_TOPK_EXACT_PASSES; pass++ )
	{
		shift = 32u - LM_TOPK_RADIX_BITS * (pass + 1u);
		for ( index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS )
			histogram[index] = 0u;
		__syncthreads();
		if ( state[LM_INDEX_SELECT_RADIX_DONE] != 0u )
			break;
		const uint32_t cur_mask = state[LM_TOPK_EXACT_MASK], cur_prefix = state[LM_TOPK_EXACT_PREFIX];
		for ( base = 0u; base < total; base += THREADS * LM_INDEX_SELECT_UNROLL )
		{
			uint32_t keys[LM_INDEX_SELECT_UNROLL], live[LM_INDEX_SELECT_UNROLL];
			#pragma unroll
			for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
			{
				keys[index] = 0u;
				live[index] = source.Key(base + index * THREADS + threadIdx.x, &keys[index]);
			}
			#pragma unroll
			for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
			{
				key = keys[index];
				bucket = live[index] != 0u && (key & cur_mask) == cur_prefix ? (key >> shift) & (LM_TOPK_BUCKETS - 1u) :
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
				state[LM_INDEX_SELECT_RADIX_DONE] = above + histogram[chosen] == need ? 1u : 0u;
			}
		}
		__syncthreads();
	}
}

static __device__ __forceinline__ uint2 LmIndexSelectSampledCandidate(SparkKvShard shard, float score, uint32_t entry)
{
	return make_uint2(__float_as_uint(score), score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
}

template<uint32_t THREADS>
static __device__ void LmIndexSelectCachedLocalRow(
	SparkKvShard shard, const float *row_scores, uint32_t local, uint32_t keep, uint2 *out,
	uint32_t *keys, uint32_t *histogram, uint32_t *state, uint32_t *warp_totals)
{
	constexpr uint32_t BUCKETS = LM_INDEX_SELECT_SAMPLED_BUCKETS;
	const uint32_t lane = threadIdx.x & 31u;
	if ( threadIdx.x == 0u )
	{
		state[LM_TOPK_EXACT_PREFIX] = 0u;
		state[LM_TOPK_EXACT_MASK] = 0u;
		state[LM_TOPK_EXACT_NEED] = keep;
		state[LM_INDEX_SELECT_RADIX_DONE] = 0u;
		state[LM_TOPK_EXACT_SEEN] = 0u;
	}
	for ( uint32_t i = threadIdx.x; i < BUCKETS; i += THREADS )
		histogram[i] = 0u;
	__syncthreads();
	if ( ((uintptr_t)row_scores & 15u) == 0u )
	{
		const uint32_t vecs = local >> 2u;
		const float4 *scores4 = (const float4 *)row_scores;
		for ( uint32_t v = threadIdx.x; v < vecs; v += THREADS )
		{
			const float4 s4 = __ldg(scores4 + v);
			const uint32_t key0 = LmTopkKey(s4.x), key1 = LmTopkKey(s4.y), key2 = LmTopkKey(s4.z), key3 = LmTopkKey(s4.w);
			const uint32_t base = v << 2u;
			keys[base] = key0; keys[base + 1u] = key1; keys[base + 2u] = key2; keys[base + 3u] = key3;
			atomicAdd(&histogram[key0 >> 21u], 1u);
			atomicAdd(&histogram[key1 >> 21u], 1u);
			atomicAdd(&histogram[key2 >> 21u], 1u);
			atomicAdd(&histogram[key3 >> 21u], 1u);
		}
		for ( uint32_t e = (vecs << 2u) + threadIdx.x; e < local; e += THREADS )
		{
			const uint32_t k0 = LmTopkKey(__ldg(row_scores + e));
			keys[e] = k0;
			atomicAdd(&histogram[k0 >> 21u], 1u);
		}
	}
	else
	{
		for ( uint32_t e = threadIdx.x; e < local; e += THREADS )
		{
			const uint32_t k0 = LmTopkKey(__ldg(row_scores + e));
			keys[e] = k0;
			atomicAdd(&histogram[k0 >> 21u], 1u);
		}
	}
	__syncthreads();
	LmIndexSelectRadix11Pick<THREADS>(histogram, 21u, 2047u, warp_totals, state);
	const uint32_t shifts[2] = { 10u, 0u };
	const uint32_t fields[2] = { 2047u, 1023u };
	for ( uint32_t pass = 0u; pass < 2u; pass++ )
	{
		if ( state[LM_INDEX_SELECT_RADIX_DONE] != 0u )
			break;
		const uint32_t shift = shifts[pass], field = fields[pass];
		const uint32_t prefix = state[LM_TOPK_EXACT_PREFIX], mask = state[LM_TOPK_EXACT_MASK];
		for ( uint32_t i = threadIdx.x; i < BUCKETS; i += THREADS )
			histogram[i] = 0u;
		__syncthreads();
		for ( uint32_t e = threadIdx.x; e < local; e += THREADS )
		{
			const uint32_t k0 = keys[e];
			if ( (k0 & mask) == prefix )
				atomicAdd(&histogram[(k0 >> shift) & field], 1u);
		}
		__syncthreads();
		LmIndexSelectRadix11Pick<THREADS>(histogram, shift, field, warp_totals, state);
	}
	const uint32_t threshold = state[LM_TOPK_EXACT_PREFIX];
	const uint32_t mask = state[LM_TOPK_EXACT_MASK];
	const uint32_t need = state[LM_TOPK_EXACT_NEED];
	const uint32_t done = state[LM_INDEX_SELECT_RADIX_DONE];
	if ( done != 0u )
	{
		for ( uint32_t base = 0u; base < local; base += THREADS )
		{
			const uint32_t e = base + threadIdx.x;
			const uint32_t k0 = e < local ? keys[e] : 0u;
			const bool keep_it = e < local && (k0 & mask) >= threshold;
			const uint32_t ballot = __ballot_sync(0xffffffffu, keep_it);
			if ( ballot != 0u )
			{
				const uint32_t leader = (uint32_t)(__ffs(ballot) - 1);
				uint32_t slot = lane == leader ? atomicAdd(&state[LM_TOPK_EXACT_SEEN], (uint32_t)__popc(ballot)) : 0u;
				slot = __shfl_sync(0xffffffffu, slot, leader) + (uint32_t)__popc(ballot & ((1u << lane) - 1u));
				if ( keep_it && slot < keep )
					out[slot] = LmIndexSelectSampledCandidate(shard, LmTopkValue(k0), e);
			}
		}
		return;
	}
	uint32_t emitted = 0u, seen = 0u;
	for ( uint32_t base = 0u; base < local; base += THREADS )
	{
		const uint32_t e = base + threadIdx.x;
		const uint32_t k0 = e < local ? keys[e] : 0u;
		const uint32_t km = k0 & mask;
		const uint32_t greater = e < local && km > threshold ? 1u : 0u;
		const uint32_t equal = e < local && km == threshold ? 1u : 0u;
		uint32_t packed_total = 0u;
		const uint32_t packed_before = LmIndexSelectExclusiveScan<THREADS>((greater << 16u) | equal, warp_totals, &packed_total);
		const uint32_t greater_before = packed_before >> 16u, equal_before = packed_before & 0xffffu;
		const uint32_t sum_greater = packed_total >> 16u, sum_equal = packed_total & 0xffffu;
		const uint32_t take = need > seen ? need - seen : 0u;
		if ( greater != 0u || (equal != 0u && equal_before < take) )
			out[emitted + greater_before + (equal_before < take ? equal_before : take)] =
				LmIndexSelectSampledCandidate(shard, LmTopkValue(k0), e);
		emitted += sum_greater + (sum_equal < take ? sum_equal : take);
		seen += sum_equal;
	}
}

template<uint32_t THREADS>
static __device__ void LmIndexSelectLocalRadixRow(SparkKvShard shard, const float *row_scores, uint32_t local, uint32_t keep, uint2 *out, uint32_t *histogram,
	uint32_t *state, uint32_t *warp_totals)
{
	LmIndexSelectScoreSource source;
	uint32_t entry, base, key, greater, equal, threshold, mask, need, emitted, seen, take;
	float score;
	source.scores = row_scores;
	source.count = local;
	LmIndexSelectRadix<THREADS>(source, keep, histogram, state);
	threshold = state[LM_TOPK_EXACT_PREFIX];
	mask = state[LM_TOPK_EXACT_MASK];
	need = state[LM_TOPK_EXACT_NEED];
	emitted = 0u;
	seen = 0u;
	for ( base = 0u; base < local; base += THREADS )
	{
		entry = base + threadIdx.x;
		score = entry < local ? __ldg(row_scores + entry) : 0.0f;
		key = entry < local ? LmTopkKey(score) & mask : 0u;
		greater = entry < local && key > threshold ? 1u : 0u;
		equal = entry < local && key == threshold ? 1u : 0u;
		uint32_t packed_total = 0u;
		const uint32_t packed_before = LmIndexSelectExclusiveScan<THREADS>((greater << 16u) | equal, warp_totals, &packed_total);
		const uint32_t greater_before = packed_before >> 16u, equal_before = packed_before & 0xffffu;
		const uint32_t sum_greater = packed_total >> 16u, sum_equal = packed_total & 0xffffu;
		take = need > seen ? need - seen : 0u;
		if ( greater != 0u || (equal != 0u && equal_before < take) )
			out[emitted + greater_before + (equal_before < take ? equal_before : take)] = LmIndexSelectSampledCandidate(shard, score, entry);
		emitted += sum_greater + (sum_equal < take ? sum_equal : take);
		seen += sum_equal;
	}
}

template<uint32_t THREADS>
static __device__ uint32_t LmIndexSelectLocalWholeRow(SparkKvShard shard, const float *row_scores, uint32_t score_stride, const uint32_t *sequence_of_row,
	uint32_t row, uint32_t local, uint32_t keep, uint2 *out, LmFrameError *error)
{
	uint32_t entry;
	float score;
	if ( local > score_stride )
	{
		if ( threadIdx.x == 0u )
			LmFrameErrorReport(error, LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, 0u, row, sequence_of_row[row], local, score_stride);
		for ( entry = threadIdx.x; entry < keep; entry += THREADS )
			out[entry] = make_uint2(__float_as_uint(-INFINITY), LM_INDEX_SHARD_NO_POSITION);
		return 1u;
	}
	if ( local > keep )
		return 0u;
	for ( entry = threadIdx.x; entry < keep; entry += THREADS )
	{
		score = entry < local ? __ldg(row_scores + entry) : -INFINITY;
		out[entry] = make_uint2(__float_as_uint(score), entry < local && score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
	}
	return 1u;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexSelectLocalKernel(SparkKvShard shard, const float *__restrict__ scores, uint32_t score_stride, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t keep, uint32_t dense_limit, uint2 *__restrict__ candidates,
	LmFrameError *__restrict__ error)
{
	__shared__ uint32_t keys[LM_INDEX_SELECT_CACHED_KEYS];
	__shared__ uint32_t histogram[LM_INDEX_SELECT_SAMPLED_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	LmDependentWait();
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row);
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	uint32_t local;
	if ( bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	local = SparkKvShardLocalKeys(shard, bound);
	if ( LmIndexSelectLocalWholeRow<THREADS>(shard, row_scores, score_stride, sequence_of_row, row, local, keep, out, error) != 0u )
	{
		LmDependentRelease();
		return;
	}
	if ( local <= LM_INDEX_SELECT_CACHED_KEYS )
		LmIndexSelectCachedLocalRow<THREADS>(shard, row_scores, local, keep, out, keys, histogram, state, warp_totals);
	else
		LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, histogram, state, warp_totals);
	LmDependentRelease();
}

#ifdef __CUDACC__
template<uint32_t THREADS>
struct LmIndexSelectSampledShared
{
	union
	{
		unsigned long long list[LM_INDEX_SELECT_SAMPLED_LIST];
		uint32_t samples[LM_INDEX_SELECT_SAMPLED_SAMPLES];
	};
	uint32_t histogram[LM_INDEX_SELECT_SAMPLED_BUCKETS];
	uint32_t state[LM_TOPK_EXACT_STATE];
	uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	uint32_t top;
	uint32_t count;
	uint32_t emitted;
	uint32_t bucket;
	uint32_t above;
};

static __device__ __forceinline__ uint32_t LmIndexSelectSampledBucket(uint32_t key, uint32_t low, uint32_t shift)
{
	const uint32_t bucket = (key - low) >> shift;
	return bucket < LM_INDEX_SELECT_SAMPLED_BUCKETS ? bucket : LM_INDEX_SELECT_SAMPLED_BUCKETS - 1u;
}

static __device__ __forceinline__ unsigned long long LmIndexSelectSampledEntry(uint32_t key, uint32_t entry)
{
	return ((unsigned long long)key << 32u) | (unsigned long long)(0xffffffffu - entry);
}

template<uint32_t THREADS>
static __device__ void LmIndexSelectSampledFinish(SparkKvShard shard, LmIndexSelectSampledShared<THREADS> &s, uint32_t count, uint32_t need,
	uint2 *out)
{
	LmIndexSelectListSource source;
	LmIndexSelectTiedSource tied;
	const uint32_t lane = threadIdx.x & 31u;
	uint32_t threshold, mask, done, tie_need, tie_threshold, tie_mask;
	source.list = s.list;
	source.count = count;
	LmIndexSelectRadix11<THREADS>(source, need, s.histogram, s.state, s.warp_totals);
	threshold = s.state[LM_TOPK_EXACT_PREFIX];
	mask = s.state[LM_TOPK_EXACT_MASK];
	done = s.state[LM_INDEX_SELECT_RADIX_DONE];
	tie_need = s.state[LM_TOPK_EXACT_NEED];
	tie_threshold = 0u;
	tie_mask = 0u;
	if ( done == 0u )
	{
		__syncthreads();
		tied.list = s.list;
		tied.count = count;
		tied.threshold = threshold;
		LmIndexSelectRadix11<THREADS>(tied, tie_need, s.histogram, s.state, s.warp_totals);
		tie_threshold = s.state[LM_TOPK_EXACT_PREFIX];
		tie_mask = s.state[LM_TOPK_EXACT_MASK];
	}
	for ( uint32_t base = 0u; base < count; base += THREADS )
	{
		const uint32_t index = base + threadIdx.x;
		unsigned long long value = 0ull;
		bool take = false;
		if ( index < count )
		{
			value = s.list[index];
			const uint32_t km = (uint32_t)(value >> 32u) & mask;
			take = km > threshold || (km == threshold && (done != 0u || ((uint32_t)value & tie_mask) >= tie_threshold));
		}
		const uint32_t ballot = __ballot_sync(0xffffffffu, take);
		if ( ballot != 0u )
		{
			const uint32_t leader = (uint32_t)(__ffs(ballot) - 1);
			uint32_t slot = lane == leader ? atomicAdd(&s.emitted, (uint32_t)__popc(ballot)) : 0u;
			slot = __shfl_sync(0xffffffffu, slot, leader) + (uint32_t)__popc(ballot & ((1u << lane) - 1u));
			if ( take )
			{
				const uint32_t entry = 0xffffffffu - (uint32_t)value;
				out[slot] = LmIndexSelectSampledCandidate(shard, LmTopkValue((uint32_t)(value >> 32u)), entry);
			}
		}
	}
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmIndexSelectSampledAccept(
	LmIndexSelectSampledShared<THREADS> &s, uint32_t key, uint32_t entry, bool inside, uint32_t low, uint32_t shift, uint32_t lane)
{
	const uint32_t mask = __ballot_sync(0xffffffffu, inside);
	if ( mask == 0u )
		return;
	if ( inside )
		atomicAdd(&s.histogram[LmIndexSelectSampledBucket(key, low, shift)], 1u);
	const uint32_t leader = (uint32_t)(__ffs(mask) - 1);
	uint32_t first = lane == leader ? atomicAdd(&s.count, (uint32_t)__popc(mask)) : 0u;
	first = __shfl_sync(0xffffffffu, first, leader);
	const uint32_t slot = first + (uint32_t)__popc(mask & ((1u << lane) - 1u));
	if ( inside && slot < LM_INDEX_SELECT_SAMPLED_LIST )
		s.list[slot] = LmIndexSelectSampledEntry(key, entry);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexSelectLocalSampledKernel(SparkKvShard shard, const float *__restrict__ scores, uint32_t score_stride, const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t keep, uint32_t dense_limit, uint2 *__restrict__ candidates,
	LmFrameError *__restrict__ error)
{
	constexpr uint32_t BUCKETS = LM_INDEX_SELECT_SAMPLED_BUCKETS, LIST = LM_INDEX_SELECT_SAMPLED_LIST, PER_THREAD = BUCKETS / THREADS;
	constexpr uint32_t UNROLL = LM_INDEX_SELECT_SAMPLED_UNROLL;
	static_assert(BUCKETS % THREADS == 0u && BUCKETS >= LM_TOPK_BUCKETS, "every thread owns whole buckets and the radix fits the histogram");
	static_assert(LM_INDEX_SELECT_SAMPLED_SAMPLES * sizeof(uint32_t) <= LIST * sizeof(unsigned long long), "the samples fit the list");
	extern __shared__ __align__(16) unsigned char lm_index_select_sampled_bytes[];
	LmIndexSelectSampledShared<THREADS> &s = *reinterpret_cast<LmIndexSelectSampledShared<THREADS> *>(lm_index_select_sampled_bytes);
	LmDependentWait();
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row), lane = threadIdx.x & 31u;
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	LmIndexSelectSampleSource sampled;
	uint32_t local, entry, index, count, step, base, key, slot, low, shift, span, total, above, own, rank, peak;
	if ( bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	local = SparkKvShardLocalKeys(shard, bound);
	if ( LmIndexSelectLocalWholeRow<THREADS>(shard, row_scores, score_stride, sequence_of_row, row, local, keep, out, error) != 0u )
	{
		LmDependentRelease();
		return;
	}
	if ( local < LM_INDEX_SELECT_SAMPLED_MIN_KEYS )
	{
		LmIndexSelectCachedLocalRow<THREADS>(shard, row_scores, local, keep, out, s.samples, s.histogram, s.state, s.warp_totals);
		LmDependentRelease();
		return;
	}
	count = local / LM_INDEX_SELECT_SAMPLED_SPACING;
	count = count < LM_INDEX_SELECT_SAMPLED_SAMPLES ? count : LM_INDEX_SELECT_SAMPLED_SAMPLES;
	step = local / count;
	if ( threadIdx.x == 0u )
	{
		s.top = 0u;
		s.count = 0u;
		s.emitted = 0u;
	}
	peak = 0u;
	for ( index = threadIdx.x; index < count; index += THREADS )
	{
		key = LmTopkKey(__ldg(row_scores + (uint64_t)index * step + ((index * 2654435761u) >> 16u) % step));
		s.samples[index] = key;
		peak = key > peak ? key : peak;
	}
	peak = __reduce_max_sync(0xffffffffu, peak);
	__syncthreads();
	if ( lane == 0u )
		atomicMax(&s.top, peak);
	{
		const float expected = (float)keep * (float)count / (float)local, half = 0.5f * LM_INDEX_SELECT_SAMPLED_MARGIN;
		const float root = half + sqrtf(half * half + expected);
		rank = (uint32_t)ceilf(root * root);
		rank = rank < count ? rank : count;
	}
	sampled.samples = s.samples;
	sampled.count = count;
	LmIndexSelectRadix11<THREADS>(sampled, rank, s.histogram, s.state, s.warp_totals);
	low = s.state[LM_TOPK_EXACT_PREFIX];
	span = s.top - low;
	for ( shift = 0u; (span >> shift) >= BUCKETS; shift++ )
		;
	__syncthreads();
	for ( index = threadIdx.x; index < BUCKETS; index += THREADS )
		s.histogram[index] = 0u;
	__syncthreads();
	if ( ((uintptr_t)row_scores & 15u) == 0u )
	{
		const uint32_t vecs = local >> 2u;
		const float4 *scores4 = (const float4 *)row_scores;
		for ( uint32_t vbase = 0u; vbase < vecs; vbase += THREADS * 2u )
		{
			const uint32_t v0 = vbase + threadIdx.x, v1 = v0 + THREADS;
			const float4 a = v0 < vecs ? __ldg(scores4 + v0) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			const float4 b = v1 < vecs ? __ldg(scores4 + v1) : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
			const uint32_t key0 = LmTopkKey(a.x), key1 = LmTopkKey(a.y), key2 = LmTopkKey(a.z), key3 = LmTopkKey(a.w);
			const bool m0 = v0 < vecs;
			if ( __any_sync(0xffffffffu, m0 && (key0 >= low || key1 >= low || key2 >= low || key3 >= low)) )
			{
				const uint32_t e0 = v0 << 2u;
				LmIndexSelectSampledAccept<THREADS>(s, key0, e0, m0 && key0 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, key1, e0 + 1u, m0 && key1 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, key2, e0 + 2u, m0 && key2 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, key3, e0 + 3u, m0 && key3 >= low, low, shift, lane);
			}
			const uint32_t k4 = LmTopkKey(b.x), k5 = LmTopkKey(b.y), k6 = LmTopkKey(b.z), k7 = LmTopkKey(b.w);
			const bool m1 = v1 < vecs;
			if ( __any_sync(0xffffffffu, m1 && (k4 >= low || k5 >= low || k6 >= low || k7 >= low)) )
			{
				const uint32_t e1 = v1 << 2u;
				LmIndexSelectSampledAccept<THREADS>(s, k4, e1, m1 && k4 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, k5, e1 + 1u, m1 && k5 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, k6, e1 + 2u, m1 && k6 >= low, low, shift, lane);
				LmIndexSelectSampledAccept<THREADS>(s, k7, e1 + 3u, m1 && k7 >= low, low, shift, lane);
			}
		}
		if ( (vecs << 2u) < local )
		{
			entry = (vecs << 2u) + threadIdx.x;
			const bool inside = entry < local;
			key = inside ? LmTopkKey(__ldg(row_scores + entry)) : 0u;
			LmIndexSelectSampledAccept<THREADS>(s, key, entry, inside && key >= low, low, shift, lane);
		}
	}
	else
	{
		for ( base = 0u; base < local; base += THREADS * UNROLL )
		{
			float loaded[UNROLL];
			#pragma unroll
			for ( index = 0u; index < UNROLL; index++ )
			{
				entry = base + index * THREADS + threadIdx.x;
				loaded[index] = entry < local ? __ldg(row_scores + entry) : 0.0f;
			}
			#pragma unroll
			for ( index = 0u; index < UNROLL; index++ )
			{
				entry = base + index * THREADS + threadIdx.x;
				key = LmTopkKey(loaded[index]);
				LmIndexSelectSampledAccept<THREADS>(s, key, entry, entry < local && key >= low, low, shift, lane);
			}
		}
	}
	__syncthreads();
	total = s.count;
	if ( total < keep )
	{
		LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, s.histogram, s.state, s.warp_totals);
		LmDependentRelease();
		return;
	}
	if ( total <= LIST )
	{
		LmIndexSelectSampledFinish<THREADS>(shard, s, total, keep, out);
		LmDependentRelease();
		return;
	}
	own = 0u;
	for ( index = 0u; index < PER_THREAD; index++ )
		own += s.histogram[BUCKETS - 1u - (threadIdx.x * PER_THREAD + index)];
	above = LmIndexSelectExclusiveScan<THREADS>(own, s.warp_totals, &total);
	for ( index = 0u; index < PER_THREAD; index++ )
	{
		const uint32_t bucket = BUCKETS - 1u - (threadIdx.x * PER_THREAD + index);
		if ( above < keep && above + s.histogram[bucket] >= keep )
		{
			s.bucket = bucket;
			s.above = above;
		}
		above += s.histogram[bucket];
	}
	if ( threadIdx.x == 0u )
		s.count = 0u;
	__syncthreads();
	for ( base = 0u; base < local; base += THREADS * UNROLL )
	{
		float loaded[UNROLL];
		for ( index = 0u; index < UNROLL; index++ )
		{
			entry = base + index * THREADS + threadIdx.x;
			loaded[index] = entry < local ? __ldg(row_scores + entry) : 0.0f;
		}
		for ( index = 0u; index < UNROLL; index++ )
		{
			entry = base + index * THREADS + threadIdx.x;
			key = LmTopkKey(loaded[index]);
			if ( entry >= local || key < low )
				continue;
			const uint32_t bucket = LmIndexSelectSampledBucket(key, low, shift);
			if ( bucket > s.bucket )
				out[atomicAdd(&s.emitted, 1u)] = LmIndexSelectSampledCandidate(shard, loaded[index], entry);
			else if ( bucket == s.bucket )
			{
				slot = atomicAdd(&s.count, 1u);
				if ( slot < LIST )
					s.list[slot] = LmIndexSelectSampledEntry(key, entry);
			}
		}
	}
	__syncthreads();
	total = s.count;
	if ( total > LIST )
	{
		LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, s.histogram, s.state, s.warp_totals);
		LmDependentRelease();
		return;
	}
	LmIndexSelectSampledFinish<THREADS>(shard, s, total, keep - s.above, out);
	LmDependentRelease();
}

template<uint32_t THREADS>
struct LmIndexSelectMergeShared
{
	union
	{
		unsigned long long list[LM_INDEX_SELECT_MERGE_LIST];
		uint32_t samples[LM_INDEX_SELECT_MERGE_SAMPLES];
	};
	union
	{
		uint32_t histogram[LM_INDEX_SELECT_SAMPLED_BUCKETS];
		uint32_t out_pos[LM_INDEX_SELECT_MERGE_MAX_K];
	};
	uint32_t state[LM_TOPK_EXACT_STATE];
	uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	uint32_t count;
	uint32_t emitted;
};

template<uint32_t THREADS>
static __device__ void LmIndexSelectBitonicSort(uint32_t *values, uint32_t padded)
{
	for ( uint32_t size = 2u; size <= padded; size <<= 1u )
		for ( uint32_t stride = size >> 1u; stride > 0u; stride >>= 1u )
		{
			for ( uint32_t index = threadIdx.x; index < (padded >> 1u); index += THREADS )
			{
				const uint32_t low = 2u * index - (index & (stride - 1u));
				const uint32_t high = low + stride;
				const uint32_t a = values[low], b = values[high];
				if ( (a > b) == ((low & size) == 0u) )
				{
					values[low] = b;
					values[high] = a;
				}
			}
			__syncthreads();
		}
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmIndexSelectMergeAccept(
	LmIndexSelectMergeShared<THREADS> &s, uint32_t key, uint32_t pos, bool take, uint32_t lane)
{
	const uint32_t mask = __ballot_sync(0xffffffffu, take);
	if ( mask == 0u )
		return;
	const uint32_t leader = (uint32_t)(__ffs(mask) - 1);
	uint32_t first = lane == leader ? atomicAdd(&s.count, (uint32_t)__popc(mask)) : 0u;
	first = __shfl_sync(0xffffffffu, first, leader);
	const uint32_t slot = first + (uint32_t)__popc(mask & ((1u << lane) - 1u));
	if ( take && slot < LM_INDEX_SELECT_MERGE_LIST )
		s.list[slot] = ((unsigned long long)key << 32u) | (unsigned long long)(0xffffffffu - pos);
}
#endif

template<uint32_t THREADS>
static __device__ void LmIndexSelectMergeBitmapRun(
	const uint2 *row_gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep, uint32_t bound, uint32_t k,
	uint32_t words, uint32_t *greater, uint32_t *equal, uint32_t *out, uint32_t *histogram, uint32_t *state, uint32_t *warp_totals)
{
	LmIndexSelectCandidateSource source;
	uint32_t index, key, threshold, mask, need, per, first, last, word, bits, bit, count_greater, count_equal, equal_before, take, sum, emitted, taken;
	uint2 candidate;
	for ( index = threadIdx.x; index < words; index += THREADS )
	{
		greater[index] = 0u;
		equal[index] = 0u;
	}
	source.gathered = row_gathered;
	source.rank_stride = rank_stride_candidates;
	source.span = SparkKvShardGatherKeys(shard, bound);
	source.span = source.span < keep ? source.span : keep;
	source.degree = shard.degree;
	source.bound = bound;
	LmIndexSelectRadix<THREADS>(source, k, histogram, state);
	threshold = state[LM_TOPK_EXACT_PREFIX];
	mask = state[LM_TOPK_EXACT_MASK];
	need = state[LM_TOPK_EXACT_NEED];
	for ( index = threadIdx.x; index < source.Total(); index += THREADS * LM_INDEX_SELECT_UNROLL )
	{
		uint2 loaded[LM_INDEX_SELECT_UNROLL];
		for ( bit = 0u; bit < LM_INDEX_SELECT_UNROLL; bit++ )
			loaded[bit] = index + bit * THREADS < source.Total() ? source.Load(index + bit * THREADS) : make_uint2(0u, bound);
		for ( bit = 0u; bit < LM_INDEX_SELECT_UNROLL; bit++ )
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
	equal_before = LmIndexSelectExclusiveScan<THREADS>(count_equal, warp_totals, &sum);
	take = need > equal_before ? need - equal_before : 0u;
	take = take < count_equal ? take : count_equal;
	emitted = LmIndexSelectExclusiveScan<THREADS>(count_greater + take, warp_totals, &sum);
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
			if ( emitted < k )
				out[emitted] = word * 32u + bit;
			emitted++;
		}
	}
	for ( index = sum + threadIdx.x; index < k; index += THREADS )
		out[index] = LM_INDEX_SELECT_NONE;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmIndexSelectMergeKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t k, uint32_t dense_limit, uint32_t row_words, uint32_t *__restrict__ bitmaps, uint32_t *__restrict__ selected, LmFrameError *__restrict__ error)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	LmDependentWait();
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row);
	uint32_t *greater = bitmaps + (uint64_t)row * 2u * row_words, *equal = greater + row_words, *out = selected + (uint64_t)row * k;
	if ( bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t words = LmIndexSelectBitmapWords(bound);
	if ( words > row_words )
	{
		if ( threadIdx.x == 0u )
			LmFrameErrorReport(error, LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, 0u, row, sequence_of_row[row], bound, row_words * 32u);
		LmDependentRelease();
		return;
	}
	LmIndexSelectMergeBitmapRun<THREADS>(gathered + (uint64_t)row * keep, rank_stride_candidates, shard, keep, bound, k, words,
		greater, equal, out, histogram, state, warp_totals);
	LmDependentRelease();
}

#ifdef __CUDACC__
template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 2)
void LmIndexSelectMergeSampledKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t k, uint32_t dense_limit, uint32_t row_words, uint32_t *__restrict__ bitmaps, uint32_t *__restrict__ selected, LmFrameError *__restrict__ error)
{
	extern __shared__ __align__(16) unsigned char lm_index_select_merge_bytes[];
	LmIndexSelectMergeShared<THREADS> &s = *reinterpret_cast<LmIndexSelectMergeShared<THREADS> *>(lm_index_select_merge_bytes);
	LmDependentWait();
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row), lane = threadIdx.x & 31u;
	uint32_t *greater = bitmaps + (uint64_t)row * 2u * row_words, *equal = greater + row_words, *out = selected + (uint64_t)row * k;
	if ( bound <= dense_limit )
	{
		LmDependentRelease();
		return;
	}
	const uint32_t words = LmIndexSelectBitmapWords(bound);
	if ( words > row_words )
	{
		if ( threadIdx.x == 0u )
			LmFrameErrorReport(error, LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, 0u, row, sequence_of_row[row], bound, row_words * 32u);
		LmDependentRelease();
		return;
	}
	const uint2 *row_gathered = gathered + (uint64_t)row * keep;
	uint32_t span = SparkKvShardGatherKeys(shard, bound);
	span = span < keep ? span : keep;
	const uint32_t degree = shard.degree, total = degree * span;
	uint32_t low = 0u;
	if ( total > LM_INDEX_SELECT_MERGE_LIST )
	{
		const uint32_t target_samples = total / 16u < LM_INDEX_SELECT_MERGE_SAMPLES ? total / 16u : LM_INDEX_SELECT_MERGE_SAMPLES;
		const uint32_t per_rank = (target_samples + degree - 1u) / degree;
		const uint32_t scount = per_rank * degree < total ? per_rank * degree : total;
		const uint32_t stride = span > per_rank ? span / per_rank : 1u;
		if ( threadIdx.x == 0u )
			s.count = 0u;
		__syncthreads();
		for ( uint32_t idx = threadIdx.x; idx < scount; idx += THREADS )
		{
			const uint32_t d = idx / per_rank, j = idx % per_rank;
			const uint32_t off = j * stride + ((idx * 2654435761u) >> 16u) % stride;
			const uint2 c = off < span ? __ldg(row_gathered + (uint64_t)d * rank_stride_candidates + off) : make_uint2(0u, bound);
			const bool valid = c.y < bound;
			const uint32_t m = __ballot_sync(0xffffffffu, valid);
			if ( m != 0u )
			{
				const uint32_t leader = (uint32_t)(__ffs(m) - 1);
				uint32_t base = lane == leader ? atomicAdd(&s.count, (uint32_t)__popc(m)) : 0u;
				base = __shfl_sync(0xffffffffu, base, leader) + (uint32_t)__popc(m & ((1u << lane) - 1u));
				if ( valid && base < LM_INDEX_SELECT_MERGE_SAMPLES )
					s.samples[base] = LmTopkKey(__uint_as_float(c.x));
			}
		}
		__syncthreads();
		const uint32_t got_samples = s.count < LM_INDEX_SELECT_MERGE_SAMPLES ? s.count : LM_INDEX_SELECT_MERGE_SAMPLES;
		if ( got_samples > 0u )
		{
			const float expected = (float)k * (float)scount / (float)total, half = 0.5f * LM_INDEX_SELECT_MERGE_MARGIN;
			const float root = half + sqrtf(half * half + expected);
			uint32_t rank = (uint32_t)ceilf(root * root);
			if ( rank < got_samples )
			{
				LmIndexSelectSampleSource sampled = { s.samples, got_samples };
				LmIndexSelectRadix11<THREADS>(sampled, rank, s.histogram, s.state, s.warp_totals);
				low = s.state[LM_TOPK_EXACT_PREFIX];
			}
		}
	}
	if ( threadIdx.x == 0u )
	{
		s.count = 0u;
		s.emitted = 0u;
	}
	__syncthreads();
	const bool vec_ok = (((uintptr_t)row_gathered & 15u) == 0u) && ((rank_stride_candidates & 1u) == 0u);
	for ( uint32_t d = 0u; d < degree; d++ )
	{
		const uint2 *rp = row_gathered + (uint64_t)d * rank_stride_candidates;
		if ( vec_ok )
		{
			const uint32_t pairs = span >> 1u;
			const uint4 *rp4 = (const uint4 *)rp;
			for ( uint32_t pbase = 0u; pbase < pairs; pbase += THREADS * 2u )
			{
				const uint32_t p0 = pbase + threadIdx.x, p1 = p0 + THREADS;
				const uint4 v0 = p0 < pairs ? __ldg(rp4 + p0) : make_uint4(0u, bound, 0u, bound);
				const uint4 v1 = p1 < pairs ? __ldg(rp4 + p1) : make_uint4(0u, bound, 0u, bound);
				const uint32_t k0 = LmTopkKey(__uint_as_float(v0.x)), k1 = LmTopkKey(__uint_as_float(v0.z));
				const bool t0 = p0 < pairs && v0.y < bound && k0 >= low;
				const bool t1 = p0 < pairs && v0.w < bound && k1 >= low;
				if ( __any_sync(0xffffffffu, t0 || t1) )
				{
					LmIndexSelectMergeAccept<THREADS>(s, k0, v0.y, t0, lane);
					LmIndexSelectMergeAccept<THREADS>(s, k1, v0.w, t1, lane);
				}
				const uint32_t key2 = LmTopkKey(__uint_as_float(v1.x)), key3 = LmTopkKey(__uint_as_float(v1.z));
				const bool t2 = p1 < pairs && v1.y < bound && key2 >= low;
				const bool t3 = p1 < pairs && v1.w < bound && key3 >= low;
				if ( __any_sync(0xffffffffu, t2 || t3) )
				{
					LmIndexSelectMergeAccept<THREADS>(s, key2, v1.y, t2, lane);
					LmIndexSelectMergeAccept<THREADS>(s, key3, v1.w, t3, lane);
				}
			}
			if ( (span & 1u) != 0u )
			{
				const bool m = threadIdx.x == 0u;
				const uint2 c = m ? __ldg(rp + span - 1u) : make_uint2(0u, bound);
				const uint32_t k0 = LmTopkKey(__uint_as_float(c.x));
				LmIndexSelectMergeAccept<THREADS>(s, k0, c.y, m && c.y < bound && k0 >= low, lane);
			}
		}
		else
		{
			for ( uint32_t base = 0u; base < span; base += THREADS )
			{
				const uint32_t j = base + threadIdx.x;
				const uint2 c = j < span ? __ldg(rp + j) : make_uint2(0u, bound);
				const uint32_t k0 = LmTopkKey(__uint_as_float(c.x));
				LmIndexSelectMergeAccept<THREADS>(s, k0, c.y, j < span && c.y < bound && k0 >= low, lane);
			}
		}
	}
	__syncthreads();
	const uint32_t found = s.count;
	if ( found > LM_INDEX_SELECT_MERGE_LIST || (found < k && low != 0u) )
	{
		LmIndexSelectMergeBitmapRun<THREADS>(row_gathered, rank_stride_candidates, shard, keep, bound, k, words,
			greater, equal, out, s.histogram, s.state, s.warp_totals);
		LmDependentRelease();
		return;
	}
	uint32_t emitted = 0u;
	if ( found <= k )
	{
		for ( uint32_t i = threadIdx.x; i < found; i += THREADS )
			s.out_pos[i] = 0xffffffffu - (uint32_t)s.list[i];
		emitted = found;
	}
	else
	{
		LmIndexSelectListSource source = { s.list, found };
		LmIndexSelectRadix11<THREADS>(source, k, s.histogram, s.state, s.warp_totals);
		const uint32_t threshold = s.state[LM_TOPK_EXACT_PREFIX];
		const uint32_t mask = s.state[LM_TOPK_EXACT_MASK];
		const uint32_t done = s.state[LM_INDEX_SELECT_RADIX_DONE];
		const uint32_t tie_need = s.state[LM_TOPK_EXACT_NEED];
		uint32_t tie_threshold = 0u, tie_mask = 0u;
		if ( done == 0u )
		{
			__syncthreads();
			LmIndexSelectTiedSource tied = { s.list, found, threshold };
			LmIndexSelectRadix11<THREADS>(tied, tie_need, s.histogram, s.state, s.warp_totals);
			tie_threshold = s.state[LM_TOPK_EXACT_PREFIX];
			tie_mask = s.state[LM_TOPK_EXACT_MASK];
		}
		__syncthreads();
		for ( uint32_t base = 0u; base < found; base += THREADS )
		{
			const uint32_t index = base + threadIdx.x;
			unsigned long long value = 0ull;
			bool take = false;
			if ( index < found )
			{
				value = s.list[index];
				const uint32_t km = (uint32_t)(value >> 32u) & mask;
				take = km > threshold || (km == threshold && (done != 0u || ((uint32_t)value & tie_mask) >= tie_threshold));
			}
			const uint32_t ballot = __ballot_sync(0xffffffffu, take);
			if ( ballot != 0u )
			{
				const uint32_t leader = (uint32_t)(__ffs(ballot) - 1);
				uint32_t slot = lane == leader ? atomicAdd(&s.emitted, (uint32_t)__popc(ballot)) : 0u;
				slot = __shfl_sync(0xffffffffu, slot, leader) + (uint32_t)__popc(ballot & ((1u << lane) - 1u));
				if ( take && slot < k )
					s.out_pos[slot] = 0xffffffffu - (uint32_t)value;
			}
		}
		__syncthreads();
		emitted = s.emitted < k ? s.emitted : k;
	}
	uint32_t padded = 1u;
	while ( padded < k )
		padded <<= 1u;
	for ( uint32_t i = emitted + threadIdx.x; i < padded; i += THREADS )
		s.out_pos[i] = LM_INDEX_SELECT_NONE;
	__syncthreads();
	LmIndexSelectBitonicSort<THREADS>(s.out_pos, padded);
	for ( uint32_t i = threadIdx.x; i < k; i += THREADS )
		out[i] = s.out_pos[i];
	LmDependentRelease();
}
#endif

template<uint32_t THREADS>
static inline SparkGpuError LmIndexSelectLocalLaunch(SparkKvShard shard, const float *scores, uint32_t score_stride, const uint32_t *sequence_of_row,
	const uint32_t *context_length, const uint32_t *row_position, uint32_t keep, uint32_t dense_limit, uint32_t rows, uint2 *candidates, LmFrameError *error,
	SparkGpuStream stream)
{
	if ( scores == 0 || sequence_of_row == 0 || context_length == 0 || candidates == 0 || error == 0 || rows == 0u || keep == 0u ||
		score_stride == 0u || shard.degree == 0u || shard.grain == 0u )
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
#ifdef __CUDACC__
	if ( score_stride >= LM_INDEX_SELECT_SAMPLED_MIN_KEYS && keep <= LM_INDEX_SELECT_SAMPLED_MAX_KEEP )
	{
		static bool granted = false;
		const uint32_t shared = (uint32_t)sizeof(LmIndexSelectSampledShared<LM_INDEX_SELECT_SAMPLED_THREADS>);
		if ( !granted )
		{
			const SparkGpuError attribute = SparkGpuFuncSetAttribute((const void *)LmIndexSelectLocalSampledKernel<LM_INDEX_SELECT_SAMPLED_THREADS>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
			if ( attribute != SPARK_GPU_SUCCESS )
				return attribute;
			granted = true;
		}
		LM_LAUNCH_DEPENDENT((LmIndexSelectLocalSampledKernel<LM_INDEX_SELECT_SAMPLED_THREADS>), rows, LM_INDEX_SELECT_SAMPLED_THREADS, shared, stream, shard, scores,
			score_stride, sequence_of_row, context_length, row_position, keep, dense_limit, candidates, error);
	}
	else
#endif
		LM_LAUNCH_DEPENDENT((LmIndexSelectLocalKernel<THREADS>), rows, THREADS, 0, stream, shard, scores, score_stride, sequence_of_row, context_length, row_position,
			keep, dense_limit, candidates, error);
	return SparkGpuPeekAtLastError();
}

template<uint32_t THREADS>
static inline SparkGpuError LmIndexSelectMergeLaunch(const uint2 *gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position, uint32_t k, uint32_t dense_limit, uint32_t rows,
	uint32_t row_words, uint32_t *bitmaps, uint32_t *selected, LmFrameError *error, SparkGpuStream stream)
{
	if ( gathered == 0 || sequence_of_row == 0 || context_length == 0 || bitmaps == 0 || selected == 0 || error == 0 || rows == 0u || keep == 0u || k == 0u || shard.degree == 0u ||
		shard.degree > SPARK_KV_SHARD_MAX_DEGREE || shard.grain == 0u || row_words == 0u || rank_stride_candidates < (uint64_t)rows * keep )
		SPARK_FAIL(SPARK_GPU_ERROR_INVALID_VALUE);
#ifdef __CUDACC__
	if ( k <= LM_INDEX_SELECT_MERGE_MAX_K )
	{
		static bool granted = false;
		const uint32_t shared = (uint32_t)sizeof(LmIndexSelectMergeShared<THREADS>);
		if ( !granted )
		{
			const SparkGpuError attribute = SparkGpuFuncSetAttribute((const void *)LmIndexSelectMergeSampledKernel<THREADS>,
				SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
			if ( attribute != SPARK_GPU_SUCCESS )
				return attribute;
			granted = true;
		}
		LM_LAUNCH_DEPENDENT((LmIndexSelectMergeSampledKernel<THREADS>), rows, THREADS, shared, stream, gathered, rank_stride_candidates, shard, keep, sequence_of_row,
			context_length, row_position, k, dense_limit, row_words, bitmaps, selected, error);
	}
	else
#endif
		LM_LAUNCH_DEPENDENT((LmIndexSelectMergeKernel<THREADS>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, shard, keep, sequence_of_row, context_length,
			row_position, k, dense_limit, row_words, bitmaps, selected, error);
	return SparkGpuPeekAtLastError();
}
