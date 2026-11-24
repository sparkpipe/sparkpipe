#pragma once

#include <stdint.h>
#include <math.h>

#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/index_shard.cuh"
#include "inference/kernels/topk_exact.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_INDEX_SELECT_THREADS 512u
#define LM_INDEX_SELECT_NONE 0xffffffffu
#define LM_INDEX_SELECT_RADIX_DONE LM_TOPK_EXACT_EMITTED
#define LM_INDEX_SELECT_UNROLL 4u
#define LM_INDEX_SELECT_SAMPLED_MIN_KEYS 16384u
#define LM_INDEX_SELECT_SAMPLED_SPACING 64u
#define LM_INDEX_SELECT_SAMPLED_SAMPLES 16384u
#define LM_INDEX_SELECT_SAMPLED_LIST 8192u
#define LM_INDEX_SELECT_SAMPLED_BUCKETS 2048u
#define LM_INDEX_SELECT_SAMPLED_MAX_KEEP 2048u
#define LM_INDEX_SELECT_SAMPLED_MARGIN 5.0f
#define LM_INDEX_SELECT_SAMPLED_THREADS 1024u
#define LM_INDEX_SELECT_SAMPLED_UNROLL 8u

static __host__ __device__ __forceinline__ uint32_t LmIndexSelectBitmapWords(uint32_t positions)
{
	return (positions + 31u) / 32u;
}

static __device__ __forceinline__ uint32_t LmIndexSelectRowBound(const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *row_position,
	uint32_t row)
{
	uint32_t bound = context_length[sequence_of_row[row]];
	if ( row_position != 0 && row_position[row] + 1u < bound )
		bound = row_position[row] + 1u;
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
		*key = LmTopkKey(scores[index]);
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
static __device__ void LmIndexSelectRadix(const Source &source, uint32_t k, uint32_t *histogram, uint32_t *state)
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
		for ( base = 0u; base < total; base += THREADS * LM_INDEX_SELECT_UNROLL )
		{
			uint32_t keys[LM_INDEX_SELECT_UNROLL], live[LM_INDEX_SELECT_UNROLL];
			for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
			{
				keys[index] = 0u;
				live[index] = source.Key(base + index * THREADS + threadIdx.x, &keys[index]);
			}
			for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
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
				state[LM_INDEX_SELECT_RADIX_DONE] = above + histogram[chosen] == need ? 1u : 0u;
			}
		}
		__syncthreads();
	}
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint32_t LmIndexSelectExclusiveScan(uint32_t value, uint32_t *warp_totals, uint32_t *total)
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
static __device__ void LmIndexSelectLocalRadixRow(SparkKvShard shard, const float *row_scores, uint32_t local, uint32_t keep, uint2 *out, uint32_t *histogram,
	uint32_t *state, uint32_t *warp_totals)
{
	LmIndexSelectScoreSource source;
	uint32_t entry, base, key, greater, equal, threshold, mask, need, equal_before, greater_before, sum_equal, sum_greater, emitted, seen, take;
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
		key = entry < local ? LmTopkKey(row_scores[entry]) & mask : 0u;
		greater = entry < local && key > threshold ? 1u : 0u;
		equal = entry < local && key == threshold ? 1u : 0u;
		greater_before = LmIndexSelectExclusiveScan<THREADS>(greater, warp_totals, &sum_greater);
		equal_before = LmIndexSelectExclusiveScan<THREADS>(equal, warp_totals, &sum_equal);
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
		score = entry < local ? row_scores[entry] : -INFINITY;
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
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row);
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	uint32_t local;
	if ( bound <= dense_limit )
		return;
	local = SparkKvShardLocalKeys(shard, bound);
	if ( LmIndexSelectLocalWholeRow<THREADS>(shard, row_scores, score_stride, sequence_of_row, row, local, keep, out, error) != 0u )
		return;
	LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, histogram, state, warp_totals);
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

static __device__ __forceinline__ uint2 LmIndexSelectSampledCandidate(SparkKvShard shard, float score, uint32_t entry)
{
	return make_uint2(__float_as_uint(score), score > -INFINITY ? SparkKvShardLocalPosition(shard, entry) : LM_INDEX_SHARD_NO_POSITION);
}

template<uint32_t THREADS>
static __device__ void LmIndexSelectSampledFinish(SparkKvShard shard, const float *row_scores, LmIndexSelectSampledShared<THREADS> &s, uint32_t count, uint32_t need,
	uint2 *out)
{
	LmIndexSelectListSource source;
	LmIndexSelectTiedSource tied;
	uint32_t threshold, mask, done, tie_need, tie_threshold, tie_mask, index, key, entry;
	unsigned long long value;
	source.list = s.list;
	source.count = count;
	LmIndexSelectRadix<THREADS>(source, need, s.histogram, s.state);
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
		LmIndexSelectRadix<THREADS>(tied, tie_need, s.histogram, s.state);
		tie_threshold = s.state[LM_TOPK_EXACT_PREFIX];
		tie_mask = s.state[LM_TOPK_EXACT_MASK];
	}
	for ( index = threadIdx.x; index < count; index += THREADS )
	{
		value = s.list[index];
		key = (uint32_t)(value >> 32u) & mask;
		if ( key > threshold || (key == threshold && (done != 0u || ((uint32_t)value & tie_mask) >= tie_threshold)) )
		{
			entry = 0xffffffffu - (uint32_t)value;
			out[atomicAdd(&s.emitted, 1u)] = LmIndexSelectSampledCandidate(shard, row_scores[entry], entry);
		}
	}
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
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row), lane = threadIdx.x % LM_TOPK_WARP_LANES;
	const float *row_scores = scores + (uint64_t)row * score_stride;
	uint2 *out = candidates + (uint64_t)row * keep;
	LmIndexSelectSampleSource sampled;
	uint32_t local, entry, index, count, step, base, key, mask, first, slot, low, shift, span, total, above, own, rank, peak;
	if ( bound <= dense_limit )
		return;
	local = SparkKvShardLocalKeys(shard, bound);
	if ( LmIndexSelectLocalWholeRow<THREADS>(shard, row_scores, score_stride, sequence_of_row, row, local, keep, out, error) != 0u )
		return;
	if ( local < LM_INDEX_SELECT_SAMPLED_MIN_KEYS )
	{
		LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, s.histogram, s.state, s.warp_totals);
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
		key = LmTopkKey(row_scores[(uint64_t)index * step + ((index * 2654435761u) >> 16u) % step]);
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
	LmIndexSelectRadix<THREADS>(sampled, rank, s.histogram, s.state);
	low = s.state[LM_TOPK_EXACT_PREFIX];
	span = s.top - low;
	for ( shift = 0u; (span >> shift) >= BUCKETS; shift++ )
		;
	__syncthreads();
	for ( index = threadIdx.x; index < BUCKETS; index += THREADS )
		s.histogram[index] = 0u;
	__syncthreads();
	for ( base = 0u; base < local; base += THREADS * UNROLL )
	{
		float loaded[UNROLL];
		for ( index = 0u; index < UNROLL; index++ )
		{
			entry = base + index * THREADS + threadIdx.x;
			loaded[index] = entry < local ? row_scores[entry] : 0.0f;
		}
		for ( index = 0u; index < UNROLL; index++ )
		{
			entry = base + index * THREADS + threadIdx.x;
			key = LmTopkKey(loaded[index]);
			const bool inside = entry < local && key >= low;
			mask = __ballot_sync(0xffffffffu, inside);
			if ( mask == 0u )
				continue;
			if ( inside )
				atomicAdd(&s.histogram[LmIndexSelectSampledBucket(key, low, shift)], 1u);
			first = lane == (uint32_t)(__ffs(mask) - 1) ? atomicAdd(&s.count, (uint32_t)__popc(mask)) : 0u;
			first = __shfl_sync(0xffffffffu, first, __ffs(mask) - 1);
			slot = first + (uint32_t)__popc(mask & ((1u << lane) - 1u));
			if ( inside && slot < LIST )
				s.list[slot] = LmIndexSelectSampledEntry(key, entry);
		}
	}
	__syncthreads();
	total = s.count;
	if ( total < keep )
	{
		LmIndexSelectLocalRadixRow<THREADS>(shard, row_scores, local, keep, out, s.histogram, s.state, s.warp_totals);
		return;
	}
	if ( total <= LIST )
	{
		LmIndexSelectSampledFinish<THREADS>(shard, row_scores, s, total, keep, out);
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
			loaded[index] = entry < local ? row_scores[entry] : 0.0f;
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
		return;
	}
	LmIndexSelectSampledFinish<THREADS>(shard, row_scores, s, total, keep - s.above, out);
}
#endif

#define LM_INDEX_SELECT_MERGE_HIST 4096u
#define LM_INDEX_SELECT_MERGE_TIE_CAP 2048u
#define LM_INDEX_SELECT_MERGE_BITMAP_WORDS 2048u

struct LmIndexSelectTieArraySource
{
	const uint2 *items;
	uint32_t count;
	__device__ __forceinline__ uint32_t Total(void) const { return count; }
	__device__ __forceinline__ uint32_t Key(uint32_t index, uint32_t *key) const
	{
		if ( index >= count )
			return 0u;
		*key = items[index].x;
		return 1u;
	}
};

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS)
void LmIndexSelectMergeKernel(const uint2 *__restrict__ gathered, uint64_t rank_stride_candidates, SparkKvShard shard, uint32_t keep,
	const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position,
	uint32_t k, uint32_t dense_limit, uint32_t row_words, uint32_t *__restrict__ bitmaps, uint32_t *__restrict__ selected, LmFrameError *__restrict__ error)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	__shared__ uint32_t state[LM_TOPK_EXACT_STATE];
	__shared__ uint32_t warp_totals[THREADS / LM_TOPK_WARP_LANES];
#ifdef __CUDACC__
	__shared__ uint32_t greater_shared[LM_INDEX_SELECT_MERGE_BITMAP_WORDS];
	__shared__ uint32_t equal_shared[LM_INDEX_SELECT_MERGE_BITMAP_WORDS];
	__shared__ union
	{
		uint32_t hist12[LM_INDEX_SELECT_MERGE_HIST];
		uint2 ties[LM_INDEX_SELECT_MERGE_TIE_CAP];
	} merge_work;
	__shared__ uint32_t merge_meta[4];
#endif
	const uint32_t row = blockIdx.x, bound = LmIndexSelectRowBound(sequence_of_row, context_length, row_position, row);
	uint32_t *greater = bitmaps + (uint64_t)row * 2u * row_words, *equal = greater + row_words, *out = selected + (uint64_t)row * k;
	uint32_t words, index, key, threshold, mask, need, per, first, last, word, bits, bit, count_greater, count_equal, equal_before, take, sum, emitted, taken;
	LmIndexSelectCandidateSource source;
	uint2 candidate;
	if ( bound <= dense_limit )
		return;
	words = LmIndexSelectBitmapWords(bound);
	if ( words > row_words )
	{
		if ( threadIdx.x == 0u )
			LmFrameErrorReport(error, LM_FRAME_ERROR_SPARSE_INDEX_OUT_OF_RANGE, 0u, row, sequence_of_row[row], bound, row_words * 32u);
		return;
	}
#ifdef __CUDACC__
	if ( words <= LM_INDEX_SELECT_MERGE_BITMAP_WORDS )
	{
		greater = greater_shared;
		equal = equal_shared;
	}
#endif
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
#ifdef __CUDACC__
	if constexpr ( THREADS == LM_INDEX_SELECT_THREADS && LM_INDEX_SELECT_MERGE_HIST % THREADS == 0u )
	{
		constexpr uint32_t PER = LM_INDEX_SELECT_MERGE_HIST / THREADS;
		const uint32_t lane = threadIdx.x % LM_TOPK_WARP_LANES;
		const uint32_t total_cand = source.Total();
		uint32_t r, base, own = 0u, above, hist_total, chosen_bucket, above_chosen, bucket_size, low_key, high_key, shift, count_above;
		if ( total_cand <= k )
		{
			for ( r = 0u; r < source.degree; r++ )
			{
				const uint2 *rank_ptr = source.gathered + (uint64_t)r * source.rank_stride;
				for ( index = threadIdx.x; index < source.span; index += THREADS )
				{
					const uint2 cand = rank_ptr[index];
					if ( cand.y < bound )
						atomicOr(&greater[cand.y >> 5u], 1u << (cand.y & 31u));
				}
			}
			need = 0u;
		}
		else
		{
			const uint32_t s_idx = (uint32_t)(((uint64_t)threadIdx.x * total_cand) / THREADS);
			const uint2 s_cand = source.gathered[(uint64_t)(s_idx / source.span) * source.rank_stride + s_idx % source.span];
			const uint32_t s_key = s_cand.y < bound ? LmTopkKey(__uint_as_float(s_cand.x)) : 0u;
			uint32_t wmax = s_key;
			merge_work.hist12[threadIdx.x] = s_key;
			for ( uint32_t off = LM_TOPK_WARP_LANES / 2u; off > 0u; off >>= 1u )
			{
				const uint32_t other = __shfl_xor_sync(0xffffffffu, wmax, off);
				wmax = other > wmax ? other : wmax;
			}
			if ( lane == 0u )
				warp_totals[threadIdx.x / LM_TOPK_WARP_LANES] = wmax;
			__syncthreads();
			if ( threadIdx.x == 0u )
			{
				uint32_t gmax = 0u;
				for ( uint32_t w = 0u; w < THREADS / LM_TOPK_WARP_LANES; w++ )
					gmax = warp_totals[w] > gmax ? warp_totals[w] : gmax;
				merge_meta[0] = gmax;
			}
			__syncthreads();
			const uint32_t sample_k = (uint32_t)(((uint64_t)k * THREADS + total_cand - 1u) / total_cand);
			const uint32_t low_rank = sample_k * 2u + 32u < THREADS ? sample_k * 2u + 32u : THREADS;
			LmIndexSelectSampleSource sample_src = { merge_work.hist12, THREADS };
			LmIndexSelectRadix<THREADS>(sample_src, low_rank, histogram, state);
			high_key = merge_meta[0];
			low_key = state[LM_TOPK_EXACT_PREFIX];
			if ( low_key > high_key )
				low_key = high_key;
			const uint32_t span_keys = high_key - low_key;
			shift = span_keys >= LM_INDEX_SELECT_MERGE_HIST ? (20u - (uint32_t)__clz(span_keys)) : 0u;
			for ( index = threadIdx.x; index < LM_INDEX_SELECT_MERGE_HIST; index += THREADS )
				merge_work.hist12[index] = 0u;
			if ( threadIdx.x < 4u )
				merge_meta[threadIdx.x] = 0u;
			__syncthreads();
			for ( r = 0u; r < source.degree; r++ )
			{
				const uint2 *rank_ptr = source.gathered + (uint64_t)r * source.rank_stride;
				for ( base = 0u; base < source.span; base += THREADS * LM_INDEX_SELECT_UNROLL )
				{
					uint2 loaded[LM_INDEX_SELECT_UNROLL];
					#pragma unroll
					for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
					{
						const uint32_t entry = base + index * THREADS + threadIdx.x;
						loaded[index] = entry < source.span ? rank_ptr[entry] : make_uint2(0u, bound);
					}
					#pragma unroll
					for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
					{
						const uint2 cand = loaded[index];
						uint32_t b = LM_INDEX_SELECT_MERGE_HIST + 1u;
						if ( cand.y < bound )
						{
							const uint32_t ck = LmTopkKey(__uint_as_float(cand.x));
							if ( ck > high_key )
								b = LM_INDEX_SELECT_MERGE_HIST;
							else if ( ck >= low_key )
								b = (ck - low_key) >> shift;
						}
						const uint32_t peers = __match_any_sync(0xffffffffu, b);
						if ( lane == (uint32_t)(__ffs(peers) - 1) )
						{
							if ( b < LM_INDEX_SELECT_MERGE_HIST )
								atomicAdd(&merge_work.hist12[b], (uint32_t)__popc(peers));
							else if ( b == LM_INDEX_SELECT_MERGE_HIST )
								atomicAdd(&merge_meta[3], (uint32_t)__popc(peers));
						}
					}
				}
			}
			__syncthreads();
			count_above = merge_meta[3];
			#pragma unroll
			for ( index = 0u; index < PER; index++ )
				own += merge_work.hist12[LM_INDEX_SELECT_MERGE_HIST - 1u - (threadIdx.x * PER + index)];
			above = count_above + LmIndexSelectExclusiveScan<THREADS>(own, warp_totals, &hist_total);
			#pragma unroll
			for ( index = 0u; index < PER; index++ )
			{
				const uint32_t b = LM_INDEX_SELECT_MERGE_HIST - 1u - (threadIdx.x * PER + index);
				const uint32_t c = merge_work.hist12[b];
				if ( above < k && above + c >= k )
				{
					merge_meta[0] = b;
					merge_meta[1] = above;
					merge_meta[2] = c;
				}
				above += c;
			}
			__syncthreads();
			chosen_bucket = merge_meta[0];
			above_chosen = merge_meta[1];
			bucket_size = merge_meta[2];
			if ( count_above < k && count_above + hist_total >= k && (shift == 0u || bucket_size <= LM_INDEX_SELECT_MERGE_TIE_CAP) )
			{
				if ( shift == 0u )
				{
					for ( r = 0u; r < source.degree; r++ )
					{
						const uint2 *rank_ptr = source.gathered + (uint64_t)r * source.rank_stride;
						for ( index = threadIdx.x; index < source.span; index += THREADS )
						{
							const uint2 cand = rank_ptr[index];
							if ( cand.y >= bound )
								continue;
							const uint32_t ck = LmTopkKey(__uint_as_float(cand.x));
							if ( ck > high_key || (ck >= low_key && ck - low_key > chosen_bucket) )
								atomicOr(&greater[cand.y >> 5u], 1u << (cand.y & 31u));
							else if ( ck >= low_key && ck <= high_key && ck - low_key == chosen_bucket )
								atomicOr(&equal[cand.y >> 5u], 1u << (cand.y & 31u));
						}
					}
					need = k - above_chosen;
				}
				else
				{
					if ( threadIdx.x == 0u )
						merge_meta[3] = 0u;
					__syncthreads();
					for ( r = 0u; r < source.degree; r++ )
					{
						const uint2 *rank_ptr = source.gathered + (uint64_t)r * source.rank_stride;
						for ( base = 0u; base < source.span; base += THREADS * LM_INDEX_SELECT_UNROLL )
						{
							uint2 loaded[LM_INDEX_SELECT_UNROLL];
							#pragma unroll
							for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
							{
								const uint32_t entry = base + index * THREADS + threadIdx.x;
								loaded[index] = entry < source.span ? rank_ptr[entry] : make_uint2(0u, bound);
							}
							#pragma unroll
							for ( index = 0u; index < LM_INDEX_SELECT_UNROLL; index++ )
							{
								const uint2 cand = loaded[index];
								if ( cand.y >= bound )
									continue;
								const uint32_t ck = LmTopkKey(__uint_as_float(cand.x));
								if ( ck > high_key )
									atomicOr(&greater[cand.y >> 5u], 1u << (cand.y & 31u));
								else if ( ck >= low_key )
								{
									const uint32_t b = (ck - low_key) >> shift;
									if ( b > chosen_bucket )
										atomicOr(&greater[cand.y >> 5u], 1u << (cand.y & 31u));
									else if ( b == chosen_bucket )
									{
										const uint32_t slot = atomicAdd(&merge_meta[3], 1u);
										if ( slot < LM_INDEX_SELECT_MERGE_TIE_CAP )
											merge_work.ties[slot] = make_uint2(ck, cand.y);
									}
								}
							}
						}
					}
					__syncthreads();
					const uint32_t tie_count = merge_meta[3];
					if ( above_chosen + tie_count == k )
					{
						for ( index = threadIdx.x; index < tie_count; index += THREADS )
						{
							const uint32_t pos = merge_work.ties[index].y;
							atomicOr(&greater[pos >> 5u], 1u << (pos & 31u));
						}
						need = 0u;
					}
					else
					{
						LmIndexSelectTieArraySource tie_source = { merge_work.ties, tie_count };
						LmIndexSelectRadix<THREADS>(tie_source, k - above_chosen, histogram, state);
						threshold = state[LM_TOPK_EXACT_PREFIX];
						mask = state[LM_TOPK_EXACT_MASK];
						need = state[LM_TOPK_EXACT_NEED];
						for ( index = threadIdx.x; index < tie_count; index += THREADS )
						{
							const uint2 item = merge_work.ties[index];
							const uint32_t ck = item.x & mask;
							if ( ck > threshold )
								atomicOr(&greater[item.y >> 5u], 1u << (item.y & 31u));
							else if ( ck == threshold )
								atomicOr(&equal[item.y >> 5u], 1u << (item.y & 31u));
						}
					}
				}
			}
			else
			{
				LmIndexSelectRadix<THREADS>(source, k, histogram, state);
				threshold = state[LM_TOPK_EXACT_PREFIX];
				mask = state[LM_TOPK_EXACT_MASK];
				need = state[LM_TOPK_EXACT_NEED];
				for ( r = 0u; r < source.degree; r++ )
				{
					const uint2 *rank_ptr = source.gathered + (uint64_t)r * source.rank_stride;
					for ( index = threadIdx.x; index < source.span; index += THREADS )
					{
						candidate = rank_ptr[index];
						if ( candidate.y >= bound )
							continue;
						key = LmTopkKey(__uint_as_float(candidate.x)) & mask;
						if ( key > threshold )
							atomicOr(&greater[candidate.y >> 5u], 1u << (candidate.y & 31u));
						else if ( key == threshold )
							atomicOr(&equal[candidate.y >> 5u], 1u << (candidate.y & 31u));
					}
				}
			}
		}
	}
	else
#endif
	{
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
static inline SparkGpuError LmIndexSelectLocalLaunch(SparkKvShard shard, const float *scores, uint32_t score_stride, const uint32_t *sequence_of_row,
	const uint32_t *context_length, const uint32_t *row_position, uint32_t keep, uint32_t dense_limit, uint32_t rows, uint2 *candidates, LmFrameError *error,
	SparkGpuStream stream)
{
	if ( scores == 0 || sequence_of_row == 0 || context_length == 0 || candidates == 0 || error == 0 || rows == 0u || keep == 0u ||
		score_stride == 0u || shard.degree == 0u || shard.grain == 0u )
		return SPARK_GPU_ERROR_INVALID_VALUE;
#ifdef __CUDACC__
	if ( score_stride >= LM_INDEX_SELECT_SAMPLED_MIN_KEYS && keep <= LM_INDEX_SELECT_SAMPLED_MAX_KEEP )
	{
		const uint32_t shared = (uint32_t)sizeof(LmIndexSelectSampledShared<LM_INDEX_SELECT_SAMPLED_THREADS>);
		const SparkGpuError attribute = SparkGpuFuncSetAttribute((const void *)LmIndexSelectLocalSampledKernel<LM_INDEX_SELECT_SAMPLED_THREADS>,
			SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)shared);
		if ( attribute != SPARK_GPU_SUCCESS )
			return attribute;
		LM_LAUNCH((LmIndexSelectLocalSampledKernel<LM_INDEX_SELECT_SAMPLED_THREADS>), rows, LM_INDEX_SELECT_SAMPLED_THREADS, shared, stream, shard, scores,
			score_stride, sequence_of_row, context_length, row_position, keep, dense_limit, candidates, error);
	}
	else
#endif
		LM_LAUNCH((LmIndexSelectLocalKernel<THREADS>), rows, THREADS, 0, stream, shard, scores, score_stride, sequence_of_row, context_length, row_position,
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
		return SPARK_GPU_ERROR_INVALID_VALUE;
	LM_LAUNCH((LmIndexSelectMergeKernel<THREADS>), rows, THREADS, 0, stream, gathered, rank_stride_candidates, shard, keep, sequence_of_row, context_length,
		row_position, k, dense_limit, row_words, bitmaps, selected, error);
	return SparkGpuPeekAtLastError();
}
