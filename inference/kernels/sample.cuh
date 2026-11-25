#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/head.cuh"
#include "inference/kernels/kernel_math.cuh"
#include "include/sparkpipe/spark_sampling.h"
#include <stdint.h>

#define LM_SAMPLE_DIGIT_BITS 8u
#define LM_SAMPLE_DIGITS (1u << LM_SAMPLE_DIGIT_BITS)
#define LM_SAMPLE_PASSES (64u / LM_SAMPLE_DIGIT_BITS)
#define LM_SAMPLE_HISTOGRAM_COPIES(threads) ((threads) >= 64u ? 8u : 1u)
#define LM_SAMPLE_COMPACT_MAX 2048u

typedef struct LmSampleVocab
{
	const float *logits;
	uint64_t shard_stride;
	uint32_t row_stride;
	uint32_t shard_tokens;
	uint32_t vocabulary;
}
LmSampleVocab;

typedef struct LmSampleRows
{
	const SparkRowSampling *rules;
	const uint32_t *positions;
	const uint32_t *source_rows;
	const uint32_t *greedy_tokens;
	const uint32_t *logprob_rows;
	uint32_t *token_out;
	float *logit_out;
	SparkSamplingLogprob *logprobs_out;
	uint32_t rows;
}
LmSampleRows;

static __device__ __forceinline__ float LmSampleLogit(const LmSampleVocab &vocab, uint32_t row, uint32_t token)
{
	const uint32_t shard = token / vocab.shard_tokens;
	return(vocab.logits[shard * vocab.shard_stride + (uint64_t)row * vocab.row_stride + (token - shard * vocab.shard_tokens)]);
}

typedef struct LmSampleCursor
{
	uint32_t token;
	uint32_t shard;
	uint32_t offset;
}
LmSampleCursor;

static __device__ __forceinline__ LmSampleCursor LmSampleCursorStart(const LmSampleVocab &vocab, uint32_t token)
{
	LmSampleCursor cursor;
	cursor.token = token;
	cursor.shard = token / vocab.shard_tokens;
	cursor.offset = token - cursor.shard * vocab.shard_tokens;
	return(cursor);
}

static __device__ __forceinline__ void LmSampleCursorAdvance(const LmSampleVocab &vocab, LmSampleCursor &cursor, uint32_t stride)
{
	cursor.token += stride;
	cursor.offset += stride;
	while ( cursor.offset >= vocab.shard_tokens )
	{
		cursor.offset -= vocab.shard_tokens;
		cursor.shard++;
	}
}

static __device__ __forceinline__ float LmSampleCursorLogit(const LmSampleVocab &vocab, uint32_t row, const LmSampleCursor &cursor)
{
	return(vocab.logits[cursor.shard * vocab.shard_stride + (uint64_t)row * vocab.row_stride + cursor.offset]);
}

template<uint32_t THREADS>
struct LmSampleShared
{
	float values[THREADS];
#if defined(__CUDA_ARCH__)
	uint64_t keys[THREADS < 32u ? 1u : (THREADS / 32u)];
#else
	uint64_t keys[THREADS];
#endif
	uint64_t histogram[LM_SAMPLE_HISTOGRAM_COPIES(THREADS)][LM_SAMPLE_DIGITS];
	uint64_t compact_keys[LM_SAMPLE_COMPACT_MAX];
	float compact_logits[LM_SAMPLE_COMPACT_MAX];
	uint64_t choice[2];
	uint64_t picked[SPARK_SAMPLING_MAX_LOGPROBS];
	uint64_t unique_key;
	uint32_t unique_count;
	uint32_t unique_weighted;
	uint32_t picked_count;
	uint32_t compact_count;
	uint32_t compact_overflow;
};

#if defined(__CUDA_ARCH__)
static __device__ __forceinline__ uint64_t LmSampleShflDownU64(uint64_t value, uint32_t offset)
{
	const uint32_t lo = __shfl_down_sync(0xffffffffu, (uint32_t)value, offset);
	const uint32_t hi = __shfl_down_sync(0xffffffffu, (uint32_t)(value >> 32u), offset);
	return(((uint64_t)hi << 32u) | (uint64_t)lo);
}
#endif

template<uint32_t THREADS>
static __device__ __forceinline__ float LmSampleBlockMax(LmSampleShared<THREADS> &shared, float value)
{
#if defined(__CUDA_ARCH__)
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u;
	constexpr uint32_t warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u);
	const uint32_t warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
		value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
	if constexpr ( warps == 1u )
		return(__shfl_sync(0xffffffffu, value, 0u));
	if ( lane == 0u )
		shared.values[warp] = value;
	__syncthreads();
	if ( warp == 0u )
	{
		value = lane < warps ? shared.values[lane] : -INFINITY;
		constexpr uint32_t start = warps > 16u ? 16u : (warps > 8u ? 8u : (warps > 4u ? 4u : (warps > 2u ? 2u : 1u)));
		#pragma unroll
		for (uint32_t offset = start; offset > 0u; offset >>= 1u)
			value = fmaxf(value, __shfl_down_sync(0xffffffffu, value, offset));
		if ( lane == 0u )
			shared.values[0] = value;
	}
	__syncthreads();
	const float result = shared.values[0];
	__syncthreads();
	return(result);
#else
	uint32_t stride;
	float result;
	shared.values[threadIdx.x] = value;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
			shared.values[threadIdx.x] = fmaxf(shared.values[threadIdx.x], shared.values[threadIdx.x + stride]);
		__syncthreads();
	}
	result = shared.values[0];
	__syncthreads();
	return(result);
#endif
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmSampleBlockSum(LmSampleShared<THREADS> &shared, float value)
{
	uint32_t stride;
	float result;
	shared.values[threadIdx.x] = value;
	__syncthreads();
#if defined(__CUDA_ARCH__)
	for (stride = THREADS / 2u; stride >= 32u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
			shared.values[threadIdx.x] += shared.values[threadIdx.x + stride];
		__syncthreads();
	}
	if ( threadIdx.x < 32u )
	{
		float acc = shared.values[threadIdx.x];
		constexpr uint32_t start = THREADS >= 32u ? 16u : (THREADS / 2u);
		#pragma unroll
		for (uint32_t offset = start; offset > 0u; offset >>= 1u)
			acc += __shfl_down_sync(0xffffffffu, acc, offset);
		if ( threadIdx.x == 0u )
			shared.values[0] = acc;
	}
	__syncthreads();
	result = shared.values[0];
	__syncthreads();
	return(result);
#else
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
			shared.values[threadIdx.x] += shared.values[threadIdx.x + stride];
		__syncthreads();
	}
	result = shared.values[0];
	__syncthreads();
	return(result);
#endif
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleBlockMaxKey(LmSampleShared<THREADS> &shared, uint64_t key)
{
#if defined(__CUDA_ARCH__)
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u;
	constexpr uint32_t warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u);
	const uint32_t warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
	{
		const uint64_t other = LmSampleShflDownU64(key, offset);
		if ( other > key )
			key = other;
	}
	if constexpr ( warps == 1u )
	{
		const uint32_t lo = __shfl_sync(0xffffffffu, (uint32_t)key, 0u);
		const uint32_t hi = __shfl_sync(0xffffffffu, (uint32_t)(key >> 32u), 0u);
		return(((uint64_t)hi << 32u) | (uint64_t)lo);
	}
	if ( lane == 0u )
		shared.keys[warp] = key;
	__syncthreads();
	if ( warp == 0u )
	{
		key = lane < warps ? shared.keys[lane] : 0u;
		constexpr uint32_t start = warps > 16u ? 16u : (warps > 8u ? 8u : (warps > 4u ? 4u : (warps > 2u ? 2u : 1u)));
		#pragma unroll
		for (uint32_t offset = start; offset > 0u; offset >>= 1u)
		{
			const uint64_t other = LmSampleShflDownU64(key, offset);
			if ( other > key )
				key = other;
		}
		if ( lane == 0u )
			shared.keys[0] = key;
	}
	__syncthreads();
	const uint64_t result = shared.keys[0];
	__syncthreads();
	return(result);
#else
	uint32_t stride;
	uint64_t result;
	shared.keys[threadIdx.x] = key;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride && shared.keys[threadIdx.x + stride] > shared.keys[threadIdx.x] )
			shared.keys[threadIdx.x] = shared.keys[threadIdx.x + stride];
		__syncthreads();
	}
	result = shared.keys[0];
	__syncthreads();
	return(result);
#endif
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleBlockSumU64(LmSampleShared<THREADS> &shared, uint64_t value)
{
#if defined(__CUDA_ARCH__)
	constexpr uint32_t lanes = THREADS < 32u ? THREADS : 32u;
	constexpr uint32_t warps = THREADS / lanes;
	const uint32_t lane = threadIdx.x & (lanes - 1u);
	const uint32_t warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = lanes / 2u; offset > 0u; offset >>= 1u)
		value += LmSampleShflDownU64(value, offset);
	if constexpr ( warps == 1u )
	{
		const uint32_t lo = __shfl_sync(0xffffffffu, (uint32_t)value, 0u);
		const uint32_t hi = __shfl_sync(0xffffffffu, (uint32_t)(value >> 32u), 0u);
		return(((uint64_t)hi << 32u) | (uint64_t)lo);
	}
	if ( lane == 0u )
		shared.keys[warp] = value;
	__syncthreads();
	if ( warp == 0u )
	{
		value = lane < warps ? shared.keys[lane] : 0u;
		constexpr uint32_t start = warps > 16u ? 16u : (warps > 8u ? 8u : (warps > 4u ? 4u : (warps > 2u ? 2u : 1u)));
		#pragma unroll
		for (uint32_t offset = start; offset > 0u; offset >>= 1u)
			value += LmSampleShflDownU64(value, offset);
		if ( lane == 0u )
			shared.keys[0] = value;
	}
	__syncthreads();
	const uint64_t result = shared.keys[0];
	__syncthreads();
	return(result);
#else
	uint32_t stride;
	uint64_t result;
	shared.keys[threadIdx.x] = value;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
			shared.keys[threadIdx.x] += shared.keys[threadIdx.x + stride];
		__syncthreads();
	}
	result = shared.keys[0];
	__syncthreads();
	return(result);
#endif
}

static __device__ __forceinline__ void LmSampleHistAdd(uint64_t *hist, uint32_t bucket, uint64_t weight, uint32_t by_mass)
{
#if defined(__CUDA_ARCH__)
	if ( by_mass == 0u )
	{
		const uint32_t active = __activemask();
		const uint32_t same = __match_any_sync(active, bucket);
		const uint32_t lane = threadIdx.x & 31u;
		if ( lane == (uint32_t)(__ffs((int)same) - 1) )
			atomicAdd((unsigned long long *)(hist + bucket), (unsigned long long)__popc(same));
		return;
	}
#endif
	atomicAdd((unsigned long long *)(hist + bucket), (unsigned long long)weight);
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleSelectCompact(LmSampleShared<THREADS> &shared, uint32_t count,
	uint64_t floor_key, uint64_t target, uint32_t by_mass, float maximum, float inverse_temperature,
	uint32_t start_pass, uint64_t prefix, uint64_t above)
{
	const uint32_t copies = LM_SAMPLE_HISTOGRAM_COPIES(THREADS);
	const uint32_t copy = copies > 1u ? (threadIdx.x / 32u) % copies : 0u;
	uint64_t key, weight, match_key = 0u;
	uint32_t pass, digit, shift, index, matches, match_weighted = 0u;
	for (pass = start_pass; pass < LM_SAMPLE_PASSES; pass++)
	{
		shift = 64u - LM_SAMPLE_DIGIT_BITS * (pass + 1u);
		for (index = threadIdx.x; index < copies * LM_SAMPLE_DIGITS; index += THREADS)
			shared.histogram[index / LM_SAMPLE_DIGITS][index % LM_SAMPLE_DIGITS] = 0u;
		if ( threadIdx.x == 0u )
			shared.unique_count = 0u;
		__syncthreads();
		matches = 0u;
		for (index = threadIdx.x; index < count; index += THREADS)
		{
			key = shared.compact_keys[index];
			if ( key < floor_key || (pass != 0u && (key >> (shift + LM_SAMPLE_DIGIT_BITS)) != (prefix >> (shift + LM_SAMPLE_DIGIT_BITS))) )
				continue;
			weight = by_mass != 0u ? SparkMathSampleMass(shared.compact_logits[index], maximum, inverse_temperature) : 1u;
			matches++;
			match_key = key;
			match_weighted = weight != 0u ? 1u : 0u;
			if ( weight == 0u )
				continue;
			LmSampleHistAdd(shared.histogram[copy], (uint32_t)((key >> shift) & (LM_SAMPLE_DIGITS - 1u)), weight, by_mass);
		}
		if ( matches != 0u )
			atomicAdd(&shared.unique_count, matches);
		if ( matches == 1u )
		{
			shared.unique_key = match_key;
			shared.unique_weighted = match_weighted;
		}
		__syncthreads();
		if ( shared.unique_count == 1u && shared.unique_weighted != 0u )
		{
			key = shared.unique_key;
			__syncthreads();
			return(key);
		}
		if ( copies > 1u )
		{
			for (digit = threadIdx.x; digit < LM_SAMPLE_DIGITS; digit += THREADS)
				for (index = 1u; index < copies; index++)
					shared.histogram[0][digit] += shared.histogram[index][digit];
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			uint64_t running = above;
			uint32_t chosen = 0u, found = 0u;
			for (digit = LM_SAMPLE_DIGITS; digit-- > 0u;)
			{
				if ( shared.histogram[0][digit] == 0u )
					continue;
				if ( running + shared.histogram[0][digit] >= target )
				{
					chosen = digit;
					found = 1u;
					break;
				}
				running += shared.histogram[0][digit];
			}
			if ( found == 0u )
			{
				for (digit = 0u; digit < LM_SAMPLE_DIGITS; digit++)
					if ( shared.histogram[0][digit] != 0u )
					{
						chosen = digit;
						break;
					}
			}
			shared.choice[0] = prefix | ((uint64_t)chosen << shift);
			shared.choice[1] = running;
		}
		__syncthreads();
		prefix = shared.choice[0];
		above = shared.choice[1];
		__syncthreads();
	}
	return(prefix);
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleSelect(LmSampleShared<THREADS> &shared, const LmSampleVocab &vocab, uint32_t row,
	uint64_t floor_key, uint64_t target, uint32_t by_mass, float maximum, float inverse_temperature)
{
	const uint32_t copies = LM_SAMPLE_HISTOGRAM_COPIES(THREADS);
	const uint32_t copy = copies > 1u ? (threadIdx.x / 32u) % copies : 0u;
	const bool single_shard = vocab.shard_tokens >= vocab.vocabulary;
	const float *row_logits = single_shard ? (vocab.logits + (uint64_t)row * vocab.row_stride) : 0;
	const bool vec_ok = single_shard && LmIsAligned16(row_logits);
	uint64_t prefix = 0u, above = 0u, key, weight, match_key = 0u;
	uint32_t pass, digit, shift, index, matches, match_weighted = 0u;
	uint32_t try_compact = 0u;
	LmSampleCursor cursor;
	for (pass = 0u; pass < LM_SAMPLE_PASSES; pass++)
	{
		shift = 64u - LM_SAMPLE_DIGIT_BITS * (pass + 1u);
		for (index = threadIdx.x; index < copies * LM_SAMPLE_DIGITS; index += THREADS)
			shared.histogram[index / LM_SAMPLE_DIGITS][index % LM_SAMPLE_DIGITS] = 0u;
		if ( threadIdx.x == 0u )
		{
			shared.unique_count = 0u;
			shared.compact_count = 0u;
			shared.compact_overflow = 0u;
		}
		__syncthreads();
		matches = 0u;
		const uint64_t prefix_hi = prefix >> (shift + LM_SAMPLE_DIGIT_BITS);
		auto process_one = [&](float logit, uint32_t tok)
		{
			key = SparkMathSampleKey(logit, tok);
			if ( key < floor_key || (pass != 0u && (key >> (shift + LM_SAMPLE_DIGIT_BITS)) != prefix_hi) )
				return;
			if ( try_compact != 0u )
			{
				const uint32_t slot = atomicAdd(&shared.compact_count, 1u);
				if ( slot < LM_SAMPLE_COMPACT_MAX )
				{
					shared.compact_keys[slot] = key;
					shared.compact_logits[slot] = logit;
				}
				else
					shared.compact_overflow = 1u;
			}
			weight = by_mass != 0u ? SparkMathSampleMass(logit, maximum, inverse_temperature) : 1u;
			matches++;
			match_key = key;
			match_weighted = weight != 0u ? 1u : 0u;
			if ( weight == 0u )
				return;
			LmSampleHistAdd(shared.histogram[copy], (uint32_t)((key >> shift) & (LM_SAMPLE_DIGITS - 1u)), weight, by_mass);
		};
		if ( vec_ok )
		{
			const uint32_t vecs = vocab.vocabulary >> 2u;
			const float4 *v4 = (const float4 *)row_logits;
			for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
			{
				const float4 x = __ldg(v4 + v);
				const uint32_t base = v << 2u;
				process_one(x.x, base);
				process_one(x.y, base + 1u);
				process_one(x.z, base + 2u);
				process_one(x.w, base + 3u);
			}
			for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
				process_one(__ldg(row_logits + tok), tok);
		}
		else
		{
			for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
				process_one(LmSampleCursorLogit(vocab, row, cursor), cursor.token);
		}
		if ( matches != 0u )
			atomicAdd(&shared.unique_count, matches);
		if ( matches == 1u )
		{
			shared.unique_key = match_key;
			shared.unique_weighted = match_weighted;
		}
		__syncthreads();
		if ( shared.unique_count == 1u && shared.unique_weighted != 0u )
		{
			key = shared.unique_key;
			__syncthreads();
			return(key);
		}
		if ( copies > 1u )
		{
			for (digit = threadIdx.x; digit < LM_SAMPLE_DIGITS; digit += THREADS)
				for (index = 1u; index < copies; index++)
					shared.histogram[0][digit] += shared.histogram[index][digit];
			__syncthreads();
		}
		if ( threadIdx.x == 0u )
		{
			uint64_t running = above;
			uint32_t chosen = 0u, found = 0u;
			for (digit = LM_SAMPLE_DIGITS; digit-- > 0u;)
			{
				if ( shared.histogram[0][digit] == 0u )
					continue;
				if ( running + shared.histogram[0][digit] >= target )
				{
					chosen = digit;
					found = 1u;
					break;
				}
				running += shared.histogram[0][digit];
			}
			if ( found == 0u )
			{
				for (digit = 0u; digit < LM_SAMPLE_DIGITS; digit++)
					if ( shared.histogram[0][digit] != 0u )
					{
						chosen = digit;
						break;
					}
			}
			shared.choice[0] = prefix | ((uint64_t)chosen << shift);
			shared.choice[1] = running;
			shared.picked_count = (by_mass == 0u && shared.histogram[0][chosen] <= LM_SAMPLE_COMPACT_MAX) ? 1u : 0u;
		}
		__syncthreads();
		prefix = shared.choice[0];
		above = shared.choice[1];
		const uint32_t compact_ok = (try_compact != 0u && shared.compact_overflow == 0u && shared.compact_count <= LM_SAMPLE_COMPACT_MAX) ? 1u : 0u;
		const uint32_t compact_n = shared.compact_count;
		const uint32_t next_compact = shared.picked_count;
		__syncthreads();
		if ( compact_ok != 0u && pass + 1u < LM_SAMPLE_PASSES )
			return(LmSampleSelectCompact<THREADS>(shared, compact_n, floor_key, target, by_mass, maximum, inverse_temperature, pass + 1u, prefix, above));
		try_compact = next_compact;
	}
	return(prefix);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmSampleTopLogprobs(LmSampleShared<THREADS> &shared, const LmSampleVocab &vocab, uint32_t row,
	uint32_t count, float lse, SparkSamplingLogprob *out)
{
	uint64_t threshold, key;
	uint32_t token, index, other, slot;
	LmSampleCursor cursor;
	threshold = LmSampleSelect<THREADS>(shared, vocab, row, 0u, count, 0u, 0.0f, 0.0f);
	if ( threadIdx.x == 0u )
		shared.picked_count = 0u;
	__syncthreads();
	for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
	{
		key = SparkMathSampleKey(LmSampleCursorLogit(vocab, row, cursor), cursor.token);
		if ( key >= threshold )
		{
			slot = atomicAdd(&shared.picked_count, 1u);
			if ( slot < count )
				shared.picked[slot] = key;
		}
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		const uint32_t picked = shared.picked_count < count ? shared.picked_count : count;
		for (index = 1u; index < picked; index++)
			for (other = index; other > 0u && shared.picked[other] > shared.picked[other - 1u]; other--)
			{
				key = shared.picked[other];
				shared.picked[other] = shared.picked[other - 1u];
				shared.picked[other - 1u] = key;
			}
		for (index = 0u; index < count; index++)
		{
			if ( index < picked )
			{
				token = SparkMathSampleKeyToken(shared.picked[index]);
				out[index].token = token;
				out[index].logprob = LmSampleLogit(vocab, row, token) - lse;
			}
			else
			{
				out[index].token = SPARK_SAMPLING_NO_TOKEN;
				out[index].logprob = 0.0f;
			}
		}
	}
	__syncthreads();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSampleRowsKernel(const LmSampleVocab vocab, const LmSampleRows rows)
{
	__shared__ LmSampleShared<THREADS> shared;
	const uint32_t row = blockIdx.x;
	uint32_t chosen;
	uint64_t best, key, floor_key, target, total;
	float maximum = -INFINITY, sum = 0.0f, lse, logit;
	SparkRowSampling rule;
	LmSampleCursor cursor;
	LmDependentWait();
	if ( row >= rows.rows )
	{
		LmDependentRelease();
		return;
	}
	rule = rows.rules[row];
	const bool single_shard = vocab.shard_tokens >= vocab.vocabulary;
	const float *row_logits = single_shard ? (vocab.logits + (uint64_t)row * vocab.row_stride) : 0;
	const bool vec_ok = single_shard && LmIsAligned16(row_logits);
	const bool need_logprobs = rows.logprobs_out != 0 && rule.logprobs != 0u;
	const bool need_max = need_logprobs || (rule.inverse_temperature != 0.0f && rule.nucleus != 0.0f);
	if ( need_max )
	{
		if ( vec_ok )
		{
			const uint32_t vecs = vocab.vocabulary >> 2u;
			const float4 *v4 = (const float4 *)row_logits;
			for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
			{
				const float4 x = __ldg(v4 + v);
				maximum = fmaxf(maximum, fmaxf(fmaxf(x.x, x.y), fmaxf(x.z, x.w)));
			}
			for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
				maximum = fmaxf(maximum, __ldg(row_logits + tok));
		}
		else
		{
			for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
				maximum = fmaxf(maximum, LmSampleCursorLogit(vocab, row, cursor));
		}
		maximum = LmSampleBlockMax<THREADS>(shared, maximum);
	}
	lse = maximum;
	if ( need_logprobs )
	{
		for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
			sum += expf(LmSampleCursorLogit(vocab, row, cursor) - maximum);
		lse = maximum + logf(LmSampleBlockSum<THREADS>(shared, sum));
	}
	best = 0u;
	if ( rule.inverse_temperature == 0.0f && rows.greedy_tokens != 0 )
		best = SparkMathSampleKey(0.0f, rows.greedy_tokens[rows.source_rows != 0 ? rows.source_rows[row] : row]);
	else if ( rule.inverse_temperature == 0.0f )
	{
		if ( vec_ok )
		{
			const uint32_t vecs = vocab.vocabulary >> 2u;
			const float4 *v4 = (const float4 *)row_logits;
			for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
			{
				const float4 x = __ldg(v4 + v);
				const uint32_t base = v << 2u;
				const uint64_t key0 = SparkMathSampleKey(x.x, base);
				const uint64_t key1 = SparkMathSampleKey(x.y, base + 1u);
				const uint64_t key2 = SparkMathSampleKey(x.z, base + 2u);
				const uint64_t key3 = SparkMathSampleKey(x.w, base + 3u);
				best = key0 > best ? key0 : best;
				best = key1 > best ? key1 : best;
				best = key2 > best ? key2 : best;
				best = key3 > best ? key3 : best;
			}
			for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
			{
				key = SparkMathSampleKey(__ldg(row_logits + tok), tok);
				best = key > best ? key : best;
			}
		}
		else
		{
			for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
			{
				key = SparkMathSampleKey(LmSampleCursorLogit(vocab, row, cursor), cursor.token);
				best = key > best ? key : best;
			}
		}
	}
	else
	{
		floor_key = 0u;
		uint32_t has_compact = 0u;
		uint32_t compact_n = 0u;
		if ( rule.top_k != 0u && rule.top_k < vocab.vocabulary )
			floor_key = LmSampleSelect<THREADS>(shared, vocab, row, 0u, rule.top_k, 0u, 0.0f, 0.0f);
		if ( rule.top_k != 0u && rule.top_k <= LM_SAMPLE_COMPACT_MAX && rule.top_k < vocab.vocabulary )
		{
			if ( threadIdx.x == 0u )
			{
				shared.compact_count = 0u;
				shared.compact_overflow = 0u;
			}
			__syncthreads();
			total = 0u;
			auto collect_one = [&](float l, uint32_t tok)
			{
				const uint64_t k = SparkMathSampleKey(l, tok);
				if ( k < floor_key )
					return;
				if ( rule.nucleus != 0.0f )
					total += SparkMathSampleMass(l, maximum, rule.inverse_temperature);
				const uint32_t slot = atomicAdd(&shared.compact_count, 1u);
				if ( slot < LM_SAMPLE_COMPACT_MAX )
				{
					shared.compact_keys[slot] = k;
					shared.compact_logits[slot] = l;
				}
				else
					shared.compact_overflow = 1u;
			};
			if ( vec_ok )
			{
				const uint32_t vecs = vocab.vocabulary >> 2u;
				const float4 *v4 = (const float4 *)row_logits;
				for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
				{
					const float4 x = __ldg(v4 + v);
					const uint32_t base = v << 2u;
					collect_one(x.x, base);
					collect_one(x.y, base + 1u);
					collect_one(x.z, base + 2u);
					collect_one(x.w, base + 3u);
				}
				for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
					collect_one(__ldg(row_logits + tok), tok);
			}
			else
			{
				for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
					collect_one(LmSampleCursorLogit(vocab, row, cursor), cursor.token);
			}
			__syncthreads();
			has_compact = (shared.compact_overflow == 0u && shared.compact_count <= LM_SAMPLE_COMPACT_MAX) ? 1u : 0u;
			compact_n = shared.compact_count;
			if ( rule.nucleus != 0.0f )
			{
				total = LmSampleBlockSumU64<THREADS>(shared, total);
				target = (uint64_t)((double)total * (double)rule.nucleus);
				target = target != 0u ? target : 1u;
				if ( has_compact != 0u )
					key = LmSampleSelectCompact<THREADS>(shared, compact_n, floor_key, target, 1u, maximum, rule.inverse_temperature, 0u, 0u, 0u);
				else
					key = LmSampleSelect<THREADS>(shared, vocab, row, floor_key, target, 1u, maximum, rule.inverse_temperature);
				floor_key = key > floor_key ? key : floor_key;
			}
		}
		else if ( rule.nucleus != 0.0f )
		{
			total = 0u;
			if ( vec_ok )
			{
				const uint32_t vecs = vocab.vocabulary >> 2u;
				const float4 *v4 = (const float4 *)row_logits;
				for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
				{
					const float4 x = __ldg(v4 + v);
					const uint32_t base = v << 2u;
					if ( SparkMathSampleKey(x.x, base) >= floor_key ) total += SparkMathSampleMass(x.x, maximum, rule.inverse_temperature);
					if ( SparkMathSampleKey(x.y, base + 1u) >= floor_key ) total += SparkMathSampleMass(x.y, maximum, rule.inverse_temperature);
					if ( SparkMathSampleKey(x.z, base + 2u) >= floor_key ) total += SparkMathSampleMass(x.z, maximum, rule.inverse_temperature);
					if ( SparkMathSampleKey(x.w, base + 3u) >= floor_key ) total += SparkMathSampleMass(x.w, maximum, rule.inverse_temperature);
				}
				for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
				{
					logit = __ldg(row_logits + tok);
					if ( SparkMathSampleKey(logit, tok) >= floor_key )
						total += SparkMathSampleMass(logit, maximum, rule.inverse_temperature);
				}
			}
			else
			{
				for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
				{
					logit = LmSampleCursorLogit(vocab, row, cursor);
					if ( SparkMathSampleKey(logit, cursor.token) >= floor_key )
						total += SparkMathSampleMass(logit, maximum, rule.inverse_temperature);
				}
			}
			total = LmSampleBlockSumU64<THREADS>(shared, total);
			target = (uint64_t)((double)total * (double)rule.nucleus);
			target = target != 0u ? target : 1u;
			key = LmSampleSelect<THREADS>(shared, vocab, row, floor_key, target, 1u, maximum, rule.inverse_temperature);
			floor_key = key > floor_key ? key : floor_key;
		}
		const uint32_t pos = rows.positions[row];
		if ( has_compact != 0u )
		{
			for (uint32_t idx = threadIdx.x; idx < compact_n; idx += THREADS)
			{
				const uint64_t ck = shared.compact_keys[idx];
				if ( ck < floor_key )
					continue;
				const uint32_t tok = SparkMathSampleKeyToken(ck);
				logit = shared.compact_logits[idx];
				key = SparkMathSampleKey(fmaf(logit, rule.inverse_temperature, SparkMathGumbelNoise(rule.seed, pos, tok)), tok);
				best = key > best ? key : best;
			}
		}
		else if ( vec_ok )
		{
			const uint32_t vecs = vocab.vocabulary >> 2u;
			const float4 *v4 = (const float4 *)row_logits;
			auto score_one = [&](float l, uint32_t tok)
			{
				if ( SparkMathSampleKey(l, tok) < floor_key )
					return;
				const uint64_t gk = SparkMathSampleKey(fmaf(l, rule.inverse_temperature, SparkMathGumbelNoise(rule.seed, pos, tok)), tok);
				best = gk > best ? gk : best;
			};
			for (uint32_t v = threadIdx.x; v < vecs; v += THREADS)
			{
				const float4 x = __ldg(v4 + v);
				const uint32_t base = v << 2u;
				score_one(x.x, base);
				score_one(x.y, base + 1u);
				score_one(x.z, base + 2u);
				score_one(x.w, base + 3u);
			}
			for (uint32_t tok = (vecs << 2u) + threadIdx.x; tok < vocab.vocabulary; tok += THREADS)
				score_one(__ldg(row_logits + tok), tok);
		}
		else
		{
			for (cursor = LmSampleCursorStart(vocab, threadIdx.x); cursor.token < vocab.vocabulary; LmSampleCursorAdvance(vocab, cursor, THREADS))
			{
				logit = LmSampleCursorLogit(vocab, row, cursor);
				if ( SparkMathSampleKey(logit, cursor.token) < floor_key )
					continue;
				key = SparkMathSampleKey(fmaf(logit, rule.inverse_temperature, SparkMathGumbelNoise(rule.seed, pos, cursor.token)), cursor.token);
				best = key > best ? key : best;
			}
		}
	}
	best = LmSampleBlockMaxKey<THREADS>(shared, best);
	chosen = SparkMathSampleKeyToken(best);
	const uint64_t logprob_base = (uint64_t)(rows.logprob_rows != 0 ? rows.logprob_rows[row] : row) * SPARK_SAMPLING_MAX_LOGPROBS;
	if ( threadIdx.x == 0u )
	{
		const uint32_t target_row = rows.source_rows != 0 ? rows.source_rows[row] : row;
		rows.token_out[target_row] = chosen;
		if ( rows.logit_out != 0 )
			rows.logit_out[target_row] = LmSampleLogit(vocab, row, chosen);
		if ( need_logprobs )
		{
			rows.logprobs_out[logprob_base].token = chosen;
			rows.logprobs_out[logprob_base].logprob = LmSampleLogit(vocab, row, chosen) - lse;
		}
	}
	__syncthreads();
	if ( rows.logprobs_out != 0 && rule.logprobs > 1u )
		LmSampleTopLogprobs<THREADS>(shared, vocab, row, rule.logprobs - 1u, lse, rows.logprobs_out + logprob_base + 1u);
	LmDependentRelease();
}
