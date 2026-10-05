#pragma once

#include "inference/kernels/head.cuh"
#include "include/sparkpipe/spark_sampling.h"
#include <stdint.h>

#define LM_SAMPLE_DIGIT_BITS 8u
#define LM_SAMPLE_DIGITS (1u << LM_SAMPLE_DIGIT_BITS)
#define LM_SAMPLE_PASSES (64u / LM_SAMPLE_DIGIT_BITS)
#define LM_SAMPLE_MASS_ONE 4294967296.0f

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

static __host__ __device__ __forceinline__ uint32_t LmSampleOrdered(float value)
{
	union { float f; uint32_t u; } bits;
	if ( value != value )
		return(0u);
	bits.f = value;
	return(bits.u ^ ((bits.u & 0x80000000u) != 0u ? 0xffffffffu : 0x80000000u));
}

static __host__ __device__ __forceinline__ uint64_t LmSampleKey(float value, uint32_t token)
{
	return(((uint64_t)LmSampleOrdered(value) << 32u) | (uint64_t)(0xffffffffu - token));
}

static __host__ __device__ __forceinline__ uint32_t LmSampleKeyToken(uint64_t key)
{
	return(0xffffffffu - (uint32_t)key);
}

static __device__ __forceinline__ float LmSampleLogit(const LmSampleVocab &vocab, uint32_t row, uint32_t token)
{
	const uint32_t shard = token / vocab.shard_tokens;
	return(vocab.logits[shard * vocab.shard_stride + (uint64_t)row * vocab.row_stride + (token - shard * vocab.shard_tokens)]);
}

template<uint32_t THREADS>
struct LmSampleShared
{
	float values[THREADS];
	uint64_t keys[THREADS];
	uint64_t histogram[LM_SAMPLE_DIGITS];
	uint64_t choice[2];
	uint64_t picked[SPARK_SAMPLING_MAX_LOGPROBS];
	uint32_t picked_count;
};

template<uint32_t THREADS>
static __device__ __forceinline__ float LmSampleBlockMax(LmSampleShared<THREADS> &shared, float value)
{
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
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmSampleBlockSum(LmSampleShared<THREADS> &shared, float value)
{
	uint32_t stride;
	float result;
	shared.values[threadIdx.x] = value;
	__syncthreads();
	for (stride = THREADS / 2u; stride > 0u; stride >>= 1u)
	{
		if ( threadIdx.x < stride )
			shared.values[threadIdx.x] += shared.values[threadIdx.x + stride];
		__syncthreads();
	}
	result = shared.values[0];
	__syncthreads();
	return(result);
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleBlockMaxKey(LmSampleShared<THREADS> &shared, uint64_t key)
{
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
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleBlockSumU64(LmSampleShared<THREADS> &shared, uint64_t value)
{
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
}

static __device__ __forceinline__ uint64_t LmSampleMass(float logit, float maximum, float inverse_temperature)
{
	return((uint64_t)(expf((logit - maximum) * inverse_temperature) * LM_SAMPLE_MASS_ONE));
}

template<uint32_t THREADS>
static __device__ __forceinline__ uint64_t LmSampleSelect(LmSampleShared<THREADS> &shared, const LmSampleVocab &vocab, uint32_t row,
	uint64_t floor_key, uint64_t target, uint32_t by_mass, float maximum, float inverse_temperature)
{
	uint64_t prefix = 0u, above = 0u, key, weight;
	uint32_t pass, token, digit, shift;
	for (pass = 0u; pass < LM_SAMPLE_PASSES; pass++)
	{
		shift = 64u - LM_SAMPLE_DIGIT_BITS * (pass + 1u);
		for (digit = threadIdx.x; digit < LM_SAMPLE_DIGITS; digit += THREADS)
			shared.histogram[digit] = 0u;
		__syncthreads();
		for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
		{
			const float logit = LmSampleLogit(vocab, row, token);
			key = LmSampleKey(logit, token);
			if ( key < floor_key || (pass != 0u && (key >> (shift + LM_SAMPLE_DIGIT_BITS)) != (prefix >> (shift + LM_SAMPLE_DIGIT_BITS))) )
				continue;
			weight = by_mass != 0u ? LmSampleMass(logit, maximum, inverse_temperature) : 1u;
			atomicAdd((unsigned long long *)&shared.histogram[(key >> shift) & (LM_SAMPLE_DIGITS - 1u)], (unsigned long long)weight);
		}
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			uint64_t running = above;
			uint32_t chosen = 0u, found = 0u;
			for (digit = LM_SAMPLE_DIGITS; digit-- > 0u;)
			{
				if ( shared.histogram[digit] == 0u )
					continue;
				if ( running + shared.histogram[digit] >= target )
				{
					chosen = digit;
					found = 1u;
					break;
				}
				running += shared.histogram[digit];
			}
			if ( found == 0u )
			{
				for (digit = 0u; digit < LM_SAMPLE_DIGITS; digit++)
					if ( shared.histogram[digit] != 0u )
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
static __device__ __forceinline__ void LmSampleTopLogprobs(LmSampleShared<THREADS> &shared, const LmSampleVocab &vocab, uint32_t row,
	uint32_t count, float lse, SparkSamplingLogprob *out)
{
	uint64_t threshold, key;
	uint32_t token, index, other, slot;
	threshold = LmSampleSelect<THREADS>(shared, vocab, row, 0u, count, 0u, 0.0f, 0.0f);
	if ( threadIdx.x == 0u )
		shared.picked_count = 0u;
	__syncthreads();
	for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
	{
		key = LmSampleKey(LmSampleLogit(vocab, row, token), token);
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
				token = LmSampleKeyToken(shared.picked[index]);
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
	uint32_t token, chosen;
	uint64_t best, key, floor_key, target, total;
	float maximum = -INFINITY, sum = 0.0f, lse, logit;
	SparkRowSampling rule;
	if ( row >= rows.rows )
		return;
	rule = rows.rules[row];
	for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
		maximum = fmaxf(maximum, LmSampleLogit(vocab, row, token));
	maximum = LmSampleBlockMax<THREADS>(shared, maximum);
	for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
		sum += expf(LmSampleLogit(vocab, row, token) - maximum);
	lse = maximum + logf(LmSampleBlockSum<THREADS>(shared, sum));
	best = 0u;
	if ( rule.inverse_temperature == 0.0f && rows.greedy_tokens != 0 )
		best = LmSampleKey(0.0f, rows.greedy_tokens[rows.source_rows != 0 ? rows.source_rows[row] : row]);
	else if ( rule.inverse_temperature == 0.0f )
	{
		for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
		{
			key = LmSampleKey(LmSampleLogit(vocab, row, token), token);
			best = key > best ? key : best;
		}
	}
	else
	{
		floor_key = 0u;
		if ( rule.top_k != 0u && rule.top_k < vocab.vocabulary )
			floor_key = LmSampleSelect<THREADS>(shared, vocab, row, 0u, rule.top_k, 0u, 0.0f, 0.0f);
		if ( rule.nucleus != 0.0f )
		{
			total = 0u;
			for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
			{
				logit = LmSampleLogit(vocab, row, token);
				if ( LmSampleKey(logit, token) >= floor_key )
					total += LmSampleMass(logit, maximum, rule.inverse_temperature);
			}
			total = LmSampleBlockSumU64<THREADS>(shared, total);
			target = (uint64_t)((double)total * (double)rule.nucleus);
			target = target != 0u ? target : 1u;
			key = LmSampleSelect<THREADS>(shared, vocab, row, floor_key, target, 1u, maximum, rule.inverse_temperature);
			floor_key = key > floor_key ? key : floor_key;
		}
		for (token = threadIdx.x; token < vocab.vocabulary; token += THREADS)
		{
			logit = LmSampleLogit(vocab, row, token);
			if ( LmSampleKey(logit, token) < floor_key )
				continue;
			key = LmSampleKey(fmaf(logit, rule.inverse_temperature, LmGumbelNoise(rule.seed, rows.positions[row], token)), token);
			best = key > best ? key : best;
		}
	}
	best = LmSampleBlockMaxKey<THREADS>(shared, best);
	chosen = LmSampleKeyToken(best);
	const uint64_t logprob_base = (uint64_t)(rows.logprob_rows != 0 ? rows.logprob_rows[row] : row) * SPARK_SAMPLING_MAX_LOGPROBS;
	if ( threadIdx.x == 0u )
	{
		const uint32_t target_row = rows.source_rows != 0 ? rows.source_rows[row] : row;
		rows.token_out[target_row] = chosen;
		if ( rows.logit_out != 0 )
			rows.logit_out[target_row] = LmSampleLogit(vocab, row, chosen);
		if ( rows.logprobs_out != 0 && rule.logprobs != 0u )
		{
			rows.logprobs_out[logprob_base].token = chosen;
			rows.logprobs_out[logprob_base].logprob = LmSampleLogit(vocab, row, chosen) - lse;
		}
	}
	__syncthreads();
	if ( rows.logprobs_out != 0 && rule.logprobs > 1u )
		LmSampleTopLogprobs<THREADS>(shared, vocab, row, rule.logprobs - 1u, lse, rows.logprobs_out + logprob_base + 1u);
}
