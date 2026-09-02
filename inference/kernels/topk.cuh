#pragma once


#include "inference/kernels/norm.cuh"
#include <stdint.h>

#define LM_TOPK_SCORE_IDENTITY 0u
#define LM_TOPK_SCORE_SIGMOID 1u
#define LM_TOPK_SCORE_SQRT_SOFTPLUS 2u

template<uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ float LmTopkScore(const float *scores, const uint16_t *logits_bf16, uint64_t index)
{
	float value = logits_bf16 != 0
		? LmBf16ToFloat(logits_bf16[index])
		: scores[index];
	if ( SCORE_TRANSFORM == LM_TOPK_SCORE_SIGMOID )
		return(1.0f / (1.0f + __expf(-value)));
	if ( SCORE_TRANSFORM == LM_TOPK_SCORE_SQRT_SOFTPLUS )
	{
		float softplus = value > 20.0f ? value : __logf(1.0f + __expf(value));
		return(sqrtf(softplus > 0.0f ? softplus : 0.0f));
	}
	return(value);
}

static __device__ __forceinline__ uint32_t LmTopkKey(float value)
{
	uint32_t bits = __float_as_uint(value);
	return(bits ^ ((bits >> 31u) ? 0xffffffffu : 0x80000000u));
}

static __device__ __forceinline__ float LmTopkValue(uint32_t key)
{
	uint32_t bits = key ^ ((key >> 31u) ? 0x80000000u : 0xffffffffu);
	return(__uint_as_float(bits));
}

#define LM_TOPK_SMALL_LIMIT 1024u

#define LM_TOPK_MAX_GROUPS 16u

template<uint32_t THREADS, uint32_t K, bool RENORMALISE = false, uint32_t GROUPS = 1u, uint32_t TOP_GROUPS = 1u, uint32_t SCORE_TRANSFORM = LM_TOPK_SCORE_IDENTITY>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkSmallKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale)
{
	extern __shared__ uint32_t lm_topk_shared[];
	uint32_t *keys = lm_topk_shared;
	uint32_t *slots = lm_topk_shared + LM_TOPK_SMALL_LIMIT;
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t index,size,stride;
	for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
	{
		keys[index] = index < n ? LmTopkKey(LmTopkScore<SCORE_TRANSFORM>(scores, logits_bf16,
			base + index) + (selection_bias != 0 ? selection_bias[index] : 0.0f)) : 0u;
		slots[index] = index;
	}
	__syncthreads();
	if ( GROUPS > 1u && GROUPS > TOP_GROUPS )
	{
		__shared__ uint32_t group_key[LM_TOPK_MAX_GROUPS];
		uint32_t group,per_group = n / GROUPS,cut;
		for (group = threadIdx.x; group < GROUPS; group += THREADS)
		{
			uint32_t best = 0u,second = 0u,member;
			for (member = 0u; member < per_group; ++member)
			{
				uint32_t value = keys[(group * per_group) + member];
				if ( value > best ) { second = best; best = value; }
				else if ( value > second ) second = value;
			}
			group_key[group] = LmTopkKey(LmTopkValue(best) + LmTopkValue(second));
		}
		__syncthreads();
		for (group = threadIdx.x; group < GROUPS; group += THREADS)
		{
			uint32_t better = 0u,other;
			for (other = 0u; other < GROUPS; ++other)
				if ( group_key[other] > group_key[group]
					|| (group_key[other] == group_key[group] && other < group) )
					++better;
			if ( better >= TOP_GROUPS )
				group_key[group] = 0xffffffffu;
		}
		(void)cut;
		__syncthreads();
		for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
			if ( index < n && group_key[index / per_group] != 0xffffffffu )
				keys[index] = 0u;
		__syncthreads();
	}
	for (size = 2u; size <= LM_TOPK_SMALL_LIMIT; size <<= 1u)
		for (stride = size >> 1u; stride > 0u; stride >>= 1u)
		{
			for (index = threadIdx.x; index < LM_TOPK_SMALL_LIMIT; index += THREADS)
			{
				uint32_t partner = index ^ stride;
				if ( partner > index )
				{
					bool descending = ((index & size) == 0u);
					bool swap = descending ? (keys[index] < keys[partner])
						: (keys[index] > keys[partner]);
					if ( swap )
					{
						uint32_t tk = keys[index],ts = slots[index];
						keys[index] = keys[partner];
						slots[index] = slots[partner];
						keys[partner] = tk;
						slots[partner] = ts;
					}
				}
			}
			__syncthreads();
		}
	for (index = threadIdx.x; index < K; index += THREADS)
	{
		out_indices[((uint64_t)blockIdx.x * K) + index] = slots[index];
		if ( out_values != 0 )
			out_values[((uint64_t)blockIdx.x * K) + index] =
				LmTopkScore<SCORE_TRANSFORM>(scores, logits_bf16, base + slots[index]);
	}
	if ( RENORMALISE && out_values != 0 )
	{
		__shared__ float reduction[LM_TOPK_SMALL_LIMIT];
		float mine = 0.0f,total;
		__syncthreads();
		for (index = threadIdx.x; index < K; index += THREADS)
			mine += out_values[((uint64_t)blockIdx.x * K) + index];
		reduction[threadIdx.x] = mine;
		__syncthreads();
		for (index = THREADS / 2u; index > 0u; index >>= 1)
		{
			if ( threadIdx.x < index && threadIdx.x + index < THREADS )
				reduction[threadIdx.x] += reduction[threadIdx.x + index];
			__syncthreads();
		}
		total = reduction[0] + 1e-20f;
		for (index = threadIdx.x; index < K; index += THREADS)
			out_values[((uint64_t)blockIdx.x * K) + index] =
				(out_values[((uint64_t)blockIdx.x * K) + index] / total) * mixture_scale;
	}
	else if ( out_values != 0 && mixture_scale != 1.0f )
	{
		for (index = threadIdx.x; index < K; index += THREADS)
			out_values[((uint64_t)blockIdx.x * K) + index] *= mixture_scale;
	}
}

#define LM_TOPK_RADIX_BITS 8u
#define LM_TOPK_BUCKETS (1u << LM_TOPK_RADIX_BITS)

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkHistogramKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, uint32_t *__restrict__ threshold_out)
{
	__shared__ uint32_t histogram[LM_TOPK_BUCKETS];
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t index,running;
	for (index = threadIdx.x; index < LM_TOPK_BUCKETS; index += THREADS)
		histogram[index] = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < n; index += THREADS)
		atomicAdd(&histogram[LmTopkKey(scores[base + index]) >> (32u - LM_TOPK_RADIX_BITS)],1u);
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		running = 0u;
		for (index = LM_TOPK_BUCKETS; index > 0u; --index)
		{
			running += histogram[index - 1u];
			if ( running >= k )
			{
				threshold_out[blockIdx.x] = index - 1u;
				return;
			}
		}
		threshold_out[blockIdx.x] = 0u;
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkGatherKernel(const float *__restrict__ scores, uint32_t n, uint32_t k, const uint32_t *__restrict__ threshold, uint32_t *__restrict__ out_indices, uint32_t *__restrict__ out_count)
{
	__shared__ uint32_t emitted;
	uint64_t base = (uint64_t)blockIdx.x * n;
	uint32_t bucket = threshold[blockIdx.x],index;
	if ( threadIdx.x == 0u )
		emitted = 0u;
	__syncthreads();
	for (index = threadIdx.x; index < n; index += THREADS)
	{
		if ( (LmTopkKey(scores[base + index]) >> (32u - LM_TOPK_RADIX_BITS)) >= bucket )
		{
			uint32_t slot = atomicAdd(&emitted,1u);
			if ( slot < k )
				out_indices[((uint64_t)blockIdx.x * k) + slot] = index;
		}
	}
	__syncthreads();
	if ( threadIdx.x == 0u && out_count != 0 )
		out_count[blockIdx.x] = emitted < k ? emitted : k;
}
