#pragma once


#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSpeculativeVerifyGreedyKernel(const uint32_t *__restrict__ draft_tokens, const uint32_t *__restrict__ target_argmax, uint32_t draft_length, uint32_t *__restrict__ accepted_count, uint32_t *__restrict__ committed_tokens, uint32_t *__restrict__ context_length)
{
	uint32_t sequence = blockIdx.x,index,accepted = 0u;
	uint64_t base = (uint64_t)sequence * draft_length;
	if ( threadIdx.x != 0u )
		return;
	while ( accepted < draft_length
		&& draft_tokens[base + accepted] == target_argmax[base + accepted] )
		++accepted;
	for (index = 0u; index < accepted; ++index)
		committed_tokens[base + index] = draft_tokens[base + index];
	committed_tokens[base + accepted] = target_argmax[base + accepted];
	accepted_count[sequence] = accepted + 1u;
	context_length[sequence] = context_length[sequence] - (draft_length - accepted);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSpeculativeVerifySampledKernel(const uint32_t *__restrict__ draft_tokens, const float *__restrict__ draft_probability, const float *__restrict__ target_probability, const float *__restrict__ uniform, uint32_t draft_length, uint32_t vocabulary, uint32_t *__restrict__ accepted_count, uint32_t *__restrict__ committed_tokens, uint32_t *__restrict__ context_length)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ uint32_t shared_accepted;
	uint32_t sequence = blockIdx.x,step,token,index;
	uint64_t draft_base = (uint64_t)sequence * draft_length;
	float residual_total,pick,running;
	if ( threadIdx.x == 0u )
		shared_accepted = 0u;
	__syncthreads();
	for (step = 0u; step < draft_length; ++step)
	{
		token = draft_tokens[draft_base + step];
		if ( threadIdx.x == 0u )
		{
			float pd = draft_probability[((draft_base + step) * vocabulary) + token];
			float pt = target_probability[((draft_base + step) * vocabulary) + token];
			shared_accepted = (uniform[draft_base + step] < fminf(1.0f,pt / fmaxf(pd,1e-20f)))
				? 1u : 0u;
		}
		__syncthreads();
		if ( shared_accepted == 0u )
			break;
		if ( threadIdx.x == 0u )
			committed_tokens[draft_base + step] = token;
		__syncthreads();
	}
	residual_total = 0.0f;
	for (index = threadIdx.x; index < vocabulary; index += THREADS)
		residual_total += fmaxf(0.0f,
			target_probability[((draft_base + step) * vocabulary) + index]
			- draft_probability[((draft_base + step) * vocabulary) + index]);
	residual_total = LmBlockSum<THREADS>(residual_total,reduction);
	if ( threadIdx.x == 0u )
	{
		pick = uniform[draft_base + draft_length] * fmaxf(residual_total,1e-20f);
		running = 0.0f;
		for (index = 0u; index < vocabulary; ++index)
		{
			running += fmaxf(0.0f,
				target_probability[((draft_base + step) * vocabulary) + index]
				- draft_probability[((draft_base + step) * vocabulary) + index]);
			if ( running >= pick )
				break;
		}
		committed_tokens[draft_base + step] = index < vocabulary ? index : 0u;
		accepted_count[sequence] = step + 1u;
		context_length[sequence] = context_length[sequence] - (draft_length - step);
	}
}
