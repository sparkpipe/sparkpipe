#pragma once

#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t GROUPS, uint32_t TOPK_GROUPS, uint32_t TOPK>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkGroupLimitedKernel(const float *__restrict__ logits, const float *__restrict__ selection_bias, uint32_t *__restrict__ out_indices,
	float *__restrict__ out_values, float scale, uint32_t *__restrict__ fault)
{
	constexpr uint32_t PER_GROUP = EXPERTS / GROUPS, WARPS = THREADS / 32u;
	static_assert(EXPERTS % GROUPS == 0u && PER_GROUP >= 2u && TOPK_GROUPS <= GROUPS && TOPK <= TOPK_GROUPS * PER_GROUP &&
		THREADS % 32u == 0u && GROUPS <= THREADS, "groups tile the experts and every pick has a candidate");
	__shared__ float score[EXPERTS];
	__shared__ float selection[EXPERTS];
	__shared__ float group_score[GROUPS];
	__shared__ uint32_t group_open[GROUPS];
	__shared__ uint32_t taken[EXPERTS];
	__shared__ unsigned long long warp_best[WARPS];
	__shared__ uint32_t chosen[TOPK];
	const uint32_t row = blockIdx.x, lane = threadIdx.x % 32u, warp = threadIdx.x / 32u;
	for ( uint32_t expert = threadIdx.x; expert < EXPERTS; expert += THREADS )
	{
		const float value = SparkMathSigmoid(logits[(uint64_t)row * EXPERTS + expert]);
		score[expert] = value;
		selection[expert] = value + (selection_bias != 0 ? selection_bias[expert] : 0.0f);
		taken[expert] = 0u;
	}
	__syncthreads();
	if ( threadIdx.x < GROUPS )
	{
		float first = -INFINITY, second = -INFINITY;
		for ( uint32_t index = 0u; index < PER_GROUP; ++index )
		{
			const float value = selection[threadIdx.x * PER_GROUP + index];
			if ( value > first )
			{
				second = first;
				first = value;
			}
			else if ( value > second )
				second = value;
		}
		group_score[threadIdx.x] = first + second;
		group_open[threadIdx.x] = 0u;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
		for ( uint32_t pick = 0u; pick < TOPK_GROUPS; ++pick )
		{
			uint32_t best = GROUPS;
			for ( uint32_t group = 0u; group < GROUPS; ++group )
				if ( group_open[group] == 0u && (best == GROUPS || group_score[group] > group_score[best]) )
					best = group;
			group_open[best] = 1u;
		}
	__syncthreads();
	for ( uint32_t pick = 0u; pick < TOPK; ++pick )
	{
		unsigned long long best = 0ull;
		for ( uint32_t expert = threadIdx.x; expert < EXPERTS; expert += THREADS )
			if ( group_open[expert / PER_GROUP] != 0u && taken[expert] == 0u )
			{
				const unsigned long long candidate = ((unsigned long long)LmTopkRouteKey(selection[expert] + 0.0f) << 32u) | (0xffffffffu - expert);
				best = candidate > best ? candidate : best;
			}
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
			best = other > best ? other : best;
		}
		if ( lane == 0u )
			warp_best[warp] = best;
		__syncthreads();
		if ( threadIdx.x == 0u )
		{
			for ( uint32_t other = 1u; other < WARPS; ++other )
				best = warp_best[other] > best ? warp_best[other] : best;
			chosen[pick] = 0xffffffffu - (uint32_t)best;
			taken[chosen[pick]] = 1u;
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
		}
		__syncthreads();
	}
	if ( threadIdx.x == 0u )
	{
		float total = 0.0f;
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			total += score[chosen[pick]];
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			out_indices[(uint64_t)row * TOPK + pick] = chosen[pick];
			out_values[(uint64_t)row * TOPK + pick] = score[chosen[pick]] / (total + 1e-20f) * scale;
		}
	}
}

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t GROUPS, uint32_t TOPK_GROUPS, uint32_t TOPK>
static inline cudaError_t LmTopkGroupLimitedLaunch(uint32_t rows, const float *logits, const float *selection_bias, uint32_t *out_indices,
	float *out_values, float scale, uint32_t *fault, cudaStream_t stream)
{
	if ( rows == 0u || logits == 0 || out_indices == 0 || out_values == 0 || fault == 0 )
		return cudaErrorInvalidValue;
	LM_LAUNCH((LmTopkGroupLimitedKernel<THREADS,EXPERTS,GROUPS,TOPK_GROUPS,TOPK>), rows, THREADS, 0, stream, logits, selection_bias, out_indices,
		out_values, scale, fault);
	return cudaPeekAtLastError();
}
