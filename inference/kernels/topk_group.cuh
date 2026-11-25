#pragma once

#include <math.h>
#include <stdint.h>

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/topk.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t GROUPS, uint32_t TOPK_GROUPS, uint32_t TOPK>
__global__ __launch_bounds__(THREADS, 1)
void LmTopkGroupLimitedKernel(const float *__restrict__ logits, const float *__restrict__ selection_bias, uint32_t *__restrict__ out_indices,
	float *__restrict__ out_values, float scale, uint32_t *__restrict__ fault)
{
	constexpr uint32_t PER_GROUP = EXPERTS / GROUPS, WARPS = THREADS / 32u;
	constexpr uint32_t PER_THREAD = (EXPERTS + THREADS - 1u) / THREADS;
	static_assert(EXPERTS % GROUPS == 0u && PER_GROUP >= 2u && TOPK_GROUPS <= GROUPS && TOPK <= TOPK_GROUPS * PER_GROUP &&
		THREADS % 32u == 0u && GROUPS <= THREADS, "groups tile the experts and every pick has a candidate");
	__shared__ float score[EXPERTS];
	__shared__ float selection[EXPERTS];
	__shared__ float group_score[GROUPS];
	__shared__ uint32_t group_open[GROUPS];
	__shared__ unsigned long long warp_best[2][WARPS];
	__shared__ uint32_t chosen[TOPK];
	const uint32_t row = blockIdx.x, lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < PER_THREAD; ++slot )
	{
		const uint32_t expert = threadIdx.x + slot * THREADS;
		if ( expert < EXPERTS )
		{
			const float value = SparkMathSigmoid(logits[(uint64_t)row * EXPERTS + expert]);
			score[expert] = value;
			selection[expert] = value + (selection_bias != 0 ? selection_bias[expert] : 0.0f);
		}
	}
	__syncthreads();
#if defined(__CUDA_ARCH__)
	if constexpr ( PER_GROUP == 32u && WARPS >= GROUPS )
	{
		if ( warp < GROUPS )
		{
			const float value = selection[warp * 32u + lane];
			const uint32_t key = value == value ? LmTopkKey(value) : 0u;
			const uint32_t max1_key = __reduce_max_sync(0xffffffffu, key);
			const uint32_t max1_mask = __ballot_sync(0xffffffffu, key == max1_key);
			const uint32_t max1_lane = (uint32_t)__ffs(max1_mask) - 1u;
			const uint32_t key2 = lane == max1_lane ? 0u : key;
			const uint32_t max2_key = __reduce_max_sync(0xffffffffu, key2);
			if ( lane == 0u )
			{
				group_score[warp] = LmTopkValue(max1_key) + LmTopkValue(max2_key);
				group_open[warp] = 0u;
			}
		}
	}
	else
#endif
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
	unsigned long long mine[PER_THREAD];
	#pragma unroll
	for ( uint32_t slot = 0u; slot < PER_THREAD; ++slot )
	{
		const uint32_t expert = threadIdx.x + slot * THREADS;
		mine[slot] = (expert < EXPERTS && group_open[expert / PER_GROUP] != 0u)
			? (((unsigned long long)LmTopkRouteKey(selection[expert] + 0.0f) << 32u) | (0xffffffffu - expert))
			: 0ull;
	}
	for ( uint32_t pick = 0u; pick < TOPK; ++pick )
	{
		unsigned long long best = 0ull;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < PER_THREAD; ++slot )
			best = mine[slot] > best ? mine[slot] : best;
#if defined(__CUDA_ARCH__)
		{
			const uint32_t high = __reduce_max_sync(0xffffffffu, (uint32_t)(best >> 32u));
			const uint32_t low = __reduce_max_sync(0xffffffffu, (uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
			best = ((unsigned long long)high << 32u) | (unsigned long long)low;
		}
#else
		for ( uint32_t offset = 16u; offset > 0u; offset >>= 1u )
		{
			const unsigned long long other = __shfl_xor_sync(0xffffffffu, best, offset);
			best = other > best ? other : best;
		}
#endif
		if ( lane == 0u )
			warp_best[pick & 1u][warp] = best;
		__syncthreads();
		best = 0ull;
		#pragma unroll
		for ( uint32_t other = 0u; other < WARPS; ++other )
			best = warp_best[pick & 1u][other] > best ? warp_best[pick & 1u][other] : best;
		#pragma unroll
		for ( uint32_t slot = 0u; slot < PER_THREAD; ++slot )
			mine[slot] = mine[slot] == best ? 0ull : mine[slot];
		if ( threadIdx.x == 0u )
		{
			chosen[pick] = 0xffffffffu - (uint32_t)best;
			LmTopkRouteFault(fault, (uint32_t)(best >> 32u));
		}
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float total = 0.0f;
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
			total += score[chosen[pick]];
		#pragma unroll
		for ( uint32_t pick = 0u; pick < TOPK; ++pick )
		{
			out_indices[(uint64_t)row * TOPK + pick] = chosen[pick];
			out_values[(uint64_t)row * TOPK + pick] = score[chosen[pick]] / (total + 1e-20f) * scale;
		}
	}
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t EXPERTS, uint32_t GROUPS, uint32_t TOPK_GROUPS, uint32_t TOPK>
static inline SparkGpuError LmTopkGroupLimitedLaunch(uint32_t rows, const float *logits, const float *selection_bias, uint32_t *out_indices,
	float *out_values, float scale, uint32_t *fault, SparkGpuStream stream)
{
	if ( rows == 0u || logits == 0 || out_indices == 0 || out_values == 0 || fault == 0 )
		return SPARK_GPU_ERROR_INVALID_VALUE;
	LM_LAUNCH_DEPENDENT((LmTopkGroupLimitedKernel<THREADS,EXPERTS,GROUPS,TOPK_GROUPS,TOPK>), rows, THREADS, 0, stream, logits, selection_bias, out_indices,
		out_values, scale, fault);
	return SparkGpuPeekAtLastError();
}
