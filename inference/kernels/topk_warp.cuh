#pragma once

#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/topk.cuh"
#include "sparkpipe/spark_gpu.h"

#define LM_TOPK_WARP_PER_LANE 16u

template<uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ uint64_t LmTopkWarpCandidate(const float *scores, const uint16_t *logits_bf16, const float *selection_bias, uint64_t base, uint32_t index, uint32_t n)
{
	float value;
	if ( index >= n )
		return(0ull);
	value = LmTopkScore<SCORE_TRANSFORM>(scores,logits_bf16,base + index) + (selection_bias != 0 ? selection_bias[index] : 0.0f);
	return(((uint64_t)LmTopkRouteKey(value) << 32u) | (uint64_t)(0xffffffffu - index));
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM, uint32_t PER_LANE = LM_TOPK_WARP_PER_LANE>
static __device__ __forceinline__ void LmTopkWarpRow(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale, uint32_t *__restrict__ fault, uint32_t row, uint32_t lane)
{
	constexpr uint32_t width = K <= 1u ? 1u : K <= 2u ? 2u : K <= 4u ? 4u : K <= 8u ? 8u : K <= 16u ? 16u : 32u;
	static_assert(K <= LM_WARP_LANES, "one lane holds each chosen expert");
	const uint64_t base = (uint64_t)row * n;
	uint64_t candidate[PER_LANE], best, winner;
	uint32_t slot, round, high = 0u, low, step, chosen = 0u;
	float value = 0.0f, tree, total;
	#pragma unroll
	for ( slot = 0u; slot < PER_LANE; slot++ )
		candidate[slot] = LmTopkWarpCandidate<SCORE_TRANSFORM>(scores,logits_bf16,selection_bias,base,lane + slot * LM_WARP_LANES,n);
	for ( round = 0u; round < K; round++ )
	{
		best = 0ull;
		#pragma unroll
		for ( slot = 0u; slot < PER_LANE; slot++ )
			best = candidate[slot] > best ? candidate[slot] : best;
		high = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u));
		low = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
		winner = ((uint64_t)high << 32u) | (uint64_t)low;
		chosen = lane == round ? 0xffffffffu - low : chosen;
		#pragma unroll
		for ( slot = 0u; slot < PER_LANE; slot++ )
			candidate[slot] = candidate[slot] == winner ? 0ull : candidate[slot];
	}
	if ( lane == 0u )
		LmTopkRouteFault(fault,high);
	if ( lane < K )
	{
		out_indices[(uint64_t)row * K + lane] = chosen;
		value = LmTopkScore<SCORE_TRANSFORM>(scores,logits_bf16,base + chosen);
	}
	if ( out_values == 0 )
		return;
	tree = lane < K ? value : 0.0f;
	#pragma unroll
	for ( step = width / 2u; step > 0u; step >>= 1u )
	{
		float other = __shfl_down_sync(0xffffffffu,tree,step);
		tree = lane < step ? tree + other : tree;
	}
	total = __shfl_sync(0xffffffffu,tree,0) + 1e-20f;
	if ( lane < K )
		out_values[(uint64_t)row * K + lane] = RENORMALISE ? (value / total) * mixture_scale : value * mixture_scale;
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM, uint32_t PER_LANE = LM_TOPK_WARP_PER_LANE>
__global__ __launch_bounds__(LM_WARP_LANES) void LmTopkWarpKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale, uint32_t *__restrict__ fault)
{
	LmDependentWait();
	LmTopkWarpRow<K,RENORMALISE,SCORE_TRANSFORM,PER_LANE>(scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault,blockIdx.x,threadIdx.x);
	LmDependentRelease();
}

#define LM_TOPK_WARPS 8u
#define LM_TOPK_WARPS_PER_LANE 4u

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ void LmTopkWarpsRow(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale, uint32_t *__restrict__ fault, uint32_t row)
{
	constexpr uint32_t width = K <= 1u ? 1u : K <= 2u ? 2u : K <= 4u ? 4u : K <= 8u ? 8u : K <= 16u ? 16u : 32u;
	constexpr uint32_t merged = (LM_TOPK_WARPS * K + LM_WARP_LANES - 1u) / LM_WARP_LANES;
	static_assert(K <= LM_WARP_LANES, "one lane holds each chosen expert");
	__shared__ uint64_t local[LM_TOPK_WARPS * K];
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	const uint64_t base = (uint64_t)row * n;
	uint64_t candidate[LM_TOPK_WARPS_PER_LANE], pool[merged], best, winner;
	uint32_t slot, round, high, low, step, chosen = 0u;
	float value = 0.0f, tree, total;
	#pragma unroll
	for ( slot = 0u; slot < LM_TOPK_WARPS_PER_LANE; slot++ )
		candidate[slot] = LmTopkWarpCandidate<SCORE_TRANSFORM>(scores,logits_bf16,selection_bias,base,
			warp * LM_WARP_LANES * LM_TOPK_WARPS_PER_LANE + lane + slot * LM_WARP_LANES,n);
	for ( round = 0u; round < K; round++ )
	{
		best = 0ull;
		#pragma unroll
		for ( slot = 0u; slot < LM_TOPK_WARPS_PER_LANE; slot++ )
			best = candidate[slot] > best ? candidate[slot] : best;
		high = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u));
		low = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
		winner = ((uint64_t)high << 32u) | (uint64_t)low;
		if ( lane == 0u )
			local[warp * K + round] = winner;
		#pragma unroll
		for ( slot = 0u; slot < LM_TOPK_WARPS_PER_LANE; slot++ )
			candidate[slot] = candidate[slot] == winner ? 0ull : candidate[slot];
	}
	__syncthreads();
	if ( warp != 0u )
		return;
	#pragma unroll
	for ( slot = 0u; slot < merged; slot++ )
		pool[slot] = lane + slot * LM_WARP_LANES < LM_TOPK_WARPS * K ? local[lane + slot * LM_WARP_LANES] : 0ull;
	for ( round = 0u; round < K; round++ )
	{
		best = 0ull;
		#pragma unroll
		for ( slot = 0u; slot < merged; slot++ )
			best = pool[slot] > best ? pool[slot] : best;
		high = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u));
		low = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
		winner = ((uint64_t)high << 32u) | (uint64_t)low;
		chosen = lane == round ? 0xffffffffu - low : chosen;
		#pragma unroll
		for ( slot = 0u; slot < merged; slot++ )
			pool[slot] = pool[slot] == winner ? 0ull : pool[slot];
	}
	if ( lane == 0u )
		LmTopkRouteFault(fault,high);
	if ( lane < K )
	{
		out_indices[(uint64_t)row * K + lane] = chosen;
		value = LmTopkScore<SCORE_TRANSFORM>(scores,logits_bf16,base + chosen);
	}
	if ( out_values == 0 )
		return;
	tree = lane < K ? value : 0.0f;
	#pragma unroll
	for ( step = width / 2u; step > 0u; step >>= 1u )
	{
		float other = __shfl_down_sync(0xffffffffu,tree,step);
		tree = lane < step ? tree + other : tree;
	}
	total = __shfl_sync(0xffffffffu,tree,0) + 1e-20f;
	if ( lane < K )
		out_values[(uint64_t)row * K + lane] = RENORMALISE ? (value / total) * mixture_scale : value * mixture_scale;
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
__global__ __launch_bounds__(LM_TOPK_WARPS * LM_WARP_LANES) void LmTopkWarpsKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale, uint32_t *__restrict__ fault)
{
	LmDependentWait();
	LmTopkWarpsRow<K,RENORMALISE,SCORE_TRANSFORM>(scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault,blockIdx.x);
	LmDependentRelease();
}

template<uint32_t THREADS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static SparkGpuError LmTopkRouteLaunch(uint32_t rows, const float *scores, uint32_t n, uint32_t *out_indices, float *out_values, const float *selection_bias, const uint16_t *logits_bf16, float mixture_scale, uint32_t *fault, SparkGpuStream stream)
{
	if ( fault == 0 || n < K )
		return(SPARK_GPU_ERROR_INVALID_VALUE);
	if ( n <= 4u * LM_WARP_LANES )
		LM_LAUNCH_DEPENDENT((LmTopkWarpKernel<K,RENORMALISE,SCORE_TRANSFORM,4u>),rows,LM_WARP_LANES,0u,stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	else if ( n <= 8u * LM_WARP_LANES )
		LM_LAUNCH_DEPENDENT((LmTopkWarpKernel<K,RENORMALISE,SCORE_TRANSFORM,8u>),rows,LM_WARP_LANES,0u,stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	else if ( n <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES )
		LM_LAUNCH_DEPENDENT((LmTopkWarpKernel<K,RENORMALISE,SCORE_TRANSFORM,LM_TOPK_WARP_PER_LANE>),rows,LM_WARP_LANES,0u,stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	else if ( n <= LM_TOPK_WARPS * LM_TOPK_WARPS_PER_LANE * LM_WARP_LANES )
		LM_LAUNCH_DEPENDENT((LmTopkWarpsKernel<K,RENORMALISE,SCORE_TRANSFORM>),rows,LM_TOPK_WARPS * LM_WARP_LANES,0u,stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	else
		LM_LAUNCH_DEPENDENT((LmTopkSmallKernel<THREADS,K,RENORMALISE,1u,1u,SCORE_TRANSFORM>),rows,THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	return(SparkGpuPeekAtLastError());
}
