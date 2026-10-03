#pragma once

#include "inference/kernels/topk.cuh"

#define LM_TOPK_WARP_PER_LANE 16u

template<uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ uint64_t LmTopkWarpCandidate(const float *scores, const uint16_t *logits_bf16, const float *selection_bias, uint64_t base, uint32_t index, uint32_t n)
{
	float value;
	if ( index >= n )
		return(0ull);
	value = LmTopkScore<SCORE_TRANSFORM>(scores,logits_bf16,base + index) + (selection_bias != 0 ? selection_bias[index] : 0.0f);
	return(((uint64_t)LmTopkKey(value) << 32u) | (uint64_t)(0xffffffffu - index));
}

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
__global__ __launch_bounds__(LM_WARP_LANES) void LmTopkWarpKernel(const float *__restrict__ scores, uint32_t n, uint32_t *__restrict__ out_indices, float *__restrict__ out_values, const float *__restrict__ selection_bias, const uint16_t *__restrict__ logits_bf16, float mixture_scale)
{
	constexpr uint32_t width = K <= 1u ? 1u : K <= 2u ? 2u : K <= 4u ? 4u : K <= 8u ? 8u : K <= 16u ? 16u : 32u;
	static_assert(K <= LM_WARP_LANES, "one lane holds each chosen expert");
	const uint32_t lane = threadIdx.x;
	const uint64_t base = (uint64_t)blockIdx.x * n;
	uint64_t candidate[LM_TOPK_WARP_PER_LANE], best, winner;
	uint32_t slot, round, high, low, step, chosen = 0u;
	float value = 0.0f, tree, total;
	#pragma unroll
	for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
		candidate[slot] = LmTopkWarpCandidate<SCORE_TRANSFORM>(scores,logits_bf16,selection_bias,base,lane + slot * LM_WARP_LANES,n);
	for ( round = 0u; round < K; round++ )
	{
		best = 0ull;
		#pragma unroll
		for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
			best = candidate[slot] > best ? candidate[slot] : best;
		high = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u));
		low = __reduce_max_sync(0xffffffffu,(uint32_t)(best >> 32u) == high ? (uint32_t)best : 0u);
		winner = ((uint64_t)high << 32u) | (uint64_t)low;
		chosen = lane == round ? 0xffffffffu - low : chosen;
		#pragma unroll
		for ( slot = 0u; slot < LM_TOPK_WARP_PER_LANE; slot++ )
			candidate[slot] = candidate[slot] == winner ? 0ull : candidate[slot];
	}
	if ( lane < K )
	{
		out_indices[(uint64_t)blockIdx.x * K + lane] = chosen;
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
		out_values[(uint64_t)blockIdx.x * K + lane] = RENORMALISE ? (value / total) * mixture_scale : value * mixture_scale;
}

template<uint32_t THREADS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static cudaError_t LmTopkRouteLaunch(uint32_t rows, const float *scores, uint32_t n, uint32_t *out_indices, float *out_values, const float *selection_bias, const uint16_t *logits_bf16, float mixture_scale, cudaStream_t stream)
{
	if ( n <= LM_TOPK_WARP_PER_LANE * LM_WARP_LANES && n >= K )
		LmTopkWarpKernel<K,RENORMALISE,SCORE_TRANSFORM><<<rows,LM_WARP_LANES,0u,stream>>>(scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale);
	else
		LmTopkSmallKernel<THREADS,K,RENORMALISE,1u,1u,SCORE_TRANSFORM><<<rows,THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream>>>(scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale);
	return(cudaPeekAtLastError());
}
