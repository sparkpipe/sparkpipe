#pragma once

#include "inference/kernels/topk.cuh"

#define LM_TOPK_WARP_PER_LANE 16u

template<uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static __device__ __forceinline__ void LmTopkWarpRow(const float *scores, uint32_t n, uint32_t *out_indices, float *out_values, const float *selection_bias, const uint16_t *logits_bf16, float mixture_scale, uint32_t *fault, uint32_t row, uint32_t lane)
{
	(void)scores;
	(void)n;
	(void)out_indices;
	(void)out_values;
	(void)selection_bias;
	(void)logits_bf16;
	(void)mixture_scale;
	(void)fault;
	(void)row;
	(void)lane;
}

template<uint32_t THREADS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static cudaError_t LmTopkRouteLaunch(uint32_t rows, const float *scores, uint32_t n, uint32_t *out_indices, float *out_values, const float *selection_bias, const uint16_t *logits_bf16, float mixture_scale, uint32_t *fault, cudaStream_t stream)
{
	if ( fault == 0 || n < K )
		return(cudaErrorInvalidValue);
	LM_LAUNCH((LmTopkSmallKernel<THREADS,K,RENORMALISE,1u,1u,SCORE_TRANSFORM>),rows,THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale,fault);
	return(cudaPeekAtLastError());
}
