#pragma once

#include "inference/kernels/topk.cuh"

template<uint32_t THREADS, uint32_t K, bool RENORMALISE, uint32_t SCORE_TRANSFORM>
static cudaError_t LmTopkRouteLaunch(uint32_t rows, const float *scores, uint32_t n, uint32_t *out_indices, float *out_values, const float *selection_bias, const uint16_t *logits_bf16, float mixture_scale, cudaStream_t stream)
{
	LM_LAUNCH((LmTopkSmallKernel<THREADS,K,RENORMALISE,1u,1u,SCORE_TRANSFORM>),rows,THREADS,2u * LM_TOPK_SMALL_LIMIT * sizeof(uint32_t),stream,scores,n,out_indices,out_values,selection_bias,logits_bf16,mixture_scale);
	return(cudaPeekAtLastError());
}
