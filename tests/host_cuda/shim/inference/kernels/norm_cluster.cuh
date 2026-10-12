#pragma once

#include "inference/kernels/norm.cuh"

template<uint32_t THREADS, class Weight>
static inline cudaError_t LmFusedResidualRmsNormLaunch(uint32_t rows, cudaStream_t stream, const uint16_t *input_bf16, const uint16_t *residual_bf16, const Weight *weight, uint16_t *residual_out_bf16, uint16_t *output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	LM_LAUNCH((LmFusedResidualRmsNormKernel<THREADS,Weight>),rows,THREADS,(dimension + 8u) * sizeof(float),stream,
		input_bf16,residual_bf16,weight,residual_out_bf16,output_bf16,dimension,row_stride,epsilon);
	return(cudaPeekAtLastError());
}
