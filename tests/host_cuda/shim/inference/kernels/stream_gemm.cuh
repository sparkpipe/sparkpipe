#pragma once

#include "inference/kernels/gemm.cuh"

template<class Format>
struct LmStreamWeight
{
	static constexpr bool kSupported = false;
};

template<class Format>
static int32_t LmStreamGemmGrouped(const void *, LmScaleTensor, const uint16_t *, uint16_t *, const uint32_t *, const uint32_t *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t)
{
	return(LM_LAUNCH_ERR_SHAPE);
}

template<class Format>
static int32_t LmStreamGemmDense(const void *, LmScaleTensor, const uint16_t *, uint16_t *, float *, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, cudaStream_t)
{
	return(LM_LAUNCH_ERR_SHAPE);
}
