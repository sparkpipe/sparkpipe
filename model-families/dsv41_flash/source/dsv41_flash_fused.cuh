#pragma once

#include "inference/kernels/fused/arrival.cuh"
#include "inference/kernels/fused/collapse_norm.cuh"
#include "inference/kernels/hc.cuh"
#include "runtime/launch.h"

#define D41FusedMixNormRows 16u

typedef struct D41FusedMixNormArguments
{
	const uint16_t *streams;
	const float *fn;
	const float *scale;
	const float *base;
	float *partials;
	float *pre;
	float *post;
	float *comb;
	uint32_t *counters;
	const float *collapse_pre;
	const uint16_t *norm;
	uint16_t *x;
	uint16_t *xq;
} D41FusedMixNormArguments;

__global__ __launch_bounds__(256) void D41FusedMixNormKernel(const __grid_constant__ D41FusedMixNormArguments arguments, uint32_t rows)
{
	extern __shared__ float D41FusedMixNormKernelShared[];
	uint32_t block = blockIdx.x;
	if ( block < rows * (LmHcSinkhornShape<D41K_HC>::kTasks) )
	{
		const uint32_t row = block / (LmHcSinkhornShape<D41K_HC>::kTasks), item = block % (LmHcSinkhornShape<D41K_HC>::kTasks);
		LmHcSinkhornPartialTask<D41K_HC>(arguments.streams, arguments.fn, arguments.partials, D41_HIDDEN, item, row);
		if ( LmFusedLastArrival(arguments.counters + row, LmHcSinkhornShape<D41K_HC>::kTasks) && threadIdx.x < 32u )
			LmHcSinkhornRow<D41K_HC>(arguments.partials, arguments.scale, arguments.base, arguments.pre, arguments.post, arguments.comb, D41_HIDDEN, D41_EPS, D41_HC_ITERATIONS, D41_HC_EPS, row, threadIdx.x, D41FusedMixNormKernelShared);
		return;
	}
	block -= rows * (LmHcSinkhornShape<D41K_HC>::kTasks);
	if ( block < rows * (D41_HIDDEN / D41_THREADS) )
	{
		const uint32_t row = block / (D41_HIDDEN / D41_THREADS), item = block % (D41_HIDDEN / D41_THREADS);
		LmHcCollapseNormRow<256u, D41K_HC, SparkDsv41FlashFp8QdqLaneOp>((const uint16_t *)0, arguments.streams, arguments.collapse_pre, arguments.norm, arguments.x, arguments.xq, row, D41_HIDDEN, item, (D41_HIDDEN / D41_THREADS), D41_EPS, D41FusedMixNormKernelShared);
		return;
	}
}

static inline int32_t D41FusedMixNormLaunch(const D41FusedMixNormArguments *arguments, uint32_t rows, SparkGpuStream stream)
{
	if ( arguments == 0 || rows == 0u || rows > D41FusedMixNormRows )
		return LM_LAUNCH_ERR_SHAPE;
	LM_LAUNCH((D41FusedMixNormKernel), rows * ((LmHcSinkhornShape<D41K_HC>::kTasks) + (D41_HIDDEN / D41_THREADS)), 256u, (D41_HIDDEN + 8u) * sizeof(float), stream, *arguments, rows);
	return SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH;
}
