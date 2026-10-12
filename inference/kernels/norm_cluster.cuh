#pragma once

#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include <cooperative_groups.h>
#include <stdint.h>

#define LM_NORM_CLUSTER_SLOTS 32u
#define LM_NORM_CLUSTER_BLOCKS 8u
#define LM_NORM_CLUSTER_ROWS 8u

template<uint32_t THREADS, class Weight, bool ROUND_SCALED>
__global__ __launch_bounds__(THREADS, 1)
void LmRmsNormClusterKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
	const uint32_t part = cluster.block_rank(),parts = cluster.num_blocks();
	const uint64_t base = (uint64_t)(blockIdx.x / parts) * row_stride;
	float values[LM_NORM_CLUSTER_SLOTS],gains[LM_NORM_CLUSTER_SLOTS];
	float total = 0.0f,scale;
	uint32_t slot,index;
	LmDependentRelease();
	#pragma unroll
	for (slot = 0u; slot < LM_NORM_CLUSTER_SLOTS; slot++)
	{
		index = threadIdx.x + slot * THREADS;
		gains[slot] = index < dimension && slot % parts == part ? LmScalarToFloat(weight[index]) : 0.0f;
	}
	LmDependentWait();
	#pragma unroll
	for (slot = 0u; slot < LM_NORM_CLUSTER_SLOTS; slot++)
	{
		index = threadIdx.x + slot * THREADS;
		float value = 0.0f;
		if ( index < dimension )
		{
			value = LmBf16ToFloat(input_bf16[base + index]);
			if ( residual_bf16 != 0 )
				value += LmBf16ToFloat(residual_bf16[base + index]);
		}
		values[slot] = value;
	}
	#pragma unroll
	for (slot = 0u; slot < LM_NORM_CLUSTER_SLOTS; slot++)
		if ( threadIdx.x + slot * THREADS < dimension )
			total += values[slot] * values[slot];
	total = LmBlockSum<THREADS>(total,reduction);
	scale = rsqrtf((total / (float)dimension) + epsilon);
	LmClusterBarrier();
	#pragma unroll
	for (slot = 0u; slot < LM_NORM_CLUSTER_SLOTS; slot++)
	{
		index = threadIdx.x + slot * THREADS;
		if ( index >= dimension || slot % parts != part )
			continue;
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[base + index] = LmFloatToBf16(values[slot]);
		output_bf16[base + index] = LmFloatToBf16((ROUND_SCALED ? LmBf16ToFloat(LmFloatToBf16(values[slot] * scale)) : values[slot] * scale) * gains[slot]);
	}
}

template<uint32_t THREADS, class Weight, bool ROUND_SCALED>
static inline uint32_t LmRmsNormClusterParts(uint32_t rows, uint32_t dimension)
{
	const uint32_t slots = (dimension + THREADS - 1u) / THREADS;
	const uint32_t parts = slots < LM_NORM_CLUSTER_BLOCKS ? slots : LM_NORM_CLUSTER_BLOCKS;
	return(rows != 0u && rows <= LM_NORM_CLUSTER_ROWS && slots <= LM_NORM_CLUSTER_SLOTS && parts > 1u ? parts : 0u);
}

template<uint32_t THREADS, class Weight>
static inline cudaError_t LmFusedResidualRmsNormLaunch(uint32_t rows, cudaStream_t stream, const uint16_t *input_bf16, const uint16_t *residual_bf16, const Weight *weight, uint16_t *residual_out_bf16, uint16_t *output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	const uint32_t parts = LmRmsNormClusterParts<THREADS,Weight,false>(rows,dimension);
	if ( parts != 0u )
		return(LmLaunchClusterKernel((LmRmsNormClusterKernel<THREADS,Weight,false>),dim3(rows * parts),dim3(THREADS),0,stream,parts,1u,
			input_bf16,residual_bf16,weight,residual_out_bf16,output_bf16,dimension,row_stride,epsilon));
	LM_LAUNCH((LmFusedResidualRmsNormKernel<THREADS,Weight>),rows,THREADS,(dimension + 8u) * sizeof(float),stream,
		input_bf16,residual_bf16,weight,residual_out_bf16,output_bf16,dimension,row_stride,epsilon);
	return(cudaPeekAtLastError());
}

template<uint32_t THREADS>
static inline cudaError_t LmBf16RmsNormLaunch(uint32_t rows, cudaStream_t stream, const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	const uint32_t parts = LmRmsNormClusterParts<THREADS,uint16_t,true>(rows,dimension);
	if ( parts != 0u )
		return(LmLaunchClusterKernel((LmRmsNormClusterKernel<THREADS,uint16_t,true>),dim3(rows * parts),dim3(THREADS),0,stream,parts,1u,
			input_bf16,(const uint16_t *)0,weight_bf16,(uint16_t *)0,output_bf16,dimension,row_stride,epsilon));
	LM_LAUNCH((LmBf16RmsNormKernel<THREADS>),rows,THREADS,(dimension + 8u) * sizeof(float),stream,input_bf16,weight_bf16,output_bf16,dimension,row_stride,epsilon);
	return(cudaPeekAtLastError());
}
