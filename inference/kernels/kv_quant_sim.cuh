#pragma once

#include "inference/kernels/dtype.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_kv_quant_sim.h"
#include <float.h>
#include <math.h>
#include <stdint.h>

#ifndef LM_KV_QUANT_SIM_LANES
#define LM_KV_QUANT_SIM_LANES 32u
#endif

#define LM_KV_QUANT_SIM_NAN_BF16 ((uint16_t)0x7fc0u)

static __device__ __forceinline__ float LmKvQuantSimFormatMax(uint32_t codec)
{
	return(codec == SPARK_KV_QUANT_SIM_MXFP4 ? LM_E2M1_MAX : LM_E4M3_MAX);
}

static __device__ __forceinline__ int32_t LmKvQuantSimScaleExponent(float amax, float format_max)
{
	int32_t amax_exponent,max_exponent,exponent;
	float amax_mantissa,max_mantissa;
	if ( amax == 0.0f )
		return(SPARK_KV_QUANT_SIM_SCALE_EXPONENT_MIN);
	amax_mantissa = frexpf(amax,&amax_exponent);
	max_mantissa = frexpf(format_max,&max_exponent);
	exponent = amax_exponent - max_exponent + (amax_mantissa > max_mantissa ? 1 : 0);
	return(exponent < SPARK_KV_QUANT_SIM_SCALE_EXPONENT_MIN ? SPARK_KV_QUANT_SIM_SCALE_EXPONENT_MIN : exponent);
}

static __device__ __forceinline__ float LmKvQuantSimRound(float scaled, uint32_t codec)
{
	float magnitude = fabsf(scaled),count;
	int32_t exponent,quantum_exponent;
	frexpf(magnitude,&exponent);
	if ( codec == SPARK_KV_QUANT_SIM_MXFP4 )
		quantum_exponent = magnitude >= 2.0f ? exponent - 2 : -1;
	else
		quantum_exponent = magnitude >= 0.015625f ? exponent - 4 : -9;
	count = rintf(ldexpf(magnitude,-quantum_exponent));
	return(copysignf(ldexpf(count,quantum_exponent),scaled));
}

static __device__ __forceinline__ uint16_t LmKvQuantSimElement(uint16_t value, int32_t exponent, uint32_t codec)
{
	float scaled = ldexpf(LmBf16ToFloat(value),-exponent);
	return(LmFloatToBf16(ldexpf(LmKvQuantSimRound(scaled,codec),exponent)));
}

template<uint32_t LANES>
static __device__ __forceinline__ void LmKvQuantSimGroup(uint16_t *row, uint32_t base, uint32_t group, uint32_t codec, uint32_t lane)
{
	float amax = 0.0f,value;
	uint32_t element,offset;
	int32_t exponent;
	for (element = lane; element < group; element += LANES)
	{
		value = fabsf(LmBf16ToFloat(row[base + element]));
		amax = fmaxf(amax,value <= FLT_MAX ? value : INFINITY);
	}
	for (offset = LANES / 2u; offset != 0u; offset >>= 1u)
		amax = fmaxf(amax,__shfl_xor_sync(0xffffffffu,amax,offset));
	if ( amax == INFINITY )
	{
		for (element = lane; element < group; element += LANES)
			row[base + element] = LM_KV_QUANT_SIM_NAN_BF16;
		return;
	}
	exponent = LmKvQuantSimScaleExponent(amax,LmKvQuantSimFormatMax(codec));
	for (element = lane; element < group; element += LANES)
		row[base + element] = LmKvQuantSimElement(row[base + element],exponent,codec);
}

template<uint32_t LANES, uint32_t CODEC>
__global__ __launch_bounds__(LANES)
void LmKvQuantSimRowsKernel(uint16_t *__restrict__ rows_bf16, uint64_t row_stride, uint32_t rows, uint32_t width, uint32_t group)
{
	uint32_t row = blockIdx.x,group_index = blockIdx.y;
	if ( row >= rows || (group_index + 1u) * group > width )
		return;
	LmKvQuantSimGroup<LANES>(rows_bf16 + ((uint64_t)row * row_stride),group_index * group,group,CODEC,threadIdx.x);
}

static int32_t LmKvQuantSimRowsLaunch(uint16_t *rows_bf16, uint64_t row_stride, uint32_t rows, uint32_t width, uint32_t codec, uint32_t group, cudaStream_t stream)
{
	dim3 grid;
	if ( SparkKvQuantSimRowValid(codec,group,width) == 0u || (uint64_t)width > row_stride )
		return(LM_LAUNCH_ERR_SHAPE);
	if ( codec == SPARK_KV_QUANT_SIM_BF16 || rows == 0u )
		return(LM_LAUNCH_OK);
	if ( rows_bf16 == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	grid = dim3(rows,width / group,1u);
	if ( codec == SPARK_KV_QUANT_SIM_FP8_E4M3 )
		LM_LAUNCH((LmKvQuantSimRowsKernel<LM_KV_QUANT_SIM_LANES,SPARK_KV_QUANT_SIM_FP8_E4M3>),grid,LM_KV_QUANT_SIM_LANES,0,stream,
			rows_bf16,row_stride,rows,width,group);
	else
		LM_LAUNCH((LmKvQuantSimRowsKernel<LM_KV_QUANT_SIM_LANES,SPARK_KV_QUANT_SIM_MXFP4>),grid,LM_KV_QUANT_SIM_LANES,0,stream,
			rows_bf16,row_stride,rows,width,group);
	return(cudaPeekAtLastError() == cudaSuccess ? LM_LAUNCH_OK : LM_LAUNCH_ERR_LAUNCH);
}

struct LmKvStateBf16Grid
{
	float value;
};

static __device__ __forceinline__ float LmScalarToFloat(LmKvStateBf16Grid state)
{
	return(state.value);
}

static __device__ __forceinline__ void LmStoreState(LmKvStateBf16Grid *slot, float value)
{
	slot->value = LmBf16ToFloat(LmFloatToBf16(value));
}

template<>
struct LmStateUpdateGrid<LmKvStateBf16Grid>
{
	static __device__ __forceinline__ float Apply(float value)
	{
		return(LmBf16ToFloat(LmFloatToBf16(value)));
	}
};
