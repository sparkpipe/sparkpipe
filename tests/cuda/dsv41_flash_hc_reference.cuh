#pragma once

#include "model-families/dsv41_flash/source/dsv41_flash_kernels.cuh"

#define SPARK_DSV41_FLASH_HC_MAX 8u
#define SPARK_DSV41_FLASH_HC_MIX_MAX ((2u + SPARK_DSV41_FLASH_HC_MAX) * SPARK_DSV41_FLASH_HC_MAX)

static __global__ void SparkDsv41FlashHcMixKernel(const uint16_t *streams_bf16, const float *fn_f32, float *mixes_f32, uint32_t flat_dimension, uint32_t mix_rows, float epsilon)
{
	__shared__ float scratch[SPARK_DSV41_FLASH_ROW_THREADS / 32u];
	uint32_t mix,element;
	float square = 0.0f,value,total,inverse;
	for (element = threadIdx.x; element < flat_dimension; element += blockDim.x)
	{
		value = LmBf16ToFloat(streams_bf16[element]);
		square += value * value;
	}
	square = SparkDsv41FlashBlockSum(square,scratch);
	inverse = rsqrtf(square / (float)flat_dimension + epsilon);
	for (mix = 0u; mix < mix_rows; mix++)
	{
		total = 0.0f;
		for (element = threadIdx.x; element < flat_dimension; element += blockDim.x)
			total += fn_f32[(uint64_t)mix * flat_dimension + element] * LmBf16ToFloat(streams_bf16[element]);
		total = SparkDsv41FlashBlockSum(total,scratch);
		if ( threadIdx.x == 0u )
			mixes_f32[mix] = total * inverse;
		__syncthreads();
	}
}

static __global__ void SparkDsv41FlashHcSinkhornKernel(const float *mixes_f32, const float *scale_f32, const float *base_f32, float *pre_f32, float *post_f32, float *comb_f32, uint32_t hc, uint32_t iterations, float epsilon)
{
	float comb[SPARK_DSV41_FLASH_HC_MAX * SPARK_DSV41_FLASH_HC_MAX],sum,top;
	uint32_t row,column,iteration;
	if ( threadIdx.x != 0u )
		return;
	for (row = 0u; row < hc; row++)
	{
		pre_f32[row] = 1.0f / (1.0f + expf(-(mixes_f32[row] * scale_f32[0] + base_f32[row]))) + epsilon;
		post_f32[row] = 2.0f / (1.0f + expf(-(mixes_f32[hc + row] * scale_f32[1] + base_f32[hc + row])));
	}
	for (row = 0u; row < hc; row++)
	{
		top = -INFINITY;
		for (column = 0u; column < hc; column++)
		{
			comb[row * hc + column] = mixes_f32[2u * hc + row * hc + column] * scale_f32[2] + base_f32[2u * hc + row * hc + column];
			top = fmaxf(top,comb[row * hc + column]);
		}
		sum = 0.0f;
		for (column = 0u; column < hc; column++)
		{
			comb[row * hc + column] = expf(comb[row * hc + column] - top);
			sum += comb[row * hc + column];
		}
		for (column = 0u; column < hc; column++)
			comb[row * hc + column] = comb[row * hc + column] / sum + epsilon;
	}
	for (column = 0u; column < hc; column++)
	{
		sum = 0.0f;
		for (row = 0u; row < hc; row++)
			sum += comb[row * hc + column];
		for (row = 0u; row < hc; row++)
			comb[row * hc + column] /= sum + epsilon;
	}
	for (iteration = 1u; iteration < iterations; iteration++)
	{
		for (row = 0u; row < hc; row++)
		{
			sum = 0.0f;
			for (column = 0u; column < hc; column++)
				sum += comb[row * hc + column];
			for (column = 0u; column < hc; column++)
				comb[row * hc + column] /= sum + epsilon;
		}
		for (column = 0u; column < hc; column++)
		{
			sum = 0.0f;
			for (row = 0u; row < hc; row++)
				sum += comb[row * hc + column];
			for (row = 0u; row < hc; row++)
				comb[row * hc + column] /= sum + epsilon;
		}
	}
	for (row = 0u; row < hc * hc; row++)
		comb_f32[row] = comb[row];
}

static inline cudaError_t SparkDsv41FlashLaunchHcMixes(cudaStream_t stream, const uint16_t *streams_bf16, const float *fn_f32, const float *scale_f32, const float *base_f32, float *mixes_f32, float *pre_f32, float *post_f32, float *comb_f32, uint32_t hc, uint32_t dimension, uint32_t iterations, float norm_epsilon, float hc_epsilon)
{
	cudaError_t status;
	if ( streams_bf16 == 0 || fn_f32 == 0 || scale_f32 == 0 || base_f32 == 0 || mixes_f32 == 0 || pre_f32 == 0 ||
		post_f32 == 0 || comb_f32 == 0 || hc == 0u || hc > SPARK_DSV41_FLASH_HC_MAX || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashHcMixKernel<<<1u,SPARK_DSV41_FLASH_ROW_THREADS,0,stream>>>(streams_bf16,fn_f32,mixes_f32,hc * dimension,(2u + hc) * hc,norm_epsilon);
	status = cudaGetLastError();
	if ( status != cudaSuccess )
		return(status);
	SparkDsv41FlashHcSinkhornKernel<<<1u,32u,0,stream>>>(mixes_f32,scale_f32,base_f32,pre_f32,post_f32,comb_f32,hc,iterations,hc_epsilon);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashHcPreKernel(const uint16_t *streams_bf16, const float *pre_f32, uint16_t *output_bf16, uint32_t hc, uint32_t dimension)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x,copy;
	float total = 0.0f;
	if ( element >= dimension )
		return;
	for (copy = 0u; copy < hc; copy++)
		total += pre_f32[copy] * LmBf16ToFloat(streams_bf16[(uint64_t)copy * dimension + element]);
	output_bf16[element] = LmFloatToBf16(total);
}

static __global__ void SparkDsv41FlashHcPostKernel(const uint16_t *sublayer_bf16, const uint16_t *residual_bf16, const float *post_f32, const float *comb_f32, uint16_t *streams_bf16, uint32_t hc, uint32_t dimension)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x,copy,source;
	float x,total;
	if ( element >= dimension )
		return;
	x = LmBf16ToFloat(sublayer_bf16[element]);
	for (copy = 0u; copy < hc; copy++)
	{
		total = 0.0f;
		for (source = 0u; source < hc; source++)
			total += comb_f32[source * hc + copy] * LmBf16ToFloat(residual_bf16[(uint64_t)source * dimension + element]);
		streams_bf16[(uint64_t)copy * dimension + element] = LmFloatToBf16(post_f32[copy] * x + total);
	}
}
