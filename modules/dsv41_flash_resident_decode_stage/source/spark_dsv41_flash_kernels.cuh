#pragma once

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"

#define SPARK_DSV41_FLASH_KV_FP4_GROUP 16u
#define SPARK_DSV41_FLASH_KV_FP4_AMAX_FLOOR (6.0f * 0.001953125f)
#define SPARK_DSV41_FLASH_ENGRAM_GATE_FLOOR 1e-6f
#define SPARK_DSV41_FLASH_ENGRAM_THREADS 256u

static __device__ __forceinline__ float SparkDsv41FlashE2m1Round(float value)
{
	return(LmE2m1PairToFloat(LmFloatPairToE2m1(value,0.0f)).x);
}

static __global__ void SparkDsv41FlashKvFp4QdqKernel(uint16_t *data_bf16, uint64_t row_stride, uint32_t row_count, uint32_t width)
{
	uint32_t groups_per_row = width / SPARK_DSV41_FLASH_KV_FP4_GROUP;
	uint64_t group = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t base;
	uint32_t row,element;
	float amax = 0.0f,scale,value;
	if ( group >= (uint64_t)row_count * groups_per_row )
		return;
	row = (uint32_t)(group / groups_per_row);
	base = (uint64_t)row * row_stride + (group % groups_per_row) * SPARK_DSV41_FLASH_KV_FP4_GROUP;
	for (element = 0u; element < SPARK_DSV41_FLASH_KV_FP4_GROUP; element++)
		amax = fmaxf(amax,fabsf(LmBf16ToFloat(data_bf16[base + element])));
	amax = fmaxf(amax,SPARK_DSV41_FLASH_KV_FP4_AMAX_FLOOR);
	scale = LmE4m3ToFloat(LmFloatToE4m3(amax / LM_E2M1_MAX));
	for (element = 0u; element < SPARK_DSV41_FLASH_KV_FP4_GROUP; element++)
	{
		value = fminf(fmaxf(LmBf16ToFloat(data_bf16[base + element]) / scale,-LM_E2M1_MAX),LM_E2M1_MAX);
		data_bf16[base + element] = LmFloatToBf16(SparkDsv41FlashE2m1Round(value) * scale);
	}
}

static inline cudaError_t SparkDsv41FlashLaunchKvFp4Qdq(cudaStream_t stream, uint16_t *data_bf16, uint64_t row_stride, uint32_t row_count, uint32_t width)
{
	uint64_t groups;
	if ( data_bf16 == 0 || row_count == 0u || width == 0u ||
		width % SPARK_DSV41_FLASH_KV_FP4_GROUP != 0u || row_stride < width )
		return(cudaErrorInvalidValue);
	groups = (uint64_t)row_count * (width / SPARK_DSV41_FLASH_KV_FP4_GROUP);
	SparkDsv41FlashKvFp4QdqKernel<<<(uint32_t)((groups + 255u) / 256u),256u,0,stream>>>(
		data_bf16,row_stride,row_count,width);
	return(cudaGetLastError());
}

static __device__ __forceinline__ float SparkDsv41FlashBlockSum(float value, float *scratch)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u,offset;
	for (offset = 16u; offset != 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	__syncthreads();
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	value = 0.0f;
	if ( threadIdx.x == 0u )
	{
		for (offset = 0u; offset < blockDim.x / 32u; offset++)
			value += scratch[offset];
		scratch[0] = value;
	}
	__syncthreads();
	return(scratch[0]);
}

static __global__ void SparkDsv41FlashEngramGateKernel(
	uint16_t *streams_bf16,
	const uint16_t *kv_bf16,
	const uint16_t *q_weight_bf16,
	const uint16_t *k_weight_bf16,
	uint32_t hc,
	uint32_t dimension,
	float epsilon)
{
	__shared__ float scratch[SPARK_DSV41_FLASH_ENGRAM_THREADS / 32u];
	uint32_t row = blockIdx.x,copy = blockIdx.y,element;
	uint64_t stream_base = ((uint64_t)row * hc + copy) * dimension;
	uint64_t kv_row = (uint64_t)row * (hc + 1u) * dimension;
	uint64_t key_base = kv_row + (uint64_t)copy * dimension;
	uint64_t value_base = kv_row + (uint64_t)hc * dimension;
	uint64_t weight_base = (uint64_t)copy * dimension;
	float h,key,weight,h_square = 0.0f,key_square = 0.0f,dot = 0.0f,rstd,gate,magnitude;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		h = LmBf16ToFloat(streams_bf16[stream_base + element]);
		key = LmBf16ToFloat(kv_bf16[key_base + element]);
		weight = LmBf16ToFloat(q_weight_bf16[weight_base + element]) * LmBf16ToFloat(k_weight_bf16[weight_base + element]);
		h_square += h * h;
		key_square += key * key;
		dot += h * weight * key;
	}
	h_square = SparkDsv41FlashBlockSum(h_square,scratch);
	key_square = SparkDsv41FlashBlockSum(key_square,scratch);
	dot = SparkDsv41FlashBlockSum(dot,scratch);
	rstd = (1.0f / sqrtf(h_square / (float)dimension + epsilon)) * (1.0f / sqrtf(key_square / (float)dimension + epsilon));
	dot = dot * rstd * (1.0f / sqrtf((float)dimension));
	magnitude = sqrtf(fmaxf(fabsf(dot),SPARK_DSV41_FLASH_ENGRAM_GATE_FLOOR));
	gate = 1.0f / (1.0f + expf(-copysignf(magnitude,dot)));
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		h = LmBf16ToFloat(streams_bf16[stream_base + element]);
		streams_bf16[stream_base + element] = LmFloatToBf16(h + gate * LmBf16ToFloat(kv_bf16[value_base + element]));
	}
}

static inline cudaError_t SparkDsv41FlashLaunchEngramGate(
	cudaStream_t stream,
	uint16_t *streams_bf16,
	const uint16_t *kv_bf16,
	const uint16_t *q_weight_bf16,
	const uint16_t *k_weight_bf16,
	uint32_t row_count,
	uint32_t hc,
	uint32_t dimension,
	float epsilon)
{
	if ( streams_bf16 == 0 || kv_bf16 == 0 || q_weight_bf16 == 0 ||
		k_weight_bf16 == 0 || row_count == 0u || hc == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashEngramGateKernel<<<dim3(row_count,hc),SPARK_DSV41_FLASH_ENGRAM_THREADS,0,stream>>>(
		streams_bf16,kv_bf16,q_weight_bf16,k_weight_bf16,hc,dimension,epsilon);
	return(cudaGetLastError());
}

#define SPARK_DSV41_FLASH_CANDIDATE_THREADS 256u

static __device__ __forceinline__ uint32_t SparkDsv41FlashOrderedKey(float value)
{
	uint32_t bits = __float_as_uint(value);
	return((bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u);
}

static __device__ __forceinline__ uint32_t SparkDsv41FlashBlockCount(uint32_t value, uint32_t *scratch)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u,offset;
	for (offset = 16u; offset != 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	__syncthreads();
	if ( lane == 0u )
		scratch[warp] = value;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		value = 0u;
		for (offset = 0u; offset < blockDim.x / 32u; offset++)
			value += scratch[offset];
		scratch[0] = value;
	}
	__syncthreads();
	value = scratch[0];
	__syncthreads();
	return(value);
}

static __global__ void SparkDsv41FlashCandidateMaskKernel(
	float *scores_f32,
	const uint32_t *widths,
	uint64_t row_stride,
	float *block_scores_f32,
	uint64_t block_stride,
	uint32_t block_size,
	uint32_t topk_blocks)
{
	__shared__ uint32_t counts[SPARK_DSV41_FLASH_CANDIDATE_THREADS / 32u];
	__shared__ uint32_t threshold_shared;
	uint32_t row = blockIdx.x,width = widths[row],block_count,block,element,bit,candidate,prefix = 0u,greater,need,taken;
	float *scores = scores_f32 + (uint64_t)row * row_stride;
	float *blocks = block_scores_f32 + (uint64_t)row * block_stride;
	float best;
	if ( width == 0u )
		return;
	block_count = (width + block_size - 1u) / block_size;
	for (block = threadIdx.x; block < block_count; block += blockDim.x)
	{
		best = -INFINITY;
		for (element = block * block_size; element < width && element < (block + 1u) * block_size; element++)
			best = fmaxf(best,scores[element]);
		blocks[block] = block == (width - 1u) / block_size ? INFINITY : best;
	}
	__syncthreads();
	if ( block_count > topk_blocks )
	{
		for (bit = 32u; bit-- > 0u;)
		{
			candidate = prefix | (1u << bit);
			greater = 0u;
			for (block = threadIdx.x; block < block_count; block += blockDim.x)
				greater += SparkDsv41FlashOrderedKey(blocks[block]) >= candidate ? 1u : 0u;
			if ( SparkDsv41FlashBlockCount(greater,counts) >= topk_blocks )
				prefix = candidate;
		}
		if ( threadIdx.x == 0u )
			threshold_shared = prefix;
		__syncthreads();
		greater = 0u;
		for (block = threadIdx.x; block < block_count; block += blockDim.x)
			greater += SparkDsv41FlashOrderedKey(blocks[block]) > threshold_shared ? 1u : 0u;
		greater = SparkDsv41FlashBlockCount(greater,counts);
		if ( threadIdx.x == 0u )
		{
			need = topk_blocks - greater;
			taken = 0u;
			for (block = 0u; block < block_count; block++)
			{
				if ( SparkDsv41FlashOrderedKey(blocks[block]) == threshold_shared )
				{
					if ( taken < need && blocks[block] > -INFINITY )
					{
						taken++;
						blocks[block] = INFINITY;
					}
					else
						blocks[block] = -INFINITY;
				}
				else if ( SparkDsv41FlashOrderedKey(blocks[block]) < threshold_shared )
					blocks[block] = -INFINITY;
			}
		}
		__syncthreads();
	}
	for (element = threadIdx.x; element < width; element += blockDim.x)
		if ( !(blocks[element / block_size] > -INFINITY) )
			scores[element] = -INFINITY;
}

static inline cudaError_t SparkDsv41FlashLaunchCandidateMask(
	cudaStream_t stream,
	float *scores_f32,
	const uint32_t *widths,
	uint64_t row_stride,
	float *block_scores_f32,
	uint64_t block_stride,
	uint32_t row_count,
	uint32_t block_size,
	uint32_t topk_blocks)
{
	if ( scores_f32 == 0 || widths == 0 || block_scores_f32 == 0 || row_count == 0u ||
		block_size == 0u || topk_blocks == 0u || block_stride * block_size < row_stride )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashCandidateMaskKernel<<<row_count,SPARK_DSV41_FLASH_CANDIDATE_THREADS,0,stream>>>(
		scores_f32,widths,row_stride,block_scores_f32,block_stride,block_size,topk_blocks);
	return(cudaGetLastError());
}

#define SPARK_DSV41_FLASH_FP8_BLOCK 32u
#define SPARK_DSV41_FLASH_FP8_AMAX_FLOOR 1e-4f
#define SPARK_DSV41_FLASH_LINEAR_WARPS 8u

static __device__ __forceinline__ float SparkDsv41FlashPow2CeilScale(float amax, float inverse_max)
{
	uint32_t bits = __float_as_uint(amax * inverse_max);
	int32_t exponent = (int32_t)((bits >> 23u) & 0xffu) - 127 + ((bits & 0x7fffffu) != 0u ? 1 : 0);
	return(__uint_as_float((uint32_t)(exponent + 127) << 23u));
}

static __global__ void SparkDsv41FlashActQuantKernel(const uint16_t *input_bf16, float *output_f32, uint32_t row_count, uint32_t width)
{
	uint32_t groups = width / SPARK_DSV41_FLASH_FP8_BLOCK;
	uint64_t group = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t base;
	uint32_t element;
	float amax = 0.0f,scale,value;
	if ( group >= (uint64_t)row_count * groups )
		return;
	base = (group / groups) * width + (group % groups) * SPARK_DSV41_FLASH_FP8_BLOCK;
	for (element = 0u; element < SPARK_DSV41_FLASH_FP8_BLOCK; element++)
		amax = fmaxf(amax,fabsf(LmBf16ToFloat(input_bf16[base + element])));
	scale = SparkDsv41FlashPow2CeilScale(fmaxf(amax,SPARK_DSV41_FLASH_FP8_AMAX_FLOOR),1.0f / LM_E4M3_MAX);
	for (element = 0u; element < SPARK_DSV41_FLASH_FP8_BLOCK; element++)
	{
		value = fminf(fmaxf(LmBf16ToFloat(input_bf16[base + element]) / scale,-LM_E4M3_MAX),LM_E4M3_MAX);
		output_f32[base + element] = LmE4m3ToFloat(LmFloatToE4m3(value)) * scale;
	}
}

static __global__ void SparkDsv41FlashFp8BlockLinearKernel(
	const uint8_t *weight_e4m3,
	const uint8_t *scale_ue8m0,
	const float *activation_f32,
	uint16_t *output_bf16,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u;
	uint32_t output = blockIdx.x * SPARK_DSV41_FLASH_LINEAR_WARPS + warp,row = blockIdx.y,k,pair,offset;
	uint32_t scale_columns = input_dimension / SPARK_DSV41_FLASH_FP8_BLOCK;
	const uint8_t *weights;
	const float *activation = activation_f32 + (uint64_t)row * input_dimension;
	float total = 0.0f,partial,scale;
	uint4 packed;
	const uint16_t *pairs;
	float2 values;
	if ( output >= output_dimension )
		return;
	weights = weight_e4m3 + (uint64_t)output * input_dimension;
	for (k = lane * 16u; k < input_dimension; k += 32u * 16u)
	{
		packed = *(const uint4 *)(weights + k);
		pairs = (const uint16_t *)&packed;
		scale = LmUe8m0ToFloat(scale_ue8m0[(uint64_t)(output / SPARK_DSV41_FLASH_FP8_BLOCK) * scale_columns + k / SPARK_DSV41_FLASH_FP8_BLOCK]);
		partial = 0.0f;
		for (pair = 0u; pair < 8u; pair++)
		{
			values = LmE4m3PairToFloat2(pairs[pair]);
			partial += values.x * activation[k + 2u * pair] + values.y * activation[k + 2u * pair + 1u];
		}
		total += partial * scale;
	}
	for (offset = 16u; offset != 0u; offset >>= 1u)
		total += __shfl_down_sync(0xffffffffu,total,offset);
	if ( lane == 0u )
		output_bf16[(uint64_t)row * output_dimension + output] = LmFloatToBf16(total);
}

static inline cudaError_t SparkDsv41FlashLaunchFp8BlockLinear(
	cudaStream_t stream,
	const uint8_t *weight_e4m3,
	const uint8_t *scale_ue8m0,
	const uint16_t *input_bf16,
	float *activation_scratch_f32,
	uint16_t *output_bf16,
	uint32_t row_count,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	uint64_t groups;
	cudaError_t status;
	if ( weight_e4m3 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || activation_scratch_f32 == 0 ||
		output_bf16 == 0 || row_count == 0u || input_dimension == 0u || input_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u ||
		output_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u ||
		((uintptr_t)weight_e4m3 % 16u) != 0u )
		return(cudaErrorInvalidValue);
	groups = (uint64_t)row_count * (input_dimension / SPARK_DSV41_FLASH_FP8_BLOCK);
	SparkDsv41FlashActQuantKernel<<<(uint32_t)((groups + 255u) / 256u),256u,0,stream>>>(
		input_bf16,activation_scratch_f32,row_count,input_dimension);
	status = cudaGetLastError();
	if ( status != cudaSuccess )
		return(status);
	SparkDsv41FlashFp8BlockLinearKernel<<<dim3((output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS,row_count),
		SPARK_DSV41_FLASH_LINEAR_WARPS * 32u,0,stream>>>(weight_e4m3,scale_ue8m0,activation_scratch_f32,output_bf16,
		input_dimension,output_dimension);
	return(cudaGetLastError());
}
