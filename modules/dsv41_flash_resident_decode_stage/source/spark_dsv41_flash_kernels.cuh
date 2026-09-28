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

#define SPARK_DSV41_FLASH_ROW_THREADS 256u
#define SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS 1024u

static __global__ void SparkDsv41FlashRmsNormKernel(const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t dimension, float epsilon)
{
	__shared__ float scratch[SPARK_DSV41_FLASH_ROW_THREADS / 32u];
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t element;
	float square = 0.0f,value,inverse;
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
	{
		value = LmBf16ToFloat(input_bf16[base + element]);
		square += value * value;
	}
	square = SparkDsv41FlashBlockSum(square,scratch);
	inverse = rsqrtf(square / (float)dimension + epsilon);
	for (element = threadIdx.x; element < dimension; element += blockDim.x)
		output_bf16[base + element] = LmFloatToBf16(LmBf16ToFloat(weight_bf16[element]) * (LmBf16ToFloat(input_bf16[base + element]) * inverse));
}

static inline cudaError_t SparkDsv41FlashLaunchRmsNorm(cudaStream_t stream, const uint16_t *input_bf16, const uint16_t *weight_bf16, uint16_t *output_bf16, uint32_t row_count, uint32_t dimension, float epsilon)
{
	if ( input_bf16 == 0 || weight_bf16 == 0 || output_bf16 == 0 || row_count == 0u || dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashRmsNormKernel<<<row_count,SPARK_DSV41_FLASH_ROW_THREADS,0,stream>>>(input_bf16,weight_bf16,output_bf16,dimension,epsilon);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashRopeKernel(uint16_t *data_bf16, const float *cos_sin_f32, uint32_t head_count, uint32_t head_dimension, uint32_t rope_dimension, uint32_t inverse)
{
	uint32_t head = blockIdx.x,pair = threadIdx.x;
	uint64_t base;
	float real,imaginary,cosine,sine;
	if ( head >= head_count || pair >= rope_dimension / 2u )
		return;
	base = (uint64_t)head * head_dimension + (head_dimension - rope_dimension) + 2u * pair;
	cosine = cos_sin_f32[2u * pair];
	sine = inverse != 0u ? -cos_sin_f32[2u * pair + 1u] : cos_sin_f32[2u * pair + 1u];
	real = LmBf16ToFloat(data_bf16[base]);
	imaginary = LmBf16ToFloat(data_bf16[base + 1u]);
	data_bf16[base] = LmFloatToBf16(real * cosine - imaginary * sine);
	data_bf16[base + 1u] = LmFloatToBf16(real * sine + imaginary * cosine);
}

static inline cudaError_t SparkDsv41FlashLaunchRope(cudaStream_t stream, uint16_t *data_bf16, const float *cos_sin_f32, uint32_t head_count, uint32_t head_dimension, uint32_t rope_dimension, uint32_t inverse)
{
	if ( data_bf16 == 0 || cos_sin_f32 == 0 || head_count == 0u || rope_dimension == 0u || rope_dimension > head_dimension || rope_dimension / 2u > 1024u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashRopeKernel<<<head_count,rope_dimension / 2u,0,stream>>>(data_bf16,cos_sin_f32,head_count,head_dimension,rope_dimension,inverse);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashFp8QdqKernel(uint16_t *data_bf16, uint32_t group_count)
{
	uint32_t group = blockIdx.x * blockDim.x + threadIdx.x,element;
	uint64_t base;
	float amax = 0.0f,scale,value;
	if ( group >= group_count )
		return;
	base = (uint64_t)group * SPARK_DSV41_FLASH_FP8_BLOCK;
	for (element = 0u; element < SPARK_DSV41_FLASH_FP8_BLOCK; element++)
		amax = fmaxf(amax,fabsf(LmBf16ToFloat(data_bf16[base + element])));
	scale = SparkDsv41FlashPow2CeilScale(fmaxf(amax,SPARK_DSV41_FLASH_FP8_AMAX_FLOOR),1.0f / LM_E4M3_MAX);
	for (element = 0u; element < SPARK_DSV41_FLASH_FP8_BLOCK; element++)
	{
		value = fminf(fmaxf(LmBf16ToFloat(data_bf16[base + element]) / scale,-LM_E4M3_MAX),LM_E4M3_MAX);
		data_bf16[base + element] = LmFloatToBf16(LmE4m3ToFloat(LmFloatToE4m3(value)) * scale);
	}
}

static inline cudaError_t SparkDsv41FlashLaunchFp8Qdq(cudaStream_t stream, uint16_t *data_bf16, uint64_t element_count)
{
	uint32_t groups;
	if ( data_bf16 == 0 || element_count == 0u || element_count % SPARK_DSV41_FLASH_FP8_BLOCK != 0u )
		return(cudaErrorInvalidValue);
	groups = (uint32_t)(element_count / SPARK_DSV41_FLASH_FP8_BLOCK);
	SparkDsv41FlashFp8QdqKernel<<<(groups + 255u) / 256u,256u,0,stream>>>(data_bf16,groups);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashSinkAttentionKernel(
	const uint16_t *query_bf16,
	const uint16_t *keys_bf16,
	const int32_t *indices,
	uint32_t index_count,
	const float *sink_f32,
	uint16_t *output_bf16,
	uint32_t head_dimension,
	float softmax_scale)
{
	__shared__ float probabilities[SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS];
	__shared__ float rounded[SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS];
	__shared__ float scratch[SPARK_DSV41_FLASH_ROW_THREADS / 32u];
	uint32_t head = blockIdx.x,lane = threadIdx.x % 32u,warp = threadIdx.x / 32u,slot,element,offset;
	const uint16_t *query = query_bf16 + (uint64_t)head * head_dimension;
	float score,top = -1e30f,denominator = 0.0f,value;
	for (slot = warp; slot < index_count; slot += blockDim.x / 32u)
	{
		score = 0.0f;
		if ( indices[slot] >= 0 )
			for (element = lane; element < head_dimension; element += 32u)
				score += LmBf16ToFloat(query[element]) * LmBf16ToFloat(keys_bf16[(uint64_t)indices[slot] * head_dimension + element]);
		for (offset = 16u; offset != 0u; offset >>= 1u)
			score += __shfl_down_sync(0xffffffffu,score,offset);
		if ( lane == 0u )
			probabilities[slot] = indices[slot] >= 0 ? score * softmax_scale : -INFINITY;
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for (slot = 0u; slot < index_count; slot++)
			top = fmaxf(top,probabilities[slot]);
		scratch[0] = top;
	}
	__syncthreads();
	top = scratch[0];
	__syncthreads();
	for (slot = threadIdx.x; slot < index_count; slot += blockDim.x)
	{
		value = expf(probabilities[slot] - top);
		probabilities[slot] = value;
		rounded[slot] = LmBf16ToFloat(LmFloatToBf16(value));
	}
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		for (slot = 0u; slot < index_count; slot++)
			denominator += probabilities[slot];
		scratch[0] = denominator + expf(sink_f32[head] - top);
	}
	__syncthreads();
	denominator = scratch[0];
	for (element = threadIdx.x; element < head_dimension; element += blockDim.x)
	{
		value = 0.0f;
		for (slot = 0u; slot < index_count; slot++)
			if ( indices[slot] >= 0 )
				value += rounded[slot] * LmBf16ToFloat(keys_bf16[(uint64_t)indices[slot] * head_dimension + element]);
		output_bf16[(uint64_t)head * head_dimension + element] = LmFloatToBf16(value / denominator);
	}
}

static inline cudaError_t SparkDsv41FlashLaunchSinkAttention(cudaStream_t stream, const uint16_t *query_bf16, const uint16_t *keys_bf16, const int32_t *indices, uint32_t index_count, const float *sink_f32, uint16_t *output_bf16, uint32_t head_count, uint32_t head_dimension, float softmax_scale)
{
	if ( query_bf16 == 0 || keys_bf16 == 0 || indices == 0 || sink_f32 == 0 || output_bf16 == 0 ||
		index_count == 0u || index_count > SPARK_DSV41_FLASH_ATTENTION_MAX_KEYS || head_count == 0u || head_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashSinkAttentionKernel<<<head_count,SPARK_DSV41_FLASH_ROW_THREADS,0,stream>>>(query_bf16,keys_bf16,indices,index_count,sink_f32,output_bf16,head_dimension,softmax_scale);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashGroupedFp8LinearKernel(
	const uint8_t *weight_e4m3,
	const uint8_t *scale_ue8m0,
	const uint16_t *input_bf16,
	uint16_t *output_bf16,
	uint32_t input_dimension,
	uint32_t output_dimension,
	uint32_t outputs_per_group)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u;
	uint32_t output = blockIdx.x * SPARK_DSV41_FLASH_LINEAR_WARPS + warp,k,pair,offset;
	uint32_t scale_columns = input_dimension / SPARK_DSV41_FLASH_FP8_BLOCK;
	const uint16_t *activation;
	const uint8_t *weights;
	float total = 0.0f,partial,scale;
	uint4 packed;
	const uint16_t *pairs;
	float2 values;
	if ( output >= output_dimension )
		return;
	activation = input_bf16 + (uint64_t)(output / outputs_per_group) * input_dimension;
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
			partial += values.x * scale * LmBf16ToFloat(activation[k + 2u * pair]) + values.y * scale * LmBf16ToFloat(activation[k + 2u * pair + 1u]);
		}
		total += partial;
	}
	for (offset = 16u; offset != 0u; offset >>= 1u)
		total += __shfl_down_sync(0xffffffffu,total,offset);
	if ( lane == 0u )
		output_bf16[output] = LmFloatToBf16(total);
}

static inline cudaError_t SparkDsv41FlashLaunchGroupedFp8Linear(cudaStream_t stream, const uint8_t *weight_e4m3, const uint8_t *scale_ue8m0, const uint16_t *input_bf16, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension, uint32_t outputs_per_group)
{
	if ( weight_e4m3 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || output_bf16 == 0 || input_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u ||
		output_dimension % SPARK_DSV41_FLASH_FP8_BLOCK != 0u || outputs_per_group == 0u || output_dimension % outputs_per_group != 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashGroupedFp8LinearKernel<<<(output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS,
		SPARK_DSV41_FLASH_LINEAR_WARPS * 32u,0,stream>>>(weight_e4m3,scale_ue8m0,input_bf16,output_bf16,input_dimension,output_dimension,outputs_per_group);
	return(cudaGetLastError());
}

#define SPARK_DSV41_FLASH_MAX_EXPERTS 512u
#define SPARK_DSV41_FLASH_MAX_ROUTED 16u

static __global__ void SparkDsv41FlashGateKernel(
	const uint16_t *input_bf16,
	const uint16_t *weight_bf16,
	const float *bias_f32,
	uint32_t *indices_out,
	float *weights_out,
	uint32_t expert_count,
	uint32_t dimension,
	uint32_t topk,
	float route_scale)
{
	__shared__ float scores[SPARK_DSV41_FLASH_MAX_EXPERTS];
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u,expert,element,offset,pick,best,chosen[SPARK_DSV41_FLASH_MAX_ROUTED];
	float total,value,best_value,sum = 0.0f;
	for (expert = warp; expert < expert_count; expert += blockDim.x / 32u)
	{
		total = 0.0f;
		for (element = lane; element < dimension; element += 32u)
			total += LmBf16ToFloat(input_bf16[element]) * LmBf16ToFloat(weight_bf16[(uint64_t)expert * dimension + element]);
		for (offset = 16u; offset != 0u; offset >>= 1u)
			total += __shfl_down_sync(0xffffffffu,total,offset);
		if ( lane == 0u )
			scores[expert] = sqrtf(total > 20.0f ? total : log1pf(expf(total)));
	}
	__syncthreads();
	if ( threadIdx.x != 0u )
		return;
	for (pick = 0u; pick < topk; pick++)
	{
		best = UINT32_MAX;
		best_value = -INFINITY;
		for (expert = 0u; expert < expert_count; expert++)
		{
			value = scores[expert] + bias_f32[expert];
			for (offset = 0u; offset < pick; offset++)
				if ( chosen[offset] == expert )
					value = -INFINITY;
			if ( value > best_value )
			{
				best_value = value;
				best = expert;
			}
		}
		chosen[pick] = best;
		sum += scores[best];
	}
	for (pick = 0u; pick < topk; pick++)
	{
		indices_out[pick] = chosen[pick];
		weights_out[pick] = scores[chosen[pick]] / (sum + 1e-20f) * route_scale;
	}
}

static inline cudaError_t SparkDsv41FlashLaunchGate(cudaStream_t stream, const uint16_t *input_bf16, const uint16_t *weight_bf16, const float *bias_f32, uint32_t *indices_out, float *weights_out, uint32_t expert_count, uint32_t dimension, uint32_t topk, float route_scale)
{
	if ( input_bf16 == 0 || weight_bf16 == 0 || bias_f32 == 0 || indices_out == 0 || weights_out == 0 ||
		expert_count == 0u || expert_count > SPARK_DSV41_FLASH_MAX_EXPERTS || topk == 0u || topk > SPARK_DSV41_FLASH_MAX_ROUTED || topk > expert_count )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashGateKernel<<<1u,SPARK_DSV41_FLASH_ROW_THREADS,0,stream>>>(input_bf16,weight_bf16,bias_f32,indices_out,weights_out,expert_count,dimension,topk,route_scale);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashMxfp4LinearKernel(
	const uint8_t *payload_e2m1,
	const uint8_t *scale_ue8m0,
	const float *activation_f32,
	uint16_t *output_bf16,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u;
	uint32_t output = blockIdx.x * SPARK_DSV41_FLASH_LINEAR_WARPS + warp,k,byte,offset;
	const uint8_t *weights;
	float total = 0.0f,partial;
	uint4 packed;
	const uint8_t *bytes;
	float2 values;
	if ( output >= output_dimension )
		return;
	weights = payload_e2m1 + (uint64_t)output * (input_dimension / 2u);
	for (k = lane * 32u; k < input_dimension; k += 32u * 32u)
	{
		packed = *(const uint4 *)(weights + k / 2u);
		bytes = (const uint8_t *)&packed;
		partial = 0.0f;
		for (byte = 0u; byte < 16u; byte++)
		{
			values = LmE2m1PairToFloat(bytes[byte]);
			partial += values.x * activation_f32[k + 2u * byte] + values.y * activation_f32[k + 2u * byte + 1u];
		}
		total += partial * LmUe8m0ToFloat(scale_ue8m0[(uint64_t)output * (input_dimension / 32u) + k / 32u]);
	}
	for (offset = 16u; offset != 0u; offset >>= 1u)
		total += __shfl_down_sync(0xffffffffu,total,offset);
	if ( lane == 0u )
		output_bf16[output] = LmFloatToBf16(total);
}

static inline cudaError_t SparkDsv41FlashLaunchMxfp4Linear(cudaStream_t stream, const uint8_t *payload_e2m1, const uint8_t *scale_ue8m0, const uint16_t *input_bf16, float *activation_scratch_f32, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	cudaError_t status;
	if ( payload_e2m1 == 0 || scale_ue8m0 == 0 || input_bf16 == 0 || activation_scratch_f32 == 0 || output_bf16 == 0 ||
		input_dimension == 0u || input_dimension % 32u != 0u || output_dimension == 0u || ((uintptr_t)payload_e2m1 % 16u) != 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashActQuantKernel<<<(input_dimension / 32u + 255u) / 256u,256u,0,stream>>>(input_bf16,activation_scratch_f32,1u,input_dimension);
	status = cudaGetLastError();
	if ( status != cudaSuccess )
		return(status);
	SparkDsv41FlashMxfp4LinearKernel<<<(output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS,
		SPARK_DSV41_FLASH_LINEAR_WARPS * 32u,0,stream>>>(payload_e2m1,scale_ue8m0,activation_scratch_f32,output_bf16,input_dimension,output_dimension);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashSwigluKernel(const uint16_t *gate_bf16, const uint16_t *up_bf16, uint16_t *output_bf16, uint32_t width, float limit, float weight)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	float gate,up;
	if ( element >= width )
		return;
	gate = fminf(LmBf16ToFloat(gate_bf16[element]),limit);
	up = fminf(fmaxf(LmBf16ToFloat(up_bf16[element]),-limit),limit);
	output_bf16[element] = LmFloatToBf16(weight * ((gate / (1.0f + expf(-gate))) * up));
}

static inline cudaError_t SparkDsv41FlashLaunchSwiglu(cudaStream_t stream, const uint16_t *gate_bf16, const uint16_t *up_bf16, uint16_t *output_bf16, uint32_t width, float limit, float weight)
{
	if ( gate_bf16 == 0 || up_bf16 == 0 || output_bf16 == 0 || width == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashSwigluKernel<<<(width + 255u) / 256u,256u,0,stream>>>(gate_bf16,up_bf16,output_bf16,width,limit,weight);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashAccumulateKernel(float *accumulator_f32, const uint16_t *value_bf16, uint32_t width)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < width )
		accumulator_f32[element] += LmBf16ToFloat(value_bf16[element]);
}

static __global__ void SparkDsv41FlashNarrowKernel(const float *input_f32, uint16_t *output_bf16, uint32_t width)
{
	uint32_t element = blockIdx.x * blockDim.x + threadIdx.x;
	if ( element < width )
		output_bf16[element] = LmFloatToBf16(input_f32[element]);
}

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

static __global__ void SparkDsv41FlashBf16LinearKernel(
	const uint16_t *weight_bf16,
	const uint16_t *input_bf16,
	float *output_f32,
	uint16_t *output_bf16,
	uint32_t input_dimension,
	uint32_t output_dimension)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u;
	uint32_t output = blockIdx.x * SPARK_DSV41_FLASH_LINEAR_WARPS + warp,k,offset;
	const uint16_t *weights;
	float total = 0.0f;
	if ( output >= output_dimension )
		return;
	weights = weight_bf16 + (uint64_t)output * input_dimension;
	for (k = lane; k < input_dimension; k += 32u)
		total += LmBf16ToFloat(weights[k]) * LmBf16ToFloat(input_bf16[k]);
	for (offset = 16u; offset != 0u; offset >>= 1u)
		total += __shfl_down_sync(0xffffffffu,total,offset);
	if ( lane != 0u )
		return;
	if ( output_f32 != 0 )
		output_f32[output] = total;
	if ( output_bf16 != 0 )
		output_bf16[output] = LmFloatToBf16(total);
}

static inline cudaError_t SparkDsv41FlashLaunchBf16Linear(cudaStream_t stream, const uint16_t *weight_bf16, const uint16_t *input_bf16, float *output_f32, uint16_t *output_bf16, uint32_t input_dimension, uint32_t output_dimension)
{
	if ( weight_bf16 == 0 || input_bf16 == 0 || (output_f32 == 0 && output_bf16 == 0) || input_dimension == 0u || output_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashBf16LinearKernel<<<(output_dimension + SPARK_DSV41_FLASH_LINEAR_WARPS - 1u) / SPARK_DSV41_FLASH_LINEAR_WARPS,
		SPARK_DSV41_FLASH_LINEAR_WARPS * 32u,0,stream>>>(weight_bf16,input_bf16,output_f32,output_bf16,input_dimension,output_dimension);
	return(cudaGetLastError());
}

static __global__ void SparkDsv41FlashCompressPoolKernel(const float *kv_state_f32, const float *score_state_f32, uint16_t *pooled_bf16, uint32_t ratio, uint32_t width)
{
	uint32_t channel = blockIdx.x * blockDim.x + threadIdx.x,slot;
	float top = -INFINITY,sum = 0.0f,total = 0.0f,weight;
	if ( channel >= width )
		return;
	for (slot = 0u; slot < ratio; slot++)
		top = fmaxf(top,score_state_f32[(uint64_t)slot * width + channel]);
	for (slot = 0u; slot < ratio; slot++)
		sum += expf(score_state_f32[(uint64_t)slot * width + channel] - top);
	for (slot = 0u; slot < ratio; slot++)
	{
		weight = expf(score_state_f32[(uint64_t)slot * width + channel] - top) / sum;
		total += kv_state_f32[(uint64_t)slot * width + channel] * weight;
	}
	pooled_bf16[channel] = LmFloatToBf16(total);
}

static __global__ void SparkDsv41FlashFp4Pow2QdqKernel(uint16_t *data_bf16, uint32_t group_count, uint32_t group_size)
{
	uint32_t group = blockIdx.x * blockDim.x + threadIdx.x,element;
	uint64_t base;
	float amax = 0.0f,scale,value;
	if ( group >= group_count )
		return;
	base = (uint64_t)group * group_size;
	for (element = 0u; element < group_size; element++)
		amax = fmaxf(amax,fabsf(LmBf16ToFloat(data_bf16[base + element])));
	scale = SparkDsv41FlashPow2CeilScale(fmaxf(amax,LM_E2M1_MAX * 1.1754943508222875e-38f),1.0f / LM_E2M1_MAX);
	for (element = 0u; element < group_size; element++)
	{
		value = fminf(fmaxf(LmBf16ToFloat(data_bf16[base + element]) / scale,-LM_E2M1_MAX),LM_E2M1_MAX);
		data_bf16[base + element] = LmFloatToBf16(SparkDsv41FlashE2m1Round(value) * scale);
	}
}

static __global__ void SparkDsv41FlashIndexScoreKernel(const uint16_t *query_bf16, const uint16_t *keys_bf16, const uint16_t *head_weights_bf16, float *scores_f32, uint32_t key_count, uint32_t head_count, uint32_t head_dimension)
{
	uint32_t lane = threadIdx.x % 32u,warp = threadIdx.x / 32u;
	uint32_t key = blockIdx.x * (blockDim.x / 32u) + warp,head,element,offset;
	float total = 0.0f,dot;
	if ( key >= key_count )
		return;
	for (head = 0u; head < head_count; head++)
	{
		dot = 0.0f;
		for (element = lane; element < head_dimension; element += 32u)
			dot += LmBf16ToFloat(query_bf16[(uint64_t)head * head_dimension + element]) * LmBf16ToFloat(keys_bf16[(uint64_t)key * head_dimension + element]);
		for (offset = 16u; offset != 0u; offset >>= 1u)
			dot += __shfl_xor_sync(0xffffffffu,dot,offset);
		total += fmaxf(dot,0.0f) * LmBf16ToFloat(head_weights_bf16[head]);
	}
	if ( lane == 0u )
		scores_f32[key] = total;
}

static inline cudaError_t SparkDsv41FlashLaunchIndexScore(cudaStream_t stream, const uint16_t *query_bf16, const uint16_t *keys_bf16, const uint16_t *head_weights_bf16, float *scores_f32, uint32_t key_count, uint32_t head_count, uint32_t head_dimension)
{
	if ( query_bf16 == 0 || keys_bf16 == 0 || head_weights_bf16 == 0 || scores_f32 == 0 || key_count == 0u || head_count == 0u || head_dimension == 0u )
		return(cudaErrorInvalidValue);
	SparkDsv41FlashIndexScoreKernel<<<(key_count + 7u) / 8u,256u,0,stream>>>(query_bf16,keys_bf16,head_weights_bf16,scores_f32,key_count,head_count,head_dimension);
	return(cudaGetLastError());
}
