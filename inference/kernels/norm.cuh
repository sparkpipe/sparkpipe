#pragma once


#include "inference/kernels/dependent_launch.cuh"
#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/bf16.cuh"
#include <stdint.h>

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockSumOnce(float value, float *shared)
{
	constexpr uint32_t warps = THREADS / LM_WARP_LANES;
	const uint32_t lane = threadIdx.x & (LM_WARP_LANES - 1u), warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	if constexpr ( warps == 1u )
		return(__shfl_sync(0xffffffffu,value,0u));
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = lane < warps ? shared[lane] : 0.0f;
	#pragma unroll
	for (uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u)
		value += __shfl_down_sync(0xffffffffu,value,offset);
	return(__shfl_sync(0xffffffffu,value,0u));
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockSum(float value, float *shared)
{
	constexpr uint32_t warps = THREADS / LM_WARP_LANES;
	value = LmBlockSumOnce<THREADS>(value,shared);
	if constexpr ( warps > 1u )
		__syncthreads();
	return(value);
}

template<uint32_t THREADS>
static __device__ __forceinline__ void LmBlockSum2Once(float *first, float *second, float *shared)
{
	constexpr uint32_t warps = THREADS / LM_WARP_LANES;
	const uint32_t lane = threadIdx.x & (LM_WARP_LANES - 1u), warp = threadIdx.x >> 5u;
	float a = *first, b = *second;
	#pragma unroll
	for (uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
	{
		a += __shfl_down_sync(0xffffffffu,a,offset);
		b += __shfl_down_sync(0xffffffffu,b,offset);
	}
	if constexpr ( warps == 1u )
	{
		*first = __shfl_sync(0xffffffffu,a,0u);
		*second = __shfl_sync(0xffffffffu,b,0u);
		return;
	}
	if ( lane == 0u )
	{
		shared[warp] = a;
		shared[warps + warp] = b;
	}
	__syncthreads();
	a = lane < warps ? shared[lane] : 0.0f;
	b = lane < warps ? shared[warps + lane] : 0.0f;
	#pragma unroll
	for (uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u)
	{
		a += __shfl_down_sync(0xffffffffu,a,offset);
		b += __shfl_down_sync(0xffffffffu,b,offset);
	}
	*first = __shfl_sync(0xffffffffu,a,0u);
	*second = __shfl_sync(0xffffffffu,b,0u);
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockMaxOnce(float value, float *shared)
{
	constexpr uint32_t warps = THREADS / LM_WARP_LANES;
	const uint32_t lane = threadIdx.x & (LM_WARP_LANES - 1u), warp = threadIdx.x >> 5u;
	#pragma unroll
	for (uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	if constexpr ( warps == 1u )
		return(__shfl_sync(0xffffffffu,value,0u));
	if ( lane == 0u )
		shared[warp] = value;
	__syncthreads();
	value = lane < warps ? shared[lane] : 0.0f;
	#pragma unroll
	for (uint32_t offset = warps / 2u; offset > 0u; offset >>= 1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	return(__shfl_sync(0xffffffffu,value,0u));
}

template<uint32_t THREADS>
static __device__ __forceinline__ float LmBlockMax(float value, float *shared)
{
	constexpr uint32_t warps = THREADS / LM_WARP_LANES;
	value = LmBlockMaxOnce<THREADS>(value,shared);
	if constexpr ( warps > 1u )
		__syncthreads();
	return(value);
}

static __device__ __forceinline__ bool LmIsAligned16(const void *pointer)
{
	return(((uintptr_t)pointer & 15u) == 0u);
}

static __device__ __forceinline__ void LmUnpackUint4Bf16(uint4 packed, uint16_t values[8])
{
	values[0] = (uint16_t)packed.x;
	values[1] = (uint16_t)(packed.x >> 16u);
	values[2] = (uint16_t)packed.y;
	values[3] = (uint16_t)(packed.y >> 16u);
	values[4] = (uint16_t)packed.z;
	values[5] = (uint16_t)(packed.z >> 16u);
	values[6] = (uint16_t)packed.w;
	values[7] = (uint16_t)(packed.w >> 16u);
}

static __device__ __forceinline__ uint4 LmPackUint4Bf16(const uint16_t values[8])
{
	return(make_uint4(
		(uint32_t)values[0] | ((uint32_t)values[1] << 16u),
		(uint32_t)values[2] | ((uint32_t)values[3] << 16u),
		(uint32_t)values[4] | ((uint32_t)values[5] << 16u),
		(uint32_t)values[6] | ((uint32_t)values[7] << 16u)));
}

static __device__ __forceinline__ uint4 LmPackUint4FloatToBf16(const float values[8])
{
	return(make_uint4(
		(uint32_t)LmFloatToBf16(values[0]) | ((uint32_t)LmFloatToBf16(values[1]) << 16u),
		(uint32_t)LmFloatToBf16(values[2]) | ((uint32_t)LmFloatToBf16(values[3]) << 16u),
		(uint32_t)LmFloatToBf16(values[4]) | ((uint32_t)LmFloatToBf16(values[5]) << 16u),
		(uint32_t)LmFloatToBf16(values[6]) | ((uint32_t)LmFloatToBf16(values[7]) << 16u)));
}

template<class Weight>
static __device__ __forceinline__ void LmLoadWeight8(const Weight *weight, uint32_t offset, float values[8])
{
	if constexpr ( sizeof(Weight) == 2u )
	{
		uint16_t bits[8];
		LmUnpackUint4Bf16(*(const uint4 *)(weight + offset), bits);
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k)
			values[k] = LmBf16ToFloat(bits[k]);
	}
	else
	{
		float4 lo = *(const float4 *)(weight + offset);
		float4 hi = *(const float4 *)(weight + offset + 4u);
		values[0] = lo.x; values[1] = lo.y; values[2] = lo.z; values[3] = lo.w;
		values[4] = hi.x; values[5] = hi.y; values[6] = hi.z; values[7] = hi.w;
	}
}

static __device__ __forceinline__ void LmStoreFloat8(float *destination, const float values[8])
{
	float4 lo = {values[0], values[1], values[2], values[3]};
	float4 hi = {values[4], values[5], values[6], values[7]};
	*(float4 *)(destination) = lo;
	*(float4 *)(destination + 4u) = hi;
}

static __device__ __forceinline__ void LmLoadFloat8(const float *source, float values[8])
{
	float4 lo = *(const float4 *)(source);
	float4 hi = *(const float4 *)(source + 4u);
	values[0] = lo.x; values[1] = lo.y; values[2] = lo.z; values[3] = lo.w;
	values[4] = hi.x; values[5] = hi.y; values[6] = hi.z; values[7] = hi.w;
}

#define LM_RMS_NORM_STAGED 24u

template<uint32_t THREADS, class Weight, bool ROUND_FIRST = false>
__global__ __launch_bounds__(THREADS, 1)
void LmFusedResidualRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16, const Weight *__restrict__ weight, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float total = 0.0f,scale;
	LmDependentRelease();
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(output_bf16) && LmIsAligned16(weight) &&
		LmIsAligned16(row) &&
		(residual_bf16 == 0 || LmIsAligned16(residual_bf16)) &&
		(residual_out_bf16 == 0 || LmIsAligned16(residual_out_bf16));
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged[vec_slots];
			uint32_t slot;
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension )
				{
					uint4 in_vec = *(const uint4 *)(input_bf16 + base + offset);
					uint16_t bits[8];
					float sq[8];
					LmUnpackUint4Bf16(in_vec, bits);
					if ( residual_bf16 != 0 )
					{
						uint16_t res_bits[8];
						LmUnpackUint4Bf16(*(const uint4 *)(residual_bf16 + base + offset), res_bits);
						#pragma unroll
						for (uint32_t k = 0u; k < 8u; ++k)
						{
							bits[k] = LmFloatToBf16(LmBf16ToFloat(bits[k]) + LmBf16ToFloat(res_bits[k]));
							const float value = LmBf16ToFloat(bits[k]);
							sq[k] = value * value;
						}
						staged[slot] = LmPackUint4Bf16(bits);
					}
					else
					{
						staged[slot] = in_vec;
						#pragma unroll
						for (uint32_t k = 0u; k < 8u; ++k)
						{
							const float value = LmBf16ToFloat(bits[k]);
							sq[k] = value * value;
						}
					}
					LmStoreFloat8(row + offset, sq);
				}
				else
					staged[slot] = make_uint4(0u, 0u, 0u, 0u);
			}
			__syncthreads();
			#pragma unroll
			for (slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
			{
				index = threadIdx.x + slot * THREADS;
				if ( index < dimension )
					total += row[index];
			}
			total = LmBlockSumOnce<THREADS>(total,reduction);
			scale = SparkMathRmsScale(total, (float)dimension, epsilon);
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				if ( residual_out_bf16 != 0 )
					*(uint4 *)(residual_out_bf16 + base + offset) = staged[slot];
				uint16_t bits[8];
				float w[8], out_f[8];
				LmUnpackUint4Bf16(staged[slot], bits);
				LmLoadWeight8(weight, offset, w);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					const float value = LmBf16ToFloat(bits[k]);
					out_f[k] = ROUND_FIRST ? SparkMathNormGainRounded(value * scale, w[k]) : value * scale * w[k];
				}
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
		{
			uint16_t bits[8];
			float values[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + offset), bits);
			if ( residual_bf16 != 0 )
			{
				uint16_t res_bits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(residual_bf16 + base + offset), res_bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					values[k] = SparkMathResidualSum(LmBf16ToFloat(bits[k]), LmBf16ToFloat(res_bits[k]));
			}
			else
			{
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					values[k] = LmBf16ToFloat(bits[k]);
			}
			LmStoreFloat8(row + offset, values);
		}
		__syncthreads();
		for (index = threadIdx.x; index < dimension; index += THREADS)
		{
			const float value = row[index];
			total += value * value;
		}
		total = LmBlockSumOnce<THREADS>(total,reduction);
		scale = SparkMathRmsScale(total, (float)dimension, epsilon);
		for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
		{
			float values[8], w[8], out_f[8];
			LmLoadFloat8(row + offset, values);
			if ( residual_out_bf16 != 0 )
				*(uint4 *)(residual_out_bf16 + base + offset) = LmPackUint4FloatToBf16(values);
			LmLoadWeight8(weight, offset, w);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = ROUND_FIRST ? SparkMathNormGainRounded(values[k] * scale, w[k]) : values[k] * scale * w[k];
			*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		uint16_t staged[LM_RMS_NORM_STAGED];
		uint32_t slot;
		#pragma unroll
		for (slot = 0u; slot < LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			if ( index < dimension )
			{
				uint16_t bits = input_bf16[base + index];
				float value = LmBf16ToFloat(bits);
				if ( residual_bf16 != 0 )
				{
					bits = LmFloatToBf16(value + LmBf16ToFloat(residual_bf16[base + index]));
					value = LmBf16ToFloat(bits);
				}
				staged[slot] = bits;
				total += value * value;
			}
			else
				staged[slot] = 0u;
		}
		total = LmBlockSumOnce<THREADS>(total,reduction);
		scale = SparkMathRmsScale(total, (float)dimension, epsilon);
		#pragma unroll
		for (slot = 0u; slot < LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			if ( index >= dimension )
				continue;
			if ( residual_out_bf16 != 0 )
				residual_out_bf16[base + index] = staged[slot];
			const float value = LmBf16ToFloat(staged[slot]);
			output_bf16[base + index] = LmFloatToBf16(ROUND_FIRST ? SparkMathNormGainRounded(value * scale, LmScalarToFloat(weight[index])) :
				value * scale * LmScalarToFloat(weight[index]));
		}
		return;
	}
	#pragma unroll 8
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		if ( residual_bf16 != 0 )
			value = SparkMathResidualSum(value, LmBf16ToFloat(residual_bf16[base + index]));
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSumOnce<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	#pragma unroll 8
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		if ( residual_out_bf16 != 0 )
			residual_out_bf16[base + index] = LmFloatToBf16(row[index]);
		output_bf16[base + index] = LmFloatToBf16(ROUND_FIRST ? SparkMathNormGainRounded(row[index] * scale, LmScalarToFloat(weight[index])) :
			row[index] * scale * LmScalarToFloat(weight[index]));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmBf16RmsNormKernel(const uint16_t *input,const uint16_t *weight,uint16_t *output,uint32_t dimension,uint32_t row_stride,float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared,*reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index,slot;
	float total = 0.0f,scale,value;
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input) && LmIsAligned16(weight) && LmIsAligned16(output) && LmIsAligned16(row);
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged_input[vec_slots];
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension )
				{
					uint4 in_vec = *(const uint4 *)(input + base + offset);
					uint16_t bits[8];
					float sq[8];
					staged_input[slot] = in_vec;
					LmUnpackUint4Bf16(in_vec, bits);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
					{
						value = LmBf16ToFloat(bits[k]);
						sq[k] = value * value;
					}
					LmStoreFloat8(row + offset, sq);
				}
				else
					staged_input[slot] = make_uint4(0u, 0u, 0u, 0u);
			}
			__syncthreads();
			#pragma unroll
			for (slot = 0u; slot < LM_RMS_NORM_STAGED; ++slot)
			{
				index = threadIdx.x + slot * THREADS;
				if ( index < dimension )
					total += row[index];
			}
			total = LmBlockSumOnce<THREADS>(total,reduction);
			scale = SparkMathRmsScale(total, (float)dimension, epsilon);
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				uint16_t bits[8], wbits[8];
				float out_f[8];
				LmUnpackUint4Bf16(staged_input[slot], bits);
				LmUnpackUint4Bf16(*(const uint4 *)(weight + offset), wbits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					value = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(bits[k]) * scale));
					out_f[k] = value * LmBf16ToFloat(wbits[k]);
				}
				*(uint4 *)(output + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
		{
			uint16_t bits[8];
			float values[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input + base + offset), bits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				values[k] = LmBf16ToFloat(bits[k]);
			LmStoreFloat8(row + offset, values);
		}
		__syncthreads();
		for (index = threadIdx.x; index < dimension; index += THREADS)
		{
			value = row[index];
			total += value * value;
		}
		total = LmBlockSumOnce<THREADS>(total,reduction);
		scale = SparkMathRmsScale(total, (float)dimension, epsilon);
		for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
		{
			uint16_t wbits[8];
			float values[8], out_f[8];
			LmLoadFloat8(row + offset, values);
			LmUnpackUint4Bf16(*(const uint4 *)(weight + offset), wbits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				value = LmBf16ToFloat(LmFloatToBf16(values[k] * scale));
				out_f[k] = value * LmBf16ToFloat(wbits[k]);
			}
			*(uint4 *)(output + base + offset) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		uint16_t staged_input[LM_RMS_NORM_STAGED];
		#pragma unroll
		for (slot=0u; slot<LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			staged_input[slot] = index < dimension ? input[base + index] : (uint16_t)0u;
		}
		#pragma unroll
		for (slot=0u; slot<LM_RMS_NORM_STAGED; slot++)
		{
			value = LmBf16ToFloat(staged_input[slot]);
			if ( threadIdx.x + slot * THREADS < dimension )
				total += value * value;
		}
		total = LmBlockSumOnce<THREADS>(total,reduction);
		scale = SparkMathRmsScale(total, (float)dimension, epsilon);
		#pragma unroll
		for (slot=0u; slot<LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			if ( index >= dimension )
				continue;
			value = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(staged_input[slot]) * scale));
			output[base + index] = LmFloatToBf16(value * LmBf16ToFloat(weight[index]));
		}
		return;
	}
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(input[base + index]);
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSumOnce<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(LmFloatToBf16(row[index] * scale));
		output[base + index] = LmFloatToBf16(value * LmBf16ToFloat(weight[index]));
	}
}

template<uint32_t THREADS, class Weight>
__global__ __launch_bounds__(THREADS, 1)
void LmLayerNormKernel(const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight, const Weight *__restrict__ bias, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	__shared__ float pair_reduction[2u * (THREADS / LM_WARP_LANES)];
	float *row = lm_norm_shared;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float value,total = 0.0f,total_squared = 0.0f,mean,variance,inverse;
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(weight) && LmIsAligned16(bias) && LmIsAligned16(output_bf16);
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged[vec_slots];
			uint32_t slot;
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension )
				{
					uint4 in_vec = *(const uint4 *)(input_bf16 + base + offset);
					uint16_t bits[8];
					staged[slot] = in_vec;
					LmUnpackUint4Bf16(in_vec, bits);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
					{
						value = LmBf16ToFloat(bits[k]);
						total += value;
						total_squared += value * value;
					}
				}
				else
					staged[slot] = make_uint4(0u, 0u, 0u, 0u);
			}
			LmBlockSum2Once<THREADS>(&total,&total_squared,pair_reduction);
			mean = total / (float)dimension;
			variance = fmaxf((total_squared / (float)dimension) - (mean * mean),0.0f);
			inverse = rsqrtf(variance + epsilon);
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				uint16_t bits[8];
				float w[8], b[8], out_f[8];
				LmUnpackUint4Bf16(staged[slot], bits);
				LmLoadWeight8(weight, offset, w);
				LmLoadWeight8(bias, offset, b);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					out_f[k] = ((LmBf16ToFloat(bits[k]) - mean) * inverse * w[k]) + b[k];
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		if ( LmIsAligned16(row) )
		{
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				uint16_t bits[8];
				float values[8];
				LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + offset), bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					value = LmBf16ToFloat(bits[k]);
					values[k] = value;
					total += value;
					total_squared += value * value;
				}
				LmStoreFloat8(row + offset, values);
			}
			LmBlockSum2Once<THREADS>(&total,&total_squared,pair_reduction);
			mean = total / (float)dimension;
			variance = fmaxf((total_squared / (float)dimension) - (mean * mean),0.0f);
			inverse = rsqrtf(variance + epsilon);
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				float values[8], w[8], b[8], out_f[8];
				LmLoadFloat8(row + offset, values);
				LmLoadWeight8(weight, offset, w);
				LmLoadWeight8(bias, offset, b);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					out_f[k] = ((values[k] - mean) * inverse * w[k]) + b[k];
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
	}
	if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
	{
		uint16_t staged[LM_RMS_NORM_STAGED];
		uint32_t slot;
		#pragma unroll
		for (slot = 0u; slot < LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			if ( index < dimension )
			{
				staged[slot] = input_bf16[base + index];
				value = LmBf16ToFloat(staged[slot]);
				total += value;
				total_squared += value * value;
			}
			else
				staged[slot] = 0u;
		}
		LmBlockSum2Once<THREADS>(&total,&total_squared,pair_reduction);
		mean = total / (float)dimension;
		variance = fmaxf((total_squared / (float)dimension) - (mean * mean),0.0f);
		inverse = rsqrtf(variance + epsilon);
		#pragma unroll
		for (slot = 0u; slot < LM_RMS_NORM_STAGED; slot++)
		{
			index = threadIdx.x + slot * THREADS;
			if ( index >= dimension )
				continue;
			output_bf16[base + index] = LmFloatToBf16(((LmBf16ToFloat(staged[slot]) - mean) * inverse * LmScalarToFloat(weight[index])) + LmScalarToFloat(bias[index]));
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		row[index] = value;
		total += value;
		total_squared += value * value;
	}
	LmBlockSum2Once<THREADS>(&total,&total_squared,pair_reduction);
	mean = total / (float)dimension;
	variance = fmaxf((total_squared / (float)dimension) - (mean * mean),0.0f);
	inverse = rsqrtf(variance + epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] = LmFloatToBf16(((row[index] - mean) * inverse * LmScalarToFloat(weight[index])) + LmScalarToFloat(bias[index]));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(gate_up_bf16) && LmIsAligned16(output_bf16) )
	{
		const uint64_t gate_base = base + (gate_first ? 0u : dimension);
		const uint64_t up_base = base + (gate_first ? dimension : 0u);
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t gbits[8], ubits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + gate_base + index), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + up_base + index), ubits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = SparkMathSiluFast(LmBf16ToFloat(gbits[k])) * LmBf16ToFloat(ubits[k]);
			*(uint4 *)(output_bf16 + out_base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
		float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
		output_bf16[out_base + index] =
			LmFloatToBf16(SparkMathSiluFast(gate) * up);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first, float limit)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(gate_up_bf16) && LmIsAligned16(output_bf16) )
	{
		const uint64_t gate_base = base + (gate_first ? 0u : dimension);
		const uint64_t up_base = base + (gate_first ? dimension : 0u);
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t gbits[8], ubits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + gate_base + index), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + up_base + index), ubits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = SparkMathClampedGlu(SparkMathSiluFast(LmBf16ToFloat(gbits[k])), LmBf16ToFloat(ubits[k]), limit);
			*(uint4 *)(output_bf16 + out_base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
		float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
		output_bf16[out_base + index] = LmFloatToBf16(SparkMathClampedGlu(SparkMathSiluFast(gate), up, limit));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluRoundedMulLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, bool gate_first, float limit)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(gate_up_bf16) && LmIsAligned16(output_bf16) )
	{
		const uint64_t gate_base = base + (gate_first ? 0u : dimension);
		const uint64_t up_base = base + (gate_first ? dimension : 0u);
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t gbits[8], ubits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + gate_base + index), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + up_base + index), ubits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = SparkMathClampedGlu(SparkMathRoundBf16(SparkMathSilu(LmBf16ToFloat(gbits[k]))), LmBf16ToFloat(ubits[k]), limit);
			*(uint4 *)(output_bf16 + out_base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? index : dimension + index)]);
		float up = LmBf16ToFloat(gate_up_bf16[base + (gate_first ? dimension + index : index)]);
		output_bf16[out_base + index] = LmFloatToBf16(SparkMathClampedGlu(SparkMathRoundBf16(SparkMathSilu(gate)), up, limit));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSiluMulGateLimitKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, float limit)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(gate_up_bf16) && LmIsAligned16(output_bf16) )
	{
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t gbits[8], ubits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + base + index), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + base + dimension + index), ubits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				float gate = fminf(LmBf16ToFloat(gbits[k]), limit);
				float up = fminf(fmaxf(LmBf16ToFloat(ubits[k]), -limit), limit);
				out_f[k] = SparkMathSilu(gate) * up;
			}
			*(uint4 *)(output_bf16 + out_base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = fminf(LmBf16ToFloat(gate_up_bf16[base + index]), limit);
		float up = fminf(fmaxf(LmBf16ToFloat(gate_up_bf16[base + dimension + index]), -limit), limit);
		output_bf16[out_base + index] = LmFloatToBf16(SparkMathSilu(gate) * up);
	}
}

#define LM_GATE_SIGMOID 0u
#define LM_GATE_SOFTPLUS 1u
#define LM_GATE_SILU 2u

template<uint32_t THREADS, class Weight = uint16_t, bool ROUND_FIRST = false>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormKernel(const uint16_t *__restrict__ input_bf16, const Weight *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon, float head_multiply)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.y,head = blockIdx.x,index;
	uint64_t base;
	float total = 0.0f,scale,value,rounded;
	if ( row >= rows || head >= heads )
		return;
	base = ((uint64_t)row * heads + head) * head_dimension;
	if ( ((head_dimension & 7u) == 0u) && LmIsAligned16(input_bf16) && LmIsAligned16(output_bf16) &&
		(weight_bf16 == 0 || LmIsAligned16(weight_bf16)) )
	{
		if ( head_dimension <= THREADS * 8u )
		{
			const uint32_t offset = threadIdx.x * 8u;
			uint4 staged = make_uint4(0u, 0u, 0u, 0u);
			if ( offset < head_dimension )
			{
				uint16_t bits[8];
				staged = *(const uint4 *)(input_bf16 + base + offset);
				LmUnpackUint4Bf16(staged, bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					value = LmBf16ToFloat(bits[k]);
					total += value * value;
				}
			}
			total = LmBlockSum<THREADS>(total,reduction);
			scale = SparkMathRmsScale(total, (float)head_dimension, epsilon);
			if ( offset < head_dimension )
			{
				uint16_t bits[8];
				float w[8], out_f[8];
				LmUnpackUint4Bf16(staged, bits);
				if ( weight_bf16 != 0 )
					LmLoadWeight8(weight_bf16, offset, w);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					value = LmBf16ToFloat(bits[k]) * scale;
					if ( weight_bf16 != 0 )
						value = ROUND_FIRST ? SparkMathNormGainRounded(value, w[k]) : value * w[k];
					rounded = LmBf16ToFloat(LmFloatToBf16(value));
					out_f[k] = rounded * head_multiply;
				}
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		for (index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u)
		{
			uint16_t bits[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + index), bits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				value = LmBf16ToFloat(bits[k]);
				total += value * value;
			}
		}
		total = LmBlockSum<THREADS>(total,reduction);
		scale = SparkMathRmsScale(total, (float)head_dimension, epsilon);
		for (index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u)
		{
			uint16_t bits[8];
			float w[8], out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + index), bits);
			if ( weight_bf16 != 0 )
				LmLoadWeight8(weight_bf16, index, w);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				value = LmBf16ToFloat(bits[k]) * scale;
				if ( weight_bf16 != 0 )
					value = ROUND_FIRST ? SparkMathNormGainRounded(value, w[k]) : value * w[k];
				rounded = LmBf16ToFloat(LmFloatToBf16(value));
				out_f[k] = rounded * head_multiply;
			}
			*(uint4 *)(output_bf16 + base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)head_dimension, epsilon);
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]) * scale;
		if ( weight_bf16 != 0 )
			value = ROUND_FIRST ? SparkMathNormGainRounded(value, LmScalarToFloat(weight_bf16[index])) : value * LmScalarToFloat(weight_bf16[index]);
		rounded = LmBf16ToFloat(LmFloatToBf16(value));
		output_bf16[base + index] = LmFloatToBf16(rounded * head_multiply);
	}
}

template<uint32_t THREADS, uint32_t GATE>
static __device__ __forceinline__ void LmHeadRmsNormGateBody(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.y,head = blockIdx.x,index;
	uint64_t base;
	float total = 0.0f,scale,value,gate;
	if ( row >= rows || head >= heads )
		return;
	base = ((uint64_t)row * heads + head) * head_dimension;
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)head_dimension, epsilon);
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		value = LmBf16ToFloat(LmFloatToBf16(LmBf16ToFloat(input_bf16[base + index]) * scale));
		value = LmBf16ToFloat(LmFloatToBf16(value * LmBf16ToFloat(weight_bf16[index])));
		gate = LmBf16ToFloat(gate_bf16[base + index]);
		output_bf16[base + index] = LmFloatToBf16(value * (GATE == LM_GATE_SIGMOID ? SparkMathSigmoidFast(gate) : SparkMathSiluFast(gate)));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSigmoidGateOnceKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.y,head = blockIdx.x,index;
	uint64_t base;
	float total = 0.0f,scale,value;
	if ( row >= rows || head >= heads )
		return;
	base = ((uint64_t)row * heads + head) * head_dimension;
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)head_dimension, epsilon);
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
		output_bf16[base + index] = LmFloatToBf16(SparkMathRmsNormGateOnce(LmBf16ToFloat(input_bf16[base + index]), scale,
			LmBf16ToFloat(weight_bf16[index]), SparkMathSigmoidFast(LmBf16ToFloat(gate_bf16[base + index]))));
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSiluGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormGateBody<THREADS,LM_GATE_SILU>(input_bf16,gate_bf16,weight_bf16,output_bf16,rows,heads,head_dimension,epsilon);
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadRmsNormSigmoidGateKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ gate_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t rows, uint32_t heads, uint32_t head_dimension, float epsilon)
{
	LmHeadRmsNormGateBody<THREADS,LM_GATE_SIGMOID>(input_bf16,gate_bf16,weight_bf16,output_bf16,rows,heads,head_dimension,epsilon);
}

template<uint32_t THREADS, uint32_t ACTIVATION>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadGateBroadcastKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,index;
	uint64_t base,at;
	float gate;
	base = (((uint64_t)row * heads) + head) * head_dimension;
	gate = LmBf16ToFloat(gate_bf16[((uint64_t)row * heads) + head]);
	if ( ACTIVATION == LM_GATE_SOFTPLUS )
		gate = gate > 20.0f ? gate : log1pf(__expf(gate));
	if ( ACTIVATION == LM_GATE_SIGMOID )
		gate = SparkMathSigmoidFast(gate);
	for (index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		at = base + index;
		heads_bf16[at] = LmFloatToBf16(LmBf16ToFloat(heads_bf16[at]) * gate);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRmsNormSigmoidGateKernel(const uint16_t *__restrict__ input_bf16,const uint16_t *__restrict__ gate_bf16,const float *__restrict__ weight,uint16_t *__restrict__ output_bf16,uint32_t dimension,float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	float value,total = 0.0f,inverse;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(input_bf16) && LmIsAligned16(gate_bf16) && LmIsAligned16(weight) && LmIsAligned16(output_bf16) )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged[vec_slots];
			uint32_t slot;
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension )
				{
					uint16_t bits[8];
					staged[slot] = *(const uint4 *)(input_bf16 + base + offset);
					LmUnpackUint4Bf16(staged[slot], bits);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
					{
						value = LmBf16ToFloat(bits[k]);
						total += value * value;
					}
				}
				else
					staged[slot] = make_uint4(0u, 0u, 0u, 0u);
			}
			inverse = SparkMathRmsScale(LmBlockSum<THREADS>(total,reduction), (float)dimension, epsilon);
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				uint16_t bits[8], gbits[8];
				float w[8], out_f[8];
				LmUnpackUint4Bf16(staged[slot], bits);
				LmUnpackUint4Bf16(*(const uint4 *)(gate_bf16 + base + offset), gbits);
				LmLoadWeight8(weight, offset, w);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					value = LmBf16ToFloat(bits[k]) * inverse * w[k];
					value *= SparkMathSigmoidFast(LmBf16ToFloat(gbits[k]));
					out_f[k] = value;
				}
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t bits[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + index), bits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				value = LmBf16ToFloat(bits[k]);
				total += value * value;
			}
		}
		inverse = SparkMathRmsScale(LmBlockSum<THREADS>(total,reduction), (float)dimension, epsilon);
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t bits[8], gbits[8];
			float w[8], out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + index), bits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_bf16 + base + index), gbits);
			LmLoadWeight8(weight, index, w);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				value = LmBf16ToFloat(bits[k]) * inverse * w[k];
				value *= SparkMathSigmoidFast(LmBf16ToFloat(gbits[k]));
				out_f[k] = value;
			}
			*(uint4 *)(output_bf16 + base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]);
		total += value * value;
	}
	inverse = SparkMathRmsScale(LmBlockSum<THREADS>(total,reduction), (float)dimension, epsilon);
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		value = LmBf16ToFloat(input_bf16[base + index]) * inverse * weight[index];
		value *= SparkMathSigmoidFast(LmBf16ToFloat(gate_bf16[base + index]));
		output_bf16[base + index] = LmFloatToBf16(value);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmClampedUpGateKernel(const uint16_t *__restrict__ up_gate_bf16,uint16_t *__restrict__ output_bf16,uint32_t dimension,float limit)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u,out_base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	float gate,up;
	if ( ((dimension & 7u) == 0u) && ((((uintptr_t)up_gate_bf16) | ((uintptr_t)output_bf16)) & 15u) == 0u )
	{
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint4 uvec = *(const uint4 *)(up_gate_bf16 + base + index);
			uint4 gvec = *(const uint4 *)(up_gate_bf16 + base + dimension + index);
			uint32_t uwords[4] = {uvec.x, uvec.y, uvec.z, uvec.w};
			uint32_t gwords[4] = {gvec.x, gvec.y, gvec.z, gvec.w};
			uint32_t owords[4];
			#pragma unroll
			for (uint32_t w = 0u; w < 4u; ++w)
			{
				up = LmBf16ToFloat((uint16_t)uwords[w]);
				gate = LmBf16ToFloat((uint16_t)gwords[w]);
				gate = gate > limit ? limit : gate;
				up = up > limit ? limit : (up < -limit ? -limit : up);
				uint32_t lo = (uint32_t)LmFloatToBf16(SparkMathSiluFast(gate) * up);
				up = LmBf16ToFloat((uint16_t)(uwords[w] >> 16u));
				gate = LmBf16ToFloat((uint16_t)(gwords[w] >> 16u));
				gate = gate > limit ? limit : gate;
				up = up > limit ? limit : (up < -limit ? -limit : up);
				uint32_t hi = (uint32_t)LmFloatToBf16(SparkMathSiluFast(gate) * up);
				owords[w] = lo | (hi << 16u);
			}
			*(uint4 *)(output_bf16 + out_base + index) = make_uint4(owords[0], owords[1], owords[2], owords[3]);
		}
		return;
	}
	for (index=threadIdx.x; index<dimension; index+=THREADS)
	{
		up = LmBf16ToFloat(up_gate_bf16[base + index]);
		gate = LmBf16ToFloat(up_gate_bf16[base + dimension + index]);
		gate = gate > limit ? limit : gate;
		up = up > limit ? limit : (up < -limit ? -limit : up);
		output_bf16[out_base + index] = LmFloatToBf16(SparkMathSiluFast(gate) * up);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSituMulKernel(const uint16_t *__restrict__ gate_up_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint64_t output_pitch, float beta, float linear_beta, uint32_t row_groups)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension * 2u;
	uint64_t out_base = (uint64_t)(blockIdx.x / row_groups) * output_pitch + (uint64_t)(blockIdx.x % row_groups) * dimension;
	uint32_t index;
	if ( ((dimension & 7u) == 0u) && ((output_pitch & 7u) == 0u) && LmIsAligned16(gate_up_bf16) && LmIsAligned16(output_bf16) )
	{
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t gbits[8], ubits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + base + index), gbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_up_bf16 + base + dimension + index), ubits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = SparkMathSitu(LmBf16ToFloat(gbits[k]), LmBf16ToFloat(ubits[k]), beta, linear_beta);
			*(uint4 *)(output_bf16 + out_base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		const float gate = LmBf16ToFloat(gate_up_bf16[base + index]);
		const float up = LmBf16ToFloat(gate_up_bf16[base + dimension + index]);
		output_bf16[out_base + index] = LmFloatToBf16(SparkMathSitu(gate, up, beta, linear_beta));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16, float *__restrict__ scores, uint32_t width)
{
	uint64_t base = (uint64_t)blockIdx.x * width;
	uint32_t index;
	for (index = threadIdx.x; index < width; index += THREADS)
	{
		float value = LmBf16ToFloat(logits_bf16[base + index]);
		scores[base + index] = SparkMathSigmoidFast(value);
	}
}

template<uint32_t THREADS, bool DOUBLED = false>
__global__ __launch_bounds__(THREADS, 1)
void LmBf16SigmoidRowsKernel(const uint16_t *__restrict__ logits_bf16,float *__restrict__ scores,uint32_t width)
{
	uint64_t base = (uint64_t)blockIdx.x * width;
	uint32_t index;
	for (index=threadIdx.x; index<width; index+=THREADS)
	{
		float value = LmBf16ToFloat(logits_bf16[base + index]);
		if constexpr ( DOUBLED )
			scores[base + index] = 2.0f * LmBf16ToFloat(LmFloatToBf16(SparkMathSigmoidFast(value)));
		else
			scores[base + index] = LmBf16ToFloat(LmFloatToBf16(SparkMathSigmoidFast(value)));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadSigmoidGateKernel(uint16_t *__restrict__ heads_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension, uint32_t gate_head_stride, uint32_t gate_offset)
{
	const uint32_t row = blockIdx.x,head = blockIdx.y;
	const uint64_t base = (((uint64_t)row * heads) + head) * head_dimension;
	const uint64_t gate_base = ((((uint64_t)row * heads) + head) * gate_head_stride) + gate_offset;
	if ( ((head_dimension & 7u) == 0u) && ((gate_head_stride & 7u) == 0u) && ((gate_offset & 7u) == 0u) &&
		LmIsAligned16(heads_bf16) && LmIsAligned16(gate_bf16) )
	{
		for (uint32_t index = threadIdx.x * 8u; index < head_dimension; index += THREADS * 8u)
		{
			uint16_t hbits[8], gbits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(heads_bf16 + base + index), hbits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_bf16 + gate_base + index), gbits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = LmBf16ToFloat(hbits[k]) * SparkMathSigmoidFast(LmBf16ToFloat(gbits[k]));
			*(uint4 *)(heads_bf16 + base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (uint32_t index = threadIdx.x; index < head_dimension; index += THREADS)
	{
		const float gate = LmBf16ToFloat(gate_bf16[gate_base + index]);
		heads_bf16[base + index] = LmFloatToBf16(LmBf16ToFloat(heads_bf16[base + index]) * (SparkMathSigmoidFast(gate)));
	}
}

template<uint32_t THREADS, uint32_t RANK>
__global__ __launch_bounds__(THREADS)
void LmLowRankGateDownKernel(const uint16_t *__restrict__ rows_bf16, const uint16_t *__restrict__ down_bf16, float *__restrict__ bottleneck, uint32_t dimension)
{
	__shared__ float partial[THREADS / LM_WARP_LANES];
	const uint32_t rank = blockIdx.x, row = blockIdx.y;
	const uint64_t base = (uint64_t)row * dimension;
	const uint32_t warp = threadIdx.x / LM_WARP_LANES,lane = threadIdx.x % LM_WARP_LANES;
	float total = 0.0f;
	for (uint32_t index = threadIdx.x; index < dimension; index += THREADS)
		total += LmBf16ToFloat(rows_bf16[base + index]) * LmBf16ToFloat(down_bf16[((uint64_t)rank * dimension) + index]);
	for (uint32_t offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		total += __shfl_down_sync(0xffffffffu,total,offset);
	if ( lane == 0u )
		partial[warp] = total;
	__syncthreads();
	if ( threadIdx.x == 0u )
	{
		float sum = 0.0f;
		for (uint32_t index = 0u; index < THREADS / LM_WARP_LANES; ++index)
			sum += partial[index];
		sum = LmBf16ToFloat(LmFloatToBf16(sum));
		bottleneck[(uint64_t)row * RANK + rank] = LmBf16ToFloat(LmFloatToBf16(SparkMathSiluFast(sum)));
	}
}

template<uint32_t THREADS, uint32_t RANK>
__global__ __launch_bounds__(THREADS)
void LmLowRankGateApplyKernel(uint16_t *__restrict__ rows_bf16, const float *__restrict__ bottleneck, const uint16_t *__restrict__ up_bf16, uint32_t dimension)
{
	const uint32_t index = blockIdx.x * THREADS + threadIdx.x, row = blockIdx.y;
	const float *factors = bottleneck + (uint64_t)row * RANK;
	float gate = 0.0f;
	if ( index >= dimension )
		return;
	for (uint32_t rank = 0u; rank < RANK; ++rank)
		gate += factors[rank] * LmBf16ToFloat(up_bf16[((uint64_t)index * RANK) + rank]);
	gate = LmBf16ToFloat(LmFloatToBf16(gate));
	rows_bf16[(uint64_t)row * dimension + index] = LmFloatToBf16(LmBf16ToFloat(rows_bf16[(uint64_t)row * dimension + index]) * SparkMathSigmoidFast(gate));
}

template<uint32_t THREADS, bool ROUND_GATE = false>
__global__ __launch_bounds__(THREADS, 1)
void LmOutputGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t dimension)
{
	uint64_t base = (uint64_t)blockIdx.x * dimension;
	uint32_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(output_bf16) && LmIsAligned16(gate_bf16) )
	{
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			uint16_t obits[8], gbits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(output_bf16 + base + index), obits);
			LmUnpackUint4Bf16(*(const uint4 *)(gate_bf16 + base + index), gbits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
			{
				const float gate = LmBf16ToFloat(gbits[k]);
				const float value = LmBf16ToFloat(obits[k]);
				out_f[k] = value * (ROUND_GATE ? SparkMathRoundBf16(SparkMathSigmoid(gate)) : SparkMathSigmoidFast(gate));
			}
			*(uint4 *)(output_bf16 + base + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float gate = LmBf16ToFloat(gate_bf16[base + index]);
		float value = LmBf16ToFloat(output_bf16[base + index]);
		output_bf16[base + index] =
			LmFloatToBf16(value * (ROUND_GATE ? SparkMathRoundBf16(SparkMathSigmoid(gate)) : SparkMathSigmoidFast(gate)));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmHeadWiseGateKernel(uint16_t *__restrict__ output_bf16, const uint16_t *__restrict__ gate_bf16, uint32_t heads, uint32_t head_dimension)
{
	uint64_t base = (uint64_t)blockIdx.x * heads * head_dimension;
	uint32_t head, index;
	for (head = 0u; head < heads; ++head)
	{
		float gate = LmBf16ToFloat(gate_bf16[base + (uint64_t)head]);
		gate = SparkMathSigmoidFast(gate);
		for (index = threadIdx.x; index < head_dimension; index += THREADS)
		{
			uint64_t element = base + (uint64_t)head * head_dimension + index;
			output_bf16[element] =
				LmFloatToBf16(gate * LmBf16ToFloat(output_bf16[element]));
		}
	}
}

template<class Format, uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmQuantiseRowsKernel(const uint16_t *__restrict__ input_bf16, const uint32_t *__restrict__ source_row_map, uint8_t *__restrict__ output_codes, uint8_t *__restrict__ output_scales, uint32_t row_count, uint32_t dimension)
{
	static_assert(Format::kScaleGroup > 0u,
		"an unquantised format has no scale groups and must not reach a "
		"quantise; the caller should skip it, not divide by zero sizing the grid");
	extern __shared__ float lm_quant_shared[];
	float *row = lm_quant_shared;
	float *reduction = lm_quant_shared + Format::kScaleGroup;
	const uint32_t groups = dimension / Format::kScaleGroup;
	uint32_t destination = blockIdx.x,group = blockIdx.y,index;
	uint64_t source;
	float absmax = 0.0f,inverse;
	uint8_t scale_code;
	if ( destination >= row_count || group >= groups )
		return;
	source = (uint64_t)(source_row_map != 0 ? source_row_map[destination] : destination)
		* (uint64_t)dimension + (group * Format::kScaleGroup);
	for (index = threadIdx.x; index < Format::kScaleGroup; index += THREADS)
	{
		row[index] = LmBf16ToFloat(input_bf16[source + index]);
		absmax = fmaxf(absmax,fabsf(row[index]));
	}
	absmax = LmBlockMax<THREADS>(absmax,reduction);
	scale_code = LmFloatToUe4m3(absmax / Format::kMax);
	if ( scale_code == 0u )
		scale_code = 1u;
	inverse = 1.0f / LmUe4m3ToFloat(scale_code);
	if ( threadIdx.x == 0u )
		output_scales[((uint64_t)destination * groups) + group] = scale_code;
	__syncthreads();
	static_assert(
		(Format::kScaleGroup % 8u) == 0u,
		"quantization groups must contain complete eight-code blocks");
	for (index = threadIdx.x * 8u;
		index < Format::kScaleGroup;
		index += THREADS * 8u)
	{
		float values[8];
		uint32_t value_index;
		uint64_t bit =
			(((uint64_t)destination * dimension) +
			 (group * Format::kScaleGroup) + index) *
			Format::kStoredBits;

		for (value_index = 0u; value_index < 8u; ++value_index)
		{
			values[value_index] = row[index + value_index] * inverse;
		}
		LmStoreCodeOctet<Format>(output_codes, bit, values);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	uint32_t token;
	uint32_t element = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t route;
	LmDependentRelease();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(packed_bf16) && LmIsAligned16(output_bf16) )
	{
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= dimension )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t vec_element = (blockIdx.x * THREADS) + (threadIdx.x << 3u);
				if ( vec_element >= dimension )
					return;
				for (token = blockIdx.y; token < tokens; token += gridDim.y)
				{
					float total[8];
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
						total[k] = 0.0f;
					for (route = 0u; route < top_k; ++route)
					{
						const uint32_t packed_row = packed_row_of_token_route[((uint64_t)token * top_k) + route];
						const float weight = route_weight[((uint64_t)token * top_k) + route];
						uint16_t bits[8];
						LmUnpackUint4Bf16(*(const uint4 *)(packed_bf16 + ((uint64_t)packed_row * dimension) + vec_element), bits);
						#pragma unroll
						for (uint32_t k = 0u; k < 8u; ++k)
							total[k] += LmBf16ToFloat(bits[k]) * weight;
					}
					*(uint4 *)(output_bf16 + ((uint64_t)token * dimension) + vec_element) = LmPackUint4FloatToBf16(total);
				}
				return;
			}
		}
		const uint32_t vec_element = element * 8u;
		if ( vec_element >= dimension )
			return;
		for (token = blockIdx.y; token < tokens; token += gridDim.y)
		{
			float total[8];
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				total[k] = 0.0f;
			for (route = 0u; route < top_k; ++route)
			{
				const uint32_t packed_row = packed_row_of_token_route[((uint64_t)token * top_k) + route];
				const float weight = route_weight[((uint64_t)token * top_k) + route];
				uint16_t bits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(packed_bf16 + ((uint64_t)packed_row * dimension) + vec_element), bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					total[k] += LmBf16ToFloat(bits[k]) * weight;
			}
			*(uint4 *)(output_bf16 + ((uint64_t)token * dimension) + vec_element) = LmPackUint4FloatToBf16(total);
		}
		return;
	}
	if ( element >= dimension )
		return;
	for (token = blockIdx.y; token < tokens; token += gridDim.y)
	{
		float total = 0.0f;
		for (route = 0u; route < top_k; ++route)
		{
			uint32_t packed_row = packed_row_of_token_route[((uint64_t)token * top_k) + route];
			total += LmBf16ToFloat(packed_bf16[((uint64_t)packed_row * dimension) + element])
				* route_weight[((uint64_t)token * top_k) + route];
		}
		output_bf16[((uint64_t)token * dimension) + element] = LmFloatToBf16(total);
	}
}

#define LM_MOE_FINALIZE_ADD_ELEMENTS 8u

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmMoeFinalizeAddKernel(const uint16_t *__restrict__ packed_bf16, const uint32_t *__restrict__ packed_row_of_token_route, const float *__restrict__ route_weight,
	const uint16_t *__restrict__ addend_bf16, uint16_t *__restrict__ output_bf16, uint32_t tokens, uint32_t top_k, uint32_t dimension)
{
	const uint32_t element = ((blockIdx.x * THREADS) + threadIdx.x) * LM_MOE_FINALIZE_ADD_ELEMENTS;
	uint32_t token, route, index, words[4];
	if ( element >= dimension )
		return;
	for (token = blockIdx.y; token < tokens; token += gridDim.y)
	{
		float total[LM_MOE_FINALIZE_ADD_ELEMENTS];
		uint4 value;
		for (index = 0u; index < LM_MOE_FINALIZE_ADD_ELEMENTS; ++index)
			total[index] = 0.0f;
		for (route = 0u; route < top_k; ++route)
		{
			const uint32_t packed_row = packed_row_of_token_route[((uint64_t)token * top_k) + route];
			const float weight = route_weight[((uint64_t)token * top_k) + route];
			value = *(const uint4 *)(packed_bf16 + ((uint64_t)packed_row * dimension) + element);
			words[0] = value.x; words[1] = value.y; words[2] = value.z; words[3] = value.w;
			for (index = 0u; index < LM_MOE_FINALIZE_ADD_ELEMENTS; ++index)
				total[index] += LmBf16ToFloat((uint16_t)(words[index / 2u] >> (16u * (index % 2u)))) * weight;
		}
		value = *(const uint4 *)(addend_bf16 + ((uint64_t)token * dimension) + element);
		words[0] = value.x; words[1] = value.y; words[2] = value.z; words[3] = value.w;
		for (index = 0u; index < LM_MOE_FINALIZE_ADD_ELEMENTS; index += 2u)
		{
			const float low = LmBf16ToFloat(LmFloatToBf16(total[index])) + LmBf16ToFloat((uint16_t)words[index / 2u]);
			const float high = LmBf16ToFloat(LmFloatToBf16(total[index + 1u])) + LmBf16ToFloat((uint16_t)(words[index / 2u] >> 16u));
			words[index / 2u] = (uint32_t)LmFloatToBf16(low) | ((uint32_t)LmFloatToBf16(high) << 16u);
		}
		*(uint4 *)(output_bf16 + ((uint64_t)token * dimension) + element) = make_uint4(words[0], words[1], words[2], words[3]);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsKernel(const uint16_t *__restrict__ a_bf16, const uint16_t *__restrict__ b_bf16, uint16_t *__restrict__ out_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(a_bf16) && LmIsAligned16(b_bf16) && LmIsAligned16(out_bf16) )
	{
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= dimension )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t vec_element = (blockIdx.x * THREADS) + (threadIdx.x << 3u);
				if ( vec_element >= dimension )
					return;
				for (row = blockIdx.y; row < rows; row += gridDim.y)
				{
					index = ((uint64_t)row * dimension) + vec_element;
					uint16_t abits[8], bbits[8];
					float out_f[8];
					LmUnpackUint4Bf16(*(const uint4 *)(a_bf16 + index), abits);
					LmUnpackUint4Bf16(*(const uint4 *)(b_bf16 + index), bbits);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
						out_f[k] = LmBf16ToFloat(abits[k]) + LmBf16ToFloat(bbits[k]);
					*(uint4 *)(out_bf16 + index) = LmPackUint4FloatToBf16(out_f);
				}
				return;
			}
		}
		const uint32_t vec_element = element * 8u;
		if ( vec_element >= dimension )
			return;
		for (row = blockIdx.y; row < rows; row += gridDim.y)
		{
			index = ((uint64_t)row * dimension) + vec_element;
			uint16_t abits[8], bbits[8];
			float out_f[8];
			LmUnpackUint4Bf16(*(const uint4 *)(a_bf16 + index), abits);
			LmUnpackUint4Bf16(*(const uint4 *)(b_bf16 + index), bbits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				out_f[k] = LmBf16ToFloat(abits[k]) + LmBf16ToFloat(bbits[k]);
			*(uint4 *)(out_bf16 + index) = LmPackUint4FloatToBf16(out_f);
		}
		return;
	}
	if ( element >= dimension )
		return;
	for (row = blockIdx.y; row < rows; row += gridDim.y)
	{
		index = ((uint64_t)row * dimension) + element;
		out_bf16[index] = LmFloatToBf16(LmBf16ToFloat(a_bf16[index]) + LmBf16ToFloat(b_bf16[index]));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCopyRowsKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t index;
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(source_bf16) && LmIsAligned16(destination_bf16) )
	{
		if constexpr ( (THREADS & 7u) == 0u )
		{
			if ( (uint64_t)gridDim.x * THREADS >= dimension )
			{
				if ( threadIdx.x >= (THREADS >> 3u) )
					return;
				const uint32_t vec_element = (blockIdx.x * THREADS) + (threadIdx.x << 3u);
				if ( vec_element >= dimension )
					return;
				for (row = blockIdx.y; row < rows; row += gridDim.y)
				{
					index = ((uint64_t)row * dimension) + vec_element;
					*(uint4 *)(destination_bf16 + index) = *(const uint4 *)(source_bf16 + index);
				}
				return;
			}
		}
		const uint32_t vec_element = element * 8u;
		if ( vec_element >= dimension )
			return;
		for (row = blockIdx.y; row < rows; row += gridDim.y)
		{
			index = ((uint64_t)row * dimension) + vec_element;
			*(uint4 *)(destination_bf16 + index) = *(const uint4 *)(source_bf16 + index);
		}
		return;
	}
	if ( element >= dimension )
		return;
	for (row = blockIdx.y; row < rows; row += gridDim.y)
	{
		index = ((uint64_t)row * dimension) + element;
		destination_bf16[index] = source_bf16[index];
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmRankMajorRowsKernel(const uint16_t *__restrict__ gathered_bf16, uint16_t *__restrict__ rows_bf16, uint32_t rows, uint32_t width, uint32_t ranks)
{
	uint32_t row,index;
	for (row = blockIdx.y; row < rows; row += gridDim.y)
		for (index = (blockIdx.x * THREADS) + threadIdx.x; index < width * ranks; index += gridDim.x * THREADS)
			rows_bf16[((uint64_t)row * width * ranks) + index] = gathered_bf16[((((uint64_t)(index / width) * rows) + row) * width) + (index % width)];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmSliceColumnsKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ target_bf16, uint32_t source_width, uint32_t offset, uint32_t width)
{
	const uint32_t row = blockIdx.y;
	uint32_t index;
	for (index = (blockIdx.x * THREADS) + threadIdx.x; index < width; index += gridDim.x * THREADS)
		target_bf16[((uint64_t)row * width) + index] = source_bf16[((uint64_t)row * source_width) + offset + index];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCopyRowsStridedKernel(const uint16_t *__restrict__ source_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension,
	uint64_t destination_stride)
{
	uint32_t row,element = (blockIdx.x * THREADS) + threadIdx.x;
	if ( element >= dimension )
		return;
	for (row = blockIdx.y; row < rows; row += gridDim.y)
		destination_bf16[((uint64_t)row * destination_stride) + element] = source_bf16[((uint64_t)row * dimension) + element];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmAddRowsStridedKernel(const uint16_t *__restrict__ source_bf16, const uint16_t *__restrict__ addend_bf16, uint16_t *__restrict__ destination_bf16,
	uint32_t rows, uint32_t dimension, uint64_t destination_stride)
{
	uint32_t row,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint64_t source;
	if ( element >= dimension )
		return;
	for (row = blockIdx.y; row < rows; row += gridDim.y)
	{
		source = ((uint64_t)row * dimension) + element;
		destination_bf16[((uint64_t)row * destination_stride) + element] = LmFloatToBf16(LmBf16ToFloat(source_bf16[source]) + LmBf16ToFloat(addend_bf16[source]));
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmStreamMeanRowsKernel(const uint16_t *__restrict__ streams_bf16, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t streams,
	uint32_t dimension, uint64_t destination_stride)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x,stream;
	const uint16_t *source;
	float total = 0.0f;
	if ( row >= rows || element >= dimension )
		return;
	source = streams_bf16 + ((uint64_t)row * streams * dimension) + element;
	for (stream = 0u; stream < streams; ++stream)
		total += LmBf16ToFloat(source[(uint64_t)stream * dimension]);
	destination_bf16[((uint64_t)row * destination_stride) + element] = LmFloatToBf16(total / (float)streams);
}

template<uint32_t THREADS, uint32_t MAX_SOURCES>
__global__ __launch_bounds__(THREADS, 1)
void LmAttnResKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const uint16_t *__restrict__ score_weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t sources, uint32_t rows, uint32_t dimension, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float score[MAX_SOURCES];
	__shared__ float weight[MAX_SOURCES];
	constexpr uint32_t softmax_lanes = THREADS < LM_WARP_LANES ? 1u : LM_WARP_LANES;
	uint32_t row = blockIdx.x,source,index;
	const bool vec_ok = ((dimension & 7u) == 0u) && LmIsAligned16(bank_bf16) && LmIsAligned16(partial_bf16) && LmIsAligned16(output_bf16);
	const bool score_vec_ok = vec_ok && LmIsAligned16(score_weight_bf16);
	for (source = 0u; source < sources; ++source)
	{
		const uint16_t *values = source + 1u == sources
			? partial_bf16 + ((uint64_t)row * dimension)
			: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
		float square = 0.0f,dot = 0.0f,inverse;
		if ( score_vec_ok )
		{
			for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
			{
				uint16_t bits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(values + index), bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					const float value = LmBf16ToFloat(bits[k]);
					square += value * value;
				}
			}
			square = LmBlockSum<THREADS>(square,reduction);
			inverse = SparkMathRmsScale(square, (float)dimension, epsilon);
			for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
			{
				uint16_t bits[8], wbits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(values + index), bits);
				LmUnpackUint4Bf16(*(const uint4 *)(score_weight_bf16 + index), wbits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					dot += LmBf16ToFloat(bits[k]) * inverse * LmBf16ToFloat(wbits[k]);
			}
		}
		else
		{
			for (index = threadIdx.x; index < dimension; index += THREADS)
			{
				float value = LmBf16ToFloat(values[index]);
				square += value * value;
			}
			square = LmBlockSum<THREADS>(square,reduction);
			inverse = SparkMathRmsScale(square, (float)dimension, epsilon);
			for (index = threadIdx.x; index < dimension; index += THREADS)
				dot += LmBf16ToFloat(values[index]) * inverse
					* LmBf16ToFloat(score_weight_bf16[index]);
		}
		dot = LmBlockSum<THREADS>(dot,reduction);
		if ( threadIdx.x == 0u )
			score[source] = dot;
		__syncthreads();
	}
	if ( threadIdx.x < softmax_lanes )
	{
		float local_max = -INFINITY;
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
			local_max = fmaxf(local_max,score[source]);
		#pragma unroll
		for (uint32_t offset = softmax_lanes / 2u; offset > 0u; offset >>= 1u)
			local_max = fmaxf(local_max,__shfl_xor_sync(0xffffffffu,local_max,offset));
		float local_sum = 0.0f;
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
		{
			const float w = __expf(score[source] - local_max);
			weight[source] = w;
			local_sum += w;
		}
		#pragma unroll
		for (uint32_t offset = softmax_lanes / 2u; offset > 0u; offset >>= 1u)
			local_sum += __shfl_xor_sync(0xffffffffu,local_sum,offset);
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
			weight[source] /= local_sum;
	}
	__syncthreads();
	if ( vec_ok )
	{
		for (index = threadIdx.x * 8u; index < dimension; index += THREADS * 8u)
		{
			float total[8];
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				total[k] = 0.0f;
			for (source = 0u; source < sources; ++source)
			{
				const uint16_t *values = source + 1u == sources
					? partial_bf16 + ((uint64_t)row * dimension)
					: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
				const float w = weight[source];
				uint16_t bits[8];
				LmUnpackUint4Bf16(*(const uint4 *)(values + index), bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					total[k] += w * LmBf16ToFloat(bits[k]);
			}
			*(uint4 *)(output_bf16 + ((uint64_t)row * dimension) + index) = LmPackUint4FloatToBf16(total);
		}
		return;
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float total = 0.0f;
		for (source = 0u; source < sources; ++source)
		{
			const uint16_t *values = source + 1u == sources
				? partial_bf16 + ((uint64_t)row * dimension)
				: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
			total += weight[source] * LmBf16ToFloat(values[index]);
		}
		output_bf16[((uint64_t)row * dimension) + index] = LmFloatToBf16(total);
	}
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS) void LmAttnResScoreKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const uint16_t *__restrict__ score_weight_bf16, float *__restrict__ score_f32, uint32_t sources, uint32_t rows, uint32_t dimension, uint32_t score_stride, float epsilon)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	const uint32_t row = blockIdx.x, source = blockIdx.y;
	const uint16_t *values = source + 1u == sources
		? partial_bf16 + ((uint64_t)row * dimension)
		: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
	float square = 0.0f,dot = 0.0f,inverse;
	uint32_t index;
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(values[index]);
		square += value * value;
	}
	square = LmBlockSum<THREADS>(square,reduction);
	inverse = SparkMathRmsScale(square, (float)dimension, epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		dot += LmBf16ToFloat(values[index]) * inverse
			* LmBf16ToFloat(score_weight_bf16[index]);
	dot = LmBlockSum<THREADS>(dot,reduction);
	if ( threadIdx.x == 0u )
		score_f32[((uint64_t)row * score_stride) + source] = dot;
}

template<uint32_t THREADS, uint32_t MAX_SOURCES>
__global__ __launch_bounds__(THREADS) void LmAttnResMixKernel(const uint16_t *__restrict__ bank_bf16, const uint16_t *__restrict__ partial_bf16, const float *__restrict__ score_f32, uint16_t *__restrict__ output_bf16, uint32_t sources, uint32_t rows, uint32_t dimension, uint32_t score_stride)
{
	__shared__ float weight[MAX_SOURCES];
	constexpr uint32_t softmax_lanes = THREADS < LM_WARP_LANES ? 1u : LM_WARP_LANES;
	const uint32_t row = blockIdx.y, index = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t source;
	if ( threadIdx.x < softmax_lanes )
	{
		const float *score = score_f32 + ((uint64_t)row * score_stride);
		float local_max = -INFINITY;
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
			local_max = fmaxf(local_max,score[source]);
		#pragma unroll
		for (uint32_t offset = softmax_lanes / 2u; offset > 0u; offset >>= 1u)
			local_max = fmaxf(local_max,__shfl_xor_sync(0xffffffffu,local_max,offset));
		float local_sum = 0.0f;
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
		{
			const float w = __expf(score[source] - local_max);
			weight[source] = w;
			local_sum += w;
		}
		#pragma unroll
		for (uint32_t offset = softmax_lanes / 2u; offset > 0u; offset >>= 1u)
			local_sum += __shfl_xor_sync(0xffffffffu,local_sum,offset);
		for (source = threadIdx.x; source < sources; source += softmax_lanes)
			weight[source] /= local_sum;
	}
	__syncthreads();
	if ( ((dimension & 7u) == 0u) && LmIsAligned16(bank_bf16) && LmIsAligned16(partial_bf16) && LmIsAligned16(output_bf16) )
	{
		const uint32_t vec_index = index * 8u;
		if ( vec_index >= dimension )
			return;
		float total[8];
		#pragma unroll
		for (uint32_t k = 0u; k < 8u; ++k)
			total[k] = 0.0f;
		for (source = 0u; source < sources; ++source)
		{
			const uint16_t *values = source + 1u == sources
				? partial_bf16 + ((uint64_t)row * dimension)
				: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
			const float w = weight[source];
			uint16_t bits[8];
			LmUnpackUint4Bf16(*(const uint4 *)(values + vec_index), bits);
			#pragma unroll
			for (uint32_t k = 0u; k < 8u; ++k)
				total[k] += w * LmBf16ToFloat(bits[k]);
		}
		*(uint4 *)(output_bf16 + ((uint64_t)row * dimension) + vec_index) = LmPackUint4FloatToBf16(total);
		return;
	}
	if ( index >= dimension )
		return;
	float total = 0.0f;
	for (source = 0u; source < sources; ++source)
	{
		const uint16_t *values = source + 1u == sources
			? partial_bf16 + ((uint64_t)row * dimension)
			: bank_bf16 + ((((uint64_t)source * rows) + row) * dimension);
		total += weight[source] * LmBf16ToFloat(values[index]);
	}
	output_bf16[((uint64_t)row * dimension) + index] = LmFloatToBf16(total);
}

template<uint32_t THREADS>
__global__ void LmGatherRowsKernel(const uint16_t *__restrict__ source_bf16, const uint32_t *__restrict__ source_row_map, uint16_t *__restrict__ destination_bf16, uint32_t rows, uint32_t dimension)
{
	uint32_t row = blockIdx.y,element = (blockIdx.x * THREADS) + threadIdx.x;
	uint32_t source_row;
	uint64_t from,to;
	if ( row >= rows || element >= dimension )
		return;
	source_row = source_row_map != 0 ? source_row_map[row] : row;
	from = ((uint64_t)source_row * dimension) + element;
	to = ((uint64_t)row * dimension) + element;
	destination_bf16[to] = source_bf16[from];
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmCenteredRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ output_bf16, uint32_t dimension, uint32_t row_stride, float epsilon)
{
	extern __shared__ float lm_norm_shared[];
	float *row = lm_norm_shared;
	float *reduction = lm_norm_shared + dimension;
	uint64_t base = (uint64_t)blockIdx.x * row_stride;
	uint32_t index;
	float total = 0.0f,scale;
	const bool vec_ok = ((dimension & 7u) == 0u) && ((row_stride & 7u) == 0u) &&
		LmIsAligned16(input_bf16) && LmIsAligned16(weight_bf16) && LmIsAligned16(output_bf16);
	if ( vec_ok )
	{
		if ( dimension <= THREADS * LM_RMS_NORM_STAGED )
		{
			constexpr uint32_t vec_slots = LM_RMS_NORM_STAGED / 8u;
			uint4 staged[vec_slots];
			uint32_t slot;
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset < dimension )
				{
					uint16_t bits[8];
					staged[slot] = *(const uint4 *)(input_bf16 + base + offset);
					LmUnpackUint4Bf16(staged[slot], bits);
					#pragma unroll
					for (uint32_t k = 0u; k < 8u; ++k)
					{
						const float value = LmBf16ToFloat(bits[k]);
						total += value * value;
					}
				}
				else
					staged[slot] = make_uint4(0u, 0u, 0u, 0u);
			}
			total = LmBlockSum<THREADS>(total,reduction);
			scale = SparkMathRmsScale(total, (float)dimension, epsilon);
			#pragma unroll
			for (slot = 0u; slot < vec_slots; ++slot)
			{
				const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
				if ( offset >= dimension )
					continue;
				uint16_t bits[8], wbits[8];
				float out_f[8];
				LmUnpackUint4Bf16(staged[slot], bits);
				LmUnpackUint4Bf16(*(const uint4 *)(weight_bf16 + offset), wbits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					out_f[k] = LmBf16ToFloat(bits[k]) * scale * (1.0f + LmBf16ToFloat(wbits[k]));
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
		if ( LmIsAligned16(row) )
		{
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				uint16_t bits[8];
				float values[8];
				LmUnpackUint4Bf16(*(const uint4 *)(input_bf16 + base + offset), bits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
				{
					const float value = LmBf16ToFloat(bits[k]);
					values[k] = value;
					total += value * value;
				}
				LmStoreFloat8(row + offset, values);
			}
			total = LmBlockSum<THREADS>(total,reduction);
			scale = SparkMathRmsScale(total, (float)dimension, epsilon);
			for (uint32_t offset = threadIdx.x * 8u; offset < dimension; offset += THREADS * 8u)
			{
				uint16_t wbits[8];
				float values[8], out_f[8];
				LmLoadFloat8(row + offset, values);
				LmUnpackUint4Bf16(*(const uint4 *)(weight_bf16 + offset), wbits);
				#pragma unroll
				for (uint32_t k = 0u; k < 8u; ++k)
					out_f[k] = values[k] * scale * (1.0f + LmBf16ToFloat(wbits[k]));
				*(uint4 *)(output_bf16 + base + offset) = LmPackUint4FloatToBf16(out_f);
			}
			return;
		}
	}
	for (index = threadIdx.x; index < dimension; index += THREADS)
	{
		float value = LmBf16ToFloat(input_bf16[base + index]);
		row[index] = value;
		total += value * value;
	}
	total = LmBlockSum<THREADS>(total,reduction);
	scale = SparkMathRmsScale(total, (float)dimension, epsilon);
	for (index = threadIdx.x; index < dimension; index += THREADS)
		output_bf16[base + index] =
			LmFloatToBf16(row[index] * scale * (1.0f + LmBf16ToFloat(weight_bf16[index])));
}
