#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/mma.cuh"
#include "../../include/sparkpipe/spark_weight_codec.h"
#include <math.h>

static __device__ __forceinline__ float LmActivationWarpMax(float value)
{
#pragma unroll
	for (uint32_t offset=16u; offset!=0u; offset>>=1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	return(value);
}

static __device__ __forceinline__ float LmActivationFp8Scale(float amax, float floor = 1e-4f)
{
	const float ratio = fmaxf(amax,floor) / LM_E4M3_MAX;
	const uint32_t bits = __float_as_uint(ratio);
	if ( bits >= 0x00800000u && bits < 0x7f000000u )
		return(__uint_as_float((bits + 0x007fffffu) & 0xff800000u));
	return(exp2f(ceilf(log2f(ratio))));
}

template<uint32_t ACTIVATION_CODEC>
static __device__ __forceinline__ float LmActivationFp8Qdq(float value, float scale)
{
	static_assert(ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,"unsupported FP8 activation codec");
	return(LmE4m3ToFloat(LmFloatToE4m3(value / scale)) * scale);
}

template<uint32_t ACTIVATION_CODEC>
static __device__ __forceinline__ float LmActivationFp8QdqInv(float value, float scale, float inv_scale)
{
	static_assert(ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,"unsupported FP8 activation codec");
	return(LmE4m3ToFloat(LmFloatToE4m3(value * inv_scale)) * scale);
}

template<uint32_t ACTIVATION_CODEC>
static __device__ void LmActivationFp8QdqFloatRow(float *row, uint32_t width)
{
	uint32_t warp = threadIdx.x / 32u,lane = threadIdx.x % 32u,base,element;
	float amax,scale,inv_scale;
	for (base=warp * 128u; base<width; base+=(blockDim.x / 32u) * 128u)
	{
		float cached[4];
		amax = 0.0f;
#pragma unroll
		for (uint32_t i=0u; i<4u; ++i)
		{
			element = base + lane + i * 32u;
			cached[i] = row[element];
			amax = fmaxf(amax,fabsf(cached[i]));
		}
		scale = LmActivationFp8Scale(__shfl_sync(0xffffffffu,LmActivationWarpMax(amax),0));
		inv_scale = 1.0f / scale;
#pragma unroll
		for (uint32_t i=0u; i<4u; ++i)
		{
			element = base + lane + i * 32u;
			row[element] = LmActivationFp8QdqInv<ACTIVATION_CODEC>(cached[i],scale,inv_scale);
		}
	}
}

template<uint32_t TILE_ROWS,uint32_t TILE_K,bool SWIZZLED,uint32_t ACTIVATION_CODEC>
static __device__ void LmActivationStageFp8Qdq(
	const void *source_bf16,
	const uint32_t *source_row_map,
	uint32_t source_row_count,
	uint32_t row_base,
	uint32_t row_limit,
	uint32_t k_base,
	uint32_t input_dimension,
	void *tile_bf16,
	uint32_t producer_warp_base,
	uint32_t producer_warp_count,
	LmFrameError *frame_error = 0)
{
	static_assert(128u % TILE_K == 0u,"FP8 activation groups must contain whole K tiles");
	static_assert(ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,"unsupported FP8 activation codec");
	const uint16_t *source = (const uint16_t *)source_bf16;
	uint16_t *tile = (uint16_t *)tile_bf16;
	uint32_t warp = threadIdx.x / 32u,lane = threadIdx.x % 32u,local_row,packed_row,source_row,element,offset;
	float amax,scale,inv_scale,value;
	if ( warp < producer_warp_base || warp >= producer_warp_base + producer_warp_count )
		return;
	const uint32_t group_base = (k_base / 128u) * 128u;
	for (local_row=warp - producer_warp_base; local_row<TILE_ROWS; local_row+=producer_warp_count)
	{
		packed_row = row_base + local_row;
		if ( packed_row >= row_limit )
			continue;
		source_row = source_row_map != 0 ? __ldg(source_row_map + packed_row) : packed_row;
		if ( source_row >= source_row_count )
		{
			LmFrameErrorReport(frame_error,
				(uint32_t)LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
				0u,packed_row,source_row,row_base,source_row_count);
			continue;
		}
		const uint16_t *src_row = source + (uint64_t)source_row * input_dimension;
		float cached[4];
		amax = 0.0f;
#pragma unroll
		for (uint32_t i=0u; i<4u; ++i)
		{
			cached[i] = LmBf16ToFloat(__ldg(src_row + group_base + lane + i * 32u));
			amax = fmaxf(amax,fabsf(cached[i]));
		}
		scale = LmActivationFp8Scale(__shfl_sync(0xffffffffu,LmActivationWarpMax(amax),0));
		inv_scale = 1.0f / scale;
		if constexpr ( TILE_K == 128u )
		{
#pragma unroll
			for (uint32_t i=0u; i<4u; ++i)
			{
				element = lane + i * 32u;
				offset = SWIZZLED ? LmSwizzledOffset(local_row,element * 2u,TILE_K * 2u,LmSwizzleSpanFor(TILE_K * 2u)) / 2u : local_row * TILE_K + element;
				tile[offset] = LmFloatToBf16(LmActivationFp8QdqInv<ACTIVATION_CODEC>(cached[i],scale,inv_scale));
			}
		}
		else
		{
			const uint32_t tile_sub = (k_base - group_base) / 32u;
#pragma unroll
			for (uint32_t i=0u; i<TILE_K / 32u; ++i)
			{
				const uint32_t idx = tile_sub + i;
				element = lane + i * 32u;
				value = idx == 0u ? cached[0] : (idx == 1u ? cached[1] : (idx == 2u ? cached[2] : cached[3]));
				offset = SWIZZLED ? LmSwizzledOffset(local_row,element * 2u,TILE_K * 2u,LmSwizzleSpanFor(TILE_K * 2u)) / 2u : local_row * TILE_K + element;
				tile[offset] = LmFloatToBf16(LmActivationFp8QdqInv<ACTIVATION_CODEC>(value,scale,inv_scale));
			}
		}
	}
}
