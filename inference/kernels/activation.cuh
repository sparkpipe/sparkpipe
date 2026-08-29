#pragma once

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/frame_error.cuh"
#include "inference/kernels/mma.cuh"
#include "../../include/sparkpipe/spark_weight_codec.h"
#include <math.h>

static __device__ __forceinline__ float LmActivationWarpMax(float value)
{
	for (uint32_t offset=16u; offset!=0u; offset>>=1u)
		value = fmaxf(value,__shfl_down_sync(0xffffffffu,value,offset));
	return(value);
}

static __device__ __forceinline__ float LmActivationFp8Scale(float amax)
{
	return(exp2f(ceilf(log2f(fmaxf(amax,1e-4f) / LM_E4M3_MAX))));
}

template<uint32_t ACTIVATION_CODEC>
static __device__ __forceinline__ float LmActivationFp8Qdq(float value, float scale)
{
	static_assert(ACTIVATION_CODEC == SPARK_ACTIVATION_CODEC_FP8_E4M3_UE8M0,"unsupported FP8 activation codec");
	return(LmE4m3ToFloat(LmFloatToE4m3(value / scale)) * scale);
}

template<uint32_t ACTIVATION_CODEC>
static __device__ void LmActivationFp8QdqFloatRow(float *row, uint32_t width)
{
	uint32_t warp = threadIdx.x / 32u,lane = threadIdx.x % 32u,base,element;
	float amax,scale;
	for (base=warp * 128u; base<width; base+=(blockDim.x / 32u) * 128u)
	{
		amax = 0.0f;
		for (element=base + lane; element<base + 128u; element+=32u)
			amax = fmaxf(amax,fabsf(row[element]));
		scale = LmActivationFp8Scale(__shfl_sync(0xffffffffu,LmActivationWarpMax(amax),0));
		for (element=base + lane; element<base + 128u; element+=32u)
			row[element] = LmActivationFp8Qdq<ACTIVATION_CODEC>(row[element],scale);
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
	float amax,scale,value;
	if ( warp < producer_warp_base || warp >= producer_warp_base + producer_warp_count )
		return;
	for (local_row=warp - producer_warp_base; local_row<TILE_ROWS; local_row+=producer_warp_count)
	{
		packed_row = row_base + local_row;
		if ( packed_row >= row_limit )
			continue;
		source_row = source_row_map != 0 ? source_row_map[packed_row] : packed_row;
		if ( source_row >= source_row_count )
		{
			// ROUTE-MAP CORRUPTION IS A FRAME FAILURE, NOT A CONTEXT
			// FAILURE (frame_error.cuh): record the first bad row and skip
			// this row's staging. These are plain shared stores with no
			// mbarrier accounting, so a skip cannot hang the tile; the row's
			// output is dead regardless and the driver fails the frame when
			// it reads the slot. The context lives.
			LmFrameErrorReport(frame_error,
				(uint32_t)LM_FRAME_ERROR_ROUTE_MAP_OUT_OF_RANGE,
				0u,packed_row,source_row,row_base,source_row_count);
			continue;
		}
		amax = 0.0f;
		for (element=lane; element<128u; element+=32u)
			amax = fmaxf(amax,fabsf(LmBf16ToFloat(source[(uint64_t)source_row * input_dimension + (k_base / 128u) * 128u + element])));
		scale = LmActivationFp8Scale(__shfl_sync(0xffffffffu,LmActivationWarpMax(amax),0));
		for (element=lane; element<TILE_K; element+=32u)
		{
			value = LmBf16ToFloat(source[(uint64_t)source_row * input_dimension + k_base + element]);
			offset = SWIZZLED ? LmSwizzledOffset(local_row,element * 2u,TILE_K * 2u,LmSwizzleSpanFor(TILE_K * 2u)) / 2u : local_row * TILE_K + element;
			tile[offset] = LmFloatToBf16(LmActivationFp8Qdq<ACTIVATION_CODEC>(value,scale));
		}
	}
}
