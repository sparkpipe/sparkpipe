#pragma once


#include "inference/kernels/formats/nvfp4.cuh"

#define LM_MXFP4_GROUP 32u

static __device__ __forceinline__ float LmE8m0ToFloat(uint8_t code)
{
	return(code == 0xffu ? __int_as_float(0x7fffffff) : exp2f((float)code - 127.0f));
}

struct LmMxfp4
{
	static constexpr uint32_t kStoredBits = 4u;
	static constexpr bool kTmaSwizzle = true;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr uint32_t kScaleGroup = LM_MXFP4_GROUP;
	static constexpr float kMax = LM_E2M1_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	static __device__ __forceinline__ float ScaleDecode(uint8_t code)
	{
		return(LmE8m0ToFloat(code));
	}
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)(LmFloatPairToE2m1(value,0.0f) & 15u));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const uint32_t span = LmSwizzleSpanFor(row_pitch_bytes);
		const uint32_t byte = span != 0u
			? LmSwizzledOffset(row,k >> 1u,row_pitch_bytes,span)
			: (row * row_pitch_bytes) + (k >> 1u);
		float2 pair = LmNvfp4Pair(tile[byte]);
		return(LmPackBf16Pair(pair.x * scale,pair.y * scale));
	}
};
