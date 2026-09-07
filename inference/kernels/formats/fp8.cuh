#pragma once


#include "inference/kernels/mma.cuh"

struct LmFp8
{
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr bool kTmaSwizzle = true;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = LM_E4M3_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return(LmFloatToE4m3(value));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const uint8_t *base = tile + LmSwizzledOffset(row,k,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes));
		return(LmPackBf16Pair(LmE4m3ToFloat(base[0]) * scale,
			LmE4m3ToFloat(base[1]) * scale));
	}
};
