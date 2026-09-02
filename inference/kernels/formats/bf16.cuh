#pragma once


#include "inference/kernels/mma.cuh"

struct LmBf16Format
{
	static constexpr uint32_t kStoredBits = 16u;
	static constexpr bool kTmaSwizzle = true;
	static constexpr uint32_t kTileK = 64u;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr uint32_t kScaleGroup = 0u;
	static constexpr float kMax = 0.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)0);
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float)
	{
		return(*(const uint32_t *)(tile + LmSwizzledOffset(row,k * 2u,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes))));
	}
};
