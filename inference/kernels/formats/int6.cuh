#pragma once


#include "inference/kernels/mma.cuh"

#define LM_INT6_VALUES_PER_GROUP 4u
#define LM_INT6_BYTES_PER_GROUP 3u

static __device__ __forceinline__ void LmInt6UnpackGroup(const uint8_t *packed, int8_t *out)
{
	uint32_t bits = (uint32_t)packed[0] | ((uint32_t)packed[1] << 8u) | ((uint32_t)packed[2] << 16u);
	uint32_t index;
	for (index = 0u; index < LM_INT6_VALUES_PER_GROUP; ++index)
		out[index] = (int8_t)(((int32_t)(bits << (26u - (index * 6u)))) >> 26);
}

static __device__ __forceinline__ uint32_t LmInt6Raw(const uint8_t *base, uint32_t index, uint32_t row_bytes)
{
	uint32_t bit = index * 6u;
	uint32_t byte = bit >> 3u;
	uint32_t word = base[byte];
	if ( byte + 1u < row_bytes )
		word |= (uint32_t)base[byte + 1u] << 8u;
	return(word >> (bit & 7u));
}

struct LmInt6
{
	typedef float Accumulator;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kStoredBits = 6u;
	static constexpr bool kTmaSwizzle = false;
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 32u;
	static constexpr float kMax = 31.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }
	static __device__ __forceinline__ void Mma(float acc[4], const uint32_t a[4], const uint32_t b[2], uint32_t, uint32_t)
	{
		LmMmaBf16(acc,a,b);
	}

	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(value,-32.0f),31.0f)));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const uint8_t *base = tile + (row * row_pitch_bytes);
		const uint32_t low = LmInt6Raw(base,k,row_pitch_bytes) & 63u;
		const uint32_t high = LmInt6Raw(base,k + 1u,row_pitch_bytes) & 63u;
		const int32_t signed_low = ((int32_t)(low << 26u)) >> 26;
		const int32_t signed_high = ((int32_t)(high << 26u)) >> 26;
		return(LmPackBf16Pair((float)signed_low * scale,(float)signed_high * scale));
	}
};
