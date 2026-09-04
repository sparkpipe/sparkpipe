#pragma once


#include "inference/kernels/mma.cuh"

#define LM_INT7_VALUES_PER_GROUP 8u
#define LM_INT7_BYTES_PER_GROUP 7u

static __device__ __forceinline__ void LmInt7UnpackGroup(const uint8_t *packed, int8_t *out)
{
	uint32_t low = *(const uint32_t *)packed;
	uint32_t high = (uint32_t)packed[4] | ((uint32_t)packed[5] << 8u) | ((uint32_t)packed[6] << 16u);
	uint64_t bits = (uint64_t)low | ((uint64_t)high << 32u);
	uint32_t index;
	for (index = 0u; index < LM_INT7_VALUES_PER_GROUP; ++index)
		out[index] = (int8_t)(((int32_t)((uint32_t)(bits >> (index * 7u)) << 25u)) >> 25);
}

static __device__ __forceinline__ uint32_t LmInt7Raw(const uint8_t *base, uint32_t index, uint32_t row_bytes)
{
	uint32_t bit = index * 7u;
	uint32_t byte = bit >> 3u;
	uint32_t word = base[byte];
	if ( byte + 1u < row_bytes )
		word |= (uint32_t)base[byte + 1u] << 8u;
	return(word >> (bit & 7u));
}

struct LmInt7
{
	typedef float Accumulator;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kStoredBits = 7u;
	static constexpr bool kTmaSwizzle = false;
	static constexpr uint32_t kTileK = 256u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr bool kScaleInMma = false;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = 63.0f;

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
		return((uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(value,-64.0f),63.0f)));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const uint8_t *base = tile + (row * row_pitch_bytes);
		const uint32_t low = LmInt7Raw(base,k,row_pitch_bytes) & 127u;
		const uint32_t high = LmInt7Raw(base,k + 1u,row_pitch_bytes) & 127u;
		const int32_t signed_low = ((int32_t)(low << 25u)) >> 25;
		const int32_t signed_high = ((int32_t)(high << 25u)) >> 25;
		return(LmPackBf16Pair((float)signed_low * scale,(float)signed_high * scale));
	}
};
