#pragma once

// INT8. Eight bits stored, decoded to BF16.
//
// The format W8LUT was for, done directly. W8LUT stored an 8-bit index into a
// per-block 256-entry lookup table - a learned codebook rather than a linear
// scale - which costs a table read per value and a pack format nothing else
// understands, in exchange for accuracy INT8 already has:
//
//     format   blk   error    coded bits
//     FP8       -    2.57%      8.06
//     INT8     128   0.647%     7.656
//
// Four times better than FP8 at fewer bits, with a scale instead of a table.
// That is why W8LUT is gone and this is here.
//
// INT8 CANNOT USE THE FREE DEQUANT. BF16 has exactly seven mantissa bits, so an
// eight-bit code overflows into the exponent and comes out doubled -
// tests/test_dequant.c has that as an explicit rejection. This path converts
// instead, which is two instructions per value against the OR that INT7 gets.
//
// That is an argument for INT7 rather than against INT8: 1.304 percent at 6.651
// coded bits, free to decode, against 0.647 percent at 7.656 bits that is not.
// Which one wins depends on whether the weight stream or the error budget is the
// binding constraint, and both are here so the choice is a config line.

#include "inference/kernels/mma.cuh"

struct LmInt8
{
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr bool kTmaSwizzle = true;
	// The K tile this format needs, in ELEMENTS. It is a property of the
	// stored width, not a free choice: the row pitch must be a whole swizzle
	// span in BYTES, so 8 bits needs 128 elements and 7 needs 256. Hardcoding
	// it at a call site is a static_assert away from being caught, and was.
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kBits = 16u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr uint32_t kScaleGroup = 128u;
	static constexpr float kMax = 127.0f;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)(int8_t)__float2int_rn(fminf(fmaxf(value,-128.0f),127.0f)));
	}

	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const int8_t *base = (const int8_t *)(tile
			+ LmSwizzledOffset(row,0u,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes)));
		return(LmPackBf16Pair((float)base[k] * scale,(float)base[k + 1u] * scale));
	}
};
