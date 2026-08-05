#pragma once

// NVFP4: E2M1 data, UE4M3 scale, one scale per 16 elements. Decoded to BF16.
//
// sm_121a has a block-scaled FP4 mma that consumes the scales directly, and this
// file does not use it. That is deliberate and worth defending: it is native,
// it is verified present, and it would do twice the FLOP per instruction.
//
// It is not used because the decode path needs sixty times less compute than the
// machine has, and because using it would fork the library. The block-scaled mma
// forces a different fragment layout, a different K depth, a scale-operand
// selection with a stride-0 mode, and an epilogue that differs from every other
// format - four divergences bought with headroom that is already spare. One mma
// and one fragment mapping is worth more than an instruction that saves time the
// kernel does not spend.
//
// kernels/mma.cuh still declares LmMmaNvfp4 and the capability gate still probes
// it, so the option remains open and measurable if a profile ever says the
// decode is the constraint.

#include "inference/kernels/mma.cuh"

// Two adjacent E2M1 nibbles share a byte when the index is even, which is the
// common case for a register pair: k is always even for the low half because
// LmMma16OperandBK returns 2*(lane%4) + 8*reg.
static __device__ __forceinline__ float2 LmNvfp4Pair(uint8_t packed)
{
	return(LmE2m1PairToFloat(packed));
}

struct LmNvfp4
{
	static constexpr uint32_t kStoredBits = 4u;
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
	static constexpr uint32_t kScaleGroup = LM_MMA4_NVFP4_GROUP;
	static constexpr float kMax = LM_E2M1_MAX;

	static __device__ __forceinline__ uint32_t OperandARow(uint32_t lane, uint32_t reg) { return(LmMma16OperandARow(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandAK(uint32_t lane, uint32_t reg) { return(LmMma16OperandAK(lane,reg)); }
	static __device__ __forceinline__ uint32_t OperandBRow(uint32_t lane) { return(LmMma16OperandBRow(lane)); }
	static __device__ __forceinline__ uint32_t OperandBK(uint32_t lane, uint32_t reg) { return(LmMma16OperandBK(lane,reg)); }

	// The mirror of Fragment(): a value becomes a code. tests/test_reference.c
	// round-trips every representable code through both.
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return((uint8_t)(LmFloatPairToE2m1(value,0.0f) & 15u));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		float2 pair = LmNvfp4Pair(tile[LmSwizzledOffset(row,k >> 1u,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes))]);
		return(LmPackBf16Pair(pair.x * scale,pair.y * scale));
	}
};
