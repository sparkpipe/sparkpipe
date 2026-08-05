#pragma once

// FP8 E4M3, decoded to BF16.
//
// Every format in this library is a decoder into the same BF16 fragment. FP8 was
// the last exception: it carried per-128-block FP32 scales applied outside the
// mma, which forced a partial accumulator, a rescale pass, and - because that
// arrangement is exactly what CUTLASS's blockwise collective implements - the
// entire vendored dependency.
//
// Decoding instead removes all four. The scale multiplies during decode, where
// it costs one fma per pair on a path with sixty times more compute than it can
// use. There is no second accumulator, no second code path, and nothing left for
// the external GEMM to do.
//
// Worth stating plainly since it is the reason this file looks trivial: FP8 is
// also the worst point on the accuracy-per-bit curve. Measured on these weights,
// FP8 is 8.06 coded bits at 2.57 percent error, where INT7 is 6.651 bits at
// 1.304 percent - fewer bytes AND half the error. FP8 exists here to read
// checkpoints that are already in it, not because it is a good choice.

#include "inference/kernels/mma.cuh"

struct LmFp8
{
	// What memory holds, and therefore what crosses the bus.
	static constexpr uint32_t kStoredBits = 8u;
	static constexpr bool kTmaSwizzle = true;
	// The K tile this format needs, in ELEMENTS. It is a property of the
	// stored width, not a free choice: the row pitch must be a whole swizzle
	// span in BYTES, so 8 bits needs 128 elements and 7 needs 256. Hardcoding
	// it at a call site is a static_assert away from being caught, and was.
	static constexpr uint32_t kTileK = 128u;
	// What the mma register holds after decode. One path for every format.
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

	// The two adjacent codes a BF16 register covers, scaled and packed. One
	// aligned 16-bit load, two hardware conversions, one pack.
	// The mirror of Fragment(): a value becomes a code. tests/test_reference.c
	// round-trips every representable code through both.
	static __device__ __forceinline__ uint8_t Encode(float value)
	{
		return(LmFloatToE4m3(value));
	}
	static __device__ __forceinline__ uint32_t Fragment(const uint8_t *tile, uint32_t row, uint32_t k, uint32_t row_pitch_bytes, float scale)
	{
		const uint8_t *base = tile + LmSwizzledOffset(row,0u,row_pitch_bytes,LmSwizzleSpanFor(row_pitch_bytes)) + k;
		return(LmPackBf16Pair(LmE4m3ToFloat(base[0]) * scale,
			LmE4m3ToFloat(base[1]) * scale));
	}
};
