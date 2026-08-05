#pragma once

// INT6. Six bits stored, eight bits computed.
//
// Four 6-bit values occupy three bytes exactly, which is the cleanest packing on
// this curve - no value straddles a 32-bit boundary and the unpack is four
// shifts of one loaded word.
//
// Measured: 2.191 percent at block 32, 5.71 bits of code entropy against 6.500
// stored. Homogeneous across planes - the first row of the measurement confirmed
// 2.190 versus 2.191 percent for the expert plane, so one curve covers both.
//
// This is a throughput lane, not an accuracy lane. INT7 is half the error for
// one more bit, and INT8 is a quarter for two. INT6 earns its place only where
// the weight stream is the binding constraint and 2.2 percent is acceptable,
// which at 96 percent of decode traffic being weights is a real place to be.
//
// Note this is NOT the native six-bit path. kernels/formats/fp6.cuh has E3M2 and
// E2M3, which the tensor core consumes directly but which tile badly: 0.75 bytes
// per element means a K tile must divide by 512 to be a whole swizzle span, and
// at that depth only a 16-row tile fits shared memory. INT6 stores just as
// narrow, unpacks to the 8-bit layout, and tiles like every other 8-bit format.
// The unpack is the price of not fighting the geometry.

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

// Extract one signed 6-bit code by index. Bit offset is index * 6, so a
// code may straddle a byte boundary; a 32-bit load covering the offset always
// contains it because 6 < 25. Sign extension is a shift to the top of a word
// and an arithmetic shift back - branchless and exact for every code, which
// tests/test_pack.c checks exhaustively.
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
	// The K tile this format needs, in ELEMENTS. It is a property of the
	// stored width, not a free choice: the row pitch must be a whole swizzle
	// span in BYTES, so 8 bits needs 128 elements and 7 needs 256. Hardcoding
	// it at a call site is a static_assert away from being caught, and was.
	static constexpr uint32_t kTileK = 128u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr bool kScaleInMma = false;
	// Block 32 rather than 128: six bits has no headroom for an outlier, and the
	// measurement is at blk32.
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

	// Decode the two adjacent codes a BF16 register covers, straight from packed
	// shared memory into the register the mma consumes. No intermediate buffer
	// and no extra barrier: shared holds only the packed form, so the 6-bit
	// saving is paid on the bus and never given back in shared memory.
	// Sign extension and block scaling happen while the two BF16 register lanes
	// are formed. Shared memory and the global-memory stream remain six-bit; no
	// expanded weight tile or persistent expanded copy exists.
	// The mirror of Fragment(): a value becomes a code. tests/test_reference.c
	// round-trips every representable code through both.
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
