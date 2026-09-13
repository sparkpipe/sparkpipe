#pragma once

// INT7. Seven bits stored, eight bits computed.
//
// There is no s7 mma - tests/test_ptx_capability_gate.py probes for one and it
// is absent at every shape. INT7 therefore stores 7 bits and unpacks to s8
// during staging, so the global stream and the TMA box pay 7 bits per weight
// while the mma sees the native 8-bit operand layout it already knows.
//
// That split is the whole point and it is why kStoredBits and kBits are separate
// trait fields. Bandwidth is set by what crosses the bus; correctness is set by
// what the tensor core receives. Conflating them is how a "6-bit kernel" ends up
// moving 8 bits.
//
// WHY THIS OVER FP8. Measured on this weight distribution:
//
//     format   blk    error    stored b/w   entropy-coded b/w
//     FP8       -     2.57%       8.000           8.06
//     INT7     128    1.304%      7.125           6.651
//     INT8     128    0.647%      8.125           7.656
//
// INT7 dominates FP8 on both axes at once - 17 percent fewer coded bits and
// half the error. FP8 is the most expensive and least accurate point on the
// curve, which is not obvious until the two are measured against each other.
//
// ENTROPY CODING IS A SEPARATE LAYER. The 6.651 figure above is what an entropy
// coder achieves over these codes; the packing here is fixed-width 7-bit, which
// gets 7.125 including the block scale. The coder replaces LmInt7Unpack and
// nothing else - the trait interface is what makes that a drop-in rather than a
// second kernel.

#include "inference/kernels/mma.cuh"

#define LM_INT7_VALUES_PER_GROUP 8u
#define LM_INT7_BYTES_PER_GROUP 7u

// Eight 7-bit values occupy seven bytes. Unpacking one group is two aligned
// 32-bit loads and a fixed shift schedule, which the compiler resolves entirely
// at compile time because every shift below is a constant.
//
// Sign extension is explicit: a 7-bit code is signed, and shifting into the top
// of a 32-bit word then arithmetic-shifting back is branchless and exact.
static __device__ __forceinline__ void LmInt7UnpackGroup(const uint8_t *packed, int8_t *out)
{
	uint32_t low = *(const uint32_t *)packed;
	uint32_t high = (uint32_t)packed[4] | ((uint32_t)packed[5] << 8u) | ((uint32_t)packed[6] << 16u);
	uint64_t bits = (uint64_t)low | ((uint64_t)high << 32u);
	uint32_t index;
	for (index = 0u; index < LM_INT7_VALUES_PER_GROUP; ++index)
		out[index] = (int8_t)(((int32_t)((uint32_t)(bits >> (index * 7u)) << 25u)) >> 25);
}

// Extract one signed 7-bit code by index. Bit offset is index * 7, so a
// code may straddle a byte boundary; a 32-bit load covering the offset always
// contains it because 7 < 25. Sign extension is a shift to the top of a word
// and an arithmetic shift back - branchless and exact for every code, which
// tests/test_pack.c checks exhaustively.
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
	// What the mma register holds after decode. Every packed format lands in
	// BF16, so there is one mma path in the library and one accumulator type.
	static constexpr uint32_t kBits = 16u;
	// What memory and the TMA box hold. This is the number that sets bandwidth.
	static constexpr uint32_t kStoredBits = 7u;
	static constexpr bool kTmaSwizzle = false;
	// The K tile this format needs, in ELEMENTS. It is a property of the
	// stored width, not a free choice: the row pitch must be a whole swizzle
	// span in BYTES, so 8 bits needs 128 elements and 7 needs 256. Hardcoding
	// it at a call site is a static_assert away from being caught, and was.
	static constexpr uint32_t kTileK = 256u;
	static constexpr uint32_t kMmaM = LM_MMA16_M;
	static constexpr uint32_t kMmaN = LM_MMA16_N;
	static constexpr uint32_t kMmaK = LM_MMA16_K;
	static constexpr bool kScaleInMma = false;
	// Block 128 is the measured knee: 1.304 percent at blk128 against 2.191 at
	// blk32 for six bits. Smaller blocks buy outlier safety and cost scale bytes.
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

	// Decode the two adjacent codes a BF16 register covers, straight from packed
	// shared memory into the register the mma consumes. No intermediate buffer
	// and no extra barrier: shared holds only the packed form, so the 7-bit
	// saving is paid on the bus and never given back in shared memory.
	// Sign extension and block scaling happen while the two BF16 register lanes
	// are formed. Shared memory and the global-memory stream remain seven-bit;
	// no expanded weight tile or persistent expanded copy exists.
	// The mirror of Fragment(): a value becomes a code. tests/test_reference.c
	// round-trips every representable code through both.
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
