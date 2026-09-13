#pragma once

// MMA atoms and their fragment mappings.
//
// The register-to-matrix-element mapping is the one part of an mma.sync kernel
// that a wrong implementation renders silently incorrect while still
// assembling. Every mapping below is transcribed from a CUTLASS MMA_Traits
// declaration, and tests/test_mma_fragment_mapping.c evaluates those CuTe
// layouts and checks each formula element by element, including a bijection
// check and a negative control.
//
// Sources, so a future change can be checked against the same place:
//   cute/atom/mma_traits_sm120.hpp  SM120_16x8x32_TN            (FP8 path)
//   cute/atom/mma_traits_sm80.hpp   SM80_16x8x32_S32S8S8S32_TN  (what it inherits)
//   cute/atom/mma_traits_sm120.hpp  SM120::BLOCKSCALED::SM120_16x8x64_TN_VS
//   cute/atom/mma_traits_sm80.hpp   SM80_16x8_Row               (accumulator)
//
// sm_121a selects SM120_16x8x32_TN, not the SM89 atom, but the two are
// byte-identical in layout - the verifier asserts that rather than assuming it.
//
// What sm_121a does NOT have, assembled rather than asserted by
// tests/test_ptx_capability_gate.py: wgmma, tcgen05, and the scale_vec::4X with
// ue8m0 combination that CUTLASS emits for its VS=16 path. That last one means
// the SM120 collective's NVFP4 emission cannot be copied here; on this target
// NVFP4 is scale_vec::4X with ue4m3.

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/layout.cuh"
#include <string.h>
#include <stdint.h>

#define LM_WARP_LANES 32u

/*
 * The native V4 decode kernels below this header are deliberately tied to
 * the architecture-qualified SM121 target.  PTX block-scaled mma is not a
 * portable fallback: compiling another real-device pass must leave a trap,
 * never a scalar or BF16-dequant implementation that can be mistaken for the
 * qualified route.  The host half of an nvcc translation also sees the trap;
 * only the SM121 device pass sees the architecture-specific instruction.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)
#define LM_SM121_NATIVE_COMPUTE_PTX 1
#else
#define LM_SM121_NATIVE_COMPUTE_PTX 0
#endif

// -- accumulator, shared by every atom ---------------------------------------
//
// SM80_16x8_Row: ((4,8),(2,2)) : ((32,1),(16,8)) over a 16x8 tile. Four floats
// per lane covering two rows and two columns.

static __device__ __forceinline__ uint32_t LmMmaAccumulatorRow(uint32_t lane, uint32_t entry)
{
	return((lane / 4u) + (8u * (entry / 2u)));
}

static __device__ __forceinline__ uint32_t LmMmaAccumulatorColumn(uint32_t lane, uint32_t entry)
{
	return((2u * (lane % 4u)) + (entry % 2u));
}

// -- 8-bit atom: m16n8k32 ----------------------------------------------------
//
// A: ((4,8),(4,2,2)) : ((64,1),(16,8,256)) over M16 x K32
//    register r, byte b -> row lane/4 + 8*(r%2), col 4*(lane%4) + b + 16*(r/2)
// B: ((4,8),(4,2))   : ((32,1),(8,128))    over N8 x K32
//    register r, byte b -> row lane/4,        col 4*(lane%4) + b + 16*r
//
// Both work out to four contiguous bytes per register, which is a single
// aligned 32-bit shared load. That is why the operand loaders below are plain
// loads and not ldmatrix: ldmatrix would need its own address derivation, and a
// second derivation is a second chance to be silently wrong.

#define LM_MMA8_M 16u
#define LM_MMA8_N 8u
#define LM_MMA8_K 32u

static __device__ __forceinline__ uint32_t LmMma8OperandARow(uint32_t lane, uint32_t reg)
{
	return((lane / 4u) + (8u * (reg % 2u)));
}

static __device__ __forceinline__ uint32_t LmMma8OperandAByte(uint32_t lane, uint32_t reg)
{
	return((4u * (lane % 4u)) + (16u * (reg / 2u)));
}

static __device__ __forceinline__ uint32_t LmMma8OperandBRow(uint32_t lane)
{
	return(lane / 4u);
}

static __device__ __forceinline__ uint32_t LmMma8OperandBByte(uint32_t lane, uint32_t reg)
{
	return((4u * (lane % 4u)) + (16u * reg));
}

static __device__ __forceinline__ void LmMmaE4m3(float accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

/*
 * SM121 mixed-width block-scaled atoms used by the V4 decode path.
 *
 * PTX represents every f8/f6/f4 element in the m16n8k32 register fragment as
 * one byte.  MXFP4 therefore remains nibble-packed in global/shared memory and
 * is placed in the central four bits of each padded byte only while the two B
 * registers are formed; it is never decoded to BF16.  A and B retain
 * independent UE8M0 scale bytes, one per K32 block.
 * The register coordinates are consequently the m16n8k32 byte coordinates
 * above for both the E4M3 and E2M1 operands.
 *
 * These exact forms are independently listed in the ptxas capability gate.
 * Static source tests prove that the production kernels call them; only a live
 * sm_121a nvcc/ptxas build and GA tensor qualification can prove deployment.
 */
static __device__ __forceinline__ void LmMmaMxf8Mxf4(
	float accumulator[4],
	const uint32_t a[4],
	const uint32_t b[2],
	uint32_t scale_a,
	uint32_t scale_b)
{
#if LM_SM121_NATIVE_COMPUTE_PTX
	asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
		".block_scale.scale_vec::1X.f32.e4m3.e2m1.f32.ue8m0 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
		"%10, {0, 0}, %11, {0, 0};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]),
		  "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
		  "r"(b[0]), "r"(b[1]), "r"(scale_a), "r"(scale_b));
#else
	(void)accumulator;
	(void)a;
	(void)b;
	(void)scale_a;
	(void)scale_b;
	asm volatile("trap;\n");
#endif
}

static __device__ __forceinline__ void LmMmaMxf8Mxf8(
	float accumulator[4],
	const uint32_t a[4],
	const uint32_t b[2],
	uint32_t scale_a,
	uint32_t scale_b)
{
#if LM_SM121_NATIVE_COMPUTE_PTX
	asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4"
		".block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
		"%10, {0, 0}, %11, {0, 0};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]),
		  "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
		  "r"(b[0]), "r"(b[1]), "r"(scale_a), "r"(scale_b));
#else
	(void)accumulator;
	(void)a;
	(void)b;
	(void)scale_a;
	(void)scale_b;
	asm volatile("trap;\n");
#endif
}

// -- 4-bit atom: m16n8k64 block-scaled ---------------------------------------
//
// A: ((4,8),(8,2,2)) : ((128,1),(16,8,512)) over M16 x K64
//    register r, nibble n -> row lane/4 + 8*(r%2), col 8*(lane%4) + n + 32*(r/2)
// B: ((4,8),(8,2))   : ((64,1),(8,256))     over N8 x K64
//    register r, nibble n -> row lane/4,         col 8*(lane%4) + n + 32*r
//
// Same four registers for A and two for B as the 8-bit atom: half the width,
// twice the depth, identical register footprint. Also four contiguous bytes per
// register once the nibble packing is accounted for.
//
// Scale factors, from SFALayout ((2,2,8),64):((8,0,1),16) and SFBLayout
// ((4,8),64):((0,1),8). Both carry a stride-0 mode, so several lanes hold the
// same scale - two for A, four for B - which is what the {byte, thread}
// selectors on the instruction exist to disambiguate. The canonical selection
// is {0,0} for both.

#define LM_MMA4_M 16u
#define LM_MMA4_N 8u
#define LM_MMA4_K 64u
#define LM_MMA4_NVFP4_GROUP 16u
#define LM_MMA4_MXFP4_GROUP 32u

static __device__ __forceinline__ uint32_t LmMma4OperandARow(uint32_t lane, uint32_t reg)
{
	return((lane / 4u) + (8u * (reg % 2u)));
}

static __device__ __forceinline__ uint32_t LmMma4OperandAByte(uint32_t lane, uint32_t reg)
{
	return((4u * (lane % 4u)) + (16u * (reg / 2u)));
}

static __device__ __forceinline__ uint32_t LmMma4OperandBRow(uint32_t lane)
{
	return(lane / 4u);
}

static __device__ __forceinline__ uint32_t LmMma4OperandBByte(uint32_t lane, uint32_t reg)
{
	return((4u * (lane % 4u)) + (16u * reg));
}

// Row of the A scale this lane supplies: SFA's stride-0 mode is bit 1 of the
// lane, so lanes differing only there carry the same value.
static __device__ __forceinline__ uint32_t LmMma4ScaleARow(uint32_t lane)
{
	return((8u * (lane % 2u)) + (lane / 4u));
}

static __device__ __forceinline__ uint32_t LmMma4ScaleBRow(uint32_t lane)
{
	return(lane / 4u);
}

// NVFP4: e2m1 data, ue4m3 scale, one scale per 16 elements.
//
// CUTLASS emits scale_vec::4X with ue8m0 for its VS=16 path, guarded by
// CUTE_ARCH_MXF4NVF4_4X_UE8M0_MMA_ENABLED. ptxas rejects that combination for
// sm_121a - it is an sm_100 capability - so the collective's emission cannot be
// transplanted. The gate keeps a negative probe on it.
static __device__ __forceinline__ void LmMmaNvfp4(float accumulator[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a, uint32_t scale_b)
{
	asm volatile("mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X"
		".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
		"%10, {0, 0}, %11, {0, 0};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
		  "r"(scale_a), "r"(scale_b));
}

// MXFP4: e2m1 data, ue8m0 scale, one scale per 32 elements.
static __device__ __forceinline__ void LmMmaMxfp4(float accumulator[4], const uint32_t a[4], const uint32_t b[2], uint32_t scale_a, uint32_t scale_b)
{
	asm volatile("mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X"
		".m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, "
		"%10, {0, 0}, %11, {0, 0};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
		  "r"(scale_a), "r"(scale_b));
}

// -- integer atoms -----------------------------------------------------------
//
// Integer mma accumulates in S32, not F32. That is a real difference and the
// reason Format carries an accumulator type: an integer GEMM's epilogue converts
// a fixed-point sum through a scale, where a float GEMM's already holds the
// value. Treating them alike gives output that is wrong by whatever the scale
// was.
//
// Operand layouts are the same as the corresponding float atoms at the same
// width - S8 shares SM80_16x8x32_S32S8S8S32_TN with the FP8 path, which is the
// layout tests/test_mma_fragment_mapping.c already verifies, and S4 at
// m16n8k64 shares the 4-bit layout with NVFP4.

static __device__ __forceinline__ void LmMmaS8(int32_t accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]), "+r"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void LmMmaS4(int32_t accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+r"(accumulator[0]), "+r"(accumulator[1]), "+r"(accumulator[2]), "+r"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -- 6-bit atoms -------------------------------------------------------------
//
// Six bits exists on this target as FLOAT, not integer: e3m2 and e2m3 through
// kind::f8f6f4. There is no s6 mma and no s7 mma of any shape - a 7-bit scheme
// would have to unpack to 8, which spends the storage saving it was for.
//
// Operands occupy a byte each in registers regardless, so the layout is the
// 8-bit one; only the stored form is narrower.

static __device__ __forceinline__ void LmMmaE3m2(float accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e3m2.e3m2.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

static __device__ __forceinline__ void LmMmaE2m3(float accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e2m3.e2m3.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -- bf16 atom: m16n8k16 -----------------------------------------------------
//
// Kept because prefill and the non-quantised families use it, and because a
// reference path that avoids quantisation entirely is the cheapest way to
// isolate a numerics bug to the quantiser.

#define LM_MMA16_M 16u
#define LM_MMA16_N 8u
#define LM_MMA16_K 16u

// A: ((4,8),(2,2,2)) : ((32,1),(16,8,128)) over M16 x K16
//    register r, half h -> row lane/4 + 8*(r%2), k 2*(lane%4) + h + 8*(r/2)
// B: ((4,8),(2,2))     : ((16,1),(8,64))    over N8 x K16
//    register r, half h -> row lane/4,        k 2*(lane%4) + h + 8*r
//
// Each register holds TWO CONSECUTIVE k values. That is the property the whole
// decode path rests on: a native BF16 tile serves a register with one aligned
// 32-bit read, and a packed format serves the same register by extracting two
// adjacent codes. Neither needs a gather, and the fragment layout is identical
// either way.

static __device__ __forceinline__ uint32_t LmMma16OperandARow(uint32_t lane, uint32_t reg)
{
	return((lane / 4u) + (8u * (reg % 2u)));
}

// First of the two k values this register covers.
static __device__ __forceinline__ uint32_t LmMma16OperandAK(uint32_t lane, uint32_t reg)
{
	return((2u * (lane % 4u)) + (8u * (reg / 2u)));
}

static __device__ __forceinline__ uint32_t LmMma16OperandBRow(uint32_t lane)
{
	return(lane / 4u);
}

static __device__ __forceinline__ uint32_t LmMma16OperandBK(uint32_t lane, uint32_t reg)
{
	return((2u * (lane % 4u)) + (8u * reg));
}

// Two floats into one register as a bf16 pair, low half first. The scale is
// applied here because a decoded code is a fixed-point integer and becomes a
// value only once multiplied - doing it later would mean carrying an integer
// through a float fragment.
// Dequantise a signed n-bit code to BF16 with no conversion instruction.
//
// BF16 is sign(1) exp(8) mantissa(7). 0x4300 is exponent 134, mantissa 0, value
// 2^7 = 128. OR-ing an m in [0,128) into the mantissa gives exactly 128 + m,
// because 128 * (1 + m/128) = 128 + m and the mantissa has precisely the bits
// the code needs. Flipping the sign bit of the code biases it into range, so a
// signed c becomes 128 + 2^(n-1) + c and one subtraction recovers it.
//
// That subtraction is free: it folds into the scale multiply that has to happen
// anyway, as v*scale - bias*scale, a single fma with a precomputed second term.
// The net saving against the arithmetic path is two conversion instructions per
// value, eight per register pair.
//
// This is the technique Marlin uses and cites to Kim et al. - "doing naive
// type-casts from INT4 to FP16 is slow" - adapted to BF16.
//
// BITS MUST BE 7 OR FEWER. BF16 has seven mantissa bits, so an eight-bit code
// overflows into the exponent and the value comes out doubled. The static_assert
// says so, and tests/test_dequant.c confirms 8 bits is rejected rather than
// silently wrong. That is an independent reason INT7 is the sweet spot: it is
// the widest code that dequantises into BF16 for free.
template<uint32_t BITS>
static __device__ __forceinline__ uint32_t LmCodeToBf16Bits(uint32_t code)
{
	static_assert(BITS >= 2u && BITS <= 7u,
		"BF16 has 7 mantissa bits; a wider code lands in the exponent and doubles the value");
	return(0x4300u | ((code & ((1u << BITS) - 1u)) ^ (1u << (BITS - 1u))));
}

// The constant that undoes the bias, to be multiplied by the block scale once
// per fragment rather than once per value.
template<uint32_t BITS>
static __host__ __device__ constexpr float LmCodeBias(void)
{
	return((float)(128 + (1 << (BITS - 1))));
}

// Two codes into one register as a BF16 pair. No conversion, no rounding: the
// bit patterns are exact and the caller applies scale and bias with one fma per
// half afterwards.
template<uint32_t BITS>
static __device__ __forceinline__ uint32_t LmPackCodePairBf16(uint32_t low, uint32_t high)
{
	return(LmCodeToBf16Bits<BITS>(low) | (LmCodeToBf16Bits<BITS>(high) << 16u));
}

// Store eight scaled values into a byte-aligned, thread-exclusive block.
//
// Eight codes occupy exactly Format::kStoredBits bytes, so consecutive owners
// never share a byte for 4-, 6-, 7-, or 8-bit formats. The old pair writer
// updated an overlapping 32-bit word; adjacent CUDA threads could lose each
// other's bits even though their logical codes were disjoint.
template<class Format>
static __device__ __forceinline__ void LmStoreCodeOctet(
    uint8_t *base,
    uint64_t bit,
    const float values[8])
{
    static_assert(
        Format::kStoredBits >= 1u && Format::kStoredBits <= 8u,
        "packed code width must fit in one byte");
    uint64_t packed = 0u;
    uint32_t index;
    const uint32_t mask = (1u << Format::kStoredBits) - 1u;
    const uint64_t byte = bit >> 3u;

    for (index = 0u; index < 8u; ++index)
    {
        packed |=
            ((uint64_t)Format::Encode(values[index]) & mask) <<
            (index * Format::kStoredBits);
    }
    memcpy(base + byte, &packed, Format::kStoredBits);
}

static __device__ __forceinline__ uint32_t LmPackBf16Pair(float low, float high)
{
	return((uint32_t)LmFloatToBf16(low) | ((uint32_t)LmFloatToBf16(high) << 16u));
}

static __device__ __forceinline__ void LmMmaBf16(float accumulator[4], const uint32_t a[4], const uint32_t b[2])
{
	asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
		"{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
		: "+f"(accumulator[0]), "+f"(accumulator[1]), "+f"(accumulator[2]), "+f"(accumulator[3])
		: "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// -- shared-memory swizzle ---------------------------------------------------
//
// Within a row, the 16-byte chunk at index c is xor'd with the row's 128-byte
// sector selector. It must match the swizzle encoded in the TMA descriptor or
// the kernel reads real data from the wrong place with no error anywhere.
//
// Justified by a count rather than by assertion: without it a 32-lane fragment
// load puts 16 lanes on one bank; with it, 4.

// The span is a property of the STORED width, not a global constant, because a
// sub-byte code that is not a power of two gives a row pitch that no large span
// divides. Seven bits over a 256-element tile is 224 bytes: 128 does not divide
// it, 64 does not, 32 does. Forcing a 128-byte span there would need a
// 1024-element tile and 252 KB of shared memory, which is how a good format gets
// rejected for a reason that has nothing to do with the format.
//
// A 32-byte span still permutes two chunks and still removes most of the
// conflict; it is a smaller win, not the absence of one.
static __device__ __forceinline__ uint32_t LmSwizzleChunk(uint32_t chunk, uint32_t row, uint32_t row_pitch_bytes, uint32_t span_bytes)
{
	return(chunk ^ LmSwizzleRowSelector(row,row_pitch_bytes,span_bytes));
}

// Byte offset of one operand register inside a swizzled row-major tile. Every
// operand load in every GEMM goes through this, so the swizzle is applied in
// exactly one place.
static __device__ __forceinline__ uint32_t LmSwizzledOffset(uint32_t row, uint32_t byte_in_row, uint32_t row_pitch_bytes, uint32_t span_bytes)
{
	uint32_t chunk = LmSwizzleChunk(byte_in_row / LM_SWIZZLE_CHUNK_BYTES,row,row_pitch_bytes,span_bytes);
	return((row * row_pitch_bytes) + (chunk * LM_SWIZZLE_CHUNK_BYTES)
		+ (byte_in_row % LM_SWIZZLE_CHUNK_BYTES));
}
