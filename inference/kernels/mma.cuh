#pragma once


#include "inference/kernels/dtype.cuh"
#include "inference/kernels/layout.cuh"
#include <string.h>
#include <stdint.h>

#define LM_WARP_LANES 32u

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 1210)
#define LM_SM121_NATIVE_COMPUTE_PTX 1
#else
#define LM_SM121_NATIVE_COMPUTE_PTX 0
#endif


static __device__ __forceinline__ uint32_t LmMmaAccumulatorRow(uint32_t lane, uint32_t entry)
{
	return((lane / 4u) + (8u * (entry / 2u)));
}

static __device__ __forceinline__ uint32_t LmMmaAccumulatorColumn(uint32_t lane, uint32_t entry)
{
	return((2u * (lane % 4u)) + (entry % 2u));
}


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

static __device__ __forceinline__ uint32_t LmMma4ScaleARow(uint32_t lane)
{
	return((8u * (lane % 2u)) + (lane / 4u));
}

static __device__ __forceinline__ uint32_t LmMma4ScaleBRow(uint32_t lane)
{
	return(lane / 4u);
}

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


#define LM_MMA16_M 16u
#define LM_MMA16_N 8u
#define LM_MMA16_K 16u


static __device__ __forceinline__ uint32_t LmMma16OperandARow(uint32_t lane, uint32_t reg)
{
	return((lane / 4u) + (8u * (reg % 2u)));
}

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

template<uint32_t BITS>
static __device__ __forceinline__ uint32_t LmCodeToBf16Bits(uint32_t code)
{
	static_assert(BITS >= 2u && BITS <= 7u,
		"BF16 has 7 mantissa bits; a wider code lands in the exponent and doubles the value");
	return(0x4300u | ((code & ((1u << BITS) - 1u)) ^ (1u << (BITS - 1u))));
}

template<uint32_t BITS>
static __host__ __device__ constexpr float LmCodeBias(void)
{
	return((float)(128 + (1 << (BITS - 1))));
}

template<uint32_t BITS>
static __device__ __forceinline__ uint32_t LmPackCodePairBf16(uint32_t low, uint32_t high)
{
	return(LmCodeToBf16Bits<BITS>(low) | (LmCodeToBf16Bits<BITS>(high) << 16u));
}

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


static __device__ __forceinline__ uint32_t LmSwizzleChunk(uint32_t chunk, uint32_t row, uint32_t row_pitch_bytes, uint32_t span_bytes)
{
	return(chunk ^ LmSwizzleRowSelector(row,row_pitch_bytes,span_bytes));
}

static __device__ __forceinline__ uint32_t LmSwizzledOffset(uint32_t row, uint32_t byte_in_row, uint32_t row_pitch_bytes, uint32_t span_bytes)
{
	uint32_t chunk = LmSwizzleChunk(byte_in_row / LM_SWIZZLE_CHUNK_BYTES,row,row_pitch_bytes,span_bytes);
	return((row * row_pitch_bytes) + (chunk * LM_SWIZZLE_CHUNK_BYTES)
		+ (byte_in_row % LM_SWIZZLE_CHUNK_BYTES));
}
