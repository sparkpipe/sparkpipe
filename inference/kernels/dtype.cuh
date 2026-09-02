#pragma once


#include <cuda_fp16.h>
#include <stdint.h>


struct LmBf16 { static constexpr uint32_t kBits = 16u; };
struct LmE4m3 { static constexpr uint32_t kBits = 8u; };
struct LmE5m2 { static constexpr uint32_t kBits = 8u; };
struct LmE2m1 { static constexpr uint32_t kBits = 4u; };
struct LmUe8m0 { static constexpr uint32_t kBits = 8u; };
struct LmUe4m3 { static constexpr uint32_t kBits = 8u; };
struct LmF32 { static constexpr uint32_t kBits = 32u; };

#define LM_E4M3_MAX 448.0f
#define LM_E5M2_MAX 57344.0f
#define LM_E2M1_MAX 6.0f

template<class T> struct LmPacking { static constexpr uint32_t kPerByte = 8u / T::kBits; };

template<class T>
static __host__ __device__ __forceinline__ uint64_t LmBytesForElements(uint64_t elements)
{
	return((elements * (uint64_t)T::kBits) / 8u);
}


static __device__ __forceinline__ float LmBf16ToFloat(uint16_t value)
{
	return(__uint_as_float((uint32_t)value << 16u));
}

static __device__ __forceinline__ float LmScalarToFloat(uint16_t value)
{
	return(LmBf16ToFloat(value));
}

static __device__ __forceinline__ float LmScalarToFloat(float value)
{
	return(value);
}

static __device__ __forceinline__ uint16_t LmFloatToBf16(float value)
{
	uint32_t bits = __float_as_uint(value);
	return((uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u));
}

static __device__ __forceinline__ float2 LmLoadBf16Pair(const void *base, uint64_t pair_index)
{
	uint32_t packed = ((const uint32_t *)base)[pair_index];
	float2 result;
	result.x = __uint_as_float(packed << 16u);
	result.y = __uint_as_float(packed & 0xffff0000u);
	return(result);
}


static __device__ __forceinline__ uint8_t LmFloatToE4m3(float value)
{
	uint16_t encoded;
	asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(0.0f), "f"(value));
	return((uint8_t)encoded);
}

static __device__ __forceinline__ uint16_t LmFloatPairToE4m3(float low, float high)
{
	uint16_t encoded;
	asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(high), "f"(low));
	return(encoded);
}

static __device__ __forceinline__ float LmE4m3ToFloat(uint8_t value)
{
	uint32_t widened;
	asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
		: "=r"(widened) : "h"((uint16_t)value));
	return(__half2float(__ushort_as_half((uint16_t)(widened & 0xffffu))));
}

static __device__ __forceinline__ float2 LmE4m3PairToFloat2(uint16_t value)
{
	uint32_t widened;
	float2 result;
	asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
		: "=r"(widened) : "h"(value));
	result.x = __half2float(__ushort_as_half((uint16_t)widened));
	result.y = __half2float(__ushort_as_half((uint16_t)(widened >> 16u)));
	return(result);
}


static __device__ __forceinline__ uint8_t LmFloatPairToE2m1(float low, float high)
{
	uint16_t encoded;
	asm volatile("{\n\t.reg .b8 packed;\n"
		"\tcvt.rn.satfinite.e2m1x2.f32 packed, %1, %2;\n"
		"\tcvt.u16.u8 %0, packed;\n\t}\n"
		: "=h"(encoded) : "f"(high), "f"(low));
	return((uint8_t)(encoded & 0xffu));
}

static __device__ __forceinline__ float2 LmE2m1PairToFloat(uint8_t packed)
{
	uint32_t widened;
	float2 result;
	asm volatile("{\n\t.reg .b8 narrow;\n"
		"\tcvt.u8.u16 narrow, %1;\n"
		"\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "h"((uint16_t)packed));
	result.x = __half2float(__ushort_as_half((uint16_t)(widened & 0xffffu)));
	result.y = __half2float(__ushort_as_half((uint16_t)(widened >> 16u)));
	return(result);
}

static __device__ __forceinline__ uint8_t LmFloatToUe8m0(float value)
{
	uint16_t encoded;
	asm volatile("cvt.rn.satfinite.ue8m0x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(0.0f), "f"(value));
	return((uint8_t)(encoded >> 8u));
}

static __device__ __forceinline__ float LmUe8m0ToFloat(uint8_t value)
{
	uint32_t widened;
	asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;\n"
		: "=r"(widened) : "h"((uint16_t)value));
	return(LmBf16ToFloat((uint16_t)(widened & 0xffffu)));
}

static __device__ __forceinline__ uint32_t LmRoundPositiveToNearestEven(
	float value)
{
	uint32_t lower = (uint32_t)value;
	float remainder = value - (float)lower;
	return(lower + (remainder > 0.5f ||
		(remainder == 0.5f && (lower & 1u) != 0u) ? 1u : 0u));
}

static __device__ __forceinline__ uint8_t LmFloatToUe4m3(float value)
{
	int32_t exponent;
	float normalized;
	uint32_t biased,mantissa;
	if ( !(value > 0.0f) )
		return(0u);
	if ( !(value < LM_E4M3_MAX) )
		return(0x7eu);
	if ( value < 0.015625f )
	{
		mantissa = LmRoundPositiveToNearestEven(value * 512.0f);
		return((uint8_t)(mantissa > 8u ? 8u : mantissa));
	}
	normalized = frexpf(value,&exponent) * 2.0f;
	biased = (uint32_t)(exponent + 6);
	mantissa = LmRoundPositiveToNearestEven((normalized - 1.0f) * 8.0f);
	if ( mantissa == 8u )
	{
		mantissa = 0u;
		biased += 1u;
	}
	if ( biased > 15u || (biased == 15u && mantissa > 6u) )
		return(0x7eu);
	return((uint8_t)((biased << 3u) | mantissa));
}

static __device__ __forceinline__ float LmUe4m3ToFloat(uint8_t value)
{
	uint32_t biased,mantissa;
	value &= 0x7fu;
	biased = (uint32_t)value >> 3u;
	mantissa = (uint32_t)value & 7u;
	if ( biased == 0u )
		return(ldexpf((float)mantissa,-9));
	if ( biased == 15u && mantissa == 7u )
		return(LM_E4M3_MAX);
	return(ldexpf(1.0f + ((float)mantissa / 8.0f),(int32_t)biased - 7));
}

template<class T>
static __device__ __forceinline__ float LmScaleFromAbsmax(float absmax)
{
	float maximum = 0.0f;
	if constexpr ( T::kBits == 4u )
		maximum = LM_E2M1_MAX;
	else
		maximum = LM_E4M3_MAX;
	return(fmaxf(absmax / maximum,1.0e-8f));
}
