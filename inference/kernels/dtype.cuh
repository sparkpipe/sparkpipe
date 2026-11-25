#pragma once


#include <cuda_fp16.h>
#include <stdint.h>

#include "inference/kernels/kernel_math.cuh"


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
	return(SparkMathBf16ToFloat(value));
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
	return(SparkMathFloatToBf16(value));
}

static __device__ __forceinline__ float2 LmLoadBf16Pair(const void *base, uint64_t pair_index)
{
	uint32_t packed = ((const uint32_t *)base)[pair_index];
	float2 result;
	result.x = __uint_as_float(packed << 16u);
	result.y = __uint_as_float(packed & 0xffff0000u);
	return(result);
}


static __device__ __forceinline__ uint8_t LmFloatToUe4m3(float value);

static __device__ __forceinline__ uint8_t LmFloatToE4m3(float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	uint16_t encoded;
	asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(0.0f), "f"(value));
	return((uint8_t)encoded);
#else
	return(LmFloatToUe4m3(fabsf(value)) | (signbit(value) ? 0x80u : 0u));
#endif
}

static __device__ __forceinline__ uint16_t LmFloatPairToE4m3(float low, float high)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	uint16_t encoded;
	asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(high), "f"(low));
	return(encoded);
#else
	return((uint16_t)LmFloatToE4m3(low) | ((uint16_t)LmFloatToE4m3(high) << 8u));
#endif
}

static __device__ __forceinline__ float2 LmHalfPairBitsToFloat2(uint32_t widened)
{
#if defined(__CUDA_ARCH__)
	float low,high;
	asm("cvt.f32.f16 %0, %1;\n" : "=f"(low) : "h"((uint16_t)widened));
	asm("cvt.f32.f16 %0, %1;\n" : "=f"(high) : "h"((uint16_t)(widened >> 16u)));
	return(make_float2(low,high));
#else
	float2 result;
	result.x = __half2float(__ushort_as_half((uint16_t)widened));
	result.y = __half2float(__ushort_as_half((uint16_t)(widened >> 16u)));
	return(result);
#endif
}

static __device__ __forceinline__ float LmE4m3ToFloat(uint8_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	uint32_t widened;
	float result;
	asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
		: "=r"(widened) : "h"((uint16_t)value));
	asm("cvt.f32.f16 %0, %1;\n" : "=f"(result) : "h"((uint16_t)widened));
	return(result);
#else
	return(SparkMathE4m3ToFloat(value));
#endif
}

static __device__ __forceinline__ float2 LmE4m3PairToFloat2(uint16_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	uint32_t widened;
	asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
		: "=r"(widened) : "h"(value));
	return(LmHalfPairBitsToFloat2(widened));
#else
	return(make_float2(SparkMathE4m3ToFloat(value & 0xffu),SparkMathE4m3ToFloat(value >> 8u)));
#endif
}

static __device__ __forceinline__ float2 LmE4m3PairToFloat2Pure(uint16_t value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	uint32_t widened;
	asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n"
		: "=r"(widened) : "h"(value));
	return(LmHalfPairBitsToFloat2(widened));
#else
	return(make_float2(SparkMathE4m3ToFloat(value & 0xffu),SparkMathE4m3ToFloat(value >> 8u)));
#endif
}


static __device__ __forceinline__ uint8_t LmFloatPairToE2m1(float low, float high)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
	uint16_t encoded;
	asm("{\n\t.reg .b8 packed;\n"
		"\tcvt.rn.satfinite.e2m1x2.f32 packed, %1, %2;\n"
		"\tcvt.u16.u8 %0, packed;\n\t}\n"
		: "=h"(encoded) : "f"(high), "f"(low));
	return((uint8_t)(encoded & 0xffu));
#else
	(void)low;
	(void)high;
	return(0u);
#endif
}

static __device__ __forceinline__ float2 LmE2m1PairToFloat(uint8_t packed)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
	uint32_t widened;
	asm("{\n\t.reg .b8 narrow;\n"
		"\tcvt.u8.u32 narrow, %1;\n"
		"\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "r"((uint32_t)packed));
	return(LmHalfPairBitsToFloat2(widened));
#else
	return(make_float2(SparkMathE2m1ToFloat(packed & 15u),SparkMathE2m1ToFloat((packed >> 4u) & 15u)));
#endif
}

static __device__ __forceinline__ float2 LmE2m1PairToFloatPure(uint8_t packed)
{
	return(LmE2m1PairToFloat(packed));
}

static __device__ __forceinline__ uint8_t LmFloatToUe8m0(float value)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
	uint16_t encoded;
	asm("cvt.rn.satfinite.ue8m0x2.f32 %0, %1, %2;\n"
		: "=h"(encoded) : "f"(0.0f), "f"(value));
	return((uint8_t)(encoded >> 8u));
#else
	(void)value;
	return(0u);
#endif
}

static __device__ __forceinline__ float LmUe8m0ToFloat(uint8_t value)
{
	return(SparkMathUe8m0ToFloat(value));
}

static __device__ __forceinline__ float LmE2m1ToFloatPure(uint32_t nibble)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
	uint32_t widened;
	float result;
	asm("{\n\t.reg .b8 narrow;\n"
		"\tcvt.u8.u32 narrow, %1;\n"
		"\tcvt.rn.f16x2.e2m1x2 %0, narrow;\n\t}\n"
		: "=r"(widened) : "r"(nibble & 15u));
	asm("cvt.f32.f16 %0, %1;\n" : "=f"(result) : "h"((uint16_t)widened));
	return(result);
#else
	return(SparkMathE2m1ToFloat(nibble));
#endif
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
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
	const uint32_t bits = (uint32_t)value & 0x7fu;
	const uint16_t clamped = (uint16_t)(bits == 0x7fu ? 0x7eu : bits);
	uint32_t widened;
	float result;
	asm("cvt.rn.f16x2.e4m3x2 %0, %1;\n" : "=r"(widened) : "h"(clamped));
	asm("cvt.f32.f16 %0, %1;\n" : "=f"(result) : "h"((uint16_t)widened));
	return(result);
#else
	return(SparkMathUe4m3ToFloat(value));
#endif
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
