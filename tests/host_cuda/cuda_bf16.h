#pragma once
// Minimal bf16 stub for the host harness. The shared model-family kernels
// (model-families/common/include/sparkpipe/spark_lm_kernels.cuh) use exactly
// the conversions below; anything else breaks the build here rather than
// silently picking up a different definition. Rounding is round-to-nearest-
// even in both directions, matching the device converts the kernels bind at
// nvcc time (__float2bfloat16 / __floats2bfloat162_rn are _rn).
#include <stdint.h>
#include <string.h>

typedef struct { uint16_t raw; } __nv_bfloat16;
typedef struct { __nv_bfloat16 x, y; } __nv_bfloat162;

static inline float __bfloat162float(__nv_bfloat16 value)
{
	uint32_t bits = (uint32_t)value.raw << 16;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static inline __nv_bfloat16 __float2bfloat16(float value)
{
	// Round to nearest even: add the truncated-half plus the low bit of the
	// kept mantissa, then truncate. Infinities and NaNs carry through.
	__nv_bfloat16 out;
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	out.raw = (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
	return out;
}

static inline float2 __bfloat1622float2(__nv_bfloat162 value)
{
	float2 out;
	out.x = __bfloat162float(value.x);
	out.y = __bfloat162float(value.y);
	return out;
}

static inline __nv_bfloat162 __floats2bfloat162_rn(float low, float high)
{
	__nv_bfloat162 out;
	out.x = __float2bfloat16(low);
	out.y = __float2bfloat16(high);
	return out;
}
