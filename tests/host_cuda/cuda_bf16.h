#pragma once
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
