#pragma once
#include <stdint.h>
#include <string.h>
typedef struct { uint16_t raw; } __half;
typedef struct { __half x, y; } __half2;
static inline float __half2float(__half value)
{
	uint32_t sign = (uint32_t)(value.raw >> 15) & 0x1u;
	uint32_t exponent = (uint32_t)(value.raw >> 10) & 0x1fu;
	uint32_t mantissa = (uint32_t)value.raw & 0x3ffu;
	uint32_t bits;
	float out;
	if (exponent == 0u)
		bits = mantissa ? ((sign << 31) | ((127u - 15u + 1u) << 23) | (mantissa << 13))
			: (sign << 31);
	else if (exponent == 31u)
		bits = (sign << 31) | (0xffu << 23) | (mantissa << 13);
	else
		bits = (sign << 31) | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
	memcpy(&out, &bits, sizeof(out));
	return out;
}
static inline __half __float2half(float value)
{
	__half out; uint32_t bits; memcpy(&bits, &value, sizeof(bits));
	uint32_t sign = (bits >> 16) & 0x8000u;
	int32_t exponent = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
	uint32_t mantissa = (bits >> 13) & 0x3ffu;
	if (exponent <= 0) out.raw = (uint16_t)sign;
	else if (exponent >= 31) out.raw = (uint16_t)(sign | 0x7c00u);
	else out.raw = (uint16_t)(sign | ((uint32_t)exponent << 10) | mantissa);
	return out;
}
