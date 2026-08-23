#pragma once

/* Minimal fp16 stub so host syntax gates can parse the kernel tree without a
 * CUDA toolkit. dtype.cuh takes exactly __half2float and __ushort_as_half from
 * <cuda_fp16.h>; if a kernel starts using more, the gate must fail here rather
 * than silently pick up different semantics. Conversion mirrors
 * tests/host_cuda/cuda_fp16.h. */
#include <stdint.h>
#include <string.h>

typedef struct { uint16_t raw; } __half;

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
		bits = (sign << 31) | (((exponent + 127u - 15u) << 23)) | (mantissa << 13);
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static inline __half __ushort_as_half(uint16_t bits)
{
	__half out;
	out.raw = bits;
	return out;
}

static inline uint16_t __half_as_ushort(__half value)
{
	return value.raw;
}
