#pragma once

/* Minimal BF16 stub so host syntax gates can parse device sources that use
 * the __nv_bfloat16 intrinsics (the drafter backend) without a CUDA toolkit.
 * The conversions are exact: a BF16 is the top half of an F32. If a source
 * needs more of <cuda_bf16.h> than this, extend it here explicitly rather
 * than letting a real toolkit silently change the parse. */
#include <stdint.h>
#include <string.h>

typedef struct { uint16_t raw; } __nv_bfloat16;

static inline float __bfloat162float(__nv_bfloat16 value)
{
	uint32_t bits = (uint32_t)value.raw << 16u;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static inline __nv_bfloat16 __float2bfloat16(float value)
{
	__nv_bfloat16 out;
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	/* Round to nearest even on the truncation boundary, matching the
	 * intrinsic's contract. */
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	out.raw = (uint16_t)(bits >> 16u);
	return out;
}
