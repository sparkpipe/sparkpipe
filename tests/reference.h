// Reference GEMM and format decoders. Obviously correct, deliberately slow.
//
// This is the oracle the fast kernel is checked against. It shares no code with
// kernels/ on purpose: a reference that reuses the implementation's helpers
// validates that they are self-consistent, not that they are right. Every
// formula here is written independently from the format definition, and where
// the two agree that agreement means something.
//
// It runs on a host with no CUDA. That is the point - the sparkdev can find a
// numerics bug on a laptop, and the first run on real hardware is a comparison
// rather than an investigation.

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#define REF_MAX_M 64
#define REF_MAX_N 256
#define REF_MAX_K 512

// -- format decoders, written from the format definitions, not from kernels/ --

static float ref_bf16(uint16_t bits)
{
	uint32_t wide = (uint32_t)bits << 16;
	float value;
	memcpy(&value,&wide,4);
	return(value);
}

// E4M3: sign(1) exponent(4, bias 7) mantissa(3), no infinities, 0x7f is NaN.
static float ref_e4m3(uint8_t code)
{
	int32_t sign = (code >> 7) & 1, exponent = (code >> 3) & 15, mantissa = code & 7;
	float magnitude;
	if ( exponent == 0 )
		magnitude = (float)mantissa / 8.0f * 0.015625f;
	else
		magnitude = (1.0f + (float)mantissa / 8.0f) * powf(2.0f,(float)(exponent - 7));
	return(sign ? -magnitude : magnitude);
}

// E2M1: the eight representable magnitudes, stated rather than computed.
static float ref_e2m1(uint8_t nibble)
{
	static const float magnitude[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
	return((nibble & 8u) ? -magnitude[nibble & 7u] : magnitude[nibble & 7u]);
}

// Signed n-bit two's complement at an arbitrary bit offset. Written as an
// explicit bit loop so it cannot share a shift bug with the kernel's version.
static int32_t ref_code(const uint8_t *base, uint32_t index, uint32_t bits)
{
	uint32_t start = index * bits, i;
	int32_t value = 0;
	for (i = 0; i < bits; ++i)
	{
		uint32_t bit = start + i;
		if ( (base[bit >> 3] >> (bit & 7)) & 1u )
			value |= (1 << i);
	}
	if ( value & (1 << (bits - 1)) )
		value -= (1 << bits);
	return(value);
}

// -- reference GEMM -----------------------------------------------------------
//
// One group, row-major A, column-major B, FP32 accumulate, BF16 out. No tiling,
// no pipeline, no swizzle - the whole point is that there is nothing here to be
// clever about.

typedef float (*ref_element)(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t scale_group);

static float ref_element_bf16(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t group)
{
	(void)scale; (void)group;
	return(ref_bf16(*(const uint16_t *)(plane + ((uint64_t)row * pitch_bits / 8u) + (k * 2u))));
}

static float ref_element_e4m3(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t group)
{
	const uint8_t *base = plane + ((uint64_t)row * pitch_bits / 8u);
	return(ref_e4m3(base[k]) * scale[k / group]);
}

static float ref_element_e2m1(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t group)
{
	const uint8_t *base = plane + ((uint64_t)row * pitch_bits / 8u);
	uint8_t byte = base[k >> 1];
	return(ref_e2m1((k & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 15u)) * scale[k / group]);
}

static float ref_element_int7(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t group)
{
	return((float)ref_code(plane + ((uint64_t)row * pitch_bits / 8u),k,7u) * scale[k / group]);
}

static float ref_element_int6(const uint8_t *plane, uint32_t row, uint32_t k, uint32_t pitch_bits, const float *scale, uint32_t group)
{
	return((float)ref_code(plane + ((uint64_t)row * pitch_bits / 8u),k,6u) * scale[k / group]);
}

// C[m][n] = sum_k A[m][k] * B[n][k], accumulated in FP32 and rounded to BF16
// exactly once at the end - which is what the fast kernel must also do, and a
// common place for the two to disagree if the fast path rounds per K tile.
static void ref_gemm(uint16_t *out, uint32_t m_count, uint32_t n_count, uint32_t k_count,
	const uint8_t *a_plane, uint32_t a_pitch_bits, ref_element a_get, const float *a_scale, uint32_t a_group,
	const uint8_t *b_plane, uint32_t b_pitch_bits, ref_element b_get, const float *b_scale, uint32_t b_group)
{
	uint32_t m,n,k;
	for (m = 0; m < m_count; ++m)
		for (n = 0; n < n_count; ++n)
		{
			float total = 0.0f;
			uint32_t bits;
			for (k = 0; k < k_count; ++k)
				total += a_get(a_plane,m,k,a_pitch_bits,a_scale + (m * (k_count / a_group ? k_count / a_group : 1u)),a_group)
					* b_get(b_plane,n,k,b_pitch_bits,b_scale + (n * (k_count / b_group ? k_count / b_group : 1u)),b_group);
			memcpy(&bits,&total,4);
			out[(m * n_count) + n] = (uint16_t)((bits + 0x7fffu + ((bits >> 16u) & 1u)) >> 16u);
		}
}
