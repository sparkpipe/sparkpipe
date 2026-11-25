#pragma once

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCH_RMS_ROWS 4096u
#define BENCH_RMS_DIM 4096u
#define BENCH_RMS_CHUNKS16 (BENCH_RMS_DIM / 16u)
#define BENCH_RMS_EPS 1e-6f

#define BENCH_SWIGLU_ROWS 1024u
#define BENCH_SWIGLU_DIM 8192u
#define BENCH_SWIGLU_ELEMS (BENCH_SWIGLU_ROWS * BENCH_SWIGLU_DIM)
#define BENCH_SWIGLU_CHUNKS16 (BENCH_SWIGLU_ELEMS / 16u)

#define BENCH_SKINNY_BF16_OUT 8192u
#define BENCH_SKINNY_BF16_K 4096u
#define BENCH_SKINNY_BF16_CHUNKS16 (BENCH_SKINNY_BF16_K / 16u)

#define BENCH_SKINNY_NVFP4_OUT 16384u
#define BENCH_SKINNY_NVFP4_K 8192u
#define BENCH_SKINNY_NVFP4_CHUNKS32 (BENCH_SKINNY_NVFP4_K / 64u)

#define BENCH_SKINNY_MXFP4_OUT 16384u
#define BENCH_SKINNY_MXFP4_K 8192u
#define BENCH_SKINNY_MXFP4_CHUNKS32 (BENCH_SKINNY_MXFP4_K / 64u)

#define BENCH_ATTN_HEADS 128u
#define BENCH_ATTN_KV_LEN 4096u
#define BENCH_ATTN_NOPE 512u
#define BENCH_ATTN_ROPE 64u
#define BENCH_ATTN_SCALE (1.0f / 24.0f)

#define BENCH_ROUTER_ROWS 32768u
#define BENCH_ROUTER_COLS 256u

#define BENCH_MX_BLOCKS 1048576u
#define BENCH_MX_ELEMS (BENCH_MX_BLOCKS * 32u)

#define BENCH_TP_CHUNKS16 1048576u
#define BENCH_TP_ELEMS (BENCH_TP_CHUNKS16 * 16u)

#define BENCH_COPY_CHUNKS32 2097152u
#define BENCH_COPY_BYTES ((uint64_t)BENCH_COPY_CHUNKS32 * 32u)

#define BENCH_NUM_KERNELS 13u

#define BENCH_WARMUP_ROUNDS 20u
#define BENCH_BATCH_SIZE 50u
#define BENCH_BATCH_TRIALS 7u
#define BENCH_LATENCY_ROUNDS 50u
#define BENCH_TIMEOUT_NS 10000000000ull

typedef struct BenchMetrics
{
	const char *name;
	double max_abs_err;
	uint32_t max_ulp_err;
	double max_abs_vs_cuda;
	uint32_t max_ulp_vs_cuda;
	uint32_t ulp_limit_fp32;
	uint32_t ulp_limit_cuda;
	double abs_limit_fp32;
	int compared_cuda;
	double gpu_us;
	double lat_us;
	double gbps;
	uint64_t traffic_bytes;
} BenchMetrics;

static inline uint16_t BenchF32ToBf16(float v)
{
	uint32_t u;
	memcpy(&u,&v,sizeof(u));
	u += 0x7fffu + ((u >> 16) & 1u);
	return((uint16_t)(u >> 16));
}

static inline float BenchBf16ToF32(uint16_t h)
{
	uint32_t u = (uint32_t)h << 16;
	float f;
	memcpy(&f,&u,sizeof(f));
	return(f);
}

static inline uint32_t BenchBf16UlpDiff(uint16_t a, uint16_t b)
{
	int32_t oa = (a & 0x8000u) != 0u ? -(int32_t)(a & 0x7fffu) : (int32_t)(a & 0x7fffu);
	int32_t ob = (b & 0x8000u) != 0u ? -(int32_t)(b & 0x7fffu) : (int32_t)(b & 0x7fffu);
	int32_t d = oa - ob;
	return((uint32_t)(d < 0 ? -d : d));
}

static inline uint32_t BenchF32UlpDiff(float fa, float fb)
{
	uint32_t a,b;
	int64_t oa,ob,d;
	memcpy(&a,&fa,sizeof(a));
	memcpy(&b,&fb,sizeof(b));
	if ( (a & 0x7fffffffu) == 0u && (b & 0x7fffffffu) == 0u )
		return(0u);
	oa = (a & 0x80000000u) != 0u ? -(int64_t)(a & 0x7fffffffu) : (int64_t)(a & 0x7fffffffu);
	ob = (b & 0x80000000u) != 0u ? -(int64_t)(b & 0x7fffffffu) : (int64_t)(b & 0x7fffffffu);
	d = oa - ob;
	if ( d < 0 )
		d = -d;
	return(d > 0xffffffffll ? 0xffffffffu : (uint32_t)d);
}

static inline float BenchE2m1ToF32(uint8_t nibble)
{
	static const float lut[16] = {
		0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
		-0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
	};
	return(lut[nibble & 0xfu]);
}

static inline float BenchUe4m3ToF32(uint8_t b)
{
	const uint32_t exp = (b >> 3) & 0xfu;
	const uint32_t mant = b & 0x7u;
	if ( exp == 0u )
		return((float)mant / 512.0f);
	{
		uint32_t bits = ((exp + 120u) << 23) | (mant << 20);
		float f;
		memcpy(&f,&bits,sizeof(f));
		return(f);
	}
}

static inline float BenchUe8m0ToF32(uint8_t b)
{
	uint32_t bits;
	float f;
	if ( b == 255u )
		return(0.0f);
	bits = (b == 0u) ? 0x00400000u : ((uint32_t)b << 23);
	memcpy(&f,&bits,sizeof(f));
	return(f);
}

static inline uint8_t BenchF32ToE4m3Sat(float v)
{
	const uint8_t sign = (v < 0.0f || (v == 0.0f && signbit(v))) ? 0x80u : 0x00u;
	float a = fabsf(v);
	uint8_t best = 0u;
	float best_err = 1e30f;
	uint32_t code;
	if ( a >= 448.0f )
		return((uint8_t)(sign | 0x7eu));
	for (code = 0u; code <= 0x7eu; code++)
	{
		const uint32_t e = (code >> 3) & 0xfu;
		const uint32_t m = code & 0x7u;
		float val;
		if ( e == 0u )
			val = (float)m / 512.0f;
		else
		{
			uint32_t bits = ((e + 120u) << 23) | (m << 20);
			memcpy(&val,&bits,sizeof(val));
		}
		{
			const float err = fabsf(a - val);
			if ( err < best_err || (err == best_err && (code & 1u) == 0u) )
			{
				best_err = err;
				best = (uint8_t)code;
			}
		}
	}
	return((uint8_t)(sign | best));
}

static inline int BenchCompareDouble(const void *a, const void *b)
{
	const double x = *(const double *)a,y = *(const double *)b;
	return(x < y ? -1 : x > y ? 1 : 0);
}

static inline double BenchMedian(double *values, uint32_t count)
{
	qsort(values,(size_t)count,sizeof(double),BenchCompareDouble);
	return(values[count / 2u]);
}

static inline void BenchFillRmsInputs(uint16_t *x, uint16_t *w, uint32_t rows, uint32_t dim)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)rows * dim; i++)
		x[i] = BenchF32ToBf16(((float)((int)((i * 13u + 7u) % 101u) - 50)) * 0.04f);
	for (i = 0u; i < dim; i++)
		w[i] = BenchF32ToBf16(0.5f + ((float)(i % 31u)) * 0.03f);
}

static inline void BenchFillFusedRmsInputs(uint16_t *x, uint16_t *res, uint16_t *w, uint32_t rows, uint32_t dim)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)rows * dim; i++)
	{
		x[i] = BenchF32ToBf16(((float)((int)((i * 13u + 7u) % 101u) - 50)) * 0.02f);
		res[i] = BenchF32ToBf16(((float)((int)((i * 17u + 3u) % 89u) - 44)) * 0.02f);
	}
	for (i = 0u; i < dim; i++)
		w[i] = BenchF32ToBf16(0.5f + ((float)(i % 31u)) * 0.03f);
}

static inline void BenchFillSwigluInputs(uint16_t *gate, uint16_t *up, uint32_t elems)
{
	uint32_t i;
	for (i = 0u; i < elems; i++)
	{
		gate[i] = BenchF32ToBf16(((float)((int)((i * 7u + 3u) % 97u) - 48)) * 0.08f);
		up[i] = BenchF32ToBf16(((float)((int)((i * 11u + 5u) % 53u) - 26)) * 0.05f);
	}
}

static inline void BenchFillSkinnyBf16Inputs(uint16_t *w, uint16_t *a, uint32_t out_dim, uint32_t k)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)out_dim * k; i++)
		w[i] = BenchF32ToBf16(((float)((int)((i * 7u + 1u) % 31u) - 15)) * 0.01f);
	for (i = 0u; i < k; i++)
		a[i] = BenchF32ToBf16(((float)((int)((i * 11u + 3u) % 29u) - 14)) * 0.02f);
}

static inline void BenchFillSkinnyNvfp4Inputs(uint8_t *w, uint8_t *scales, uint16_t *a, float *gscale, uint32_t out_dim, uint32_t k)
{
	const uint32_t c32 = k / 64u;
	uint64_t i;
	*gscale = 0.25f;
	for (i = 0u; i < (uint64_t)out_dim * c32 * 32u; i++)
		w[i] = (uint8_t)((i * 37u + 13u) & 0xffu);
	for (i = 0u; i < (uint64_t)out_dim * c32 * 4u; i++)
		scales[i] = (uint8_t)(0x38u + (i % 16u));
	for (i = 0u; i < k; i++)
		a[i] = BenchF32ToBf16(((float)((int)((i * 11u + 5u) % 29u) - 14)) * 0.02f);
}

static inline void BenchFillSkinnyMxfp4Inputs(uint8_t *w, uint8_t *scales, uint16_t *a, uint32_t out_dim, uint32_t k)
{
	const uint32_t c32 = k / 64u;
	uint64_t i;
	for (i = 0u; i < (uint64_t)out_dim * c32 * 32u; i++)
		w[i] = (uint8_t)((i * 41u + 19u) & 0xffu);
	for (i = 0u; i < (uint64_t)out_dim * c32 * 2u; i++)
		scales[i] = (uint8_t)(123u + (i % 7u));
	scales[0] = 0u;
	scales[1] = 255u;
	for (i = 0u; i < k; i++)
		a[i] = BenchF32ToBf16(((float)((int)((i * 13u + 7u) % 29u) - 14)) * 0.02f);
}

static inline void BenchFillAttnInputs(uint16_t *qn, uint16_t *qr, uint16_t *kn, uint16_t *kr, uint32_t heads, uint32_t kv_len)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)heads * BENCH_ATTN_NOPE; i++)
		qn[i] = BenchF32ToBf16(((float)((int)((i * 7u + 3u) % 41u) - 20)) * 0.05f);
	for (i = 0u; i < (uint64_t)heads * BENCH_ATTN_ROPE; i++)
		qr[i] = BenchF32ToBf16(((float)((int)((i * 11u + 5u) % 37u) - 18)) * 0.05f);
	for (i = 0u; i < (uint64_t)kv_len * BENCH_ATTN_NOPE; i++)
		kn[i] = BenchF32ToBf16(((float)((int)((i * 13u + 1u) % 43u) - 21)) * 0.04f);
	for (i = 0u; i < (uint64_t)kv_len * BENCH_ATTN_ROPE; i++)
		kr[i] = BenchF32ToBf16(((float)((int)((i * 19u + 9u) % 31u) - 15)) * 0.04f);
}

static inline void BenchFillRouterF32Inputs(float *logits, uint32_t rows, uint32_t cols)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)rows * cols; i++)
		logits[i] = ((float)((int)((i * 23u + 11u) % 127u) - 63)) * 0.08f;
}

static inline void BenchFillRouterBf16Inputs(uint16_t *logits, uint32_t rows, uint32_t cols)
{
	uint64_t i;
	for (i = 0u; i < (uint64_t)rows * cols; i++)
		logits[i] = BenchF32ToBf16(((float)((int)((i * 23u + 11u) % 127u) - 63)) * 0.08f);
}

static inline void BenchFillMxQuantInputs(uint16_t *src, uint32_t elems)
{
	uint32_t i;
	for (i = 0u; i < elems; i++)
		src[i] = BenchF32ToBf16(((float)((int)((i * 19u + 5u) % 113u) - 56)) * 0.125f);
	for (i = 0u; i < 32u; i++)
		src[i] = 0u;
}

static inline void BenchFillTpF32Inputs(float *src, uint32_t elems)
{
	uint32_t i;
	for (i = 0u; i < elems; i++)
		src[i] = ((float)((int)((i * 29u + 7u) % 199u) - 99)) * 0.03125f;
}

static inline void BenchFillTpBf16Inputs(uint16_t *src, uint32_t elems)
{
	uint32_t i;
	for (i = 0u; i < elems; i++)
		src[i] = BenchF32ToBf16(((float)((int)((i * 31u + 13u) % 197u) - 98)) * 0.03125f);
}

static inline void BenchFillCopyInputs(uint32_t *src, uint64_t words)
{
	uint64_t i;
	for (i = 0u; i < words; i++)
		src[i] = (uint32_t)((i * 2654435761ull) ^ (i >> 7));
}

static inline void BenchVerifyRms(const uint16_t *x, const uint16_t *w, const uint16_t *out, uint32_t rows, uint32_t dim, float eps,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t r,j;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < rows; r++)
	{
		double s = 0.0;
		float inv;
		for (j = 0u; j < dim; j++)
		{
			const float v = BenchBf16ToF32(x[(uint64_t)r * dim + j]);
			s += (double)v * (double)v;
		}
		inv = 1.0f / sqrtf((float)(s / (double)dim) + eps);
		for (j = 0u; j < dim; j++)
		{
			const float ref = BenchBf16ToF32(x[(uint64_t)r * dim + j]) * inv * BenchBf16ToF32(w[j]);
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[(uint64_t)r * dim + j];
			const double err = fabs((double)BenchBf16ToF32(got_h) - (double)ref);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyFusedRms(const uint16_t *x, const uint16_t *res_in, const uint16_t *w, const uint16_t *res_out,
	const uint16_t *out, uint32_t rows, uint32_t dim, float eps, double *max_abs, uint32_t *max_ulp)
{
	uint32_t r,j;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < rows; r++)
	{
		double s = 0.0;
		float inv;
		for (j = 0u; j < dim; j++)
		{
			const uint64_t idx = (uint64_t)r * dim + j;
			const float sum = BenchBf16ToF32(x[idx]) + BenchBf16ToF32(res_in[idx]);
			const uint16_t ref_res = BenchF32ToBf16(sum);
			const uint32_t rulp = BenchBf16UlpDiff(res_out[idx],ref_res);
			const double rerr = fabs((double)BenchBf16ToF32(res_out[idx]) - (double)sum);
			if ( rerr > ma )
				ma = rerr;
			if ( rulp > mu )
				mu = rulp;
			s += (double)sum * (double)sum;
		}
		inv = 1.0f / sqrtf((float)(s / (double)dim) + eps);
		for (j = 0u; j < dim; j++)
		{
			const uint64_t idx = (uint64_t)r * dim + j;
			const float hv = BenchBf16ToF32(BenchF32ToBf16(BenchBf16ToF32(x[idx]) + BenchBf16ToF32(res_in[idx])));
			const float ref = hv * inv * BenchBf16ToF32(w[j]);
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[idx];
			const double err = fabs((double)BenchBf16ToF32(got_h) - (double)ref);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifySwiglu(const uint16_t *gate, const uint16_t *up, const uint16_t *out, uint32_t elems,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t i;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (i = 0u; i < elems; i++)
	{
		const float g = BenchBf16ToF32(gate[i]);
		const float u = BenchBf16ToF32(up[i]);
		const float ref = (g / (1.0f + expf(-g))) * u;
		const uint16_t ref_h = BenchF32ToBf16(ref);
		const uint16_t got_h = out[i];
		const double err = fabs((double)BenchBf16ToF32(got_h) - (double)ref);
		const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
		if ( err > ma )
			ma = err;
		if ( ulp > mu )
			mu = ulp;
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifySkinnyBf16(const uint16_t *w, const uint16_t *a, const uint16_t *out, uint32_t out_dim, uint32_t k,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t r,j;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < out_dim; r++)
	{
		double acc = 0.0;
		for (j = 0u; j < k; j++)
			acc += (double)BenchBf16ToF32(w[(uint64_t)r * k + j]) * (double)BenchBf16ToF32(a[j]);
		{
			const float ref = (float)acc;
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[r];
			const double err = fabs((double)BenchBf16ToF32(got_h) - acc);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifySkinnyNvfp4(const uint8_t *w, const uint8_t *scales, const uint16_t *a, float gscale, const uint16_t *out,
	uint32_t out_dim, uint32_t k, double *max_abs, uint32_t *max_ulp)
{
	const uint32_t c32 = k / 64u;
	uint32_t r,c,g,b;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < out_dim; r++)
	{
		double acc = 0.0;
		for (c = 0u; c < c32; c++)
		{
			const uint8_t *wp = w + ((uint64_t)r * c32 + c) * 32u;
			const uint8_t *sp = scales + ((uint64_t)r * c32 + c) * 4u;
			for (g = 0u; g < 4u; g++)
			{
				const float sg = BenchUe4m3ToF32(sp[g]);
				double grp = 0.0;
				for (b = 0u; b < 8u; b++)
				{
					const uint8_t byte = wp[g * 8u + b];
					const float wlo = BenchE2m1ToF32(byte & 0xfu);
					const float whi = BenchE2m1ToF32((byte >> 4) & 0xfu);
					const float alo = BenchBf16ToF32(a[c * 64u + g * 16u + b * 2u]);
					const float ahi = BenchBf16ToF32(a[c * 64u + g * 16u + b * 2u + 1u]);
					grp += (double)wlo * (double)alo + (double)whi * (double)ahi;
				}
				acc += (double)sg * grp;
			}
		}
		acc *= (double)gscale;
		{
			const float ref = (float)acc;
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[r];
			const double err = fabs((double)BenchBf16ToF32(got_h) - acc);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifySkinnyMxfp4(const uint8_t *w, const uint8_t *scales, const uint16_t *a, const uint16_t *out,
	uint32_t out_dim, uint32_t k, double *max_abs, uint32_t *max_ulp)
{
	const uint32_t c32 = k / 64u;
	uint32_t r,c,g,b;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < out_dim; r++)
	{
		double acc = 0.0;
		for (c = 0u; c < c32; c++)
		{
			const uint8_t *wp = w + ((uint64_t)r * c32 + c) * 32u;
			const uint8_t *sp = scales + ((uint64_t)r * c32 + c) * 2u;
			for (g = 0u; g < 2u; g++)
			{
				const float sg = BenchUe8m0ToF32(sp[g]);
				double grp = 0.0;
				for (b = 0u; b < 16u; b++)
				{
					const uint8_t byte = wp[g * 16u + b];
					const float wlo = BenchE2m1ToF32(byte & 0xfu);
					const float whi = BenchE2m1ToF32((byte >> 4) & 0xfu);
					const float alo = BenchBf16ToF32(a[c * 64u + g * 32u + b * 2u]);
					const float ahi = BenchBf16ToF32(a[c * 64u + g * 32u + b * 2u + 1u]);
					grp += (double)wlo * (double)alo + (double)whi * (double)ahi;
				}
				acc += (double)sg * grp;
			}
		}
		{
			const float ref = (float)acc;
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[r];
			const double err = fabs((double)BenchBf16ToF32(got_h) - acc);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyAttn(const uint16_t *qn, const uint16_t *qr, const uint16_t *kn, const uint16_t *kr, const uint16_t *out,
	uint32_t heads, uint32_t kv_len, float scale, double *max_abs, uint32_t *max_ulp)
{
	double ma = 0.0;
	uint32_t mu = 0u;
	uint32_t h,t,d;
	float *logits = (float *)malloc((size_t)kv_len * sizeof(float));
	float *qn_f = (float *)malloc((size_t)BENCH_ATTN_NOPE * sizeof(float));
	float *qr_f = (float *)malloc((size_t)BENCH_ATTN_ROPE * sizeof(float));
	double *acc = (double *)malloc((size_t)BENCH_ATTN_NOPE * sizeof(double));
	for (h = 0u; h < heads; h++)
	{
		double mval = -1e30,lval = 0.0;
		for (d = 0u; d < BENCH_ATTN_NOPE; d++)
		{
			qn_f[d] = BenchBf16ToF32(qn[(uint64_t)h * BENCH_ATTN_NOPE + d]);
			acc[d] = 0.0;
		}
		for (d = 0u; d < BENCH_ATTN_ROPE; d++)
			qr_f[d] = BenchBf16ToF32(qr[(uint64_t)h * BENCH_ATTN_ROPE + d]);
		for (t = 0u; t < kv_len; t++)
		{
			double dot = 0.0;
			const uint16_t *knp = kn + (uint64_t)t * BENCH_ATTN_NOPE;
			const uint16_t *krp = kr + (uint64_t)t * BENCH_ATTN_ROPE;
			for (d = 0u; d < BENCH_ATTN_NOPE; d++)
				dot += (double)qn_f[d] * (double)BenchBf16ToF32(knp[d]);
			for (d = 0u; d < BENCH_ATTN_ROPE; d++)
				dot += (double)qr_f[d] * (double)BenchBf16ToF32(krp[d]);
			logits[t] = (float)(dot * (double)scale);
			if ( (double)logits[t] > mval )
				mval = (double)logits[t];
		}
		for (t = 0u; t < kv_len; t++)
		{
			const double p = exp((double)logits[t] - mval);
			const uint16_t *knp = kn + (uint64_t)t * BENCH_ATTN_NOPE;
			lval += p;
			for (d = 0u; d < BENCH_ATTN_NOPE; d++)
				acc[d] += p * (double)BenchBf16ToF32(knp[d]);
		}
		for (d = 0u; d < BENCH_ATTN_NOPE; d++)
		{
			const float ref = (float)(acc[d] / lval);
			const uint16_t ref_h = BenchF32ToBf16(ref);
			const uint16_t got_h = out[(uint64_t)h * BENCH_ATTN_NOPE + d];
			const double err = fabs((double)BenchBf16ToF32(got_h) - (double)ref);
			const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	free(acc);
	free(qr_f);
	free(qn_f);
	free(logits);
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyRouterF32(const float *logits, const float *probs, uint32_t rows, uint32_t cols,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t r,c;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < rows; r++)
	{
		const float *in = logits + (uint64_t)r * cols;
		const float *out = probs + (uint64_t)r * cols;
		double rmax = -1e30,sum = 0.0;
		for (c = 0u; c < cols; c++)
			if ( (double)in[c] > rmax )
				rmax = (double)in[c];
		for (c = 0u; c < cols; c++)
			sum += exp((double)in[c] - rmax);
		for (c = 0u; c < cols; c++)
		{
			const float ref = (float)(exp((double)in[c] - rmax) / sum);
			const double err = fabs((double)out[c] - (double)ref);
			const uint32_t ulp = BenchF32UlpDiff(out[c],ref);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyRouterBf16(const uint16_t *logits, const float *probs, uint32_t rows, uint32_t cols,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t r,c;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (r = 0u; r < rows; r++)
	{
		const uint16_t *in = logits + (uint64_t)r * cols;
		const float *out = probs + (uint64_t)r * cols;
		double rmax = -1e30,sum = 0.0;
		for (c = 0u; c < cols; c++)
		{
			const double v = (double)BenchBf16ToF32(in[c]);
			if ( v > rmax )
				rmax = v;
		}
		for (c = 0u; c < cols; c++)
			sum += exp((double)BenchBf16ToF32(in[c]) - rmax);
		for (c = 0u; c < cols; c++)
		{
			const float ref = (float)(exp((double)BenchBf16ToF32(in[c]) - rmax) / sum);
			const double err = fabs((double)out[c] - (double)ref);
			const uint32_t ulp = BenchF32UlpDiff(out[c],ref);
			if ( err > ma )
				ma = err;
			if ( ulp > mu )
				mu = ulp;
		}
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyMxQuant(const uint16_t *src, const uint8_t *fp8, const uint8_t *scales, uint32_t blocks,
	double *max_abs, uint32_t *max_ulp)
{
	uint32_t b,i;
	uint32_t mismatches = 0u;
	for (b = 0u; b < blocks; b++)
	{
		float amax = 0.0f;
		uint32_t e8m0 = 0u;
		float inv_scale = 0.0f;
		for (i = 0u; i < 32u; i++)
		{
			const float v = fabsf(BenchBf16ToF32(src[(uint64_t)b * 32u + i]));
			if ( v > amax )
				amax = v;
		}
		if ( amax > 0.0f )
		{
			float sc = amax * (1.0f / 448.0f);
			uint32_t bits;
			memcpy(&bits,&sc,sizeof(bits));
			e8m0 = (bits + 0x007fffffu) >> 23;
			if ( e8m0 > 254u )
				e8m0 = 254u;
			{
				uint32_t ibits = (254u - e8m0) << 23;
				memcpy(&inv_scale,&ibits,sizeof(inv_scale));
			}
		}
		if ( scales[b] != (uint8_t)e8m0 )
			mismatches++;
		for (i = 0u; i < 32u; i++)
		{
			const float s = BenchBf16ToF32(src[(uint64_t)b * 32u + i]) * inv_scale;
			const uint8_t ref = BenchF32ToE4m3Sat(s);
			const uint8_t got = fp8[(uint64_t)b * 32u + i];
			if ( got != ref )
			{
				const uint8_t gn = got & 0x7fu;
				const uint8_t rn = ref & 0x7fu;
				if ( !(gn == 0u && rn == 0u) )
					mismatches++;
			}
		}
	}
	*max_abs = (double)mismatches;
	*max_ulp = mismatches;
}

static inline void BenchVerifyTpF32ToBf16(const float *src, const uint16_t *out, uint32_t elems, double *max_abs, uint32_t *max_ulp)
{
	uint32_t i;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (i = 0u; i < elems; i++)
	{
		const uint16_t ref_h = BenchF32ToBf16(src[i]);
		const uint16_t got_h = out[i];
		const double err = fabs((double)BenchBf16ToF32(got_h) - (double)BenchBf16ToF32(ref_h));
		const uint32_t ulp = BenchBf16UlpDiff(got_h,ref_h);
		if ( err > ma )
			ma = err;
		if ( ulp > mu )
			mu = ulp;
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyTpBf16ToF32(const uint16_t *src, const float *out, uint32_t elems, double *max_abs, uint32_t *max_ulp)
{
	uint32_t i;
	double ma = 0.0;
	uint32_t mu = 0u;
	for (i = 0u; i < elems; i++)
	{
		const float ref = BenchBf16ToF32(src[i]);
		const double err = fabs((double)out[i] - (double)ref);
		const uint32_t ulp = BenchF32UlpDiff(out[i],ref);
		if ( err > ma )
			ma = err;
		if ( ulp > mu )
			mu = ulp;
	}
	*max_abs = ma;
	*max_ulp = mu;
}

static inline void BenchVerifyCopy(const uint32_t *src, const uint32_t *dst, uint64_t words, double *max_abs, uint32_t *max_ulp)
{
	uint64_t i;
	uint32_t mismatches = 0u;
	for (i = 0u; i < words; i++)
		if ( src[i] != dst[i] )
			mismatches++;
	*max_abs = (double)mismatches;
	*max_ulp = mismatches;
}

static inline int BenchDumpBytes(const char *dir, const char *name, const void *data, size_t bytes)
{
	char path[1024];
	FILE *f;
	if ( dir == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.bin",dir,name);
	f = fopen(path,"wb");
	if ( f == 0 )
		return(-1);
	if ( fwrite(data,1u,bytes,f) != bytes )
	{
		fclose(f);
		return(-1);
	}
	fclose(f);
	return(0);
}

static inline int BenchDumpBf16(const char *dir, const char *name, const uint16_t *data, size_t count)
{
	return(BenchDumpBytes(dir,name,data,count * sizeof(uint16_t)));
}

static inline int BenchCompareBf16File(const char *dir, const char *name, const uint16_t *got, size_t count,
	double *max_abs, uint32_t *max_ulp)
{
	char path[1024];
	FILE *f;
	uint16_t *ref;
	size_t i;
	double ma = 0.0;
	uint32_t mu = 0u;
	if ( dir == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.bin",dir,name);
	f = fopen(path,"rb");
	if ( f == 0 )
		return(-1);
	ref = (uint16_t *)malloc(count * sizeof(uint16_t));
	if ( ref == 0 )
	{
		fclose(f);
		return(-1);
	}
	if ( fread(ref,sizeof(uint16_t),count,f) != count )
	{
		free(ref);
		fclose(f);
		return(-1);
	}
	fclose(f);
	for (i = 0u; i < count; i++)
	{
		const double err = fabs((double)BenchBf16ToF32(got[i]) - (double)BenchBf16ToF32(ref[i]));
		const uint32_t ulp = BenchBf16UlpDiff(got[i],ref[i]);
		if ( err > ma )
			ma = err;
		if ( ulp > mu )
			mu = ulp;
	}
	free(ref);
	*max_abs = ma;
	*max_ulp = mu;
	return(1);
}

static inline int BenchCompareF32File(const char *dir, const char *name, const float *got, size_t count,
	double *max_abs, uint32_t *max_ulp)
{
	char path[1024];
	FILE *f;
	float *ref;
	size_t i;
	double ma = 0.0;
	uint32_t mu = 0u;
	if ( dir == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.bin",dir,name);
	f = fopen(path,"rb");
	if ( f == 0 )
		return(-1);
	ref = (float *)malloc(count * sizeof(float));
	if ( ref == 0 )
	{
		fclose(f);
		return(-1);
	}
	if ( fread(ref,sizeof(float),count,f) != count )
	{
		free(ref);
		fclose(f);
		return(-1);
	}
	fclose(f);
	for (i = 0u; i < count; i++)
	{
		const double err = fabs((double)got[i] - (double)ref[i]);
		const uint32_t ulp = BenchF32UlpDiff(got[i],ref[i]);
		if ( err > ma )
			ma = err;
		if ( ulp > mu )
			mu = ulp;
	}
	free(ref);
	*max_abs = ma;
	*max_ulp = mu;
	return(1);
}

static inline int BenchCompareBytesFile(const char *dir, const char *name, const uint8_t *got, size_t bytes,
	double *max_abs, uint32_t *max_ulp)
{
	char path[1024];
	FILE *f;
	uint8_t *ref;
	size_t i;
	uint32_t mismatches = 0u;
	if ( dir == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.bin",dir,name);
	f = fopen(path,"rb");
	if ( f == 0 )
		return(-1);
	ref = (uint8_t *)malloc(bytes);
	if ( ref == 0 )
	{
		fclose(f);
		return(-1);
	}
	if ( fread(ref,1u,bytes,f) != bytes )
	{
		free(ref);
		fclose(f);
		return(-1);
	}
	fclose(f);
	for (i = 0u; i < bytes; i++)
		if ( got[i] != ref[i] )
			mismatches++;
	free(ref);
	*max_abs = (double)mismatches;
	*max_ulp = mismatches;
	return(1);
}

static inline void BenchPrintRow(const char *label, const BenchMetrics *m)
{
	if ( m->compared_cuda )
		printf("RESULT %-10s %-30s gpu_us=%8.2f lat_us=%8.2f bw_gbs=%7.2f max_err_fp32=%.6f max_ulp_fp32=%u max_err_cuda=%.6f max_ulp_cuda=%u\n",
			label,m->name,m->gpu_us,m->lat_us,m->gbps,m->max_abs_err,m->max_ulp_err,m->max_abs_vs_cuda,m->max_ulp_vs_cuda);
	else
		printf("RESULT %-10s %-30s gpu_us=%8.2f lat_us=%8.2f bw_gbs=%7.2f max_err_fp32=%.6f max_ulp_fp32=%u\n",
			label,m->name,m->gpu_us,m->lat_us,m->gbps,m->max_abs_err,m->max_ulp_err);
}

static inline int BenchCheckParity(const BenchMetrics *m)
{
	const uint32_t fp32_lim = m->ulp_limit_fp32 ? m->ulp_limit_fp32 : 2u;
	const uint32_t cuda_lim = m->ulp_limit_cuda ? m->ulp_limit_cuda : 2u;
	if ( m->abs_limit_fp32 > 0.0 )
	{
		if ( m->max_abs_err > m->abs_limit_fp32 )
			return(0);
	}
	else if ( m->max_ulp_err > fp32_lim )
		return(0);
	if ( m->compared_cuda && m->max_ulp_vs_cuda > cuda_lim )
		return(0);
	return(1);
}
