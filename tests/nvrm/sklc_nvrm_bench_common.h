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

static inline int BenchDumpBf16(const char *dir, const char *name, const uint16_t *data, size_t count)
{
	char path[1024];
	FILE *f;
	if ( dir == 0 )
		return(0);
	snprintf(path,sizeof(path),"%s/%s.bin",dir,name);
	f = fopen(path,"wb");
	if ( f == 0 )
		return(-1);
	if ( fwrite(data,sizeof(uint16_t),count,f) != count )
	{
		fclose(f);
		return(-1);
	}
	fclose(f);
	return(0);
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

static inline void BenchPrintRow(const char *label, const BenchMetrics *m)
{
	if ( m->compared_cuda )
		printf("RESULT %-10s %-22s gpu_us=%8.2f lat_us=%8.2f bw_gbs=%7.2f max_err_fp32=%.6f max_ulp_fp32=%u max_err_cuda=%.6f max_ulp_cuda=%u\n",
			label,m->name,m->gpu_us,m->lat_us,m->gbps,m->max_abs_err,m->max_ulp_err,m->max_abs_vs_cuda,m->max_ulp_vs_cuda);
	else
		printf("RESULT %-10s %-22s gpu_us=%8.2f lat_us=%8.2f bw_gbs=%7.2f max_err_fp32=%.6f max_ulp_fp32=%u\n",
			label,m->name,m->gpu_us,m->lat_us,m->gbps,m->max_abs_err,m->max_ulp_err);
}
