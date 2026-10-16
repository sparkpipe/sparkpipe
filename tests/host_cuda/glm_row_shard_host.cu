#include "tests/host_cuda/lm_host_cuda.cuh"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "inference/kernels/dtype.cuh"
#define __CUDACC__ 1
#include "runtime/launch.h"
#include "tests/host_cuda/lm_host_threads.cuh"
LmHostDim3 blockDim,gridDim;
#undef __shared__
#define __shared__
#include "inference/kernels/norm.cuh"
float lm_norm_shared[LM_HOST_SHARED_BYTES / sizeof(float)];
#include "model-families/glm52/source/glm52_row_shard.cuh"

#define T_THREADS 256u
#define T_HIDDEN 6144u
#define T_DEGREE 16u
#define T_LAYERS 3u
#define T_EPSILON 1e-5f

static uint32_t lcg = 0x2545f491u;

static float Uniform(void)
{
	lcg = lcg * 1664525u + 1013904223u;
	return (float)((lcg >> 8) & 0xffffffu) / 8388608.0f - 1.0f;
}

static float Bf16Float(uint16_t value)
{
	uint32_t bits = (uint32_t)value << 16u;
	float out;
	memcpy(&out, &bits, sizeof(out));
	return out;
}

static uint16_t Bf16Rne(float value)
{
	uint32_t bits;
	memcpy(&bits, &value, sizeof(bits));
	if ( (bits & 0x7fffffffu) > 0x7f800000u )
		return (uint16_t)((bits >> 16u) | 0x40u);
	bits += 0x7fffu + ((bits >> 16u) & 1u);
	return (uint16_t)(bits >> 16u);
}

static void Fill(uint16_t *out, uint64_t count, float base, float scale)
{
	for ( uint64_t index = 0u; index < count; index++ )
		out[index] = Bf16Rne(base + scale * Uniform());
}

static void FillWide(uint16_t *out, uint64_t count)
{
	for ( uint64_t index = 0u; index < count; index++ )
	{
		const float magnitude = ldexpf(1.0f, (int)((lcg >> 11) % 21u) - 16);
		out[index] = Bf16Rne(magnitude * Uniform());
	}
}

static void Reduce(const uint16_t *partials, uint64_t stride, uint32_t first_row, uint32_t rows, uint32_t reverse, uint16_t *out)
{
	for ( uint64_t index = 0u; index < (uint64_t)rows * T_HIDDEN; index++ )
	{
		float sum = 0.0f;
		for ( uint32_t step = 0u; step < T_DEGREE; step++ )
		{
			const uint32_t peer = reverse != 0u ? T_DEGREE - 1u - step : step;
			sum += Bf16Float(partials[(uint64_t)peer * stride + (uint64_t)first_row * T_HIDDEN + index]);
		}
		out[index] = Bf16Rne(sum);
	}
}

static uint64_t Run(uint32_t rows, uint32_t reverse)
{
	const uint32_t share = (rows + T_DEGREE - 1u) / T_DEGREE, padded = share * T_DEGREE;
	const uint64_t row_bytes = (uint64_t)T_HIDDEN * sizeof(uint16_t), stride = (uint64_t)padded * T_HIDDEN;
	uint16_t *weights = (uint16_t *)malloc((2u * T_LAYERS + 1u) * row_bytes);
	uint16_t *hidden = (uint16_t *)malloc(rows * row_bytes), *residual = (uint16_t *)calloc(rows, row_bytes);
	uint16_t *normed = (uint16_t *)malloc(rows * row_bytes), *reduced = (uint16_t *)malloc(rows * row_bytes);
	uint16_t *partials = (uint16_t *)malloc(T_DEGREE * stride * sizeof(uint16_t)), *gathered = (uint16_t *)malloc(padded * row_bytes);
	uint16_t *rank_residual[T_DEGREE], *row_share = (uint16_t *)malloc(share * row_bytes), *row_normed[T_DEGREE];
	uint64_t mismatches = 0u;
	uint32_t covered = 0u;
	for ( uint32_t rank = 0u; rank < T_DEGREE; rank++ )
	{
		const GlmRowShare row = GlmRowShareOf(rows, T_DEGREE, rank);
		if ( row.share != share || row.first + row.count > rows || (row.count != 0u && row.first != rank * share) )
		{
			fprintf(stderr, "rows=%u rank=%u share=%u first=%u count=%u: the share is not the gather slot\n", rows, rank, row.share, row.first, row.count);
			exit(1);
		}
		covered += row.count;
	}
	if ( covered != rows )
	{
		fprintf(stderr, "rows=%u: the shares cover %u rows\n", rows, covered);
		exit(1);
	}
	Fill(weights, (2u * T_LAYERS + 1u) * T_HIDDEN, 1.0f, 0.5f);
	Fill(hidden, (uint64_t)rows * T_HIDDEN, 0.0f, 2.0f);
	LM_LAUNCH((LmFusedResidualRmsNormKernel<T_THREADS,uint16_t>), rows, T_THREADS, (T_HIDDEN + 8u) * sizeof(float), 0,
		hidden, residual, weights, residual, normed, T_HIDDEN, T_HIDDEN, T_EPSILON);
	for ( uint32_t rank = 0u; rank < T_DEGREE; rank++ )
	{
		rank_residual[rank] = (uint16_t *)malloc(rows * row_bytes);
		row_normed[rank] = (uint16_t *)malloc(share * row_bytes);
		memcpy(rank_residual[rank], residual, rows * row_bytes);
	}
	for ( uint32_t phase = 0u; phase < 2u * T_LAYERS; phase++ )
	{
		const uint16_t *weight = weights + (uint64_t)(phase + 1u) * T_HIDDEN;
		const uint32_t keep = phase + 1u < 2u * T_LAYERS ? 1u : 0u;
		FillWide(partials, T_DEGREE * stride);
		for ( uint32_t peer = 0u; peer < T_DEGREE; peer++ )
			for ( uint64_t index = (uint64_t)rows * T_HIDDEN; index < stride; index++ )
				partials[peer * stride + index] = 0x7fc1u;
		Reduce(partials, stride, 0u, rows, 0u, reduced);
		LM_LAUNCH((LmFusedResidualRmsNormKernel<T_THREADS,uint16_t>), rows, T_THREADS, (T_HIDDEN + 8u) * sizeof(float), 0,
			reduced, residual, weight, keep != 0u ? residual : (uint16_t *)0, normed, T_HIDDEN, T_HIDDEN, T_EPSILON);
		for ( uint32_t rank = 0u; rank < T_DEGREE; rank++ )
		{
			const GlmRowShare row = GlmRowShareOf(rows, T_DEGREE, rank);
			Reduce(partials, stride, rank * share, share, reverse, row_share);
			for ( uint64_t index = 0u; index < (uint64_t)share * T_HIDDEN; index++ )
				row_normed[rank][index] = 0xffc3u;
			GlmRowShareNorm<T_THREADS>(&row, row_share, rank_residual[rank], weight, keep, row_normed[rank], T_HIDDEN, T_EPSILON, 0);
		}
		for ( uint32_t rank = 0u; rank < T_DEGREE; rank++ )
			memcpy(gathered + (uint64_t)rank * share * T_HIDDEN, row_normed[rank], share * row_bytes);
		for ( uint64_t index = 0u; index < (uint64_t)rows * T_HIDDEN; index++ )
			mismatches += gathered[index] != normed[index];
	}
	for ( uint32_t rank = 0u; rank < T_DEGREE; rank++ )
	{
		const GlmRowShare row = GlmRowShareOf(rows, T_DEGREE, rank);
		for ( uint64_t index = (uint64_t)row.first * T_HIDDEN; index < (uint64_t)(row.first + row.count) * T_HIDDEN; index++ )
			mismatches += rank_residual[rank][index] != residual[index];
		free(rank_residual[rank]);
		free(row_normed[rank]);
	}
	free(weights); free(hidden); free(residual); free(normed); free(reduced); free(partials); free(gathered); free(row_share);
	return mismatches;
}

int main(void)
{
	static const uint32_t cases[] = { 37u, 300u, 1024u };
	for ( uint32_t index = 0u; index < sizeof(cases) / sizeof(cases[0]); index++ )
	{
		const uint64_t exact = Run(cases[index], 0u);
		printf("rows=%u row-sharded vs replicated: %llu mismatched elements\n", cases[index], (unsigned long long)exact);
		if ( exact != 0u )
			return 1;
	}
	{
		const uint64_t reordered = Run(300u, 1u);
		printf("rows=300 reverse peer order: %llu mismatched elements\n", (unsigned long long)reordered);
		if ( reordered == 0u )
		{
			fprintf(stderr, "a changed summation order went undetected\n");
			return 1;
		}
	}
	printf("glm row shard: PASS\n");
	return 0;
}
