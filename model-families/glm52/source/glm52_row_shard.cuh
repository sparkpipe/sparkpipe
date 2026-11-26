#ifndef SPARKPIPE_GLM52_ROW_SHARD_CUH
#define SPARKPIPE_GLM52_ROW_SHARD_CUH

#include <stdint.h>
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define GLM_ROW_SHARD_ROWS 256u

struct GlmRowShare
{
	uint32_t share;
	uint32_t first;
	uint32_t count;
};

static inline GlmRowShare GlmRowShareOf(uint32_t rows, uint32_t degree, uint32_t rank)
{
	GlmRowShare row;
	row.share = (rows + degree - 1u) / degree;
	row.first = rank * row.share < rows ? rank * row.share : rows;
	row.count = rows - row.first < row.share ? rows - row.first : row.share;
	return row;
}

template<uint32_t THREADS, uint32_t VEC_SLOTS, bool ROUND_FIRST = false>
__global__ __launch_bounds__(THREADS, 4)
static void GlmResidualRmsNormKernel(const uint16_t *__restrict__ input_bf16, const uint16_t *__restrict__ residual_bf16,
	const uint16_t *__restrict__ weight_bf16, uint16_t *__restrict__ residual_out_bf16, uint16_t *__restrict__ output_bf16,
	uint32_t width, float epsilon)
{
	__shared__ float sq_wave[THREADS * 8u];
	__shared__ float reduction[LmBlockWarps<THREADS>()];
	const uint64_t base = (uint64_t)blockIdx.x * width;
	uint4 staged[VEC_SLOTS];
	float total = 0.0f;
	LmDependentWait();
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t wave_base = slot * THREADS * 8u;
		const uint32_t offset = wave_base + threadIdx.x * 8u;
		if ( offset < width )
		{
			float vals[8];
			staged[slot] = LmRmsLoadAdd8<true>(input_bf16 + base + offset, residual_bf16 != 0 ? residual_bf16 + base + offset : (const uint16_t *)0, vals);
			#pragma unroll
			for ( uint32_t k = 0u; k < 8u; ++k )
				vals[k] *= vals[k];
			LmStoreFloat8(sq_wave + threadIdx.x * 8u, vals);
		}
		else
			staged[slot] = make_uint4(0u, 0u, 0u, 0u);
		__syncthreads();
		#pragma unroll
		for ( uint32_t s = 0u; s < 8u; ++s )
			if ( wave_base + s * THREADS + threadIdx.x < width )
				total += sq_wave[s * THREADS + threadIdx.x];
		if ( slot + 1u < VEC_SLOTS )
			__syncthreads();
	}
	const float scale = SparkMathRmsScale(LmBlockSumOnce<THREADS>(total, reduction), (float)width, epsilon);
	#pragma unroll
	for ( uint32_t slot = 0u; slot < VEC_SLOTS; ++slot )
	{
		const uint32_t offset = (slot * THREADS + threadIdx.x) * 8u;
		if ( offset >= width )
			continue;
		if ( residual_out_bf16 != 0 )
			*(uint4 *)(residual_out_bf16 + base + offset) = staged[slot];
		float vals[8];
		LmLoadBf16x8AsFloat(staged[slot], vals);
		LmRmsStore8(vals, weight_bf16, offset, scale, output_bf16 + base + offset, LmRmsGainOp<ROUND_FIRST>{});
	}
	LmDependentRelease();
}

template<uint32_t THREADS, bool ROUND_FIRST = false>
static inline void GlmResidualRmsNormLaunch(const uint16_t *input_bf16, const uint16_t *residual_bf16, const uint16_t *weight_bf16,
	uint16_t *residual_out_bf16, uint16_t *output_bf16, uint32_t rows, uint32_t width, float epsilon, SparkGpuStream stream)
{
	if ( (width & 7u) == 0u && width <= THREADS * LM_RMS_NORM_STAGED && LmIsAligned16(input_bf16) && LmIsAligned16(output_bf16) &&
		LmIsAligned16(weight_bf16) && (residual_bf16 == 0 || LmIsAligned16(residual_bf16)) &&
		(residual_out_bf16 == 0 || LmIsAligned16(residual_out_bf16)) )
	{
		if ( width <= THREADS * 8u )
			LM_LAUNCH_DEPENDENT((GlmResidualRmsNormKernel<THREADS, 1u, ROUND_FIRST>), rows, THREADS, 0, stream,
				input_bf16, residual_bf16, weight_bf16, residual_out_bf16, output_bf16, width, epsilon);
		else if ( width <= THREADS * 16u )
			LM_LAUNCH_DEPENDENT((GlmResidualRmsNormKernel<THREADS, 2u, ROUND_FIRST>), rows, THREADS, 0, stream,
				input_bf16, residual_bf16, weight_bf16, residual_out_bf16, output_bf16, width, epsilon);
		else if ( width <= THREADS * 24u )
			LM_LAUNCH_DEPENDENT((GlmResidualRmsNormKernel<THREADS, 3u, ROUND_FIRST>), rows, THREADS, 0, stream,
				input_bf16, residual_bf16, weight_bf16, residual_out_bf16, output_bf16, width, epsilon);
		else
			LM_LAUNCH_DEPENDENT((GlmResidualRmsNormKernel<THREADS, 4u, ROUND_FIRST>), rows, THREADS, 0, stream,
				input_bf16, residual_bf16, weight_bf16, residual_out_bf16, output_bf16, width, epsilon);
		return;
	}
	LM_LAUNCH_DEPENDENT((LmFusedResidualRmsNormKernel<THREADS, uint16_t, ROUND_FIRST>), rows, THREADS, (width + 8u) * sizeof(float), stream,
		input_bf16, residual_bf16, weight_bf16, residual_out_bf16, output_bf16, width, width, epsilon);
}

template<uint32_t THREADS, bool ROUND_FIRST = false>
static void GlmRowShareNorm(const GlmRowShare *row, const uint16_t *reduced, uint16_t *residual, const uint16_t *weight, uint32_t keep_residual,
	uint16_t *normed, uint32_t width, float epsilon, SparkGpuStream stream)
{
	uint16_t *own = residual + (uint64_t)row->first * width;
	if ( row->count == 0u )
		return;
	GlmResidualRmsNormLaunch<THREADS, ROUND_FIRST>(reduced, own, weight, keep_residual != 0u ? own : (uint16_t *)0, normed, row->count, width, epsilon, stream);
}

#endif
