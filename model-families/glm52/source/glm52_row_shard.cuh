#ifndef SPARKPIPE_GLM52_ROW_SHARD_CUH
#define SPARKPIPE_GLM52_ROW_SHARD_CUH

#include <stdint.h>
#include "inference/kernels/norm.cuh"
#include "runtime/launch.h"

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

template<uint32_t THREADS>
static void GlmRowShareNorm(const GlmRowShare *row, const uint16_t *reduced, uint16_t *residual, const uint16_t *weight, uint32_t keep_residual,
	uint16_t *normed, uint32_t dimension, float epsilon, cudaStream_t stream)
{
	uint16_t *own = residual + (uint64_t)row->first * dimension;
	if ( row->count == 0u )
		return;
	LM_LAUNCH((LmFusedResidualRmsNormKernel<THREADS,uint16_t>), row->count, THREADS, (dimension + 8u) * sizeof(float), stream,
		reduced, own, weight, keep_residual != 0u ? own : (uint16_t *)0, normed, dimension, dimension, epsilon);
}

#endif
