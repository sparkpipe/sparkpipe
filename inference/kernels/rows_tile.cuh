#pragma once

#include "inference/kernels/dtype.cuh"
#include <stdint.h>

#define LM_ROWS_TILE_N 64u
#define LM_ROWS_TILE_K 64u

typedef struct LmRowsTileOperand
{
	const uint16_t *input;
	const uint16_t *weight;
	const uint32_t *ids;
	uint64_t input_stride;
	uint64_t weight_stride;
	uint32_t rows;
	uint32_t count;
	uint32_t depth;
}
LmRowsTileOperand;

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmRowsTileLoad(const LmRowsTileOperand *operand, float (*weight_tile)[LM_ROWS_TILE_K + 1u], float (*input_tile)[LM_ROWS_TILE_K], uint32_t first_index, uint32_t first_row, uint32_t k0)
{
	uint32_t element, index, row;
	for (element = threadIdx.x; element < LM_ROWS_TILE_N * LM_ROWS_TILE_K; element += THREADS)
	{
		index = first_index + element / LM_ROWS_TILE_K;
		weight_tile[element / LM_ROWS_TILE_K][element % LM_ROWS_TILE_K] = index < operand->count ? LmBf16ToFloat(operand->weight[((uint64_t)(operand->ids != 0 ? operand->ids[index] : index) * operand->weight_stride) + k0 + element % LM_ROWS_TILE_K]) : 0.0f;
	}
	for (element = threadIdx.x; element < ROWS * LM_ROWS_TILE_K; element += THREADS)
	{
		row = first_row + element / LM_ROWS_TILE_K;
		input_tile[element / LM_ROWS_TILE_K][element % LM_ROWS_TILE_K] = row < operand->rows ? LmBf16ToFloat(operand->input[((uint64_t)row * operand->input_stride) + k0 + element % LM_ROWS_TILE_K]) : 0.0f;
	}
}

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmRowsTileDot(const LmRowsTileOperand *operand, float (*weight_tile)[LM_ROWS_TILE_K + 1u], float (*input_tile)[LM_ROWS_TILE_K], uint32_t first_index, uint32_t first_row, float *total)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(THREADS % LM_ROWS_TILE_N == 0u && ROWS % (THREADS / LM_ROWS_TILE_N) == 0u, "row tiles must divide evenly across thread groups");
	const uint32_t lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	uint32_t k0, k, j;
	for (j = 0u; j < per_thread; ++j)
		total[j] = 0.0f;
	for (k0 = 0u; k0 < operand->depth; k0 += LM_ROWS_TILE_K)
	{
		__syncthreads();
		LmRowsTileLoad<THREADS, ROWS>(operand, weight_tile, input_tile, first_index, first_row, k0);
		__syncthreads();
		for (k = 0u; k < LM_ROWS_TILE_K; ++k)
			for (j = 0u; j < per_thread; ++j)
				total[j] += input_tile[group * per_thread + j][k] * weight_tile[lane][k];
	}
}
