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
struct LmRowsTileStage
{
	static constexpr uint32_t kWeights = (LM_ROWS_TILE_N * LM_ROWS_TILE_K + THREADS - 1u) / THREADS;
	static constexpr uint32_t kInputs = (ROWS * LM_ROWS_TILE_K + THREADS - 1u) / THREADS;
};

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmRowsTileFetch(const LmRowsTileOperand *operand, uint32_t first_index, uint32_t first_row, uint32_t k0, float *weights, float *inputs)
{
	uint32_t slot, element, index, row;
	#pragma unroll
	for (slot = 0u; slot < LmRowsTileStage<THREADS, ROWS>::kWeights; slot++)
	{
		element = threadIdx.x + slot * THREADS;
		index = first_index + element / LM_ROWS_TILE_K;
		weights[slot] = element < LM_ROWS_TILE_N * LM_ROWS_TILE_K && index < operand->count ? LmBf16ToFloat(operand->weight[((uint64_t)(operand->ids != 0 ? operand->ids[index] : index) * operand->weight_stride) + k0 + element % LM_ROWS_TILE_K]) : 0.0f;
	}
	#pragma unroll
	for (slot = 0u; slot < LmRowsTileStage<THREADS, ROWS>::kInputs; slot++)
	{
		element = threadIdx.x + slot * THREADS;
		row = first_row + element / LM_ROWS_TILE_K;
		inputs[slot] = element < ROWS * LM_ROWS_TILE_K && row < operand->rows ? LmBf16ToFloat(operand->input[((uint64_t)row * operand->input_stride) + k0 + element % LM_ROWS_TILE_K]) : 0.0f;
	}
}

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmRowsTileStore(float (*weight_tile)[LM_ROWS_TILE_K + 1u], float (*input_tile)[LM_ROWS_TILE_K], const float *weights, const float *inputs)
{
	uint32_t slot, element;
	#pragma unroll
	for (slot = 0u; slot < LmRowsTileStage<THREADS, ROWS>::kWeights; slot++)
	{
		element = threadIdx.x + slot * THREADS;
		if ( element < LM_ROWS_TILE_N * LM_ROWS_TILE_K )
			weight_tile[element / LM_ROWS_TILE_K][element % LM_ROWS_TILE_K] = weights[slot];
	}
	#pragma unroll
	for (slot = 0u; slot < LmRowsTileStage<THREADS, ROWS>::kInputs; slot++)
	{
		element = threadIdx.x + slot * THREADS;
		if ( element < ROWS * LM_ROWS_TILE_K )
			input_tile[element / LM_ROWS_TILE_K][element % LM_ROWS_TILE_K] = inputs[slot];
	}
}

template<uint32_t THREADS, uint32_t ROWS>
static __device__ __forceinline__ void LmRowsTileDot(const LmRowsTileOperand *operand, float (*weight_tile)[LM_ROWS_TILE_K + 1u], float (*input_tile)[LM_ROWS_TILE_K], uint32_t first_index, uint32_t first_row, float *total)
{
	constexpr uint32_t per_thread = ROWS / (THREADS / LM_ROWS_TILE_N);
	static_assert(THREADS % LM_ROWS_TILE_N == 0u && ROWS % (THREADS / LM_ROWS_TILE_N) == 0u, "row tiles must divide evenly across thread groups");
	const uint32_t lane = threadIdx.x % LM_ROWS_TILE_N, group = threadIdx.x / LM_ROWS_TILE_N;
	float weights[LmRowsTileStage<THREADS, ROWS>::kWeights], inputs[LmRowsTileStage<THREADS, ROWS>::kInputs];
	uint32_t k0, k, j;
	for (j = 0u; j < per_thread; ++j)
		total[j] = 0.0f;
	if ( operand->depth != 0u )
		LmRowsTileFetch<THREADS, ROWS>(operand, first_index, first_row, 0u, weights, inputs);
	for (k0 = 0u; k0 < operand->depth; k0 += LM_ROWS_TILE_K)
	{
		__syncthreads();
		LmRowsTileStore<THREADS, ROWS>(weight_tile, input_tile, weights, inputs);
		__syncthreads();
		if ( k0 + LM_ROWS_TILE_K < operand->depth )
			LmRowsTileFetch<THREADS, ROWS>(operand, first_index, first_row, k0 + LM_ROWS_TILE_K, weights, inputs);
		for (k = 0u; k < LM_ROWS_TILE_K; ++k)
			for (j = 0u; j < per_thread; ++j)
				total[j] += input_tile[group * per_thread + j][k] * weight_tile[lane][k];
	}
}
