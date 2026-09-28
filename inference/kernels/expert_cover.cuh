#pragma once

#include <stdint.h>
#include "sparkpipe/spark_step_verdict.h"

static __device__ __forceinline__ uint32_t LmExpertCoverFirst(
	const uint32_t *layer_words,
	uint32_t cover_stride,
	uint32_t experts)
{
	uint32_t word,first = experts;
	for (word=0u; word<cover_stride && first == experts; word++)
		if ( layer_words[word] != 0u )
			first = word * 32u + (uint32_t)__ffs((int)layer_words[word]) - 1u;
	return(first < experts ? first : experts);
}

static __device__ __forceinline__ uint32_t LmExpertCovered(
	const uint32_t *layer_words,
	uint32_t expert,
	uint32_t experts)
{
	return(expert < experts && (layer_words[expert >> 5u] & (1u << (expert & 31u))) != 0u ? 1u : 0u);
}

static __global__ void LmExpertCoverKernel(
	uint32_t *route_expert,
	const uint32_t *cover,
	uint32_t cover_stride,
	uint32_t experts,
	uint32_t layer,
	uint32_t pack_stride,
	uint32_t ring_capacity,
	uint32_t packed_rows,
	uint32_t *route_log,
	volatile uint32_t *miss)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t *layer_words = cover + (uint64_t)layer * cover_stride;
	uint32_t expert,substitute,slot;
	if ( index >= packed_rows )
		return;
	expert = route_expert[index];
	if ( route_log != 0 )
		route_log[index] = expert;
	if ( LmExpertCovered(layer_words,expert,experts) != 0u )
		return;
	substitute = LmExpertCoverFirst(layer_words,cover_stride,experts);
	if ( substitute >= experts )
		__trap();
	slot = atomicAdd((unsigned int *)(miss + SPARK_STEP_MISS_COUNT),1u);
	if ( slot < ring_capacity )
		miss[SPARK_STEP_MISS_ENTRIES + slot] = layer * pack_stride + (expert < experts ? expert : experts);
	route_expert[index] = substitute;
	miss[SPARK_STEP_MISS_FLAG] = 1u;
}

static __global__ void LmHeadMissPoisonKernel(
	const volatile uint32_t *miss,
	uint64_t *maxloc,
	uint32_t rows)
{
	uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row < rows && miss[SPARK_STEP_MISS_FLAG] != 0u )
		maxloc[row] = UINT64_MAX;
}
