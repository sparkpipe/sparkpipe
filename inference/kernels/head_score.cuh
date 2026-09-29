#pragma once

#include <math.h>
#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "include/sparkpipe/spark_score_dump.h"

#ifndef LM_WARP_LANES
#define LM_WARP_LANES 32u
#endif

#define LM_HEAD_SCORE_THREADS 256u
#define LM_HEAD_SCORE_ROWS_PER_PASS 16u

template<uint32_t ROWS_PER_PASS>
__global__ void LmHeadScoreLogitsKernel(
	const uint16_t *__restrict__ hidden_bf16,
	const uint16_t *__restrict__ head_bf16,
	float *__restrict__ logits,
	uint32_t rows,
	uint32_t dimension,
	uint32_t vocabulary)
{
	float accumulator[ROWS_PER_PASS];
	uint32_t warps = blockDim.x / LM_WARP_LANES;
	uint32_t warp = threadIdx.x / LM_WARP_LANES;
	uint32_t lane = threadIdx.x % LM_WARP_LANES;
	uint32_t id = blockIdx.x * warps + warp;
	uint32_t first = blockIdx.y * ROWS_PER_PASS;
	uint32_t count, row, element, offset;
	const uint16_t *weight;
	float value;
	if ( id >= vocabulary || first >= rows )
		return;
	count = rows - first < ROWS_PER_PASS ? rows - first : ROWS_PER_PASS;
	weight = head_bf16 + (uint64_t)id * dimension;
	for (row = 0u; row < ROWS_PER_PASS; row++)
		accumulator[row] = 0.0f;
	for (element = lane; element < dimension; element += LM_WARP_LANES)
	{
		value = LmBf16ToFloat(weight[element]);
		for (row = 0u; row < ROWS_PER_PASS; row++)
			if ( row < count )
				accumulator[row] = fmaf(LmBf16ToFloat(hidden_bf16[(uint64_t)(first + row) * dimension + element]), value, accumulator[row]);
	}
	for (offset = LM_WARP_LANES / 2u; offset > 0u; offset >>= 1u)
		for (row = 0u; row < ROWS_PER_PASS; row++)
			accumulator[row] += __shfl_xor_sync(0xffffffffu, accumulator[row], (int)offset);
	if ( lane == 0u )
		for (row = 0u; row < count; row++)
			logits[(uint64_t)(first + row) * vocabulary + id] = accumulator[row];
}

static __device__ __forceinline__ uint32_t LmHeadScoreBetter(float value, uint32_t id, float best_value, uint32_t best_id)
{
	return(value > best_value || (value == best_value && id < best_id) ? 1u : 0u);
}

template<uint32_t THREADS>
__global__ void LmHeadScoreRowsKernel(
	const float *__restrict__ logits,
	uint32_t rows,
	uint32_t vocabulary,
	uint32_t id_base,
	const uint32_t *__restrict__ probe_offsets,
	const uint32_t *__restrict__ probe_local,
	float *__restrict__ probe_logits,
	SparkScoreDumpStats *__restrict__ stats)
{
	__shared__ float shared_value[THREADS];
	__shared__ uint32_t shared_id[THREADS];
	__shared__ double shared_sum[THREADS];
	uint32_t row = blockIdx.x;
	uint32_t threads = blockDim.x;
	uint32_t tid = threadIdx.x;
	uint32_t index, stride, rank, bad, best_id, previous_id, id;
	float maximum, best_value, previous_value, value;
	double sum;
	const float *x;
	SparkScoreDumpStats *out;
	if ( row >= rows )
		return;
	x = logits + (uint64_t)row * vocabulary;
	out = stats + row;
	maximum = -INFINITY;
	bad = 0u;
	for (index = tid; index < vocabulary; index += threads)
	{
		value = x[index];
		if ( !isfinite(value) )
			bad = 1u;
		else if ( value > maximum )
			maximum = value;
	}
	shared_value[tid] = maximum;
	shared_id[tid] = bad;
	__syncthreads();
	for (stride = threads / 2u; stride > 0u; stride >>= 1u)
	{
		if ( tid < stride )
		{
			if ( shared_value[tid + stride] > shared_value[tid] )
				shared_value[tid] = shared_value[tid + stride];
			shared_id[tid] |= shared_id[tid + stride];
		}
		__syncthreads();
	}
	maximum = shared_value[0];
	bad = shared_id[0];
	__syncthreads();
	if ( bad != 0u )
	{
		for (index = tid; index < SPARK_SCORE_DUMP_TOP_K; index += threads)
		{
			out->top_ids[index] = SPARK_SCORE_DUMP_NO_TOKEN;
			out->top_logits[index] = 0.0f;
		}
		if ( tid == 0u )
		{
			out->local_max = 0.0f;
			out->local_sum_exp = 0.0;
			out->flags = SPARK_SCORE_DUMP_ROW_NONFINITE;
		}
		for (index = probe_offsets[row] + tid; index < probe_offsets[row + 1u]; index += threads)
			probe_logits[index] = 0.0f;
		return;
	}
	sum = 0.0;
	for (index = tid; index < vocabulary; index += threads)
		sum += exp((double)x[index] - (double)maximum);
	shared_sum[tid] = sum;
	__syncthreads();
	for (stride = threads / 2u; stride > 0u; stride >>= 1u)
	{
		if ( tid < stride )
			shared_sum[tid] += shared_sum[tid + stride];
		__syncthreads();
	}
	if ( tid == 0u )
	{
		out->local_max = maximum;
		out->local_sum_exp = shared_sum[0];
		out->flags = 0u;
	}
	previous_value = INFINITY;
	previous_id = 0u;
	for (rank = 0u; rank < SPARK_SCORE_DUMP_TOP_K; rank++)
	{
		best_value = -INFINITY;
		best_id = SPARK_SCORE_DUMP_NO_TOKEN;
		for (index = tid; index < vocabulary; index += threads)
		{
			value = x[index];
			id = id_base + index;
			if ( (value < previous_value || (value == previous_value && id > previous_id)) && LmHeadScoreBetter(value, id, best_value, best_id) != 0u )
			{
				best_value = value;
				best_id = id;
			}
		}
		shared_value[tid] = best_value;
		shared_id[tid] = best_id;
		__syncthreads();
		for (stride = threads / 2u; stride > 0u; stride >>= 1u)
		{
			if ( tid < stride && LmHeadScoreBetter(shared_value[tid + stride], shared_id[tid + stride], shared_value[tid], shared_id[tid]) != 0u )
			{
				shared_value[tid] = shared_value[tid + stride];
				shared_id[tid] = shared_id[tid + stride];
			}
			__syncthreads();
		}
		previous_value = shared_value[0];
		previous_id = shared_id[0];
		if ( tid == 0u )
		{
			out->top_ids[rank] = previous_id;
			out->top_logits[rank] = previous_id == SPARK_SCORE_DUMP_NO_TOKEN ? 0.0f : previous_value;
		}
		__syncthreads();
	}
	for (index = probe_offsets[row] + tid; index < probe_offsets[row + 1u]; index += threads)
		probe_logits[index] = x[probe_local[index]];
}

static inline uint32_t LmHeadScoreLogitBlocks(uint32_t vocabulary)
{
	uint32_t warps = LM_HEAD_SCORE_THREADS / LM_WARP_LANES;
	return((vocabulary + warps - 1u) / warps);
}

static inline uint32_t LmHeadScoreRowPasses(uint32_t rows)
{
	return((rows + LM_HEAD_SCORE_ROWS_PER_PASS - 1u) / LM_HEAD_SCORE_ROWS_PER_PASS);
}
