#pragma once

#include <stdint.h>

#include "inference/kernels/dtype.cuh"
#include "inference/kernels/formats/mxfp4.cuh"

#ifdef __CUDACC__
static __global__ void LmRowLanesKernel(const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	uint32_t *__restrict__ row_lane, uint32_t *__restrict__ row_ordinal)
{
	const uint32_t lane = blockIdx.x, begin = sequence_row_begin[lane], end = sequence_row_begin[lane + 1u];
	for ( uint32_t k = begin + threadIdx.x; k < end; k += blockDim.x )
	{
		row_lane[sequence_row_indices[k]] = lane;
		row_ordinal[sequence_row_indices[k]] = k - begin;
	}
}

static __device__ __forceinline__ int32_t LmNgramToken(const int32_t *__restrict__ token_map, uint32_t token)
{
	return token_map != 0 ? token_map[token] : (int32_t)token;
}

static __global__ void LmNgramHashKernel(const uint32_t *__restrict__ token_ids, const int32_t *__restrict__ token_map,
	const uint32_t *__restrict__ positions, const uint32_t *__restrict__ row_lane, const uint32_t *__restrict__ row_ordinal,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices,
	const uint32_t *__restrict__ recurrent_slot, const int32_t *__restrict__ history, uint32_t history_stride,
	const int64_t *__restrict__ multipliers, const int64_t *__restrict__ primes, const int64_t *__restrict__ offsets, int64_t *__restrict__ ids,
	uint32_t rows, uint32_t orders, uint32_t heads, int32_t pad, int32_t block_token, uint32_t block_from)
{
	const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
	if ( row >= rows )
		return;
	const uint32_t lane = row_lane[row], position = positions[row], into_wave = row_ordinal[row];
	const uint32_t *lane_rows = sequence_row_indices + sequence_row_begin[lane];
	const int32_t *carried = history + (uint64_t)recurrent_slot[lane] * history_stride;
	int64_t tokens[4];
	uint32_t blocked = 0u;
	for ( uint32_t shift = 0u; shift <= orders; ++shift )
	{
		int32_t source = -1;
		if ( position >= shift )
			source = shift <= into_wave ? LmNgramToken(token_map, token_ids[lane_rows[into_wave - shift]]) : carried[shift - into_wave - 1u];
		if ( position < shift || (shift >= block_from && source == block_token) )
			blocked = 1u;
		tokens[shift] = blocked != 0u ? pad : source;
	}
	int64_t rolling = tokens[0] * multipliers[0];
	for ( uint32_t order = 1u; order <= orders; ++order )
	{
		rolling ^= tokens[order] * multipliers[order];
		for ( uint32_t head = 0u; head < heads; ++head )
		{
			const uint32_t column = (order - 1u) * heads + head;
			ids[(uint64_t)row * orders * heads + column] = rolling % primes[column] + offsets[column];
		}
	}
}

static __global__ void LmNgramHistoryKernel(const uint32_t *__restrict__ token_ids, const int32_t *__restrict__ token_map,
	const uint32_t *__restrict__ sequence_row_begin, const uint32_t *__restrict__ sequence_row_indices, const uint32_t *__restrict__ recurrent_slot,
	int32_t *__restrict__ history, uint32_t history_stride, uint32_t sequences, uint32_t orders, uint32_t commit)
{
	const uint32_t sequence = blockIdx.x * blockDim.x + threadIdx.x;
	if ( commit == 0u || sequence >= sequences )
		return;
	const uint32_t begin = sequence_row_begin[sequence];
	const uint32_t into_wave = sequence_row_begin[sequence + 1u] - begin - 1u;
	const uint32_t *lane_rows = sequence_row_indices + begin;
	int32_t *slot = history + (uint64_t)recurrent_slot[sequence] * history_stride;
	int32_t carried[4], updated[4];
	for ( uint32_t shift = 0u; shift < orders; ++shift )
		carried[shift] = slot[shift];
	for ( uint32_t shift = 0u; shift < orders; ++shift )
		updated[shift] = shift <= into_wave ? LmNgramToken(token_map, token_ids[lane_rows[into_wave - shift]]) : carried[shift - into_wave - 1u];
	for ( uint32_t shift = 0u; shift < orders; ++shift )
		slot[shift] = updated[shift];
}

static __global__ void LmNgramGatherE8m0Kernel(const int64_t *__restrict__ ids, const uint8_t *__restrict__ payload, const uint8_t *__restrict__ scale,
	uint16_t *__restrict__ embed, uint64_t first_row, uint64_t local_rows, uint32_t columns, uint32_t head_dim)
{
	const uint32_t row = blockIdx.x, column = blockIdx.y;
	const int64_t id = ids[(uint64_t)row * columns + column];
	uint16_t *target = embed + ((uint64_t)row * columns + column) * head_dim;
	const uint32_t owned = id >= (int64_t)first_row && id < (int64_t)(first_row + local_rows);
	const uint64_t local = owned != 0u ? (uint64_t)(id - (int64_t)first_row) : 0u;
	for ( uint32_t c = threadIdx.x; c < head_dim; c += blockDim.x )
		target[c] = owned != 0u ? LmFloatToBf16(LmE4m3ToFloat(payload[local * head_dim + c]) *
			LmE8m0ToFloat(scale[local * (head_dim / 32u) + c / 32u])) : (uint16_t)0u;
}

static __global__ void LmNgramGatherScaledKernel(const int64_t *__restrict__ ids, const uint8_t *__restrict__ payload, const float *__restrict__ scale,
	uint16_t *__restrict__ embed, uint64_t first_row, uint64_t local_rows, uint32_t columns, uint32_t head_dim)
{
	const uint32_t row = blockIdx.x, column = blockIdx.y;
	const int64_t id = ids[(uint64_t)row * columns + column];
	uint16_t *target = embed + ((uint64_t)row * columns + column) * head_dim;
	const uint32_t owned = id >= (int64_t)first_row && id < (int64_t)(first_row + local_rows);
	const uint64_t local = owned != 0u ? (uint64_t)(id - (int64_t)first_row) : 0u;
	const float factor = scale[0];
	for ( uint32_t c = threadIdx.x; c < head_dim; c += blockDim.x )
		target[c] = owned != 0u ? LmFloatToBf16(LmE4m3ToFloat(payload[local * head_dim + c]) * factor) : (uint16_t)0u;
}
#endif
