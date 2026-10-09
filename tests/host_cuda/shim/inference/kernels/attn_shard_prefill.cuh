#pragma once

#include "inference/kernels/attn_shard.cuh"

#define LM_LATENT_SHARD_PREFILL_MIN_ROWS 17u

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream)
{
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, partial_rank_stride, rows, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentGatherPrefillLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *partials,
	uint32_t rows,
	cudaStream_t stream)
{
	return LmLatentShardPartialOwnHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, rows, stream);
}
