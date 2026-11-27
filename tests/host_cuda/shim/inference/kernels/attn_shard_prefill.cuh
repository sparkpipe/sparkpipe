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
	cudaStream_t stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, partial_rank_stride, rows, stream, fork_positions, slot_offsets);
}

static inline uint32_t LmLatentShardDecodeSplits(uint32_t rows)
{
	return rows >= 16u ? 1u : 16u / rows;
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardSplitLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	uint32_t splits,
	cudaStream_t stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	(void)split_partials;
	(void)split_capacity_floats;
	if (rows == 0u || splits == 0u || heads_per_rank == 0u)
		return cudaErrorInvalidValue;
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, partial_rank_stride, rows, stream, fork_positions, slot_offsets);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardDecodeLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	float qk_scale,
	float *split_partials,
	uint64_t split_capacity_floats,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	(void)split_partials;
	(void)split_capacity_floats;
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, partial_rank_stride, rows, stream, fork_positions, slot_offsets);
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
	if (LmKvKeyWindowBegin(cache) != 0u || LmKvKeyWindowEnd(cache) != LM_KV_SHARD_KEYS_ALL)
		return cudaErrorNotSupported;
	return LmLatentShardPartialOwnHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, heads_per_rank,
		sequence_of_row, context_length, row_position, 0, 0u, 0u, qk_scale, partials, rows, stream);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardAttendHeadsLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	uint32_t total_heads,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t record_heads,
	uint32_t record_copies,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *split_positions = 0,
	const uint32_t *split_offsets = 0,
	uint32_t heads_per_block_limit = 4u,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	return LmLatentShardPartialHeadsLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, total_heads, sequence_of_row,
		context_length, row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, record_heads, record_copies, rows,
		stream, split_positions, split_offsets, heads_per_block_limit, fork_positions, slot_offsets);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardAttendCoopLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *coop_partials,
	uint64_t coop_capacity_floats,
	uint32_t *coop_done,
	uint32_t coop_done_entries,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0,
	uint32_t coop_chunk_tiles = 0u)
{
	(void)coop_partials;
	(void)coop_capacity_floats;
	(void)coop_done;
	(void)coop_done_entries;
	(void)coop_chunk_tiles;
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, rows, stream, fork_positions, slot_offsets);
}

template<class Geometry, class Pages, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentShardAttendLaunch(
	Pages cache,
	const uint16_t *query_bf16,
	uint64_t query_rank_stride,
	uint32_t heads_per_rank,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *row_position,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t dense_limit,
	float qk_scale,
	float *partials,
	uint64_t partial_rank_stride,
	uint32_t rows,
	cudaStream_t stream,
	const uint32_t *fork_positions = 0,
	const uint32_t *slot_offsets = 0)
{
	return LmLatentShardPartialLaunch<Geometry, Pages, LATENT, ROPE>(cache, query_bf16, query_rank_stride, heads_per_rank, sequence_of_row, context_length,
		row_position, selected_positions, selected_count, dense_limit, qk_scale, partials, partial_rank_stride, rows, stream, fork_positions, slot_offsets);
}
