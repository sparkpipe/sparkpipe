#pragma once

#include "inference/kernels/attn.cuh"
#include <stdint.h>

#define LM_ROPE_HEADS_THREADS 256u
#define LM_ROPE_HEADS_MIN_ROWS 4u

template<class Geometry, uint32_t THREADS, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentRopeHeadsSplitKernel(
	const uint16_t *__restrict__ query_latent_bf16,
	const uint16_t *__restrict__ query_rope_bf16,
	LmKvView cache,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ selected_positions,
	uint32_t selected_count,
	uint32_t partitions,
	float qk_scale,
	float *__restrict__ partials,
	const uint32_t *__restrict__ row_position)
{
	constexpr uint32_t VALUES = LM_LATENT_ATTN_SPLIT_VALUES(LATENT + ROPE, THREADS);
	static_assert(LATENT <= 8u * THREADS, "the latent must fit the per-thread accumulator");
	__shared__ float group_reduction[LM_LATENT_ATTN_SPLIT_GROUP][THREADS / LM_WARP_LANES];
	__shared__ float group_total[LM_LATENT_ATTN_SPLIT_GROUP];
	__shared__ float shared_query[HEADS][LATENT + ROPE];
	const uint32_t row = blockIdx.x, partition = blockIdx.y, sequence = sequence_of_row[row];
	float accumulator[HEADS][8], running_max[HEADS], running_sum[HEADS], scores[LM_LATENT_ATTN_SPLIT_GROUP];
	uint32_t head, index, element, step, position_count, first_position, last_position, partition_span;
	uint64_t partial_base;
	LmDependentRelease();
	if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
	{
		LmKvReportRequiredAccessFailure(cache, !LmKvViewIsConfigured(cache) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		running_max[head] = -INFINITY;
		running_sum[head] = 0.0f;
		#pragma unroll
		for (index = 0u; index < 8u; ++index)
			accumulator[head][index] = 0.0f;
	}
	for (index = threadIdx.x; index < HEADS * (LATENT + ROPE); index += THREADS)
	{
		head = index / (LATENT + ROPE);
		element = index % (LATENT + ROPE);
		shared_query[head][element] = element < LATENT
			? LmBf16ToFloat(query_latent_bf16[((uint64_t)row * HEADS + head) * LATENT + element])
			: LmBf16ToFloat(query_rope_bf16[((uint64_t)row * HEADS + head) * ROPE + element - LATENT]);
	}
	__syncthreads();
	position_count = selected_positions != 0 ? selected_count : context_length[sequence];
	if (ROW_INVARIANT && selected_positions == 0 && row_position != 0 && row_position[row] + 1u < position_count)
		position_count = row_position[row] + 1u;
	partition_span = (position_count + partitions - 1u) / partitions;
	first_position = partition * partition_span;
	last_position = first_position + partition_span < position_count ? first_position + partition_span : position_count;
	step = first_position;
	while (step < last_position)
	{
		const uint8_t *slots[LM_LATENT_ATTN_SPLIT_GROUP];
		uint16_t values[LM_LATENT_ATTN_SPLIT_GROUP][VALUES];
		uint32_t count = 0u, missing = 0u, group, value;
		while (count < LM_LATENT_ATTN_SPLIT_GROUP && step < last_position)
		{
			uint32_t position = selected_positions != 0 ? selected_positions[(row * selected_count) + step] : step;
			const uint8_t *slot;
			++step;
			if (row_position != 0 && position > row_position[row])
				continue;
			slot = LmKvSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
			if (slot == 0)
			{
				missing = 1u;
				break;
			}
			slots[count++] = slot;
		}
		#pragma unroll
		for (group = 0u; group < LM_LATENT_ATTN_SPLIT_GROUP; ++group)
			#pragma unroll
			for (value = 0u; value < VALUES; ++value)
			{
				index = threadIdx.x + value * THREADS;
				values[group][value] = group < count && index < LATENT + ROPE ? ((const uint16_t *)slots[group])[index] : (uint16_t)0u;
			}
		#pragma unroll
		for (head = 0u; head < HEADS; ++head)
		{
			#pragma unroll
			for (group = 0u; group < LM_LATENT_ATTN_SPLIT_GROUP; ++group)
			{
				scores[group] = 0.0f;
				#pragma unroll
				for (value = 0u; value < VALUES; ++value)
				{
					index = threadIdx.x + value * THREADS;
					if (index < LATENT + ROPE)
						scores[group] += shared_query[head][index] * LmBf16ToFloat(values[group][value]);
				}
			}
			LmBlockSumGroup<THREADS, LM_LATENT_ATTN_SPLIT_GROUP>(scores, group_reduction, group_total);
			#pragma unroll
			for (group = 0u; group < LM_LATENT_ATTN_SPLIT_GROUP; ++group)
			{
				float score, scaled_previous, scaled_current, previous_max;
				if (group >= count)
					continue;
				score = scores[group] * qk_scale;
				previous_max = running_max[head];
				running_max[head] = fmaxf(running_max[head], score);
				scaled_previous = __expf(previous_max - running_max[head]);
				scaled_current = __expf(score - running_max[head]);
				running_sum[head] = (running_sum[head] * scaled_previous) + scaled_current;
				#pragma unroll
				for (index = 0u; index < 8u; ++index)
				{
					element = (index * THREADS) + threadIdx.x;
					if (index < VALUES && element < LATENT)
						accumulator[head][index] = (accumulator[head][index] * scaled_previous) + (scaled_current * LmBf16ToFloat(values[group][index]));
				}
			}
		}
		if (missing != 0u)
			break;
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		partial_base = (((uint64_t)row * HEADS + head) * partitions + partition) * (LATENT + 2u);
		if (threadIdx.x == 0u)
		{
			partials[partial_base] = running_max[head];
			partials[partial_base + 1u] = running_sum[head];
		}
		#pragma unroll
		for (index = 0u; index < 8u; ++index)
		{
			element = (index * THREADS) + threadIdx.x;
			if (element < LATENT)
				partials[partial_base + 2u + element] = accumulator[head][index];
		}
	}
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT, uint32_t HEADS>
static inline cudaError_t LmLatentRopeHeadsSplitShape(const uint16_t *query_latent_bf16, const uint16_t *query_rope_bf16, LmKvView cache, const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *selected_positions, uint32_t selected_count, uint32_t partitions, float qk_scale, uint16_t *output_bf16, float *split_partials, const uint32_t *row_position, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmLatentRopeHeadsSplitKernel<Geometry, THREADS, HEADS, LATENT, ROPE, ROW_INVARIANT>), dim3(rows, partitions), THREADS, 0, stream, query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, split_partials, row_position);
	if (cudaPeekAtLastError() != cudaSuccess)
		return cudaPeekAtLastError();
	LM_LAUNCH_DEPENDENT((LmLatentAttentionDecodeSplitCombineKernel<THREADS, LATENT>), dim3(rows, HEADS), THREADS, 0, stream, split_partials, output_bf16, HEADS, partitions);
	return cudaPeekAtLastError();
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT = false>
static inline cudaError_t LmLatentRopeHeadsSplitLaunch(
	const uint16_t *query_latent_bf16,
	const uint16_t *query_rope_bf16,
	LmKvView cache,
	const uint32_t *sequence_of_row,
	const uint32_t *context_length,
	const uint32_t *selected_positions,
	uint32_t selected_count,
	uint32_t heads,
	float qk_scale,
	uint16_t *output_bf16,
	const uint32_t *row_position,
	uint32_t rows,
	uint32_t position_bound,
	uint32_t split_context_threshold,
	float *split_partials,
	uint32_t split_partial_blocks,
	uint32_t multiprocessor_count,
	cudaStream_t stream)
{
	uint32_t blocks = rows * heads, wanted, partitions;
	wanted = multiprocessor_count == 0u || blocks == 0u ? 1u : (multiprocessor_count * LM_LATENT_ATTN_SPLIT_CTAS_PER_SM + (ROW_INVARIANT ? heads : blocks) - 1u) / (ROW_INVARIANT ? heads : blocks);
	partitions = wanted > LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ? LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS : wanted < 1u ? 1u : wanted;
	if (LmLatentAttentionContextSplits(position_bound, split_context_threshold) == 0u || split_partials == 0 || partitions < 2u || blocks == 0u || (uint64_t)blocks * partitions > split_partial_blocks || THREADS != LM_ROPE_HEADS_THREADS || rows < LM_ROPE_HEADS_MIN_ROWS)
		return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
	if (heads == 4u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 4u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	if (heads == 2u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 2u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	if (heads == 1u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 1u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
}
