#pragma once

#include "inference/kernels/attn.cuh"

#define LM_LATENT_ROWS_WARPS 8u
#define LM_LATENT_ROWS_THREADS (LM_LATENT_ROWS_WARPS * LM_WARP_LANES)
#define LM_LATENT_ROWS_HEAD_GROUP 4u

template<uint32_t THREADS, uint32_t WIDTH>
struct LmLatentRowsLayout
{
	static constexpr uint32_t kWarps = THREADS / LM_WARP_LANES;
	static constexpr uint32_t kValues = (WIDTH + THREADS - 1u) / THREADS;
	static constexpr uint32_t kSlots = kValues * kWarps;
};

template<uint32_t THREADS, uint32_t WIDTH>
static __device__ __forceinline__ float LmLatentRowsBlockScore(const float *query, const float *key, uint32_t lane)
{
	typedef LmLatentRowsLayout<THREADS, WIDTH> Layout;
	float partial[Layout::kWarps], pair[4], quad[2], single, send, keep;
	uint32_t warp, value, index;
	#pragma unroll
	for (warp = 0u; warp < Layout::kWarps; ++warp)
	{
		partial[warp] = 0.0f;
		#pragma unroll
		for (value = 0u; value < Layout::kValues; ++value)
		{
			if (value * THREADS + warp * LM_WARP_LANES + lane < WIDTH)
				partial[warp] += query[value * Layout::kWarps + warp] * key[value * Layout::kWarps + warp];
		}
	}
	#pragma unroll
	for (index = 0u; index < 4u; ++index)
	{
		send = (lane & 16u) != 0u ? partial[index] : partial[4u + index];
		keep = (lane & 16u) != 0u ? partial[4u + index] : partial[index];
		pair[index] = keep + __shfl_xor_sync(0xffffffffu, send, 16);
	}
	#pragma unroll
	for (index = 0u; index < 2u; ++index)
	{
		send = (lane & 8u) != 0u ? pair[index] : pair[2u + index];
		keep = (lane & 8u) != 0u ? pair[2u + index] : pair[index];
		quad[index] = keep + __shfl_xor_sync(0xffffffffu, send, 8);
	}
	send = (lane & 4u) != 0u ? quad[0] : quad[1];
	keep = (lane & 4u) != 0u ? quad[1] : quad[0];
	single = keep + __shfl_xor_sync(0xffffffffu, send, 4);
	single = single + __shfl_xor_sync(0xffffffffu, single, 2);
	single = single + __shfl_xor_sync(0xffffffffu, single, 1);
	single = single + __shfl_xor_sync(0xffffffffu, single, 16);
	single = single + __shfl_xor_sync(0xffffffffu, single, 8);
	return single + __shfl_xor_sync(0xffffffffu, single, 4);
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE, uint32_t HEAD_GROUP>
__global__ __launch_bounds__(LM_LATENT_ROWS_THREADS, 1)
void LmLatentAttentionRowsSplitKernel(
	const uint16_t *__restrict__ query_latent_bf16,
	const uint16_t *__restrict__ query_rope_bf16,
	LmKvView cache,
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ selected_positions,
	uint32_t selected_count,
	uint32_t heads,
	uint32_t partitions,
	float qk_scale,
	float *__restrict__ partials,
	const uint32_t *__restrict__ row_position,
	uint32_t rows)
{
	typedef LmLatentRowsLayout<THREADS, LATENT + ROPE> Layout;
	static_assert(THREADS == 8u * LM_WARP_LANES, "the score tree emulates an eight-warp block sum");
	static_assert(Layout::kValues * THREADS <= 8u * THREADS, "the latent must fit the emulated accumulator");
	float query[HEAD_GROUP][Layout::kSlots], accumulator[HEAD_GROUP][Layout::kSlots], key[Layout::kSlots];
	float running_max[HEAD_GROUP], running_sum[HEAD_GROUP];
	uint32_t lane = threadIdx.x % LM_WARP_LANES, row = blockIdx.x * LM_LATENT_ROWS_WARPS + threadIdx.x / LM_WARP_LANES;
	uint32_t partition = blockIdx.y, first_head = blockIdx.z * HEAD_GROUP, head, value, warp, slot_index, element;
	uint32_t sequence, position_count, partition_span, first_position, last_position, step;
	uint64_t partial_base;
	if (row >= rows || partition >= partitions)
		return;
	sequence = sequence_of_row[row];
	if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
	{
		if (lane == 0u)
			LmKvReportRequiredAccessFailure(cache, !LmKvViewIsConfigured(cache) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	position_count = selected_positions != 0 ? selected_count : context_length[sequence];
	if (selected_positions == 0 && row_position != 0 && row_position[row] + 1u < position_count)
		position_count = row_position[row] + 1u;
	partition_span = (position_count + partitions - 1u) / partitions;
	first_position = partition * partition_span;
	last_position = first_position + partition_span;
	if (last_position > position_count)
		last_position = position_count;
	#pragma unroll
	for (head = 0u; head < HEAD_GROUP; ++head)
	{
		running_max[head] = -INFINITY;
		running_sum[head] = 0.0f;
		#pragma unroll
		for (value = 0u; value < Layout::kValues; ++value)
		{
			#pragma unroll
			for (warp = 0u; warp < Layout::kWarps; ++warp)
			{
				slot_index = value * Layout::kWarps + warp;
				element = value * THREADS + warp * LM_WARP_LANES + lane;
				accumulator[head][slot_index] = 0.0f;
				query[head][slot_index] = element < LATENT
					? LmBf16ToFloat(query_latent_bf16[((uint64_t)row * heads + first_head + head) * LATENT + element])
					: element < LATENT + ROPE ? LmBf16ToFloat(query_rope_bf16[((uint64_t)row * heads + first_head + head) * ROPE + element - LATENT]) : 0.0f;
			}
		}
	}
	for (step = first_position; step < last_position; ++step)
	{
		uint32_t position = selected_positions != 0 ? selected_positions[((uint64_t)row * selected_count) + step] : step;
		const uint16_t *slot;
		if (row_position != 0 && position > row_position[row])
			continue;
		slot = (const uint16_t *)LmKvSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
		if (slot == 0)
			break;
		#pragma unroll
		for (value = 0u; value < Layout::kValues; ++value)
		{
			#pragma unroll
			for (warp = 0u; warp < Layout::kWarps; ++warp)
			{
				element = value * THREADS + warp * LM_WARP_LANES + lane;
				key[value * Layout::kWarps + warp] = element < LATENT + ROPE ? LmBf16ToFloat(slot[element]) : 0.0f;
			}
		}
		#pragma unroll
		for (head = 0u; head < HEAD_GROUP; ++head)
		{
			float score, previous_max, scaled_previous, scaled_current;
			score = LmLatentRowsBlockScore<THREADS, LATENT + ROPE>(query[head], key, lane);
			score = score * qk_scale;
			previous_max = running_max[head];
			running_max[head] = fmaxf(running_max[head], score);
			scaled_previous = __expf(previous_max - running_max[head]);
			scaled_current = __expf(score - running_max[head]);
			running_sum[head] = (running_sum[head] * scaled_previous) + scaled_current;
			#pragma unroll
			for (value = 0u; value < Layout::kValues; ++value)
			{
				#pragma unroll
				for (warp = 0u; warp < Layout::kWarps; ++warp)
				{
					slot_index = value * Layout::kWarps + warp;
					if (value * THREADS + warp * LM_WARP_LANES + lane < LATENT)
						accumulator[head][slot_index] = (accumulator[head][slot_index] * scaled_previous) + (scaled_current * key[slot_index]);
				}
			}
		}
	}
	#pragma unroll
	for (head = 0u; head < HEAD_GROUP; ++head)
	{
		partial_base = (((uint64_t)row * heads + first_head + head) * partitions + partition) * (LATENT + 2u);
		if (lane == 0u)
		{
			partials[partial_base] = running_max[head];
			partials[partial_base + 1u] = running_sum[head];
		}
		#pragma unroll
		for (value = 0u; value < Layout::kValues; ++value)
		{
			#pragma unroll
			for (warp = 0u; warp < Layout::kWarps; ++warp)
			{
				element = value * THREADS + warp * LM_WARP_LANES + lane;
				if (element < LATENT)
					partials[partial_base + 2u + element] = accumulator[head][value * Layout::kWarps + warp];
			}
		}
	}
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentAttentionRowsSplitLaunch(
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
	uint32_t partitions;
	partitions = multiprocessor_count == 0u || rows == 0u || heads == 0u ? 1u : (multiprocessor_count * LM_LATENT_ATTN_SPLIT_CTAS_PER_SM + heads - 1u) / heads;
	if (partitions > LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS)
		partitions = LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
	if (THREADS != 8u * LM_WARP_LANES || heads % LM_LATENT_ROWS_HEAD_GROUP != 0u || LmLatentAttentionContextSplits(position_bound, split_context_threshold) == 0u || split_partials == 0 || partitions < 2u || (uint64_t)rows * heads * partitions > split_partial_blocks)
		return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, true>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
	LM_LAUNCH(
		(LmLatentAttentionRowsSplitKernel<Geometry, THREADS, LATENT, ROPE, LM_LATENT_ROWS_HEAD_GROUP>),
		dim3((rows + LM_LATENT_ROWS_WARPS - 1u) / LM_LATENT_ROWS_WARPS, partitions, heads / LM_LATENT_ROWS_HEAD_GROUP),
		LM_LATENT_ROWS_THREADS,
		0,
		stream,
		query_latent_bf16,
		query_rope_bf16,
		cache,
		sequence_of_row,
		context_length,
		selected_positions,
		selected_count,
		heads,
		partitions,
		qk_scale,
		split_partials,
		row_position,
		rows);
	if (cudaPeekAtLastError() != cudaSuccess)
		return cudaPeekAtLastError();
	LM_LAUNCH(
		(LmLatentAttentionDecodeSplitCombineKernel<THREADS, LATENT>),
		dim3(rows, heads),
		THREADS,
		0,
		stream,
		split_partials,
		output_bf16,
		heads,
		partitions);
	return cudaPeekAtLastError();
}
