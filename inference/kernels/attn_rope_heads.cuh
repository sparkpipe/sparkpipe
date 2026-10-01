#pragma once

#include "inference/kernels/attn.cuh"
#include <stdint.h>

#define LM_ROPE_HEADS_THREADS 256u
#define LM_LATENT_WARP_MIN_SPAN 32u
#define LM_LATENT_WARP_SPAN(count) ((count) <= 16u * LM_LATENT_WARP_MIN_SPAN ? LM_LATENT_WARP_MIN_SPAN : (count) <= 32u * LM_LATENT_WARP_MIN_SPAN ? 2u * LM_LATENT_WARP_MIN_SPAN : 4u * LM_LATENT_WARP_MIN_SPAN)
#define LM_LATENT_WARP_WARPS 4u
#define LM_LATENT_WARP_THREADS (LM_LATENT_WARP_WARPS * LM_WARP_LANES)

static __device__ __forceinline__ void LmLatentWarpUnpack(uint4 word, float *out)
{
	const uint32_t parts[4] = {word.x, word.y, word.z, word.w};
	#pragma unroll
	for (uint32_t i = 0u; i < 4u; ++i)
	{
		out[2u * i] = __uint_as_float(parts[i] << 16u);
		out[2u * i + 1u] = __uint_as_float(parts[i] & 0xffff0000u);
	}
}

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT>
__global__ __launch_bounds__(LM_LATENT_WARP_THREADS)
void LmLatentWarpSplitKernel(
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
	static_assert(LATENT == 16u * LM_WARP_LANES && ROPE == 64u, "one warp covers the latent with two 16-byte loads per lane and the rope with eight lanes");
	__shared__ float warp_max[LM_LATENT_WARP_WARPS][HEADS];
	__shared__ float warp_sum[LM_LATENT_WARP_WARPS][HEADS];
	__shared__ float warp_acc[LM_LATENT_WARP_WARPS][HEADS][LATENT];
	const uint32_t row = blockIdx.x, partition = blockIdx.y, sequence = sequence_of_row[row];
	const uint32_t lane = threadIdx.x % LM_WARP_LANES, warp = threadIdx.x / LM_WARP_LANES;
	float q_latent[HEADS][16], q_rope[HEADS][8], acc[HEADS][16], run_max[HEADS], run_sum[HEADS];
	uint32_t head, i, position_count, first_position, last_position, step;
	uint64_t partial_base;
	LmDependentRelease();
	if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
	{
		if (threadIdx.x == 0u)
			LmKvReportRequiredAccessFailure(cache, !LmKvViewIsConfigured(cache) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		const uint16_t *ql = query_latent_bf16 + ((uint64_t)row * HEADS + head) * LATENT + lane * 16u;
		const uint16_t *qr = query_rope_bf16 + ((uint64_t)row * HEADS + head) * ROPE + (lane % 8u) * 8u;
		LmLatentWarpUnpack(*(const uint4 *)ql, q_latent[head]);
		LmLatentWarpUnpack(*(const uint4 *)(ql + 8u), q_latent[head] + 8u);
		LmLatentWarpUnpack(*(const uint4 *)qr, q_rope[head]);
		run_max[head] = -INFINITY;
		run_sum[head] = 0.0f;
		#pragma unroll
		for (i = 0u; i < 16u; ++i)
			acc[head][i] = 0.0f;
	}
	position_count = selected_positions != 0 ? selected_count : context_length[sequence];
	if (ROW_INVARIANT && selected_positions == 0 && row_position != 0 && row_position[row] + 1u < position_count)
		position_count = row_position[row] + 1u;
	first_position = partition * LM_LATENT_WARP_SPAN(position_count);
	last_position = first_position + LM_LATENT_WARP_SPAN(position_count) < position_count ? first_position + LM_LATENT_WARP_SPAN(position_count) : position_count;
	for (step = first_position + warp; step < last_position; step += LM_LATENT_WARP_WARPS)
	{
		uint32_t position = selected_positions != 0 ? selected_positions[(row * selected_count) + step] : step;
		const uint8_t *slot;
		float key[16], rope[8], score[HEADS];
		if (row_position != 0 && position > row_position[row])
			continue;
		slot = LmKvSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
		if (slot == 0)
			break;
		LmLatentWarpUnpack(((const uint4 *)slot)[2u * lane], key);
		LmLatentWarpUnpack(((const uint4 *)slot)[2u * lane + 1u], key + 8u);
		LmLatentWarpUnpack(((const uint4 *)(slot + LATENT * sizeof(uint16_t)))[lane % 8u], rope);
		#pragma unroll
		for (head = 0u; head < HEADS; ++head)
		{
			float partial = 0.0f;
			#pragma unroll
			for (i = 0u; i < 16u; ++i)
				partial += q_latent[head][i] * key[i];
			if (lane < 8u)
			{
				#pragma unroll
				for (i = 0u; i < 8u; ++i)
					partial += q_rope[head][i] * rope[i];
			}
			#pragma unroll
			for (i = LM_WARP_LANES / 2u; i > 0u; i /= 2u)
				partial += __shfl_xor_sync(0xffffffffu, partial, i);
			score[head] = partial * qk_scale;
		}
		#pragma unroll
		for (head = 0u; head < HEADS; ++head)
		{
			float previous = run_max[head], scaled_previous, weight;
			run_max[head] = fmaxf(previous, score[head]);
			scaled_previous = __expf(previous - run_max[head]);
			weight = __expf(score[head] - run_max[head]);
			run_sum[head] = run_sum[head] * scaled_previous + weight;
			#pragma unroll
			for (i = 0u; i < 16u; ++i)
				acc[head][i] = acc[head][i] * scaled_previous + weight * key[i];
		}
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		warp_max[warp][head] = run_max[head];
		warp_sum[warp][head] = run_sum[head];
		#pragma unroll
		for (i = 0u; i < 16u; ++i)
			warp_acc[warp][head][lane * 16u + i] = acc[head][i];
	}
	__syncthreads();
	for (i = threadIdx.x; i < HEADS * (LATENT + 2u); i += LM_LATENT_WARP_THREADS)
	{
		uint32_t w, element;
		float global_max = -INFINITY, value = 0.0f;
		head = i / (LATENT + 2u);
		element = i % (LATENT + 2u);
		#pragma unroll
		for (w = 0u; w < LM_LATENT_WARP_WARPS; ++w)
			global_max = fmaxf(global_max, warp_max[w][head]);
		if (element == 0u)
			value = global_max;
		else
		{
			#pragma unroll
			for (w = 0u; w < LM_LATENT_WARP_WARPS; ++w)
			{
				float scale = warp_max[w][head] == -INFINITY ? 0.0f : __expf(warp_max[w][head] - global_max);
				value += scale * (element == 1u ? warp_sum[w][head] : warp_acc[w][head][element - 2u]);
			}
		}
		partial_base = (((uint64_t)row * HEADS + head) * partitions + partition) * (LATENT + 2u);
		partials[partial_base + element] = value;
	}
}

template<class Geometry, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT, uint32_t HEADS>
static inline cudaError_t LmLatentWarpSplitShape(const uint16_t *query_latent_bf16, const uint16_t *query_rope_bf16, LmKvView cache, const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *selected_positions, uint32_t selected_count, uint32_t partitions, float qk_scale, uint16_t *output_bf16, float *split_partials, const uint32_t *row_position, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmLatentWarpSplitKernel<Geometry, HEADS, LATENT, ROPE, ROW_INVARIANT>), dim3(rows, partitions), LM_LATENT_WARP_THREADS, 0, stream, query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, split_partials, row_position);
	if (cudaPeekAtLastError() != cudaSuccess)
		return cudaPeekAtLastError();
	LM_LAUNCH_DEPENDENT((LmLatentAttentionDecodeSplitCombineKernel<LM_ROPE_HEADS_THREADS, LATENT>), dim3(rows, HEADS), LM_ROPE_HEADS_THREADS, 0, stream, split_partials, output_bf16, HEADS, partitions);
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
	uint32_t blocks = rows * heads, bound = selected_positions != 0 ? selected_count : position_bound, partitions;
	partitions = bound == 0u ? 1u : (bound + LM_LATENT_WARP_MIN_SPAN - 1u) / LM_LATENT_WARP_MIN_SPAN;
	partitions = partitions > LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS ? LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS : partitions;
	if (ROW_INVARIANT && LATENT == 16u * LM_WARP_LANES && ROPE == 64u && split_partials != 0 && bound <= 64u * LM_LATENT_WARP_MIN_SPAN && (uint64_t)blocks * partitions <= split_partial_blocks)
	{
		if (heads == 4u || heads == 2u || heads == 1u)
			return heads == 4u ? LmLatentWarpSplitShape<Geometry, LATENT, ROPE, ROW_INVARIANT, 4u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream) : heads == 2u ? LmLatentWarpSplitShape<Geometry, LATENT, ROPE, ROW_INVARIANT, 2u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream) : LmLatentWarpSplitShape<Geometry, LATENT, ROPE, ROW_INVARIANT, 1u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	}
	return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
}
