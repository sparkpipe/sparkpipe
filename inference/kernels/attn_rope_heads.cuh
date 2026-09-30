#pragma once

#include "inference/kernels/attn.cuh"
#include <stdint.h>

#define LM_ROPE_HEADS_WARPS 8u
#define LM_ROPE_HEADS_THREADS (LM_ROPE_HEADS_WARPS * 32u)
#define LM_ROPE_HEADS_UNROLL 4u

template<uint32_t LATENT, uint32_t ROPE>
struct LmRopeHeadsShape
{
	static constexpr uint32_t kLatentPerLane = LATENT / 32u;
	static constexpr uint32_t kRopePerLane = ROPE / 32u;
	static constexpr uint32_t kLatentWords = kLatentPerLane / 2u;
	static constexpr uint32_t kRopeWords = kRopePerLane / 2u;
	static constexpr uint32_t kWords = kLatentWords + kRopeWords;
	static_assert(LATENT % 256u == 0u, "each lane loads whole 16-byte latent chunks");
	static_assert(ROPE % 64u == 0u, "each lane loads whole 4-byte rope words");
	static_assert(kRopeWords == 1u, "one rope word per lane");
};

template<uint32_t LATENT, uint32_t ROPE>
static __device__ __forceinline__ void LmRopeHeadsLoadSlot(const uint8_t *slot, uint32_t lane, uint32_t *words)
{
	using Shape = LmRopeHeadsShape<LATENT, ROPE>;
	const uint4 *latent = (const uint4 *)slot + lane * (Shape::kLatentWords / 4u);
	uint32_t chunk;
	uint4 packed;
	#pragma unroll
	for (chunk = 0u; chunk < Shape::kLatentWords / 4u; ++chunk)
	{
		packed = __ldcs(latent + chunk);
		words[chunk * 4u] = packed.x;
		words[chunk * 4u + 1u] = packed.y;
		words[chunk * 4u + 2u] = packed.z;
		words[chunk * 4u + 3u] = packed.w;
	}
	words[Shape::kLatentWords] = __ldcs((const uint32_t *)(slot + LATENT * sizeof(uint16_t)) + lane);
}

template<uint32_t HEADS, uint32_t LATENT, uint32_t ROPE>
static __device__ __forceinline__ void LmRopeHeadsStep(const float (*query)[LATENT / 32u + ROPE / 32u], const uint32_t *words, float qk_scale, float *running_max, float *running_sum, float (*accumulator)[LATENT / 32u])
{
	using Shape = LmRopeHeadsShape<LATENT, ROPE>;
	float value[Shape::kLatentPerLane + Shape::kRopePerLane];
	float score, previous, scaled_previous, scaled_current;
	uint32_t head, element, mask;
	#pragma unroll
	for (element = 0u; element < Shape::kWords; ++element)
	{
		value[element * 2u] = __uint_as_float(words[element] << 16u);
		value[element * 2u + 1u] = __uint_as_float(words[element] & 0xffff0000u);
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		score = 0.0f;
		#pragma unroll
		for (element = 0u; element < Shape::kLatentPerLane + Shape::kRopePerLane; ++element)
			score = fmaf(query[head][element], value[element], score);
		#pragma unroll
		for (mask = 16u; mask > 0u; mask >>= 1u)
			score += __shfl_xor_sync(0xffffffffu, score, mask);
		score *= qk_scale;
		previous = running_max[head];
		running_max[head] = fmaxf(previous, score);
		scaled_previous = __expf(previous - running_max[head]);
		scaled_current = __expf(score - running_max[head]);
		running_sum[head] = (running_sum[head] * scaled_previous) + scaled_current;
		#pragma unroll
		for (element = 0u; element < Shape::kLatentPerLane; ++element)
			accumulator[head][element] = (accumulator[head][element] * scaled_previous) + (scaled_current * value[element]);
	}
}

template<uint32_t HEADS, uint32_t LATENT>
static __device__ __forceinline__ void LmRopeHeadsMergeStore(float (*warp_max)[HEADS], float (*warp_sum)[HEADS], float4 (*stage)[LATENT / 4u], const float *running_max, const float *running_sum, const float (*accumulator)[LATENT / 32u], float *__restrict__ partials, uint64_t first_block, uint32_t partitions)
{
	constexpr uint32_t PER_LANE = LATENT / 32u;
	const uint32_t warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
	float global_max, global_sum, scale, total;
	uint32_t head, index, element;
	uint64_t base;
	if (lane == 0u)
		for (head = 0u; head < HEADS; ++head)
		{
			warp_max[warp][head] = running_max[head];
			warp_sum[warp][head] = running_sum[head];
		}
	__syncthreads();
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		global_max = -INFINITY;
		for (index = 0u; index < LM_ROPE_HEADS_WARPS; ++index)
			global_max = fmaxf(global_max, warp_max[index][head]);
		scale = LmLatentHeadsScale<HEADS, LATENT>(warp_max, warp, head, global_max);
		#pragma unroll
		for (element = 0u; element < PER_LANE / 4u; ++element)
			stage[warp][lane * (PER_LANE / 4u) + element] = make_float4(accumulator[head][element * 4u] * scale, accumulator[head][element * 4u + 1u] * scale, accumulator[head][element * 4u + 2u] * scale, accumulator[head][element * 4u + 3u] * scale);
		__syncthreads();
		base = (first_block + (uint64_t)head * partitions) * (LATENT + 2u);
		for (element = threadIdx.x; element < LATENT; element += LM_ROPE_HEADS_THREADS)
		{
			total = 0.0f;
			for (index = 0u; index < LM_ROPE_HEADS_WARPS; ++index)
				total += ((const float *)stage[index])[element];
			partials[base + 2u + element] = total;
		}
		if (threadIdx.x == 0u)
		{
			global_sum = 0.0f;
			for (index = 0u; index < LM_ROPE_HEADS_WARPS; ++index)
				global_sum = fmaf(warp_sum[index][head], LmLatentHeadsScale<HEADS, LATENT>(warp_max, index, head, global_max), global_sum);
			partials[base] = global_max;
			partials[base + 1u] = global_sum;
		}
		__syncthreads();
	}
}

template<class Geometry, uint32_t HEADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT>
__global__ __launch_bounds__(LM_ROPE_HEADS_THREADS, 1)
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
	using Shape = LmRopeHeadsShape<LATENT, ROPE>;
	constexpr uint32_t PER_LANE = Shape::kLatentPerLane;
	constexpr uint32_t QUERY = Shape::kLatentPerLane + Shape::kRopePerLane;
	__shared__ float warp_max[LM_ROPE_HEADS_WARPS][HEADS];
	__shared__ float warp_sum[LM_ROPE_HEADS_WARPS][HEADS];
	__shared__ float4 stage[LM_ROPE_HEADS_WARPS][LATENT / 4u];
	const uint32_t row = blockIdx.x, partition = blockIdx.y, warp = threadIdx.x / 32u, lane = threadIdx.x % 32u;
	const uint32_t sequence = sequence_of_row[row];
	float query[HEADS][QUERY], accumulator[HEADS][PER_LANE];
	float running_max[HEADS], running_sum[HEADS];
	uint32_t words[LM_ROPE_HEADS_UNROLL][Shape::kWords];
	const uint8_t *slot[LM_ROPE_HEADS_UNROLL];
	uint32_t head, element, index, position_count, span, first, last, step, unroll, position, failed = 0u;
	if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
	{
		LmKvReportRequiredAccessFailure(cache, !LmKvViewIsConfigured(cache) ? LM_KV_ACCESS_ERROR_INVALID_VIEW : LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE, LM_KV_ACCESS_READ, row, sequence, 0xffffffffu, 0xffffffffu);
		return;
	}
	#pragma unroll
	for (head = 0u; head < HEADS; ++head)
	{
		const uint16_t *latent = query_latent_bf16 + ((uint64_t)row * HEADS + head) * LATENT + lane * PER_LANE;
		const uint16_t *rope = query_rope_bf16 + ((uint64_t)row * HEADS + head) * ROPE + lane * Shape::kRopePerLane;
		running_max[head] = -INFINITY;
		running_sum[head] = 0.0f;
		#pragma unroll
		for (element = 0u; element < PER_LANE; ++element)
		{
			query[head][element] = LmBf16ToFloat(latent[element]);
			accumulator[head][element] = 0.0f;
		}
		#pragma unroll
		for (element = 0u; element < Shape::kRopePerLane; ++element)
			query[head][PER_LANE + element] = LmBf16ToFloat(rope[element]);
	}
	position_count = selected_positions != 0 ? selected_count : context_length[sequence];
	if (ROW_INVARIANT && selected_positions == 0 && row_position != 0 && row_position[row] + 1u < position_count)
		position_count = row_position[row] + 1u;
	span = (position_count + partitions - 1u) / partitions;
	first = partition * span;
	last = first + span < position_count ? first + span : position_count;
	for (step = first + warp; step < last && failed == 0u; step += LM_ROPE_HEADS_UNROLL * LM_ROPE_HEADS_WARPS)
	{
		#pragma unroll
		for (unroll = 0u; unroll < LM_ROPE_HEADS_UNROLL; ++unroll)
		{
			index = step + unroll * LM_ROPE_HEADS_WARPS;
			slot[unroll] = 0;
			if (index >= last || failed != 0u)
				continue;
			position = selected_positions != 0 ? selected_positions[(row * selected_count) + index] : index;
			if (row_position != 0 && position > row_position[row])
				continue;
			slot[unroll] = LmKvSlotRequired<Geometry>(cache, sequence, position, row, LM_KV_ACCESS_READ);
			failed = slot[unroll] == 0 ? 1u : 0u;
		}
		#pragma unroll
		for (unroll = 0u; unroll < LM_ROPE_HEADS_UNROLL; ++unroll)
			if (slot[unroll] != 0)
				LmRopeHeadsLoadSlot<LATENT, ROPE>(slot[unroll], lane, words[unroll]);
		#pragma unroll
		for (unroll = 0u; unroll < LM_ROPE_HEADS_UNROLL; ++unroll)
			if (slot[unroll] != 0)
				LmRopeHeadsStep<HEADS, LATENT, ROPE>(query, words[unroll], qk_scale, running_max, running_sum, accumulator);
	}
	LmRopeHeadsMergeStore<HEADS, LATENT>(warp_max, warp_sum, stage, running_max, running_sum, accumulator, partials, ((uint64_t)row * HEADS) * partitions + partition, partitions);
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE, bool ROW_INVARIANT, uint32_t HEADS>
static inline cudaError_t LmLatentRopeHeadsSplitShape(const uint16_t *query_latent_bf16, const uint16_t *query_rope_bf16, LmKvView cache, const uint32_t *sequence_of_row, const uint32_t *context_length, const uint32_t *selected_positions, uint32_t selected_count, uint32_t partitions, float qk_scale, uint16_t *output_bf16, float *split_partials, const uint32_t *row_position, uint32_t rows, cudaStream_t stream)
{
	LM_LAUNCH((LmLatentRopeHeadsSplitKernel<Geometry, HEADS, LATENT, ROPE, ROW_INVARIANT>), dim3(rows, partitions), LM_ROPE_HEADS_THREADS, 0, stream, query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, split_partials, row_position);
	if (cudaPeekAtLastError() != cudaSuccess)
		return cudaPeekAtLastError();
	LM_LAUNCH((LmLatentAttentionDecodeSplitCombineKernel<THREADS, LATENT>), dim3(rows, HEADS), THREADS, 0, stream, split_partials, output_bf16, HEADS, partitions);
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
	if (LmLatentAttentionContextSplits(position_bound, split_context_threshold) == 0u || split_partials == 0 || partitions < 2u || blocks == 0u || (uint64_t)blocks * partitions > split_partial_blocks || THREADS != LM_ROPE_HEADS_THREADS)
		return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
	if (heads == 4u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 4u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	if (heads == 2u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 2u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	if (heads == 1u)
		return LmLatentRopeHeadsSplitShape<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT, 1u>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, partitions, qk_scale, output_bf16, split_partials, row_position, rows, stream);
	return LmLatentAttentionDecodeSplitLaunch<Geometry, THREADS, LATENT, ROPE, ROW_INVARIANT>(query_latent_bf16, query_rope_bf16, cache, sequence_of_row, context_length, selected_positions, selected_count, heads, qk_scale, output_bf16, row_position, rows, position_bound, split_context_threshold, split_partials, split_partial_blocks, multiprocessor_count, stream);
}
