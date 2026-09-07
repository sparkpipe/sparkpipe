#pragma once


#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <math.h>
#include <stdint.h>

#define LM_KV_POSITION_UNUSED 0xffffffffu

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmBuildSlidingWindowPositionsKernel(
	const uint32_t *__restrict__ sequence_of_row,
	const uint32_t *__restrict__ context_length,
	const uint32_t *__restrict__ position_of_row,
	uint32_t row_count,
	uint32_t window,
	uint32_t *__restrict__ positions_out)
{
	uint32_t row = blockIdx.x;
	uint32_t sequence;
	uint32_t available;
	uint32_t selected;
	uint32_t start;
	uint32_t index;

	if ( row >= row_count || window == 0u )
		return;
	sequence = sequence_of_row[row];
	available = context_length[sequence];
	if ( position_of_row != 0 && position_of_row[row] != 0xffffffffu
		&& available > position_of_row[row] + 1u )
		available = position_of_row[row] + 1u;
	selected = available < window ? available : window;
	start = available - selected;
	for ( index = threadIdx.x; index < window; index += THREADS )
		positions_out[((uint64_t)row * window) + index] = index < selected
			? start + index
			: LM_KV_POSITION_UNUSED;
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmExpandHeadsKernel(const uint16_t *__restrict__ input_bf16, uint16_t *__restrict__ output_bf16, uint32_t source_heads, uint32_t head_dim, uint32_t group, uint32_t rows)
{
	uint32_t row = blockIdx.x,head,index;
	for (head = 0u; head < source_heads * group; ++head)
		for (index = threadIdx.x; index < head_dim; index += THREADS)
			output_bf16[(((uint64_t)row * source_heads * group) + head) * head_dim + index] =
				input_bf16[(((uint64_t)row * source_heads) + (head / group)) * head_dim + index];
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaKvStoreKernel(LmKvView view, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count)
{
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"the slot is [K: heads x head_dim][V: heads x value_dim] bf16 and nothing else");
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count )
		return;
	slot = LmKvSlotMutableRequired<Geometry>(
		view,sequence_of_row[row],position_of_row[row],row);
	if ( slot == 0 )
		return;
	for (index = threadIdx.x; index < KV_HEADS * (HEAD_DIM + VALUE_DIM); index += THREADS)
		((uint16_t *)slot)[index] = index < KV_HEADS * HEAD_DIM
			? key_bf16[((uint64_t)row * KV_HEADS * HEAD_DIM) + index]
			: value_bf16[((uint64_t)row * KV_HEADS * VALUE_DIM)
				+ (index - (KV_HEADS * HEAD_DIM))];
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaAttentionDecodeKernel(const uint16_t *__restrict__ query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[HEAD_DIM];
	float accumulator[(VALUE_DIM + THREADS - 1u) / THREADS];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"store and decode must agree on the slot layout");
	uint32_t sequence = sequence_of_row[row];
	uint32_t kv_head;
	uint64_t query_base;
	float running_max = -INFINITY,running_sum = 0.0f;

	if ( !LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			!LmKvViewIsConfigured(cache)
				? LM_KV_ACCESS_ERROR_INVALID_VIEW
				: LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			0xffffffffu,
			0xffffffffu);
		return;
	}
	if ( heads < KV_HEADS || ( heads % KV_HEADS ) != 0u || head >= heads )
	{
		LmKvReportRequiredAccessFailure(
			cache,
			LM_KV_ACCESS_ERROR_INVALID_GQA_GEOMETRY,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			head,
			heads);
		return;
	}
	kv_head = head / (heads / KV_HEADS);
	query_base = ((uint64_t)row * heads + head) * HEAD_DIM;
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
		shared_query[index] = LmBf16ToFloat(query_bf16[query_base + index]);
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : context_length[sequence];
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? selected_positions[(row * selected_count) + step] : step;
		const uint8_t *slot;
		if ( position == LM_KV_POSITION_UNUSED )
			continue;
		const uint16_t *key,*value;
		float score = 0.0f,scaled,previous;
		if ( row_position != 0 && position > row_position[row] )
			continue;
		slot = LmKvSlotRequired<Geometry>(
			cache,sequence,position,row,LM_KV_ACCESS_READ);
		if ( slot == 0 )
			return;
		key = (const uint16_t *)slot + (kv_head * HEAD_DIM);
		value = (const uint16_t *)slot + (KV_HEADS * HEAD_DIM) + (kv_head * VALUE_DIM);
		for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(key[index]);
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		running_sum = (running_sum * scaled) + __expf(score - running_max);
		for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < VALUE_DIM )
				accumulator[index] = (accumulator[index] * scaled)
					+ (__expf(score - running_max) * LmBf16ToFloat(value[element]));
		}
	}
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < VALUE_DIM )
			output_bf16[(((uint64_t)row * heads) + head) * VALUE_DIM + element] =
				LmFloatToBf16(accumulator[index] / fmaxf(running_sum,1.0e-20f));
	}
}
