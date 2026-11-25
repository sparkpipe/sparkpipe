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

	LmDependentWait();
	if ( row >= row_count || window == 0u )
	{
		LmDependentRelease();
		return;
	}
	sequence = __ldg(sequence_of_row + row);
	available = __ldg(context_length + sequence);
	if ( position_of_row != 0 )
	{
		const uint32_t row_pos = __ldg(position_of_row + row);
		if ( row_pos != 0xffffffffu && available > row_pos + 1u )
			available = row_pos + 1u;
	}
	selected = available < window ? available : window;
	start = available - selected;
	for ( index = threadIdx.x; index < window; index += THREADS )
		positions_out[((uint64_t)row * window) + index] = index < selected
			? start + index
			: LM_KV_POSITION_UNUSED;
	LmDependentRelease();
}

template<uint32_t THREADS>
__global__ __launch_bounds__(THREADS, 1)
void LmExpandHeadsKernel(const uint16_t *__restrict__ input_bf16, uint16_t *__restrict__ output_bf16, uint32_t source_heads, uint32_t head_dim, uint32_t group, uint32_t rows)
{
	LmDependentWait();
	uint32_t row = blockIdx.x,head,index;
	if ( row >= rows )
	{
		LmDependentRelease();
		return;
	}
	if ( (head_dim % 8u) == 0u && ((((uintptr_t)input_bf16 | (uintptr_t)output_bf16) & 15u) == 0u) )
	{
		const uint32_t head_chunks = head_dim >> 3u;
		const uint32_t total_chunks = source_heads * group * head_chunks;
		const uint4 *in4 = (const uint4 *)(input_bf16 + (uint64_t)row * source_heads * head_dim);
		uint4 *out4 = (uint4 *)(output_bf16 + (uint64_t)row * source_heads * group * head_dim);
		for (index = threadIdx.x; index < total_chunks; index += THREADS)
		{
			const uint32_t out_head = index / head_chunks;
			const uint32_t chunk = index % head_chunks;
			out4[index] = __ldg(in4 + (out_head / group) * head_chunks + chunk);
		}
		LmDependentRelease();
		return;
	}
	for (head = 0u; head < source_heads * group; ++head)
		for (index = threadIdx.x; index < head_dim; index += THREADS)
			output_bf16[(((uint64_t)row * source_heads * group) + head) * head_dim + index] =
				__ldg(input_bf16 + (((uint64_t)row * source_heads) + (head / group)) * head_dim + index);
	LmDependentRelease();
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaKvStoreKernel(LmKvView view, const uint16_t *__restrict__ key_bf16, const uint16_t *__restrict__ value_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ position_of_row, uint32_t row_count)
{
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"the slot is [K: heads x head_dim][V: heads x value_dim] bf16 and nothing else");
	LmDependentWait();
	uint32_t row = blockIdx.x,index;
	uint8_t *slot;
	if ( row >= row_count )
	{
		LmDependentRelease();
		return;
	}
	slot = LmKvSlotMutableRequired<Geometry>(
		view,__ldg(sequence_of_row + row),__ldg(position_of_row + row),row);
	if ( slot == 0 )
	{
		LmDependentRelease();
		return;
	}
	if constexpr ( ((KV_HEADS * HEAD_DIM) % 8u) == 0u && ((KV_HEADS * VALUE_DIM) % 8u) == 0u )
	{
		if ( ((((uintptr_t)slot | (uintptr_t)key_bf16 | (uintptr_t)value_bf16) & 15u) == 0u) )
		{
			constexpr uint32_t KEY_CHUNKS = (KV_HEADS * HEAD_DIM) / 8u;
			constexpr uint32_t TOTAL_CHUNKS = (KV_HEADS * (HEAD_DIM + VALUE_DIM)) / 8u;
			const uint4 *key4 = (const uint4 *)(key_bf16 + (uint64_t)row * KV_HEADS * HEAD_DIM);
			const uint4 *val4 = (const uint4 *)(value_bf16 + (uint64_t)row * KV_HEADS * VALUE_DIM);
			uint4 *dst4 = (uint4 *)slot;
			for (index = threadIdx.x; index < TOTAL_CHUNKS; index += THREADS)
				dst4[index] = index < KEY_CHUNKS ? __ldg(key4 + index) : __ldg(val4 + (index - KEY_CHUNKS));
			LmDependentRelease();
			return;
		}
	}
	for (index = threadIdx.x; index < KV_HEADS * (HEAD_DIM + VALUE_DIM); index += THREADS)
		((uint16_t *)slot)[index] = index < KV_HEADS * HEAD_DIM
			? __ldg(key_bf16 + ((uint64_t)row * KV_HEADS * HEAD_DIM) + index)
			: __ldg(value_bf16 + ((uint64_t)row * KV_HEADS * VALUE_DIM)
				+ (index - (KV_HEADS * HEAD_DIM)));
	LmDependentRelease();
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
static __device__ __forceinline__ void LmGqaAttentionDecodeRow(const uint16_t *__restrict__ query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position, float initial_max, float initial_sum)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[HEAD_DIM];
	float accumulator[(VALUE_DIM + THREADS - 1u) / THREADS];
	LmDependentWait();
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	static_assert(Geometry::kSlotBytes == (KV_HEADS * (HEAD_DIM + VALUE_DIM) * 2u),
		"store and decode must agree on the slot layout");
	uint32_t sequence = __ldg(sequence_of_row + row);
	const uint32_t max_pos = row_position != 0 ? __ldg(row_position + row) : 0xffffffffu;
	uint32_t kv_head;
	uint64_t query_base;
	float running_max = initial_max,running_sum = initial_sum;

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
		LmDependentRelease();
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
		LmDependentRelease();
		return;
	}
	kv_head = head / (heads / KV_HEADS);
	query_base = ((uint64_t)row * heads + head) * HEAD_DIM;
	#pragma unroll
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
		shared_query[index] = LmBf16ToFloat(__ldg(query_bf16 + query_base + index));
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : __ldg(context_length + sequence);
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? __ldg(selected_positions + (row * selected_count) + step) : step;
		const uint8_t *slot;
		if ( position == LM_KV_POSITION_UNUSED )
			continue;
		const uint16_t *key,*value;
		float score = 0.0f,scaled,previous,current_prob,value_weight;
		if ( row_position != 0 && position > max_pos )
			continue;
		slot = LmKvSlotRequired<Geometry>(
			cache,sequence,position,row,LM_KV_ACCESS_READ);
		if ( slot == 0 )
		{
			LmDependentRelease();
			return;
		}
		key = (const uint16_t *)slot + (kv_head * HEAD_DIM);
		value = (const uint16_t *)slot + (KV_HEADS * HEAD_DIM) + (kv_head * VALUE_DIM);
		for (index = threadIdx.x; index < HEAD_DIM; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(__ldg(key + index));
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		current_prob = __expf(score - running_max);
		running_sum = (running_sum * scaled) + current_prob;
		value_weight = SparkMathAttentionValueWeight(current_prob);
		#pragma unroll
		for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < VALUE_DIM )
				accumulator[index] = (accumulator[index] * scaled)
					+ (value_weight * LmBf16ToFloat(__ldg(value + element)));
		}
	}
	const float denom = fmaxf(running_sum,1.0e-20f);
	#pragma unroll
	for (index = 0u; index < (VALUE_DIM + THREADS - 1u) / THREADS; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < VALUE_DIM )
			output_bf16[(((uint64_t)row * heads) + head) * VALUE_DIM + element] =
				LmFloatToBf16(accumulator[index] / denom);
	}
	LmDependentRelease();
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaAttentionDecodeKernel(const uint16_t *__restrict__ query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
{
	LmGqaAttentionDecodeRow<Geometry,THREADS,KV_HEADS,HEAD_DIM,VALUE_DIM>(query_bf16,cache,sequence_of_row,context_length,selected_positions,selected_count,heads,qk_scale,output_bf16,row_position,-INFINITY,0.0f);
}

template<class Geometry, uint32_t THREADS, uint32_t KV_HEADS, uint32_t HEAD_DIM, uint32_t VALUE_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmGqaSinkAttentionDecodeKernel(const uint16_t *__restrict__ query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position, const uint16_t *__restrict__ sink_bf16)
{
	LmGqaAttentionDecodeRow<Geometry,THREADS,KV_HEADS,HEAD_DIM,VALUE_DIM>(query_bf16,cache,sequence_of_row,context_length,selected_positions,selected_count,heads,qk_scale,output_bf16,row_position,LmBf16ToFloat(sink_bf16[blockIdx.y < heads ? blockIdx.y : 0u]),1.0f);
}
