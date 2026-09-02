#pragma once


#include "inference/kernels/kv.cuh"
#include "inference/kernels/norm.cuh"
#include <stdint.h>

static __device__ __forceinline__ void LmRopePair(float *low, float *high, float angle)
{
	float c = __cosf(angle),s = __sinf(angle);
	float a = *low,b = *high;
	*low = (a * c) - (b * s);
	*high = (a * s) + (b * c);
}

enum LmRopePairing
{
	LM_ROPE_HALF_SPLIT = 0,
	LM_ROPE_INTERLEAVED = 1
};

template<LmRopePairing PAIRING>
static __device__ __forceinline__ void LmRopeRotate(uint16_t *rows_bf16, uint64_t base, uint32_t index, uint32_t half, float angle)
{
	uint32_t low_offset,high_offset;
	float low,high;
	low_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? (index * 2u) : index;
	high_offset = (PAIRING == LM_ROPE_INTERLEAVED) ? ((index * 2u) + 1u) : (half + index);
	low = LmBf16ToFloat(rows_bf16[base + low_offset]);
	high = LmBf16ToFloat(rows_bf16[base + high_offset]);
	LmRopePair(&low,&high,angle);
	rows_bf16[base + low_offset] = LmFloatToBf16(low);
	rows_bf16[base + high_offset] = LmFloatToBf16(high);
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dim));
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopePerHeadKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t head_count, uint32_t head_dimension, uint32_t rope_offset, uint32_t rope_dimension, float theta)
{
	uint32_t row = blockIdx.x,head = blockIdx.y,half = rope_dimension / 2u,index;
	uint64_t base;
	float position;
	if ( head >= head_count )
		return;
	base = (((uint64_t)row * head_count) + head) * head_dimension + rope_offset;
	position = (float)positions[row];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,
			position * __powf(theta,-2.0f * (float)index / (float)rope_dimension));
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmAttentionDecodeKernel(const uint16_t *__restrict__ query_latent_bf16, const uint16_t *__restrict__ query_rope_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_positions, uint32_t selected_count, uint32_t heads, float qk_scale, uint16_t *__restrict__ output_bf16, const uint32_t *__restrict__ row_position)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	__shared__ float shared_query[LATENT + ROPE];
	float accumulator[(LATENT + THREADS - 1u) / THREADS];
	uint32_t row = blockIdx.x,head = blockIdx.y,index,step,positions;
	uint32_t sequence = sequence_of_row[row];
	uint64_t query_base = ((uint64_t)row * heads + head) * (LATENT + ROPE);
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
	for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
		accumulator[index] = 0.0f;
	for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
		shared_query[index] = LmBf16ToFloat(query_latent_bf16[query_base + index]);
	__syncthreads();
	positions = selected_positions != 0 ? selected_count : context_length[sequence];
	for (step = 0u; step < positions; ++step)
	{
		uint32_t position = selected_positions != 0
			? selected_positions[(row * selected_count) + step] : step;
		if ( row_position != 0 && position > row_position[row] )
			continue;
		const uint8_t *slot = LmKvSlotRequired<Geometry>(
			cache,sequence,position,row,LM_KV_ACCESS_READ);
		float score = 0.0f,scaled,previous;
		if ( slot == 0 )
			return;
		for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
			score += shared_query[index] * LmBf16ToFloat(((const uint16_t *)slot)[index]);
		score = LmBlockSum<THREADS>(score,reduction) * qk_scale;
		previous = running_max;
		running_max = fmaxf(running_max,score);
		scaled = __expf(previous - running_max);
		running_sum = (running_sum * scaled) + __expf(score - running_max);
		for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
		{
			uint32_t element = (index * THREADS) + threadIdx.x;
			if ( element < LATENT )
				accumulator[index] = (accumulator[index] * scaled)
					+ (__expf(score - running_max)
						* LmBf16ToFloat(((const uint16_t *)slot)[element]));
		}
	}
	for (index = 0u; index < (LATENT + THREADS - 1u) / THREADS; ++index)
	{
		uint32_t element = (index * THREADS) + threadIdx.x;
		if ( element < LATENT )
			output_bf16[(((uint64_t)row * heads) + head) * LATENT + element] =
				LmFloatToBf16(accumulator[index] / fmaxf(running_sum,1.0e-20f));
	}
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ query_rope_bf16,
    LmKvView cache,
    const uint32_t *__restrict__ sequence_of_row,
    const uint32_t *__restrict__ context_length,
    const uint32_t *__restrict__ selected_positions,
    uint32_t selected_count,
    uint32_t heads,
    float qk_scale,
    uint16_t *__restrict__ output_bf16,
    const uint32_t *__restrict__ row_position)
{
    static_assert(
        LATENT <= 8u * THREADS,
        "the latent must fit the per-thread accumulator");
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    __shared__ float shared_query[LATENT + ROPE];
    float accumulator[8];
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t index;
    uint32_t step;
    uint32_t position_count;
    uint32_t sequence = sequence_of_row[row];
    uint64_t latent_base = ((uint64_t)row * heads + head) * LATENT;
    uint64_t rope_base = ((uint64_t)row * heads + head) * ROPE;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
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

    for (index = 0u; index < 8u; ++index)
    {
        accumulator[index] = 0.0f;
    }
    for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
    {
        shared_query[index] = index < LATENT
            ? LmBf16ToFloat(query_latent_bf16[latent_base + index])
            : LmBf16ToFloat(query_rope_bf16[rope_base + index - LATENT]);
    }
    __syncthreads();

    position_count = selected_positions != 0
        ? selected_count
        : context_length[sequence];
    for (step = 0u; step < position_count; ++step)
    {
        uint32_t position = selected_positions != 0
            ? selected_positions[(row * selected_count) + step]
            : step;
        const uint8_t *slot;
        float score = 0.0f;
        float scaled_previous;
        float scaled_current;
        float previous_max;

        if (row_position != 0 && position > row_position[row])
        {
            continue;
        }
        slot = LmKvSlotRequired<Geometry>(
            cache, sequence, position, row, LM_KV_ACCESS_READ);
        if (slot == 0)
        {
            return;
        }
        for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
        {
            score += shared_query[index] *
                LmBf16ToFloat(((const uint16_t *)slot)[index]);
        }
        score = LmBlockSum<THREADS>(score, reduction) * qk_scale;
        previous_max = running_max;
        running_max = fmaxf(running_max, score);
        scaled_previous = __expf(previous_max - running_max);
        scaled_current = __expf(score - running_max);
        running_sum = (running_sum * scaled_previous) + scaled_current;
        for (index = 0u; index < 8u; ++index)
        {
            uint32_t element = (index * THREADS) + threadIdx.x;

            if (element < LATENT)
            {
                accumulator[index] =
                    (accumulator[index] * scaled_previous) +
                    (scaled_current *
                        LmBf16ToFloat(((const uint16_t *)slot)[element]));
            }
        }
    }
    for (index = 0u; index < 8u; ++index)
    {
        uint32_t element = (index * THREADS) + threadIdx.x;

        if (element < LATENT)
        {
            output_bf16[
                (((uint64_t)row * heads) + head) * LATENT + element] =
                LmFloatToBf16(
                    accumulator[index] / fmaxf(running_sum, 1.0e-20f));
        }
    }
}

#define LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS 16u
#define LM_LATENT_ATTN_SPLIT_CTAS_PER_SM 4u
#define LM_LATENT_ATTN_SPLIT_BLOCK_FLOATS(l) ((l) + 2u)

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeSplitKernel(
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
    const uint32_t *__restrict__ row_position)
{
    static_assert(
        LATENT <= 8u * THREADS,
        "the latent must fit the per-thread accumulator");
    __shared__ float reduction[THREADS / LM_WARP_LANES];
    __shared__ float shared_query[LATENT + ROPE];
    float accumulator[8];
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t partition = blockIdx.z;
    uint32_t index;
    uint32_t step;
    uint32_t position_count;
    uint32_t first_position;
    uint32_t last_position;
    uint32_t partition_span;
    uint32_t sequence = sequence_of_row[row];
    uint64_t latent_base = ((uint64_t)row * heads + head) * LATENT;
    uint64_t rope_base = ((uint64_t)row * heads + head) * ROPE;
    uint64_t partial_base;
    float running_max = -INFINITY;
    float running_sum = 0.0f;

    if (!LmKvViewIsConfigured(cache) || sequence >= cache.sequence_count)
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
    if (partition >= partitions)
    {
        return;
    }

    for (index = 0u; index < 8u; ++index)
    {
        accumulator[index] = 0.0f;
    }
    for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
    {
        shared_query[index] = index < LATENT
            ? LmBf16ToFloat(query_latent_bf16[latent_base + index])
            : LmBf16ToFloat(query_rope_bf16[rope_base + index - LATENT]);
    }
    __syncthreads();

    position_count = selected_positions != 0
        ? selected_count
        : context_length[sequence];
    partition_span = (position_count + partitions - 1u) / partitions;
    first_position = partition * partition_span;
    last_position = first_position + partition_span;
    if (last_position > position_count)
    {
        last_position = position_count;
    }
    partial_base = (((uint64_t)row * heads + head) * partitions + partition) *
                   (LATENT + 2u);
    if (first_position >= last_position)
    {
        if (threadIdx.x == 0u)
        {
            partials[partial_base] = -INFINITY;
            partials[partial_base + 1u] = 0.0f;
        }
        for (index = threadIdx.x; index < LATENT; index += THREADS)
        {
            partials[partial_base + 2u + index] = 0.0f;
        }
        return;
    }
    for (step = first_position; step < last_position; ++step)
    {
        uint32_t position = selected_positions != 0
            ? selected_positions[(row * selected_count) + step]
            : step;
        const uint8_t *slot;
        float score = 0.0f;
        float scaled_previous;
        float scaled_current;
        float previous_max;

        if (row_position != 0 && position > row_position[row])
        {
            continue;
        }
        slot = LmKvSlotRequired<Geometry>(
            cache, sequence, position, row, LM_KV_ACCESS_READ);
        if (slot == 0)
        {
            if (threadIdx.x == 0u)
            {
                partials[partial_base] = running_max;
                partials[partial_base + 1u] = running_sum;
            }
            for (index = 0u; index < 8u; ++index)
            {
                uint32_t element = (index * THREADS) + threadIdx.x;

                if (element < LATENT)
                {
                    partials[partial_base + 2u + element] = accumulator[index];
                }
            }
            return;
        }
        for (index = threadIdx.x; index < LATENT + ROPE; index += THREADS)
        {
            score += shared_query[index] *
                LmBf16ToFloat(((const uint16_t *)slot)[index]);
        }
        score = LmBlockSum<THREADS>(score, reduction) * qk_scale;
        previous_max = running_max;
        running_max = fmaxf(running_max, score);
        scaled_previous = __expf(previous_max - running_max);
        scaled_current = __expf(score - running_max);
        running_sum = (running_sum * scaled_previous) + scaled_current;
        for (index = 0u; index < 8u; ++index)
        {
            uint32_t element = (index * THREADS) + threadIdx.x;

            if (element < LATENT)
            {
                accumulator[index] =
                    (accumulator[index] * scaled_previous) +
                    (scaled_current *
                        LmBf16ToFloat(((const uint16_t *)slot)[element]));
            }
        }
    }
    if (threadIdx.x == 0u)
    {
        partials[partial_base] = running_max;
        partials[partial_base + 1u] = running_sum;
    }
    for (index = 0u; index < 8u; ++index)
    {
        uint32_t element = (index * THREADS) + threadIdx.x;

        if (element < LATENT)
        {
            partials[partial_base + 2u + element] = accumulator[index];
        }
    }
}

template<uint32_t THREADS, uint32_t LATENT>
__global__ __launch_bounds__(THREADS, 1)
void LmLatentAttentionDecodeSplitCombineKernel(
    const float *__restrict__ partials,
    uint16_t *__restrict__ output_bf16,
    uint32_t heads,
    uint32_t partitions)
{
    __shared__ float scales[LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS];
    __shared__ float denominator_shared;
    uint32_t row = blockIdx.x;
    uint32_t head = blockIdx.y;
    uint32_t partition;
    uint32_t element;
    uint64_t block_base;
    float global_max;
    float denominator;

    block_base = ((uint64_t)row * heads + head) * partitions *
                 (LATENT + 2u);
    if (threadIdx.x == 0u)
    {
        global_max = -INFINITY;
        for (partition = 0u; partition < partitions; ++partition)
        {
            float candidate = partials[
                block_base + (uint64_t)partition * (LATENT + 2u)];

            global_max = candidate > global_max ? candidate : global_max;
        }
        denominator = 0.0f;
        for (partition = 0u; partition < partitions; ++partition)
        {
            float partition_max = partials[
                block_base + (uint64_t)partition * (LATENT + 2u)];
            float scale = (global_max == -INFINITY ||
                           partition_max == -INFINITY)
                ? 0.0f
                : __expf(partition_max - global_max);

            scales[partition] = scale;
            denominator = fmaf(
                partials[block_base + (uint64_t)partition * (LATENT + 2u) +
                         1u],
                scale,
                denominator);
        }
        denominator_shared = denominator;
    }
    __syncthreads();
    for (element = threadIdx.x; element < LATENT; element += THREADS)
    {
        float merged = 0.0f;

        for (partition = 0u; partition < partitions; ++partition)
        {
            merged = fmaf(
                partials[block_base + (uint64_t)partition * (LATENT + 2u) +
                         2u + element],
                scales[partition],
                merged);
        }
        output_bf16[((uint64_t)row * heads + head) * LATENT + element] =
            LmFloatToBf16(merged / fmaxf(denominator_shared, 1.0e-20f));
    }
}

template<class Geometry, uint32_t THREADS, uint32_t LATENT, uint32_t ROPE>
static inline cudaError_t LmLatentAttentionDecodeSplitLaunch(
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
    uint32_t blocks;
    uint32_t wanted;
    uint32_t partitions;

    blocks = rows * heads;
    wanted = multiprocessor_count == 0u || blocks == 0u
        ? 1u
        : (multiprocessor_count * LM_LATENT_ATTN_SPLIT_CTAS_PER_SM +
           blocks - 1u) / blocks;
    partitions = wanted < 1u ? 1u : wanted;
    if (partitions > LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS)
    {
        partitions = LM_LATENT_ATTN_SPLIT_MAX_PARTITIONS;
    }
    if (split_context_threshold == 0u || split_partials == 0 ||
        position_bound < split_context_threshold || partitions < 2u ||
        blocks == 0u ||
        (uint64_t)blocks * partitions > split_partial_blocks)
    {
        partitions = 1u;
    }
    if (partitions == 1u)
    {
        LM_LAUNCH(
            (LmLatentAttentionDecodeKernel<Geometry, THREADS, LATENT, ROPE>),
            dim3(rows, heads),
            THREADS,
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
            qk_scale,
            output_bf16,
            row_position);
        return cudaPeekAtLastError();
    }
    LM_LAUNCH(
        (LmLatentAttentionDecodeSplitKernel<
            Geometry, THREADS, LATENT, ROPE>),
        dim3(rows, heads, partitions),
        THREADS,
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
        row_position);
    if (cudaPeekAtLastError() != cudaSuccess)
    {
        return cudaPeekAtLastError();
    }
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

template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, uint32_t index_heads, float *__restrict__ scores)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,position = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot;
	float total = 0.0f;
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
			position,
			0xffffffffu);
		return;
	}
	if ( position >= context_length[sequence] )
		return;
	slot = LmKvSlotRequired<Geometry>(
		cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.y) + position] = total;
}

template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmWeightedSparseScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ head_weight_bf16, LmKvView index_cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ row_position, uint32_t index_heads, float qk_scale, float *__restrict__ scores)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t position = blockIdx.x,row = blockIdx.y,sequence,head,index;
	const uint8_t *slot;
	float total = 0.0f,partial;
	if ( scores == 0 )
		return;
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.x) + position] = -INFINITY;
	if ( sequence_of_row == 0 || context_length == 0 || index_query_bf16 == 0
		|| head_weight_bf16 == 0 || !LmKvViewIsConfigured(index_cache) )
	{
		LmKvReportRequiredAccessFailure(
			index_cache,
			LM_KV_ACCESS_ERROR_INVALID_VIEW,
			LM_KV_ACCESS_READ,
			row,
			0xffffffffu,
			position,
			0xffffffffu);
		return;
	}
	sequence = sequence_of_row[row];
	if ( sequence >= index_cache.sequence_count )
	{
		LmKvReportRequiredAccessFailure(
			index_cache,
			LM_KV_ACCESS_ERROR_SEQUENCE_OUT_OF_RANGE,
			LM_KV_ACCESS_READ,
			row,
			sequence,
			position,
			0xffffffffu);
		return;
	}
	if ( position >= context_length[sequence] || (row_position != 0 && position > row_position[row]) )
		return;
	slot = LmKvSlotRequired<Geometry>(index_cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; head++)
	{
		partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[(((uint64_t)row * index_heads) + head) * INDEX_DIM + index]) * LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction) * LmBf16ToFloat(head_weight_bf16[((uint64_t)row * index_heads) + head]);
	}
	if ( threadIdx.x == 0u )
		scores[((uint64_t)row * gridDim.x) + position] = total * qk_scale;
}

static __device__ __forceinline__ float LmYarnFrequency(uint32_t index, uint32_t rope_dimension, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	float exponent = 2.0f * (float)index / (float)rope_dimension;
	float inverse = __powf(theta,-exponent);
	float wavelength = 6.2831853f / inverse;
	float rotations = original_positions / wavelength;
	float ramp = (rotations - low_band) / fmaxf(high_band - low_band,1e-3f);
	float blend = fminf(fmaxf(ramp,0.0f),1.0f);
	return((inverse * blend) + ((inverse / scale_factor) * (1.0f - blend)));
}

template<uint32_t THREADS, LmRopePairing PAIRING = LM_ROPE_HALF_SPLIT>
__global__ __launch_bounds__(THREADS, 1)
void LmRopeYarnKernel(uint16_t *__restrict__ rows_bf16, const uint32_t *__restrict__ positions, uint32_t row_stride, uint32_t rope_offset, uint32_t rope_dim, float theta, float scale_factor, float original_positions, float low_band, float high_band)
{
	uint64_t base = ((uint64_t)blockIdx.x * row_stride) + rope_offset;
	uint32_t half = rope_dim / 2u,index;
	float position = (float)positions[blockIdx.x];
	for (index = threadIdx.x; index < half; index += THREADS)
		LmRopeRotate<PAIRING>(rows_bf16,base,index,half,position *
			LmYarnFrequency(index,rope_dim,theta,scale_factor,
				original_positions,low_band,high_band));
}

template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryBuildKernel(LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ dirty_block, uint32_t block_positions, uint16_t *__restrict__ summary_bf16, uint32_t blocks_per_sequence)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = dirty_block != 0 ? dirty_block[row] : blockIdx.y;
	uint32_t sequence = sequence_of_row[row],position,index;
	for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
	{
		float best = 0.0f;
		for (position = 0u; position < block_positions; ++position)
		{
			uint32_t absolute_position = (block * block_positions) + position;
			const uint8_t *slot;
			if ( absolute_position >= context_length[sequence] )
				continue;
			slot = LmKvSlotRequired<Geometry>(
				cache,sequence,absolute_position,row,LM_KV_ACCESS_READ);
			if ( slot == 0 )
				return;
			best = fmaxf(best,fabsf(LmBf16ToFloat(((const uint16_t *)slot)[index])));
		}
		summary_bf16[(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index] =
			LmFloatToBf16(best);
	}
	(void)reduction;
}

template<uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseSummaryScoreKernel(const uint16_t *__restrict__ index_query_bf16, const uint16_t *__restrict__ summary_bf16, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ block_count, uint32_t index_heads, uint32_t blocks_per_sequence, float *__restrict__ block_score)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,block = blockIdx.y;
	uint32_t sequence = sequence_of_row[row],head,index;
	float total = 0.0f;
	if ( block >= block_count[sequence] )
	{
		if ( threadIdx.x == 0u )
			block_score[(row * blocks_per_sequence) + block] = -INFINITY;
		return;
	}
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += fabsf(LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index]))
				* LmBf16ToFloat(summary_bf16[
				(((uint64_t)sequence * blocks_per_sequence) + block) * INDEX_DIM + index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
		block_score[(row * blocks_per_sequence) + block] = total;
}

template<class Geometry, uint32_t THREADS, uint32_t INDEX_DIM>
__global__ __launch_bounds__(THREADS, 1)
void LmSparseRefineKernel(const uint16_t *__restrict__ index_query_bf16, LmKvView cache, const uint32_t *__restrict__ sequence_of_row, const uint32_t *__restrict__ context_length, const uint32_t *__restrict__ selected_block, uint32_t block_positions, uint32_t index_heads, float *__restrict__ scores, uint32_t *__restrict__ positions_out)
{
	__shared__ float reduction[THREADS / LM_WARP_LANES];
	uint32_t row = blockIdx.x,slot_index = blockIdx.y;
	uint32_t block = selected_block[(row * gridDim.y) + (slot_index / block_positions)];
	uint32_t position = (block * block_positions) + (slot_index % block_positions);
	uint32_t sequence = sequence_of_row[row],head,index;
	const uint8_t *slot;
	float total = 0.0f;
	if ( position >= context_length[sequence] )
	{
		if ( threadIdx.x == 0u )
		{
			scores[(row * gridDim.y) + slot_index] = -INFINITY;
			positions_out[(row * gridDim.y) + slot_index] = position;
		}
		return;
	}
	slot = LmKvSlotRequired<Geometry>(
		cache,sequence,position,row,LM_KV_ACCESS_READ);
	if ( slot == 0 )
		return;
	for (head = 0u; head < index_heads; ++head)
	{
		float partial = 0.0f;
		for (index = threadIdx.x; index < INDEX_DIM; index += THREADS)
			partial += LmBf16ToFloat(index_query_bf16[
				(((uint64_t)row * index_heads) + head) * INDEX_DIM + index])
				* LmBf16ToFloat(((const uint16_t *)slot)[index]);
		total += LmBlockSum<THREADS>(partial,reduction);
	}
	if ( threadIdx.x == 0u )
	{
		scores[(row * gridDim.y) + slot_index] = total;
		positions_out[(row * gridDim.y) + slot_index] = position;
	}
}
