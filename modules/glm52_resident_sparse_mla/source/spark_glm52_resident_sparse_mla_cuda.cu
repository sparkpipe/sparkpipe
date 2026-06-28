#include "spark_glm52_resident_sparse_mla_backend.h"

#include <cuda_runtime.h>

#include <float.h>
#include <stdint.h>

#define SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS 256u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES 32u
#define SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT UINT32_MAX

static __device__ __forceinline__ float SparkGlm52ResidentSparseMlaBf16ToFloat(
    uint16_t value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;

    conversion.bits = ((uint32_t)value) << 16u;
    return conversion.value;
}

static __device__ __forceinline__ uint16_t SparkGlm52ResidentSparseMlaFloatToBf16(
    float value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;
    uint32_t rounding_bias;

    conversion.value = value;
    rounding_bias = 0x7fffu + ((conversion.bits >> 16u) & 1u);
    return (uint16_t)((conversion.bits + rounding_bias) >> 16u);
}

static __device__ __forceinline__ void SparkGlm52ResidentSparseMlaApplyRopePair(
    uint16_t first_input_bf16,
    uint16_t second_input_bf16,
    float cosine,
    float sine,
    uint16_t *first_output_bf16,
    uint16_t *second_output_bf16)
{
    float first_input;
    float second_input;

    first_input = SparkGlm52ResidentSparseMlaBf16ToFloat(first_input_bf16);
    second_input = SparkGlm52ResidentSparseMlaBf16ToFloat(second_input_bf16);
    *first_output_bf16 = SparkGlm52ResidentSparseMlaFloatToBf16(
        (first_input * cosine) - (second_input * sine));
    *second_output_bf16 = SparkGlm52ResidentSparseMlaFloatToBf16(
        (first_input * sine) + (second_input * cosine));
}

static __global__ void SparkGlm52ResidentSparseMlaPrepareKernel(
    const uint16_t *__restrict__ query_rope_input_bf16,
    const uint16_t *__restrict__ key_rope_input_bf16,
    const uint16_t *__restrict__ current_kv_latent_bf16,
    const uint32_t *__restrict__ positions,
    const uint32_t *__restrict__ slot_mapping,
    const float *__restrict__ cos_table,
    const float *__restrict__ sin_table,
    uint16_t *__restrict__ rotated_query_rope_bf16,
    uint16_t *__restrict__ mla_cache_bf16,
    uint32_t active_sequence_count,
    uint32_t position_count,
    uint32_t cache_token_capacity)
{
    const uint32_t rope_pair_count =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u;
    uint64_t query_rope_work_count;
    uint64_t cache_latent_work_count;
    uint64_t cache_rope_work_count;
    uint64_t total_work_count;
    uint64_t work_index;
    uint64_t work_stride;

    query_rope_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        (uint64_t)rope_pair_count;
    cache_latent_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
    cache_rope_work_count =
        (uint64_t)active_sequence_count *
        (uint64_t)rope_pair_count;
    total_work_count =
        query_rope_work_count +
        cache_latent_work_count +
        cache_rope_work_count;
    work_index =
        ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) +
        (uint64_t)threadIdx.x;
    work_stride = (uint64_t)gridDim.x * (uint64_t)blockDim.x;

    while (work_index < total_work_count)
    {
        if (work_index < query_rope_work_count)
        {
            uint64_t query_row_index;
            uint32_t rope_pair_index;
            uint32_t sequence_index;
            uint32_t position;
            uint64_t input_offset;
            uint64_t table_offset;

            query_row_index = work_index / (uint64_t)rope_pair_count;
            rope_pair_index =
                (uint32_t)(work_index % (uint64_t)rope_pair_count);
            sequence_index = (uint32_t)(
                query_row_index /
                (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT);
            position = positions[sequence_index];
            input_offset =
                (query_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                ((uint64_t)rope_pair_index * 2u);
            if (position < position_count)
            {
                table_offset =
                    ((uint64_t)position * (uint64_t)rope_pair_count) +
                    (uint64_t)rope_pair_index;
                SparkGlm52ResidentSparseMlaApplyRopePair(
                    query_rope_input_bf16[input_offset],
                    query_rope_input_bf16[input_offset + 1u],
                    cos_table[table_offset],
                    sin_table[table_offset],
                    &rotated_query_rope_bf16[input_offset],
                    &rotated_query_rope_bf16[input_offset + 1u]);
            }
            else
            {
                rotated_query_rope_bf16[input_offset] = 0u;
                rotated_query_rope_bf16[input_offset + 1u] = 0u;
            }
        }
        else if (work_index <
                 query_rope_work_count + cache_latent_work_count)
        {
            uint64_t cache_work_index;
            uint32_t sequence_index;
            uint32_t latent_dimension_index;
            uint32_t cache_slot_index;
            uint64_t cache_offset;

            cache_work_index = work_index - query_rope_work_count;
            sequence_index = (uint32_t)(
                cache_work_index /
                (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION);
            latent_dimension_index = (uint32_t)(
                cache_work_index %
                (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION);
            cache_slot_index = slot_mapping[sequence_index];
            if (cache_slot_index < cache_token_capacity)
            {
                cache_offset =
                    ((uint64_t)cache_slot_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                    (uint64_t)latent_dimension_index;
                mla_cache_bf16[cache_offset] = current_kv_latent_bf16[
                    ((uint64_t)sequence_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
                    (uint64_t)latent_dimension_index];
            }
        }
        else
        {
            uint64_t cache_rope_work_index;
            uint32_t sequence_index;
            uint32_t rope_pair_index;
            uint32_t cache_slot_index;
            uint32_t position;
            uint64_t input_offset;
            uint64_t cache_offset;
            uint64_t table_offset;

            cache_rope_work_index =
                work_index - query_rope_work_count - cache_latent_work_count;
            sequence_index = (uint32_t)(
                cache_rope_work_index / (uint64_t)rope_pair_count);
            rope_pair_index = (uint32_t)(
                cache_rope_work_index % (uint64_t)rope_pair_count);
            cache_slot_index = slot_mapping[sequence_index];
            position = positions[sequence_index];
            if (cache_slot_index < cache_token_capacity &&
                position < position_count)
            {
                input_offset =
                    ((uint64_t)sequence_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                    ((uint64_t)rope_pair_index * 2u);
                cache_offset =
                    ((uint64_t)cache_slot_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                    (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION +
                    ((uint64_t)rope_pair_index * 2u);
                table_offset =
                    ((uint64_t)position * (uint64_t)rope_pair_count) +
                    (uint64_t)rope_pair_index;
                SparkGlm52ResidentSparseMlaApplyRopePair(
                    key_rope_input_bf16[input_offset],
                    key_rope_input_bf16[input_offset + 1u],
                    cos_table[table_offset],
                    sin_table[table_offset],
                    &mla_cache_bf16[cache_offset],
                    &mla_cache_bf16[cache_offset + 1u]);
            }
        }
        work_index += work_stride;
    }
}

static __device__ __forceinline__ float SparkGlm52ResidentSparseMlaWarpReduceSum(
    float value)
{
    value += __shfl_down_sync(0xffffffffu, value, 16);
    value += __shfl_down_sync(0xffffffffu, value, 8);
    value += __shfl_down_sync(0xffffffffu, value, 4);
    value += __shfl_down_sync(0xffffffffu, value, 2);
    value += __shfl_down_sync(0xffffffffu, value, 1);
    return value;
}

static __device__ float SparkGlm52ResidentSparseMlaBlockReduceMax(
    float value,
    float *shared_reduction)
{
    uint32_t stride;

    shared_reduction[threadIdx.x] = value;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride &&
            shared_reduction[threadIdx.x + stride] >
                shared_reduction[threadIdx.x])
        {
            shared_reduction[threadIdx.x] =
                shared_reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared_reduction[0];
}

static __device__ float SparkGlm52ResidentSparseMlaBlockReduceSum(
    float value,
    float *shared_reduction)
{
    uint32_t stride;

    shared_reduction[threadIdx.x] = value;
    __syncthreads();
    for (stride = blockDim.x >> 1u; stride != 0u; stride >>= 1u)
    {
        if (threadIdx.x < stride)
        {
            shared_reduction[threadIdx.x] +=
                shared_reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return shared_reduction[0];
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentSparseMlaResolveCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity)
{
    uint64_t addressed_token_index;
    uint64_t logical_block_index;
    uint64_t block_token_offset;
    uint64_t block_table_offset;
    uint32_t physical_block_index;
    uint64_t cache_slot_index;

    addressed_token_index =
        (uint64_t)first_block_token_offset + (uint64_t)token_index;
    logical_block_index =
        addressed_token_index /
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS;
    if (logical_block_index >= (uint64_t)max_blocks_per_sequence)
    {
        return SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
    }
    block_token_offset =
        addressed_token_index %
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS;
    block_table_offset =
        ((uint64_t)sequence_index *
         (uint64_t)max_blocks_per_sequence) +
        logical_block_index;
    physical_block_index = block_table[block_table_offset];
    if (physical_block_index >= kv_block_count)
    {
        return SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
    }
    cache_slot_index =
        ((uint64_t)physical_block_index *
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS) +
        block_token_offset;
    if (cache_slot_index >= (uint64_t)cache_token_capacity ||
        cache_slot_index > (uint64_t)UINT32_MAX)
    {
        return SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
    }
    return (uint32_t)cache_slot_index;
}

static __device__ __forceinline__ uint32_t SparkGlm52ResidentSparseMlaWarpResolveCacheSlot(
    const uint32_t *block_table,
    uint32_t sequence_index,
    uint32_t token_index,
    uint32_t first_block_token_offset,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    uint32_t lane_index)
{
    uint32_t cache_slot_index;

    cache_slot_index = SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
    if (lane_index == 0u)
    {
        cache_slot_index = SparkGlm52ResidentSparseMlaResolveCacheSlot(
            block_table,
            sequence_index,
            token_index,
            first_block_token_offset,
            max_blocks_per_sequence,
            kv_block_count,
            cache_token_capacity);
    }
    return __shfl_sync(0xffffffffu, cache_slot_index, 0);
}

static __device__ __forceinline__ float SparkGlm52ResidentSparseMlaWarpDotProduct(
    const float *shared_query,
    const uint16_t *mla_cache_bf16,
    uint32_t cache_slot_index,
    uint32_t lane_index,
    float qk_scale)
{
    const uint32_t *cache_pairs;
    uint64_t cache_element_offset;
    float accumulated_dot_product;
    uint32_t pair_index;

    cache_element_offset =
        (uint64_t)cache_slot_index *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
    cache_pairs = (const uint32_t *)&mla_cache_bf16[cache_element_offset];
    accumulated_dot_product = 0.0f;
    for (pair_index = lane_index;
         pair_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS / 2u;
         pair_index += SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES)
    {
        uint32_t packed_values;
        uint32_t first_dimension;

        packed_values = cache_pairs[pair_index];
        first_dimension = pair_index * 2u;
        accumulated_dot_product +=
            shared_query[first_dimension] *
            SparkGlm52ResidentSparseMlaBf16ToFloat(
                (uint16_t)(packed_values & 0xffffu));
        accumulated_dot_product +=
            shared_query[first_dimension + 1u] *
            SparkGlm52ResidentSparseMlaBf16ToFloat(
                (uint16_t)(packed_values >> 16u));
    }
    return SparkGlm52ResidentSparseMlaWarpReduceSum(
        accumulated_dot_product) * qk_scale;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS, 2)
void SparkGlm52ResidentSparseMlaAttentionKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint16_t *__restrict__ output_latent_bf16,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_query[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS];
    __shared__ float shared_scores[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT];
    __shared__ uint32_t shared_cache_slots[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT];
    __shared__ float shared_reduction[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS];
    uint32_t sequence_index;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint64_t query_row_index;
    uint64_t sparse_row_offset;
    uint64_t output_row_offset;
    uint32_t dimension_index;
    uint32_t candidate_index;
    float local_maximum;
    float row_maximum;
    float local_exponential_sum;
    float row_exponential_sum;

    sequence_index = blockIdx.x;
    head_index = blockIdx.y;
    lane_index =
        threadIdx.x &
        (SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES - 1u);
    warp_index =
        threadIdx.x / SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES;
    warp_count =
        blockDim.x / SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES;
    context_length = context_lengths[sequence_index];
    first_block_token_offset = first_block_token_offsets[sequence_index];
    query_row_index =
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT) +
        (uint64_t)head_index;
    sparse_row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
    output_row_offset =
        query_row_index *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;

    for (dimension_index = threadIdx.x;
         dimension_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[dimension_index] =
            SparkGlm52ResidentSparseMlaBf16ToFloat(
                query_latent_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    for (dimension_index = threadIdx.x;
         dimension_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[
            SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION +
            dimension_index] = SparkGlm52ResidentSparseMlaBf16ToFloat(
                rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    __syncthreads();

    local_maximum = -FLT_MAX;
    for (candidate_index = warp_index;
         candidate_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
         candidate_index += warp_count)
    {
        uint32_t token_index;
        uint32_t cache_slot_index;
        float attention_score;

        token_index = sparse_token_indices[
            sparse_row_offset + (uint64_t)candidate_index];
        cache_slot_index =
            SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
        if (token_index < context_length)
        {
            cache_slot_index =
                SparkGlm52ResidentSparseMlaWarpResolveCacheSlot(
                    block_table,
                    sequence_index,
                    token_index,
                    first_block_token_offset,
                    max_blocks_per_sequence,
                    kv_block_count,
                    cache_token_capacity,
                    lane_index);
        }
        attention_score =
            cache_slot_index !=
                SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT
            ? SparkGlm52ResidentSparseMlaWarpDotProduct(
                  shared_query,
                  mla_cache_bf16,
                  cache_slot_index,
                  lane_index,
                  qk_scale)
            : -FLT_MAX;
        if (lane_index == 0u)
        {
            shared_scores[candidate_index] = attention_score;
            shared_cache_slots[candidate_index] = cache_slot_index;
            if (attention_score > local_maximum)
            {
                local_maximum = attention_score;
            }
        }
    }

    row_maximum = SparkGlm52ResidentSparseMlaBlockReduceMax(
        local_maximum,
        shared_reduction);
    local_exponential_sum = 0.0f;
    for (candidate_index = threadIdx.x;
         candidate_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
         candidate_index += blockDim.x)
    {
        float exponential_score;

        exponential_score = row_maximum <= (-FLT_MAX * 0.5f)
            ? 0.0f
            : __expf(shared_scores[candidate_index] - row_maximum);
        shared_scores[candidate_index] = exponential_score;
        local_exponential_sum += exponential_score;
    }
    row_exponential_sum = SparkGlm52ResidentSparseMlaBlockReduceSum(
        local_exponential_sum,
        shared_reduction);

    for (dimension_index = threadIdx.x;
         dimension_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
         dimension_index += blockDim.x)
    {
        float accumulated_value;

        accumulated_value = 0.0f;
        if (row_exponential_sum > 0.0f)
        {
            for (candidate_index = 0u;
                 candidate_index <
                     SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
                 ++candidate_index)
            {
                uint32_t cache_slot_index;

                cache_slot_index = shared_cache_slots[candidate_index];
                if (cache_slot_index !=
                    SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT)
                {
                    uint64_t cache_element_offset;

                    cache_element_offset =
                        ((uint64_t)cache_slot_index *
                         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                        (uint64_t)dimension_index;
                    accumulated_value +=
                        (shared_scores[candidate_index] /
                         row_exponential_sum) *
                        SparkGlm52ResidentSparseMlaBf16ToFloat(
                            mla_cache_bf16[cache_element_offset]);
                }
            }
        }
        output_latent_bf16[
            output_row_offset + (uint64_t)dimension_index] =
            SparkGlm52ResidentSparseMlaFloatToBf16(accumulated_value);
    }
}


static __device__ __forceinline__ void SparkGlm52ResidentSparseMlaOnlineAccumulate(
    float attention_score,
    float value,
    float *online_maximum,
    float *online_denominator,
    float *accumulated_value)
{
    float next_maximum;
    float old_scale;
    float score_scale;

    next_maximum = attention_score > *online_maximum
        ? attention_score
        : *online_maximum;
    old_scale = *online_denominator > 0.0f
        ? __expf(*online_maximum - next_maximum)
        : 0.0f;
    score_scale = __expf(attention_score - next_maximum);
    *accumulated_value = (*accumulated_value * old_scale) +
        (value * score_scale);
    *online_denominator = (*online_denominator * old_scale) + score_scale;
    *online_maximum = next_maximum;
}

static __global__ __launch_bounds__(SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS, 2)
void SparkGlm52ResidentSparseMlaAttentionOnlineKernel(
    const uint16_t *__restrict__ query_latent_bf16,
    const uint16_t *__restrict__ rotated_query_rope_bf16,
    const uint16_t *__restrict__ mla_cache_bf16,
    const uint32_t *__restrict__ block_table,
    const uint32_t *__restrict__ context_lengths,
    const uint32_t *__restrict__ first_block_token_offsets,
    const uint32_t *__restrict__ sparse_token_indices,
    uint16_t *__restrict__ output_latent_bf16,
    uint32_t max_blocks_per_sequence,
    uint32_t kv_block_count,
    uint32_t cache_token_capacity,
    float qk_scale)
{
    __shared__ float shared_query[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS];
    __shared__ float shared_tile_scores[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES];
    __shared__ uint32_t shared_tile_cache_slots[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS /
        SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES];
    uint32_t sequence_index;
    uint32_t head_index;
    uint32_t lane_index;
    uint32_t warp_index;
    uint32_t warp_count;
    uint32_t context_length;
    uint32_t first_block_token_offset;
    uint64_t query_row_index;
    uint64_t sparse_row_offset;
    uint64_t output_row_offset;
    uint32_t dimension_index;
    uint32_t candidate_base;
    uint32_t first_output_dimension;
    uint32_t second_output_dimension;
    float first_maximum;
    float first_denominator;
    float first_accumulated_value;
    float second_maximum;
    float second_denominator;
    float second_accumulated_value;

    sequence_index = blockIdx.x;
    head_index = blockIdx.y;
    lane_index = threadIdx.x &
        (SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES - 1u);
    warp_index = threadIdx.x / SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES;
    warp_count = blockDim.x / SPARK_GLM52_RESIDENT_SPARSE_MLA_WARP_LANES;
    context_length = context_lengths[sequence_index];
    first_block_token_offset = first_block_token_offsets[sequence_index];
    query_row_index =
        ((uint64_t)sequence_index *
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT) +
        (uint64_t)head_index;
    sparse_row_offset =
        (uint64_t)sequence_index *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
    output_row_offset =
        query_row_index *
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;

    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[dimension_index] = SparkGlm52ResidentSparseMlaBf16ToFloat(
            query_latent_bf16[
                (query_row_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
                (uint64_t)dimension_index]);
    }
    for (dimension_index = threadIdx.x;
         dimension_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
         dimension_index += blockDim.x)
    {
        shared_query[
            SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION +
            dimension_index] = SparkGlm52ResidentSparseMlaBf16ToFloat(
                rotated_query_rope_bf16[
                    (query_row_index *
                     (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                    (uint64_t)dimension_index]);
    }
    __syncthreads();

    first_output_dimension = threadIdx.x;
    second_output_dimension = threadIdx.x + blockDim.x;
    first_maximum = -FLT_MAX;
    first_denominator = 0.0f;
    first_accumulated_value = 0.0f;
    second_maximum = -FLT_MAX;
    second_denominator = 0.0f;
    second_accumulated_value = 0.0f;

    for (candidate_base = 0u;
         candidate_base < SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
         candidate_base += warp_count)
    {
        uint32_t candidate_index;
        uint32_t token_index;
        uint32_t cache_slot_index;
        float attention_score;
        uint32_t tile_index;

        candidate_index = candidate_base + warp_index;
        cache_slot_index = SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT;
        attention_score = -FLT_MAX;
        if (candidate_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT)
        {
            token_index = sparse_token_indices[
                sparse_row_offset + (uint64_t)candidate_index];
            if (token_index < context_length)
            {
                cache_slot_index = SparkGlm52ResidentSparseMlaWarpResolveCacheSlot(
                    block_table,
                    sequence_index,
                    token_index,
                    first_block_token_offset,
                    max_blocks_per_sequence,
                    kv_block_count,
                    cache_token_capacity,
                    lane_index);
            }
            if (cache_slot_index != SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT)
            {
                attention_score = SparkGlm52ResidentSparseMlaWarpDotProduct(
                    shared_query,
                    mla_cache_bf16,
                    cache_slot_index,
                    lane_index,
                    qk_scale);
            }
        }
        if (lane_index == 0u)
        {
            shared_tile_scores[warp_index] = attention_score;
            shared_tile_cache_slots[warp_index] = cache_slot_index;
        }
        __syncthreads();

        for (tile_index = 0u; tile_index < warp_count; ++tile_index)
        {
            float tile_score;
            uint32_t tile_cache_slot;

            tile_score = shared_tile_scores[tile_index];
            tile_cache_slot = shared_tile_cache_slots[tile_index];
            if (tile_cache_slot != SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_CACHE_SLOT &&
                tile_score > (-FLT_MAX * 0.5f))
            {
                uint64_t cache_element_offset;

                if (first_output_dimension <
                    SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION)
                {
                    cache_element_offset =
                        ((uint64_t)tile_cache_slot *
                         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                        (uint64_t)first_output_dimension;
                    SparkGlm52ResidentSparseMlaOnlineAccumulate(
                        tile_score,
                        SparkGlm52ResidentSparseMlaBf16ToFloat(
                            mla_cache_bf16[cache_element_offset]),
                        &first_maximum,
                        &first_denominator,
                        &first_accumulated_value);
                }
                if (second_output_dimension <
                    SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION)
                {
                    cache_element_offset =
                        ((uint64_t)tile_cache_slot *
                         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                        (uint64_t)second_output_dimension;
                    SparkGlm52ResidentSparseMlaOnlineAccumulate(
                        tile_score,
                        SparkGlm52ResidentSparseMlaBf16ToFloat(
                            mla_cache_bf16[cache_element_offset]),
                        &second_maximum,
                        &second_denominator,
                        &second_accumulated_value);
                }
            }
        }
        __syncthreads();
    }

    if (first_output_dimension < SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION)
    {
        output_latent_bf16[output_row_offset + (uint64_t)first_output_dimension] =
            SparkGlm52ResidentSparseMlaFloatToBf16(
                first_denominator > 0.0f
                    ? first_accumulated_value / first_denominator
                    : 0.0f);
    }
    if (second_output_dimension < SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION)
    {
        output_latent_bf16[output_row_offset + (uint64_t)second_output_dimension] =
            SparkGlm52ResidentSparseMlaFloatToBf16(
                second_denominator > 0.0f
                    ? second_accumulated_value / second_denominator
                    : 0.0f);
    }
}

static uint32_t SparkGlm52ResidentSparseMlaPrepareBlockCount(
    uint32_t active_sequence_count)
{
    uint64_t rope_pair_count;
    uint64_t total_work_count;
    uint64_t block_count;

    rope_pair_count =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u;
    total_work_count =
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
         rope_pair_count) +
        ((uint64_t)active_sequence_count *
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
        ((uint64_t)active_sequence_count * rope_pair_count);
    block_count =
        (total_work_count +
         (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS - 1u) /
        (uint64_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS;
    if (block_count == 0u)
    {
        block_count = 1u;
    }
    if (block_count > 65535u)
    {
        block_count = 65535u;
    }
    return (uint32_t)block_count;
}

static SparkStatus SparkGlm52ResidentSparseMlaCheckCudaLaunch(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context,
    SparkGlm52ResidentSparseMlaCudaPipelineSlotState *cuda_slot_state,
    cudaStream_t cuda_stream)
{
    cudaError_t cuda_status;

    if (node_context == 0 ||
        node_context->launch_check_mode ==
            SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_NONE)
    {
        return SPARK_STATUS_OK;
    }

    cuda_status = node_context->launch_check_mode ==
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_PEEK
        ? cudaPeekAtLastError()
        : cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->launch_error_count += 1u;
        }
        if (node_context->launch_check_mode ==
            SPARK_GLM52_RESIDENT_SPARSE_MLA_LAUNCH_CHECK_SYNC_ON_ERROR)
        {
            cudaStreamSynchronize(cuda_stream);
        }
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

static void CUDART_CB SparkGlm52ResidentSparseMlaCudaCompletion(
    void *completion_context)
{
    SparkGlm52ResidentSparseMlaBackendCompletion *completion;

    completion =
        (SparkGlm52ResidentSparseMlaBackendCompletion *)completion_context;
    if (completion != 0 && completion->function != 0)
    {
        completion->function(completion->context);
    }
}

static SparkStatus SparkGlm52ResidentSparseMlaEnqueueCompletion(
    cudaStream_t cuda_stream,
    SparkGlm52ResidentSparseMlaCudaPipelineSlotState *cuda_slot_state,
    SparkGlm52ResidentSparseMlaBackendCompletion *completion)
{
    cudaError_t cuda_status;

    cuda_status = cudaLaunchHostFunc(
        cuda_stream,
        SparkGlm52ResidentSparseMlaCudaCompletion,
        completion);
    if (cuda_status != cudaSuccess)
    {
        if (cuda_slot_state != 0)
        {
            cuda_slot_state->launch_error_count += 1u;
        }
        cudaStreamSynchronize(cuda_stream);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkGlm52ResidentSparseMlaBackendSubmit(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context,
    uint32_t pipeline_slot_index,
    uint32_t active_sequence_count,
    SparkGlm52ResidentSparseMlaBackendCompletion *completion)
{
    const SparkGlm52ResidentSparseMlaPipelineSlot *pipeline_slot;
    SparkGlm52ResidentSparseMlaCudaPipelineSlotState *cuda_slot_state;
    cudaStream_t cuda_stream;
    cudaError_t cuda_status;
    cudaGraph_t captured_graph;
    cudaGraphExec_t captured_graph_exec;
    uint32_t graph_capture_active;
    dim3 attention_grid;
    SparkStatus status;

    if (node_context == 0 || completion == 0 ||
        completion->function == 0 ||
        pipeline_slot_index >= node_context->pipeline_slot_count ||
        active_sequence_count == 0u ||
        active_sequence_count > node_context->max_active_sequence_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    pipeline_slot = &node_context->pipeline_slots[pipeline_slot_index];
    cuda_slot_state = node_context->cuda_pipeline_slot_states != 0
        ? &node_context->cuda_pipeline_slot_states[pipeline_slot_index]
        : 0;
    cuda_stream = (cudaStream_t)pipeline_slot->cuda_stream;
    captured_graph = 0;
    captured_graph_exec = 0;
    graph_capture_active = 0u;

    if (node_context->enable_cuda_graph_replay != 0u &&
        cuda_slot_state != 0 &&
        cuda_slot_state->cuda_graph_exec != 0 &&
        cuda_slot_state->graph_active_sequence_count == active_sequence_count)
    {
        cuda_status = cudaGraphLaunch(
            (cudaGraphExec_t)cuda_slot_state->cuda_graph_exec,
            cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->graph_replay_count += 1u;
        return SparkGlm52ResidentSparseMlaEnqueueCompletion(
            cuda_stream,
            cuda_slot_state,
            completion);
    }

    if (node_context->enable_cuda_graph_replay != 0u && cuda_slot_state != 0)
    {
        if (cuda_slot_state->cuda_graph_exec != 0)
        {
            cudaGraphExecDestroy((cudaGraphExec_t)cuda_slot_state->cuda_graph_exec);
            cuda_slot_state->cuda_graph_exec = 0;
            cuda_slot_state->graph_active_sequence_count = 0u;
        }
        cuda_status = cudaStreamBeginCapture(
            cuda_stream,
            cudaStreamCaptureModeThreadLocal);
        if (cuda_status == cudaSuccess)
        {
            graph_capture_active = 1u;
        }
        else
        {
            cuda_slot_state->launch_error_count += 1u;
            return SPARK_STATUS_INTERNAL_ERROR;
        }
    }

    if (cuda_slot_state != 0)
    {
        cuda_slot_state->launch_chain_count += 1u;
    }

    SparkGlm52ResidentSparseMlaPrepareKernel<<<
        SparkGlm52ResidentSparseMlaPrepareBlockCount(active_sequence_count),
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS,
        0u,
        cuda_stream>>>(
        (const uint16_t *)pipeline_slot->query_rope_input_bf16,
        (const uint16_t *)pipeline_slot->key_rope_input_bf16,
        (const uint16_t *)pipeline_slot->current_kv_latent_bf16,
        pipeline_slot->positions,
        pipeline_slot->slot_mapping,
        node_context->cos_table,
        node_context->sin_table,
        (uint16_t *)pipeline_slot->rotated_query_rope_bf16,
        (uint16_t *)node_context->mla_cache_bf16,
        active_sequence_count,
        node_context->position_count,
        node_context->cache_token_capacity);
    status = SparkGlm52ResidentSparseMlaCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    attention_grid = dim3(
        active_sequence_count,
        SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT,
        1u);
    if (node_context->attention_execution_mode ==
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ATTENTION_EXECUTION_TILED_ONLINE_SOFTMAX)
    {
        SparkGlm52ResidentSparseMlaAttentionOnlineKernel<<<
            attention_grid,
            SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->query_latent_bf16,
            (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
            (const uint16_t *)node_context->mla_cache_bf16,
            pipeline_slot->block_table,
            pipeline_slot->context_lengths,
            pipeline_slot->first_block_token_offsets,
            pipeline_slot->sparse_token_indices,
            (uint16_t *)pipeline_slot->output_latent_bf16,
            node_context->max_blocks_per_sequence,
            node_context->kv_block_count,
            node_context->cache_token_capacity,
            node_context->qk_scale);
    }
    else
    {
        SparkGlm52ResidentSparseMlaAttentionKernel<<<
            attention_grid,
            SPARK_GLM52_RESIDENT_SPARSE_MLA_CUDA_THREADS,
            0u,
            cuda_stream>>>(
            (const uint16_t *)pipeline_slot->query_latent_bf16,
            (const uint16_t *)pipeline_slot->rotated_query_rope_bf16,
            (const uint16_t *)node_context->mla_cache_bf16,
            pipeline_slot->block_table,
            pipeline_slot->context_lengths,
            pipeline_slot->first_block_token_offsets,
            pipeline_slot->sparse_token_indices,
            (uint16_t *)pipeline_slot->output_latent_bf16,
            node_context->max_blocks_per_sequence,
            node_context->kv_block_count,
            node_context->cache_token_capacity,
            node_context->qk_scale);
    }
    status = SparkGlm52ResidentSparseMlaCheckCudaLaunch(
        node_context,
        cuda_slot_state,
        cuda_stream);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if (graph_capture_active != 0u)
    {
        cuda_status = cudaStreamEndCapture(cuda_stream, &captured_graph);
        if (cuda_status != cudaSuccess)
        {
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_status = cudaGraphInstantiate(
            &captured_graph_exec,
            captured_graph,
            0,
            0,
            0);
        cudaGraphDestroy(captured_graph);
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->cuda_graph_exec = (void *)captured_graph_exec;
        cuda_slot_state->graph_active_sequence_count = active_sequence_count;
        cuda_slot_state->graph_capture_count += 1u;
        cuda_status = cudaGraphLaunch(captured_graph_exec, cuda_stream);
        if (cuda_status != cudaSuccess)
        {
            cuda_slot_state->launch_error_count += 1u;
            cudaStreamSynchronize(cuda_stream);
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        cuda_slot_state->graph_replay_count += 1u;
    }
    return SparkGlm52ResidentSparseMlaEnqueueCompletion(
        cuda_stream,
        cuda_slot_state,
        completion);
}

extern "C" void SparkGlm52ResidentSparseMlaBackendQuiesce(
    const SparkGlm52ResidentSparseMlaNodeContext *node_context)
{
    uint32_t pipeline_slot_index;

    if (node_context == 0 || node_context->pipeline_slots == 0)
    {
        return;
    }
    for (pipeline_slot_index = 0u;
         pipeline_slot_index < node_context->pipeline_slot_count;
         ++pipeline_slot_index)
    {
        cudaStreamSynchronize((cudaStream_t)
            node_context->pipeline_slots[pipeline_slot_index].cuda_stream);
        if (node_context->cuda_pipeline_slot_states != 0 &&
            node_context->cuda_pipeline_slot_states[pipeline_slot_index].cuda_graph_exec != 0)
        {
            cudaGraphExecDestroy((cudaGraphExec_t)
                node_context->cuda_pipeline_slot_states[
                    pipeline_slot_index].cuda_graph_exec);
            node_context->cuda_pipeline_slot_states[
                pipeline_slot_index].cuda_graph_exec = 0;
            node_context->cuda_pipeline_slot_states[
                pipeline_slot_index].graph_active_sequence_count = 0u;
        }
    }
}
