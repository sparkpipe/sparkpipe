#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>

#include <atomic>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sparkpipe/spark_glm52_resident_sparse_mla_firmware.h"

#define SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT 1u
#define SPARK_VALIDATION_CONTEXT_TOKEN_COUNT 3u
#define SPARK_VALIDATION_CACHE_TOKEN_CAPACITY 128u
#define SPARK_VALIDATION_KV_BLOCK_COUNT 2u
#define SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE 2u
#define SPARK_VALIDATION_POSITION_COUNT 64u
#define SPARK_VALIDATION_POSITION 42u
#define SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET 63u
#define SPARK_VALIDATION_CURRENT_CACHE_SLOT 1u
#define SPARK_VALIDATION_WARMUP_COUNT 3u
#define SPARK_VALIDATION_MEASUREMENT_COUNT 10u
#define SPARK_VALIDATION_WAIT_TIMEOUT_SECONDS 10.0
#define SPARK_VALIDATION_TIMEOUT_CHECK_SPIN_COUNT 1024u
#define SPARK_VALIDATION_OUTPUT_TOLERANCE 0.05f
#define SPARK_VALIDATION_QK_SCALE 0.0416666679f

typedef struct SparkValidationCompletionState
{
    std::atomic<uint32_t> completion_count;
    SparkModelDriverCompletion last_completion;
} SparkValidationCompletionState;

typedef struct SparkValidationHostBuffers
{
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_input_bf16;
    uint16_t *key_rope_input_bf16;
    uint16_t *current_kv_latent_bf16;
    uint16_t *input_cache_bf16;
    uint16_t *reference_cache_bf16;
    uint16_t *rotated_query_rope_bf16;
    uint16_t *output_latent_bf16;
    uint16_t *device_cache_result_bf16;
    uint16_t *device_rotated_query_result_bf16;
    uint16_t *device_output_result_bf16;
    float *expected_output;
    float *cos_table;
    float *sin_table;
    uint32_t *sparse_token_indices;
} SparkValidationHostBuffers;

typedef struct SparkValidationDeviceBuffers
{
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_input_bf16;
    uint16_t *key_rope_input_bf16;
    uint16_t *current_kv_latent_bf16;
    uint16_t *mla_cache_bf16;
    uint16_t *rotated_query_rope_bf16;
    uint16_t *output_latent_bf16;
    float *cos_table;
    float *sin_table;
    uint32_t *positions;
    uint32_t *slot_mapping;
    uint32_t *block_table;
    uint32_t *context_lengths;
    uint32_t *first_block_token_offsets;
    uint32_t *sparse_token_indices;
} SparkValidationDeviceBuffers;

static uint16_t SparkValidationFloatToBf16(float value)
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

static float SparkValidationBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t bits;
        float value;
    } conversion;

    conversion.bits = ((uint32_t)value) << 16u;
    return conversion.value;
}

static double SparkValidationMonotonicSeconds(void)
{
    struct timespec timestamp;

    if (clock_gettime(CLOCK_MONOTONIC, &timestamp) != 0)
    {
        return 0.0;
    }
    return (double)timestamp.tv_sec + ((double)timestamp.tv_nsec * 1.0e-9);
}

static bool SparkValidationCudaSucceeded(
    cudaError_t cuda_status,
    const char *operation)
{
    if (cuda_status == cudaSuccess)
    {
        return true;
    }
    fprintf(
        stderr,
        "%s failed: %s\n",
        operation,
        cudaGetErrorString(cuda_status));
    return false;
}

static void SparkValidationCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkValidationCompletionState *state;

    state = (SparkValidationCompletionState *)completion_context;
    if (state == 0 || completion == 0)
    {
        return;
    }
    state->last_completion = *completion;
    state->completion_count.fetch_add(1u, std::memory_order_release);
}

static bool SparkValidationWaitForCompletion(
    SparkValidationCompletionState *completion_state,
    uint32_t expected_completion_count)
{
    double deadline;
    uint32_t spin_count;

    deadline = SparkValidationMonotonicSeconds() +
        SPARK_VALIDATION_WAIT_TIMEOUT_SECONDS;
    spin_count = 0u;
    while (completion_state->completion_count.load(std::memory_order_acquire) <
           expected_completion_count)
    {
        spin_count += 1u;
        if (spin_count == SPARK_VALIDATION_TIMEOUT_CHECK_SPIN_COUNT)
        {
            spin_count = 0u;
            if (SparkValidationMonotonicSeconds() >= deadline)
            {
                return false;
            }
        }
    }
    return true;
}

static float SparkValidationPattern(uint64_t index, uint32_t multiplier)
{
    int32_t centered;

    centered = (int32_t)((index * multiplier) % 31u) - 15;
    return (float)centered * 0.015625f;
}

static uint32_t SparkValidationLogicalTokenToCacheSlot(uint32_t token_index)
{
    static const uint32_t BlockTable[
        SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE] = { 1u, 0u };
    uint32_t addressed_token_index;
    uint32_t logical_block_index;
    uint32_t block_token_offset;

    addressed_token_index =
        SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET + token_index;
    logical_block_index =
        addressed_token_index /
        SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS;
    block_token_offset =
        addressed_token_index %
        SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS;
    return
        (BlockTable[logical_block_index] *
         SPARK_GLM52_RESIDENT_SPARSE_MLA_BLOCK_TOKENS) +
        block_token_offset;
}

static bool SparkValidationAllocateHostBuffers(
    SparkValidationHostBuffers *buffers)
{
    const size_t query_latent_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
    const size_t query_rope_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
    const size_t cache_count =
        (size_t)SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
    const size_t rope_table_count =
        (size_t)SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u);

    memset(buffers, 0, sizeof(*buffers));
    buffers->query_latent_bf16 =
        (uint16_t *)calloc(query_latent_count, sizeof(uint16_t));
    buffers->query_rope_input_bf16 =
        (uint16_t *)calloc(query_rope_count, sizeof(uint16_t));
    buffers->key_rope_input_bf16 = (uint16_t *)calloc(
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION,
        sizeof(uint16_t));
    buffers->current_kv_latent_bf16 = (uint16_t *)calloc(
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION,
        sizeof(uint16_t));
    buffers->input_cache_bf16 =
        (uint16_t *)calloc(cache_count, sizeof(uint16_t));
    buffers->reference_cache_bf16 =
        (uint16_t *)calloc(cache_count, sizeof(uint16_t));
    buffers->rotated_query_rope_bf16 =
        (uint16_t *)calloc(query_rope_count, sizeof(uint16_t));
    buffers->output_latent_bf16 =
        (uint16_t *)calloc(query_latent_count, sizeof(uint16_t));
    buffers->device_cache_result_bf16 =
        (uint16_t *)calloc(cache_count, sizeof(uint16_t));
    buffers->device_rotated_query_result_bf16 =
        (uint16_t *)calloc(query_rope_count, sizeof(uint16_t));
    buffers->device_output_result_bf16 =
        (uint16_t *)calloc(query_latent_count, sizeof(uint16_t));
    buffers->expected_output =
        (float *)calloc(query_latent_count, sizeof(float));
    buffers->cos_table = (float *)calloc(rope_table_count, sizeof(float));
    buffers->sin_table = (float *)calloc(rope_table_count, sizeof(float));
    buffers->sparse_token_indices = (uint32_t *)calloc(
        SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT,
        sizeof(uint32_t));
    return buffers->query_latent_bf16 != 0 &&
        buffers->query_rope_input_bf16 != 0 &&
        buffers->key_rope_input_bf16 != 0 &&
        buffers->current_kv_latent_bf16 != 0 &&
        buffers->input_cache_bf16 != 0 &&
        buffers->reference_cache_bf16 != 0 &&
        buffers->rotated_query_rope_bf16 != 0 &&
        buffers->output_latent_bf16 != 0 &&
        buffers->device_cache_result_bf16 != 0 &&
        buffers->device_rotated_query_result_bf16 != 0 &&
        buffers->device_output_result_bf16 != 0 &&
        buffers->expected_output != 0 &&
        buffers->cos_table != 0 &&
        buffers->sin_table != 0 &&
        buffers->sparse_token_indices != 0;
}

static void SparkValidationFreeHostBuffers(
    SparkValidationHostBuffers *buffers)
{
    if (buffers == 0)
    {
        return;
    }
    free(buffers->sparse_token_indices);
    free(buffers->sin_table);
    free(buffers->cos_table);
    free(buffers->expected_output);
    free(buffers->device_output_result_bf16);
    free(buffers->device_rotated_query_result_bf16);
    free(buffers->device_cache_result_bf16);
    free(buffers->output_latent_bf16);
    free(buffers->rotated_query_rope_bf16);
    free(buffers->reference_cache_bf16);
    free(buffers->input_cache_bf16);
    free(buffers->current_kv_latent_bf16);
    free(buffers->key_rope_input_bf16);
    free(buffers->query_rope_input_bf16);
    free(buffers->query_latent_bf16);
    memset(buffers, 0, sizeof(*buffers));
}

static bool SparkValidationCudaAllocate(
    void **device_pointer,
    size_t allocation_bytes,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMalloc(device_pointer, allocation_bytes),
        name);
}

static bool SparkValidationAllocateDeviceBuffers(
    SparkValidationDeviceBuffers *buffers)
{
    const size_t query_latent_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION *
        sizeof(uint16_t);
    const size_t query_rope_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION *
        sizeof(uint16_t);
    const size_t cache_bytes =
        (size_t)SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS *
        sizeof(uint16_t);
    const size_t rope_table_bytes =
        (size_t)SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u) *
        sizeof(float);

    memset(buffers, 0, sizeof(*buffers));
    return SparkValidationCudaAllocate(
               (void **)&buffers->query_latent_bf16,
               query_latent_bytes,
               "cudaMalloc query latent") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->query_rope_input_bf16,
               query_rope_bytes,
               "cudaMalloc query RoPE input") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->key_rope_input_bf16,
               SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION *
                   sizeof(uint16_t),
               "cudaMalloc key RoPE input") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->current_kv_latent_bf16,
               SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION *
                   sizeof(uint16_t),
               "cudaMalloc current KV latent") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->mla_cache_bf16,
               cache_bytes,
               "cudaMalloc MLA cache") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->rotated_query_rope_bf16,
               query_rope_bytes,
               "cudaMalloc rotated query RoPE") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->output_latent_bf16,
               query_latent_bytes,
               "cudaMalloc output latent") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->cos_table,
               rope_table_bytes,
               "cudaMalloc cosine table") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->sin_table,
               rope_table_bytes,
               "cudaMalloc sine table") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->positions,
               sizeof(uint32_t),
               "cudaMalloc positions") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->slot_mapping,
               sizeof(uint32_t),
               "cudaMalloc slot mapping") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->block_table,
               SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE * sizeof(uint32_t),
               "cudaMalloc block table") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->context_lengths,
               sizeof(uint32_t),
               "cudaMalloc context lengths") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->first_block_token_offsets,
               sizeof(uint32_t),
               "cudaMalloc first block token offsets") &&
        SparkValidationCudaAllocate(
               (void **)&buffers->sparse_token_indices,
               SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT *
                   sizeof(uint32_t),
               "cudaMalloc sparse token indices");
}

static void SparkValidationFreeDeviceBuffers(
    SparkValidationDeviceBuffers *buffers)
{
    if (buffers == 0)
    {
        return;
    }
    cudaFree(buffers->sparse_token_indices);
    cudaFree(buffers->first_block_token_offsets);
    cudaFree(buffers->context_lengths);
    cudaFree(buffers->block_table);
    cudaFree(buffers->slot_mapping);
    cudaFree(buffers->positions);
    cudaFree(buffers->sin_table);
    cudaFree(buffers->cos_table);
    cudaFree(buffers->output_latent_bf16);
    cudaFree(buffers->rotated_query_rope_bf16);
    cudaFree(buffers->mla_cache_bf16);
    cudaFree(buffers->current_kv_latent_bf16);
    cudaFree(buffers->key_rope_input_bf16);
    cudaFree(buffers->query_rope_input_bf16);
    cudaFree(buffers->query_latent_bf16);
    memset(buffers, 0, sizeof(*buffers));
}

static void SparkValidationApplyRopePair(
    uint16_t first_input_bf16,
    uint16_t second_input_bf16,
    float cosine,
    float sine,
    uint16_t *first_output_bf16,
    uint16_t *second_output_bf16)
{
    float first_input;
    float second_input;

    first_input = SparkValidationBf16ToFloat(first_input_bf16);
    second_input = SparkValidationBf16ToFloat(second_input_bf16);
    *first_output_bf16 = SparkValidationFloatToBf16(
        (first_input * cosine) - (second_input * sine));
    *second_output_bf16 = SparkValidationFloatToBf16(
        (first_input * sine) + (second_input * cosine));
}

static void SparkValidationFillInputs(
    SparkValidationHostBuffers *buffers)
{
    const uint32_t rope_pair_count =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u;
    const size_t query_latent_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
    const size_t query_rope_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
    const size_t cache_count =
        (size_t)SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
    uint32_t position_index;
    uint32_t rope_pair_index;
    size_t element_index;
    uint32_t logical_token_index;

    for (position_index = 0u;
         position_index < SPARK_VALIDATION_POSITION_COUNT;
         ++position_index)
    {
        for (rope_pair_index = 0u;
             rope_pair_index < rope_pair_count;
             ++rope_pair_index)
        {
            float angle;
            size_t table_index;

            angle =
                (float)(position_index + 1u) *
                (float)(rope_pair_index + 1u) *
                0.0007f;
            table_index =
                ((size_t)position_index * rope_pair_count) +
                rope_pair_index;
            buffers->cos_table[table_index] = cosf(angle);
            buffers->sin_table[table_index] = sinf(angle);
        }
    }
    for (element_index = 0u;
         element_index < query_latent_count;
         ++element_index)
    {
        buffers->query_latent_bf16[element_index] =
            SparkValidationFloatToBf16(
                SparkValidationPattern(element_index, 13u));
    }
    for (element_index = 0u;
         element_index < query_rope_count;
         ++element_index)
    {
        buffers->query_rope_input_bf16[element_index] =
            SparkValidationFloatToBf16(
                SparkValidationPattern(element_index, 17u));
    }
    for (element_index = 0u;
         element_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
         ++element_index)
    {
        buffers->key_rope_input_bf16[element_index] =
            SparkValidationFloatToBf16(
                SparkValidationPattern(element_index, 19u));
    }
    for (element_index = 0u;
         element_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
         ++element_index)
    {
        buffers->current_kv_latent_bf16[element_index] =
            SparkValidationFloatToBf16(
                SparkValidationPattern(element_index, 23u));
    }

    memset(buffers->input_cache_bf16, 0, cache_count * sizeof(uint16_t));
    for (logical_token_index = 0u;
         logical_token_index < SPARK_VALIDATION_CONTEXT_TOKEN_COUNT - 1u;
         ++logical_token_index)
    {
        uint32_t cache_slot_index;
        uint32_t cache_dimension_index;

        cache_slot_index =
            SparkValidationLogicalTokenToCacheSlot(logical_token_index);
        for (cache_dimension_index = 0u;
             cache_dimension_index <
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
             ++cache_dimension_index)
        {
            uint64_t pattern_index;
            size_t cache_index;

            pattern_index =
                ((uint64_t)logical_token_index *
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                cache_dimension_index;
            cache_index =
                ((size_t)cache_slot_index *
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                cache_dimension_index;
            buffers->input_cache_bf16[cache_index] =
                SparkValidationFloatToBf16(
                    SparkValidationPattern(pattern_index, 29u));
        }
    }
    memcpy(
        buffers->reference_cache_bf16,
        buffers->input_cache_bf16,
        cache_count * sizeof(uint16_t));

    for (element_index = 0u;
         element_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT;
         ++element_index)
    {
        buffers->sparse_token_indices[element_index] =
            element_index < SPARK_VALIDATION_CONTEXT_TOKEN_COUNT
            ? (uint32_t)element_index
            : SPARK_GLM52_RESIDENT_SPARSE_MLA_INVALID_TOKEN_INDEX;
    }
}

static void SparkValidationBuildReference(
    SparkValidationHostBuffers *buffers)
{
    const uint32_t rope_pair_count =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u;
    const size_t table_base =
        (size_t)SPARK_VALIDATION_POSITION * rope_pair_count;
    uint32_t head_index;
    uint32_t rope_pair_index;
    uint32_t latent_dimension_index;
    uint32_t current_cache_slot;

    for (head_index = 0u;
         head_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT;
         ++head_index)
    {
        for (rope_pair_index = 0u;
             rope_pair_index < rope_pair_count;
             ++rope_pair_index)
        {
            size_t rope_offset;

            rope_offset =
                ((size_t)head_index *
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                ((size_t)rope_pair_index * 2u);
            SparkValidationApplyRopePair(
                buffers->query_rope_input_bf16[rope_offset],
                buffers->query_rope_input_bf16[rope_offset + 1u],
                buffers->cos_table[table_base + rope_pair_index],
                buffers->sin_table[table_base + rope_pair_index],
                &buffers->rotated_query_rope_bf16[rope_offset],
                &buffers->rotated_query_rope_bf16[rope_offset + 1u]);
        }
    }

    current_cache_slot = SPARK_VALIDATION_CURRENT_CACHE_SLOT;
    for (latent_dimension_index = 0u;
         latent_dimension_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
         ++latent_dimension_index)
    {
        buffers->reference_cache_bf16[
            ((size_t)current_cache_slot *
             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
            latent_dimension_index] =
            buffers->current_kv_latent_bf16[latent_dimension_index];
    }
    for (rope_pair_index = 0u;
         rope_pair_index < rope_pair_count;
         ++rope_pair_index)
    {
        size_t input_offset;
        size_t cache_offset;

        input_offset = (size_t)rope_pair_index * 2u;
        cache_offset =
            ((size_t)current_cache_slot *
             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
            SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION +
            input_offset;
        SparkValidationApplyRopePair(
            buffers->key_rope_input_bf16[input_offset],
            buffers->key_rope_input_bf16[input_offset + 1u],
            buffers->cos_table[table_base + rope_pair_index],
            buffers->sin_table[table_base + rope_pair_index],
            &buffers->reference_cache_bf16[cache_offset],
            &buffers->reference_cache_bf16[cache_offset + 1u]);
    }

    for (head_index = 0u;
         head_index < SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT;
         ++head_index)
    {
        float scores[SPARK_VALIDATION_CONTEXT_TOKEN_COUNT];
        float maximum_score;
        float exponential_sum;
        uint32_t logical_token_index;

        maximum_score = -INFINITY;
        for (logical_token_index = 0u;
             logical_token_index < SPARK_VALIDATION_CONTEXT_TOKEN_COUNT;
             ++logical_token_index)
        {
            uint32_t cache_slot_index;
            float dot_product;
            uint32_t dimension_index;

            cache_slot_index =
                SparkValidationLogicalTokenToCacheSlot(logical_token_index);
            dot_product = 0.0f;
            for (dimension_index = 0u;
                 dimension_index <
                     SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
                 ++dimension_index)
            {
                dot_product +=
                    SparkValidationBf16ToFloat(
                        buffers->query_latent_bf16[
                            ((size_t)head_index *
                             SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
                            dimension_index]) *
                    SparkValidationBf16ToFloat(
                        buffers->reference_cache_bf16[
                            ((size_t)cache_slot_index *
                             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                            dimension_index]);
            }
            for (dimension_index = 0u;
                 dimension_index <
                     SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
                 ++dimension_index)
            {
                dot_product +=
                    SparkValidationBf16ToFloat(
                        buffers->rotated_query_rope_bf16[
                            ((size_t)head_index *
                             SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION) +
                            dimension_index]) *
                    SparkValidationBf16ToFloat(
                        buffers->reference_cache_bf16[
                            ((size_t)cache_slot_index *
                             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                            SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION +
                            dimension_index]);
            }
            scores[logical_token_index] =
                dot_product * SPARK_VALIDATION_QK_SCALE;
            if (scores[logical_token_index] > maximum_score)
            {
                maximum_score = scores[logical_token_index];
            }
        }

        exponential_sum = 0.0f;
        for (logical_token_index = 0u;
             logical_token_index < SPARK_VALIDATION_CONTEXT_TOKEN_COUNT;
             ++logical_token_index)
        {
            scores[logical_token_index] =
                expf(scores[logical_token_index] - maximum_score);
            exponential_sum += scores[logical_token_index];
        }
        for (latent_dimension_index = 0u;
             latent_dimension_index <
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
             ++latent_dimension_index)
        {
            float output_value;

            output_value = 0.0f;
            for (logical_token_index = 0u;
                 logical_token_index < SPARK_VALIDATION_CONTEXT_TOKEN_COUNT;
                 ++logical_token_index)
            {
                uint32_t cache_slot_index;

                cache_slot_index =
                    SparkValidationLogicalTokenToCacheSlot(
                        logical_token_index);
                output_value +=
                    (scores[logical_token_index] / exponential_sum) *
                    SparkValidationBf16ToFloat(
                        buffers->reference_cache_bf16[
                            ((size_t)cache_slot_index *
                             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS) +
                            latent_dimension_index]);
            }
            buffers->expected_output[
                ((size_t)head_index *
                 SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION) +
                latent_dimension_index] = output_value;
        }
    }
}

static bool SparkValidationCopyInputsToDevice(
    const SparkValidationHostBuffers *host_buffers,
    const SparkValidationDeviceBuffers *device_buffers)
{
    static const uint32_t Position = SPARK_VALIDATION_POSITION;
    static const uint32_t SlotMapping =
        SPARK_VALIDATION_CURRENT_CACHE_SLOT;
    static const uint32_t BlockTable[
        SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE] = { 1u, 0u };
    static const uint32_t ContextLength =
        SPARK_VALIDATION_CONTEXT_TOKEN_COUNT;
    static const uint32_t FirstBlockTokenOffset =
        SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET;
    const size_t query_latent_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION *
        sizeof(uint16_t);
    const size_t query_rope_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION *
        sizeof(uint16_t);
    const size_t cache_bytes =
        (size_t)SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS *
        sizeof(uint16_t);
    const size_t rope_table_bytes =
        (size_t)SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION / 2u) *
        sizeof(float);

#define SPARK_COPY_TO_DEVICE(destination, source, bytes, name) \
    if (!SparkValidationCudaSucceeded( \
            cudaMemcpy(destination, source, bytes, cudaMemcpyHostToDevice), \
            name)) \
    { \
        return false; \
    }
    SPARK_COPY_TO_DEVICE(
        device_buffers->query_latent_bf16,
        host_buffers->query_latent_bf16,
        query_latent_bytes,
        "copy query latent")
    SPARK_COPY_TO_DEVICE(
        device_buffers->query_rope_input_bf16,
        host_buffers->query_rope_input_bf16,
        query_rope_bytes,
        "copy query RoPE input")
    SPARK_COPY_TO_DEVICE(
        device_buffers->key_rope_input_bf16,
        host_buffers->key_rope_input_bf16,
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION * sizeof(uint16_t),
        "copy key RoPE input")
    SPARK_COPY_TO_DEVICE(
        device_buffers->current_kv_latent_bf16,
        host_buffers->current_kv_latent_bf16,
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION * sizeof(uint16_t),
        "copy current KV latent")
    SPARK_COPY_TO_DEVICE(
        device_buffers->mla_cache_bf16,
        host_buffers->input_cache_bf16,
        cache_bytes,
        "copy MLA cache")
    SPARK_COPY_TO_DEVICE(
        device_buffers->cos_table,
        host_buffers->cos_table,
        rope_table_bytes,
        "copy cosine table")
    SPARK_COPY_TO_DEVICE(
        device_buffers->sin_table,
        host_buffers->sin_table,
        rope_table_bytes,
        "copy sine table")
    SPARK_COPY_TO_DEVICE(
        device_buffers->positions,
        &Position,
        sizeof(Position),
        "copy positions")
    SPARK_COPY_TO_DEVICE(
        device_buffers->slot_mapping,
        &SlotMapping,
        sizeof(SlotMapping),
        "copy slot mapping")
    SPARK_COPY_TO_DEVICE(
        device_buffers->block_table,
        BlockTable,
        sizeof(BlockTable),
        "copy block table")
    SPARK_COPY_TO_DEVICE(
        device_buffers->context_lengths,
        &ContextLength,
        sizeof(ContextLength),
        "copy context length")
    SPARK_COPY_TO_DEVICE(
        device_buffers->first_block_token_offsets,
        &FirstBlockTokenOffset,
        sizeof(FirstBlockTokenOffset),
        "copy first block token offset")
    SPARK_COPY_TO_DEVICE(
        device_buffers->sparse_token_indices,
        host_buffers->sparse_token_indices,
        SPARK_GLM52_RESIDENT_SPARSE_MLA_SELECTED_TOKEN_COUNT *
            sizeof(uint32_t),
        "copy sparse token indices")
#undef SPARK_COPY_TO_DEVICE
    return true;
}

static bool SparkValidationCopyResultsToHost(
    SparkValidationHostBuffers *host_buffers,
    const SparkValidationDeviceBuffers *device_buffers)
{
    const size_t query_latent_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION *
        sizeof(uint16_t);
    const size_t query_rope_bytes =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION *
        sizeof(uint16_t);
    const size_t cache_bytes =
        (size_t)SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS *
        sizeof(uint16_t);

    return SparkValidationCudaSucceeded(
               cudaMemcpy(
                   host_buffers->device_rotated_query_result_bf16,
                   device_buffers->rotated_query_rope_bf16,
                   query_rope_bytes,
                   cudaMemcpyDeviceToHost),
               "copy rotated query result") &&
        SparkValidationCudaSucceeded(
               cudaMemcpy(
                   host_buffers->device_cache_result_bf16,
                   device_buffers->mla_cache_bf16,
                   cache_bytes,
                   cudaMemcpyDeviceToHost),
               "copy cache result") &&
        SparkValidationCudaSucceeded(
               cudaMemcpy(
                   host_buffers->device_output_result_bf16,
                   device_buffers->output_latent_bf16,
                   query_latent_bytes,
                   cudaMemcpyDeviceToHost),
               "copy output result");
}

static bool SparkValidationCompareResults(
    const SparkValidationHostBuffers *buffers)
{
    const size_t query_latent_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_LATENT_DIMENSION;
    const size_t query_rope_count =
        (size_t)SPARK_GLM52_RESIDENT_SPARSE_MLA_HEAD_COUNT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_ROPE_DIMENSION;
    const size_t current_cache_offset =
        (size_t)SPARK_VALIDATION_CURRENT_CACHE_SLOT *
        SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
    size_t element_index;

    for (element_index = 0u;
         element_index < query_rope_count;
         ++element_index)
    {
        if (buffers->device_rotated_query_result_bf16[element_index] !=
            buffers->rotated_query_rope_bf16[element_index])
        {
            fprintf(stderr, "rotated query mismatch at %zu\n", element_index);
            return false;
        }
    }
    for (element_index = 0u;
         element_index <
             SPARK_GLM52_RESIDENT_SPARSE_MLA_CACHE_TOKEN_ELEMENTS;
         ++element_index)
    {
        if (buffers->device_cache_result_bf16[
                current_cache_offset + element_index] !=
            buffers->reference_cache_bf16[
                current_cache_offset + element_index])
        {
            fprintf(stderr, "cache write mismatch at %zu\n", element_index);
            return false;
        }
    }
    for (element_index = 0u;
         element_index < query_latent_count;
         ++element_index)
    {
        float actual_value;
        float absolute_error;

        actual_value = SparkValidationBf16ToFloat(
            buffers->device_output_result_bf16[element_index]);
        absolute_error = fabsf(
            actual_value - buffers->expected_output[element_index]);
        if (!isfinite(actual_value) ||
            absolute_error > SPARK_VALIDATION_OUTPUT_TOLERANCE)
        {
            fprintf(
                stderr,
                "attention output mismatch at %zu actual=%f expected=%f error=%f\n",
                element_index,
                actual_value,
                buffers->expected_output[element_index],
                absolute_error);
            return false;
        }
    }
    return true;
}

static bool SparkValidationSubmitAndWait(
    void *module_state,
    SparkValidationCompletionState *completion_state,
    uint64_t request_id,
    uint32_t expected_completion_count)
{
    SparkModelDriverFrame frame;
    SparkStatus status;

    memset(&frame, 0, sizeof(frame));
    frame.request_id = request_id;
    frame.active_slot_count = SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    frame.program_id = 1u;
    frame.scalar[
        SPARK_GLM52_RESIDENT_SPARSE_MLA_PIPELINE_SLOT_SCALAR_INDEX] = 0u;
    status = SparkGlm52ResidentSparseMlaExecute(module_state, &frame);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(
            stderr,
            "firmware execute failed: %s\n",
            SparkStatusToString(status));
        return false;
    }
    if (!SparkValidationWaitForCompletion(
            completion_state,
            expected_completion_count))
    {
        fprintf(stderr, "firmware completion timed out\n");
        return false;
    }
    if (completion_state->last_completion.request_id != request_id ||
        completion_state->last_completion.program_id != 1u ||
        completion_state->last_completion.status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "firmware completion record is invalid\n");
        return false;
    }
    return true;
}

static bool SparkValidationMeasureStage(
    void *module_state,
    SparkValidationCompletionState *completion_state,
    double maximum_stage_microseconds,
    double *average_microseconds,
    double *maximum_microseconds)
{
    uint32_t completion_count;
    uint32_t iteration_index;
    double total_microseconds;

    *average_microseconds = 0.0;
    *maximum_microseconds = 0.0;
    completion_count = completion_state->completion_count.load(
        std::memory_order_acquire);
    for (iteration_index = 0u;
         iteration_index < SPARK_VALIDATION_WARMUP_COUNT;
         ++iteration_index)
    {
        completion_count += 1u;
        if (!SparkValidationSubmitAndWait(
                module_state,
                completion_state,
                1000u + iteration_index,
                completion_count))
        {
            return false;
        }
    }

    total_microseconds = 0.0;
    for (iteration_index = 0u;
         iteration_index < SPARK_VALIDATION_MEASUREMENT_COUNT;
         ++iteration_index)
    {
        double start_seconds;
        double end_seconds;
        double elapsed_microseconds;

        completion_count += 1u;
        start_seconds = SparkValidationMonotonicSeconds();
        if (!SparkValidationSubmitAndWait(
                module_state,
                completion_state,
                2000u + iteration_index,
                completion_count))
        {
            return false;
        }
        end_seconds = SparkValidationMonotonicSeconds();
        elapsed_microseconds = (end_seconds - start_seconds) * 1.0e6;
        total_microseconds += elapsed_microseconds;
        if (elapsed_microseconds > *maximum_microseconds)
        {
            *maximum_microseconds = elapsed_microseconds;
        }
    }
    *average_microseconds =
        total_microseconds / (double)SPARK_VALIDATION_MEASUREMENT_COUNT;
    return *maximum_microseconds <= maximum_stage_microseconds;
}

int main(int argument_count, char **arguments)
{
    SparkValidationHostBuffers host_buffers;
    SparkValidationDeviceBuffers device_buffers;
    SparkValidationCompletionState completion_state;
    SparkGlm52ResidentSparseMlaPipelineSlot pipeline_slot;
    SparkGlm52ResidentSparseMlaNodeContext node_context;
    SparkFirmwareModuleConfiguration module_configuration;
    SparkFirmwareModuleHostServices host_services;
    void *module_state;
    cudaStream_t cuda_stream;
    char *threshold_end;
    double maximum_stage_microseconds;
    double average_microseconds;
    double maximum_microseconds;
    bool passed;

    if (argument_count != 2)
    {
        fprintf(stderr, "usage: %s MAX_STAGE_MICROSECONDS\n", arguments[0]);
        return 2;
    }
    threshold_end = 0;
    maximum_stage_microseconds = strtod(arguments[1], &threshold_end);
    if (threshold_end == arguments[1] || *threshold_end != '\0' ||
        !isfinite(maximum_stage_microseconds) ||
        maximum_stage_microseconds <= 0.0)
    {
        fprintf(stderr, "invalid stage latency limit\n");
        return 2;
    }

    memset(&host_buffers, 0, sizeof(host_buffers));
    memset(&device_buffers, 0, sizeof(device_buffers));
    module_state = 0;
    cuda_stream = 0;
    passed = false;
    average_microseconds = 0.0;
    maximum_microseconds = 0.0;
    completion_state.completion_count.store(0u, std::memory_order_relaxed);
    memset(&completion_state.last_completion, 0, sizeof(completion_state.last_completion));

    if (!SparkValidationAllocateHostBuffers(&host_buffers) ||
        !SparkValidationAllocateDeviceBuffers(&device_buffers) ||
        !SparkValidationCudaSucceeded(
            cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking),
            "cudaStreamCreateWithFlags"))
    {
        goto cleanup;
    }
    SparkValidationFillInputs(&host_buffers);
    SparkValidationBuildReference(&host_buffers);
    if (!SparkValidationCopyInputsToDevice(&host_buffers, &device_buffers))
    {
        goto cleanup;
    }

    memset(&pipeline_slot, 0, sizeof(pipeline_slot));
    pipeline_slot.cuda_stream = (void *)cuda_stream;
    pipeline_slot.query_latent_bf16 = device_buffers.query_latent_bf16;
    pipeline_slot.query_rope_input_bf16 =
        device_buffers.query_rope_input_bf16;
    pipeline_slot.key_rope_input_bf16 = device_buffers.key_rope_input_bf16;
    pipeline_slot.current_kv_latent_bf16 =
        device_buffers.current_kv_latent_bf16;
    pipeline_slot.positions = device_buffers.positions;
    pipeline_slot.slot_mapping = device_buffers.slot_mapping;
    pipeline_slot.block_table = device_buffers.block_table;
    pipeline_slot.context_lengths = device_buffers.context_lengths;
    pipeline_slot.first_block_token_offsets =
        device_buffers.first_block_token_offsets;
    pipeline_slot.sparse_token_indices = device_buffers.sparse_token_indices;
    pipeline_slot.rotated_query_rope_bf16 =
        device_buffers.rotated_query_rope_bf16;
    pipeline_slot.output_latent_bf16 = device_buffers.output_latent_bf16;

    memset(&node_context, 0, sizeof(node_context));
    node_context.abi_version =
        SPARK_GLM52_RESIDENT_SPARSE_MLA_NODE_CONTEXT_ABI_VERSION;
    node_context.pipeline_slot_count = 1u;
    node_context.max_active_sequence_count =
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    node_context.cache_token_capacity =
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY;
    node_context.kv_block_count = SPARK_VALIDATION_KV_BLOCK_COUNT;
    node_context.max_blocks_per_sequence =
        SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE;
    node_context.position_count = SPARK_VALIDATION_POSITION_COUNT;
    node_context.qk_scale = SPARK_VALIDATION_QK_SCALE;
    node_context.cos_table = device_buffers.cos_table;
    node_context.sin_table = device_buffers.sin_table;
    node_context.mla_cache_bf16 = device_buffers.mla_cache_bf16;
    node_context.pipeline_slots = &pipeline_slot;

    memset(&module_configuration, 0, sizeof(module_configuration));
    module_configuration.abi_version = SPARK_FIRMWARE_MODULE_ABI_VERSION;
    module_configuration.model_id = "validation.glm52";
    module_configuration.model_revision = "validation.v1";
    module_configuration.stage_name = "resident_attention";
    module_configuration.program_name = "decode_attention";
    module_configuration.operation_name = "resident_sparse_mla";
    module_configuration.configuration_json = "{}";
    module_configuration.configuration_json_bytes = 2u;

    memset(&host_services, 0, sizeof(host_services));
    host_services.completion_function = SparkValidationCompletion;
    host_services.completion_context = &completion_state;
    host_services.node_id = "validation-node";
    host_services.node_target = SPARK_GLM52_RESIDENT_SPARSE_MLA_TARGET;
    host_services.node_context = &node_context;

    if (SparkGlm52ResidentSparseMlaInitialize(
            &module_configuration,
            &host_services,
            &module_state) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "firmware initialization failed\n");
        goto cleanup;
    }
    if (!SparkValidationSubmitAndWait(
            module_state,
            &completion_state,
            1u,
            1u) ||
        !SparkValidationCopyResultsToHost(
            &host_buffers,
            &device_buffers) ||
        !SparkValidationCompareResults(&host_buffers))
    {
        goto cleanup;
    }
    if (!SparkValidationMeasureStage(
            module_state,
            &completion_state,
            maximum_stage_microseconds,
            &average_microseconds,
            &maximum_microseconds))
    {
        fprintf(
            stderr,
            "stage latency failed average_us=%.3f maximum_us=%.3f limit_us=%.3f\n",
            average_microseconds,
            maximum_microseconds,
            maximum_stage_microseconds);
        goto cleanup;
    }

    printf(
        "glm52_resident_sparse_mla validation passed average_us=%.3f maximum_us=%.3f limit_us=%.3f\n",
        average_microseconds,
        maximum_microseconds,
        maximum_stage_microseconds);
    passed = true;

cleanup:
    SparkGlm52ResidentSparseMlaDestroy(module_state);
    if (cuda_stream != 0)
    {
        cudaStreamDestroy(cuda_stream);
    }
    SparkValidationFreeDeviceBuffers(&device_buffers);
    SparkValidationFreeHostBuffers(&host_buffers);
    return passed ? 0 : 1;
}
