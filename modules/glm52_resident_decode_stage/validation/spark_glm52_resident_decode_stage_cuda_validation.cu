#define _POSIX_C_SOURCE 200809L

#include <cuda_runtime.h>

#include <atomic>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "spark_glm52_resident_decode_stage_backend.h"
#include "sparkpipe/spark_orchestrator.h"
#include "sparkpipe/spark_status.h"

#define SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT 1u
#define SPARK_VALIDATION_CACHE_TOKEN_CAPACITY 128u
#define SPARK_VALIDATION_KV_BLOCK_COUNT 2u
#define SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE 2u
#define SPARK_VALIDATION_POSITION_COUNT 128u
#define SPARK_VALIDATION_CONTEXT_LENGTH 4u
#define SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET 61u
#define SPARK_VALIDATION_CURRENT_POSITION 64u
#define SPARK_VALIDATION_CURRENT_CACHE_SLOT 0u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT0 125u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT1 126u
#define SPARK_VALIDATION_REMAP_CACHE_SLOT2 127u
#define SPARK_VALIDATION_CHECKED_HEAD_COUNT 4u
#define SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT 8u
#define SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT 4u
#define SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN 1009u
#define SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN 1011u
#define SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN 1003u
#define SPARK_VALIDATION_MEASUREMENT_COUNT 3u
#define SPARK_VALIDATION_WARMUP_COUNT 1u
#define SPARK_VALIDATION_ATTENTION_TOLERANCE 0.030f

typedef struct SparkValidationCompletionState
{
    std::atomic<uint32_t> completion_count;
} SparkValidationCompletionState;

typedef struct SparkValidationDriverCompletionState
{
    std::atomic<uint32_t> completion_count;
    SparkModelDriverCompletion completion;
} SparkValidationDriverCompletionState;

typedef struct SparkValidationDeviceBuffers
{
    uint16_t *input_hidden_bf16;
    uint16_t *normalized_hidden_bf16;
    uint16_t *query_latent_bf16;
    uint16_t *query_rope_input_bf16;
    uint16_t *key_rope_input_bf16;
    uint16_t *current_kv_latent_bf16;
    uint16_t *raw_query_a_bf16;
    uint16_t *raw_query_a_normalized_bf16;
    uint16_t *raw_query_b_bf16;
    uint16_t *raw_kv_a_bf16;
    uint16_t *raw_kv_a_normalized_bf16;
    uint16_t *raw_kv_b_bf16;
    uint16_t *mla_cache_bf16;
    uint16_t *rotated_query_rope_bf16;
    uint16_t *attention_output_latent_bf16;
    uint16_t *attention_projected_hidden_bf16;
    uint16_t *post_attention_hidden_bf16;
    uint16_t *mtp_draft_hidden_bf16;
    uint16_t *attention_norm_weight_bf16;
    uint16_t *query_latent_weight_bf16;
    uint16_t *query_rope_weight_bf16;
    uint16_t *key_rope_weight_bf16;
    uint16_t *kv_latent_weight_bf16;
    uint16_t *raw_query_a_weight_bf16;
    uint16_t *raw_query_a_norm_weight_bf16;
    uint16_t *raw_query_b_weight_bf16;
    uint16_t *raw_kv_a_weight_bf16;
    uint16_t *raw_kv_a_norm_weight_bf16;
    uint16_t *raw_kv_b_weight_bf16;
    uint8_t *raw_query_a_weight_fp8_e4m3;
    float *raw_query_a_weight_scale_inv_f32;
    uint8_t *raw_query_b_weight_fp8_e4m3;
    float *raw_query_b_weight_scale_inv_f32;
    uint8_t *raw_kv_a_weight_fp8_e4m3;
    float *raw_kv_a_weight_scale_inv_f32;
    uint8_t *raw_kv_b_weight_fp8_e4m3;
    float *raw_kv_b_weight_scale_inv_f32;
    uint16_t *attention_output_weight_bf16;
    uint16_t *final_norm_weight_bf16;
    uint16_t *restricted_lm_head_weight_bf16;
    uint8_t *mtp_mxfp4_weight_payload_u8;
    uint8_t *mtp_mxfp4_scale_e8m0_u8;
    float *cos_table;
    float *sin_table;
    float *dsa_token_scores;
    float *restricted_logits;
    float *restricted_selected_token_scores;
    float *mtp_draft_logits;
    uint32_t *positions;
    uint32_t *slot_mapping;
    uint32_t *block_table;
    uint32_t *context_lengths;
    uint32_t *first_block_token_offsets;
    uint32_t *sparse_token_indices;
    uint32_t *restricted_token_ids;
    uint32_t *restricted_selected_token_ids;
    uint32_t *mtp_draft_token_ids;
    uint32_t *mtp_target_token_ids;
    uint32_t *mtp_accept_mask;
    uint32_t *mtp_committed_token_ids;
    uint32_t *mtp_event_counters;
    uint64_t *phase_clock_cycles;
} SparkValidationDeviceBuffers;

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

static bool SparkValidationAllocateZeroed(
    void **device_pointer,
    uint64_t byte_count,
    const char *name)
{
    if (!SparkValidationCudaSucceeded(
            cudaMalloc(device_pointer, (size_t)byte_count),
            name))
    {
        return false;
    }
    return SparkValidationCudaSucceeded(
        cudaMemset(*device_pointer, 0, (size_t)byte_count),
        name);
}

static bool SparkValidationCopyToDevice(
    void *device_pointer,
    const void *host_pointer,
    uint64_t byte_count,
    const char *name)
{
    return SparkValidationCudaSucceeded(
        cudaMemcpy(
            device_pointer,
            host_pointer,
            (size_t)byte_count,
            cudaMemcpyHostToDevice),
        name);
}

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

static float SparkValidationFp8E4m3ToFloat(uint8_t value)
{
    uint32_t sign;
    uint32_t exponent;
    uint32_t mantissa;
    float decoded_value;

    sign = (uint32_t)(value >> 7u);
    exponent = (uint32_t)((value >> 3u) & 15u);
    mantissa = (uint32_t)(value & 7u);
    if ((value & 0x7fu) == 0u)
    {
        return 0.0f;
    }
    if (exponent == 0u)
    {
        decoded_value = ldexpf((float)mantissa / 8.0f, -6);
    }
    else
    {
        decoded_value =
            ldexpf(1.0f + ((float)mantissa / 8.0f), (int32_t)exponent - 7);
    }
    return sign != 0u ? -decoded_value : decoded_value;
}

static uint8_t SparkValidationFloatToFp8E4m3(float value)
{
    float best_error;
    uint8_t best_byte;
    uint32_t byte_value;

    best_error = fabsf(value);
    best_byte = 0u;
    for (byte_value = 1u; byte_value < 255u; ++byte_value)
    {
        float decoded_value;
        float error;

        decoded_value = SparkValidationFp8E4m3ToFloat((uint8_t)byte_value);
        error = fabsf(decoded_value - value);
        if (error < best_error)
        {
            best_error = error;
            best_byte = (uint8_t)byte_value;
        }
    }
    return best_byte;
}

static float SparkValidationSeedLatentValue(
    uint32_t token_index,
    uint32_t dimension_index)
{
    return (0.03125f * (float)(token_index + 1u)) +
        (0.0078125f * (float)(dimension_index + 1u));
}

static float SparkValidationSeedRopeValue(
    uint32_t token_index,
    uint32_t dimension_index)
{
    return (0.015625f * (float)(token_index + 1u)) +
        (0.00390625f * (float)(dimension_index + 1u));
}

static void SparkValidationSeedCacheSlot(
    uint16_t *cache_seed,
    uint32_t cache_slot_index,
    uint32_t token_index)
{
    uint64_t cache_offset;
    uint32_t dimension_index;

    cache_offset =
        (uint64_t)cache_slot_index *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_seed[cache_offset + (uint64_t)dimension_index] =
            SparkValidationFloatToBf16(
                SparkValidationSeedLatentValue(token_index, dimension_index));
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_seed[
            cache_offset +
            (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
            (uint64_t)dimension_index] =
            SparkValidationFloatToBf16(
                SparkValidationSeedRopeValue(token_index, dimension_index));
    }
}

static bool SparkValidationCopyU8Value(
    uint8_t *device_pointer,
    uint64_t index,
    uint8_t value,
    const char *name)
{
    return SparkValidationCopyToDevice(
        &device_pointer[index],
        &value,
        sizeof(value),
        name);
}

static bool SparkValidationCopyU16Value(
    uint16_t *device_pointer,
    uint64_t index,
    uint16_t value,
    const char *name)
{
    return SparkValidationCopyToDevice(
        &device_pointer[index],
        &value,
        sizeof(value),
        name);
}

static bool SparkValidationCopyFp8FixtureValue(
    uint8_t *device_pointer,
    uint64_t index,
    float value,
    const char *name)
{
    return SparkValidationCopyU8Value(
        device_pointer,
        index,
        SparkValidationFloatToFp8E4m3(value),
        name);
}

static bool SparkValidationSeedProjectionFixtureWeights(
    SparkValidationDeviceBuffers *buffers)
{
    uint32_t head_index;
    uint32_t dimension_index;

    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0030f * (float)(head_index + 1u)) +
                (0.0005f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->query_latent_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy query latent fixture"))
            {
                return false;
            }
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0020f * (float)(head_index + 1u)) +
                (0.00025f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->query_rope_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy query rope fixture"))
            {
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->kv_latent_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0040f + (0.0005f * (float)dimension_index)),
                "copy kv latent fixture"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->key_rope_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0050f + (0.00075f * (float)dimension_index)),
                "copy key rope fixture"))
        {
            return false;
        }
    }
    if (!SparkValidationCopyU16Value(
            buffers->raw_query_a_weight_bf16,
            0u,
            SparkValidationFloatToBf16(0.0010f),
            "copy raw q_a fixture") ||
        !SparkValidationCopyU16Value(
            buffers->raw_kv_b_weight_bf16,
            0u,
            SparkValidationFloatToBf16(0.0010f),
            "copy raw kv_b fixture") ||
        !SparkValidationCopyFp8FixtureValue(
            buffers->raw_query_a_weight_fp8_e4m3,
            0u,
            0.0010f,
            "copy raw fp8 q_a fixture") ||
        !SparkValidationCopyFp8FixtureValue(
            buffers->raw_kv_b_weight_fp8_e4m3,
            0u,
            0.0010f,
            "copy raw fp8 kv_b fixture"))
    {
        return false;
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)dimension_index;
            weight_value =
                (0.0030f * (float)(head_index + 1u)) +
                (0.0005f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->raw_query_b_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy raw q_b latent fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_query_b_weight_fp8_e4m3,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    weight_value,
                    "copy raw fp8 q_b latent fixture"))
            {
                return false;
            }
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            uint64_t row_index;
            float weight_value;

            row_index =
                ((uint64_t)head_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_HEAD_DIMENSION) +
                (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QK_NOPE_HEAD_DIMENSION +
                (uint64_t)dimension_index;
            weight_value =
                (0.0020f * (float)(head_index + 1u)) +
                (0.00025f * (float)(dimension_index + 1u));
            if (!SparkValidationCopyU16Value(
                    buffers->raw_query_b_weight_bf16,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    SparkValidationFloatToBf16(weight_value),
                    "copy raw q_b rope fixture") ||
                !SparkValidationCopyFp8FixtureValue(
                    buffers->raw_query_b_weight_fp8_e4m3,
                    row_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION,
                    weight_value,
                    "copy raw fp8 q_b rope fixture"))
            {
                return false;
            }
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->raw_kv_a_weight_bf16,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0040f + (0.0005f * (float)dimension_index)),
                "copy raw kv_a latent fixture") ||
            !SparkValidationCopyFp8FixtureValue(
                buffers->raw_kv_a_weight_fp8_e4m3,
                (uint64_t)dimension_index *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0.0040f + (0.0005f * (float)dimension_index),
                "copy raw fp8 kv_a latent fixture"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (!SparkValidationCopyU16Value(
                buffers->raw_kv_a_weight_bf16,
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                 (uint64_t)dimension_index) *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                SparkValidationFloatToBf16(
                    0.0050f + (0.00075f * (float)dimension_index)),
                "copy raw kv_a rope fixture") ||
            !SparkValidationCopyFp8FixtureValue(
                buffers->raw_kv_a_weight_fp8_e4m3,
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
                 (uint64_t)dimension_index) *
                    (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION,
                0.0050f + (0.00075f * (float)dimension_index),
                "copy raw fp8 kv_a rope fixture"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        uint32_t hidden_output_index;
        uint64_t attention_input_index;

        hidden_output_index = 3u + dimension_index;
        attention_input_index = dimension_index;
        if (!SparkValidationCopyU16Value(
                buffers->attention_output_weight_bf16,
                ((uint64_t)hidden_output_index *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION) +
                    attention_input_index,
                SparkValidationFloatToBf16(
                    0.0200f + (0.0010f * (float)dimension_index)),
                "copy attention output fixture") ||
            !SparkValidationCopyU16Value(
                buffers->restricted_lm_head_weight_bf16,
                ((uint64_t)4u *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_output_index,
                SparkValidationFloatToBf16(0.0100f),
                "copy restricted logit fixture 4") ||
            !SparkValidationCopyU16Value(
                buffers->restricted_lm_head_weight_bf16,
                ((uint64_t)9u *
                 (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION) +
                    (uint64_t)hidden_output_index,
                SparkValidationFloatToBf16(0.0350f),
                "copy restricted logit fixture 9"))
        {
            return false;
        }
    }
    return
        SparkValidationCopyU8Value(
            buffers->mtp_mxfp4_weight_payload_u8,
            11u *
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION / 2u),
            7u,
            "copy mtp payload fixture") &&
        SparkValidationCopyU8Value(
            buffers->mtp_mxfp4_scale_e8m0_u8,
            11u *
                ((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION /
                 SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE),
            127u,
            "copy mtp scale fixture");
}

static void SparkValidationCompletion(void *completion_context)
{
    SparkValidationCompletionState *state;

    state = (SparkValidationCompletionState *)completion_context;
    if (state != 0)
    {
        state->completion_count.fetch_add(1u, std::memory_order_release);
    }
}

static void SparkValidationDriverCompletion(
    void *completion_context,
    const SparkModelDriverCompletion *completion)
{
    SparkValidationDriverCompletionState *state;

    state = (SparkValidationDriverCompletionState *)completion_context;
    if (state != 0 && completion != 0)
    {
        state->completion = *completion;
        state->completion_count.fetch_add(1u, std::memory_order_release);
    }
}

static bool SparkValidationAllocateDeviceBuffers(
    SparkValidationDeviceBuffers *buffers)
{
    uint64_t hidden_count;
    uint64_t query_latent_count;
    uint64_t query_rope_count;
    uint64_t raw_query_a_count;
    uint64_t raw_query_b_count;
    uint64_t raw_kv_a_count;
    uint64_t raw_kv_b_count;
    uint64_t cache_count;
    uint64_t big_projection_weight_count;
    uint64_t raw_query_a_weight_count;
    uint64_t raw_query_b_weight_count;
    uint64_t raw_kv_a_weight_count;
    uint64_t raw_kv_b_weight_count;
    uint64_t raw_query_a_scale_count;
    uint64_t raw_query_b_scale_count;
    uint64_t raw_kv_a_scale_count;
    uint64_t raw_kv_b_scale_count;
    uint64_t restricted_weight_count;
    uint64_t mtp_payload_count;
    uint64_t mtp_scale_count;
    uint64_t rope_table_count;

    memset(buffers, 0, sizeof(*buffers));
    hidden_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    query_latent_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION;
    query_rope_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_ROPE_PROJECTION_DIMENSION;
    raw_query_a_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION;
    raw_query_b_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION;
    raw_kv_a_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION;
    raw_kv_b_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION;
    cache_count =
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS;
    big_projection_weight_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_ATTENTION_PROJECTION_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    raw_query_a_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    raw_query_b_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION;
    raw_kv_a_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    raw_kv_b_weight_count =
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION;
    raw_query_a_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_query_b_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_kv_a_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    raw_kv_b_scale_count =
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_KV_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        (((uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK);
    restricted_weight_count =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT *
        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
    mtp_payload_count = restricted_weight_count / 2u;
    mtp_scale_count =
        restricted_weight_count /
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MXFP4_GROUP_SIZE;
    rope_table_count =
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u);
    return
        SparkValidationAllocateZeroed((void **)&buffers->input_hidden_bf16, hidden_count * 2u, "cudaMalloc input_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->normalized_hidden_bf16, hidden_count * 2u, "cudaMalloc normalized_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_latent_bf16, query_latent_count * 2u, "cudaMalloc query_latent") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_rope_input_bf16, query_rope_count * 2u, "cudaMalloc query_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->key_rope_input_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION * 2u, "cudaMalloc key_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->current_kv_latent_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc current_kv") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_bf16, raw_query_a_count * 2u, "cudaMalloc raw_query_a") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_normalized_bf16, raw_query_a_count * 2u, "cudaMalloc raw_query_a_normalized") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_bf16, raw_query_b_count * 2u, "cudaMalloc raw_query_b") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_bf16, raw_kv_a_count * 2u, "cudaMalloc raw_kv_a") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_normalized_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc raw_kv_a_normalized") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_bf16, raw_kv_b_count * 2u, "cudaMalloc raw_kv_b") &&
        SparkValidationAllocateZeroed((void **)&buffers->mla_cache_bf16, cache_count * 2u, "cudaMalloc mla_cache") &&
        SparkValidationAllocateZeroed((void **)&buffers->rotated_query_rope_bf16, query_rope_count * 2u, "cudaMalloc rotated_query_rope") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_latent_bf16, query_latent_count * 2u, "cudaMalloc attention_output_latent") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_projected_hidden_bf16, hidden_count * 2u, "cudaMalloc attention_projected_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->post_attention_hidden_bf16, hidden_count * 2u, "cudaMalloc post_attention_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_hidden_bf16, hidden_count * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 2u, "cudaMalloc mtp_draft_hidden") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_norm_weight_bf16, hidden_count * 2u, "cudaMalloc attention_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_latent_weight_bf16, big_projection_weight_count * 2u, "cudaMalloc query_latent_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->query_rope_weight_bf16, query_rope_count * hidden_count * 2u, "cudaMalloc query_rope_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->key_rope_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION * hidden_count * 2u, "cudaMalloc key_rope_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->kv_latent_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * hidden_count * 2u, "cudaMalloc kv_latent_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_bf16, raw_query_a_weight_count * 2u, "cudaMalloc raw_query_a_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_norm_weight_bf16, raw_query_a_count * 2u, "cudaMalloc raw_query_a_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_bf16, raw_query_b_weight_count * 2u, "cudaMalloc raw_query_b_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_bf16, raw_kv_a_weight_count * 2u, "cudaMalloc raw_kv_a_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_norm_weight_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "cudaMalloc raw_kv_a_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_bf16, raw_kv_b_weight_count * 2u, "cudaMalloc raw_kv_b_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_fp8_e4m3, raw_query_a_weight_count, "cudaMalloc raw_query_a_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_a_weight_scale_inv_f32, raw_query_a_scale_count * 4u, "cudaMalloc raw_query_a_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_fp8_e4m3, raw_query_b_weight_count, "cudaMalloc raw_query_b_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_query_b_weight_scale_inv_f32, raw_query_b_scale_count * 4u, "cudaMalloc raw_query_b_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_fp8_e4m3, raw_kv_a_weight_count, "cudaMalloc raw_kv_a_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_a_weight_scale_inv_f32, raw_kv_a_scale_count * 4u, "cudaMalloc raw_kv_a_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_fp8_e4m3, raw_kv_b_weight_count, "cudaMalloc raw_kv_b_weight_fp8") &&
        SparkValidationAllocateZeroed((void **)&buffers->raw_kv_b_weight_scale_inv_f32, raw_kv_b_scale_count * 4u, "cudaMalloc raw_kv_b_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->attention_output_weight_bf16, big_projection_weight_count * 2u, "cudaMalloc attention_output_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->final_norm_weight_bf16, hidden_count * 2u, "cudaMalloc final_norm_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_lm_head_weight_bf16, restricted_weight_count * 2u, "cudaMalloc restricted_lm_head_weight") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_mxfp4_weight_payload_u8, mtp_payload_count, "cudaMalloc mtp_payload") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_mxfp4_scale_e8m0_u8, mtp_scale_count, "cudaMalloc mtp_scale") &&
        SparkValidationAllocateZeroed((void **)&buffers->cos_table, rope_table_count * 4u, "cudaMalloc cos_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->sin_table, rope_table_count * 4u, "cudaMalloc sin_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->dsa_token_scores, 64u * 4u, "cudaMalloc dsa_scores") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_logits, SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * 4u, "cudaMalloc restricted_logits") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_selected_token_scores, 4u, "cudaMalloc selected_scores") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_logits, SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 4u, "cudaMalloc mtp_logits") &&
        SparkValidationAllocateZeroed((void **)&buffers->positions, 4u, "cudaMalloc positions") &&
        SparkValidationAllocateZeroed((void **)&buffers->slot_mapping, 4u, "cudaMalloc slot_mapping") &&
        SparkValidationAllocateZeroed((void **)&buffers->block_table, SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE * 4u, "cudaMalloc block_table") &&
        SparkValidationAllocateZeroed((void **)&buffers->context_lengths, 4u, "cudaMalloc context_lengths") &&
        SparkValidationAllocateZeroed((void **)&buffers->first_block_token_offsets, 4u, "cudaMalloc first_block_token_offsets") &&
        SparkValidationAllocateZeroed((void **)&buffers->sparse_token_indices, SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT * 4u, "cudaMalloc sparse_indices") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_token_ids, SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT * 4u, "cudaMalloc restricted_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->restricted_selected_token_ids, 4u, "cudaMalloc selected_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_draft_token_ids, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 4u, "cudaMalloc mtp_draft_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_target_token_ids, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 4u, "cudaMalloc mtp_target_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_accept_mask, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 4u, "cudaMalloc mtp_accept_mask") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_committed_token_ids, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT * 4u, "cudaMalloc mtp_committed_token_ids") &&
        SparkValidationAllocateZeroed((void **)&buffers->mtp_event_counters, SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT * 4u, "cudaMalloc mtp_event_counters") &&
        SparkValidationAllocateZeroed((void **)&buffers->phase_clock_cycles, SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT * 8u, "cudaMalloc phase_clocks");
}

static bool SparkValidationInitializeDeviceInputs(
    SparkValidationDeviceBuffers *buffers)
{
    uint16_t cache_seed[
        SPARK_VALIDATION_CACHE_TOKEN_CAPACITY *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS];
    float cos_table[
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)];
    float sin_table[
        SPARK_VALIDATION_POSITION_COUNT *
        (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)];
    uint16_t one_bf16[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    float fp8_scale_inv[
        ((SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_B_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK) *
        ((SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION +
          SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK - 1u) /
         SPARK_GLM52_RESIDENT_DECODE_STAGE_FP8_SCALE_BLOCK)];
    uint16_t input_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint16_t mtp_draft_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT *
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION];
    uint32_t restricted_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT];
    uint32_t sparse_token_indices[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT];
    uint32_t context_length;
    uint32_t position;
    uint32_t slot_mapping;
    uint32_t block_table[SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE];
    uint32_t first_block_token_offset;
    uint32_t mtp_target_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t index;
    uint32_t rope_pair_index;

    memset(cache_seed, 0, sizeof(cache_seed));
    memset(input_hidden, 0, sizeof(input_hidden));
    memset(mtp_draft_hidden, 0, sizeof(mtp_draft_hidden));
    for (index = 0u;
         index < SPARK_VALIDATION_POSITION_COUNT *
            (SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u);
         ++index)
    {
        cos_table[index] = 1.0f;
        sin_table[index] = 0.0f;
    }
    for (rope_pair_index = 0u;
         rope_pair_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT / 2u;
         ++rope_pair_index)
    {
        uint64_t table_offset;
        float angle;

        table_offset =
            ((uint64_t)SPARK_VALIDATION_CURRENT_POSITION *
             (uint64_t)(SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION / 2u)) +
            (uint64_t)rope_pair_index;
        angle = 0.125f * (float)(rope_pair_index + 1u);
        cos_table[table_offset] = cosf(angle);
        sin_table[table_offset] = sinf(angle);
    }
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION;
         ++index)
    {
        one_bf16[index] = SparkValidationFloatToBf16(1.0f);
    }
    for (index = 0u; index < sizeof(fp8_scale_inv) / sizeof(fp8_scale_inv[0]); ++index)
    {
        fp8_scale_inv[index] = 1.0f;
    }
    input_hidden[0] = SparkValidationFloatToBf16(1.0f);
    mtp_draft_hidden[0] = SparkValidationFloatToBf16(1.0f);
    mtp_draft_hidden[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_HIDDEN_DIMENSION] =
        SparkValidationFloatToBf16(1.0f);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT0,
        0u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT1,
        1u);
    SparkValidationSeedCacheSlot(
        cache_seed,
        SPARK_VALIDATION_REMAP_CACHE_SLOT2,
        2u);
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_RESTRICTED_VOCAB_COUNT;
         ++index)
    {
        restricted_token_ids[index] = 1000u + index;
    }
    for (index = 0u;
         index < SPARK_GLM52_RESIDENT_DECODE_STAGE_SELECTED_TOKEN_COUNT;
         ++index)
    {
        sparse_token_indices[index] = index < SPARK_VALIDATION_CONTEXT_LENGTH
            ? index
            : SPARK_GLM52_RESIDENT_DECODE_STAGE_INVALID_TOKEN_ID;
    }
    context_length = SPARK_VALIDATION_CONTEXT_LENGTH;
    position = SPARK_VALIDATION_CURRENT_POSITION;
    slot_mapping = SPARK_VALIDATION_CURRENT_CACHE_SLOT;
    block_table[0] = 1u;
    block_table[1] = 0u;
    first_block_token_offset = SPARK_VALIDATION_FIRST_BLOCK_TOKEN_OFFSET;
    mtp_target_token_ids[0] = SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN;
    mtp_target_token_ids[1] = SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN;
    return
        SparkValidationCopyToDevice(buffers->mla_cache_bf16, cache_seed, sizeof(cache_seed), "copy seeded cache") &&
        SparkValidationCopyToDevice(buffers->input_hidden_bf16, input_hidden, sizeof(input_hidden), "copy input_hidden") &&
        SparkValidationCopyToDevice(buffers->mtp_draft_hidden_bf16, mtp_draft_hidden, sizeof(mtp_draft_hidden), "copy mtp_draft_hidden") &&
        SparkValidationCopyToDevice(buffers->cos_table, cos_table, sizeof(cos_table), "copy cos_table") &&
        SparkValidationCopyToDevice(buffers->sin_table, sin_table, sizeof(sin_table), "copy sin_table") &&
        SparkValidationCopyToDevice(buffers->attention_norm_weight_bf16, one_bf16, sizeof(one_bf16), "copy attention_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_query_a_norm_weight_bf16, one_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_QUERY_A_DIMENSION * 2u, "copy raw_query_a_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_kv_a_norm_weight_bf16, one_bf16, SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION * 2u, "copy raw_kv_a_norm_weight") &&
        SparkValidationCopyToDevice(buffers->raw_query_a_weight_scale_inv_f32, fp8_scale_inv, 16u * 48u * 4u, "copy raw_query_a_scale") &&
        SparkValidationCopyToDevice(buffers->raw_query_b_weight_scale_inv_f32, fp8_scale_inv, 128u * 16u * 4u, "copy raw_query_b_scale") &&
        SparkValidationCopyToDevice(buffers->raw_kv_a_weight_scale_inv_f32, fp8_scale_inv, 5u * 48u * 4u, "copy raw_kv_a_scale") &&
        SparkValidationCopyToDevice(buffers->raw_kv_b_weight_scale_inv_f32, fp8_scale_inv, 224u * 4u * 4u, "copy raw_kv_b_scale") &&
        SparkValidationCopyToDevice(buffers->final_norm_weight_bf16, one_bf16, sizeof(one_bf16), "copy final_norm_weight") &&
        SparkValidationCopyToDevice(buffers->restricted_token_ids, restricted_token_ids, sizeof(restricted_token_ids), "copy restricted_token_ids") &&
        SparkValidationCopyToDevice(buffers->sparse_token_indices, sparse_token_indices, sizeof(sparse_token_indices), "copy sparse indices") &&
        SparkValidationCopyToDevice(buffers->context_lengths, &context_length, sizeof(context_length), "copy context_length") &&
        SparkValidationCopyToDevice(buffers->positions, &position, sizeof(position), "copy position") &&
        SparkValidationCopyToDevice(buffers->slot_mapping, &slot_mapping, sizeof(slot_mapping), "copy slot_mapping") &&
        SparkValidationCopyToDevice(buffers->block_table, block_table, sizeof(block_table), "copy block_table") &&
        SparkValidationCopyToDevice(buffers->first_block_token_offsets, &first_block_token_offset, sizeof(first_block_token_offset), "copy first_block_token_offset") &&
        SparkValidationCopyToDevice(buffers->mtp_target_token_ids, mtp_target_token_ids, sizeof(mtp_target_token_ids), "copy mtp targets") &&
        SparkValidationSeedProjectionFixtureWeights(buffers);
}

static void SparkValidationConfigureNode(
    SparkValidationDeviceBuffers *buffers,
    cudaStream_t cuda_stream,
    SparkGlm52ResidentDecodeStagePipelineSlot *pipeline_slot,
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState *cuda_slot_state,
    SparkGlm52ResidentDecodeStageNodeContext *node_context)
{
    memset(pipeline_slot, 0, sizeof(*pipeline_slot));
    memset(cuda_slot_state, 0, sizeof(*cuda_slot_state));
    memset(node_context, 0, sizeof(*node_context));
    pipeline_slot->cuda_stream = (void *)cuda_stream;
    pipeline_slot->input_hidden_bf16 = buffers->input_hidden_bf16;
    pipeline_slot->normalized_hidden_bf16 = buffers->normalized_hidden_bf16;
    pipeline_slot->query_latent_bf16 = buffers->query_latent_bf16;
    pipeline_slot->query_rope_input_bf16 = buffers->query_rope_input_bf16;
    pipeline_slot->key_rope_input_bf16 = buffers->key_rope_input_bf16;
    pipeline_slot->current_kv_latent_bf16 = buffers->current_kv_latent_bf16;
    pipeline_slot->raw_query_a_bf16 = buffers->raw_query_a_bf16;
    pipeline_slot->raw_query_a_normalized_bf16 =
        buffers->raw_query_a_normalized_bf16;
    pipeline_slot->raw_query_b_bf16 = buffers->raw_query_b_bf16;
    pipeline_slot->raw_kv_a_bf16 = buffers->raw_kv_a_bf16;
    pipeline_slot->raw_kv_a_normalized_bf16 =
        buffers->raw_kv_a_normalized_bf16;
    pipeline_slot->raw_kv_b_bf16 = buffers->raw_kv_b_bf16;
    pipeline_slot->positions = buffers->positions;
    pipeline_slot->slot_mapping = buffers->slot_mapping;
    pipeline_slot->block_table = buffers->block_table;
    pipeline_slot->context_lengths = buffers->context_lengths;
    pipeline_slot->first_block_token_offsets = buffers->first_block_token_offsets;
    pipeline_slot->dsa_token_scores = buffers->dsa_token_scores;
    pipeline_slot->sparse_token_indices = buffers->sparse_token_indices;
    pipeline_slot->rotated_query_rope_bf16 = buffers->rotated_query_rope_bf16;
    pipeline_slot->attention_output_latent_bf16 = buffers->attention_output_latent_bf16;
    pipeline_slot->attention_projected_hidden_bf16 = buffers->attention_projected_hidden_bf16;
    pipeline_slot->post_attention_hidden_bf16 = buffers->post_attention_hidden_bf16;
    pipeline_slot->mtp_draft_hidden_bf16 = buffers->mtp_draft_hidden_bf16;
    pipeline_slot->restricted_logits = buffers->restricted_logits;
    pipeline_slot->mtp_draft_logits = buffers->mtp_draft_logits;
    pipeline_slot->restricted_selected_token_ids = buffers->restricted_selected_token_ids;
    pipeline_slot->restricted_selected_token_scores = buffers->restricted_selected_token_scores;
    pipeline_slot->mtp_draft_token_ids = buffers->mtp_draft_token_ids;
    pipeline_slot->mtp_target_token_ids = buffers->mtp_target_token_ids;
    pipeline_slot->mtp_accept_mask = buffers->mtp_accept_mask;
    pipeline_slot->mtp_committed_token_ids = buffers->mtp_committed_token_ids;
    pipeline_slot->mtp_event_counters = buffers->mtp_event_counters;
    pipeline_slot->phase_clock_cycles = buffers->phase_clock_cycles;
    cuda_slot_state->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_CUDA_SLOT_STATE_ABI_VERSION;
    node_context->abi_version =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_NODE_CONTEXT_ABI_VERSION;
    node_context->pipeline_slot_count = 1u;
    node_context->max_active_sequence_count =
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    node_context->cache_token_capacity = SPARK_VALIDATION_CACHE_TOKEN_CAPACITY;
    node_context->kv_block_count = SPARK_VALIDATION_KV_BLOCK_COUNT;
    node_context->max_blocks_per_sequence =
        SPARK_VALIDATION_MAX_BLOCKS_PER_SEQUENCE;
    node_context->position_count = SPARK_VALIDATION_POSITION_COUNT;
    node_context->dsa_candidate_count = 64u;
    node_context->qk_scale = 0.0416666679f;
    node_context->rms_norm_epsilon = 0.000001f;
    node_context->cos_table = buffers->cos_table;
    node_context->sin_table = buffers->sin_table;
    node_context->mla_cache_bf16 = buffers->mla_cache_bf16;
    node_context->attention_norm_weight_bf16 =
        buffers->attention_norm_weight_bf16;
    node_context->query_latent_weight_bf16 =
        buffers->query_latent_weight_bf16;
    node_context->query_rope_weight_bf16 = buffers->query_rope_weight_bf16;
    node_context->key_rope_weight_bf16 = buffers->key_rope_weight_bf16;
    node_context->kv_latent_weight_bf16 = buffers->kv_latent_weight_bf16;
    node_context->raw_query_a_weight_bf16 =
        buffers->raw_query_a_weight_bf16;
    node_context->raw_query_a_norm_weight_bf16 =
        buffers->raw_query_a_norm_weight_bf16;
    node_context->raw_query_b_weight_bf16 =
        buffers->raw_query_b_weight_bf16;
    node_context->raw_kv_a_weight_bf16 = buffers->raw_kv_a_weight_bf16;
    node_context->raw_kv_a_norm_weight_bf16 =
        buffers->raw_kv_a_norm_weight_bf16;
    node_context->raw_kv_b_weight_bf16 = buffers->raw_kv_b_weight_bf16;
    node_context->raw_query_a_weight_fp8_e4m3 =
        buffers->raw_query_a_weight_fp8_e4m3;
    node_context->raw_query_a_weight_scale_inv_f32 =
        buffers->raw_query_a_weight_scale_inv_f32;
    node_context->raw_query_b_weight_fp8_e4m3 =
        buffers->raw_query_b_weight_fp8_e4m3;
    node_context->raw_query_b_weight_scale_inv_f32 =
        buffers->raw_query_b_weight_scale_inv_f32;
    node_context->raw_kv_a_weight_fp8_e4m3 =
        buffers->raw_kv_a_weight_fp8_e4m3;
    node_context->raw_kv_a_weight_scale_inv_f32 =
        buffers->raw_kv_a_weight_scale_inv_f32;
    node_context->raw_kv_b_weight_fp8_e4m3 =
        buffers->raw_kv_b_weight_fp8_e4m3;
    node_context->raw_kv_b_weight_scale_inv_f32 =
        buffers->raw_kv_b_weight_scale_inv_f32;
    node_context->attention_output_weight_bf16 =
        buffers->attention_output_weight_bf16;
    node_context->final_norm_weight_bf16 = buffers->final_norm_weight_bf16;
    node_context->restricted_lm_head_weight_bf16 =
        buffers->restricted_lm_head_weight_bf16;
    node_context->mtp_mxfp4_weight_payload_u8 =
        buffers->mtp_mxfp4_weight_payload_u8;
    node_context->mtp_mxfp4_scale_e8m0_u8 =
        buffers->mtp_mxfp4_scale_e8m0_u8;
    node_context->restricted_token_ids = buffers->restricted_token_ids;
    node_context->pipeline_slots = pipeline_slot;
    node_context->cuda_pipeline_slot_states = cuda_slot_state;
    node_context->projection_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PROJECTION_RAW_GLM_FP8_E4M3;
    node_context->sparse_index_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_SPARSE_INDEX_PRESELECTED;
    node_context->launch_check_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_LAUNCH_CHECK_SYNC_ON_ERROR;
    node_context->phase_clock_mode =
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_DEVICE_CLOCK64;
}

static bool SparkValidationRunOnce(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    float *elapsed_microseconds)
{
    SparkValidationCompletionState completion_state;
    SparkGlm52ResidentDecodeStageBackendCompletion completion;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    SparkStatus status;
    float elapsed_milliseconds;
    bool succeeded;

    completion_state.completion_count.store(0u, std::memory_order_release);
    completion.function = SparkValidationCompletion;
    completion.context = &completion_state;
    start_event = 0;
    stop_event = 0;
    succeeded =
        SparkValidationCudaSucceeded(cudaEventCreate(&start_event), "cudaEventCreate start") &&
        SparkValidationCudaSucceeded(cudaEventCreate(&stop_event), "cudaEventCreate stop") &&
        SparkValidationCudaSucceeded(cudaEventRecord(start_event, cuda_stream), "cudaEventRecord start");
    if (!succeeded)
    {
        return false;
    }
    status = SparkGlm52ResidentDecodeStageBackendSubmit(
        node_context,
        0u,
        SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT,
        &completion);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "backend submit failed: %s\n", SparkStatusToString(status));
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventRecord(stop_event, cuda_stream), "cudaEventRecord stop") ||
        !SparkValidationCudaSucceeded(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop"))
    {
        return false;
    }
    if (completion_state.completion_count.load(std::memory_order_acquire) != 1u)
    {
        fprintf(stderr, "backend completion callback did not run\n");
        return false;
    }
    if (!SparkValidationCudaSucceeded(
            cudaEventElapsedTime(&elapsed_milliseconds, start_event, stop_event),
            "cudaEventElapsedTime"))
    {
        return false;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    *elapsed_microseconds = elapsed_milliseconds * 1000.0f;
    return true;
}

static float SparkValidationReferenceAttentionValue(
    const float *query_latent_values,
    const float *query_rope_values,
    const float cache_latent_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT],
    const float cache_rope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT],
    uint32_t output_dimension)
{
    float scores[SPARK_VALIDATION_CONTEXT_LENGTH];
    float maximum_score;
    float exponential_sum;
    float expected_value;
    uint32_t token_index;
    uint32_t dimension_index;
    uint32_t rope_dimension_index;

    maximum_score = -1.0e30f;
    for (token_index = 0u;
         token_index < SPARK_VALIDATION_CONTEXT_LENGTH;
         ++token_index)
    {
        scores[token_index] = 0.0f;
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            scores[token_index] +=
                query_latent_values[dimension_index] *
                cache_latent_values[token_index][dimension_index];
        }
        for (rope_dimension_index = 0u;
             rope_dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++rope_dimension_index)
        {
            scores[token_index] +=
                query_rope_values[rope_dimension_index] *
                cache_rope_values[token_index][rope_dimension_index];
        }
        scores[token_index] *= 0.0416666679f;
        if (scores[token_index] > maximum_score)
        {
            maximum_score = scores[token_index];
        }
    }
    exponential_sum = 0.0f;
    expected_value = 0.0f;
    for (token_index = 0u;
         token_index < SPARK_VALIDATION_CONTEXT_LENGTH;
         ++token_index)
    {
        float weight;

        weight = expf(scores[token_index] - maximum_score);
        exponential_sum += weight;
        expected_value +=
            weight * cache_latent_values[token_index][output_dimension];
    }
    return expected_value / exponential_sum;
}

static bool SparkValidationCheckOutputs(
    SparkValidationDeviceBuffers *buffers)
{
    uint16_t current_kv_value[
        SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t cached_kv_value[
        SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t key_rope_value[SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT];
    uint16_t cached_rope_value[SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT];
    uint16_t query_latent_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint16_t query_rope_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT];
    uint16_t attention_output_value[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    uint32_t selected_token_id;
    uint32_t mtp_draft_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t mtp_accept_mask[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t mtp_committed_token_ids[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT];
    uint32_t counters[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_EVENT_COUNTER_COUNT];
    uint64_t phase_clocks[
        SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_CLOCK_COUNT];
    float query_latent_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    float query_rope_values[
        SPARK_VALIDATION_CHECKED_HEAD_COUNT]
        [SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT];
    float cache_latent_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT];
    float cache_rope_values[
        SPARK_VALIDATION_CONTEXT_LENGTH]
        [SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT];
    uint32_t head_index;
    uint32_t dimension_index;
    uint32_t token_index;

    memset(current_kv_value, 0, sizeof(current_kv_value));
    memset(cached_kv_value, 0, sizeof(cached_kv_value));
    memset(key_rope_value, 0, sizeof(key_rope_value));
    memset(cached_rope_value, 0, sizeof(cached_rope_value));
    memset(query_latent_value, 0, sizeof(query_latent_value));
    memset(query_rope_value, 0, sizeof(query_rope_value));
    memset(attention_output_value, 0, sizeof(attention_output_value));
    selected_token_id = 0u;
    memset(mtp_draft_token_ids, 0, sizeof(mtp_draft_token_ids));
    memset(mtp_accept_mask, 0, sizeof(mtp_accept_mask));
    memset(mtp_committed_token_ids, 0, sizeof(mtp_committed_token_ids));
    memset(counters, 0, sizeof(counters));
    memset(phase_clocks, 0, sizeof(phase_clocks));
    if (!SparkValidationCudaSucceeded(
            cudaMemcpy(
                current_kv_value,
                buffers->current_kv_latent_bf16,
                sizeof(current_kv_value),
                cudaMemcpyDeviceToHost),
            "copy current_kv") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                cached_kv_value,
                &buffers->mla_cache_bf16[
                    (SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    0u],
                sizeof(cached_kv_value),
                cudaMemcpyDeviceToHost),
            "copy cached current kv") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                key_rope_value,
                buffers->key_rope_input_bf16,
                sizeof(key_rope_value),
                cudaMemcpyDeviceToHost),
            "copy key_rope_input") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                cached_rope_value,
                &buffers->mla_cache_bf16[
                    (SPARK_VALIDATION_CURRENT_CACHE_SLOT *
                     SPARK_GLM52_RESIDENT_DECODE_STAGE_CACHE_TOKEN_ELEMENTS) +
                    SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                sizeof(cached_rope_value),
                cudaMemcpyDeviceToHost),
            "copy cached current rope") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                &selected_token_id,
                buffers->restricted_selected_token_ids,
                sizeof(selected_token_id),
                cudaMemcpyDeviceToHost),
            "copy selected_token") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_draft_token_ids,
                buffers->mtp_draft_token_ids,
                sizeof(mtp_draft_token_ids),
                cudaMemcpyDeviceToHost),
            "copy mtp_draft_token_ids") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_accept_mask,
                buffers->mtp_accept_mask,
                sizeof(mtp_accept_mask),
                cudaMemcpyDeviceToHost),
            "copy mtp_accept_mask") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                mtp_committed_token_ids,
                buffers->mtp_committed_token_ids,
                sizeof(mtp_committed_token_ids),
                cudaMemcpyDeviceToHost),
            "copy mtp_committed_token_ids") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                counters,
                buffers->mtp_event_counters,
                sizeof(counters),
                cudaMemcpyDeviceToHost),
            "copy mtp_event_counters") ||
        !SparkValidationCudaSucceeded(
            cudaMemcpy(
                phase_clocks,
                buffers->phase_clock_cycles,
                sizeof(phase_clocks),
                cudaMemcpyDeviceToHost),
            "copy phase_clock_cycles"))
    {
        return false;
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        if (!SparkValidationCudaSucceeded(
                cudaMemcpy(
                    query_latent_value[head_index],
                    &buffers->query_latent_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                    sizeof(query_latent_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy query_latent checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    query_rope_value[head_index],
                    &buffers->rotated_query_rope_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_ROPE_DIMENSION],
                    sizeof(query_rope_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy rotated_query_rope checked head") ||
            !SparkValidationCudaSucceeded(
                cudaMemcpy(
                    attention_output_value[head_index],
                    &buffers->attention_output_latent_bf16[
                        (uint64_t)head_index *
                        (uint64_t)SPARK_GLM52_RESIDENT_DECODE_STAGE_LATENT_DIMENSION],
                    sizeof(attention_output_value[head_index]),
                    cudaMemcpyDeviceToHost),
                "copy attention_output checked head"))
        {
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        if (current_kv_value[dimension_index] == 0u ||
            cached_kv_value[dimension_index] != current_kv_value[dimension_index])
        {
            fprintf(
                stderr,
                "current KV value was not written to expected cache slot dim=%u\n",
                dimension_index);
            return false;
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         dimension_index += 2u)
    {
        float first_input;
        float second_input;
        float cosine;
        float sine;
        float expected_first;
        float expected_second;
        float observed_first;
        float observed_second;

        first_input = SparkValidationBf16ToFloat(
            key_rope_value[dimension_index]);
        second_input = SparkValidationBf16ToFloat(
            key_rope_value[dimension_index + 1u]);
        cosine = cosf(0.125f * (float)((dimension_index / 2u) + 1u));
        sine = sinf(0.125f * (float)((dimension_index / 2u) + 1u));
        expected_first = (first_input * cosine) - (second_input * sine);
        expected_second = (first_input * sine) + (second_input * cosine);
        observed_first = SparkValidationBf16ToFloat(
            cached_rope_value[dimension_index]);
        observed_second = SparkValidationBf16ToFloat(
            cached_rope_value[dimension_index + 1u]);
        if (first_input == 0.0f ||
            second_input == 0.0f ||
            fabsf(observed_first - expected_first) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE ||
            fabsf(observed_second - expected_second) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE)
        {
            fprintf(
                stderr,
                "current key RoPE rotation mismatch pair=%u observed=(%.6f,%.6f) expected=(%.6f,%.6f)\n",
                dimension_index / 2u,
                observed_first,
                observed_second,
                expected_first,
                expected_second);
            return false;
        }
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            if (query_latent_value[head_index][dimension_index] == 0u)
            {
                fprintf(stderr, "query latent fixture did not become nonzero\n");
                return false;
            }
            query_latent_values[head_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    query_latent_value[head_index][dimension_index]);
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            if (query_rope_value[head_index][dimension_index] == 0u)
            {
                fprintf(stderr, "query RoPE fixture did not become nonzero\n");
                return false;
            }
            query_rope_values[head_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    query_rope_value[head_index][dimension_index]);
        }
    }
    for (token_index = 0u;
         token_index < SPARK_VALIDATION_CONTEXT_LENGTH - 1u;
         ++token_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            cache_latent_values[token_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    SparkValidationFloatToBf16(
                        SparkValidationSeedLatentValue(
                            token_index,
                            dimension_index)));
        }
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
             ++dimension_index)
        {
            cache_rope_values[token_index][dimension_index] =
                SparkValidationBf16ToFloat(
                    SparkValidationFloatToBf16(
                        SparkValidationSeedRopeValue(
                            token_index,
                            dimension_index)));
        }
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_latent_values[
            SPARK_VALIDATION_CONTEXT_LENGTH - 1u][dimension_index] =
            SparkValidationBf16ToFloat(cached_kv_value[dimension_index]);
    }
    for (dimension_index = 0u;
         dimension_index < SPARK_VALIDATION_CHECKED_ROPE_DIMENSION_COUNT;
         ++dimension_index)
    {
        cache_rope_values[
            SPARK_VALIDATION_CONTEXT_LENGTH - 1u][dimension_index] =
            SparkValidationBf16ToFloat(cached_rope_value[dimension_index]);
    }
    for (head_index = 0u;
         head_index < SPARK_VALIDATION_CHECKED_HEAD_COUNT;
         ++head_index)
    {
        for (dimension_index = 0u;
             dimension_index < SPARK_VALIDATION_CHECKED_LATENT_DIMENSION_COUNT;
             ++dimension_index)
        {
            float expected_attention;
            float observed_attention;

            expected_attention = SparkValidationReferenceAttentionValue(
                query_latent_values[head_index],
                query_rope_values[head_index],
                cache_latent_values,
                cache_rope_values,
                dimension_index);
            observed_attention = SparkValidationBf16ToFloat(
                attention_output_value[head_index][dimension_index]);
            if (fabsf(observed_attention - expected_attention) >
                SPARK_VALIDATION_ATTENTION_TOLERANCE)
            {
                fprintf(
                    stderr,
                    "attention reference mismatch head=%u dim=%u observed=%.6f expected=%.6f\n",
                    head_index,
                    dimension_index,
                    observed_attention,
                    expected_attention);
                return false;
            }
        }
    }
    if (selected_token_id != SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN)
    {
        fprintf(
            stderr,
            "restricted argmax mismatch observed=%u expected=%u\n",
            selected_token_id,
            SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN);
        return false;
    }
    if (mtp_draft_token_ids[0] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_draft_token_ids[1] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_accept_mask[0] != 1u ||
        mtp_accept_mask[1] != 0u ||
        mtp_committed_token_ids[0] != SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN ||
        mtp_committed_token_ids[1] != SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN)
    {
        fprintf(stderr, "MTP draft/verify fixture did not produce expected tokens\n");
        return false;
    }
    if (phase_clocks[
            SPARK_GLM52_RESIDENT_DECODE_STAGE_PHASE_COMPLETION_READY] == 0u)
    {
        fprintf(stderr, "phase clock completion marker was not written\n");
        return false;
    }
    if (counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] == 0u)
    {
        fprintf(stderr, "MTP commit counter was not incremented\n");
        return false;
    }
    if (counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ACCEPTED] != 1u ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_REJECTED] != 1u ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_COMMITTED] != 2u ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_ROLLBACK] != 1u ||
        counters[SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_CANCELLED] != 0u)
    {
        fprintf(stderr, "MTP event counters do not match accept/reject fixture\n");
        return false;
    }
    return true;
}

static bool SparkValidationRunDriverOnce(
    SparkGlm52ResidentDecodeStageNodeContext *node_context,
    cudaStream_t cuda_stream,
    const char *driver_path,
    float *elapsed_microseconds)
{
    SparkValidationDriverCompletionState completion_state;
    SparkOrchestratorConfiguration orchestrator_configuration;
    SparkOrchestrator *orchestrator;
    SparkOrchestratorNodeHandle node_handle;
    SparkOrchestratorDriverHandle driver_handle;
    SparkOrchestratorRouteHandle route_handle;
    SparkModelDriverRuntimeSnapshot runtime_snapshot;
    SparkModelDriverFrame frame;
    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    SparkStatus status;
    char error_buffer[1024];
    float elapsed_milliseconds;

    memset(&completion_state, 0, sizeof(completion_state));
    memset(&orchestrator_configuration, 0, sizeof(orchestrator_configuration));
    memset(&frame, 0, sizeof(frame));
    orchestrator_configuration.node_capacity = 1u;
    orchestrator_configuration.driver_capacity = 1u;
    orchestrator_configuration.route_capacity = 1u;
    orchestrator_configuration.route_endpoint_capacity = 1u;
    orchestrator_configuration.completion_function =
        SparkValidationDriverCompletion;
    orchestrator_configuration.completion_context = &completion_state;
    orchestrator = 0;
    start_event = 0;
    stop_event = 0;
    status = SparkCreateOrchestrator(
        &orchestrator_configuration,
        &orchestrator);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkCreateOrchestrator failed: %s\n", SparkStatusToString(status));
        return false;
    }
    status = SparkOrchestratorAddNode(
        orchestrator,
        "cuda-node-0",
        SPARK_GLM52_RESIDENT_DECODE_STAGE_TARGET,
        node_context,
        &node_handle);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorAddNode failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorAttachDriver(
        orchestrator,
        node_handle,
        driver_path,
        &driver_handle,
        error_buffer,
        sizeof(error_buffer));
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorAttachDriver failed: %s: %s\n", SparkStatusToString(status), error_buffer);
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorResolveRoute(
        orchestrator,
        "zai.glm-5.2.resident-decode-stage-firmware",
        "bf16-h6144-h64-d512-r64-k2048-b64-rv256-mtp2-v1",
        "resident_decode",
        "decode",
        &route_handle);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorResolveRoute failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventCreate(&start_event), "cudaEventCreate start") ||
        !SparkValidationCudaSucceeded(cudaEventCreate(&stop_event), "cudaEventCreate stop") ||
        !SparkValidationCudaSucceeded(cudaEventRecord(start_event, cuda_stream), "cudaEventRecord start"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    frame.request_id = 9001u;
    frame.sequence_id = 70001u;
    frame.sequence_position = 17u;
    frame.active_slot_count = SPARK_VALIDATION_ACTIVE_SEQUENCE_COUNT;
    frame.new_token_count = SPARK_GLM52_RESIDENT_DECODE_STAGE_MTP_DRAFT_TOKEN_COUNT + 1u;
    frame.residency.owner = 1u;
    status = SparkOrchestratorSubmit(orchestrator, route_handle, &frame);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorSubmit failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if ((frame.flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u ||
        frame.driver_dispatch_slot != 0u)
    {
        fprintf(stderr, "driver dispatch slot was not assigned by admission\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(cudaEventRecord(stop_event, cuda_stream), "cudaEventRecord stop") ||
        !SparkValidationCudaSucceeded(cudaEventSynchronize(stop_event), "cudaEventSynchronize stop"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (completion_state.completion_count.load(std::memory_order_acquire) != 1u ||
        completion_state.completion.request_id != frame.request_id ||
        completion_state.completion.status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "orchestrator completion did not match submitted frame\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    status = SparkOrchestratorGetDriverProgramSnapshot(
        orchestrator,
        driver_handle,
        "decode",
        &runtime_snapshot);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "SparkOrchestratorGetDriverProgramSnapshot failed: %s\n", SparkStatusToString(status));
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (runtime_snapshot.submitted_count != 1u ||
        runtime_snapshot.completed_count != 1u ||
        runtime_snapshot.active_submission_count != 0u ||
        runtime_snapshot.host_callback_completion_count != 1u ||
        runtime_snapshot.host_staging_bytes_per_submit != 0u ||
        runtime_snapshot.device_memcpy_bytes_per_submit != 0u)
    {
        fprintf(stderr, "orchestrator snapshot counters are not clean\n");
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    if (!SparkValidationCudaSucceeded(
            cudaEventElapsedTime(&elapsed_milliseconds, start_event, stop_event),
            "cudaEventElapsedTime"))
    {
        SparkDestroyOrchestrator(orchestrator);
        return false;
    }
    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);
    *elapsed_microseconds = elapsed_milliseconds * 1000.0f;
    SparkDestroyOrchestrator(orchestrator);
    return true;
}

int main(int argc, char **argv)
{
    SparkValidationDeviceBuffers buffers;
    SparkGlm52ResidentDecodeStagePipelineSlot pipeline_slot;
    SparkGlm52ResidentDecodeStageCudaPipelineSlotState cuda_slot_state;
    SparkGlm52ResidentDecodeStageNodeContext node_context;
    cudaStream_t cuda_stream;
    double maximum_stage_microseconds;
    double total_microseconds;
    double maximum_observed_microseconds;
    uint32_t iteration;

    if (argc != 2 && argc != 3)
    {
        fprintf(stderr, "usage: %s MAX_STAGE_MICROSECONDS [DRIVER_SO]\n", argv[0]);
        return 2;
    }
    maximum_stage_microseconds = atof(argv[1]);
    if (maximum_stage_microseconds <= 0.0)
    {
        fprintf(stderr, "MAX_STAGE_MICROSECONDS must be positive\n");
        return 2;
    }
    cuda_stream = 0;
    if (!SparkValidationCudaSucceeded(cudaSetDevice(0), "cudaSetDevice") ||
        !SparkValidationCudaSucceeded(
            cudaStreamCreateWithFlags(&cuda_stream, cudaStreamNonBlocking),
            "cudaStreamCreate") ||
        !SparkValidationAllocateDeviceBuffers(&buffers) ||
        !SparkValidationInitializeDeviceInputs(&buffers))
    {
        return 2;
    }
    SparkValidationConfigureNode(
        &buffers,
        cuda_stream,
        &pipeline_slot,
        &cuda_slot_state,
        &node_context);
    if (argc == 3)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunDriverOnce(
                &node_context,
                cuda_stream,
                argv[2],
                &elapsed_microseconds) ||
            !SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize") ||
            !SparkValidationCheckOutputs(&buffers))
        {
            return 2;
        }
        if ((double)elapsed_microseconds > maximum_stage_microseconds)
        {
            fprintf(
                stderr,
                "glm52_resident_decode_stage orchestrator validation failed elapsed_us=%.3f limit_us=%.3f\n",
                (double)elapsed_microseconds,
                maximum_stage_microseconds);
            return 1;
        }
        printf(
            "glm52_resident_decode_stage orchestrator validation passed fixture=remapped_nonzero_context4_h4_d8_r4 elapsed_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
            (double)elapsed_microseconds,
            maximum_stage_microseconds,
            SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
            SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
            (unsigned long long)cuda_slot_state.launch_chain_count,
            (unsigned long long)cuda_slot_state.graph_capture_count,
            (unsigned long long)cuda_slot_state.graph_replay_count);
        return 0;
    }
    total_microseconds = 0.0;
    maximum_observed_microseconds = 0.0;
    for (iteration = 0u;
         iteration < SPARK_VALIDATION_WARMUP_COUNT + SPARK_VALIDATION_MEASUREMENT_COUNT;
         ++iteration)
    {
        float elapsed_microseconds;

        if (!SparkValidationRunOnce(
                &node_context,
                cuda_stream,
                &elapsed_microseconds))
        {
            return 2;
        }
        if (iteration >= SPARK_VALIDATION_WARMUP_COUNT)
        {
            total_microseconds += (double)elapsed_microseconds;
            if ((double)elapsed_microseconds > maximum_observed_microseconds)
            {
                maximum_observed_microseconds = (double)elapsed_microseconds;
            }
        }
    }
    if (!SparkValidationCudaSucceeded(cudaStreamSynchronize(cuda_stream), "cudaStreamSynchronize") ||
        !SparkValidationCheckOutputs(&buffers))
    {
        return 2;
    }
    SparkGlm52ResidentDecodeStageBackendQuiesce(&node_context);
    if (maximum_observed_microseconds > maximum_stage_microseconds)
    {
        fprintf(
            stderr,
            "glm52_resident_decode_stage validation failed average_us=%.3f maximum_us=%.3f limit_us=%.3f\n",
            total_microseconds / SPARK_VALIDATION_MEASUREMENT_COUNT,
            maximum_observed_microseconds,
            maximum_stage_microseconds);
        return 1;
    }
    printf(
        "glm52_resident_decode_stage validation passed fixture=remapped_nonzero_context4_h4_d8_r4 average_us=%.3f maximum_us=%.3f limit_us=%.3f restricted_token=%u mtp_draft=%u mtp_reject=%u launch_chains=%llu graph_captures=%llu graph_replays=%llu\n",
        total_microseconds / SPARK_VALIDATION_MEASUREMENT_COUNT,
        maximum_observed_microseconds,
        maximum_stage_microseconds,
        SPARK_VALIDATION_EXPECTED_RESTRICTED_TOKEN,
        SPARK_VALIDATION_EXPECTED_MTP_DRAFT_TOKEN,
        SPARK_VALIDATION_EXPECTED_MTP_REJECT_TOKEN,
        (unsigned long long)cuda_slot_state.launch_chain_count,
        (unsigned long long)cuda_slot_state.graph_capture_count,
        (unsigned long long)cuda_slot_state.graph_replay_count);
    return 0;
}
