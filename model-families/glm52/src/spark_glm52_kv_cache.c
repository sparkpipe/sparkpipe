#include "sparkpipe/spark_glm52_kv_cache.h"

#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <unistd.h>
#endif

static uint64_t SparkGlm52KvCacheMulU64(
    uint64_t left,
    uint64_t right)
{
    return left * right;
}

static uint64_t SparkGlm52KvCacheCeilDivU64(
    uint64_t numerator,
    uint64_t denominator)
{
    if (denominator == 0u)
    {
        return 0u;
    }
    return (numerator + denominator - 1u) / denominator;
}

static SparkStatus SparkGlm52KvCacheCheckedMulU64(
    uint64_t left,
    uint64_t right,
    uint64_t *value_out)
{
    if (value_out == 0 || (right != 0u && left > UINT64_MAX / right))
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *value_out = left * right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheCheckedAddU64(
    uint64_t left,
    uint64_t right,
    uint64_t *value_out)
{
    if (value_out == 0 || left > UINT64_MAX - right)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *value_out = left + right;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheAlignUpU64(
    uint64_t value,
    uint32_t alignment,
    uint64_t *value_out)
{
    uint64_t aligned_value;

    if (value_out == 0 || alignment == 0u ||
        (alignment & (alignment - 1u)) != 0u ||
        value > UINT64_MAX - ((uint64_t)alignment - 1u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    aligned_value = (value + alignment - 1u) & ~((uint64_t)alignment - 1u);
    if (aligned_value < value)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    *value_out = aligned_value;
    return SPARK_STATUS_OK;
}

uint32_t SparkGlm52KvCacheDsaSourceLayer(uint32_t layer_index)
{
    uint32_t adjusted_layer_index;

    if (layer_index >= SPARK_GLM52_MODEL_LAYER_COUNT)
    {
        return UINT32_MAX;
    }
    if (layer_index < SPARK_GLM52_MODEL_FIRST_ROUTED_LAYER)
    {
        return layer_index;
    }
    adjusted_layer_index = layer_index -
        (SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET - 1u);
    return (SPARK_GLM52_MODEL_DSA_INDEX_SKIP_TOPK_OFFSET - 1u) +
        (adjusted_layer_index -
         (adjusted_layer_index %
          SPARK_GLM52_MODEL_DSA_INDEX_SHARE_GROUP_LAYER_COUNT));
}

SparkStatus SparkGlm52KvCacheCalculateJitStageBudget(
    const SparkGlm52KvJitStageBudgetRequest *request,
    SparkGlm52KvJitStageBudget *budget)
{
    uint64_t attention_bytes_per_token_per_layer;
    uint64_t dsa_bytes_per_token_per_layer;
    uint64_t summary_bytes_per_index_layer_block;
    uint64_t resident_token_bytes;
    uint64_t payload_token_bytes;
    uint64_t record_unaligned_bytes;
    uint64_t active_token_capacity;
    uint64_t backing_token_capacity;
    uint64_t compact_selected_token_count;
    uint64_t key_nope_elements;
    uint64_t value_elements;
    uint32_t layer_offset;
    uint32_t local_dsa_index_layer_count;
    SparkStatus status;

    if (request == 0 || budget == 0 ||
        request->abi_version != SPARK_GLM52_KV_JIT_STAGE_BUDGET_ABI_VERSION ||
        request->descriptor_bytes !=
            SPARK_GLM52_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES ||
        request->layer_count == 0u ||
        request->first_layer_index >= SPARK_GLM52_MODEL_LAYER_COUNT ||
        request->layer_count >
            SPARK_GLM52_MODEL_LAYER_COUNT - request->first_layer_index ||
        request->physical_pool_token_capacity == 0u ||
        request->backing_block_capacity == 0u ||
        request->active_sequence_count == 0u ||
        request->backing_request_count == 0u ||
        request->selected_token_count == 0u ||
        request->block_token_count == 0u ||
        (request->attention_cache_layout !=
            SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE &&
         request->attention_cache_layout !=
            SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED &&
         request->attention_cache_layout !=
            SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED_FP8_E4M3 &&
         request->attention_cache_layout !=
            SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3) ||
        ((request->attention_cache_layout ==
            SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED_FP8_E4M3 ||
          request->attention_cache_layout ==
            SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3) &&
         (request->fp8_scale_block_size == 0u ||
          (request->fp8_scale_block_size & 15u) != 0u)) ||
        request->physical_pool_token_capacity % request->block_token_count != 0u ||
        request->backing_block_capacity <
            request->physical_pool_token_capacity / request->block_token_count ||
        request->include_mtp_layer > 1u ||
        (request->include_mtp_layer != 0u &&
         request->first_layer_index + request->layer_count !=
            SPARK_GLM52_MODEL_LAYER_COUNT) ||
        request->record_alignment_bytes <
            SPARK_GLM52_KV_JIT_DEFAULT_RECORD_ALIGNMENT ||
        (request->record_alignment_bytes &
            (request->record_alignment_bytes - 1u)) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    local_dsa_index_layer_count = 0u;
    for (layer_offset = 0u; layer_offset < request->layer_count; ++layer_offset)
    {
        uint32_t layer_index;
        layer_index = request->first_layer_index + layer_offset;
        if (SparkGlm52KvCacheDsaSourceLayer(layer_index) == layer_index)
        {
            local_dsa_index_layer_count += 1u;
        }
    }
    if (request->attention_cache_layout ==
        SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED_FP8_E4M3)
    {
        attention_bytes_per_token_per_layer =
            (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
            ((SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
             request->fp8_scale_block_size - 1u) /
             request->fp8_scale_block_size) * sizeof(float);
    }
    else if (request->attention_cache_layout ==
        SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3)
    {
        key_nope_elements =
            (uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION;
        value_elements =
            (uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
        attention_bytes_per_token_per_layer =
            SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
            key_nope_elements + value_elements +
            (((SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
                request->fp8_scale_block_size - 1u) /
               request->fp8_scale_block_size) +
             ((key_nope_elements + request->fp8_scale_block_size - 1u) /
               request->fp8_scale_block_size) +
             ((value_elements + request->fp8_scale_block_size - 1u) /
               request->fp8_scale_block_size)) * sizeof(float);
    }
    else if (request->attention_cache_layout ==
        SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE)
    {
        attention_bytes_per_token_per_layer =
            ((uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
             ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
              SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
             ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
              SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION)) * sizeof(uint16_t);
    }
    else
    {
        attention_bytes_per_token_per_layer =
            (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t);
    }
    dsa_bytes_per_token_per_layer =
        (uint64_t)SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * sizeof(uint16_t);
    summary_bytes_per_index_layer_block =
        (2u * dsa_bytes_per_token_per_layer) + sizeof(uint8_t);

    memset(budget, 0, sizeof(*budget));
    budget->abi_version = SPARK_GLM52_KV_JIT_STAGE_BUDGET_ABI_VERSION;
    budget->descriptor_bytes = SPARK_GLM52_KV_JIT_STAGE_BUDGET_DESCRIPTOR_BYTES;
    budget->first_layer_index = request->first_layer_index;
    budget->layer_count = request->layer_count;
    budget->local_dsa_index_layer_count = local_dsa_index_layer_count;
    budget->include_mtp_layer = request->include_mtp_layer;
    budget->physical_block_capacity =
        request->physical_pool_token_capacity / request->block_token_count;
    budget->backing_block_capacity = request->backing_block_capacity;
    budget->mla_bytes_per_token =
        (uint64_t)request->layer_count * attention_bytes_per_token_per_layer;
    budget->dsa_index_bytes_per_token =
        (uint64_t)local_dsa_index_layer_count * dsa_bytes_per_token_per_layer;
    budget->mtp_bytes_per_token = request->include_mtp_layer != 0u
        ? attention_bytes_per_token_per_layer : 0u;
    status = SparkGlm52KvCacheCheckedAddU64(
        budget->mla_bytes_per_token,
        budget->dsa_index_bytes_per_token,
        &resident_token_bytes);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedAddU64(
            resident_token_bytes,
            budget->mtp_bytes_per_token,
            &budget->resident_bytes_per_token);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            budget->physical_block_capacity,
            (uint64_t)local_dsa_index_layer_count *
                summary_bytes_per_index_layer_block,
            &budget->resident_summary_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            request->physical_pool_token_capacity,
            budget->resident_bytes_per_token,
            &resident_token_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedAddU64(
            resident_token_bytes,
            budget->resident_summary_bytes,
            &budget->resident_pool_bytes);
    }
    payload_token_bytes = 0u;
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            request->block_token_count,
            budget->resident_bytes_per_token,
            &payload_token_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedAddU64(
            payload_token_bytes,
            (uint64_t)local_dsa_index_layer_count *
                summary_bytes_per_index_layer_block,
            &budget->nvme_payload_bytes_per_block);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedAddU64(
            request->record_alignment_bytes,
            budget->nvme_payload_bytes_per_block,
            &record_unaligned_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheAlignUpU64(
            record_unaligned_bytes,
            request->record_alignment_bytes,
            &budget->nvme_record_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            request->backing_block_capacity,
            budget->nvme_record_bytes,
            &budget->nvme_capacity_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            request->active_sequence_count,
            request->selected_token_count,
            &compact_selected_token_count);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkGlm52KvCacheCheckedMulU64(
            compact_selected_token_count,
            budget->mla_bytes_per_token + budget->mtp_bytes_per_token,
            &budget->compact_selected_mla_working_set_bytes);
    }
    if (status != SPARK_STATUS_OK)
    {
        memset(budget, 0, sizeof(*budget));
        return status;
    }

    active_token_capacity = request->physical_pool_token_capacity;
    status = SparkGlm52KvCacheCheckedMulU64(
        request->backing_block_capacity,
        request->block_token_count,
        &backing_token_capacity);
    if (status != SPARK_STATUS_OK)
    {
        memset(budget, 0, sizeof(*budget));
        return status;
    }
    budget->maximum_average_active_context_tokens =
        (uint32_t)(active_token_capacity / request->active_sequence_count);
    backing_token_capacity /= request->backing_request_count;
    budget->maximum_average_backing_context_tokens =
        backing_token_capacity > UINT32_MAX ? UINT32_MAX : (uint32_t)backing_token_capacity;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheCalculateAttentionBytesPerTokenLayer(
    const SparkGlm52KvCacheCapacityRequest *request,
    uint64_t *bytes_per_token_per_layer_out)
{
    uint64_t element_count;
    uint64_t scale_count;

    if (request == 0 || bytes_per_token_per_layer_out == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    switch (request->layout)
    {
        case SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE:
            element_count =
                (uint64_t)request->latent_dimension +
                (uint64_t)request->rope_dimension +
                ((uint64_t)request->head_count *
                 (uint64_t)request->qk_nope_head_dimension) +
                ((uint64_t)request->head_count *
                 (uint64_t)request->value_head_dimension);
            *bytes_per_token_per_layer_out =
                element_count * (uint64_t)request->bytes_per_scalar;
            return SPARK_STATUS_OK;

        case SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED:
            element_count =
                (uint64_t)request->latent_dimension +
                (uint64_t)request->rope_dimension;
            *bytes_per_token_per_layer_out =
                element_count * (uint64_t)request->bytes_per_scalar;
            return SPARK_STATUS_OK;

        case SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED_FP8_E4M3:
            if (request->fp8_scale_block_size == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            element_count =
                (uint64_t)request->latent_dimension +
                (uint64_t)request->rope_dimension;
            scale_count = SparkGlm52KvCacheCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out =
                element_count + (scale_count * sizeof(float));
            return SPARK_STATUS_OK;

        case SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3:
            if (request->fp8_scale_block_size == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            element_count =
                (uint64_t)request->latent_dimension +
                (uint64_t)request->rope_dimension;
            scale_count = SparkGlm52KvCacheCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out =
                element_count + (scale_count * sizeof(float));
            element_count =
                (uint64_t)request->head_count *
                (uint64_t)request->qk_nope_head_dimension;
            scale_count = SparkGlm52KvCacheCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out +=
                element_count + (scale_count * sizeof(float));
            element_count =
                (uint64_t)request->head_count *
                (uint64_t)request->value_head_dimension;
            scale_count = SparkGlm52KvCacheCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out +=
                element_count + (scale_count * sizeof(float));
            return SPARK_STATUS_OK;

        default:
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
}

SparkStatus SparkGlm52KvCacheEstimateCapacity(
    const SparkGlm52KvCacheCapacityRequest *request,
    SparkGlm52KvCacheCapacityEstimate *estimate)
{
    uint64_t attention_bytes_per_token_per_layer;
    uint64_t dsa_index_bytes_per_token;
    uint64_t block_count_per_context;
    uint64_t bytes_per_block_per_layer;
    uint64_t bytes_per_context_per_rank;
    SparkStatus status;

    if (request == 0 || estimate == 0 ||
        request->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        request->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES ||
        request->context_token_count == 0u ||
        request->block_token_count == 0u ||
        request->layer_count == 0u ||
        request->latent_dimension == 0u ||
        request->rope_dimension == 0u ||
        request->bytes_per_scalar == 0u ||
        request->cache_bytes_per_rank == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52KvCacheCalculateAttentionBytesPerTokenLayer(
        request,
        &attention_bytes_per_token_per_layer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    dsa_index_bytes_per_token = 0u;
    if (request->index_key_layer_count != 0u ||
        request->index_key_dimension != 0u ||
        request->index_key_bytes_per_scalar != 0u)
    {
        if (request->index_key_layer_count == 0u ||
            request->index_key_dimension == 0u ||
            request->index_key_bytes_per_scalar == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        dsa_index_bytes_per_token =
            (uint64_t)request->index_key_layer_count *
            (uint64_t)request->index_key_dimension *
            (uint64_t)request->index_key_bytes_per_scalar;
    }

    block_count_per_context = SparkGlm52KvCacheCeilDivU64(
        (uint64_t)request->context_token_count,
        (uint64_t)request->block_token_count);
    bytes_per_block_per_layer =
        (uint64_t)request->block_token_count *
        attention_bytes_per_token_per_layer;
    bytes_per_context_per_rank =
        ((uint64_t)request->context_token_count *
         (uint64_t)request->layer_count *
         attention_bytes_per_token_per_layer) +
        ((uint64_t)request->context_token_count *
         dsa_index_bytes_per_token);
    if (bytes_per_context_per_rank == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(estimate, 0, sizeof(*estimate));
    estimate->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    estimate->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES;
    estimate->layout = request->layout;
    estimate->context_token_count = request->context_token_count;
    estimate->block_token_count = request->block_token_count;
    estimate->layer_count = request->layer_count;
    estimate->block_count_per_context = (uint32_t)block_count_per_context;
    estimate->contexts_per_rank =
        (uint32_t)(request->cache_bytes_per_rank / bytes_per_context_per_rank);
    estimate->attention_bytes_per_token_per_layer =
        attention_bytes_per_token_per_layer;
    estimate->dsa_index_bytes_per_token = dsa_index_bytes_per_token;
    estimate->bytes_per_block_per_layer = bytes_per_block_per_layer;
    estimate->bytes_per_context_per_rank = bytes_per_context_per_rank;
    estimate->unused_cache_bytes_per_rank =
        request->cache_bytes_per_rank -
        ((uint64_t)estimate->contexts_per_rank * bytes_per_context_per_rank);
    return SPARK_STATUS_OK;
}

static uint64_t SparkGlm52KvCacheDefaultBlockStrideBytes(
    const SparkGlm52KvCacheConfiguration *configuration)
{
    uint64_t stride_bytes;

    stride_bytes = SparkGlm52KvCacheMulU64(
        configuration->block_token_count,
        configuration->layer_count);
    stride_bytes = SparkGlm52KvCacheMulU64(
        stride_bytes,
        configuration->kv_head_count);
    stride_bytes = SparkGlm52KvCacheMulU64(
        stride_bytes,
        configuration->head_dim);
    stride_bytes = SparkGlm52KvCacheMulU64(
        stride_bytes,
        configuration->bytes_per_scalar);
    return stride_bytes;
}

static uint32_t SparkGlm52KvCacheConfigurationHasValuePayload(
    const SparkGlm52KvCacheConfiguration *configuration)
{
    return configuration != 0 &&
        (configuration->value_device_base != 0 ||
         configuration->value_block_stride_bytes != 0u);
}

static uint32_t SparkGlm52KvCacheArenaHasValuePayload(
    const SparkGlm52KvCacheArena *arena)
{
    return arena != 0 &&
        (arena->value_device_base != 0u ||
         arena->value_block_stride_bytes != 0u);
}

static uint32_t SparkGlm52KvCacheConfigurationIsValid(
    const SparkGlm52KvCacheConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->physical_block_count == 0u ||
        configuration->block_token_count == 0u ||
        configuration->block_token_count > SPARK_GLM52_KV_CACHE_MAX_BLOCK_TOKENS ||
        configuration->resident_block_capacity > configuration->physical_block_count ||
        configuration->layer_count == 0u ||
        configuration->layer_count > SPARK_GLM52_KV_CACHE_MAX_LAYER_COUNT ||
        configuration->kv_head_count == 0u ||
        configuration->head_dim == 0u ||
        configuration->bytes_per_scalar == 0u ||
        configuration->key_device_base == 0 ||
        configuration->blocks == 0)
    {
        return 0u;
    }
    if (configuration->value_device_base == 0 &&
        configuration->value_block_stride_bytes != 0u)
    {
        return 0u;
    }
    return 1u;
}

static SparkStatus SparkGlm52KvCacheArenaValidate(
    const SparkGlm52KvCacheArena *arena)
{
    if (arena == 0 ||
        arena->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        arena->descriptor_bytes != SPARK_GLM52_KV_CACHE_ARENA_DESCRIPTOR_BYTES ||
        arena->physical_block_count == 0u ||
        arena->block_token_count == 0u ||
        arena->block_token_count > SPARK_GLM52_KV_CACHE_MAX_BLOCK_TOKENS ||
        arena->resident_block_capacity == 0u ||
        arena->resident_block_capacity > arena->physical_block_count ||
        arena->layer_count == 0u ||
        arena->layer_count > SPARK_GLM52_KV_CACHE_MAX_LAYER_COUNT ||
        arena->kv_head_count == 0u ||
        arena->head_dim == 0u ||
        arena->bytes_per_scalar == 0u ||
        arena->key_device_base == 0u ||
        arena->key_block_stride_bytes == 0u ||
        arena->blocks == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arena->value_device_base == 0u && arena->value_block_stride_bytes != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (arena->value_device_base != 0u && arena->value_block_stride_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52KvCacheInitializeBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;

    block = &arena->blocks[physical_block_index];
    memset(block, 0, sizeof(*block));
    block->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    block->descriptor_bytes = SPARK_GLM52_KV_CACHE_BLOCK_DESCRIPTOR_BYTES;
    block->physical_block_index = physical_block_index;
    block->token_capacity = arena->block_token_count;
    block->key_device_address = arena->key_device_base +
        (uintptr_t)(arena->key_block_stride_bytes * physical_block_index);
    block->value_device_address =
        SparkGlm52KvCacheArenaHasValuePayload(arena) != 0u
        ? arena->value_device_base +
            (uintptr_t)(arena->value_block_stride_bytes * physical_block_index)
        : 0u;
}

SparkStatus SparkGlm52KvCacheArenaInitialize(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCacheConfiguration *configuration)
{
    uint64_t default_stride_bytes;
    uint32_t physical_block_index;

    if (arena == 0 || !SparkGlm52KvCacheConfigurationIsValid(configuration))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    default_stride_bytes = SparkGlm52KvCacheDefaultBlockStrideBytes(configuration);
    memset(arena, 0, sizeof(*arena));
    arena->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    arena->descriptor_bytes = SPARK_GLM52_KV_CACHE_ARENA_DESCRIPTOR_BYTES;
    arena->physical_block_count = configuration->physical_block_count;
    arena->block_token_count = configuration->block_token_count;
    arena->resident_block_capacity = configuration->resident_block_capacity != 0u
        ? configuration->resident_block_capacity
        : configuration->physical_block_count;
    arena->layer_count = configuration->layer_count;
    arena->kv_head_count = configuration->kv_head_count;
    arena->head_dim = configuration->head_dim;
    arena->bytes_per_scalar = configuration->bytes_per_scalar;
    arena->key_block_stride_bytes = configuration->key_block_stride_bytes != 0u
        ? configuration->key_block_stride_bytes
        : default_stride_bytes;
    arena->value_block_stride_bytes =
        SparkGlm52KvCacheConfigurationHasValuePayload(configuration) != 0u
        ? configuration->value_block_stride_bytes != 0u
            ? configuration->value_block_stride_bytes
            : default_stride_bytes
        : 0u;
    arena->key_device_base = (uintptr_t)configuration->key_device_base;
    arena->value_device_base =
        SparkGlm52KvCacheConfigurationHasValuePayload(configuration) != 0u
        ? (uintptr_t)configuration->value_device_base
        : 0u;
    arena->blocks = configuration->blocks;

    for (physical_block_index = 0u;
         physical_block_index < arena->physical_block_count;
         ++physical_block_index)
    {
        SparkGlm52KvCacheInitializeBlock(arena, physical_block_index);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaAcquireBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t *physical_block_index_out)
{
    uint32_t physical_block_index;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK || physical_block_index_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    for (physical_block_index = 0u;
         physical_block_index < arena->physical_block_count;
         ++physical_block_index)
    {
        SparkGlm52KvCacheBlock *block;

        block = &arena->blocks[physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
        {
            arena->epoch += 1u;
            block->flags = SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED;
            block->reference_count = 0u;
            block->generation += 1u;
            block->last_used_epoch = arena->epoch;
            arena->allocated_block_count += 1u;
            *physical_block_index_out = physical_block_index;
            return SPARK_STATUS_OK;
        }
    }

    *physical_block_index_out = SPARK_GLM52_KV_CACHE_NO_BLOCK;
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}

SparkStatus SparkGlm52KvCacheArenaRecycleBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (block->reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }

    arena->epoch += 1u;
    block->flags = SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED;
    block->generation += 1u;
    block->last_used_epoch = arena->epoch;
    arena->recycled_block_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaRetainBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block->reference_count += 1u;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    arena->retained_block_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaReleaseBlockReference(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        block->reference_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block->reference_count -= 1u;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    arena->released_reference_count += 1u;
    return SPARK_STATUS_OK;
}


static uint32_t SparkGlm52KvCacheProtectedBlockListContainsBlock(
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t physical_block_index)
{
    uint32_t protected_block_index;

    if (protected_physical_block_count != 0u &&
        protected_physical_block_indices == 0)
    {
        return 0u;
    }
    for (protected_block_index = 0u;
         protected_block_index < protected_physical_block_count;
         ++protected_block_index)
    {
        if (protected_physical_block_indices[protected_block_index] ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkGlm52KvCachePrefetchPlanContainsBlock(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    uint32_t physical_block_index)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return 0u;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        if (prefetch_plan->blocks[block_index].physical_block_index ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkGlm52KvCacheBlockIsProtectedFromResidentEviction(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t physical_block_index)
{
    return SparkGlm52KvCachePrefetchPlanContainsBlock(
            prefetch_plan,
            physical_block_index) ||
        SparkGlm52KvCacheProtectedBlockListContainsBlock(
            protected_physical_block_indices,
            protected_physical_block_count,
            physical_block_index);
}

static SparkGlm52KvCacheBlock *SparkGlm52KvCacheArenaSelectResidentEvictionVictim(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count)
{
    SparkGlm52KvCacheBlock *victim;
    uint32_t physical_block_index;

    victim = 0;
    for (physical_block_index = 0u;
         physical_block_index < arena->physical_block_count;
         ++physical_block_index)
    {
        SparkGlm52KvCacheBlock *block;

        block = &arena->blocks[physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            SparkGlm52KvCacheBlockIsProtectedFromResidentEviction(
                prefetch_plan,
                protected_physical_block_indices,
                protected_physical_block_count,
                physical_block_index))
        {
            continue;
        }
        if (victim == 0 ||
            block->reference_count < victim->reference_count ||
            (block->reference_count == victim->reference_count &&
             block->last_used_epoch < victim->last_used_epoch))
        {
            victim = block;
        }
    }
    return victim;
}

static SparkStatus SparkGlm52KvCacheArenaEvictResidentBlock(
    SparkGlm52KvCacheArena *arena,
    SparkGlm52KvCacheBlock *block)
{
    if (arena == 0 || block == 0 ||
        (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block->flags &= ~SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT;
    if (arena->resident_block_count != 0u)
    {
        arena->resident_block_count -= 1u;
    }
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    arena->resident_evicted_block_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheArenaTrimResidentBlocksWithPrefetchProtection(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((protected_physical_block_count != 0u &&
         protected_physical_block_indices == 0) ||
        target_resident_block_count > arena->resident_block_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (arena->resident_block_count > target_resident_block_count)
    {
        SparkGlm52KvCacheBlock *victim;

        victim = SparkGlm52KvCacheArenaSelectResidentEvictionVictim(
            arena,
            prefetch_plan,
            protected_physical_block_indices,
            protected_physical_block_count);
        if (victim == 0)
        {
            arena->resident_capacity_stall_count += 1u;
            if (evicted_block_count_out != 0)
            {
                *evicted_block_count_out = evicted_block_count;
            }
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkGlm52KvCacheArenaEvictResidentBlock(arena, victim);
        if (status != SPARK_STATUS_OK)
        {
            if (evicted_block_count_out != 0)
            {
                *evicted_block_count_out = evicted_block_count;
            }
            return status;
        }
        evicted_block_count += 1u;
    }

    if (evicted_block_count_out != 0)
    {
        *evicted_block_count_out = evicted_block_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaTrimResidentBlocks(
    SparkGlm52KvCacheArena *arena,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out)
{
    return SparkGlm52KvCacheArenaTrimResidentBlocksWithPrefetchProtection(
        arena,
        0,
        protected_physical_block_indices,
        protected_physical_block_count,
        target_resident_block_count,
        evicted_block_count_out);
}

static SparkStatus SparkGlm52KvCacheArenaMakeRoomForResidentBlocks(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t new_resident_block_count)
{
    uint32_t target_resident_block_count;

    if (new_resident_block_count > arena->resident_block_capacity)
    {
        arena->resident_capacity_stall_count += 1u;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (arena->resident_block_count + new_resident_block_count <=
        arena->resident_block_capacity)
    {
        return SPARK_STATUS_OK;
    }
    target_resident_block_count =
        arena->resident_block_capacity - new_resident_block_count;
    return SparkGlm52KvCacheArenaTrimResidentBlocksWithPrefetchProtection(
        arena,
        prefetch_plan,
        protected_physical_block_indices,
        protected_physical_block_count,
        target_resident_block_count,
        0);
}


SparkStatus SparkGlm52KvCacheArenaMarkBlockResident(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    uint32_t protected_physical_block_index;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
    {
        protected_physical_block_index = physical_block_index;
        status = SparkGlm52KvCacheArenaMakeRoomForResidentBlocks(
            arena,
            0,
            &protected_physical_block_index,
            1u,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        arena->resident_block_count += 1u;
    }
    block->flags |= SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaFreeBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (block->reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u &&
        arena->resident_block_count != 0u)
    {
        arena->resident_block_count -= 1u;
    }
    arena->epoch += 1u;
    block->flags = 0u;
    block->generation += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}


SparkStatus SparkGlm52KvCacheArenaMarkBlockNonResident(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index)
{
    SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
    {
        block->flags &= ~SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT;
        if (arena->resident_block_count != 0u)
        {
            arena->resident_block_count -= 1u;
        }
    }
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

static void SparkGlm52KvCachePrefetchPlanInitialize(
    SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    uint32_t lane_count,
    uint32_t requested_physical_block_count)
{
    memset(prefetch_plan, 0, sizeof(*prefetch_plan));
    prefetch_plan->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    prefetch_plan->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES;
    prefetch_plan->lane_count = lane_count;
    prefetch_plan->requested_physical_block_count =
        requested_physical_block_count;
}

static uint32_t SparkGlm52KvCachePrefetchPlanAlreadyContainsBlock(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    uint32_t physical_block_index)
{
    uint32_t block_index;

    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        if (prefetch_plan->blocks[block_index].physical_block_index ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static void SparkGlm52KvCachePrefetchSourceInitializeFromPhysicalBlock(
    SparkGlm52KvCachePrefetchSourceBlock *source_block,
    uint32_t physical_block_index,
    uint32_t include_value_payload)
{
    memset(source_block, 0, sizeof(*source_block));
    source_block->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    source_block->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block->physical_block_index = physical_block_index;
    source_block->flags = SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY;
    if (include_value_payload != 0u)
    {
        source_block->flags |= SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE;
    }
}

static SparkStatus SparkGlm52KvCacheValidatePrefetchSourceBlock(
    const SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchSourceBlock *source_block)
{
    if (source_block == 0 ||
        source_block->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        source_block->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES ||
        source_block->physical_block_index >= arena->physical_block_count ||
        (source_block->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) == 0u ||
        (source_block->flags &
            ~SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((source_block->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE) != 0u &&
        SparkGlm52KvCacheArenaHasValuePayload(arena) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (source_block->token_capacity != 0u &&
        source_block->token_count > source_block->token_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static void SparkGlm52KvCachePrefetchPlanAddSourceBlock(
    SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const SparkGlm52KvCacheBlock *block,
    const SparkGlm52KvCachePrefetchSourceBlock *source_block)
{
    SparkGlm52KvCachePrefetchBlock *prefetch_block;
    uint32_t lane_index;

    lane_index = prefetch_plan->prefetch_block_count %
        prefetch_plan->lane_count;
    prefetch_block = &prefetch_plan->blocks[
        prefetch_plan->prefetch_block_count];
    memset(prefetch_block, 0, sizeof(*prefetch_block));
    prefetch_block->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    prefetch_block->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES;
    prefetch_block->lane_index = lane_index;
    prefetch_block->physical_block_index = block->physical_block_index;
    prefetch_block->token_capacity = block->token_capacity;
    prefetch_block->first_token_index = source_block->first_token_index;
    prefetch_block->token_count = source_block->token_count;
    prefetch_block->flags = source_block->flags;
    prefetch_block->generation = block->generation;
    prefetch_block->parent_hash = source_block->parent_hash;
    prefetch_block->block_hash = source_block->block_hash;
    prefetch_block->content_hash = source_block->content_hash;
    prefetch_block->key_device_address = block->key_device_address;
    prefetch_block->value_device_address = block->value_device_address;
    prefetch_plan->prefetch_block_count += 1u;
    prefetch_plan->lane_block_counts[lane_index] += 1u;
}

SparkStatus SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
    const SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_count,
    uint32_t lane_count,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_plan == 0 || lane_count == 0u ||
        lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        (source_block_count != 0u && source_blocks == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkGlm52KvCachePrefetchPlanInitialize(
        prefetch_plan,
        lane_count,
        source_block_count);
    for (block_index = 0u; block_index < source_block_count; ++block_index)
    {
        const SparkGlm52KvCacheBlock *block;
        const SparkGlm52KvCachePrefetchSourceBlock *source_block;
        uint32_t physical_block_index;

        source_block = &source_blocks[block_index];
        status = SparkGlm52KvCacheValidatePrefetchSourceBlock(
            arena,
            source_block);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        physical_block_index = source_block->physical_block_index;
        if (SparkGlm52KvCachePrefetchPlanAlreadyContainsBlock(
                prefetch_plan,
                physical_block_index))
        {
            prefetch_plan->duplicate_block_count += 1u;
            continue;
        }

        block = &arena->blocks[physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
        {
            prefetch_plan->missing_block_count += 1u;
            return SPARK_STATUS_NOT_FOUND;
        }
        if (source_block->generation != 0u &&
            source_block->generation != block->generation)
        {
            return SPARK_STATUS_HASH_MISMATCH;
        }
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
        {
            prefetch_plan->resident_block_count += 1u;
            continue;
        }
        if (prefetch_plan->prefetch_block_count >=
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
        {
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        SparkGlm52KvCachePrefetchPlanAddSourceBlock(
            prefetch_plan,
            block,
            source_block);
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaBuildPrefetchPlan(
    const SparkGlm52KvCacheArena *arena,
    const uint32_t *physical_block_indices,
    uint32_t physical_block_count,
    uint32_t lane_count,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCachePrefetchSourceBlock source_blocks[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint32_t block_index;

    if (physical_block_count > SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (physical_block_count != 0u && physical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (block_index = 0u; block_index < physical_block_count; ++block_index)
    {
        SparkGlm52KvCachePrefetchSourceInitializeFromPhysicalBlock(
            &source_blocks[block_index],
            physical_block_indices[block_index],
            SparkGlm52KvCacheArenaHasValuePayload(arena));
    }
    return SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        arena,
        physical_block_count != 0u ? source_blocks : 0,
        physical_block_count,
        lane_count,
        prefetch_plan);
}

static SparkStatus SparkGlm52KvCacheValidatePrefetchPlan(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (prefetch_plan == 0 ||
        prefetch_plan->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        prefetch_plan->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES ||
        prefetch_plan->lane_count == 0u ||
        prefetch_plan->lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        prefetch_plan->prefetch_block_count >
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        const SparkGlm52KvCachePrefetchBlock *prefetch_block;

        prefetch_block = &prefetch_plan->blocks[block_index];
        if (prefetch_block->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
            prefetch_block->descriptor_bytes !=
                SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES ||
            prefetch_block->lane_index >= prefetch_plan->lane_count ||
            (prefetch_block->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) == 0u ||
            (prefetch_block->flags &
                ~SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52KvCachePrefetchPlanContainsBlockBeforeIndex(
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    uint32_t block_index_limit,
    uint32_t physical_block_index)
{
    uint32_t block_index;

    if (prefetch_plan == 0)
    {
        return 0u;
    }
    if (block_index_limit > prefetch_plan->prefetch_block_count)
    {
        block_index_limit = prefetch_plan->prefetch_block_count;
    }

    for (block_index = 0u; block_index < block_index_limit; ++block_index)
    {
        if (prefetch_plan->blocks[block_index].physical_block_index ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

SparkStatus SparkGlm52KvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count)
{
    uint32_t block_index;
    uint32_t new_resident_block_count;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkGlm52KvCacheValidatePrefetchPlan(prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (protected_physical_block_count != 0u &&
        protected_physical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    new_resident_block_count = 0u;
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        const SparkGlm52KvCachePrefetchBlock *prefetch_block;
        const SparkGlm52KvCacheBlock *block;

        prefetch_block = &prefetch_plan->blocks[block_index];
        if (prefetch_block->physical_block_index >= arena->physical_block_count)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        block = &arena->blocks[prefetch_block->physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
        {
            return SPARK_STATUS_NOT_FOUND;
        }
        if (block->generation != prefetch_block->generation)
        {
            return SPARK_STATUS_HASH_MISMATCH;
        }
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u &&
            !SparkGlm52KvCachePrefetchPlanContainsBlockBeforeIndex(
                prefetch_plan,
                block_index,
                prefetch_block->physical_block_index))
        {
            new_resident_block_count += 1u;
        }
    }

    status = SparkGlm52KvCacheArenaMakeRoomForResidentBlocks(
        arena,
        prefetch_plan,
        protected_physical_block_indices,
        protected_physical_block_count,
        new_resident_block_count);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        SparkGlm52KvCacheBlock *block;

        block = &arena->blocks[
            prefetch_plan->blocks[block_index].physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
        {
            arena->resident_block_count += 1u;
        }
        block->flags |= SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT;
        arena->epoch += 1u;
        block->last_used_epoch = arena->epoch;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    return SparkGlm52KvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
        arena,
        prefetch_plan,
        0,
        0u);
}

static uint32_t SparkGlm52KvCachePhysicalBlockIsProtected(
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t physical_block_index)
{
    uint32_t protected_index;

    if (protected_physical_block_indices == 0)
    {
        return 0u;
    }
    for (protected_index = 0u;
         protected_index < protected_physical_block_count;
         ++protected_index)
    {
        if (protected_physical_block_indices[protected_index] ==
            physical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkGlm52KvCacheSelectResidentEvictionVictim(
    const SparkGlm52KvCacheArena *arena,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count)
{
    uint32_t physical_block_index;
    uint32_t victim_physical_block_index;
    const SparkGlm52KvCacheBlock *victim_block;

    victim_physical_block_index = SPARK_GLM52_KV_CACHE_NO_BLOCK;
    victim_block = 0;
    for (physical_block_index = 0u;
         physical_block_index < arena->physical_block_count;
         ++physical_block_index)
    {
        const SparkGlm52KvCacheBlock *block;

        block = &arena->blocks[physical_block_index];
        if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            SparkGlm52KvCachePhysicalBlockIsProtected(
                protected_physical_block_indices,
                protected_physical_block_count,
                physical_block_index))
        {
            continue;
        }
        if (victim_block == 0 ||
            block->reference_count < victim_block->reference_count ||
            (block->reference_count == victim_block->reference_count &&
             block->last_used_epoch < victim_block->last_used_epoch))
        {
            victim_block = block;
            victim_physical_block_index = physical_block_index;
        }
    }
    return victim_physical_block_index;
}

SparkStatus SparkGlm52KvCacheArenaEvictResidentBlocksToLimit(
    SparkGlm52KvCacheArena *arena,
    uint32_t max_resident_block_count,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (protected_physical_block_count != 0u &&
        protected_physical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (arena->resident_block_count > max_resident_block_count)
    {
        uint32_t victim_physical_block_index;

        victim_physical_block_index = SparkGlm52KvCacheSelectResidentEvictionVictim(
            arena,
            protected_physical_block_indices,
            protected_physical_block_count);
        if (victim_physical_block_index == SPARK_GLM52_KV_CACHE_NO_BLOCK)
        {
            break;
        }
        status = SparkGlm52KvCacheArenaMarkBlockNonResident(
            arena,
            victim_physical_block_index);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        arena->resident_evicted_block_count += 1u;
        evicted_block_count += 1u;
    }

    if (evicted_block_count_out != 0)
    {
        *evicted_block_count_out = evicted_block_count;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaResolveBlock(
    const SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index,
    SparkGlm52KvCacheBlockView *block_view)
{
    const SparkGlm52KvCacheBlock *block;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (physical_block_index >= arena->physical_block_count || block_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[physical_block_index];
    if ((block->flags & SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(block_view, 0, sizeof(*block_view));
    block_view->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    block_view->descriptor_bytes = SPARK_GLM52_KV_CACHE_BLOCK_VIEW_DESCRIPTOR_BYTES;
    block_view->physical_block_index = physical_block_index;
    block_view->token_capacity = arena->block_token_count;
    block_view->layer_count = arena->layer_count;
    block_view->kv_head_count = arena->kv_head_count;
    block_view->head_dim = arena->head_dim;
    block_view->bytes_per_scalar = arena->bytes_per_scalar;
    block_view->generation = block->generation;
    block_view->key_block_stride_bytes = arena->key_block_stride_bytes;
    block_view->value_block_stride_bytes = arena->value_block_stride_bytes;
    block_view->key_device_address = block->key_device_address;
    block_view->value_device_address = block->value_device_address;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheArenaReset(
    SparkGlm52KvCacheArena *arena)
{
    uint32_t physical_block_index;
    SparkStatus status;

    status = SparkGlm52KvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (physical_block_index = 0u;
         physical_block_index < arena->physical_block_count;
         ++physical_block_index)
    {
        SparkGlm52KvCacheInitializeBlock(arena, physical_block_index);
    }
    arena->epoch = 0u;
    arena->allocated_block_count = 0u;
    arena->recycled_block_count = 0u;
    arena->resident_block_count = 0u;
    arena->retained_block_count = 0u;
    arena->released_reference_count = 0u;
    return SPARK_STATUS_OK;
}


static uint32_t SparkGlm52KvCacheAsyncPrefetchBackendSourceModeIsValid(
    uint32_t flags)
{
    uint32_t source_mode_count;

    source_mode_count = 0u;
    if ((flags & SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        source_mode_count += 1u;
    }
    if ((flags & SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE) != 0u)
    {
        source_mode_count += 1u;
    }
    return source_mode_count == 1u;
}

static uint32_t SparkGlm52KvCacheAsyncPrefetchBackendCopyFlagsAreValid(
    uint32_t flags)
{
    return (flags & SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS) != 0u;
}

static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendValidateConfiguration(
    const SparkGlm52KvCacheAsyncPrefetchBackendConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES ||
        (configuration->flags & ~SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS) != 0u ||
        !SparkGlm52KvCacheAsyncPrefetchBackendSourceModeIsValid(
            configuration->flags) ||
        !SparkGlm52KvCacheAsyncPrefetchBackendCopyFlagsAreValid(
            configuration->flags) ||
        configuration->lane_count == 0u ||
        configuration->lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        configuration->max_inflight_prefetch_count == 0u ||
        configuration->max_inflight_prefetch_count >
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY ||
        configuration->physical_block_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (configuration->key_source_stride_bytes == 0u ||
         configuration->key_transfer_bytes == 0u ||
         configuration->key_transfer_bytes > configuration->key_source_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (configuration->value_source_stride_bytes == 0u ||
         configuration->value_transfer_bytes == 0u ||
         configuration->value_transfer_bytes > configuration->value_source_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        if (((configuration->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
                configuration->key_source_base == 0) ||
            ((configuration->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
                configuration->value_source_base == 0))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if ((configuration->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE) != 0u)
    {
        if (((configuration->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
                configuration->key_file_descriptor < 0) ||
            ((configuration->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
                configuration->value_file_descriptor < 0))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if (configuration->source_entry_count != 0u &&
        configuration->source_entries == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendValidate(
    const SparkGlm52KvCacheAsyncPrefetchBackend *backend)
{
    if (backend == 0 ||
        backend->abi_version != SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        backend->descriptor_bytes != SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES ||
        (backend->flags & ~SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS) != 0u ||
        !SparkGlm52KvCacheAsyncPrefetchBackendSourceModeIsValid(backend->flags) ||
        !SparkGlm52KvCacheAsyncPrefetchBackendCopyFlagsAreValid(backend->flags) ||
        backend->lane_count == 0u ||
        backend->lane_count > SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        backend->max_inflight_prefetch_count == 0u ||
        backend->max_inflight_prefetch_count >
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY ||
        backend->physical_block_count == 0u ||
        backend->blocks_per_poll == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (backend->key_source_stride_bytes == 0u || backend->key_transfer_bytes == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (backend->value_source_stride_bytes == 0u || backend->value_transfer_bytes == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendInitialize(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCacheAsyncPrefetchBackendConfiguration *configuration)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkGlm52KvCacheAsyncPrefetchBackendValidateConfiguration(
        configuration);
    if (backend == 0 || status != SPARK_STATUS_OK)
    {
        return backend == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    memset(backend, 0, sizeof(*backend));
    backend->abi_version = SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    backend->descriptor_bytes = SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES;
    backend->flags = configuration->flags;
    backend->lane_count = configuration->lane_count;
    backend->max_inflight_prefetch_count =
        configuration->max_inflight_prefetch_count;
    backend->physical_block_count = configuration->physical_block_count;
    backend->source_entry_count = configuration->source_entry_count;
    backend->blocks_per_poll = configuration->blocks_per_poll != 0u
        ? configuration->blocks_per_poll
        : configuration->lane_count;
    backend->key_source_stride_bytes = configuration->key_source_stride_bytes;
    backend->value_source_stride_bytes = configuration->value_source_stride_bytes;
    backend->key_transfer_bytes = configuration->key_transfer_bytes;
    backend->value_transfer_bytes = configuration->value_transfer_bytes;
    backend->key_source_base = configuration->key_source_base;
    backend->value_source_base = configuration->value_source_base;
    backend->source_entries = configuration->source_entries;
    backend->key_file_descriptor = configuration->key_file_descriptor;
    backend->value_file_descriptor = configuration->value_file_descriptor;

    for (request_index = 0u;
         request_index < SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY;
         ++request_index)
    {
        backend->requests[request_index].abi_version =
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
        backend->requests[request_index].descriptor_bytes =
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
    }
    return SPARK_STATUS_OK;
}

static SparkGlm52KvCacheAsyncPrefetchRequest *
SparkGlm52KvCacheAsyncPrefetchBackendFindRequest(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    uint64_t prefetch_id)
{
    uint32_t request_index;

    for (request_index = 0u;
         request_index < backend->max_inflight_prefetch_count;
         ++request_index)
    {
        if (backend->requests[request_index].active != 0u &&
            backend->requests[request_index].prefetch_id == prefetch_id)
        {
            return &backend->requests[request_index];
        }
    }
    return 0;
}

static SparkGlm52KvCacheAsyncPrefetchRequest *
SparkGlm52KvCacheAsyncPrefetchBackendFindFreeRequest(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend)
{
    uint32_t request_index;

    for (request_index = 0u;
         request_index < backend->max_inflight_prefetch_count;
         ++request_index)
    {
        if (backend->requests[request_index].active == 0u)
        {
            return &backend->requests[request_index];
        }
    }
    return 0;
}

static void SparkGlm52KvCacheAsyncPrefetchBackendClearRequest(
    SparkGlm52KvCacheAsyncPrefetchRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    request->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
}

static uint32_t SparkGlm52KvCachePrefetchBackendSourceEntryMatches(
    const SparkGlm52KvCachePrefetchBackendSourceEntry *source_entry,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block)
{
    if (source_entry->abi_version !=
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        source_entry->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_SOURCE_ENTRY_DESCRIPTOR_BYTES)
    {
        return 0u;
    }
    if (source_entry->content_hash != 0u &&
        source_entry->content_hash == prefetch_block->content_hash)
    {
        return 1u;
    }
    if (source_entry->block_hash != 0u &&
        source_entry->block_hash == prefetch_block->block_hash &&
        (source_entry->parent_hash == 0u ||
         source_entry->parent_hash == prefetch_block->parent_hash))
    {
        return 1u;
    }
    if (source_entry->physical_block_index != SPARK_GLM52_KV_CACHE_NO_BLOCK &&
        source_entry->physical_block_index == prefetch_block->physical_block_index)
    {
        return 1u;
    }
    return 0u;
}

static const SparkGlm52KvCachePrefetchBackendSourceEntry *
SparkGlm52KvCacheAsyncPrefetchBackendFindSourceEntry(
    const SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block)
{
    uint32_t source_index;

    if (backend->source_entries == 0 || backend->source_entry_count == 0u)
    {
        return 0;
    }
    for (source_index = 0u;
         source_index < backend->source_entry_count;
         ++source_index)
    {
        if (SparkGlm52KvCachePrefetchBackendSourceEntryMatches(
                &backend->source_entries[source_index],
                prefetch_block))
        {
            return &backend->source_entries[source_index];
        }
    }
    return 0;
}

static uint64_t SparkGlm52KvCacheAsyncPrefetchBackendDefaultSourceOffset(
    uint32_t physical_block_index,
    uint64_t source_stride_bytes)
{
    return (uint64_t)physical_block_index * source_stride_bytes;
}

static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendResolveSource(
    const SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block,
    uint32_t key_source,
    const void **memory_source_out,
    uint64_t *file_offset_out)
{
    const SparkGlm52KvCachePrefetchBackendSourceEntry *source_entry;
    uint64_t offset_bytes;
    uint64_t stride_bytes;
    const void *base;

    if (memory_source_out == 0 || file_offset_out == 0 ||
        prefetch_block->physical_block_index >= backend->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    source_entry = SparkGlm52KvCacheAsyncPrefetchBackendFindSourceEntry(
        backend,
        prefetch_block);
    stride_bytes = key_source != 0u
        ? backend->key_source_stride_bytes
        : backend->value_source_stride_bytes;
    offset_bytes = SparkGlm52KvCacheAsyncPrefetchBackendDefaultSourceOffset(
        prefetch_block->physical_block_index,
        stride_bytes);
    base = key_source != 0u
        ? backend->key_source_base
        : backend->value_source_base;

    if (source_entry != 0)
    {
        offset_bytes = key_source != 0u
            ? source_entry->key_source_offset_bytes
            : source_entry->value_source_offset_bytes;
        if ((backend->flags &
                SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
        {
            const void *entry_address;

            entry_address = key_source != 0u
                ? source_entry->key_source_address
                : source_entry->value_source_address;
            if (entry_address != 0)
            {
                *memory_source_out = entry_address;
                *file_offset_out = offset_bytes;
                return SPARK_STATUS_OK;
            }
        }
    }

    *file_offset_out = offset_bytes;
    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        if (base == 0)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
        *memory_source_out = (const void *)((const unsigned char *)base + offset_bytes);
    }
    else
    {
        *memory_source_out = 0;
    }
    return SPARK_STATUS_OK;
}

#if defined(__unix__) || defined(__APPLE__)
static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendReadExact(
    int32_t file_descriptor,
    uint64_t offset_bytes,
    void *destination,
    uint64_t byte_count)
{
    unsigned char *cursor;
    uint64_t remaining_bytes;

    if (file_descriptor < 0 || destination == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (lseek(file_descriptor, (off_t)offset_bytes, SEEK_SET) == (off_t)-1)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    cursor = (unsigned char *)destination;
    remaining_bytes = byte_count;
    while (remaining_bytes != 0u)
    {
        ssize_t read_bytes;

        read_bytes = read(file_descriptor, cursor, (size_t)remaining_bytes);
        if (read_bytes <= 0)
        {
            return SPARK_STATUS_IO_ERROR;
        }
        cursor += (uint64_t)read_bytes;
        remaining_bytes -= (uint64_t)read_bytes;
    }
    return SPARK_STATUS_OK;
}
#else
static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendReadExact(
    int32_t file_descriptor,
    uint64_t offset_bytes,
    void *destination,
    uint64_t byte_count)
{
    (void)file_descriptor;
    (void)offset_bytes;
    (void)destination;
    (void)byte_count;
    return SPARK_STATUS_IO_ERROR;
}
#endif

static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendCopyOnePayload(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block,
    uint32_t key_payload)
{
    const void *memory_source;
    uint64_t file_offset_bytes;
    uint64_t transfer_bytes;
    uintptr_t destination_address;
    SparkStatus status;

    transfer_bytes = key_payload != 0u
        ? backend->key_transfer_bytes
        : backend->value_transfer_bytes;
    destination_address = key_payload != 0u
        ? prefetch_block->key_device_address
        : prefetch_block->value_device_address;
    if (destination_address == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkGlm52KvCacheAsyncPrefetchBackendResolveSource(
        backend,
        prefetch_block,
        key_payload,
        &memory_source,
        &file_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        memcpy((void *)destination_address, memory_source, (size_t)transfer_bytes);
    }
    else
    {
        int32_t file_descriptor;

        file_descriptor = key_payload != 0u
            ? backend->key_file_descriptor
            : backend->value_file_descriptor;
        status = SparkGlm52KvCacheAsyncPrefetchBackendReadExact(
            file_descriptor,
            file_offset_bytes,
            (void *)destination_address,
            transfer_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }

    if (key_payload != 0u)
    {
        backend->copied_key_block_count += 1u;
    }
    else
    {
        backend->copied_value_block_count += 1u;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendCopyOneBlock(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCachePrefetchBlock *prefetch_block)
{
    SparkStatus status;

    if (prefetch_block == 0 ||
        prefetch_block->abi_version != SPARK_GLM52_KV_CACHE_ABI_VERSION ||
        prefetch_block->descriptor_bytes !=
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES ||
        prefetch_block->physical_block_index >= backend->physical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (prefetch_block->flags & SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY) != 0u)
    {
        status = SparkGlm52KvCacheAsyncPrefetchBackendCopyOnePayload(
            backend,
            prefetch_block,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if ((backend->flags &
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (prefetch_block->flags & SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE) != 0u)
    {
        status = SparkGlm52KvCacheAsyncPrefetchBackendCopyOnePayload(
            backend,
            prefetch_block,
            0u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendStart(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCacheAsyncPrefetchBackend *backend;
    SparkGlm52KvCacheAsyncPrefetchRequest *request;
    SparkStatus status;

    backend = (SparkGlm52KvCacheAsyncPrefetchBackend *)context;
    status = SparkGlm52KvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_id == 0u ||
        SparkGlm52KvCacheValidatePrefetchPlan(prefetch_plan) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkGlm52KvCacheAsyncPrefetchBackendFindRequest(
            backend,
            prefetch_id) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    request = SparkGlm52KvCacheAsyncPrefetchBackendFindFreeRequest(backend);
    if (request == 0)
    {
        return SPARK_STATUS_BUSY;
    }

    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    request->descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
    request->active = 1u;
    request->prefetch_id = prefetch_id;
    request->terminal_status = SPARK_STATUS_BUSY;
    request->prefetch_plan = *prefetch_plan;
    backend->started_prefetch_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendPoll(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCacheAsyncPrefetchBackend *backend;
    SparkGlm52KvCacheAsyncPrefetchRequest *request;
    SparkStatus status;
    uint32_t block_budget;
    uint32_t processed_block_count;

    backend = (SparkGlm52KvCacheAsyncPrefetchBackend *)context;
    status = SparkGlm52KvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_id == 0u || prefetch_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    request = SparkGlm52KvCacheAsyncPrefetchBackendFindRequest(
        backend,
        prefetch_id);
    if (request == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (request->prefetch_plan.prefetch_block_count !=
            prefetch_plan->prefetch_block_count ||
        request->prefetch_plan.lane_count != prefetch_plan->lane_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block_budget = backend->blocks_per_poll;
    processed_block_count = 0u;
    while (request->completed_block_count <
            request->prefetch_plan.prefetch_block_count &&
        processed_block_count < block_budget)
    {
        status = SparkGlm52KvCacheAsyncPrefetchBackendCopyOneBlock(
            backend,
            &request->prefetch_plan.blocks[request->completed_block_count]);
        if (status != SPARK_STATUS_OK)
        {
            backend->failed_prefetch_count += 1u;
            SparkGlm52KvCacheAsyncPrefetchBackendClearRequest(request);
            return status;
        }
        request->completed_block_count += 1u;
        processed_block_count += 1u;
    }

    if (request->completed_block_count < request->prefetch_plan.prefetch_block_count)
    {
        backend->busy_poll_count += 1u;
        return SPARK_STATUS_BUSY;
    }

    backend->completed_prefetch_count += 1u;
    SparkGlm52KvCacheAsyncPrefetchBackendClearRequest(request);
    return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendSubmitSynchronous(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan)
{
    SparkGlm52KvCacheAsyncPrefetchBackend *backend;
    uint64_t prefetch_id;
    SparkStatus status;

    backend = (SparkGlm52KvCacheAsyncPrefetchBackend *)context;
    status = SparkGlm52KvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    prefetch_id = backend->started_prefetch_count + 1u;
    status = SparkGlm52KvCacheAsyncPrefetchBackendStart(
        backend,
        prefetch_id,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    do
    {
        status = SparkGlm52KvCacheAsyncPrefetchBackendPoll(
            backend,
            prefetch_id,
            prefetch_plan);
    } while (status == SPARK_STATUS_BUSY);
    return status;
}
