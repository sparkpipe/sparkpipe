#include "sparkpipe/spark_kv_cache.h"

#include <string.h>
#if defined(__unix__) || defined(__APPLE__)
#include <sys/types.h>
#include <unistd.h>
#endif

static uint64_t SparkKvCacheMulU64(
    uint64_t left,
    uint64_t right)
{
    return left * right;
}

static SparkStatus SparkKvCacheCheckedMulU64(
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

static SparkStatus SparkKvCacheCheckedAddU64(
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

static SparkStatus SparkKvCacheAlignUpU64(
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

uint32_t SparkKvCacheIndexSourceLayer(uint32_t layer_index)
{
    return(layer_index < SPARK_KV_CACHE_MAX_LAYER_COUNT ? layer_index : UINT32_MAX);
}

static SparkStatus SparkKvCacheCalculateAttentionBytesPerTokenLayer(
    const SparkKvCacheCapacityRequest *request,
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
        case SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE:
            element_count =
                (uint64_t)request->compressed_dimension +
                (uint64_t)request->position_dimension +
                ((uint64_t)request->head_count *
                 (uint64_t)request->query_key_head_dimension) +
                ((uint64_t)request->head_count *
                 (uint64_t)request->value_head_dimension);
            *bytes_per_token_per_layer_out =
                element_count * (uint64_t)request->bytes_per_scalar;
            return SPARK_STATUS_OK;

        case SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE:
            element_count =
                (uint64_t)request->compressed_dimension +
                (uint64_t)request->position_dimension;
            *bytes_per_token_per_layer_out =
                element_count * (uint64_t)request->bytes_per_scalar;
            return SPARK_STATUS_OK;

        case SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3:
            if (request->fp8_scale_block_size == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            element_count =
                (uint64_t)request->compressed_dimension +
                (uint64_t)request->position_dimension;
            scale_count = SparkCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out =
                element_count + (scale_count * sizeof(float));
            return SPARK_STATUS_OK;

        case SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3:
            if (request->fp8_scale_block_size == 0u)
            {
                return SPARK_STATUS_INVALID_ARGUMENT;
            }
            element_count =
                (uint64_t)request->compressed_dimension +
                (uint64_t)request->position_dimension;
            scale_count = SparkCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out =
                element_count + (scale_count * sizeof(float));
            element_count =
                (uint64_t)request->head_count *
                (uint64_t)request->query_key_head_dimension;
            scale_count = SparkCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out +=
                element_count + (scale_count * sizeof(float));
            element_count =
                (uint64_t)request->head_count *
                (uint64_t)request->value_head_dimension;
            scale_count = SparkCeilDivU64(
                element_count,
                (uint64_t)request->fp8_scale_block_size);
            *bytes_per_token_per_layer_out +=
                element_count + (scale_count * sizeof(float));
            return SPARK_STATUS_OK;

        default:
            return SPARK_STATUS_INVALID_ARGUMENT;
    }
}

SparkStatus SparkKvCacheCalculateJitStageBudget(
    const SparkKvJitStageBudgetRequest *request,
    SparkKvJitStageBudget *budget)
{
    uint64_t attention_bytes_per_token_per_layer;
    uint64_t index_key_bytes_per_token_per_layer;
    uint64_t summary_bytes_per_index_layer_block;
    uint64_t resident_token_bytes;
    uint64_t payload_token_bytes;
    uint64_t record_unaligned_bytes;
    uint64_t active_token_capacity;
    uint64_t backing_token_capacity;
    uint64_t compact_selected_token_count;
    uint32_t local_index_key_layer_count;
    SparkStatus status;

    if (request == 0 || budget == 0 ||
        request->abi_version != SPARK_KV_JIT_STAGE_BUDGET_ABI_VERSION ||
        request->descriptor_bytes !=
            SPARK_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES ||
        request->layer_count == 0u ||
        request->first_layer_index >= SPARK_KV_CACHE_MAX_LAYER_COUNT ||
        request->layer_count >
            SPARK_KV_CACHE_MAX_LAYER_COUNT - request->first_layer_index ||
        request->physical_pool_token_capacity == 0u ||
        request->backing_block_capacity == 0u ||
        request->active_sequence_count == 0u ||
        request->backing_request_count == 0u ||
        request->selected_token_count == 0u ||
        request->block_token_count == 0u ||
        (request->attention_cache_layout !=
            SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE &&
         request->attention_cache_layout !=
            SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE &&
         request->attention_cache_layout !=
            SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3 &&
         request->attention_cache_layout !=
            SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3) ||
        ((request->attention_cache_layout ==
            SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3 ||
          request->attention_cache_layout ==
            SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3) &&
         (request->fp8_scale_block_size == 0u ||
          (request->fp8_scale_block_size & 15u) != 0u)) ||
        request->physical_pool_token_capacity % request->block_token_count != 0u ||
        request->backing_block_capacity <
            request->physical_pool_token_capacity / request->block_token_count ||
        request->include_auxiliary_layer > 1u ||
        request->record_alignment_bytes <
            SPARK_KV_JIT_DEFAULT_RECORD_ALIGNMENT ||
        (request->record_alignment_bytes &
            (request->record_alignment_bytes - 1u)) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    local_index_key_layer_count = 0u;
    // A MODEL WITHOUT AN INDEX CACHE ASKS FOR ZERO INDEX LAYERS. The
    // request says how many layers carry index keys; scanning a model layer
    // schedule here priced every other model's cache with model's calendar.
    local_index_key_layer_count = request->index_key_layer_count;
    {
        // ONE FORMULA SET, SHARED WITH THE ESTIMATOR. The stage budget
        // builds the estimator's request from its own geometry fields and
        // asks the same helper; four duplicated layout formulas retired.
        SparkKvCacheCapacityRequest geometry;
        memset(&geometry,0,sizeof(geometry));
        geometry.layout = request->attention_cache_layout;
        geometry.fp8_scale_block_size = request->fp8_scale_block_size;
        geometry.head_count = request->head_count;
        geometry.query_key_head_dimension = request->query_key_head_dimension;
        geometry.value_head_dimension = request->value_head_dimension;
        geometry.compressed_dimension = request->compressed_dimension;
        geometry.position_dimension = request->position_dimension;
        geometry.bytes_per_scalar = request->bytes_per_scalar;
        if (SparkKvCacheCalculateAttentionBytesPerTokenLayer(&geometry,
            &attention_bytes_per_token_per_layer) != SPARK_STATUS_OK)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    index_key_bytes_per_token_per_layer =
        (uint64_t)request->index_key_dimension *
        request->index_key_bytes_per_scalar;
    summary_bytes_per_index_layer_block =
        (2u * index_key_bytes_per_token_per_layer) + sizeof(uint8_t);

    memset(budget, 0, sizeof(*budget));
    budget->abi_version = SPARK_KV_JIT_STAGE_BUDGET_ABI_VERSION;
    budget->descriptor_bytes = SPARK_KV_JIT_STAGE_BUDGET_DESCRIPTOR_BYTES;
    budget->first_layer_index = request->first_layer_index;
    budget->layer_count = request->layer_count;
    budget->local_index_key_layer_count = local_index_key_layer_count;
    budget->include_auxiliary_layer = request->include_auxiliary_layer;
    budget->logical_block_capacity =
        request->physical_pool_token_capacity / request->block_token_count;
    budget->backing_block_capacity = request->backing_block_capacity;
    budget->attention_bytes_per_token =
        (uint64_t)request->layer_count * attention_bytes_per_token_per_layer;
    budget->index_key_bytes_per_token =
        (uint64_t)local_index_key_layer_count * index_key_bytes_per_token_per_layer;
    budget->auxiliary_bytes_per_token = request->include_auxiliary_layer != 0u
        ? attention_bytes_per_token_per_layer : 0u;
    status = SparkKvCacheCheckedAddU64(
        budget->attention_bytes_per_token,
        budget->index_key_bytes_per_token,
        &resident_token_bytes);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedAddU64(
            resident_token_bytes,
            budget->auxiliary_bytes_per_token,
            &budget->resident_bytes_per_token);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            budget->logical_block_capacity,
            (uint64_t)local_index_key_layer_count *
                summary_bytes_per_index_layer_block,
            &budget->resident_summary_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            request->physical_pool_token_capacity,
            budget->resident_bytes_per_token,
            &resident_token_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedAddU64(
            resident_token_bytes,
            budget->resident_summary_bytes,
            &budget->resident_pool_bytes);
    }
    payload_token_bytes = 0u;
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            request->block_token_count,
            budget->resident_bytes_per_token,
            &payload_token_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedAddU64(
            payload_token_bytes,
            (uint64_t)local_index_key_layer_count *
                summary_bytes_per_index_layer_block,
            &budget->nvme_payload_bytes_per_block);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedAddU64(
            request->record_alignment_bytes,
            budget->nvme_payload_bytes_per_block,
            &record_unaligned_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheAlignUpU64(
            record_unaligned_bytes,
            request->record_alignment_bytes,
            &budget->nvme_record_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            request->backing_block_capacity,
            budget->nvme_record_bytes,
            &budget->nvme_capacity_bytes);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            request->active_sequence_count,
            request->selected_token_count,
            &compact_selected_token_count);
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkKvCacheCheckedMulU64(
            compact_selected_token_count,
            budget->attention_bytes_per_token + budget->auxiliary_bytes_per_token,
            &budget->selected_working_set_bytes);
    }
    if (status != SPARK_STATUS_OK)
    {
        memset(budget, 0, sizeof(*budget));
        return status;
    }

    active_token_capacity = request->physical_pool_token_capacity;
    status = SparkKvCacheCheckedMulU64(
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

SparkStatus SparkKvCacheEstimateCapacity(
    const SparkKvCacheCapacityRequest *request,
    SparkKvCacheCapacityEstimate *estimate)
{
    uint64_t attention_bytes_per_token_per_layer;
    uint64_t index_key_bytes_per_token;
    uint64_t block_count_per_context;
    uint64_t bytes_per_block_per_layer;
    uint64_t bytes_per_context_per_rank;
    SparkStatus status;

    if (request == 0 || estimate == 0 ||
        request->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        request->descriptor_bytes !=
            SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES ||
        request->context_token_count == 0u ||
        request->block_token_count == 0u ||
        request->layer_count == 0u ||
        request->bytes_per_scalar == 0u ||
        request->cache_bytes_per_rank == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    // Layout-conditional shape requirements: the compressed layouts
    // carry a latent and a rope slice; the full layouts carry heads.
    // Compressed layouts carry two shared slices while full layouts carry
    // per-head key and value slices.
    if (request->layout == SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE ||
        request->layout == SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3)
    {
        if (request->compressed_dimension == 0u ||
            request->position_dimension == 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    else if (request->head_count == 0u ||
        request->query_key_head_dimension == 0u ||
        request->value_head_dimension == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkKvCacheCalculateAttentionBytesPerTokenLayer(
        request,
        &attention_bytes_per_token_per_layer);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    index_key_bytes_per_token = 0u;
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
        index_key_bytes_per_token =
            (uint64_t)request->index_key_layer_count *
            (uint64_t)request->index_key_dimension *
            (uint64_t)request->index_key_bytes_per_scalar;
    }

    block_count_per_context = SparkCeilDivU64(
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
         index_key_bytes_per_token);
    if (bytes_per_context_per_rank == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    memset(estimate, 0, sizeof(*estimate));
    estimate->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    estimate->descriptor_bytes =
        SPARK_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES;
    estimate->layout = request->layout;
    estimate->context_token_count = request->context_token_count;
    estimate->block_token_count = request->block_token_count;
    estimate->layer_count = request->layer_count;
    estimate->block_count_per_context = (uint32_t)block_count_per_context;
    estimate->contexts_per_rank =
        (uint32_t)(request->cache_bytes_per_rank / bytes_per_context_per_rank);
    estimate->attention_bytes_per_token_per_layer =
        attention_bytes_per_token_per_layer;
    estimate->index_key_bytes_per_token = index_key_bytes_per_token;
    estimate->bytes_per_block_per_layer = bytes_per_block_per_layer;
    estimate->bytes_per_context_per_rank = bytes_per_context_per_rank;
    estimate->unused_cache_bytes_per_rank =
        request->cache_bytes_per_rank -
        ((uint64_t)estimate->contexts_per_rank * bytes_per_context_per_rank);
    return SPARK_STATUS_OK;
}

static uint64_t SparkKvCacheDefaultBlockStrideBytes(
    const SparkKvCacheConfiguration *configuration)
{
    uint64_t stride_bytes;

    stride_bytes = SparkKvCacheMulU64(
        configuration->block_token_count,
        configuration->layer_count);
    stride_bytes = SparkKvCacheMulU64(
        stride_bytes,
        configuration->kv_head_count);
    stride_bytes = SparkKvCacheMulU64(
        stride_bytes,
        configuration->head_dim);
    stride_bytes = SparkKvCacheMulU64(
        stride_bytes,
        configuration->bytes_per_scalar);
    return stride_bytes;
}

static uint32_t SparkKvCacheConfigurationHasValuePayload(
    const SparkKvCacheConfiguration *configuration)
{
    return configuration != 0 &&
        (configuration->value_device_base != 0 ||
         configuration->value_block_stride_bytes != 0u);
}

static uint32_t SparkKvCacheArenaHasValuePayload(
    const SparkKvCacheArena *arena)
{
    return arena != 0 &&
        (arena->value_device_base != 0u ||
         arena->value_block_stride_bytes != 0u);
}

static uint32_t SparkKvCacheConfigurationIsValid(
    const SparkKvCacheConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES ||
        configuration->logical_block_count == 0u ||
        configuration->block_token_count == 0u ||
        configuration->block_token_count > SPARK_KV_CACHE_MAX_BLOCK_TOKENS ||
        configuration->resident_block_capacity > configuration->logical_block_count ||
        configuration->layer_count == 0u ||
        configuration->layer_count > SPARK_KV_CACHE_MAX_LAYER_COUNT ||
        configuration->kv_head_count == 0u ||
        configuration->head_dim == 0u ||
        configuration->bytes_per_scalar == 0u ||
        configuration->key_device_base == 0 ||
        configuration->blocks == 0 ||
        configuration->resident_slot_logical_block_indices == 0 ||
        (configuration->evict_function == 0 &&
         configuration->evict_context != 0))
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

static SparkStatus SparkKvCacheArenaValidate(
    const SparkKvCacheArena *arena)
{
	uint32_t unassigned_resident_block_count;

	unassigned_resident_block_count = arena != 0 ?
		atomic_load(&arena->unassigned_resident_block_count) : 0u;
    if (arena == 0 ||
        arena->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        arena->descriptor_bytes != SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES ||
        arena->logical_block_count == 0u ||
        arena->block_token_count == 0u ||
        arena->block_token_count > SPARK_KV_CACHE_MAX_BLOCK_TOKENS ||
        arena->resident_block_capacity == 0u ||
        arena->resident_block_capacity > arena->logical_block_count ||
        arena->layer_count == 0u ||
        arena->layer_count > SPARK_KV_CACHE_MAX_LAYER_COUNT ||
        arena->kv_head_count == 0u ||
        arena->head_dim == 0u ||
        arena->bytes_per_scalar == 0u ||
        arena->key_device_base == 0u ||
        arena->key_block_stride_bytes == 0u ||
        arena->blocks == 0 ||
        arena->resident_slot_logical_block_indices == 0 ||
        (uint64_t)arena->resident_block_count + arena->reserved_block_count +
            unassigned_resident_block_count >
            arena->resident_block_capacity)
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

static void SparkKvCacheInitializeBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;

    block = &arena->blocks[logical_block_index];
    memset(block, 0, sizeof(*block));
    block->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    block->descriptor_bytes = SPARK_KV_CACHE_BLOCK_DESCRIPTOR_BYTES;
    block->logical_block_index = logical_block_index;
    block->token_capacity = arena->block_token_count;
    block->resident_slot_index = SPARK_KV_CACHE_NO_RESIDENT_SLOT;
    block->free_next = logical_block_index + 1u < arena->logical_block_count
        ? logical_block_index + 1u : SPARK_KV_CACHE_NO_BLOCK;
}

SparkStatus SparkKvCacheArenaInitialize(
    SparkKvCacheArena *arena,
    const SparkKvCacheConfiguration *configuration)
{
    uint64_t default_stride_bytes;
    uint32_t logical_block_index,resident_slot_index;

    if (arena == 0 || !SparkKvCacheConfigurationIsValid(configuration))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    default_stride_bytes = SparkKvCacheDefaultBlockStrideBytes(configuration);
    memset(arena, 0, sizeof(*arena));
    arena->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    arena->descriptor_bytes = SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES;
	atomic_init(&arena->unassigned_resident_block_count,0u);
    arena->logical_block_count = configuration->logical_block_count;
    arena->block_token_count = configuration->block_token_count;
    arena->resident_block_capacity = configuration->resident_block_capacity != 0u
        ? configuration->resident_block_capacity
        : configuration->logical_block_count;
    arena->layer_count = configuration->layer_count;
    arena->kv_head_count = configuration->kv_head_count;
    arena->head_dim = configuration->head_dim;
    arena->bytes_per_scalar = configuration->bytes_per_scalar;
    arena->key_block_stride_bytes = configuration->key_block_stride_bytes != 0u
        ? configuration->key_block_stride_bytes
        : default_stride_bytes;
    arena->value_block_stride_bytes =
        SparkKvCacheConfigurationHasValuePayload(configuration) != 0u
        ? configuration->value_block_stride_bytes != 0u
            ? configuration->value_block_stride_bytes
            : default_stride_bytes
        : 0u;
    arena->key_device_base = (uintptr_t)configuration->key_device_base;
    arena->value_device_base =
        SparkKvCacheConfigurationHasValuePayload(configuration) != 0u
        ? (uintptr_t)configuration->value_device_base
        : 0u;
    arena->blocks = configuration->blocks;
    arena->resident_slot_logical_block_indices =
        configuration->resident_slot_logical_block_indices;
    arena->evict_function = configuration->evict_function;
    arena->evict_context = configuration->evict_context;
    arena->free_logical_block_head = 0u;

    for (logical_block_index = 0u;
         logical_block_index < arena->logical_block_count;
         ++logical_block_index)
    {
        SparkKvCacheInitializeBlock(arena, logical_block_index);
    }
    for (resident_slot_index = 0u;
         resident_slot_index < arena->resident_block_capacity;
         ++resident_slot_index)
    {
        arena->resident_slot_logical_block_indices[resident_slot_index] =
            SPARK_KV_CACHE_NO_BLOCK;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaReserveUnassignedResidentBlocks(
    SparkKvCacheArena *arena,
    uint32_t block_count)
{
	uint32_t current,next,target;
	SparkStatus status;
	status = SparkKvCacheArenaValidate(arena);
	if ( status != SPARK_STATUS_OK || block_count == 0u )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	current = atomic_load(&arena->unassigned_resident_block_count);
	for (;;)
	{
		if ( current > arena->resident_block_capacity ||
			block_count > arena->resident_block_capacity - current ||
			(uint64_t)arena->reserved_block_count + current + block_count >
				arena->resident_block_capacity )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		target = arena->resident_block_capacity - arena->reserved_block_count -
			current - block_count;
		if ( arena->resident_block_count > target )
		{
			status = SparkKvCacheArenaTrimResidentBlocks(arena,0,0u,target,0);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		next = current + block_count;
		if ( atomic_compare_exchange_weak(
			&arena->unassigned_resident_block_count,&current,next) )
			return(SPARK_STATUS_OK);
	}
}

static SparkStatus SparkKvCacheArenaRemoveUnassignedResidentBlocks(
	SparkKvCacheArena *arena,
	uint32_t block_count)
{
	uint32_t current,next;
	SparkStatus status;

	status = SparkKvCacheArenaValidate(arena);
	if ( status != SPARK_STATUS_OK || block_count == 0u )
		return(status != SPARK_STATUS_OK ? status : SPARK_STATUS_INVALID_ARGUMENT);
	current = atomic_load(&arena->unassigned_resident_block_count);
	for (;;)
	{
		if ( current < block_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		next = current - block_count;
		if ( atomic_compare_exchange_weak(
			&arena->unassigned_resident_block_count,&current,next) )
			return(SPARK_STATUS_OK);
	}
}

SparkStatus SparkKvCacheArenaConsumeUnassignedResidentBlocks(
    SparkKvCacheArena *arena,
    uint32_t block_count)
{
	return(SparkKvCacheArenaRemoveUnassignedResidentBlocks(arena,block_count));
}

SparkStatus SparkKvCacheArenaReleaseUnassignedResidentBlocks(
    SparkKvCacheArena *arena,
    uint32_t block_count)
{
	return(SparkKvCacheArenaRemoveUnassignedResidentBlocks(arena,block_count));
}

uint32_t SparkKvCacheArenaUnassignedResidentBlockCount(
    const SparkKvCacheArena *arena)
{
	if ( arena == 0 || arena->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
		arena->descriptor_bytes != SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES )
		return(0u);
	return(atomic_load(&arena->unassigned_resident_block_count));
}

SparkStatus SparkKvCacheArenaAcquireBlock(
    SparkKvCacheArena *arena,
    uint32_t *logical_block_index_out)
{
    SparkKvCacheBlock *block;
    uint32_t logical_block_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK || logical_block_index_out == 0)
    {
        return status == SPARK_STATUS_OK ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    logical_block_index = arena->free_logical_block_head;
    if (logical_block_index == SPARK_KV_CACHE_NO_BLOCK)
    {
        *logical_block_index_out = SPARK_KV_CACHE_NO_BLOCK;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    arena->free_logical_block_head = block->free_next;
    arena->epoch += 1u;
    block->flags = SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED;
    block->reference_count = 0u;
    block->free_next = SPARK_KV_CACHE_NO_BLOCK;
    block->generation += 1u;
    block->last_used_epoch = arena->epoch;
    arena->allocated_block_count += 1u;
    *logical_block_index_out = logical_block_index;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaRecycleBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (block->reference_count != 0u ||
        block->residency_reference_count != 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }

    arena->epoch += 1u;
    block->flags &= SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT;
    block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED;
    block->generation += 1u;
    block->last_used_epoch = arena->epoch;
    arena->recycled_block_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaRetainBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block->reference_count += 1u;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    arena->retained_block_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaReleaseBlockReference(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
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

SparkStatus SparkKvCacheArenaPinResidentBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if (block->residency_reference_count == UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    block->residency_reference_count += 1u;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaUnpinResidentBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block->residency_reference_count -= 1u;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaMarkBlockDirty(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkKvCacheArenaReleaseResidentSlot(
    SparkKvCacheArena *arena,
    SparkKvCacheBlock *block)
{
    uint32_t resident_slot_index;

    resident_slot_index = block->resident_slot_index;
    if (resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT)
    {
        block->key_device_address = 0u;
        block->value_device_address = 0u;
        return SPARK_STATUS_OK;
    }
    if (resident_slot_index >= arena->resident_block_capacity ||
        arena->resident_slot_logical_block_indices[resident_slot_index] !=
            block->logical_block_index)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    arena->resident_slot_logical_block_indices[resident_slot_index] =
        SPARK_KV_CACHE_NO_BLOCK;
    arena->next_resident_slot_scan = resident_slot_index;
    block->resident_slot_index = SPARK_KV_CACHE_NO_RESIDENT_SLOT;
    block->key_device_address = 0u;
    block->value_device_address = 0u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkKvCacheArenaAssignResidentSlot(
    SparkKvCacheArena *arena,
    SparkKvCacheBlock *block)
{
    uint32_t offset,resident_slot_index;

    if (block->resident_slot_index != SPARK_KV_CACHE_NO_RESIDENT_SLOT)
    {
        return SPARK_STATUS_OK;
    }
    for (offset = 0u; offset < arena->resident_block_capacity; ++offset)
    {
        resident_slot_index =
            (arena->next_resident_slot_scan + offset) %
            arena->resident_block_capacity;
        if (arena->resident_slot_logical_block_indices[resident_slot_index] !=
                SPARK_KV_CACHE_NO_BLOCK)
        {
            continue;
        }
        arena->resident_slot_logical_block_indices[resident_slot_index] =
            block->logical_block_index;
        arena->next_resident_slot_scan =
            (resident_slot_index + 1u) % arena->resident_block_capacity;
        block->resident_slot_index = resident_slot_index;
        block->key_device_address = arena->key_device_base +
            (uintptr_t)(arena->key_block_stride_bytes * resident_slot_index);
        block->value_device_address =
            SparkKvCacheArenaHasValuePayload(arena) != 0u
            ? arena->value_device_base +
                (uintptr_t)(arena->value_block_stride_bytes * resident_slot_index)
            : 0u;
        return SPARK_STATUS_OK;
    }
    arena->resident_capacity_stall_count += 1u;
    return SPARK_STATUS_CAPACITY_EXCEEDED;
}


static uint32_t SparkKvCachePrefetchPlanContainsBlock(
    const SparkKvCachePrefetchPlan *prefetch_plan,
    uint32_t logical_block_index)
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
        if (prefetch_plan->blocks[block_index].logical_block_index ==
            logical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkKvCacheBlockIsProtectedFromResidentEviction(
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t logical_block_index)
{
    return SparkKvCachePrefetchPlanContainsBlock(
            prefetch_plan,
            logical_block_index) ||
        SparkKvProtectedBlockListContainsBlock(
            protected_logical_block_indices,
            protected_logical_block_count,
            logical_block_index);
}

/*
 * C5 REUSE-VALUE KEEPNESS (docs/JIT_KV_RESPONSE.md C5). One number, higher
 * = keep, computed only when the arena's eviction policy is REUSE_VALUE:
 *
 *   keepness = restore_count * RESTORE_WEIGHT - age + dirty * DIRTY_BONUS
 *
 * The restored-again history dominates: a block the workload restored twice
 * came back on purpose, and no plausible recency gap outvotes the second
 * restore (weights are orders of magnitude apart by construction - see the
 * SPARK_KV_CACHE_REUSE_VALUE_* constants in spark_kv_cache.h). Dirtiness
 * outranks recency too: parking a clean block is free (no write-back, no
 * flash wear), so a clean block drops before a dirty one even when the
 * clean one is the younger. Recency is the residual term, exactly the LRU
 * direction. Inputs are clamped (ages beyond 2^54 epochs, histories beyond
 * 2^16 restores) so the int64_t arithmetic can never overflow and the
 * ordering stays exact inside the documented domain.
 */
static int64_t SparkKvCacheReuseValueKeepness(
    const SparkKvCacheArena *arena,
    const SparkKvCacheBlock *block)
{
    uint64_t age;
    uint64_t restore_count;
    int64_t keepness;

    age = arena->epoch - block->last_used_epoch;
    if (age > SPARK_KV_CACHE_REUSE_VALUE_AGE_CLAMP)
    {
        age = SPARK_KV_CACHE_REUSE_VALUE_AGE_CLAMP;
    }
    restore_count = block->restore_count;
    if (restore_count > SPARK_KV_CACHE_REUSE_VALUE_RESTORE_CLAMP)
    {
        restore_count = SPARK_KV_CACHE_REUSE_VALUE_RESTORE_CLAMP;
    }
    keepness = (int64_t)(restore_count *
        SPARK_KV_CACHE_REUSE_VALUE_RESTORE_WEIGHT);
    keepness -= (int64_t)age;
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) != 0u)
    {
        keepness += (int64_t)SPARK_KV_CACHE_REUSE_VALUE_DIRTY_BONUS;
    }
    return keepness;
}

/* True when `block` is the better victim than `victim` under the arena's
 * eviction policy. LRU (the default) is the historical order: fewer
 * references first, then the older epoch. REUSE_VALUE keeps the
 * reference-count primary and replaces the recency tiebreak with the
 * keepness score above (references still lose first - a referenced block is
 * someone's working set; the reuse-value policy only ranks the rest). */
static uint32_t SparkKvCacheBlockIsBetterEvictionVictim(
    const SparkKvCacheArena *arena,
    const SparkKvCacheBlock *victim,
    int64_t victim_keepness,
    const SparkKvCacheBlock *block)
{
    int64_t keepness;

    if (block->reference_count != victim->reference_count)
    {
        return block->reference_count < victim->reference_count;
    }
    if (arena->eviction_policy == SPARK_KV_CACHE_EVICTION_POLICY_REUSE_VALUE)
    {
        keepness = SparkKvCacheReuseValueKeepness(arena, block);
        if (keepness != victim_keepness)
        {
            return keepness < victim_keepness;
        }
        return block->last_used_epoch < victim->last_used_epoch;
    }
    return block->last_used_epoch < victim->last_used_epoch;
}

static SparkKvCacheBlock *SparkKvCacheArenaSelectResidentEvictionVictim(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count)
{
    SparkKvCacheBlock *victim;
    uint32_t logical_block_index,resident_slot_index;
    int64_t victim_keepness;

    victim = 0;
    victim_keepness = 0;
    for (resident_slot_index = 0u;
         resident_slot_index < arena->resident_block_capacity;
         ++resident_slot_index)
    {
        SparkKvCacheBlock *block;

        logical_block_index =
            arena->resident_slot_logical_block_indices[resident_slot_index];
        if (logical_block_index == SPARK_KV_CACHE_NO_BLOCK)
        {
            continue;
        }
        if (logical_block_index >= arena->logical_block_count)
        {
            return 0;
        }
        block = &arena->blocks[logical_block_index];
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
            block->residency_reference_count != 0u ||
            (arena->evict_function == 0 && block->reference_count != 0u) ||
            SparkKvCacheBlockIsProtectedFromResidentEviction(
                prefetch_plan,
                protected_logical_block_indices,
                protected_logical_block_count,
                logical_block_index))
        {
            continue;
        }
        if (victim == 0 ||
            SparkKvCacheBlockIsBetterEvictionVictim(
                arena, victim, victim_keepness, block))
        {
            victim = block;
            victim_keepness = SparkKvCacheReuseValueKeepness(arena, block);
        }
    }
    return victim;
}

static SparkStatus SparkKvCacheArenaEvictResidentBlock(
    SparkKvCacheArena *arena,
    SparkKvCacheBlock *block)
{
    SparkStatus status;

    if (arena == 0 || block == 0 ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (arena->evict_function != 0 &&
        ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) != 0u ||
         (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u))
    {
        status = arena->evict_function(
            arena->evict_context,
            block->logical_block_index,
            block->resident_slot_index,
            block->generation,
            block->key_device_address,
            arena->key_block_stride_bytes,
            block->value_device_address,
            arena->value_block_stride_bytes);
        if (status != SPARK_STATUS_OK)
        {
            /* B1 WRITE-BACK WEDGE (docs/JIT_KV_RESPONSE.md): the backing
             * store refused the write (ENOSPC / full disk surface here as
             * IO_ERROR; no free backing slot or no device room as
             * CAPACITY_EXCEEDED). Retrying forever wedges the arena: the
             * block can never be written, so it can never stop being a
             * victim, so no new resident slot can ever be granted and
             * admission stalls permanently. DEGRADE instead, per the JIT-KV
             * contract: drop the block and let the sequence recompute it on
             * demand. Clearing BACKING_VALID alongside DIRTY is what makes
             * the drop safe - restore (SparkKvPageStorePrefetch) gates on
             * BACKING_VALID and answers NOT_FOUND, never reading a stale or
             * partial slot. Transient statuses are NOT degradation: BUSY is
             * the async write-back in flight (the ordinary backpressure
             * path), and any other status is a program error that must stay
             * loud. */
            if (status == SPARK_STATUS_IO_ERROR ||
                status == SPARK_STATUS_CAPACITY_EXCEEDED)
            {
                if (arena->write_back_degraded_block_count != UINT32_MAX)
                {
                    arena->write_back_degraded_block_count += 1u;
                }
                block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
                block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID;
            }
            else
            {
                return status;
            }
        }
        else
        {
            block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID;
            block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
        }
    }
    status = SparkKvCacheArenaReleaseResidentSlot(arena, block);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT;
    if (arena->resident_block_count != 0u)
    {
        arena->resident_block_count -= 1u;
    }
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    arena->resident_evicted_block_count += 1u;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkKvCacheArenaTrimResidentBlocksWithPrefetchProtection(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if ((protected_logical_block_count != 0u &&
         protected_logical_block_indices == 0) ||
        target_resident_block_count > arena->resident_block_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (arena->resident_block_count > target_resident_block_count)
    {
        SparkKvCacheBlock *victim;

        victim = SparkKvCacheArenaSelectResidentEvictionVictim(
            arena,
            prefetch_plan,
            protected_logical_block_indices,
            protected_logical_block_count);
        if (victim == 0)
        {
            arena->resident_capacity_stall_count += 1u;
            if (evicted_block_count_out != 0)
            {
                *evicted_block_count_out = evicted_block_count;
            }
            return SPARK_STATUS_CAPACITY_EXCEEDED;
        }
        status = SparkKvCacheArenaEvictResidentBlock(arena, victim);
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

SparkStatus SparkKvCacheArenaTrimResidentBlocks(
    SparkKvCacheArena *arena,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out)
{
    return SparkKvCacheArenaTrimResidentBlocksWithPrefetchProtection(
        arena,
        0,
        protected_logical_block_indices,
        protected_logical_block_count,
        target_resident_block_count,
        evicted_block_count_out);
}

static SparkStatus SparkKvCacheArenaMakeRoomForResidentBlocks(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t new_resident_block_count)
{
    uint32_t target_resident_block_count,unassigned_resident_block_count;

	unassigned_resident_block_count = atomic_load(
		&arena->unassigned_resident_block_count);

    if (new_resident_block_count > arena->resident_block_capacity)
    {
        arena->resident_capacity_stall_count += 1u;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
	if ((uint64_t)arena->resident_block_count + arena->reserved_block_count +
			unassigned_resident_block_count +
            new_resident_block_count <=
        arena->resident_block_capacity)
    {
        return SPARK_STATUS_OK;
    }
	if ( (uint64_t)arena->reserved_block_count +
		unassigned_resident_block_count +
		new_resident_block_count > arena->resident_block_capacity )
	{
		arena->resident_capacity_stall_count += 1u;
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
    target_resident_block_count =
        arena->resident_block_capacity -
        (uint32_t)arena->reserved_block_count -
		unassigned_resident_block_count - new_resident_block_count;
    return SparkKvCacheArenaTrimResidentBlocksWithPrefetchProtection(
        arena,
        prefetch_plan,
        protected_logical_block_indices,
        protected_logical_block_count,
        target_resident_block_count,
        0);
}


SparkStatus SparkKvCacheArenaMarkBlockResident(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    uint32_t protected_logical_block_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
    {
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u)
        {
            return SPARK_STATUS_BUSY;
        }
        protected_logical_block_index = logical_block_index;
        status = SparkKvCacheArenaMakeRoomForResidentBlocks(
            arena,
            0,
            &protected_logical_block_index,
            1u,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkKvCacheArenaAssignResidentSlot(arena, block);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        arena->resident_block_count += 1u;
    }
    block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT;
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

/*
 * The JIT-KV restore half (docs/JIT_KV_DESIGN.md): a block the pager PARKED -
 * ALLOCATED, non-resident, BACKING_VALID, exactly the state eviction leaves -
 * becomes resident again after the pager has brought its bytes back from the
 * backing tier. SparkKvCacheArenaMarkBlockResident refuses backing-valid
 * blocks on purpose, because a plain mark would hand out a resident block
 * nobody re-filled; THIS is the re-fill path's counterpart, for the caller
 * that owns the backing truth (the pager restored the bytes digest-verified
 * at the tier boundary). BACKING_VALID survives - the backing copy still
 * matches, so re-parking the block later is a deduplicated no-write - and
 * DIRTY stays clear. Room is made the same way as every other residency
 * grant: the resident victim picked by the arena's eviction policy is
 * evicted through the arena's evict function, which under the pager IS a
 * page-out. Each fresh re-attachment also bumps the block's restore_count -
 * the restored-again history the C5 REUSE_VALUE victim policy ranks (a block
 * the workload keeps restoring is hot; LRU never reads it). Blocks that were
 * never parked (blank) are refused: they go through MarkBlockResident.
 */
SparkStatus SparkKvCacheArenaMarkParkedBlockResident(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    uint32_t protected_logical_block_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
    {
        arena->epoch += 1u;
        block->last_used_epoch = arena->epoch;
        return SPARK_STATUS_OK;
    }
    protected_logical_block_index = logical_block_index;
    status = SparkKvCacheArenaMakeRoomForResidentBlocks(
        arena,
        0,
        &protected_logical_block_index,
        1u,
        1u);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkKvCacheArenaAssignResidentSlot(arena, block);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    arena->resident_block_count += 1u;
    block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT;
    /* C5: a completed page-in is the restored-again event. Saturating: the
     * count ranks victims, it is not an accounting quantity. */
    if (block->restore_count != UINT32_MAX)
    {
        block->restore_count += 1u;
    }
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

/*
 * The parkability predicate (see spark_kv_cache.h): the resident-eviction
 * selector's structural exclusions as a single answer. The selector and the
 * pager's admission pool count must agree with this test, not restate it.
 */
uint32_t SparkKvCacheArenaBlockIsParkable(
    const SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    const SparkKvCacheBlock *block;

    if (arena == 0 || logical_block_index >= arena->logical_block_count)
    {
        return 0u;
    }
    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
        (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return 0u;
    }
    return 1u;
}

SparkStatus SparkKvCacheArenaFreeBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (block->reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
    {
        status = SparkKvCacheArenaReleaseResidentSlot(arena, block);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        if (arena->resident_block_count == 0u)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        arena->resident_block_count -= 1u;
    }
    arena->epoch += 1u;
    block->flags = 0u;
    block->generation += 1u;
    block->last_used_epoch = arena->epoch;
    block->free_next = arena->free_logical_block_head;
    arena->free_logical_block_head = logical_block_index;
    return SPARK_STATUS_OK;
}


SparkStatus SparkKvCacheArenaMarkBlockNonResident(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index)
{
    SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
        block->residency_reference_count != 0u)
    {
        return SPARK_STATUS_BUSY;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
    {
        return SparkKvCacheArenaEvictResidentBlock(arena, block);
    }
    arena->epoch += 1u;
    block->last_used_epoch = arena->epoch;
    return SPARK_STATUS_OK;
}

static void SparkKvCachePrefetchPlanInitialize(
    SparkKvCachePrefetchPlan *prefetch_plan,
    uint32_t lane_count,
    uint32_t requested_logical_block_count)
{
    memset(prefetch_plan, 0, sizeof(*prefetch_plan));
    prefetch_plan->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    prefetch_plan->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES;
    prefetch_plan->lane_count = lane_count;
    prefetch_plan->requested_logical_block_count =
        requested_logical_block_count;
}

static uint32_t SparkKvCachePrefetchPlanAlreadyContainsBlock(
    const SparkKvCachePrefetchPlan *prefetch_plan,
    uint32_t logical_block_index)
{
    uint32_t block_index;

    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        if (prefetch_plan->blocks[block_index].logical_block_index ==
            logical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static void SparkKvCachePrefetchSourceInitializeFromLogicalBlock(
    SparkKvCachePrefetchSourceBlock *source_block,
    uint32_t logical_block_index,
    uint32_t include_value_payload)
{
    memset(source_block, 0, sizeof(*source_block));
    source_block->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    source_block->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block->logical_block_index = logical_block_index;
    source_block->flags = SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY;
    if (include_value_payload != 0u)
    {
        source_block->flags |= SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE;
    }
}

static SparkStatus SparkKvCacheValidatePrefetchSourceBlock(
    const SparkKvCacheArena *arena,
    const SparkKvCachePrefetchSourceBlock *source_block)
{
    if (source_block == 0 ||
        source_block->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        source_block->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES ||
        source_block->logical_block_index >= arena->logical_block_count ||
        (source_block->flags &
            SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) == 0u ||
        (source_block->flags &
            ~SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((source_block->flags &
            SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE) != 0u &&
        SparkKvCacheArenaHasValuePayload(arena) == 0u)
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

static void SparkKvCachePrefetchPlanAddSourceBlock(
    SparkKvCachePrefetchPlan *prefetch_plan,
    const SparkKvCacheBlock *block,
    const SparkKvCachePrefetchSourceBlock *source_block)
{
    SparkKvCachePrefetchBlock *prefetch_block;
    uint32_t lane_index;

    lane_index = prefetch_plan->prefetch_block_count %
        prefetch_plan->lane_count;
    prefetch_block = &prefetch_plan->blocks[
        prefetch_plan->prefetch_block_count];
    memset(prefetch_block, 0, sizeof(*prefetch_block));
    prefetch_block->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    prefetch_block->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES;
    prefetch_block->lane_index = lane_index;
    prefetch_block->logical_block_index = block->logical_block_index;
    prefetch_block->resident_slot_index = block->resident_slot_index;
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

static SparkStatus SparkKvCacheArenaReserveBlockResidency(
    SparkKvCacheArena *arena,
    SparkKvCacheBlock *block,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkStatus status;

    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
    {
        return SPARK_STATUS_OK;
    }
    if (block->residency_reference_count == UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) == 0u)
    {
        status = SparkKvCacheArenaMakeRoomForResidentBlocks(
            arena,
            prefetch_plan,
            0,
            0u,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        status = SparkKvCacheArenaAssignResidentSlot(arena, block);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
        block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED;
        arena->reserved_block_count += 1u;
    }
    block->residency_reference_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_count,
    uint32_t lane_count,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_plan == 0 || lane_count == 0u ||
        lane_count > SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        (source_block_count != 0u && source_blocks == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkKvCachePrefetchPlanInitialize(
        prefetch_plan,
        lane_count,
        source_block_count);
    for (block_index = 0u; block_index < source_block_count; ++block_index)
    {
        SparkKvCacheBlock *block;
        const SparkKvCachePrefetchSourceBlock *source_block;
        uint32_t logical_block_index;

        source_block = &source_blocks[block_index];
        status = SparkKvCacheValidatePrefetchSourceBlock(
            arena,
            source_block);
        if (status != SPARK_STATUS_OK)
        {
            goto rollback;
        }
        logical_block_index = source_block->logical_block_index;
        if (SparkKvCachePrefetchPlanAlreadyContainsBlock(
                prefetch_plan,
                logical_block_index))
        {
            prefetch_plan->duplicate_block_count += 1u;
            continue;
        }

        block = &arena->blocks[logical_block_index];
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
        {
            prefetch_plan->missing_block_count += 1u;
            status = SPARK_STATUS_NOT_FOUND;
            goto rollback;
        }
        if (source_block->generation != 0u &&
            source_block->generation != block->generation)
        {
            status = SPARK_STATUS_HASH_MISMATCH;
            goto rollback;
        }
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u)
        {
            prefetch_plan->resident_block_count += 1u;
            continue;
        }
        if (prefetch_plan->prefetch_block_count >=
            SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
        {
            status = SPARK_STATUS_CAPACITY_EXCEEDED;
            goto rollback;
        }
        status = SparkKvCacheArenaReserveBlockResidency(
            arena,
            block,
            prefetch_plan);
        if (status != SPARK_STATUS_OK)
        {
            goto rollback;
        }
        SparkKvCachePrefetchPlanAddSourceBlock(
            prefetch_plan,
            block,
            source_block);
        prefetch_plan->reserved_block_count += 1u;
    }
    return SPARK_STATUS_OK;

rollback:
    (void)SparkKvCacheArenaCancelPrefetchPlan(arena, prefetch_plan);
    return status;
}

SparkStatus SparkKvCacheArenaBuildPrefetchPlan(
    SparkKvCacheArena *arena,
    const uint32_t *logical_block_indices,
    uint32_t logical_block_count,
    uint32_t lane_count,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCachePrefetchSourceBlock source_blocks[
        SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
    uint32_t block_index;

    if (logical_block_count > SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (logical_block_count != 0u && logical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (block_index = 0u; block_index < logical_block_count; ++block_index)
    {
        SparkKvCachePrefetchSourceInitializeFromLogicalBlock(
            &source_blocks[block_index],
            logical_block_indices[block_index],
            SparkKvCacheArenaHasValuePayload(arena));
    }
    return SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        arena,
        logical_block_count != 0u ? source_blocks : 0,
        logical_block_count,
        lane_count,
        prefetch_plan);
}

SparkStatus SparkKvCachePrefetchCursorInitialize(
    SparkKvCachePrefetchCursor *cursor,
    uint32_t logical_block_count)
{
    if (cursor == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(cursor, 0, sizeof(*cursor));
    cursor->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    cursor->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_CURSOR_DESCRIPTOR_BYTES;
    cursor->logical_block_count = logical_block_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaBuildNextPrefetchPlan(
    SparkKvCacheArena *arena,
    const uint32_t *logical_block_indices,
    uint32_t lane_count,
    SparkKvCachePrefetchCursor *cursor,
    SparkKvCachePrefetchPlan *prefetch_plan)
{
    uint32_t remaining_block_count,chunk_block_count;
    SparkStatus status;

    if (cursor == 0 || prefetch_plan == 0 ||
        cursor->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        cursor->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_CURSOR_DESCRIPTOR_BYTES ||
        cursor->next_logical_block_index > cursor->logical_block_count ||
        (cursor->logical_block_count != 0u && logical_block_indices == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    remaining_block_count = cursor->logical_block_count -
        cursor->next_logical_block_index;
    chunk_block_count = remaining_block_count >
        SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY ?
        SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY : remaining_block_count;
    status = SparkKvCacheArenaBuildPrefetchPlan(
        arena,
        chunk_block_count != 0u ? logical_block_indices +
            cursor->next_logical_block_index : 0,
        chunk_block_count,
        lane_count,
        prefetch_plan);
    if (status == SPARK_STATUS_OK)
    {
        cursor->next_logical_block_index += chunk_block_count;
    }
    return status;
}

static SparkStatus SparkKvCacheValidatePrefetchPlan(
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    uint32_t block_index;

    if (prefetch_plan == 0 ||
        prefetch_plan->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        prefetch_plan->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES ||
        prefetch_plan->lane_count == 0u ||
        prefetch_plan->lane_count > SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        prefetch_plan->prefetch_block_count >
            SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        const SparkKvCachePrefetchBlock *prefetch_block;

        prefetch_block = &prefetch_plan->blocks[block_index];
        if (prefetch_block->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
            prefetch_block->descriptor_bytes !=
                SPARK_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES ||
            prefetch_block->lane_index >= prefetch_plan->lane_count ||
            prefetch_block->resident_slot_index ==
                SPARK_KV_CACHE_NO_RESIDENT_SLOT ||
            prefetch_block->key_device_address == 0u ||
            prefetch_block->reserved0 != 0u ||
            (prefetch_block->flags &
                SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) == 0u ||
            (prefetch_block->flags &
                ~SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS) != 0u)
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkKvCacheArenaValidatePlanBlock(
    const SparkKvCacheArena *arena,
    const SparkKvCachePrefetchBlock *prefetch_block)
{
    const SparkKvCacheBlock *block;

    if (prefetch_block->logical_block_index >= arena->logical_block_count ||
        prefetch_block->resident_slot_index >= arena->resident_block_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    block = &arena->blocks[prefetch_block->logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (block->generation != prefetch_block->generation)
    {
        return SPARK_STATUS_HASH_MISMATCH;
    }
    if (block->resident_slot_index != prefetch_block->resident_slot_index ||
        block->key_device_address != prefetch_block->key_device_address ||
        block->value_device_address != prefetch_block->value_device_address ||
        block->residency_reference_count == 0u ||
        arena->resident_slot_logical_block_indices[
            block->resident_slot_index] != block->logical_block_index ||
        ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u &&
         (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) == 0u))
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count)
{
    SparkKvCacheBlock *block;
    uint32_t block_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkKvCacheValidatePrefetchPlan(prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (protected_logical_block_count != 0u &&
        protected_logical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        status = SparkKvCacheArenaValidatePlanBlock(
            arena,
            &prefetch_plan->blocks[block_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        block = &arena->blocks[
            prefetch_plan->blocks[block_index].logical_block_index];
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u)
        {
            if (arena->reserved_block_count == 0u)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            arena->reserved_block_count -= 1u;
            arena->resident_block_count += 1u;
            block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED;
            block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT;
        }
        block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID;
        block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
        block->residency_reference_count -= 1u;
        arena->epoch += 1u;
        block->last_used_epoch = arena->epoch;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheArenaMarkPrefetchPlanResident(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    return SparkKvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
        arena,
        prefetch_plan,
        0,
        0u);
}

SparkStatus SparkKvCacheArenaCancelPrefetchPlan(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCacheBlock *block;
    SparkStatus status;
    uint32_t block_index;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkKvCacheValidatePrefetchPlan(prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        status = SparkKvCacheArenaValidatePlanBlock(
            arena,
            &prefetch_plan->blocks[block_index]);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    for (block_index = 0u;
         block_index < prefetch_plan->prefetch_block_count;
         ++block_index)
    {
        block = &arena->blocks[
            prefetch_plan->blocks[block_index].logical_block_index];
        block->residency_reference_count -= 1u;
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u &&
            block->residency_reference_count == 0u)
        {
            if (arena->reserved_block_count == 0u)
            {
                return SPARK_STATUS_INTERNAL_ERROR;
            }
            arena->reserved_block_count -= 1u;
            block->flags &= ~SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED;
            status = SparkKvCacheArenaReleaseResidentSlot(arena, block);
            if (status != SPARK_STATUS_OK)
            {
                return status;
            }
        }
    }
    return SPARK_STATUS_OK;
}

static uint32_t SparkKvCacheLogicalBlockIsProtected(
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t logical_block_index)
{
    uint32_t protected_index;

    if (protected_logical_block_indices == 0)
    {
        return 0u;
    }
    for (protected_index = 0u;
         protected_index < protected_logical_block_count;
         ++protected_index)
    {
        if (protected_logical_block_indices[protected_index] ==
            logical_block_index)
        {
            return 1u;
        }
    }
    return 0u;
}

static uint32_t SparkKvCacheSelectResidentEvictionVictim(
    const SparkKvCacheArena *arena,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count)
{
    uint32_t logical_block_index;
    uint32_t victim_logical_block_index;
    const SparkKvCacheBlock *victim_block;
    int64_t victim_keepness;

    victim_logical_block_index = SPARK_KV_CACHE_NO_BLOCK;
    victim_block = 0;
    victim_keepness = 0;
    for (logical_block_index = 0u;
         logical_block_index < arena->logical_block_count;
         ++logical_block_index)
    {
        const SparkKvCacheBlock *block;

        block = &arena->blocks[logical_block_index];
        if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u ||
            (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
            block->residency_reference_count != 0u ||
            (arena->evict_function == 0 && block->reference_count != 0u) ||
            SparkKvCacheLogicalBlockIsProtected(
                protected_logical_block_indices,
                protected_logical_block_count,
                logical_block_index))
        {
            continue;
        }
        if (victim_block == 0 ||
            SparkKvCacheBlockIsBetterEvictionVictim(
                arena, victim_block, victim_keepness, block))
        {
            victim_block = block;
            victim_logical_block_index = logical_block_index;
            victim_keepness = SparkKvCacheReuseValueKeepness(arena, block);
        }
    }
    return victim_logical_block_index;
}

SparkStatus SparkKvCacheArenaEvictResidentBlocksToLimit(
    SparkKvCacheArena *arena,
    uint32_t max_resident_block_count,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t *evicted_block_count_out)
{
    uint32_t evicted_block_count;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (protected_logical_block_count != 0u &&
        protected_logical_block_indices == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    evicted_block_count = 0u;
    while (arena->resident_block_count > max_resident_block_count)
    {
        uint32_t victim_logical_block_index;

        victim_logical_block_index = SparkKvCacheSelectResidentEvictionVictim(
            arena,
            protected_logical_block_indices,
            protected_logical_block_count);
        if (victim_logical_block_index == SPARK_KV_CACHE_NO_BLOCK)
        {
            break;
        }
        status = SparkKvCacheArenaMarkBlockNonResident(
            arena,
            victim_logical_block_index);
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

SparkStatus SparkKvCacheArenaResolveBlock(
    const SparkKvCacheArena *arena,
    uint32_t logical_block_index,
    SparkKvCacheBlockView *block_view)
{
    const SparkKvCacheBlock *block;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (logical_block_index >= arena->logical_block_count || block_view == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    block = &arena->blocks[logical_block_index];
    if ((block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u)
    {
        return SPARK_STATUS_NOT_FOUND;
    }

    memset(block_view, 0, sizeof(*block_view));
    block_view->abi_version = SPARK_KV_CACHE_ABI_VERSION;
    block_view->descriptor_bytes = SPARK_KV_CACHE_BLOCK_VIEW_DESCRIPTOR_BYTES;
    block_view->logical_block_index = logical_block_index;
    block_view->resident_slot_index = block->resident_slot_index;
    block_view->flags = block->flags;
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

SparkStatus SparkKvCacheArenaReset(
    SparkKvCacheArena *arena)
{
    uint32_t logical_block_index,resident_slot_index;
    SparkStatus status;

    status = SparkKvCacheArenaValidate(arena);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    for (logical_block_index = 0u;
         logical_block_index < arena->logical_block_count;
         ++logical_block_index)
    {
        SparkKvCacheInitializeBlock(arena, logical_block_index);
    }
    for (resident_slot_index = 0u;
         resident_slot_index < arena->resident_block_capacity;
         ++resident_slot_index)
    {
        arena->resident_slot_logical_block_indices[resident_slot_index] =
            SPARK_KV_CACHE_NO_BLOCK;
    }
    arena->free_logical_block_head = 0u;
    arena->next_resident_slot_scan = 0u;
    arena->epoch = 0u;
    arena->allocated_block_count = 0u;
    arena->recycled_block_count = 0u;
    arena->resident_block_count = 0u;
    arena->reserved_block_count = 0u;
	atomic_store(&arena->unassigned_resident_block_count,0u);
    arena->retained_block_count = 0u;
    arena->released_reference_count = 0u;
    return SPARK_STATUS_OK;
}


static uint32_t SparkKvCacheAsyncPrefetchBackendSourceModeIsValid(
    uint32_t flags)
{
    uint32_t source_mode_count;

    source_mode_count = 0u;
    if ((flags & SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        source_mode_count += 1u;
    }
    if ((flags & SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE) != 0u)
    {
        source_mode_count += 1u;
    }
    return source_mode_count == 1u;
}

static uint32_t SparkKvCacheAsyncPrefetchBackendCopyFlagsAreValid(
    uint32_t flags)
{
    return (flags & SPARK_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS) != 0u;
}

static SparkStatus SparkKvCacheAsyncPrefetchBackendValidateConfiguration(
    const SparkKvCacheAsyncPrefetchBackendConfiguration *configuration)
{
    if (configuration == 0 ||
        configuration->abi_version != SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        configuration->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES ||
        (configuration->flags & ~SPARK_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS) != 0u ||
        !SparkKvCacheAsyncPrefetchBackendSourceModeIsValid(
            configuration->flags) ||
        !SparkKvCacheAsyncPrefetchBackendCopyFlagsAreValid(
            configuration->flags) ||
        configuration->lane_count == 0u ||
        configuration->lane_count > SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        configuration->max_inflight_prefetch_count == 0u ||
        configuration->max_inflight_prefetch_count >
            SPARK_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY ||
        configuration->logical_block_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((configuration->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (configuration->key_source_stride_bytes == 0u ||
         configuration->key_transfer_bytes == 0u ||
         configuration->key_transfer_bytes > configuration->key_source_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (configuration->value_source_stride_bytes == 0u ||
         configuration->value_transfer_bytes == 0u ||
         configuration->value_transfer_bytes > configuration->value_source_stride_bytes))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((configuration->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        if (((configuration->flags &
                SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
                configuration->key_source_base == 0) ||
            ((configuration->flags &
                SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
                configuration->value_source_base == 0))
        {
            return SPARK_STATUS_INVALID_ARGUMENT;
        }
    }
    if ((configuration->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE) != 0u)
    {
        if (((configuration->flags &
                SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
                configuration->key_file_descriptor < 0) ||
            ((configuration->flags &
                SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
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

static SparkStatus SparkKvCacheAsyncPrefetchBackendValidate(
    const SparkKvCacheAsyncPrefetchBackend *backend)
{
    if (backend == 0 ||
        backend->abi_version != SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        backend->descriptor_bytes != SPARK_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES ||
        (backend->flags & ~SPARK_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS) != 0u ||
        !SparkKvCacheAsyncPrefetchBackendSourceModeIsValid(backend->flags) ||
        !SparkKvCacheAsyncPrefetchBackendCopyFlagsAreValid(backend->flags) ||
        backend->lane_count == 0u ||
        backend->lane_count > SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT ||
        backend->max_inflight_prefetch_count == 0u ||
        backend->max_inflight_prefetch_count >
            SPARK_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY ||
        backend->logical_block_count == 0u ||
        backend->blocks_per_poll == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((backend->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (backend->key_source_stride_bytes == 0u || backend->key_transfer_bytes == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((backend->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (backend->value_source_stride_bytes == 0u || backend->value_transfer_bytes == 0u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheAsyncPrefetchBackendInitialize(
    SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCacheAsyncPrefetchBackendConfiguration *configuration)
{
    uint32_t request_index;
    SparkStatus status;

    status = SparkKvCacheAsyncPrefetchBackendValidateConfiguration(
        configuration);
    if (backend == 0 || status != SPARK_STATUS_OK)
    {
        return backend == 0 ? SPARK_STATUS_INVALID_ARGUMENT : status;
    }

    memset(backend, 0, sizeof(*backend));
    backend->abi_version = SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    backend->descriptor_bytes = SPARK_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES;
    backend->flags = configuration->flags;
    backend->lane_count = configuration->lane_count;
    backend->max_inflight_prefetch_count =
        configuration->max_inflight_prefetch_count;
    backend->logical_block_count = configuration->logical_block_count;
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
         request_index < SPARK_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY;
         ++request_index)
    {
        backend->requests[request_index].abi_version =
            SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
        backend->requests[request_index].descriptor_bytes =
            SPARK_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
    }
    return SPARK_STATUS_OK;
}

static SparkKvCacheAsyncPrefetchRequest *
SparkKvCacheAsyncPrefetchBackendFindRequest(
    SparkKvCacheAsyncPrefetchBackend *backend,
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

static SparkKvCacheAsyncPrefetchRequest *
SparkKvCacheAsyncPrefetchBackendFindFreeRequest(
    SparkKvCacheAsyncPrefetchBackend *backend)
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

static void SparkKvCacheAsyncPrefetchBackendClearRequest(
    SparkKvCacheAsyncPrefetchRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    request->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
}

static uint32_t SparkKvCachePrefetchBackendSourceEntryMatches(
    const SparkKvCachePrefetchBackendSourceEntry *source_entry,
    const SparkKvCachePrefetchBlock *prefetch_block)
{
    if (source_entry->abi_version !=
            SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION ||
        source_entry->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_BACKEND_SOURCE_ENTRY_DESCRIPTOR_BYTES)
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
    if (source_entry->logical_block_index != SPARK_KV_CACHE_NO_BLOCK &&
        source_entry->logical_block_index == prefetch_block->logical_block_index)
    {
        return 1u;
    }
    return 0u;
}

static const SparkKvCachePrefetchBackendSourceEntry *
SparkKvCacheAsyncPrefetchBackendFindSourceEntry(
    const SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCachePrefetchBlock *prefetch_block)
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
        if (SparkKvCachePrefetchBackendSourceEntryMatches(
                &backend->source_entries[source_index],
                prefetch_block))
        {
            return &backend->source_entries[source_index];
        }
    }
    return 0;
}

static uint64_t SparkKvCacheAsyncPrefetchBackendDefaultSourceOffset(
    uint32_t logical_block_index,
    uint64_t source_stride_bytes)
{
    return (uint64_t)logical_block_index * source_stride_bytes;
}

static SparkStatus SparkKvCacheAsyncPrefetchBackendResolveSource(
    const SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCachePrefetchBlock *prefetch_block,
    uint32_t key_source,
    const void **memory_source_out,
    uint64_t *file_offset_out)
{
    const SparkKvCachePrefetchBackendSourceEntry *source_entry;
    uint64_t offset_bytes;
    uint64_t stride_bytes;
    const void *base;

    if (memory_source_out == 0 || file_offset_out == 0 ||
        prefetch_block->logical_block_index >= backend->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    source_entry = SparkKvCacheAsyncPrefetchBackendFindSourceEntry(
        backend,
        prefetch_block);
    stride_bytes = key_source != 0u
        ? backend->key_source_stride_bytes
        : backend->value_source_stride_bytes;
    offset_bytes = SparkKvCacheAsyncPrefetchBackendDefaultSourceOffset(
        prefetch_block->logical_block_index,
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
                SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
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
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
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
static SparkStatus SparkKvCacheAsyncPrefetchBackendReadExact(
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
static SparkStatus SparkKvCacheAsyncPrefetchBackendReadExact(
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

static SparkStatus SparkKvCacheAsyncPrefetchBackendCopyOnePayload(
    SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCachePrefetchBlock *prefetch_block,
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

    status = SparkKvCacheAsyncPrefetchBackendResolveSource(
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
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE) != 0u)
    {
        memcpy((void *)destination_address, memory_source, (size_t)transfer_bytes);
    }
    else
    {
        int32_t file_descriptor;

        file_descriptor = key_payload != 0u
            ? backend->key_file_descriptor
            : backend->value_file_descriptor;
        status = SparkKvCacheAsyncPrefetchBackendReadExact(
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

static SparkStatus SparkKvCacheAsyncPrefetchBackendCopyOneBlock(
    SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCachePrefetchBlock *prefetch_block)
{
    SparkStatus status;

    if (prefetch_block == 0 ||
        prefetch_block->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        prefetch_block->descriptor_bytes !=
            SPARK_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES ||
        prefetch_block->logical_block_index >= backend->logical_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if ((backend->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS) != 0u &&
        (prefetch_block->flags & SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY) != 0u)
    {
        status = SparkKvCacheAsyncPrefetchBackendCopyOnePayload(
            backend,
            prefetch_block,
            1u);
        if (status != SPARK_STATUS_OK)
        {
            return status;
        }
    }
    if ((backend->flags &
            SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS) != 0u &&
        (prefetch_block->flags & SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE) != 0u)
    {
        status = SparkKvCacheAsyncPrefetchBackendCopyOnePayload(
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

SparkStatus SparkKvCacheAsyncPrefetchBackendStart(
    void *context,
    uint64_t prefetch_id,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCacheAsyncPrefetchBackend *backend;
    SparkKvCacheAsyncPrefetchRequest *request;
    SparkStatus status;

    backend = (SparkKvCacheAsyncPrefetchBackend *)context;
    status = SparkKvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_id == 0u ||
        SparkKvCacheValidatePrefetchPlan(prefetch_plan) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkKvCacheAsyncPrefetchBackendFindRequest(
            backend,
            prefetch_id) != 0)
    {
        return SPARK_STATUS_DUPLICATE;
    }
    request = SparkKvCacheAsyncPrefetchBackendFindFreeRequest(backend);
    if (request == 0)
    {
        return SPARK_STATUS_BUSY;
    }

    memset(request, 0, sizeof(*request));
    request->abi_version = SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    request->descriptor_bytes =
        SPARK_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES;
    request->active = 1u;
    request->prefetch_id = prefetch_id;
    request->terminal_status = SPARK_STATUS_BUSY;
    request->prefetch_plan = *prefetch_plan;
    backend->started_prefetch_count += 1u;
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheAsyncPrefetchBackendPoll(
    void *context,
    uint64_t prefetch_id,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCacheAsyncPrefetchBackend *backend;
    SparkKvCacheAsyncPrefetchRequest *request;
    SparkStatus status;
    uint32_t block_budget;
    uint32_t processed_block_count;

    backend = (SparkKvCacheAsyncPrefetchBackend *)context;
    status = SparkKvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (prefetch_id == 0u || prefetch_plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    request = SparkKvCacheAsyncPrefetchBackendFindRequest(
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
        status = SparkKvCacheAsyncPrefetchBackendCopyOneBlock(
            backend,
            &request->prefetch_plan.blocks[request->completed_block_count]);
        if (status != SPARK_STATUS_OK)
        {
            backend->failed_prefetch_count += 1u;
            SparkKvCacheAsyncPrefetchBackendClearRequest(request);
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
    SparkKvCacheAsyncPrefetchBackendClearRequest(request);
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvCacheAsyncPrefetchBackendSubmitSynchronous(
    void *context,
    const SparkKvCachePrefetchPlan *prefetch_plan)
{
    SparkKvCacheAsyncPrefetchBackend *backend;
    uint64_t prefetch_id;
    SparkStatus status;

    backend = (SparkKvCacheAsyncPrefetchBackend *)context;
    status = SparkKvCacheAsyncPrefetchBackendValidate(backend);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    prefetch_id = backend->started_prefetch_count + 1u;
    status = SparkKvCacheAsyncPrefetchBackendStart(
        backend,
        prefetch_id,
        prefetch_plan);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    do
    {
        status = SparkKvCacheAsyncPrefetchBackendPoll(
            backend,
            prefetch_id,
            prefetch_plan);
    } while (status == SPARK_STATUS_BUSY);
    return status;
}
