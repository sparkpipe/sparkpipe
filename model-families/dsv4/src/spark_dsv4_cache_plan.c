#include "sparkpipe/spark_dsv4_cache_plan.h"

#include <limits.h>
#include <stddef.h>
#include <string.h>

#include "sparkpipe/spark_dsv4_model.h"
#include "sparkpipe/spark_dsv4_pro_model.h"

typedef struct SparkDsv4ModelCacheGeometry
{
    uint32_t backbone_layer_count;
    uint32_t mtp_layer_count;
    uint32_t head_dimension;
    uint32_t rope_dimension;
    uint32_t index_head_dimension;
    uint32_t sliding_window_tokens;
    uint32_t maximum_context_tokens;
} SparkDsv4ModelCacheGeometry;

typedef struct SparkDsv4ClassFootprint
{
    uint64_t sliding_sequence_stride_bytes;
    uint64_t compressed_history_bytes;
    uint64_t compressor_state_sequence_stride_bytes;
} SparkDsv4ClassFootprint;

static uint32_t SparkDsv4IsPowerOfTwo(uint32_t value)
{
    return(value != 0u && (value & (value - 1u)) == 0u);
}

static uint32_t SparkDsv4ElementBitWidthIsSupported(uint32_t bit_width)
{
    return(bit_width == 4u || bit_width == 8u || bit_width == 16u || bit_width == 32u);
}

static SparkStatus SparkDsv4CheckedAddU64(
    uint64_t left,
    uint64_t right,
    uint64_t *result)
{
    if (result == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (UINT64_MAX - left < right)
    {
        return(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    *result = left + right;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4CheckedMultiplyU64(
    uint64_t left,
    uint64_t right,
    uint64_t *result)
{
    if (result == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (left != 0u && right > UINT64_MAX / left)
    {
        return(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    *result = left * right;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4AlignUpU64(
    uint64_t value,
    uint32_t alignment,
    uint64_t *result)
{
    uint64_t mask;

    if (result == 0 || !SparkDsv4IsPowerOfTwo(alignment))
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    mask = (uint64_t)alignment - 1u;
    if (value > UINT64_MAX - mask)
    {
        return(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    *result = (value + mask) & ~mask;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4RoundUpToMultipleU64(
    uint64_t value,
    uint32_t multiple,
    uint64_t *result)
{
    uint64_t remainder;
    uint64_t increment;

    if (result == 0 || multiple == 0u)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    remainder = value % multiple;
    if (remainder == 0u)
    {
        *result = value;
        return(SPARK_STATUS_OK);
    }
    increment = multiple - remainder;
    return(SparkDsv4CheckedAddU64(value, increment, result));
}

static SparkStatus SparkDsv4ElementsToPackedBytes(
    uint64_t element_count,
    uint32_t element_bits,
    uint64_t *byte_count)
{
    SparkStatus status;
    uint64_t bit_count;

    status = SparkDsv4CheckedMultiplyU64(element_count, element_bits, &bit_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    if (bit_count > UINT64_MAX - 7u)
    {
        return(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    *byte_count = (bit_count + 7u) / 8u;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4AppendAlignedRegion(
    uint64_t region_bytes,
    uint32_t alignment,
    uint64_t *cursor,
    uint64_t *offset)
{
    SparkStatus status;
    uint64_t aligned_cursor;
    uint64_t end;

    if (cursor == 0 || offset == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (region_bytes == 0u)
    {
        *offset = 0u;
        return(SPARK_STATUS_OK);
    }
    status = SparkDsv4AlignUpU64(*cursor, alignment, &aligned_cursor);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedAddU64(aligned_cursor, region_bytes, &end);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    *offset = aligned_cursor;
    *cursor = end;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4ResolveModelGeometry(
    SparkDsv4ModelVariant model_variant,
    SparkDsv4ModelCacheGeometry *geometry)
{
    if (geometry == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(geometry, 0, sizeof(*geometry));
    switch (model_variant)
    {
        case SPARK_DSV4_MODEL_VARIANT_FLASH:
            geometry->backbone_layer_count = SPARK_DSV4_MODEL_LAYER_COUNT;
            geometry->mtp_layer_count = SPARK_DSV4_MODEL_MTP_LAYER_COUNT;
            geometry->head_dimension = SPARK_DSV4_MODEL_HEAD_DIMENSION;
            geometry->rope_dimension = SPARK_DSV4_MODEL_ATTN_ROPE_DIMENSION;
            geometry->index_head_dimension = SPARK_DSV4_MODEL_INDEX_HEAD_DIMENSION;
            geometry->sliding_window_tokens = SPARK_DSV4_MODEL_SLIDING_WINDOW_TOKENS;
            geometry->maximum_context_tokens = SPARK_DSV4_MODEL_MAX_POSITIONS;
            return(SPARK_STATUS_OK);
        case SPARK_DSV4_MODEL_VARIANT_PRO:
            geometry->backbone_layer_count = SPARK_DSV4_PRO_LAYER_COUNT;
            geometry->mtp_layer_count = SPARK_DSV4_PRO_MTP_LAYER_COUNT;
            geometry->head_dimension = SPARK_DSV4_PRO_HEAD_DIMENSION;
            geometry->rope_dimension = SPARK_DSV4_PRO_QK_ROPE_HEAD_DIMENSION;
            geometry->index_head_dimension = SPARK_DSV4_PRO_INDEX_HEAD_DIMENSION;
            geometry->sliding_window_tokens = SPARK_DSV4_PRO_SLIDING_WINDOW_TOKENS;
            geometry->maximum_context_tokens = SPARK_DSV4_PRO_MAXIMUM_CONTEXT_TOKENS;
            return(SPARK_STATUS_OK);
        default:
            return(SPARK_STATUS_INVALID_ARGUMENT);
    }
}

static uint16_t SparkDsv4CompressionRatioForLayer(
    SparkDsv4ModelVariant model_variant,
    uint32_t absolute_layer_index,
    uint32_t is_mtp_layer)
{
    if (model_variant == SPARK_DSV4_MODEL_VARIANT_FLASH)
    {
        return(is_mtp_layer != 0u
            ? SparkDsv4ModelMtpCompressionRatio()
            : SparkDsv4ModelBackboneCompressionRatio(absolute_layer_index));
    }
    if (model_variant == SPARK_DSV4_MODEL_VARIANT_PRO)
    {
        return(is_mtp_layer != 0u
            ? SparkDsv4ProMtpCompressionRatio()
            : SparkDsv4ProBackboneCompressionRatio(absolute_layer_index));
    }
    return(UINT16_MAX);
}

static SparkStatus SparkDsv4AttentionClassFromCompressionRatio(
    uint32_t compression_ratio,
    SparkDsv4AttentionClass *attention_class)
{
    if (attention_class == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    switch (compression_ratio)
    {
        case 0u:
            *attention_class = SPARK_DSV4_ATTENTION_CLASS_SLIDING;
            return(SPARK_STATUS_OK);
        case 4u:
            *attention_class = SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE;
            return(SPARK_STATUS_OK);
        case 128u:
            *attention_class = SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED;
            return(SPARK_STATUS_OK);
        default:
            return(SPARK_STATUS_INVALID_ARGUMENT);
    }
}

static SparkStatus SparkDsv4ValidateConfiguration(
    const SparkDsv4CachePlanConfiguration *configuration,
    SparkDsv4ModelCacheGeometry *geometry,
    uint32_t *planned_layer_count)
{
    SparkStatus status;
    uint64_t backbone_end;
    uint32_t reserved_index;

    if (configuration == 0 || geometry == 0 || planned_layer_count == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (configuration->abi_version != SPARK_DSV4_CACHE_PLAN_ABI_VERSION ||
        configuration->descriptor_bytes != sizeof(*configuration) ||
        configuration->backbone_layer_count == 0u ||
        configuration->include_mtp_layer > 1u ||
        configuration->active_sequence_capacity == 0u ||
        configuration->maximum_context_tokens_per_sequence == 0u ||
        configuration->aggregate_context_token_capacity == 0u ||
        configuration->aggregate_context_token_capacity < configuration->active_sequence_capacity ||
        configuration->compressed_history_page_entries == 0u ||
        !SparkDsv4IsPowerOfTwo(configuration->compressed_history_page_entries) ||
        !SparkDsv4ElementBitWidthIsSupported(configuration->attention_content_element_bits) ||
        !SparkDsv4ElementBitWidthIsSupported(configuration->attention_rope_element_bits) ||
        !SparkDsv4ElementBitWidthIsSupported(configuration->indexer_element_bits) ||
        !SparkDsv4ElementBitWidthIsSupported(configuration->compressor_state_element_bits) ||
        !SparkDsv4IsPowerOfTwo(configuration->allocation_alignment_bytes))
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    for (reserved_index = 0u;
         reserved_index < sizeof(configuration->reserved_u32) / sizeof(configuration->reserved_u32[0]);
         ++reserved_index)
    {
        if (configuration->reserved_u32[reserved_index] != 0u)
        {
            return(SPARK_STATUS_INVALID_ARGUMENT);
        }
    }
    status = SparkDsv4ResolveModelGeometry(configuration->model_variant, geometry);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    if (geometry->head_dimension <= geometry->rope_dimension ||
        configuration->maximum_context_tokens_per_sequence > geometry->maximum_context_tokens)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    backbone_end = (uint64_t)configuration->first_backbone_layer_index +
        configuration->backbone_layer_count;
    if (backbone_end > geometry->backbone_layer_count)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (configuration->include_mtp_layer != 0u && geometry->mtp_layer_count != 1u)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    *planned_layer_count = configuration->backbone_layer_count +
        configuration->include_mtp_layer;
    if (*planned_layer_count > SPARK_DSV4_CACHE_PLAN_MAXIMUM_LAYERS)
    {
        return(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4BuildSlidingLayout(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    SparkDsv4LayerCachePlan *layer_plan)
{
    SparkStatus status;
    uint64_t cursor;
    uint64_t element_count;
    uint64_t region_bytes;

    cursor = 0u;
    layer_plan->sliding_entry_capacity_per_sequence = geometry->sliding_window_tokens - 1u;
    status = SparkDsv4CheckedMultiplyU64(
        layer_plan->sliding_entry_capacity_per_sequence,
        geometry->head_dimension - geometry->rope_dimension,
        &element_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4ElementsToPackedBytes(
        element_count,
        configuration->attention_content_element_bits,
        &region_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AppendAlignedRegion(
        region_bytes,
        configuration->allocation_alignment_bytes,
        &cursor,
        &layer_plan->sliding_content_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        layer_plan->sliding_entry_capacity_per_sequence,
        geometry->rope_dimension,
        &element_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4ElementsToPackedBytes(
        element_count,
        configuration->attention_rope_element_bits,
        &region_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AppendAlignedRegion(
        region_bytes,
        configuration->allocation_alignment_bytes,
        &cursor,
        &layer_plan->sliding_rope_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    return(SparkDsv4AlignUpU64(
        cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->sliding_sequence_stride_bytes));
}

static SparkStatus SparkDsv4BuildCompressedHistoryLayout(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    SparkDsv4LayerCachePlan *layer_plan)
{
    SparkStatus status;
    uint64_t unrounded_entry_capacity;
    uint64_t rounded_entry_capacity;
    uint64_t cursor;
    uint64_t element_count;
    uint64_t region_bytes;

    if (layer_plan->compression_ratio == 0u)
    {
        return(SPARK_STATUS_OK);
    }
    unrounded_entry_capacity = configuration->aggregate_context_token_capacity /
        layer_plan->compression_ratio;
    if ((configuration->aggregate_context_token_capacity %
         layer_plan->compression_ratio) != 0u)
    {
        status = SparkDsv4CheckedAddU64(
            unrounded_entry_capacity,
            1u,
            &unrounded_entry_capacity);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
    }
    status = SparkDsv4RoundUpToMultipleU64(
        unrounded_entry_capacity,
        configuration->compressed_history_page_entries,
        &rounded_entry_capacity);
    if (status != SPARK_STATUS_OK || rounded_entry_capacity > UINT32_MAX)
    {
        return(status == SPARK_STATUS_OK ? SPARK_STATUS_CAPACITY_EXCEEDED : status);
    }
    layer_plan->compressed_entry_capacity = (uint32_t)rounded_entry_capacity;
    cursor = 0u;

    status = SparkDsv4CheckedMultiplyU64(
        rounded_entry_capacity,
        geometry->head_dimension - geometry->rope_dimension,
        &element_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4ElementsToPackedBytes(
        element_count,
        configuration->attention_content_element_bits,
        &region_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AppendAlignedRegion(
        region_bytes,
        configuration->allocation_alignment_bytes,
        &cursor,
        &layer_plan->compressed_content_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    status = SparkDsv4CheckedMultiplyU64(
        rounded_entry_capacity,
        geometry->rope_dimension,
        &element_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4ElementsToPackedBytes(
        element_count,
        configuration->attention_rope_element_bits,
        &region_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AppendAlignedRegion(
        region_bytes,
        configuration->allocation_alignment_bytes,
        &cursor,
        &layer_plan->compressed_rope_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    if (layer_plan->attention_class == SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE)
    {
        status = SparkDsv4CheckedMultiplyU64(
            rounded_entry_capacity,
            geometry->index_head_dimension,
            &element_count);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4ElementsToPackedBytes(
            element_count,
            configuration->indexer_element_bits,
            &region_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendAlignedRegion(
            region_bytes,
            configuration->allocation_alignment_bytes,
            &cursor,
            &layer_plan->indexer_history_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
    }
    return(SparkDsv4AlignUpU64(
        cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->compressed_history_arena_bytes));
}

static SparkStatus SparkDsv4AppendStateTensor(
    uint64_t token_count,
    uint64_t feature_count,
    const SparkDsv4CachePlanConfiguration *configuration,
    uint64_t *cursor,
    uint64_t *offset)
{
    SparkStatus status;
    uint64_t element_count;
    uint64_t region_bytes;

    status = SparkDsv4CheckedMultiplyU64(token_count, feature_count, &element_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4ElementsToPackedBytes(
        element_count,
        configuration->compressor_state_element_bits,
        &region_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    return(SparkDsv4AppendAlignedRegion(
        region_bytes,
        configuration->allocation_alignment_bytes,
        cursor,
        offset));
}

static SparkStatus SparkDsv4BuildCompressorStateLayout(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    SparkDsv4LayerCachePlan *layer_plan)
{
    SparkStatus status;
    uint64_t cursor;
    uint64_t compressor_projection_width;
    uint64_t indexer_projection_width;

    if (layer_plan->compression_ratio == 0u)
    {
        return(SPARK_STATUS_OK);
    }
    layer_plan->compressor_buffer_token_capacity_per_sequence =
        layer_plan->compression_ratio - 1u;
    cursor = 0u;

    compressor_projection_width = geometry->head_dimension;
    if (layer_plan->attention_class == SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE)
    {
        compressor_projection_width *= 2u;
    }
    status = SparkDsv4AppendStateTensor(
        layer_plan->compressor_buffer_token_capacity_per_sequence,
        compressor_projection_width,
        configuration,
        &cursor,
        &layer_plan->compressor_buffer_kv_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AppendStateTensor(
        layer_plan->compressor_buffer_token_capacity_per_sequence,
        compressor_projection_width,
        configuration,
        &cursor,
        &layer_plan->compressor_buffer_gate_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    if (layer_plan->attention_class == SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE)
    {
        layer_plan->indexer_buffer_token_capacity_per_sequence =
            layer_plan->compression_ratio - 1u;
        indexer_projection_width = 2u * (uint64_t)geometry->index_head_dimension;
        status = SparkDsv4AppendStateTensor(
            layer_plan->indexer_buffer_token_capacity_per_sequence,
            indexer_projection_width,
            configuration,
            &cursor,
            &layer_plan->indexer_buffer_kv_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendStateTensor(
            layer_plan->indexer_buffer_token_capacity_per_sequence,
            indexer_projection_width,
            configuration,
            &cursor,
            &layer_plan->indexer_buffer_gate_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendStateTensor(
            layer_plan->compression_ratio,
            geometry->head_dimension,
            configuration,
            &cursor,
            &layer_plan->compressor_overlap_kv_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendStateTensor(
            layer_plan->compression_ratio,
            geometry->head_dimension,
            configuration,
            &cursor,
            &layer_plan->compressor_overlap_gate_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendStateTensor(
            layer_plan->compression_ratio,
            geometry->index_head_dimension,
            configuration,
            &cursor,
            &layer_plan->indexer_overlap_kv_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        status = SparkDsv4AppendStateTensor(
            layer_plan->compression_ratio,
            geometry->index_head_dimension,
            configuration,
            &cursor,
            &layer_plan->indexer_overlap_gate_offset_bytes);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
    }
    return(SparkDsv4AlignUpU64(
        cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->compressor_state_sequence_stride_bytes));
}

static SparkStatus SparkDsv4BuildLayerPlanForClass(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    uint32_t absolute_layer_index,
    uint32_t is_mtp_layer,
    SparkDsv4AttentionClass attention_class,
    uint32_t compression_ratio,
    SparkDsv4LayerCachePlan *layer_plan)
{
    SparkStatus status;

    if (configuration == 0 || geometry == 0 || layer_plan == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(layer_plan, 0, sizeof(*layer_plan));
    layer_plan->absolute_layer_index = absolute_layer_index;
    layer_plan->is_mtp_layer = is_mtp_layer;
    layer_plan->attention_class = attention_class;
    layer_plan->compression_ratio = compression_ratio;

    status = SparkDsv4BuildSlidingLayout(configuration, geometry, layer_plan);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4BuildCompressedHistoryLayout(configuration, geometry, layer_plan);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    return(SparkDsv4BuildCompressorStateLayout(configuration, geometry, layer_plan));
}

static SparkStatus SparkDsv4BuildLayerPlan(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    uint32_t absolute_layer_index,
    uint32_t is_mtp_layer,
    SparkDsv4LayerCachePlan *layer_plan)
{
    SparkStatus status;
    uint32_t compression_ratio;
    SparkDsv4AttentionClass attention_class;

    compression_ratio = SparkDsv4CompressionRatioForLayer(
        configuration->model_variant,
        absolute_layer_index,
        is_mtp_layer);
    status = SparkDsv4AttentionClassFromCompressionRatio(
        compression_ratio,
        &attention_class);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    return(SparkDsv4BuildLayerPlanForClass(
        configuration,
        geometry,
        absolute_layer_index,
        is_mtp_layer,
        attention_class,
        compression_ratio,
        layer_plan));
}

static SparkStatus SparkDsv4BuildClassFootprint(
    const SparkDsv4CachePlanConfiguration *configuration,
    const SparkDsv4ModelCacheGeometry *geometry,
    SparkDsv4AttentionClass attention_class,
    uint32_t compression_ratio,
    SparkDsv4ClassFootprint *footprint)
{
    SparkDsv4LayerCachePlan layer_plan;
    SparkStatus status;

    if (footprint == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    status = SparkDsv4BuildLayerPlanForClass(
        configuration,
        geometry,
        0u,
        0u,
        attention_class,
        compression_ratio,
        &layer_plan);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    footprint->sliding_sequence_stride_bytes = layer_plan.sliding_sequence_stride_bytes;
    footprint->compressed_history_bytes = layer_plan.compressed_history_arena_bytes;
    footprint->compressor_state_sequence_stride_bytes =
        layer_plan.compressor_state_sequence_stride_bytes;
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkDsv4AppendLayerArenas(
    const SparkDsv4CachePlanConfiguration *configuration,
    SparkDsv4LayerCachePlan *layer_plan,
    uint64_t *sliding_cursor,
    uint64_t *history_cursor,
    uint64_t *state_cursor)
{
    SparkStatus status;
    uint64_t arena_bytes;

    status = SparkDsv4AlignUpU64(
        *sliding_cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->sliding_arena_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        layer_plan->sliding_sequence_stride_bytes,
        configuration->active_sequence_capacity,
        &arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    layer_plan->sliding_arena_bytes = arena_bytes;
    status = SparkDsv4CheckedAddU64(
        layer_plan->sliding_arena_offset_bytes,
        arena_bytes,
        sliding_cursor);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    status = SparkDsv4AlignUpU64(
        *history_cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->compressed_history_arena_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedAddU64(
        layer_plan->compressed_history_arena_offset_bytes,
        layer_plan->compressed_history_arena_bytes,
        history_cursor);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    status = SparkDsv4AlignUpU64(
        *state_cursor,
        configuration->allocation_alignment_bytes,
        &layer_plan->compressor_state_arena_offset_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        layer_plan->compressor_state_sequence_stride_bytes,
        configuration->active_sequence_capacity,
        &arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    layer_plan->compressor_state_arena_bytes = arena_bytes;
    return(SparkDsv4CheckedAddU64(
        layer_plan->compressor_state_arena_offset_bytes,
        arena_bytes,
        state_cursor));
}

SparkStatus SparkDsv4CachePlanBuild(
    const SparkDsv4CachePlanConfiguration *configuration,
    SparkDsv4CachePlan *plan)
{
    SparkDsv4ModelCacheGeometry geometry;
    SparkDsv4ClassFootprint csa_footprint;
    SparkDsv4ClassFootprint hca_footprint;
    SparkDsv4LayerCachePlan *layer_plan;
    SparkStatus status;
    uint32_t planned_layer_count;
    uint32_t planned_index;
    uint32_t absolute_layer_index;
    uint32_t is_mtp_layer;
    uint64_t sliding_cursor;
    uint64_t history_cursor;
    uint64_t state_cursor;
    uint64_t per_layer_bytes;

    if (plan == 0)
    {
        return(SPARK_STATUS_INVALID_ARGUMENT);
    }
    status = SparkDsv4ValidateConfiguration(
        configuration,
        &geometry,
        &planned_layer_count);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    memset(plan, 0, sizeof(*plan));
    plan->abi_version = SPARK_DSV4_CACHE_PLAN_ABI_VERSION;
    plan->descriptor_bytes = sizeof(*plan);
    plan->model_variant = configuration->model_variant;
    plan->planned_layer_count = planned_layer_count;
    plan->active_sequence_capacity = configuration->active_sequence_capacity;
    plan->maximum_context_tokens_per_sequence =
        configuration->maximum_context_tokens_per_sequence;
    plan->aggregate_context_token_capacity =
        configuration->aggregate_context_token_capacity;
    plan->compressed_history_page_entries =
        configuration->compressed_history_page_entries;
    plan->allocation_alignment_bytes = configuration->allocation_alignment_bytes;
    sliding_cursor = 0u;
    history_cursor = 0u;
    state_cursor = 0u;

    for (planned_index = 0u; planned_index < planned_layer_count; ++planned_index)
    {
        is_mtp_layer = planned_index >= configuration->backbone_layer_count;
        absolute_layer_index = is_mtp_layer != 0u
            ? geometry.backbone_layer_count
            : configuration->first_backbone_layer_index + planned_index;
        layer_plan = &plan->layers[planned_index];
        status = SparkDsv4BuildLayerPlan(
            configuration,
            &geometry,
            absolute_layer_index,
            is_mtp_layer,
            layer_plan);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
        switch (layer_plan->attention_class)
        {
            case SPARK_DSV4_ATTENTION_CLASS_SLIDING:
                plan->sliding_layer_count += 1u;
                break;
            case SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE:
                plan->compressed_sparse_layer_count += 1u;
                break;
            case SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED:
                plan->heavily_compressed_layer_count += 1u;
                break;
            default:
                return(SPARK_STATUS_INVALID_ARGUMENT);
        }
        status = SparkDsv4AppendLayerArenas(
            configuration,
            layer_plan,
            &sliding_cursor,
            &history_cursor,
            &state_cursor);
        if (status != SPARK_STATUS_OK)
        {
            return(status);
        }
    }

    status = SparkDsv4AlignUpU64(
        sliding_cursor,
        configuration->allocation_alignment_bytes,
        &plan->sliding_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AlignUpU64(
        history_cursor,
        configuration->allocation_alignment_bytes,
        &plan->compressed_history_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4AlignUpU64(
        state_cursor,
        configuration->allocation_alignment_bytes,
        &plan->compressor_state_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedAddU64(
        plan->sliding_arena_bytes,
        plan->compressed_history_arena_bytes,
        &plan->total_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedAddU64(
        plan->total_arena_bytes,
        plan->compressor_state_arena_bytes,
        &plan->total_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    status = SparkDsv4BuildClassFootprint(
        configuration,
        &geometry,
        SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE,
        4u,
        &csa_footprint);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4BuildClassFootprint(
        configuration,
        &geometry,
        SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED,
        128u,
        &hca_footprint);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }

    status = SparkDsv4CheckedMultiplyU64(
        csa_footprint.sliding_sequence_stride_bytes,
        configuration->active_sequence_capacity,
        &per_layer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        per_layer_bytes,
        planned_layer_count,
        &plan->worst_class_sliding_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        csa_footprint.compressed_history_bytes,
        planned_layer_count,
        &plan->worst_class_history_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        hca_footprint.compressor_state_sequence_stride_bytes,
        configuration->active_sequence_capacity,
        &per_layer_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedMultiplyU64(
        per_layer_bytes,
        planned_layer_count,
        &plan->worst_class_state_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    status = SparkDsv4CheckedAddU64(
        plan->worst_class_sliding_arena_bytes,
        plan->worst_class_history_arena_bytes,
        &plan->worst_class_total_arena_bytes);
    if (status != SPARK_STATUS_OK)
    {
        return(status);
    }
    return(SparkDsv4CheckedAddU64(
        plan->worst_class_total_arena_bytes,
        plan->worst_class_state_arena_bytes,
        &plan->worst_class_total_arena_bytes));
}
