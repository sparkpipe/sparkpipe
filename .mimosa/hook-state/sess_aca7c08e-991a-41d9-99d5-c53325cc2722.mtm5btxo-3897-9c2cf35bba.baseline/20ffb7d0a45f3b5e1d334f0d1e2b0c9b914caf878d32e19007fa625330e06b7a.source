#ifndef SPARKPIPE_SPARK_DSV4_CACHE_PLAN_H
#define SPARKPIPE_SPARK_DSV4_CACHE_PLAN_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_DSV4_CACHE_PLAN_ABI_VERSION 2u
#define SPARK_DSV4_CACHE_PLAN_MAXIMUM_LAYERS 63u
#define SPARK_DSV4_CACHE_PLAN_DEFAULT_ALIGNMENT_BYTES 256u
#define SPARK_DSV4_CACHE_PLAN_DEFAULT_PAGE_ENTRIES 512u

typedef enum SparkDsv4ModelVariant
{
    SPARK_DSV4_MODEL_VARIANT_INVALID = 0,
    SPARK_DSV4_MODEL_VARIANT_FLASH = 1,
    SPARK_DSV4_MODEL_VARIANT_PRO = 2
} SparkDsv4ModelVariant;

typedef enum SparkDsv4AttentionClass
{
    SPARK_DSV4_ATTENTION_CLASS_INVALID = 0,
    SPARK_DSV4_ATTENTION_CLASS_SLIDING = 1,
    SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE = 2,
    SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED = 3
} SparkDsv4AttentionClass;

typedef struct SparkDsv4CachePlanConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkDsv4ModelVariant model_variant;
    uint32_t first_backbone_layer_index;
    uint32_t backbone_layer_count;
    uint32_t include_mtp_layer;
    uint32_t active_sequence_capacity;
    uint32_t maximum_context_tokens_per_sequence;
    uint64_t aggregate_context_token_capacity;
    uint32_t compressed_history_page_entries;
    uint32_t attention_content_element_bits;
    uint32_t attention_rope_element_bits;
    uint32_t indexer_element_bits;
    uint32_t compressor_state_element_bits;
    uint32_t allocation_alignment_bytes;
    uint32_t reserved_u32[5];
} SparkDsv4CachePlanConfiguration;

typedef struct SparkDsv4LayerCachePlan
{
    uint32_t absolute_layer_index;
    uint32_t is_mtp_layer;
    SparkDsv4AttentionClass attention_class;
    uint32_t compression_ratio;
    uint32_t sliding_entry_capacity_per_sequence;
    uint32_t compressed_entry_capacity;
    uint32_t compressor_buffer_token_capacity_per_sequence;
    uint32_t indexer_buffer_token_capacity_per_sequence;

    uint64_t sliding_content_offset_bytes;
    uint64_t sliding_rope_offset_bytes;
    uint64_t sliding_sequence_stride_bytes;
    uint64_t sliding_arena_offset_bytes;
    uint64_t sliding_arena_bytes;

    uint64_t compressed_content_offset_bytes;
    uint64_t compressed_rope_offset_bytes;
    uint64_t indexer_history_offset_bytes;
    uint64_t compressed_history_arena_offset_bytes;
    uint64_t compressed_history_arena_bytes;

    uint64_t compressor_buffer_kv_offset_bytes;
    uint64_t compressor_buffer_gate_offset_bytes;
    uint64_t indexer_buffer_kv_offset_bytes;
    uint64_t indexer_buffer_gate_offset_bytes;
    uint64_t compressor_overlap_kv_offset_bytes;
    uint64_t compressor_overlap_gate_offset_bytes;
    uint64_t indexer_overlap_kv_offset_bytes;
    uint64_t indexer_overlap_gate_offset_bytes;
    uint64_t compressor_state_sequence_stride_bytes;
    uint64_t compressor_state_arena_offset_bytes;
    uint64_t compressor_state_arena_bytes;
} SparkDsv4LayerCachePlan;

typedef struct SparkDsv4CachePlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkDsv4ModelVariant model_variant;
    uint32_t planned_layer_count;
    uint32_t active_sequence_capacity;
    uint32_t maximum_context_tokens_per_sequence;
    uint64_t aggregate_context_token_capacity;
    uint32_t compressed_history_page_entries;
    uint32_t allocation_alignment_bytes;

    uint32_t sliding_layer_count;
    uint32_t compressed_sparse_layer_count;
    uint32_t heavily_compressed_layer_count;
    uint32_t reserved_u32;

    uint64_t sliding_arena_bytes;
    uint64_t compressed_history_arena_bytes;
    uint64_t compressor_state_arena_bytes;
    uint64_t total_arena_bytes;

    uint64_t worst_class_sliding_arena_bytes;
    uint64_t worst_class_history_arena_bytes;
    uint64_t worst_class_state_arena_bytes;
    uint64_t worst_class_total_arena_bytes;

    SparkDsv4LayerCachePlan layers[SPARK_DSV4_CACHE_PLAN_MAXIMUM_LAYERS];
} SparkDsv4CachePlan;

SparkStatus SparkDsv4CachePlanBuild(
    const SparkDsv4CachePlanConfiguration *configuration,
    SparkDsv4CachePlan *plan);

#ifdef __cplusplus
}
#endif

#endif
