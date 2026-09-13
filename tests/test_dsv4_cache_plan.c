#include "sparkpipe/spark_dsv4_cache_arena.h"
#include "sparkpipe/spark_dsv4_cache_plan.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define SPARK_TEST_CONTEXT_TOKEN_CAPACITY 1048576u
#define SPARK_TEST_ACTIVE_SEQUENCE_CAPACITY 8u

static void SparkTestInitializeConfiguration(
    SparkDsv4CachePlanConfiguration *configuration,
    SparkDsv4ModelVariant model_variant)
{
    memset(configuration, 0, sizeof(*configuration));
    configuration->abi_version = SPARK_DSV4_CACHE_PLAN_ABI_VERSION;
    configuration->descriptor_bytes = sizeof(*configuration);
    configuration->model_variant = model_variant;
    configuration->active_sequence_capacity =
        SPARK_TEST_ACTIVE_SEQUENCE_CAPACITY;
    configuration->maximum_context_tokens_per_sequence =
        SPARK_TEST_CONTEXT_TOKEN_CAPACITY;
    configuration->aggregate_context_token_capacity =
        SPARK_TEST_CONTEXT_TOKEN_CAPACITY;
    configuration->compressed_history_page_entries =
        SPARK_DSV4_CACHE_PLAN_DEFAULT_PAGE_ENTRIES;
    configuration->attention_content_element_bits = 16u;
    configuration->attention_rope_element_bits = 16u;
    configuration->indexer_element_bits = 8u;
    configuration->compressor_state_element_bits = 16u;
    configuration->allocation_alignment_bytes =
        SPARK_DSV4_CACHE_PLAN_DEFAULT_ALIGNMENT_BYTES;
}

static const SparkDsv4LayerCachePlan *SparkTestFindAttentionClass(
    const SparkDsv4CachePlan *plan,
    SparkDsv4AttentionClass attention_class)
{
    uint32_t layer_index;

    for (layer_index = 0u;
         layer_index < plan->planned_layer_count;
         ++layer_index)
    {
        if (plan->layers[layer_index].attention_class == attention_class)
        {
            return(&plan->layers[layer_index]);
        }
    }
    return(0);
}

static void SparkTestValidateArenaPlacement(const SparkDsv4CachePlan *plan)
{
    uint32_t layer_index;
    uint64_t sliding_end;
    uint64_t history_end;
    uint64_t state_end;

    sliding_end = 0u;
    history_end = 0u;
    state_end = 0u;
    for (layer_index = 0u;
         layer_index < plan->planned_layer_count;
         ++layer_index)
    {
        const SparkDsv4LayerCachePlan *layer_plan;

        layer_plan = &plan->layers[layer_index];
        assert(layer_plan->sliding_arena_offset_bytes >= sliding_end);
        assert(layer_plan->compressed_history_arena_offset_bytes >= history_end);
        assert(layer_plan->compressor_state_arena_offset_bytes >= state_end);
        sliding_end = layer_plan->sliding_arena_offset_bytes +
            layer_plan->sliding_arena_bytes;
        history_end = layer_plan->compressed_history_arena_offset_bytes +
            layer_plan->compressed_history_arena_bytes;
        state_end = layer_plan->compressor_state_arena_offset_bytes +
            layer_plan->compressor_state_arena_bytes;
    }
    assert(sliding_end <= plan->sliding_arena_bytes);
    assert(history_end <= plan->compressed_history_arena_bytes);
    assert(state_end <= plan->compressor_state_arena_bytes);
}

static void SparkTestFlashUsesExactAttentionClassReservations(void)
{
    SparkDsv4CachePlanConfiguration configuration;
    SparkDsv4CachePlan plan;
    const SparkDsv4LayerCachePlan *sliding_plan;
    const SparkDsv4LayerCachePlan *csa_plan;
    const SparkDsv4LayerCachePlan *hca_plan;

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.backbone_layer_count = 43u;
    configuration.include_mtp_layer = 0u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) == SPARK_STATUS_OK);
    assert(plan.planned_layer_count == 43u);
    assert(plan.sliding_layer_count == 2u);
    assert(plan.compressed_sparse_layer_count == 21u);
    assert(plan.heavily_compressed_layer_count == 20u);
    assert(plan.total_arena_bytes < plan.worst_class_total_arena_bytes);

    sliding_plan = SparkTestFindAttentionClass(
        &plan,
        SPARK_DSV4_ATTENTION_CLASS_SLIDING);
    csa_plan = SparkTestFindAttentionClass(
        &plan,
        SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE);
    hca_plan = SparkTestFindAttentionClass(
        &plan,
        SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED);
    assert(sliding_plan != 0);
    assert(csa_plan != 0);
    assert(hca_plan != 0);

    assert(sliding_plan->compression_ratio == 0u);
    assert(sliding_plan->compressed_entry_capacity == 0u);
    assert(sliding_plan->compressor_state_sequence_stride_bytes == 0u);

    assert(csa_plan->compression_ratio == 4u);
    assert(csa_plan->compressed_entry_capacity == 262144u);
    assert(csa_plan->compressor_buffer_token_capacity_per_sequence == 3u);
    assert(csa_plan->indexer_buffer_token_capacity_per_sequence == 3u);
    assert(csa_plan->indexer_history_offset_bytes !=
        csa_plan->compressed_rope_offset_bytes);

    assert(hca_plan->compression_ratio == 128u);
    assert(hca_plan->compressed_entry_capacity == 8192u);
    assert(hca_plan->compressor_buffer_token_capacity_per_sequence == 127u);
    assert(hca_plan->indexer_buffer_token_capacity_per_sequence == 0u);
    assert(hca_plan->indexer_history_offset_bytes == 0u);
    assert(hca_plan->compressed_history_arena_bytes <
        csa_plan->compressed_history_arena_bytes);
    assert(csa_plan->compressor_state_sequence_stride_bytes <
        hca_plan->compressor_state_sequence_stride_bytes);

    SparkTestValidateArenaPlacement(&plan);
    configuration.include_mtp_layer = 1u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestProUsesItsOwnLayerSchedule(void)
{
    SparkDsv4CachePlanConfiguration configuration;
    SparkDsv4CachePlan plan;

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_PRO);
    configuration.backbone_layer_count = 61u;
    configuration.include_mtp_layer = 1u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) == SPARK_STATUS_OK);
    assert(plan.planned_layer_count == 62u);
    assert(plan.sliding_layer_count == 1u);
    assert(plan.compressed_sparse_layer_count == 30u);
    assert(plan.heavily_compressed_layer_count == 31u);
    assert(plan.layers[0u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED);
    assert(plan.layers[1u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED);
    assert(plan.layers[2u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE);
    assert(plan.layers[61u].is_mtp_layer == 1u);
    assert(plan.layers[61u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_SLIDING);
    assert(plan.total_arena_bytes < plan.worst_class_total_arena_bytes);
    SparkTestValidateArenaPlacement(&plan);
}

static void SparkTestAggregateHistoryUsesCeilingDivision(void)
{
    SparkDsv4CachePlanConfiguration configuration;
    SparkDsv4CachePlan plan;

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.first_backbone_layer_index = 2u;
    configuration.backbone_layer_count = 2u;
    configuration.active_sequence_capacity = 2u;
    configuration.maximum_context_tokens_per_sequence = 1025u;
    configuration.aggregate_context_token_capacity = 1025u;
    configuration.compressed_history_page_entries = 1u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) == SPARK_STATUS_OK);
    assert(plan.planned_layer_count == 2u);
    assert(plan.layers[0u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_COMPRESSED_SPARSE);
    assert(plan.layers[1u].attention_class ==
        SPARK_DSV4_ATTENTION_CLASS_HEAVILY_COMPRESSED);
    assert(plan.layers[0u].compressed_entry_capacity == 257u);
    assert(plan.layers[1u].compressed_entry_capacity == 9u);
}


static void SparkTestExactArenaAllocationUsesPlanBytes(void)
{
    SparkDsv4CachePlanConfiguration configuration;
    SparkDsv4CacheArena arena;
    SparkDsv4LayerCacheArenaView view;
    SparkStageModuleLedger ledger;
    uint32_t layer_index;

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.first_backbone_layer_index = 0u;
    configuration.backbone_layer_count = 4u;
    configuration.include_mtp_layer = 0u;
    configuration.active_sequence_capacity = 2u;
    configuration.maximum_context_tokens_per_sequence = 1025u;
    configuration.aggregate_context_token_capacity = 1025u;
    configuration.compressed_history_page_entries = 1u;
    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "dsv4_cache_arena_test";
    memset(&arena, 0, sizeof(arena));
    assert(SparkDsv4CacheArenaAllocate(&configuration, &ledger, &arena) ==
        SPARK_STATUS_OK);
    assert(ledger.device_allocation_count == 3u);
    assert(ledger.device_bytes_resident == arena.plan.total_arena_bytes);
    assert(ledger.device_bytes_resident <
        arena.plan.worst_class_total_arena_bytes);
    for (layer_index = 0u;
         layer_index < arena.plan.planned_layer_count;
         ++layer_index)
    {
        assert(SparkDsv4CacheArenaLayerView(&arena, layer_index, &view) ==
            SPARK_STATUS_OK);
        assert(view.plan == &arena.plan.layers[layer_index]);
        if (view.plan->sliding_arena_bytes != 0u)
        {
            assert(view.sliding_arena != 0);
        }
        if (view.plan->compressed_history_arena_bytes != 0u)
        {
            assert(view.compressed_history_arena != 0);
        }
        if (view.plan->compressor_state_arena_bytes != 0u)
        {
            assert(view.compressor_state_arena != 0);
        }
    }
    SparkStageModuleLedgerRelease(&ledger);
    assert(ledger.device_allocation_count == 0u);
    assert(ledger.device_bytes_resident == 0u);
}

static void SparkTestInvalidConfigurationFailsClosed(void)
{
    SparkDsv4CachePlanConfiguration configuration;
    SparkDsv4CachePlan plan;

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.backbone_layer_count = 43u;
    configuration.allocation_alignment_bytes = 192u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.backbone_layer_count = 44u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) ==
        SPARK_STATUS_INVALID_ARGUMENT);

    SparkTestInitializeConfiguration(
        &configuration,
        SPARK_DSV4_MODEL_VARIANT_FLASH);
    configuration.backbone_layer_count = 43u;
    configuration.reserved_u32[0u] = 1u;
    assert(SparkDsv4CachePlanBuild(&configuration, &plan) ==
        SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestFlashUsesExactAttentionClassReservations();
    SparkTestProUsesItsOwnLayerSchedule();
    SparkTestAggregateHistoryUsesCeilingDivision();
    SparkTestExactArenaAllocationUsesPlanBytes();
    SparkTestInvalidConfigurationFailsClosed();
    printf("test_dsv4_cache_plan PASS\n");
    return(0);
}
