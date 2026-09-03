#include "sparkpipe/spark_kv_model_table.h"

#include <string.h>

SparkStatus SparkKvModelTableValidate(const SparkKvModelTable *table)
{
    if (table == 0 ||
        table->abi_version != SPARK_KV_MODEL_TABLE_ABI_VERSION ||
        table->descriptor_bytes != SPARK_KV_MODEL_TABLE_BYTES ||
        table->capacity_request.abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        table->capacity_request.descriptor_bytes !=
            SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES ||
        table->arena_configuration.abi_version != SPARK_KV_CACHE_ABI_VERSION ||
        table->arena_configuration.descriptor_bytes !=
            SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES ||
        table->page_store_config.abi_version != SPARK_KV_PAGE_STORE_ABI_VERSION ||
        table->page_store_config.descriptor_bytes !=
            SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (table->sequence_capacity == 0u ||
        table->entry_capacity == 0u ||
        table->hash_bucket_count == 0u ||
        table->entries == 0 ||
        table->sequences == 0 ||
        table->hash_bucket_heads == 0 ||
        table->entry_indices_by_logical_page == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (table->entry_capacity > table->arena_configuration.logical_block_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkKvBackendInitialize(
    const SparkKvModelTable *table,
    SparkKvCacheArena *arena,
    SparkKvPageCache *page_cache,
    SparkKvPageStore *page_store)
{
    SparkKvPageCacheConfiguration page_cache_configuration;
    SparkStatus status;

    status = SparkKvModelTableValidate(table);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (arena == 0 || page_cache == 0 || page_store == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    status = SparkKvCacheArenaInitialize(arena, &table->arena_configuration);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    status = SparkKvPageStoreInitialize(page_store, &table->page_store_config);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    memset(&page_cache_configuration, 0, sizeof(page_cache_configuration));
    page_cache_configuration.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
    page_cache_configuration.descriptor_bytes =
        SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
    page_cache_configuration.sequence_capacity = table->sequence_capacity;
    page_cache_configuration.entry_capacity = table->entry_capacity;
    page_cache_configuration.hash_bucket_count = table->hash_bucket_count;
    page_cache_configuration.kv_cache_arena = arena;
    page_cache_configuration.page_store = page_store;
    page_cache_configuration.entries = table->entries;
    page_cache_configuration.sequences = table->sequences;
    page_cache_configuration.hash_bucket_heads = table->hash_bucket_heads;
    page_cache_configuration.entry_indices_by_logical_page =
        table->entry_indices_by_logical_page;
    status = SparkKvPageCacheInitialize(page_cache, &page_cache_configuration);
    if (status != SPARK_STATUS_OK)
    {
        SparkKvPageStoreDestroy(page_store);
        return status;
    }
    return SPARK_STATUS_OK;
}
