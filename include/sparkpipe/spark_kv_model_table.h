#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_MODEL_TABLE_ABI_VERSION 1u
#define SPARK_KV_MODEL_TABLE_BYTES ((uint32_t)sizeof(SparkKvModelTable))

typedef struct SparkKvModelTable
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    SparkKvCacheCapacityRequest capacity_request;
    SparkKvCacheConfiguration arena_configuration;
    SparkKvPageStoreConfiguration page_store_config;
    uint32_t sequence_capacity;
    uint32_t entry_capacity;
    uint32_t hash_bucket_count;
    SparkKvPageCacheEntry *entries;
    SparkKvPageCacheSequence *sequences;
    uint32_t *hash_bucket_heads;
    uint32_t *entry_indices_by_logical_page;
    const char *model_id;
    const char *model_revision;
    const char *cache_layout_fingerprint;
} SparkKvModelTable;

SparkStatus SparkKvModelTableValidate(const SparkKvModelTable *table);

SparkStatus SparkKvBackendInitialize(
    const SparkKvModelTable *table,
    SparkKvCacheArena *arena,
    SparkKvPageCache *page_cache,
    SparkKvPageStore *page_store);

#ifdef __cplusplus
}
#endif
