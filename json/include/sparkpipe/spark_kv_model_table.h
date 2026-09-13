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

/*
 * The single token-free fill point a model driver uses to describe its KV stack
 * to the common machinery. It bundles the three config structs the paged-cache
 * core already consumes - arena geometry, page-store backing (including the
 * model's copy primitive in page_store_config.copy_function), and the page
 * directory's capacities plus its caller-owned tables - together with the
 * identity strings the store key and backing path are derived from.
 *
 * Geometry and mechanics live here; nothing in this struct is model-specific.
 * The JIT_KV vs DRIVER_OWNS_KV capability only selects WHO calls
 * SparkKvBackendInitialize (the runtime resident vs the model driver), never
 * what a table may contain. See docs/PROPOSAL_KV_SEAM.md 1.2/1.3.
 */
typedef struct SparkKvModelTable
{
    uint32_t abi_version;              /* SPARK_KV_MODEL_TABLE_ABI_VERSION */
    uint32_t descriptor_bytes;         /* SPARK_KV_MODEL_TABLE_BYTES       */
    /* geometry -> sizing authority (SparkKvCacheEstimateCapacity) */
    SparkKvCacheCapacityRequest capacity_request;
    /* arena block + resident-slot geometry; the caller pre-sizes the arena and
       provides blocks[] and resident_slot_logical_block_indices[] */
    SparkKvCacheConfiguration arena_configuration;
    /* host backing + the model-specific copy primitive */
    SparkKvPageStoreConfiguration page_store_config;
    /* page directory: capacities + caller-owned tables */
    uint32_t sequence_capacity;
    uint32_t entry_capacity;
    uint32_t hash_bucket_count;
    SparkKvPageCacheEntry *entries;
    SparkKvPageCacheSequence *sequences;
    uint32_t *hash_bucket_heads;
    uint32_t *entry_indices_by_logical_page;
    /* identity: store-key + backing-path derivation (optional here; required
       by SparkKvPageStoreBuildPath and the stage store client) */
    const char *model_id;
    const char *model_revision;
    const char *cache_layout_fingerprint;
} SparkKvModelTable;

/* Table-level validation: ABI/descriptor bytes for the table and its three
 * nested configs, non-zero page-directory capacities, non-null caller tables,
 * and the entry_capacity <= arena.logical_block_count invariant. Deep geometry
 * checks stay in the three Initialize calls, which fail closed on their own. */
SparkStatus SparkKvModelTableValidate(const SparkKvModelTable *table);

/* Initialize the arena, page store, and page directory from one table, in the
 * order their internal cross-references require (arena, then page store, then
 * page directory). On a later failure the page store is destroyed; the arena is
 * plain memory and needs no teardown. */
SparkStatus SparkKvBackendInitialize(
    const SparkKvModelTable *table,
    SparkKvCacheArena *arena,
    SparkKvPageCache *page_cache,
    SparkKvPageStore *page_store);

#ifdef __cplusplus
}
#endif
