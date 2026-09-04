#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "spark_dsv4_pool_layout.h"

#define SPARK_DSV4_PAGED_CACHE_NO_PAGE UINT32_MAX

typedef struct SparkDsv4PagedCacheConfiguration
{
	uint32_t first_layer_index;
	uint32_t layer_count;
	uint32_t resident_sequence_capacity;
	uint32_t maximum_sequence_positions;
	uint32_t logical_page_capacity;
	uint32_t physical_page_capacity;
}
SparkDsv4PagedCacheConfiguration;

typedef struct SparkDsv4PagedCacheLane
{
	uint32_t logical_page_count;
	uint32_t mutable_logical_page;
	uint32_t mutable_physical_page;
	uint32_t parent_physical_page;
	uint64_t mutable_generation;
	uint32_t requires_initialization;
	uint32_t mutation_flags;
}
SparkDsv4PagedCacheLane;

typedef struct SparkDsv4PagedCache
{
	SparkDsv4PagedPoolLayout layout;
	SparkKvCacheArena arena;
	SparkKvPageCache page_cache;
	SparkKvCacheBlock *blocks;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *resident_page_owners;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
	uint64_t *physical_page_generations;
	uint32_t *physical_page_content_logical_pages;
	uint32_t *host_device_page_table;
	void *device_page_pool;
	uint32_t *device_page_table;
	uint32_t logical_page_capacity;
	uint32_t physical_page_capacity;
	uint32_t lane_page_capacity;
}
SparkDsv4PagedCache;

SparkStatus SparkDsv4PagedCacheInitialize(
	SparkDsv4PagedCache *cache,
	const SparkDsv4PagedCacheConfiguration *configuration,
	SparkStageModuleLedger *ledger);
void SparkDsv4PagedCacheDestroyHost(SparkDsv4PagedCache *cache);
SparkStatus SparkDsv4PagedCachePrepareLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t *physical_pages,
	uint32_t page_capacity,
	SparkDsv4PagedCacheLane *prepared_lane);
SparkStatus SparkDsv4PagedCachePrefetchLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_pages,
	uint32_t page_capacity);
SparkStatus SparkDsv4PagedCachePinLane(
	SparkDsv4PagedCache *cache,
	const uint32_t *logical_pages,
	uint32_t logical_page_count);
SparkStatus SparkDsv4PagedCacheUnpinLane(
	SparkDsv4PagedCache *cache,
	const uint32_t *logical_pages,
	uint32_t logical_page_count);
SparkStatus SparkDsv4PagedCacheCompleteLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	const SparkDsv4PagedCacheLane *prepared_lane);
SparkStatus SparkDsv4PagedCacheAbortLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane);
SparkStatus SparkDsv4PagedCacheRollbackLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane,
	const SparkDsv4PagedCacheLane *prepared_lane);
SparkStatus SparkDsv4PagedCacheReleaseLane(
	SparkDsv4PagedCache *cache,
	const SparkModelDriverCacheLane *lane);
