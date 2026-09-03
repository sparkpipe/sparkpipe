#pragma once

#include <stdint.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_model_driver.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_PAGE_CACHE_ABI_VERSION 3u
#define SPARK_KV_PAGE_CACHE_NO_INDEX UINT32_MAX
#define SPARK_KV_PAGE_CACHE_ENTRY_FLAG_VALID UINT32_C(0x00000001)
#define SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE UINT32_C(0x00000001)
#define SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE UINT32_C(0x00000002)
#define SPARK_KV_PAGE_CACHE_KNOWN_MUTATIONS \
	(SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE | \
	 SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE)


typedef struct SparkKvPageCacheEntry
{
	uint32_t flags;
	uint32_t token_count;
	uint32_t page_count;
	uint32_t parent_entry_index;
	uint32_t logical_page_index;
	uint32_t reference_count;
	uint32_t hash_next;
	uint32_t free_next;
	uint64_t last_used_epoch;
	SparkModelDriverCacheIdentity identity;
}
SparkKvPageCacheEntry;

typedef struct SparkKvPageCacheSequence
{
	uint64_t sequence_id;
	uint64_t generation;
	uint32_t next_token_position;
	uint32_t terminal_entry_index;
	uint32_t mutable_logical_page_index;
	uint32_t mutable_first_token_index;
}
SparkKvPageCacheSequence;

typedef struct SparkKvPageCacheConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t sequence_capacity;
	uint32_t entry_capacity;
	uint32_t hash_bucket_count;
	uint32_t reserved0;
	SparkKvCacheArena *kv_cache_arena;
	SparkKvPageStore *page_store;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
}
SparkKvPageCacheConfiguration;

typedef struct SparkKvPageCache
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t sequence_capacity;
	uint32_t entry_capacity;
	uint32_t hash_bucket_count;
	uint32_t free_entry_head;
	uint32_t live_sequence_count;
	SparkKvCacheArena *kv_cache_arena;
	SparkKvPageStore *page_store;
	SparkKvPageCacheEntry *entries;
	SparkKvPageCacheSequence *sequences;
	uint32_t *hash_bucket_heads;
	uint32_t *entry_indices_by_logical_page;
	uint64_t epoch;
	uint64_t prefix_hit_count;
	uint64_t prefix_miss_count;
	uint64_t published_page_count;
	uint64_t deduplicated_page_count;
	uint64_t evicted_entry_count;
	uint64_t released_sequence_count;
}
SparkKvPageCache;

#define SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES \
	((uint32_t)sizeof(SparkKvPageCacheConfiguration))
#define SPARK_KV_PAGE_CACHE_BYTES \
	((uint32_t)sizeof(SparkKvPageCache))

SparkStatus SparkKvPageCacheInitialize(
	SparkKvPageCache *cache,
	const SparkKvPageCacheConfiguration *configuration);
SparkStatus SparkKvPageCachePrepareLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);
SparkStatus SparkKvPageCacheResolveLanePages(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);
SparkStatus SparkKvPageCacheGetLaneMutablePageDemand(
	const SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_page_demand_out);
SparkStatus SparkKvPageCacheBeginLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out);
SparkStatus SparkKvPageCacheBeginLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t *mutable_logical_page_index_out,
	uint32_t *mutation_flags_out);
SparkStatus SparkKvPageCacheRollbackLaneTransaction(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane,
	uint32_t mutation_flags);
SparkStatus SparkKvPageCacheCompleteLane(
	SparkKvPageCache *cache,
	const SparkModelDriverCacheLane *lane);
SparkStatus SparkKvPageCacheReleaseLane(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id);
SparkStatus SparkKvPageCacheBuildLaneTable(
	SparkKvPageCache *cache,
	uint32_t resident_sequence_slot,
	uint64_t sequence_id,
	uint32_t *logical_page_indices,
	uint32_t logical_page_capacity,
	uint32_t *logical_page_count_out);

#ifdef __cplusplus
}
#endif
