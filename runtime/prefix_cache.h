 




















#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PREFIX_CACHE_CORE_ABI_VERSION 1u
#define SPARK_PREFIX_CACHE_CORE_CONFIGURATION_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreConfiguration))
#define SPARK_PREFIX_CACHE_CORE_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCore))
#define SPARK_PREFIX_CACHE_CORE_STATS_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreStats))
#define SPARK_PREFIX_CACHE_CORE_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES \
	((uint32_t)sizeof(SparkPrefixCacheCoreBlockTableView))

#define SPARK_PREFIX_CACHE_CORE_MAX_BLOCK_TOKENS 256u
#define SPARK_PREFIX_CACHE_CORE_NO_BLOCK 0xffffffffu

#define SPARK_PREFIX_CACHE_CORE_BLOCK_FREE 0u
#define SPARK_PREFIX_CACHE_CORE_BLOCK_PRIVATE 1u
#define SPARK_PREFIX_CACHE_CORE_BLOCK_PUBLISHED 2u

typedef struct SparkPrefixCacheCoreConfiguration
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	 


	uint32_t block_token_count;
	uint32_t reserved0;
	uint64_t block_stride_bytes;
	 
	uint32_t block_count;
	uint32_t max_sequence_count;
	uint32_t sequence_block_capacity;
	 
	uint32_t hash_bucket_count;
	uint32_t reserved1;
} SparkPrefixCacheCoreConfiguration;

typedef struct SparkPrefixCacheCoreStats
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t live_sequence_count;
	uint32_t used_block_count;
	uint32_t published_block_count;
	uint32_t free_block_count;
	uint64_t admit_count;
	uint64_t matched_block_count;
	uint64_t missed_block_count;
	uint64_t appended_token_count;
	uint64_t published_block_count_total;
	uint64_t evicted_block_count;
	uint64_t capacity_stall_count;
	uint64_t bytes_reused;
	uint64_t ticks;
} SparkPrefixCacheCoreStats;

 







typedef struct SparkPrefixCacheCoreBlockTableView
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t lane_count;
	uint32_t lane_stride;
	const uint32_t *device_physical_block_indices;
	const uint32_t *device_lane_block_counts;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_block_counts;
} SparkPrefixCacheCoreBlockTableView;

typedef struct SparkPrefixCacheCoreBlock
{
	uint32_t state;
	uint32_t reference_count;
	uint32_t token_count;
	uint32_t hash_next;
	uint32_t free_next;
	uint32_t lru_prev;
	uint32_t lru_next;
	uint64_t chain_hash;
	uint64_t last_used_tick;
	uint32_t *token_ids;
} SparkPrefixCacheCoreBlock;

typedef struct SparkPrefixCacheCoreSequence
{
	uint64_t sequence_id;
	uint32_t used;
	uint32_t block_count;
	uint64_t running_chain_hash;
	uint32_t *blocks;
} SparkPrefixCacheCoreSequence;

typedef struct SparkPrefixCacheCore
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	uint32_t block_token_count;
	uint32_t reserved0;
	uint64_t block_stride_bytes;
	uint32_t block_count;
	uint32_t max_sequence_count;
	uint32_t sequence_block_capacity;
	uint32_t hash_bucket_mask;
	SparkPrefixCacheCoreBlock *blocks;
	uint32_t *block_tokens;
	SparkPrefixCacheCoreSequence *sequences;
	uint32_t *sequence_blocks;
	uint32_t *hash_buckets;
	uint32_t free_block_head;
	uint32_t live_sequence_count;
	uint32_t lru_head;
	uint32_t lru_tail;
	uint64_t tick;
	uint64_t admit_count;
	uint64_t matched_block_count;
	uint64_t missed_block_count;
	uint64_t appended_token_count;
	uint64_t published_block_total;
	uint64_t evicted_block_count;
	uint64_t capacity_stall_count;
	 







	uint32_t *id_slots;
	uint64_t *id_keys;
	uint32_t id_bucket_mask;
} SparkPrefixCacheCore;

 




















SparkStatus SparkPrefixCacheCoreValidateGeometry(
    const SparkPrefixCacheCoreConfiguration *configuration);

SparkStatus SparkPrefixCacheCoreInitialize(
    SparkPrefixCacheCore *core,
    const SparkPrefixCacheCoreConfiguration *configuration);

void SparkPrefixCacheCoreDestroy(SparkPrefixCacheCore *core);

 






SparkStatus SparkPrefixCacheCoreAdmitSequence(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count,
    uint32_t *matched_token_count_out);

 

SparkStatus SparkPrefixCacheCoreAppendTokens(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    const uint32_t *token_ids,
    uint32_t token_count);

 

SparkStatus SparkPrefixCacheCoreBuildBlockTable(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id,
    uint32_t token_count,
    uint32_t *block_indices_out,
    uint32_t block_index_capacity,
    uint32_t *block_count_out);

uint32_t SparkPrefixCacheCoreSequenceTokenCount(
    const SparkPrefixCacheCore *core,
    uint64_t sequence_id);

 




SparkStatus SparkPrefixCacheCoreReleaseSequence(
    SparkPrefixCacheCore *core,
    uint64_t sequence_id);

 

SparkStatus SparkPrefixCacheCoreTrim(
    SparkPrefixCacheCore *core,
    uint32_t free_target,
    uint32_t *evicted_block_count_out);

void SparkPrefixCacheCoreQueryStats(
    const SparkPrefixCacheCore *core,
    SparkPrefixCacheCoreStats *stats);

#ifdef __cplusplus
}
#endif
