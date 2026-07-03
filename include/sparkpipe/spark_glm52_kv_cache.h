#ifndef SPARKPIPE_SPARK_GLM52_KV_CACHE_H
#define SPARKPIPE_SPARK_GLM52_KV_CACHE_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_GLM52_KV_CACHE_ABI_VERSION 2u
#define SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheConfiguration))
#define SPARK_GLM52_KV_CACHE_ARENA_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheArena))
#define SPARK_GLM52_KV_CACHE_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheBlock))
#define SPARK_GLM52_KV_CACHE_BLOCK_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheBlockView))
#define SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvBlockTableView))

#define SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED 0x00000001u
#define SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT 0x00000002u

#define SPARK_GLM52_KV_CACHE_NO_BLOCK 0xffffffffu

#ifndef SPARK_GLM52_KV_CACHE_MAX_BLOCK_TOKENS
#define SPARK_GLM52_KV_CACHE_MAX_BLOCK_TOKENS 256u
#endif

#ifndef SPARK_GLM52_KV_CACHE_MAX_LAYER_COUNT
#define SPARK_GLM52_KV_CACHE_MAX_LAYER_COUNT 256u
#endif


#define SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCachePrefetchSourceBlock))
#define SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCachePrefetchBlock))
#define SPARK_GLM52_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCachePrefetchPlan))
#define SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT 13u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY 1024u

#define SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY 0x00000001u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE 0x00000002u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS \
    (SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY | \
     SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE)

#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION 1u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheAsyncPrefetchBackendConfiguration))
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheAsyncPrefetchBackend))
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_SOURCE_ENTRY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCachePrefetchBackendSourceEntry))
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkGlm52KvCacheAsyncPrefetchRequest))
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY 8u

#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE 0x00000001u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE 0x00000002u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS 0x00000004u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS 0x00000008u
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS \
    (SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS | \
     SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS)
#define SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS \
    (SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE | \
     SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE | \
     SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS)


typedef struct SparkGlm52KvCachePrefetchSourceBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t physical_block_index;
    uint32_t token_capacity;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t generation;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
} SparkGlm52KvCachePrefetchSourceBlock;

typedef struct SparkGlm52KvCachePrefetchBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_index;
    uint32_t physical_block_index;
    uint32_t token_capacity;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t flags;
    uint64_t generation;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
} SparkGlm52KvCachePrefetchBlock;

typedef struct SparkGlm52KvCachePrefetchPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_count;
    uint32_t requested_physical_block_count;
    uint32_t prefetch_block_count;
    uint32_t resident_block_count;
    uint32_t duplicate_block_count;
    uint32_t missing_block_count;
    uint32_t lane_block_counts[SPARK_GLM52_KV_CACHE_MAX_PREFETCH_LANE_COUNT];
    SparkGlm52KvCachePrefetchBlock blocks[
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
} SparkGlm52KvCachePrefetchPlan;

typedef struct SparkGlm52KvCachePrefetchBackendSourceEntry
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t physical_block_index;
    uint32_t reserved0;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t key_source_offset_bytes;
    uint64_t value_source_offset_bytes;
    const void *key_source_address;
    const void *value_source_address;
} SparkGlm52KvCachePrefetchBackendSourceEntry;

typedef struct SparkGlm52KvCacheAsyncPrefetchBackendConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t lane_count;
    uint32_t max_inflight_prefetch_count;
    uint32_t physical_block_count;
    uint32_t source_entry_count;
    uint32_t blocks_per_poll;
    uint64_t key_source_stride_bytes;
    uint64_t value_source_stride_bytes;
    uint64_t key_transfer_bytes;
    uint64_t value_transfer_bytes;
    const void *key_source_base;
    const void *value_source_base;
    const SparkGlm52KvCachePrefetchBackendSourceEntry *source_entries;
    int32_t key_file_descriptor;
    int32_t value_file_descriptor;
} SparkGlm52KvCacheAsyncPrefetchBackendConfiguration;

typedef struct SparkGlm52KvCacheAsyncPrefetchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t active;
    uint32_t completed_block_count;
    uint64_t prefetch_id;
    SparkStatus terminal_status;
    uint32_t reserved0;
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
} SparkGlm52KvCacheAsyncPrefetchRequest;

typedef struct SparkGlm52KvCacheAsyncPrefetchBackend
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t lane_count;
    uint32_t max_inflight_prefetch_count;
    uint32_t physical_block_count;
    uint32_t source_entry_count;
    uint32_t blocks_per_poll;
    uint64_t key_source_stride_bytes;
    uint64_t value_source_stride_bytes;
    uint64_t key_transfer_bytes;
    uint64_t value_transfer_bytes;
    const void *key_source_base;
    const void *value_source_base;
    const SparkGlm52KvCachePrefetchBackendSourceEntry *source_entries;
    int32_t key_file_descriptor;
    int32_t value_file_descriptor;
    uint64_t started_prefetch_count;
    uint64_t completed_prefetch_count;
    uint64_t failed_prefetch_count;
    uint64_t copied_key_block_count;
    uint64_t copied_value_block_count;
    uint64_t busy_poll_count;
    SparkGlm52KvCacheAsyncPrefetchRequest requests[
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY];
} SparkGlm52KvCacheAsyncPrefetchBackend;

typedef struct SparkGlm52KvCacheBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t physical_block_index;
    uint32_t token_capacity;
    uint32_t reference_count;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t generation;
    uint64_t last_used_epoch;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
} SparkGlm52KvCacheBlock;

typedef struct SparkGlm52KvCacheConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t physical_block_count;
    uint32_t block_token_count;
    uint32_t resident_block_capacity;
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t bytes_per_scalar;
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    void *key_device_base;
    void *value_device_base;
    SparkGlm52KvCacheBlock *blocks;
} SparkGlm52KvCacheConfiguration;

typedef struct SparkGlm52KvCacheBlockView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t physical_block_index;
    uint32_t token_capacity;
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t bytes_per_scalar;
    uint64_t generation;
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
} SparkGlm52KvCacheBlockView;

typedef struct SparkGlm52KvBlockTableView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t lane_count;
    uint32_t lane_stride;
    uint32_t lane_capacity;
    const uint32_t *physical_block_indices;
    const uint32_t *lane_physical_block_counts;
    const uint32_t *host_physical_block_indices;
    uint32_t reserved0;
    uint32_t reserved1;
} SparkGlm52KvBlockTableView;

typedef struct SparkGlm52KvCacheArena
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t physical_block_count;
    uint32_t block_token_count;
    uint32_t resident_block_capacity;
    uint32_t layer_count;
    uint32_t kv_head_count;
    uint32_t head_dim;
    uint32_t bytes_per_scalar;
    uint64_t key_block_stride_bytes;
    uint64_t value_block_stride_bytes;
    uintptr_t key_device_base;
    uintptr_t value_device_base;
    SparkGlm52KvCacheBlock *blocks;
    uint64_t epoch;
    uint64_t allocated_block_count;
    uint64_t recycled_block_count;
    uint64_t resident_block_count;
    uint64_t resident_evicted_block_count;
    uint64_t resident_capacity_stall_count;
    uint64_t retained_block_count;
    uint64_t released_reference_count;
} SparkGlm52KvCacheArena;

SparkStatus SparkGlm52KvCacheArenaInitialize(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCacheConfiguration *configuration);

SparkStatus SparkGlm52KvCacheArenaAcquireBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t *physical_block_index_out);

SparkStatus SparkGlm52KvCacheArenaRecycleBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);

SparkStatus SparkGlm52KvCacheArenaRetainBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);

SparkStatus SparkGlm52KvCacheArenaReleaseBlockReference(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);

SparkStatus SparkGlm52KvCacheArenaMarkBlockResident(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);

SparkStatus SparkGlm52KvCacheArenaFreeBlock(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);


SparkStatus SparkGlm52KvCacheArenaMarkBlockNonResident(
    SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index);

SparkStatus SparkGlm52KvCacheArenaBuildPrefetchPlan(
    const SparkGlm52KvCacheArena *arena,
    const uint32_t *physical_block_indices,
    uint32_t physical_block_count,
    uint32_t lane_count,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
    const SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_count,
    uint32_t lane_count,
    SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
    SparkGlm52KvCacheArena *arena,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count);

SparkStatus SparkGlm52KvCacheArenaTrimResidentBlocks(
    SparkGlm52KvCacheArena *arena,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out);

SparkStatus SparkGlm52KvCacheArenaEvictResidentBlocksToLimit(
    SparkGlm52KvCacheArena *arena,
    uint32_t max_resident_block_count,
    const uint32_t *protected_physical_block_indices,
    uint32_t protected_physical_block_count,
    uint32_t *evicted_block_count_out);

SparkStatus SparkGlm52KvCacheArenaResolveBlock(
    const SparkGlm52KvCacheArena *arena,
    uint32_t physical_block_index,
    SparkGlm52KvCacheBlockView *block_view);

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendInitialize(
    SparkGlm52KvCacheAsyncPrefetchBackend *backend,
    const SparkGlm52KvCacheAsyncPrefetchBackendConfiguration *configuration);

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendStart(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendPoll(
    void *context,
    uint64_t prefetch_id,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheAsyncPrefetchBackendSubmitSynchronous(
    void *context,
    const SparkGlm52KvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkGlm52KvCacheArenaReset(
    SparkGlm52KvCacheArena *arena);

#ifdef __cplusplus
}
#endif

#endif
