#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_CACHE_ABI_VERSION 6u
#define SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheConfiguration))
#define SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheArena))
#define SPARK_KV_CACHE_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheBlock))
#define SPARK_KV_CACHE_BLOCK_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheBlockView))
#define SPARK_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvBlockTableView))

#define SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED 0x00000001u
#define SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT 0x00000002u
#define SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED 0x00000004u
#define SPARK_KV_CACHE_BLOCK_FLAG_DIRTY 0x00000008u
#define SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID 0x00000010u

#define SPARK_KV_CACHE_NO_BLOCK 0xffffffffu
#define SPARK_KV_CACHE_NO_RESIDENT_SLOT 0xffffffffu

#define SPARK_KV_CACHE_MAX_BLOCK_TOKENS 256u

#define SPARK_KV_CACHE_MAX_LAYER_COUNT 256u

#define SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE 1u
#define SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE 2u
#define SPARK_KV_CACHE_LAYOUT_COMPRESSED_KEY_VALUE_FP8_E4M3 3u
#define SPARK_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3 4u

#define SPARK_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheCapacityRequest))
#define SPARK_KV_CACHE_CAPACITY_ESTIMATE_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheCapacityEstimate))
#define SPARK_KV_JIT_STAGE_BUDGET_ABI_VERSION 3u
#define SPARK_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvJitStageBudgetRequest))
#define SPARK_KV_JIT_STAGE_BUDGET_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvJitStageBudget))
#define SPARK_KV_JIT_DEFAULT_RECORD_ALIGNMENT 4096u

typedef struct SparkKvCacheCapacityRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t layout;
    uint32_t context_token_count;
    uint32_t block_token_count;
    uint32_t layer_count;
    uint32_t head_count;
    uint32_t query_key_head_dimension;
    uint32_t value_head_dimension;
    uint32_t compressed_dimension;
    uint32_t position_dimension;
    uint32_t bytes_per_scalar;
    uint32_t fp8_scale_block_size;
    uint32_t index_key_layer_count;
    uint32_t index_key_dimension;
    uint32_t index_key_bytes_per_scalar;
    uint64_t cache_bytes_per_rank;
} SparkKvCacheCapacityRequest;

typedef struct SparkKvCacheCapacityEstimate
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t layout;
    uint32_t context_token_count;
    uint32_t block_token_count;
    uint32_t layer_count;
    uint32_t block_count_per_context;
    uint32_t contexts_per_rank;
    uint32_t reserved0;
    uint64_t attention_bytes_per_token_per_layer;
    uint64_t index_key_bytes_per_token;
    uint64_t bytes_per_block_per_layer;
    uint64_t bytes_per_context_per_rank;
    uint64_t unused_cache_bytes_per_rank;
} SparkKvCacheCapacityEstimate;

typedef struct SparkKvJitStageBudgetRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t physical_pool_token_capacity;
    uint32_t backing_block_capacity;
    uint32_t active_sequence_count;
    uint32_t backing_request_count;
    uint32_t selected_token_count;
    uint32_t include_auxiliary_layer;
    uint32_t block_token_count;
    uint32_t record_alignment_bytes;
    uint32_t attention_cache_layout;
    uint32_t fp8_scale_block_size;
    uint32_t head_count;
    uint32_t query_key_head_dimension;
    uint32_t value_head_dimension;
    uint32_t compressed_dimension;
    uint32_t position_dimension;
    uint32_t bytes_per_scalar;
    uint32_t index_key_layer_count;
    uint32_t index_key_dimension;
    uint32_t index_key_bytes_per_scalar;
} SparkKvJitStageBudgetRequest;

typedef struct SparkKvJitStageBudget
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t first_layer_index;
    uint32_t layer_count;
    uint32_t local_index_key_layer_count;
    uint32_t include_auxiliary_layer;
    uint32_t logical_block_capacity;
    uint32_t backing_block_capacity;
    uint32_t maximum_average_active_context_tokens;
    uint32_t maximum_average_backing_context_tokens;
    uint64_t attention_bytes_per_token;
    uint64_t index_key_bytes_per_token;
    uint64_t auxiliary_bytes_per_token;
    uint64_t resident_bytes_per_token;
    uint64_t resident_summary_bytes;
    uint64_t resident_pool_bytes;
    uint64_t nvme_payload_bytes_per_block;
    uint64_t nvme_record_bytes;
    uint64_t nvme_capacity_bytes;
    uint64_t selected_working_set_bytes;
} SparkKvJitStageBudget;


#define SPARK_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCachePrefetchSourceBlock))
#define SPARK_KV_CACHE_PREFETCH_BLOCK_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCachePrefetchBlock))
#define SPARK_KV_CACHE_PREFETCH_PLAN_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCachePrefetchPlan))
#define SPARK_KV_CACHE_PREFETCH_CURSOR_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCachePrefetchCursor))
#define SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT 1024u
#define SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY 4096u

#define SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY 0x00000001u
#define SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE 0x00000002u
#define SPARK_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS \
    (SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY | \
     SPARK_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE)

#define SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION 1u
#define SPARK_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheAsyncPrefetchBackendConfiguration))
#define SPARK_KV_CACHE_PREFETCH_BACKEND_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheAsyncPrefetchBackend))
#define SPARK_KV_CACHE_PREFETCH_BACKEND_SOURCE_ENTRY_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCachePrefetchBackendSourceEntry))
#define SPARK_KV_CACHE_PREFETCH_BACKEND_REQUEST_DESCRIPTOR_BYTES \
    ((uint32_t)sizeof(SparkKvCacheAsyncPrefetchRequest))
#define SPARK_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY 8u

#define SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE 0x00000001u
#define SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE 0x00000002u
#define SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS 0x00000004u
#define SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS 0x00000008u
#define SPARK_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS \
    (SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS | \
     SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_VALUE_BLOCKS)
#define SPARK_KV_CACHE_PREFETCH_BACKEND_KNOWN_FLAGS \
    (SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE | \
     SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_POSIX_FD_SOURCE | \
     SPARK_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS)


typedef struct SparkKvCachePrefetchSourceBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t token_capacity;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t generation;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
} SparkKvCachePrefetchSourceBlock;

typedef struct SparkKvCachePrefetchBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_index;
    uint32_t logical_block_index;
    uint32_t resident_slot_index;
    uint32_t token_capacity;
    uint32_t first_token_index;
    uint32_t token_count;
    uint32_t flags;
    uint32_t reserved0;
    uint64_t generation;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
} SparkKvCachePrefetchBlock;

typedef struct SparkKvCachePrefetchPlan
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t lane_count;
    uint32_t requested_logical_block_count;
    uint32_t prefetch_block_count;
    uint32_t resident_block_count;
    uint32_t reserved_block_count;
    uint32_t duplicate_block_count;
    uint32_t missing_block_count;
    uint32_t lane_block_counts[SPARK_KV_CACHE_MAX_PREFETCH_LANE_COUNT];
    SparkKvCachePrefetchBlock blocks[
        SPARK_KV_CACHE_PREFETCH_BLOCK_CAPACITY];
} SparkKvCachePrefetchPlan;

typedef struct SparkKvCachePrefetchCursor
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_count;
    uint32_t next_logical_block_index;
} SparkKvCachePrefetchCursor;

typedef struct SparkKvCachePrefetchBackendSourceEntry
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t reserved0;
    uint64_t parent_hash;
    uint64_t block_hash;
    uint64_t content_hash;
    uint64_t key_source_offset_bytes;
    uint64_t value_source_offset_bytes;
    const void *key_source_address;
    const void *value_source_address;
} SparkKvCachePrefetchBackendSourceEntry;

typedef struct SparkKvCacheAsyncPrefetchBackendConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t lane_count;
    uint32_t max_inflight_prefetch_count;
    uint32_t logical_block_count;
    uint32_t source_entry_count;
    uint32_t blocks_per_poll;
    uint64_t key_source_stride_bytes;
    uint64_t value_source_stride_bytes;
    uint64_t key_transfer_bytes;
    uint64_t value_transfer_bytes;
    const void *key_source_base;
    const void *value_source_base;
    const SparkKvCachePrefetchBackendSourceEntry *source_entries;
    int32_t key_file_descriptor;
    int32_t value_file_descriptor;
} SparkKvCacheAsyncPrefetchBackendConfiguration;

typedef struct SparkKvCacheAsyncPrefetchRequest
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t active;
    uint32_t completed_block_count;
    uint64_t prefetch_id;
    SparkStatus terminal_status;
    uint32_t reserved0;
    SparkKvCachePrefetchPlan prefetch_plan;
} SparkKvCacheAsyncPrefetchRequest;

typedef struct SparkKvCacheAsyncPrefetchBackend
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t lane_count;
    uint32_t max_inflight_prefetch_count;
    uint32_t logical_block_count;
    uint32_t source_entry_count;
    uint32_t blocks_per_poll;
    uint64_t key_source_stride_bytes;
    uint64_t value_source_stride_bytes;
    uint64_t key_transfer_bytes;
    uint64_t value_transfer_bytes;
    const void *key_source_base;
    const void *value_source_base;
    const SparkKvCachePrefetchBackendSourceEntry *source_entries;
    int32_t key_file_descriptor;
    int32_t value_file_descriptor;
    uint64_t started_prefetch_count;
    uint64_t completed_prefetch_count;
    uint64_t failed_prefetch_count;
    uint64_t copied_key_block_count;
    uint64_t copied_value_block_count;
    uint64_t busy_poll_count;
    SparkKvCacheAsyncPrefetchRequest requests[
        SPARK_KV_CACHE_PREFETCH_BACKEND_INFLIGHT_CAPACITY];
} SparkKvCacheAsyncPrefetchBackend;

typedef struct SparkKvCacheBlock
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t flags;
    uint32_t logical_block_index;
    uint32_t token_capacity;
    uint32_t reference_count;
    uint32_t resident_slot_index;
    uint32_t residency_reference_count;
    uint32_t free_next;
    uint32_t reserved0;
    uint64_t generation;
    uint64_t last_used_epoch;
    uintptr_t key_device_address;
    uintptr_t value_device_address;
} SparkKvCacheBlock;

typedef SparkStatus (*SparkKvCacheEvictFunction)(
    void *context,
    uint32_t logical_block_index,
    uint32_t resident_slot_index,
    uint64_t generation,
    uintptr_t key_device_address,
    uint64_t key_bytes,
    uintptr_t value_device_address,
    uint64_t value_bytes);

typedef struct SparkKvCacheConfiguration
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_count;
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
    SparkKvCacheBlock *blocks;
    uint32_t *resident_slot_logical_block_indices;
    SparkKvCacheEvictFunction evict_function;
    void *evict_context;
} SparkKvCacheConfiguration;

typedef struct SparkKvCacheBlockView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_index;
    uint32_t resident_slot_index;
    uint32_t flags;
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
} SparkKvCacheBlockView;

typedef struct SparkKvBlockTableView
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t block_token_count;
    uint32_t lane_count;
    uint32_t lane_stride;
    uint32_t lane_capacity;
    const uint32_t *logical_block_indices;
    const uint32_t *resident_slot_indices;
    const uint32_t *lane_logical_block_counts;
    const uint32_t *host_logical_block_indices;
    const uint32_t *host_resident_slot_indices;
    const uint32_t *host_lane_logical_block_counts;
} SparkKvBlockTableView;

typedef struct SparkKvCacheArena
{
    uint32_t abi_version;
    uint32_t descriptor_bytes;
    uint32_t logical_block_count;
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
    SparkKvCacheBlock *blocks;
    uint32_t *resident_slot_logical_block_indices;
    SparkKvCacheEvictFunction evict_function;
    void *evict_context;
    uint32_t free_logical_block_head;
    uint32_t next_resident_slot_scan;
    uint32_t reserved0;
    uint32_t reserved1;
    uint64_t epoch;
    uint64_t allocated_block_count;
    uint64_t recycled_block_count;
    uint64_t resident_block_count;
    uint64_t reserved_block_count;
    uint64_t resident_evicted_block_count;
    uint64_t resident_capacity_stall_count;
    uint64_t retained_block_count;
    uint64_t released_reference_count;
} SparkKvCacheArena;

SparkStatus SparkKvCacheEstimateCapacity(
    const SparkKvCacheCapacityRequest *request,
    SparkKvCacheCapacityEstimate *estimate);

uint32_t SparkKvCacheIndexSourceLayer(uint32_t layer_index);

SparkStatus SparkKvCacheCalculateJitStageBudget(
    const SparkKvJitStageBudgetRequest *request,
    SparkKvJitStageBudget *budget);

SparkStatus SparkKvCacheArenaInitialize(
    SparkKvCacheArena *arena,
    const SparkKvCacheConfiguration *configuration);

SparkStatus SparkKvCacheArenaAcquireBlock(
    SparkKvCacheArena *arena,
    uint32_t *logical_block_index_out);

SparkStatus SparkKvCacheArenaRecycleBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaRetainBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaReleaseBlockReference(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaPinResidentBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaUnpinResidentBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaMarkBlockDirty(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaMarkBlockResident(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaFreeBlock(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);


SparkStatus SparkKvCacheArenaMarkBlockNonResident(
    SparkKvCacheArena *arena,
    uint32_t logical_block_index);

SparkStatus SparkKvCacheArenaBuildPrefetchPlan(
    SparkKvCacheArena *arena,
    const uint32_t *logical_block_indices,
    uint32_t logical_block_count,
    uint32_t lane_count,
    SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCachePrefetchCursorInitialize(
    SparkKvCachePrefetchCursor *cursor,
    uint32_t logical_block_count);

SparkStatus SparkKvCacheArenaBuildNextPrefetchPlan(
    SparkKvCacheArena *arena,
    const uint32_t *logical_block_indices,
    uint32_t lane_count,
    SparkKvCachePrefetchCursor *cursor,
    SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheArenaBuildPrefetchPlanFromSourceBlocks(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchSourceBlock *source_blocks,
    uint32_t source_block_count,
    uint32_t lane_count,
    SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheArenaMarkPrefetchPlanResident(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheArenaMarkPrefetchPlanResidentWithProtectedBlocks(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count);

SparkStatus SparkKvCacheArenaCancelPrefetchPlan(
    SparkKvCacheArena *arena,
    const SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheArenaTrimResidentBlocks(
    SparkKvCacheArena *arena,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t target_resident_block_count,
    uint32_t *evicted_block_count_out);

SparkStatus SparkKvCacheArenaEvictResidentBlocksToLimit(
    SparkKvCacheArena *arena,
    uint32_t max_resident_block_count,
    const uint32_t *protected_logical_block_indices,
    uint32_t protected_logical_block_count,
    uint32_t *evicted_block_count_out);

SparkStatus SparkKvCacheArenaResolveBlock(
    const SparkKvCacheArena *arena,
    uint32_t logical_block_index,
    SparkKvCacheBlockView *block_view);

SparkStatus SparkKvCacheAsyncPrefetchBackendInitialize(
    SparkKvCacheAsyncPrefetchBackend *backend,
    const SparkKvCacheAsyncPrefetchBackendConfiguration *configuration);

SparkStatus SparkKvCacheAsyncPrefetchBackendStart(
    void *context,
    uint64_t prefetch_id,
    const SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheAsyncPrefetchBackendPoll(
    void *context,
    uint64_t prefetch_id,
    const SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheAsyncPrefetchBackendSubmitSynchronous(
    void *context,
    const SparkKvCachePrefetchPlan *prefetch_plan);

SparkStatus SparkKvCacheArenaReset(
    SparkKvCacheArena *arena);

#ifdef __cplusplus
}
#endif


static inline uint32_t SparkKvProtectedBlockListContainsBlock(const uint32_t *protected_logical_block_indices, uint32_t protected_logical_block_count, uint32_t logical_block_index)
{
	uint32_t protected_block_index;
	if ( protected_logical_block_count != 0u && protected_logical_block_indices == 0 )
		return 0u;
	for (protected_block_index = 0u; protected_block_index < protected_logical_block_count; ++protected_block_index)
	{
		if ( protected_logical_block_indices[protected_block_index] == logical_block_index )
			return 1u;
	}
	return 0u;
}
