#include <assert.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_glm52_kv_cache.h"
#include "sparkpipe/spark_glm52_model.h"
#include "sparkpipe/spark_glm52_stage_plan.h"

#define SPARK_TEST_KV_BLOCK_TOKENS 16u
#define SPARK_TEST_KV_HEAD_COUNT 8u
#define SPARK_TEST_KV_HEAD_DIMENSION 128u
#define SPARK_TEST_KV_INDEX_KEY_LAYER_COUNT 2u
#define SPARK_TEST_KV_STAGE_LAYER_COUNT 6u
#define SPARK_TEST_KV_TRANSFER_BYTES 64u

static void SparkTestInitializeKvCacheArena(
    SparkGlm52KvCacheArena *arena,
    SparkGlm52KvCacheBlock *blocks,
    uint32_t physical_block_count)
{
    SparkGlm52KvCacheConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.physical_block_count = physical_block_count;
    configuration.block_token_count = SPARK_TEST_KV_BLOCK_TOKENS;
    configuration.layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
    configuration.kv_head_count = SPARK_TEST_KV_HEAD_COUNT;
    configuration.head_dim = SPARK_TEST_KV_HEAD_DIMENSION;
    configuration.bytes_per_scalar = sizeof(uint16_t);
    configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    configuration.blocks = blocks;
    assert(SparkGlm52KvCacheArenaInitialize(arena, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestInitializeKvCacheArenaWithResidentCapacity(
    SparkGlm52KvCacheArena *arena,
    SparkGlm52KvCacheBlock *blocks,
    uint32_t physical_block_count,
    uint32_t resident_block_capacity)
{
    SparkGlm52KvCacheConfiguration configuration;

    memset(&configuration, 0, sizeof(configuration));
    configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    configuration.physical_block_count = physical_block_count;
    configuration.block_token_count = SPARK_TEST_KV_BLOCK_TOKENS;
    configuration.resident_block_capacity = resident_block_capacity;
    configuration.layer_count = SPARK_GLM52_MODEL_LAYER_COUNT;
    configuration.kv_head_count = SPARK_TEST_KV_HEAD_COUNT;
    configuration.head_dim = SPARK_TEST_KV_HEAD_DIMENSION;
    configuration.bytes_per_scalar = sizeof(uint16_t);
    configuration.key_device_base = (void *)(uintptr_t)0x100000000ull;
    configuration.value_device_base = (void *)(uintptr_t)0x200000000ull;
    configuration.blocks = blocks;
    assert(SparkGlm52KvCacheArenaInitialize(arena, &configuration) ==
        SPARK_STATUS_OK);
}

static void SparkTestKvCacheAllocatesResidentDeviceBlocks(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[4u];
    SparkGlm52KvCacheBlockView block_view;
    uint32_t first_block;
    uint32_t second_block;
    uint64_t expected_stride;

    SparkTestInitializeKvCacheArena(&arena, blocks, 4u);
    expected_stride =
        (uint64_t)SPARK_TEST_KV_BLOCK_TOKENS *
        SPARK_GLM52_MODEL_LAYER_COUNT *
        SPARK_TEST_KV_HEAD_COUNT *
        SPARK_TEST_KV_HEAD_DIMENSION *
        sizeof(uint16_t);
    assert(arena.key_block_stride_bytes == expected_stride);
    assert(arena.value_block_stride_bytes == expected_stride);

    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &first_block) ==
        SPARK_STATUS_OK);
    assert(first_block == 0u);
    assert(SparkGlm52KvCacheArenaRetainBlock(&arena, first_block) ==
        SPARK_STATUS_OK);
    assert(blocks[first_block].reference_count == 1u);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(&arena, first_block) ==
        SPARK_STATUS_OK);
    assert((blocks[first_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert(SparkGlm52KvCacheArenaResolveBlock(
        &arena,
        first_block,
        &block_view) == SPARK_STATUS_OK);
    assert(block_view.key_device_address == 0x100000000ull);
    assert(block_view.value_device_address == 0x200000000ull);

    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &second_block) ==
        SPARK_STATUS_OK);
    assert(second_block == 1u);
    assert(SparkGlm52KvCacheArenaResolveBlock(
        &arena,
        second_block,
        &block_view) == SPARK_STATUS_OK);
    assert(block_view.key_device_address == 0x100000000ull + expected_stride);
    assert(block_view.value_device_address == 0x200000000ull + expected_stride);
}



static void SparkTestKvCacheCanEvictResidentOwnedBlocks(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    uint32_t physical_block_index;
    uint32_t physical_block_indices[1u];

    SparkTestInitializeKvCacheArena(&arena, blocks, 2u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_index) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaRetainBlock(
        &arena,
        physical_block_index) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(
        &arena,
        physical_block_index) == SPARK_STATUS_OK);
    assert(arena.resident_block_count == 1u);

    assert(SparkGlm52KvCacheArenaMarkBlockNonResident(
        &arena,
        physical_block_index) == SPARK_STATUS_OK);
    assert(blocks[physical_block_index].reference_count == 1u);
    assert((blocks[physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u);
    assert((blocks[physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(arena.resident_block_count == 0u);

    physical_block_indices[0] = physical_block_index;
    assert(SparkGlm52KvCacheArenaBuildPrefetchPlan(
        &arena,
        physical_block_indices,
        1u,
        1u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.prefetch_block_count == 1u);
    assert(prefetch_plan.blocks[0].physical_block_index == physical_block_index);

    assert(SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        &arena,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert((blocks[physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert(arena.resident_block_count == 1u);
}



static void SparkTestKvCacheBuildsHashAddressedPrefetchPlan(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    SparkGlm52KvCachePrefetchSourceBlock source_block;
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    uint32_t physical_block_index;

    SparkTestInitializeKvCacheArena(&arena, blocks, 2u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_index) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaRetainBlock(
        &arena,
        physical_block_index) == SPARK_STATUS_OK);

    memset(&source_block, 0, sizeof(source_block));
    source_block.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    source_block.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block.physical_block_index = physical_block_index;
    source_block.token_capacity = 16u;
    source_block.first_token_index = 32u;
    source_block.token_count = 16u;
    source_block.flags = SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS;
    source_block.generation = blocks[physical_block_index].generation;
    source_block.parent_hash = 0x1111222233334444ull;
    source_block.block_hash = 0x2222333344445555ull;
    source_block.content_hash = 0x3333444455556666ull;

    assert(SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        &arena,
        &source_block,
        1u,
        1u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.requested_physical_block_count == 1u);
    assert(prefetch_plan.prefetch_block_count == 1u);
    assert(prefetch_plan.blocks[0u].physical_block_index ==
        physical_block_index);
    assert(prefetch_plan.blocks[0u].generation ==
        blocks[physical_block_index].generation);
    assert(prefetch_plan.blocks[0u].first_token_index == 32u);
    assert(prefetch_plan.blocks[0u].token_count == 16u);
    assert(prefetch_plan.blocks[0u].parent_hash == source_block.parent_hash);
    assert(prefetch_plan.blocks[0u].block_hash == source_block.block_hash);
    assert(prefetch_plan.blocks[0u].content_hash == source_block.content_hash);
    assert(prefetch_plan.blocks[0u].key_device_address ==
        blocks[physical_block_index].key_device_address);
    assert(prefetch_plan.blocks[0u].value_device_address ==
        blocks[physical_block_index].value_device_address);

    source_block.generation += 1u;
    assert(SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        &arena,
        &source_block,
        1u,
        1u,
        &prefetch_plan) == SPARK_STATUS_HASH_MISMATCH);
}



static void SparkTestKvCacheResidentEvictionRespectsProtectedBlocks(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[3u];
    uint32_t physical_block_indices[3u];
    uint32_t protected_physical_block_indices[1u];
    uint32_t evicted_block_count;
    uint32_t block_index;

    SparkTestInitializeKvCacheArena(&arena, blocks, 3u);
    for (block_index = 0u; block_index < 3u; ++block_index)
    {
        assert(SparkGlm52KvCacheArenaAcquireBlock(
            &arena,
            &physical_block_indices[block_index]) == SPARK_STATUS_OK);
        assert(physical_block_indices[block_index] == block_index);
        assert(SparkGlm52KvCacheArenaRetainBlock(
            &arena,
            physical_block_indices[block_index]) == SPARK_STATUS_OK);
        assert(SparkGlm52KvCacheArenaMarkBlockResident(
            &arena,
            physical_block_indices[block_index]) == SPARK_STATUS_OK);
    }
    assert(arena.resident_block_count == 3u);

    protected_physical_block_indices[0u] = physical_block_indices[0u];
    evicted_block_count = 0u;
    assert(SparkGlm52KvCacheArenaEvictResidentBlocksToLimit(
        &arena,
        1u,
        protected_physical_block_indices,
        1u,
        &evicted_block_count) == SPARK_STATUS_OK);
    assert(evicted_block_count == 2u);
    assert(arena.resident_block_count == 1u);
    assert((blocks[physical_block_indices[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((blocks[physical_block_indices[1u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert((blocks[physical_block_indices[2u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
}

static void SparkTestKvCacheRejectsStalePrefetchGeneration(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    uint32_t physical_block_index;
    uint32_t physical_block_indices[1u];

    SparkTestInitializeKvCacheArena(&arena, blocks, 2u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_index) == SPARK_STATUS_OK);
    physical_block_indices[0] = physical_block_index;
    assert(SparkGlm52KvCacheArenaBuildPrefetchPlan(
        &arena,
        physical_block_indices,
        1u,
        1u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.prefetch_block_count == 1u);
    assert(prefetch_plan.blocks[0].generation ==
        blocks[physical_block_index].generation);

    assert(SparkGlm52KvCacheArenaRecycleBlock(
        &arena,
        physical_block_index) == SPARK_STATUS_OK);
    assert(blocks[physical_block_index].generation !=
        prefetch_plan.blocks[0].generation);
    assert(SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        &arena,
        &prefetch_plan) == SPARK_STATUS_HASH_MISMATCH);
    assert((blocks[physical_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
}


static void SparkTestKvCacheEvictsUnprotectedResidentBlocksToLimit(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[4u];
    uint32_t first_block;
    uint32_t second_block;
    uint32_t third_block;
    uint32_t protected_blocks[1u];
    uint32_t evicted_block_count;

    SparkTestInitializeKvCacheArena(&arena, blocks, 4u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &first_block) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &second_block) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &third_block) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(&arena, first_block) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(&arena, second_block) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(&arena, third_block) ==
        SPARK_STATUS_OK);
    assert(arena.resident_block_count == 3u);

    protected_blocks[0u] = first_block;
    evicted_block_count = 0u;
    assert(SparkGlm52KvCacheArenaEvictResidentBlocksToLimit(
        &arena,
        1u,
        protected_blocks,
        1u,
        &evicted_block_count) == SPARK_STATUS_OK);
    assert(evicted_block_count == 2u);
    assert(arena.resident_block_count == 1u);
    assert((blocks[first_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((blocks[second_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert((blocks[third_block].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
}

static void SparkTestKvCachePrefetchPlanResidencyIsAtomicUnderCapacity(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    uint32_t physical_block_indices[2u];

    SparkTestInitializeKvCacheArenaWithResidentCapacity(
        &arena,
        blocks,
        2u,
        1u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_indices[0u]) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_indices[1u]) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaBuildPrefetchPlan(
        &arena,
        physical_block_indices,
        2u,
        1u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.prefetch_block_count == 2u);

    assert(SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        &arena,
        &prefetch_plan) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(arena.resident_block_count == 0u);
    assert((blocks[physical_block_indices[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert((blocks[physical_block_indices[1u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert(arena.resident_capacity_stall_count == 1u);
}

static void SparkTestKvCachePrefetchPlanEvictsColdBlocksAsGroup(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[3u];
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    uint32_t cold_block_index;
    uint32_t prefetch_block_indices[2u];

    SparkTestInitializeKvCacheArenaWithResidentCapacity(
        &arena,
        blocks,
        3u,
        2u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &cold_block_index) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &prefetch_block_indices[0u]) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &prefetch_block_indices[1u]) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkBlockResident(
        &arena,
        cold_block_index) == SPARK_STATUS_OK);
    assert(arena.resident_block_count == 1u);

    assert(SparkGlm52KvCacheArenaBuildPrefetchPlan(
        &arena,
        prefetch_block_indices,
        2u,
        2u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        &arena,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(arena.resident_block_count == 2u);
    assert((blocks[cold_block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
    assert((blocks[prefetch_block_indices[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((blocks[prefetch_block_indices[1u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}


static void SparkTestKvCacheAsyncPrefetchBackendCopiesHashAddressedBlocks(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[4u];
    SparkGlm52KvCacheConfiguration arena_configuration;
    SparkGlm52KvCachePrefetchSourceBlock source_blocks[2u];
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    SparkGlm52KvCachePrefetchBackendSourceEntry source_entries[2u];
    SparkGlm52KvCacheAsyncPrefetchBackendConfiguration backend_configuration;
    SparkGlm52KvCacheAsyncPrefetchBackend backend;
    uint8_t key_source[4u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint8_t value_source[4u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint8_t key_destination[4u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint8_t value_destination[4u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint32_t physical_block_indices[2u];
    uint32_t block_index;
    uint32_t byte_index;

    memset(key_source, 0, sizeof(key_source));
    memset(value_source, 0, sizeof(value_source));
    memset(key_destination, 0, sizeof(key_destination));
    memset(value_destination, 0, sizeof(value_destination));
    for (block_index = 0u; block_index < 4u; ++block_index)
    {
        for (byte_index = 0u;
            byte_index < SPARK_TEST_KV_TRANSFER_BYTES;
            ++byte_index)
        {
            key_source[block_index][byte_index] =
                (unsigned char)(0x10u + block_index + byte_index);
            value_source[block_index][byte_index] =
                (unsigned char)(0x80u + block_index + byte_index);
        }
    }

    memset(&arena_configuration, 0, sizeof(arena_configuration));
    arena_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    arena_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    arena_configuration.physical_block_count = 4u;
    arena_configuration.block_token_count = 16u;
    arena_configuration.layer_count = 1u;
    arena_configuration.kv_head_count = 1u;
    arena_configuration.head_dim = 32u;
    arena_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
    arena_configuration.key_block_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    arena_configuration.value_block_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    arena_configuration.key_device_base = key_destination;
    arena_configuration.value_device_base = value_destination;
    arena_configuration.blocks = blocks;
    assert(SparkGlm52KvCacheArenaInitialize(
        &arena,
        &arena_configuration) == SPARK_STATUS_OK);

    for (block_index = 0u; block_index < 2u; ++block_index)
    {
        assert(SparkGlm52KvCacheArenaAcquireBlock(
            &arena,
            &physical_block_indices[block_index]) == SPARK_STATUS_OK);
        assert(SparkGlm52KvCacheArenaRetainBlock(
            &arena,
            physical_block_indices[block_index]) == SPARK_STATUS_OK);

        memset(&source_blocks[block_index], 0, sizeof(source_blocks[block_index]));
        source_blocks[block_index].abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
        source_blocks[block_index].descriptor_bytes =
            SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
        source_blocks[block_index].physical_block_index =
            physical_block_indices[block_index];
        source_blocks[block_index].token_capacity = 16u;
        source_blocks[block_index].token_count = 16u;
        source_blocks[block_index].flags =
            SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_DEFAULT_FLAGS;
        source_blocks[block_index].generation =
            blocks[physical_block_indices[block_index]].generation;
        source_blocks[block_index].parent_hash = 0x11110000ull + block_index;
        source_blocks[block_index].block_hash = 0x22220000ull + block_index;
        source_blocks[block_index].content_hash = 0x33330000ull + block_index;

        memset(&source_entries[block_index], 0, sizeof(source_entries[block_index]));
        source_entries[block_index].abi_version =
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
        source_entries[block_index].descriptor_bytes =
            SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_SOURCE_ENTRY_DESCRIPTOR_BYTES;
        source_entries[block_index].physical_block_index =
            SPARK_GLM52_KV_CACHE_NO_BLOCK;
        source_entries[block_index].parent_hash =
            source_blocks[block_index].parent_hash;
        source_entries[block_index].block_hash =
            source_blocks[block_index].block_hash;
        source_entries[block_index].content_hash =
            source_blocks[block_index].content_hash;
        source_entries[block_index].key_source_address =
            key_source[block_index + 2u];
        source_entries[block_index].value_source_address =
            value_source[block_index + 2u];
    }

    assert(SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        &arena,
        source_blocks,
        2u,
        2u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.prefetch_block_count == 2u);

    memset(&backend_configuration, 0, sizeof(backend_configuration));
    backend_configuration.abi_version =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    backend_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
    backend_configuration.flags =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE |
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_DEFAULT_COPY_FLAGS;
    backend_configuration.lane_count = 2u;
    backend_configuration.max_inflight_prefetch_count = 2u;
    backend_configuration.physical_block_count = 4u;
    backend_configuration.source_entry_count = 2u;
    backend_configuration.blocks_per_poll = 1u;
    backend_configuration.key_source_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.value_source_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.key_transfer_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.value_transfer_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.key_source_base = key_source;
    backend_configuration.value_source_base = value_source;
    backend_configuration.source_entries = source_entries;
    assert(SparkGlm52KvCacheAsyncPrefetchBackendInitialize(
        &backend,
        &backend_configuration) == SPARK_STATUS_OK);

    assert(SparkGlm52KvCacheAsyncPrefetchBackendStart(
        &backend,
        99u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheAsyncPrefetchBackendPoll(
        &backend,
        99u,
        &prefetch_plan) == SPARK_STATUS_BUSY);
    assert(memcmp(key_destination[physical_block_indices[0u]],
        key_source[2u],
        sizeof(key_destination[physical_block_indices[0u]])) == 0);
    assert(memcmp(value_destination[physical_block_indices[0u]],
        value_source[2u],
        sizeof(value_destination[physical_block_indices[0u]])) == 0);
    assert(SparkGlm52KvCacheAsyncPrefetchBackendPoll(
        &backend,
        99u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(memcmp(key_destination[physical_block_indices[1u]],
        key_source[3u],
        sizeof(key_destination[physical_block_indices[1u]])) == 0);
    assert(memcmp(value_destination[physical_block_indices[1u]],
        value_source[3u],
        sizeof(value_destination[physical_block_indices[1u]])) == 0);
    assert(backend.started_prefetch_count == 1u);
    assert(backend.completed_prefetch_count == 1u);
    assert(backend.copied_key_block_count == 2u);
    assert(backend.copied_value_block_count == 2u);

    assert(SparkGlm52KvCacheArenaMarkPrefetchPlanResident(
        &arena,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert((blocks[physical_block_indices[0u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
    assert((blocks[physical_block_indices[1u]].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}

static void SparkTestKvCacheRejectsRecyclingRetainedBlocks(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    uint32_t block_index;
    uint64_t generation;

    SparkTestInitializeKvCacheArena(&arena, blocks, 2u);
    assert(SparkGlm52KvCacheArenaAcquireBlock(&arena, &block_index) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaRetainBlock(&arena, block_index) ==
        SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheArenaRecycleBlock(&arena, block_index) ==
        SPARK_STATUS_BUSY);
    assert(SparkGlm52KvCacheArenaReleaseBlockReference(&arena, block_index) ==
        SPARK_STATUS_OK);
    generation = blocks[block_index].generation;
    assert(SparkGlm52KvCacheArenaRecycleBlock(&arena, block_index) ==
        SPARK_STATUS_OK);
    assert(blocks[block_index].generation == generation + 1u);
    assert(SparkGlm52KvCacheArenaFreeBlock(&arena, block_index) ==
        SPARK_STATUS_OK);
    assert((blocks[block_index].flags &
        SPARK_GLM52_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u);
}

static void SparkTestKvCacheSupportsMlaPrimaryOnlyArenaAndPrefetch(void)
{
    SparkGlm52KvCacheArena arena;
    SparkGlm52KvCacheBlock blocks[2u];
    SparkGlm52KvCacheConfiguration arena_configuration;
    SparkGlm52KvCachePrefetchSourceBlock source_block;
    SparkGlm52KvCachePrefetchPlan prefetch_plan;
    SparkGlm52KvCacheAsyncPrefetchBackendConfiguration backend_configuration;
    SparkGlm52KvCacheAsyncPrefetchBackend backend;
    uint8_t source[2u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint8_t destination[2u][SPARK_TEST_KV_TRANSFER_BYTES];
    uint32_t physical_block_index;
    uint32_t byte_index;

    memset(source, 0, sizeof(source));
    memset(destination, 0, sizeof(destination));
    for (byte_index = 0u;
        byte_index < SPARK_TEST_KV_TRANSFER_BYTES;
        ++byte_index)
    {
        source[1u][byte_index] = (unsigned char)(0x40u + byte_index);
    }

    memset(&arena_configuration, 0, sizeof(arena_configuration));
    arena_configuration.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    arena_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
    arena_configuration.physical_block_count = 2u;
    arena_configuration.block_token_count = 16u;
    arena_configuration.layer_count = 1u;
    arena_configuration.kv_head_count = 1u;
    arena_configuration.head_dim = 32u;
    arena_configuration.bytes_per_scalar = (uint32_t)sizeof(uint16_t);
    arena_configuration.key_block_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    arena_configuration.key_device_base = destination;
    arena_configuration.blocks = blocks;
    assert(SparkGlm52KvCacheArenaInitialize(
        &arena,
        &arena_configuration) == SPARK_STATUS_OK);
    assert(arena.value_device_base == 0u);
    assert(arena.value_block_stride_bytes == 0u);

    assert(SparkGlm52KvCacheArenaAcquireBlock(
        &arena,
        &physical_block_index) == SPARK_STATUS_OK);
    assert(blocks[physical_block_index].value_device_address == 0u);

    memset(&source_block, 0, sizeof(source_block));
    source_block.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    source_block.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_SOURCE_BLOCK_DESCRIPTOR_BYTES;
    source_block.physical_block_index = physical_block_index;
    source_block.token_capacity = 16u;
    source_block.token_count = 16u;
    source_block.flags = SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_KEY;
    source_block.generation = blocks[physical_block_index].generation;
    source_block.parent_hash = 0x101u;
    source_block.block_hash = 0x202u;
    source_block.content_hash = 0x303u;
    assert(SparkGlm52KvCacheArenaBuildPrefetchPlanFromSourceBlocks(
        &arena,
        &source_block,
        1u,
        1u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(prefetch_plan.prefetch_block_count == 1u);
    assert((prefetch_plan.blocks[0u].flags &
        SPARK_GLM52_KV_CACHE_PREFETCH_BLOCK_FLAG_VALUE) == 0u);

    memset(&backend_configuration, 0, sizeof(backend_configuration));
    backend_configuration.abi_version =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
    backend_configuration.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
    backend_configuration.flags =
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE |
        SPARK_GLM52_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS;
    backend_configuration.lane_count = 1u;
    backend_configuration.max_inflight_prefetch_count = 1u;
    backend_configuration.physical_block_count = 2u;
    backend_configuration.blocks_per_poll = 1u;
    backend_configuration.key_source_stride_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.key_transfer_bytes = SPARK_TEST_KV_TRANSFER_BYTES;
    backend_configuration.key_source_base = source;
    assert(SparkGlm52KvCacheAsyncPrefetchBackendInitialize(
        &backend,
        &backend_configuration) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheAsyncPrefetchBackendStart(
        &backend,
        7u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(SparkGlm52KvCacheAsyncPrefetchBackendPoll(
        &backend,
        7u,
        &prefetch_plan) == SPARK_STATUS_OK);
    assert(memcmp(
        destination[physical_block_index],
        source[physical_block_index],
        sizeof(destination[physical_block_index])) == 0);
    assert(backend.copied_key_block_count == 1u);
    assert(backend.copied_value_block_count == 0u);
}

static void SparkTestKvCacheCapacityEstimatorAccountsForMlaCompression(void)
{
    SparkGlm52KvCacheCapacityRequest request;
    SparkGlm52KvCacheCapacityEstimate full_estimate;
    SparkGlm52KvCacheCapacityEstimate full_fp8_estimate;
    SparkGlm52KvCacheCapacityEstimate mla_estimate;
    uint64_t full_fp8_bytes_per_token_per_layer;

    memset(&request, 0, sizeof(request));
    request.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
    request.descriptor_bytes =
        SPARK_GLM52_KV_CACHE_CAPACITY_REQUEST_DESCRIPTOR_BYTES;
    request.layout = SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
    request.context_token_count = SPARK_GLM52_MODEL_MAXIMUM_CONTEXT_TOKENS;
    request.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    request.layer_count = SPARK_TEST_KV_STAGE_LAYER_COUNT;
    request.head_count = SPARK_GLM52_MODEL_HEAD_COUNT;
    request.qk_nope_head_dimension = SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION;
    request.value_head_dimension = SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION;
    request.latent_dimension = SPARK_GLM52_MODEL_LATENT_DIMENSION;
    request.rope_dimension = SPARK_GLM52_MODEL_ROPE_DIMENSION;
    request.bytes_per_scalar = sizeof(uint16_t);
    request.index_key_layer_count = SPARK_TEST_KV_INDEX_KEY_LAYER_COUNT;
    request.index_key_dimension = SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION;
    request.index_key_bytes_per_scalar = sizeof(uint16_t);
    request.cache_bytes_per_rank = 1024ull * 1024ull * 1024ull * 1024ull;

    assert(SparkGlm52KvCacheEstimateCapacity(
        &request,
        &full_estimate) == SPARK_STATUS_OK);

    request.layout = SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3;
    request.fp8_scale_block_size = SPARK_GLM52_MODEL_FP8_SCALE_BLOCK;
    assert(SparkGlm52KvCacheEstimateCapacity(
        &request,
        &full_fp8_estimate) == SPARK_STATUS_OK);

    request.layout = SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED;
    request.fp8_scale_block_size = 0u;
    assert(SparkGlm52KvCacheEstimateCapacity(
        &request,
        &mla_estimate) == SPARK_STATUS_OK);

    assert(full_estimate.contexts_per_rank < mla_estimate.contexts_per_rank);
    assert(full_estimate.contexts_per_rank <
        full_fp8_estimate.contexts_per_rank);
    assert(full_fp8_estimate.contexts_per_rank <
        mla_estimate.contexts_per_rank);
    assert(full_estimate.attention_bytes_per_token_per_layer ==
        ((uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
         ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
          SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
         ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
          SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION)) * sizeof(uint16_t));
    full_fp8_bytes_per_token_per_layer =
        (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
        ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
         SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
        ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
         SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION) +
        (((SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK) +
         ((((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK) +
         ((((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION) +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK)) * sizeof(float);
    assert(full_fp8_estimate.attention_bytes_per_token_per_layer ==
        full_fp8_bytes_per_token_per_layer);
    assert(mla_estimate.attention_bytes_per_token_per_layer ==
        SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS * sizeof(uint16_t));
    assert(mla_estimate.dsa_index_bytes_per_token ==
        SPARK_TEST_KV_INDEX_KEY_LAYER_COUNT *
        SPARK_GLM52_MODEL_DSA_INDEX_HEAD_DIMENSION * sizeof(uint16_t));
    assert(mla_estimate.block_count_per_context ==
        SPARK_GLM52_KV_CONTEXT_TOKENS / SPARK_GLM52_KV_BLOCK_TOKENS);
}

static void SparkTestKvJitStageBudgetsMatchRingStorage(void)
{
    SparkGlm52KvJitStageBudgetRequest request;
    SparkGlm52KvJitStageBudget budget;
    uint64_t full_bf16_bytes_per_token_per_layer;
    uint64_t full_fp8_bytes_per_token_per_layer;

    memset(&request, 0, sizeof(request));
    full_bf16_bytes_per_token_per_layer =
        ((uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
         ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
          SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
         ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
          SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION)) * sizeof(uint16_t);
    full_fp8_bytes_per_token_per_layer =
        (uint64_t)SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
        ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
         SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
        ((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
         SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION) +
        (((SPARK_GLM52_MODEL_CACHE_TOKEN_ELEMENTS +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK) +
         ((((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_QK_NOPE_HEAD_DIMENSION) +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK) +
         ((((uint64_t)SPARK_GLM52_MODEL_HEAD_COUNT *
            SPARK_GLM52_MODEL_VALUE_HEAD_DIMENSION) +
           SPARK_GLM52_MODEL_FP8_SCALE_BLOCK - 1u) /
          SPARK_GLM52_MODEL_FP8_SCALE_BLOCK)) * sizeof(float);
    request.abi_version = SPARK_GLM52_KV_JIT_STAGE_BUDGET_ABI_VERSION;
    request.descriptor_bytes =
        SPARK_GLM52_KV_JIT_STAGE_BUDGET_REQUEST_DESCRIPTOR_BYTES;
    request.layer_count = 6u;
    request.physical_pool_token_capacity = SPARK_GLM52_KV_POOL_TOKENS;
    request.backing_block_capacity = 1048576u;
    request.active_sequence_count = 1024u;
    request.backing_request_count =
        SPARK_GLM52_STAGE_PLAN_PIPELINE_INFLIGHT_REQUEST_CAPACITY;
    request.selected_token_count = SPARK_GLM52_MODEL_DSA_SELECTED_TOKEN_COUNT;
    request.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
    request.record_alignment_bytes =
        SPARK_GLM52_KV_JIT_DEFAULT_RECORD_ALIGNMENT;
    request.attention_cache_layout =
        SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED;

    request.first_layer_index = 0u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.local_dsa_index_layer_count == 3u);
    assert(budget.mla_bytes_per_token == 6912u);
    assert(budget.dsa_index_bytes_per_token == 768u);
    assert(budget.resident_bytes_per_token == 7680u);
    assert(budget.resident_summary_bytes == 100859904u);
    assert(budget.resident_pool_bytes == UINT64_C(32313114624));
    assert(budget.nvme_payload_bytes_per_block == 493059u);
    assert(budget.nvme_record_bytes == 499712u);
    assert(budget.nvme_capacity_bytes == UINT64_C(523986010112));
    assert(budget.compact_selected_mla_working_set_bytes ==
        UINT64_C(14495514624));
    assert(budget.maximum_average_active_context_tokens == 4096u);
    assert(budget.maximum_average_backing_context_tokens == 5041u);

    request.attention_cache_layout =
        SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED_FP8_E4M3;
    request.fp8_scale_block_size = SPARK_GLM52_MODEL_FP8_SCALE_BLOCK;
    request.first_layer_index = 0u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.mla_bytes_per_token == 3576u);
    assert(budget.dsa_index_bytes_per_token == 768u);
    assert(budget.resident_bytes_per_token == 4344u);

    request.attention_cache_layout =
        SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE_FP8_E4M3;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.mla_bytes_per_token ==
        6u * full_fp8_bytes_per_token_per_layer);
    assert(budget.resident_bytes_per_token ==
        budget.mla_bytes_per_token + budget.dsa_index_bytes_per_token);

    request.attention_cache_layout =
        SPARK_GLM52_KV_CACHE_LAYOUT_FULL_KEY_VALUE;
    request.fp8_scale_block_size = 0u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.mla_bytes_per_token ==
        6u * full_bf16_bytes_per_token_per_layer);

    request.attention_cache_layout =
        SPARK_GLM52_KV_CACHE_LAYOUT_MLA_COMPRESSED;

    request.first_layer_index = 6u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.local_dsa_index_layer_count == 2u);
    assert(budget.resident_bytes_per_token == 7424u);
    assert(budget.nvme_record_bytes == 483328u);

    request.first_layer_index = 12u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.local_dsa_index_layer_count == 1u);
    assert(budget.resident_bytes_per_token == 7168u);
    assert(budget.nvme_record_bytes == 466944u);

    request.first_layer_index = 72u;
    request.include_mtp_layer = 1u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_OK);
    assert(budget.local_dsa_index_layer_count == 1u);
    assert(budget.mtp_bytes_per_token == 1152u);
    assert(budget.resident_bytes_per_token == 8320u);
    assert(budget.resident_pool_bytes == UINT64_C(34930229248));
    assert(budget.nvme_record_bytes == 540672u);
    assert(budget.nvme_capacity_bytes == UINT64_C(566935683072));
    assert(budget.compact_selected_mla_working_set_bytes ==
        UINT64_C(16911433728));

    request.first_layer_index = 66u;
    assert(SparkGlm52KvCacheCalculateJitStageBudget(
        &request, &budget) == SPARK_STATUS_INVALID_ARGUMENT);
}

int main(void)
{
    SparkTestKvCacheAllocatesResidentDeviceBlocks();
    SparkTestKvCacheCanEvictResidentOwnedBlocks();
    SparkTestKvCacheBuildsHashAddressedPrefetchPlan();
    SparkTestKvCacheResidentEvictionRespectsProtectedBlocks();
    SparkTestKvCacheRejectsStalePrefetchGeneration();
    SparkTestKvCacheEvictsUnprotectedResidentBlocksToLimit();
    SparkTestKvCachePrefetchPlanResidencyIsAtomicUnderCapacity();
    SparkTestKvCachePrefetchPlanEvictsColdBlocksAsGroup();
    SparkTestKvCacheAsyncPrefetchBackendCopiesHashAddressedBlocks();
    SparkTestKvCacheSupportsMlaPrimaryOnlyArenaAndPrefetch();
    SparkTestKvCacheRejectsRecyclingRetainedBlocks();
    SparkTestKvCacheCapacityEstimatorAccountsForMlaCompression();
    SparkTestKvJitStageBudgetsMatchRingStorage();
    return 0;
}
