#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <fcntl.h>
#include <sched.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_kv_cache.h"
#include "sparkpipe/spark_kv_page_cache.h"
#include "sparkpipe/spark_kv_page_store.h"
#include "sparkpipe/spark_prefix_cache.h"

#define SPARK_TEST_LOGICAL_BLOCK_COUNT 8u
#define SPARK_TEST_RESIDENT_SLOT_COUNT 2u
#define SPARK_TEST_BLOCK_TOKENS 4u
#define SPARK_TEST_BLOCK_BYTES 32u

typedef struct SparkTestKvFixture
{
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[SPARK_TEST_LOGICAL_BLOCK_COUNT];
	uint32_t resident_owners[SPARK_TEST_RESIDENT_SLOT_COUNT];
	uint8_t device[SPARK_TEST_RESIDENT_SLOT_COUNT * SPARK_TEST_BLOCK_BYTES];
	uint8_t backing[SPARK_TEST_LOGICAL_BLOCK_COUNT * SPARK_TEST_BLOCK_BYTES];
	SparkStatus evict_status;
	uint32_t evict_count;
	uint32_t evicted_logical_block;
}
SparkTestKvFixture;

static SparkStatus SparkTestKvEvict(
	void *context,
	uint32_t logical_block_index,
	uint32_t resident_slot_index,
	uint64_t generation,
	uintptr_t key_device_address,
	uint64_t key_bytes,
	uintptr_t value_device_address,
	uint64_t value_bytes)
{
	SparkTestKvFixture *fixture;
	(void)generation;
	fixture = (SparkTestKvFixture *)context;
	assert(fixture != 0);
	assert(resident_slot_index < SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(key_device_address == (uintptr_t)(fixture->device +
		((uint64_t)resident_slot_index * SPARK_TEST_BLOCK_BYTES)));
	assert(key_bytes == SPARK_TEST_BLOCK_BYTES);
	assert(value_device_address == 0u && value_bytes == 0u);
	fixture->evict_count++;
	fixture->evicted_logical_block = logical_block_index;
	return(fixture->evict_status);
}

static void SparkTestKvInitialize(SparkTestKvFixture *fixture)
{
	SparkKvCacheConfiguration configuration;
	uint32_t index;
	memset(fixture,0,sizeof(*fixture));
	for (index=0u; index<sizeof(fixture->backing); index++)
		fixture->backing[index] = (uint8_t)(index + 1u);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.logical_block_count = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.block_token_count = SPARK_TEST_BLOCK_TOKENS;
	configuration.resident_block_capacity = SPARK_TEST_RESIDENT_SLOT_COUNT;
	configuration.layer_count = 1u;
	configuration.kv_head_count = 1u;
	configuration.head_dim = 8u;
	configuration.bytes_per_scalar = 1u;
	configuration.key_block_stride_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.key_device_base = fixture->device;
	configuration.blocks = fixture->blocks;
	configuration.resident_slot_logical_block_indices = fixture->resident_owners;
	configuration.evict_function = SparkTestKvEvict;
	configuration.evict_context = fixture;
	assert(SparkKvCacheArenaInitialize(&fixture->arena,&configuration) == SPARK_STATUS_OK);
}

static uint32_t SparkTestKvAcquire(SparkTestKvFixture *fixture)
{
	uint32_t logical_block_index;
	assert(SparkKvCacheArenaAcquireBlock(&fixture->arena,&logical_block_index) == SPARK_STATUS_OK);
	return(logical_block_index);
}

static void SparkTestKvLogicalBlocksReuseBoundedResidentSlots(void)
{
	SparkTestKvFixture fixture;
	SparkKvCacheBlockView view;
	uint32_t block0,block1,block2,slot0;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) == SPARK_STATUS_OK);
	slot0 = fixture.blocks[block0].resident_slot_index;
	assert(slot0 < SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) == SPARK_STATUS_OK);
	assert(fixture.arena.resident_block_count == SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(fixture.blocks[block0].resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT);
	assert(fixture.blocks[block0].key_device_address == 0u);
	assert(fixture.blocks[block2].resident_slot_index == slot0);
	assert(fixture.evict_count == 1u);
	assert(fixture.evicted_logical_block == block0);
	assert(SparkKvCacheArenaResolveBlock(&fixture.arena,block2,&view) == SPARK_STATUS_OK);
	assert(view.resident_slot_index == slot0);
	assert(view.key_device_address == (uintptr_t)(fixture.device + ((uint64_t)slot0 * SPARK_TEST_BLOCK_BYTES)));
}

static void SparkTestKvEvictionBackpressurePreservesResidentOwner(void)
{
	SparkTestKvFixture fixture;
	uint32_t block0,block1,block2,owner0,owner1;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) == SPARK_STATUS_OK);
	owner0 = fixture.resident_owners[0];
	owner1 = fixture.resident_owners[1];
	fixture.evict_status = SPARK_STATUS_BUSY;
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) == SPARK_STATUS_BUSY);
	assert(fixture.resident_owners[0] == owner0);
	assert(fixture.resident_owners[1] == owner1);
	assert(fixture.blocks[block2].resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT);
	fixture.evict_status = SPARK_STATUS_OK;
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) == SPARK_STATUS_OK);
	assert(fixture.evict_count == 2u);
}

static void SparkTestKvLogicalBlockFreeListReusesReleasedHead(void)
{
	SparkTestKvFixture fixture;
	uint32_t block0,block1,reused;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	assert(block0 == 0u && block1 == 1u);
	assert(SparkKvCacheArenaFreeBlock(&fixture.arena,block0) == SPARK_STATUS_OK);
	reused = SparkTestKvAcquire(&fixture);
	assert(reused == block0);
	assert(fixture.blocks[reused].free_next == SPARK_KV_CACHE_NO_BLOCK);
}

static void SparkTestKvFramePinProtectsResidentBlock(void)
{
	SparkTestKvFixture fixture;
	uint32_t block0,block1,block2;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaPinResidentBlock(&fixture.arena,block0) == SPARK_STATUS_OK);
	assert(fixture.blocks[block0].residency_reference_count == 1u);
	assert(SparkKvCacheArenaMarkBlockNonResident(&fixture.arena,block0) == SPARK_STATUS_BUSY);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) == SPARK_STATUS_OK);
	assert((fixture.blocks[block0].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	assert((fixture.blocks[block1].flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
	assert(SparkKvCacheArenaUnpinResidentBlock(&fixture.arena,block0) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaUnpinResidentBlock(&fixture.arena,block0) == SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkKvCacheArenaMarkBlockNonResident(&fixture.arena,block0) == SPARK_STATUS_OK);
}

static void SparkTestKvInitializeBackend(
	SparkTestKvFixture *fixture,
	SparkKvCacheAsyncPrefetchBackend *backend)
{
	SparkKvCacheAsyncPrefetchBackendConfiguration configuration;
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_CACHE_PREFETCH_BACKEND_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_CACHE_PREFETCH_BACKEND_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.flags = SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_MEMORY_SOURCE | SPARK_KV_CACHE_PREFETCH_BACKEND_FLAG_COPY_KEY_BLOCKS;
	configuration.lane_count = 2u;
	configuration.max_inflight_prefetch_count = 2u;
	configuration.logical_block_count = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.blocks_per_poll = 1u;
	configuration.key_source_stride_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.key_transfer_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.key_source_base = fixture->backing;
	assert(SparkKvCacheAsyncPrefetchBackendInitialize(backend,&configuration) == SPARK_STATUS_OK);
}

static void SparkTestKvPrefetchUsesReservedDeviceSlot(void)
{
	SparkTestKvFixture fixture;
	SparkKvCacheAsyncPrefetchBackend backend;
	SparkKvCachePrefetchPlan plan;
	uint32_t block,slot;
	SparkTestKvInitialize(&fixture);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaBuildPrefetchPlan(&fixture.arena,&block,1u,1u,&plan) == SPARK_STATUS_OK);
	assert(plan.prefetch_block_count == 1u);
	assert(fixture.arena.reserved_block_count == 1u);
	slot = plan.blocks[0u].resident_slot_index;
	assert(slot < SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(plan.blocks[0u].key_device_address == (uintptr_t)(fixture.device + ((uint64_t)slot * SPARK_TEST_BLOCK_BYTES)));
	SparkTestKvInitializeBackend(&fixture,&backend);
	assert(SparkKvCacheAsyncPrefetchBackendStart(&backend,1u,&plan) == SPARK_STATUS_OK);
	assert(SparkKvCacheAsyncPrefetchBackendPoll(&backend,1u,&plan) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkPrefetchPlanResident(&fixture.arena,&plan) == SPARK_STATUS_OK);
	assert(fixture.arena.reserved_block_count == 0u);
	assert(fixture.arena.resident_block_count == 1u);
	assert(memcmp(fixture.device + ((uint64_t)slot * SPARK_TEST_BLOCK_BYTES),fixture.backing + ((uint64_t)block * SPARK_TEST_BLOCK_BYTES),SPARK_TEST_BLOCK_BYTES) == 0);
}

static void SparkTestKvOverlappingPrefetchReservationsAreReferenceCounted(void)
{
	SparkTestKvFixture fixture;
	SparkKvCachePrefetchPlan first,second;
	uint32_t block,slot;
	SparkTestKvInitialize(&fixture);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaBuildPrefetchPlan(&fixture.arena,&block,1u,1u,&first) == SPARK_STATUS_OK);
	slot = first.blocks[0u].resident_slot_index;
	assert(SparkKvCacheArenaBuildPrefetchPlan(&fixture.arena,&block,1u,1u,&second) == SPARK_STATUS_OK);
	assert(second.blocks[0u].resident_slot_index == slot);
	assert(fixture.blocks[block].residency_reference_count == 2u);
	assert(fixture.arena.reserved_block_count == 1u);
	assert(SparkKvCacheArenaMarkPrefetchPlanResident(&fixture.arena,&first) == SPARK_STATUS_OK);
	assert(fixture.blocks[block].residency_reference_count == 1u);
	assert(fixture.arena.resident_block_count == 1u);
	assert(fixture.arena.reserved_block_count == 0u);
	assert(SparkKvCacheArenaCancelPrefetchPlan(&fixture.arena,&second) == SPARK_STATUS_OK);
	assert(fixture.blocks[block].residency_reference_count == 0u);
	assert(fixture.blocks[block].resident_slot_index == slot);
}

static void SparkTestKvCancelledPrefetchReleasesDeviceSlot(void)
{
	SparkTestKvFixture fixture;
	SparkKvCachePrefetchPlan plan;
	uint32_t block,slot;
	SparkTestKvInitialize(&fixture);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaBuildPrefetchPlan(&fixture.arena,&block,1u,1u,&plan) == SPARK_STATUS_OK);
	slot = plan.blocks[0u].resident_slot_index;
	assert(SparkKvCacheArenaCancelPrefetchPlan(&fixture.arena,&plan) == SPARK_STATUS_OK);
	assert(fixture.arena.reserved_block_count == 0u);
	assert(fixture.blocks[block].resident_slot_index == SPARK_KV_CACHE_NO_RESIDENT_SLOT);
	assert(fixture.resident_owners[slot] == SPARK_KV_CACHE_NO_BLOCK);
}

static void SparkTestKvPrefetchCursorOwnsPlanChunking(void)
{
	SparkTestKvFixture fixture;
	SparkKvCachePrefetchCursor cursor;
	SparkKvCachePrefetchPlan plan;
	uint32_t blocks[2u];
	SparkTestKvInitialize(&fixture);
	blocks[0u] = SparkTestKvAcquire(&fixture);
	blocks[1u] = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[0u]) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[1u]) == SPARK_STATUS_OK);
	assert(SparkKvCachePrefetchCursorInitialize(&cursor,2u) == SPARK_STATUS_OK);
	assert(SparkKvCacheArenaBuildNextPrefetchPlan(&fixture.arena,blocks,1u,&cursor,&plan) == SPARK_STATUS_OK);
	assert(cursor.next_logical_block_index == 2u);
	assert(plan.requested_logical_block_count == 2u);
	assert(plan.resident_block_count == 2u);
	assert(plan.prefetch_block_count == 0u);
	assert(SparkKvCacheArenaBuildNextPrefetchPlan(&fixture.arena,blocks,1u,&cursor,&plan) == SPARK_STATUS_OK);
	assert(plan.requested_logical_block_count == 0u);
}

static void SparkTestKvPageStoreWritesDirtyOnceAndRestores(void)
{
	SparkTestKvFixture fixture;
	SparkKvPageStoreConfiguration configuration;
	SparkKvPageStore store;
	char path[] = "/tmp/sparkpipe-kv-page-store-XXXXXX";
	uint8_t staging[SPARK_TEST_BLOCK_BYTES],expected[SPARK_TEST_BLOCK_BYTES];
	uint32_t block0,block1,block2,index,slot;
	int32_t descriptor;
	SparkStatus status;
	SparkTestKvInitialize(&fixture);
	descriptor = mkstemp(path);
	assert(descriptor >= 0);
	assert(close(descriptor) == 0);
	assert(unlink(path) == 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.transfer_capacity = SPARK_TEST_RESIDENT_SLOT_COUNT;
	configuration.page_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.maximum_backing_bytes =
		2u * SPARK_TEST_BLOCK_BYTES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.staging_bytes = sizeof(staging);
	assert(SparkKvPageStoreInitialize(&store,&configuration) == SPARK_STATUS_OK);
	fixture.arena.evict_function = SparkKvPageStoreWriteback;
	fixture.arena.evict_context = &store;
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	slot = fixture.blocks[block0].resident_slot_index;
	for (index=0u; index<sizeof(expected); index++)
		expected[index] = fixture.device[(uint64_t)slot * SPARK_TEST_BLOCK_BYTES +
			index] = (uint8_t)(0xa0u + index);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2);
	assert(status == SPARK_STATUS_BUSY);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2);
	}
	assert(status == SPARK_STATUS_OK);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u);
	assert((fixture.blocks[block0].flags & SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) == 0u);
	assert(store.write_count == 1u);
	status = SparkKvPageStorePrefetch(&store,&fixture.arena,block0);
	assert(status == SPARK_STATUS_BUSY);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvPageStorePrefetch(&store,&fixture.arena,block0);
	}
	assert(status == SPARK_STATUS_OK);
	slot = fixture.blocks[block0].resident_slot_index;
	assert(memcmp(fixture.device + (uint64_t)slot * SPARK_TEST_BLOCK_BYTES,
		expected,sizeof(expected)) == 0);
	assert(store.write_count == 2u && store.read_count == 1u);
	assert(store.backing_page_count == 2u);
	assert(SparkKvPageStoreWriteback(&store,block2,
		fixture.blocks[block2].resident_slot_index,
		fixture.blocks[block2].generation,
		fixture.blocks[block2].key_device_address,SPARK_TEST_BLOCK_BYTES,
		0u,0u) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkKvCacheArenaMarkBlockNonResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(store.write_count == 2u);
	SparkKvPageStoreDestroy(&store);
	assert(unlink(path) == 0);
}

static void SparkTestPrefixCacheReusesCommittedLogicalBlocks(void)
{
	SparkTestKvFixture fixture;
	SparkPrefixCache cache;
	SparkPrefixCacheConfiguration configuration;
	SparkPrefixCacheEntry entries[8u];
	SparkPrefixCacheSequenceBinding bindings[16u];
	SparkPrefixCacheLookup lookup;
	uint32_t tokens[16u],index;
	SparkTestKvInitialize(&fixture);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_PREFIX_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_PREFIX_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	configuration.block_token_count = SPARK_TEST_BLOCK_TOKENS;
	configuration.entry_count = 8u;
	configuration.logical_block_count = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.sequence_binding_count = 16u;
	configuration.entries = entries;
	configuration.sequence_bindings = bindings;
	configuration.kv_cache_arena = &fixture.arena;
	assert(SparkPrefixCacheInitialize(&cache,&configuration) == SPARK_STATUS_OK);
	for (index=0u; index<16u; index++)
		tokens[index] = 1000u + index;
	assert(SparkPrefixCacheCommitPrompt(&cache,1u,tokens,16u,&lookup) == SPARK_STATUS_OK);
	assert(lookup.matched_token_count == 16u);
	assert(SparkPrefixCacheLookupPrompt(&cache,2u,tokens,16u,&lookup) == SPARK_STATUS_OK);
	assert(lookup.matched_token_count == 12u);
	assert(lookup.matched_block_count == 3u);
	assert(cache.hit_count == 1u);
}

typedef struct SparkTestKvPageFixture
{
	SparkTestKvFixture kv;
	SparkKvPageCache cache;
	SparkKvPageCacheEntry entries[SPARK_TEST_LOGICAL_BLOCK_COUNT];
	SparkKvPageCacheSequence sequences[4u];
	uint32_t hash_heads[SPARK_TEST_LOGICAL_BLOCK_COUNT];
	uint32_t entry_indices_by_logical_page[SPARK_TEST_LOGICAL_BLOCK_COUNT];
}
SparkTestKvPageFixture;

static void SparkTestKvIdentity(
	SparkModelDriverCacheIdentity *identity,
	uint8_t seed)
{
	uint32_t index;
	memset(identity,0,sizeof(*identity));
	for (index=0u; index<sizeof(identity->sha256); index++)
		identity->sha256[index] = (uint8_t)(seed + index);
}

static void SparkTestKvPageInitialize(SparkTestKvPageFixture *fixture)
{
	SparkKvPageCacheConfiguration configuration;
	SparkTestKvInitialize(&fixture->kv);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_CACHE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_CACHE_CONFIGURATION_BYTES;
	configuration.sequence_capacity = 4u;
	configuration.entry_capacity = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.hash_bucket_count = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.kv_cache_arena = &fixture->kv.arena;
	configuration.entries = fixture->entries;
	configuration.sequences = fixture->sequences;
	configuration.hash_bucket_heads = fixture->hash_heads;
	configuration.entry_indices_by_logical_page =
		fixture->entry_indices_by_logical_page;
	assert(SparkKvPageCacheInitialize(&fixture->cache,&configuration) ==
		SPARK_STATUS_OK);
}

static void SparkTestKvPageLane(
	SparkModelDriverCacheLane *lane,
	uint64_t sequence_id,
	uint32_t resident_slot,
	uint32_t position,
	uint32_t context)
{
	memset(lane,0,sizeof(*lane));
	lane->sequence_id = sequence_id;
	lane->resident_sequence_slot = resident_slot;
	lane->sequence_position = position;
	lane->context_token_count = context;
}

static void SparkTestKvPagePublish(
	SparkModelDriverCacheLane *lane,
	uint32_t token_count,
	uint8_t seed)
{
	lane->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PUBLISH;
	lane->publish_token_count = token_count;
	SparkTestKvIdentity(&lane->publish_identity,seed);
}

static void SparkTestKvPagePrefix(
	SparkModelDriverCacheLane *lane,
	uint32_t token_count,
	uint8_t seed)
{
	lane->flags |= SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane->prefix_token_count = token_count;
	SparkTestKvIdentity(&lane->prefix_identity,seed);
}

static uint32_t SparkTestKvPageBegin(
	SparkTestKvPageFixture *fixture,
	const SparkModelDriverCacheLane *lane)
{
	uint32_t logical_page;
	assert(SparkKvPageCacheBeginLane(&fixture->cache,lane,&logical_page) ==
		SPARK_STATUS_OK);
	return(logical_page);
}

static void SparkTestKvPageCacheSharesImmutableChains(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t pages[4u],count,page0,page1;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,1u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,11u);
	assert(SparkKvPageCachePrepareLane(&fixture.cache,&lane,pages,4u,&count) ==
		SPARK_STATUS_OK && count == 0u);
	page0 = SparkTestKvPageBegin(&fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,1u,0u,4u,5u);
	assert(SparkKvPageCachePrepareLane(&fixture.cache,&lane,pages,4u,&count) ==
		SPARK_STATUS_OK && count == 1u && pages[0u] == page0);
	page1 = SparkTestKvPageBegin(&fixture,&lane);
	assert(page1 != page0);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,1u,0u,5u,8u);
	SparkTestKvPagePublish(&lane,8u,22u);
	assert(SparkTestKvPageBegin(&fixture,&lane) == page1);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,1u) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,2u,1u,8u,9u);
	SparkTestKvPagePrefix(&lane,8u,22u);
	assert(SparkKvPageCachePrepareLane(&fixture.cache,&lane,pages,4u,&count) ==
		SPARK_STATUS_OK && count == 2u && pages[0u] == page0 && pages[1u] == page1);
	assert(SparkTestKvPageBegin(&fixture,&lane) != SPARK_KV_CACHE_NO_BLOCK);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheBuildLaneTable(&fixture.cache,1u,2u,pages,4u,&count) ==
		SPARK_STATUS_OK && count == 3u);
}

static void SparkTestKvPageCacheDeduplicatesAndRequiresRelease(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane,other;
	uint32_t pages[4u],count,page0;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,1u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,33u);
	page0 = SparkTestKvPageBegin(&fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,1u) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,2u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,33u);
	assert(SparkTestKvPageBegin(&fixture,&lane) != page0);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(fixture.cache.deduplicated_page_count == 1u);
	assert(SparkKvPageCacheBuildLaneTable(&fixture.cache,0u,2u,pages,4u,&count) ==
		SPARK_STATUS_OK && count == 1u && pages[0u] == page0);
	SparkTestKvPageLane(&other,3u,0u,0u,1u);
	assert(SparkKvPageCacheBeginLane(&fixture.cache,&other,&count) ==
		SPARK_STATUS_BUSY);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,2u) == SPARK_STATUS_OK);
	assert(SparkTestKvPageBegin(&fixture,&other) != SPARK_KV_CACHE_NO_BLOCK);
}

static void SparkTestKvPageCacheRejectsMissingPrefix(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t pages[2u],count;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,7u,0u,4u,5u);
	SparkTestKvPagePrefix(&lane,4u,99u);
	assert(SparkKvPageCachePrepareLane(&fixture.cache,&lane,pages,2u,&count) ==
		SPARK_STATUS_NOT_FOUND);
}

static void SparkTestKvPageCachePrefetchAndBeginAreTransactional(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t pages[4u],count,mutable_page,mutation_flags,published_page;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,1u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,71u);
	published_page = SparkTestKvPageBegin(&fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) ==
		SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,1u,0u,4u,5u);
	assert(SparkKvPageCacheBeginLaneTransaction(&fixture.cache,&lane,
		&mutable_page,&mutation_flags) == SPARK_STATUS_OK);
	assert(mutable_page != published_page);
	assert(mutation_flags ==
		SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE);
	assert(SparkKvPageCacheRollbackLaneTransaction(&fixture.cache,&lane,
		mutation_flags) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheBuildLaneTable(&fixture.cache,0u,1u,pages,4u,
		&count) == SPARK_STATUS_OK);
	assert(count == 1u && pages[0u] == published_page);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,1u) ==
		SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,2u,1u,4u,5u);
	SparkTestKvPagePrefix(&lane,4u,71u);
	assert(SparkKvPageCachePrepareLane(&fixture.cache,&lane,pages,4u,&count) ==
		SPARK_STATUS_OK);
	assert(count == 1u && pages[0u] == published_page);
	assert(fixture.cache.sequences[1u].sequence_id == 0u);
	assert(SparkKvPageCacheBeginLaneTransaction(&fixture.cache,&lane,
		&mutable_page,&mutation_flags) == SPARK_STATUS_OK);
	assert(mutation_flags ==
		(SPARK_KV_PAGE_CACHE_MUTATION_BOUND_SEQUENCE |
		 SPARK_KV_PAGE_CACHE_MUTATION_ALLOCATED_MUTABLE));
	assert(SparkKvPageCacheRollbackLaneTransaction(&fixture.cache,&lane,
		mutation_flags) == SPARK_STATUS_OK);
	assert(fixture.cache.sequences[1u].sequence_id == 0u);
	assert(fixture.cache.sequences[1u].mutable_logical_page_index ==
		SPARK_KV_CACHE_NO_BLOCK);
}

static void SparkTestKvPageCacheReclaimsColdPrefixUnderPressure(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t page;
	SparkTestKvPageInitialize(&fixture);
	fixture.kv.arena.evict_function = 0;
	fixture.kv.arena.evict_context = 0;
	SparkTestKvPageLane(&lane,1u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,41u);
	page = SparkTestKvPageBegin(&fixture,&lane);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,1u) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,2u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,42u);
	assert(SparkTestKvPageBegin(&fixture,&lane) != page);
	assert(SparkKvPageCacheCompleteLane(&fixture.cache,&lane) == SPARK_STATUS_OK);
	assert(SparkKvPageCacheReleaseLane(&fixture.cache,0u,2u) == SPARK_STATUS_OK);
	SparkTestKvPageLane(&lane,3u,0u,0u,1u);
	assert(SparkTestKvPageBegin(&fixture,&lane) != SPARK_KV_CACHE_NO_BLOCK);
	assert(fixture.cache.evicted_entry_count == 1u);
}

int main(void)
{
	SparkTestKvLogicalBlocksReuseBoundedResidentSlots();
	SparkTestKvEvictionBackpressurePreservesResidentOwner();
	SparkTestKvLogicalBlockFreeListReusesReleasedHead();
	SparkTestKvFramePinProtectsResidentBlock();
	SparkTestKvPrefetchUsesReservedDeviceSlot();
	SparkTestKvOverlappingPrefetchReservationsAreReferenceCounted();
	SparkTestKvCancelledPrefetchReleasesDeviceSlot();
	SparkTestKvPrefetchCursorOwnsPlanChunking();
	SparkTestKvPageStoreWritesDirtyOnceAndRestores();
	SparkTestPrefixCacheReusesCommittedLogicalBlocks();
	SparkTestKvPageCacheSharesImmutableChains();
	SparkTestKvPageCacheDeduplicatesAndRequiresRelease();
	SparkTestKvPageCacheRejectsMissingPrefix();
	SparkTestKvPageCachePrefetchAndBeginAreTransactional();
	SparkTestKvPageCacheReclaimsColdPrefixUnderPressure();
	return(0);
}
