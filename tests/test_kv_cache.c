#define _POSIX_C_SOURCE 200809L

#include <assert.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/stat.h>
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

static void SparkTestKvEvictionIoErrorDegradesInsteadOfWedging(void)
{
	SparkTestKvFixture fixture;
	SparkKvCacheBlockView view;
	uint32_t block0,block1,block2,block3;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	fixture.evict_status = SPARK_STATUS_IO_ERROR;
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) ==
		SPARK_STATUS_OK);
	block3 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block3) ==
		SPARK_STATUS_OK);
	assert(fixture.arena.resident_block_count == SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(fixture.arena.write_back_degraded_block_count == 2u);
	assert(fixture.evict_count == 2u);
	assert(SparkKvCacheArenaResolveBlock(&fixture.arena,block0,&view) ==
		SPARK_STATUS_OK);
	assert((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
	assert((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) == 0u);
	assert((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	assert(fixture.arena.write_back_degraded_block_count == 3u);
	fixture.evict_status = SPARK_STATUS_OK;
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block3) ==
		SPARK_STATUS_OK);
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) ==
		SPARK_STATUS_OK);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) == 0u);
	assert(fixture.arena.write_back_degraded_block_count == 3u);
	assert(fixture.evict_count == 4u);
}

static void SparkTestKvEvictionInternalErrorStaysLoud(void)
{
	SparkTestKvFixture fixture;
	uint32_t block0,block1,block2;
	SparkTestKvInitialize(&fixture);
	block0 = SparkTestKvAcquire(&fixture);
	block1 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	fixture.evict_status = SPARK_STATUS_INTERNAL_ERROR;
	block2 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2) ==
		SPARK_STATUS_INTERNAL_ERROR);
	assert((fixture.blocks[block1].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	assert((fixture.blocks[block1].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) != 0u);
	assert(fixture.arena.write_back_degraded_block_count == 0u);
	assert(fixture.arena.resident_block_count == SPARK_TEST_RESIDENT_SLOT_COUNT);
}

static void SparkTestKvPageStoreFullDiskDegradesAndServingContinues(void)
{
	SparkTestKvFixture fixture;
	SparkKvPageStoreConfiguration configuration;
	SparkKvPageStore store;
	SparkKvCacheBlockView view;
	char path[] = "/tmp/sparkpipe-kv-full-disk-XXXXXX";
	uint8_t staging[SPARK_TEST_BLOCK_BYTES],expected[SPARK_TEST_BLOCK_BYTES];
	uint32_t block0,block1,block2,block3,block4,index,slot;
	struct rlimit old_limit,full_limit;
	void *old_handler;
	int32_t descriptor;
	SparkStatus status;
	descriptor = mkstemp(path);
	assert(descriptor >= 0);
	assert(close(descriptor) == 0);
	assert(unlink(path) == 0);
	old_handler = signal(SIGXFSZ,SIG_IGN);
	assert(old_handler != SIG_ERR);
	assert(getrlimit(RLIMIT_FSIZE,&old_limit) == 0);
	full_limit.rlim_cur = SPARK_TEST_BLOCK_BYTES;
	full_limit.rlim_max = old_limit.rlim_max;
	assert(setrlimit(RLIMIT_FSIZE,&full_limit) == 0);
	SparkTestKvInitialize(&fixture);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.transfer_capacity = SPARK_TEST_RESIDENT_SLOT_COUNT;
	configuration.page_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.maximum_backing_bytes = 2u * SPARK_TEST_BLOCK_BYTES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.staging_bytes = sizeof(staging);
	assert(SparkKvPageStoreInitialize(&store,&configuration) == SPARK_STATUS_OK);
	fixture.arena.evict_function = SparkKvPageStoreWriteback;
	fixture.arena.evict_context = &store;
	block0 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	slot = fixture.blocks[block0].resident_slot_index;
	for ( index=0u; index<sizeof(expected); index++ )
		expected[index] = fixture.device[(uint64_t)slot * SPARK_TEST_BLOCK_BYTES +
			index] = (uint8_t)(0x5au + index);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	block1 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block1) ==
		SPARK_STATUS_OK);
	block2 = SparkTestKvAcquire(&fixture);
	status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block2);
	}
	assert(status == SPARK_STATUS_OK);
	assert(store.write_count == 1u);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u);
	block3 = SparkTestKvAcquire(&fixture);
	status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block3);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block3);
	}
	assert(status == SPARK_STATUS_OK);
	block4 = SparkTestKvAcquire(&fixture);
	status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block4);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvCacheArenaMarkBlockResident(&fixture.arena,block4);
	}
	assert(status == SPARK_STATUS_OK);
	assert(fixture.arena.resident_block_count == SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(fixture.arena.write_back_degraded_block_count == 2u);
	assert(store.write_count == 1u);
	assert(store.backing_page_count == 1u);
	assert(SparkKvCacheArenaResolveBlock(&fixture.arena,block1,&view) ==
		SPARK_STATUS_OK);
	assert((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u);
	assert(SparkKvPageStorePrefetch(&store,&fixture.arena,block1) ==
		SPARK_STATUS_NOT_FOUND);
	for ( index = 0u; index < 10000u; ++index )
	{
		if ( (fixture.blocks[block0].flags &
				SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
			break;
		(void)sched_yield();
		(void)SparkKvPageStoreProgress(&store,&fixture.arena,1u);
		(void)SparkKvPageStorePrefetch(&store,&fixture.arena,block0);
	}
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
	slot = fixture.blocks[block0].resident_slot_index;
	assert(memcmp(fixture.device + (uint64_t)slot * SPARK_TEST_BLOCK_BYTES,
		expected,sizeof(expected)) == 0);
	assert(store.read_count == 1u);
	assert(setrlimit(RLIMIT_FSIZE,&old_limit) == 0);
	(void)signal(SIGXFSZ,old_handler);
	SparkKvPageStoreDestroy(&store);
	assert(unlink(path) == 0);
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

static int32_t SparkTestKvPinnedTableUsesPhysicalMapping(void)
{
	SparkTestKvFixture fixture;
	uint32_t blocks[3],pages[3],physical[3],index;
	SparkTestKvInitialize(&fixture);
	for (index=0u; index<3u; index++)
		blocks[index] = SparkTestKvAcquire(&fixture);
	if ( SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[2]) != SPARK_STATUS_OK || SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[0]) != SPARK_STATUS_OK )
		return(-1);
	pages[0] = blocks[2];
	pages[1] = blocks[0];
	pages[2] = blocks[2];
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,3u,physical) != SPARK_STATUS_OK )
		return(-2);
	if ( physical[0] == pages[0] || physical[0] != physical[2] || physical[0] == physical[1] || fixture.blocks[blocks[2]].residency_reference_count != 2u )
		return(-3);
	if ( SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[1]) != SPARK_STATUS_CAPACITY_EXCEEDED )
		return(-4);
	if ( SparkKvCacheArenaUnpinResidentTable(&fixture.arena,pages,3u) != SPARK_STATUS_OK || fixture.blocks[blocks[2]].residency_reference_count != 0u )
		return(-5);
	if ( SparkKvCacheArenaMarkBlockResident(&fixture.arena,blocks[1]) != SPARK_STATUS_OK )
		return(-6);
	return(0);
}

static int32_t SparkTestKvPinnedTableFailurePreservesOtherOwners(void)
{
	SparkTestKvFixture fixture;
	uint32_t pages[3],physical[3],index;
	SparkTestKvInitialize(&fixture);
	for (index=0u; index<3u; index++)
		pages[index] = SparkTestKvAcquire(&fixture);
	if ( SparkKvCacheArenaMarkBlockResident(&fixture.arena,pages[0]) != SPARK_STATUS_OK || SparkKvCacheArenaMarkBlockResident(&fixture.arena,pages[1]) != SPARK_STATUS_OK )
		return(-7);
	if ( SparkKvCacheArenaPinResidentBlock(&fixture.arena,pages[0]) != SPARK_STATUS_OK )
		return(-8);
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,3u,physical) != SPARK_STATUS_BUSY )
		return(-9);
	if ( fixture.blocks[pages[0]].residency_reference_count != 1u || fixture.blocks[pages[1]].residency_reference_count != 0u )
		return(-10);
	fixture.blocks[pages[1]].residency_reference_count = UINT32_MAX;
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,2u,physical) != SPARK_STATUS_CAPACITY_EXCEEDED || fixture.blocks[pages[0]].residency_reference_count != 1u )
		return(-11);
	fixture.blocks[pages[1]].residency_reference_count = 0u;
	pages[1] = fixture.arena.logical_block_count;
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,2u,physical) != SPARK_STATUS_INVALID_ARGUMENT || fixture.blocks[pages[0]].residency_reference_count != 1u )
		return(-12);
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,1u,pages) != SPARK_STATUS_INVALID_ARGUMENT || SparkKvCacheArenaPinResidentTable(&fixture.arena,pages,2u,pages + 1u) != SPARK_STATUS_INVALID_ARGUMENT )
		return(-13);
	if ( SparkKvCacheArenaPinResidentTable(&fixture.arena,0,0u,0) != SPARK_STATUS_OK || SparkKvCacheArenaUnpinResidentTable(&fixture.arena,0,0u) != SPARK_STATUS_OK )
		return(-14);
	pages[1] = pages[0];
	pages[0] = fixture.arena.logical_block_count;
	if ( SparkKvCacheArenaUnpinResidentTable(&fixture.arena,pages,2u) != SPARK_STATUS_INVALID_ARGUMENT || fixture.blocks[pages[1]].residency_reference_count != 0u )
		return(-15);
	return(0);
}

static void SparkTestKvUnassignedResidentCapacityOwnership(void)
{
	SparkTestKvFixture fixture;
	uint32_t block;
	SparkTestKvInitialize(&fixture);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaReserveUnassignedResidentBlocks(&fixture.arena,2u) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaUnassignedResidentBlockCount(&fixture.arena) == 2u);
	assert(SparkKvCacheArenaReserveUnassignedResidentBlocks(&fixture.arena,1u) ==
		SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block) ==
		SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(SparkKvCacheArenaConsumeUnassignedResidentBlocks(&fixture.arena,1u) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaReleaseUnassignedResidentBlocks(&fixture.arena,1u) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaUnassignedResidentBlockCount(&fixture.arena) == 0u);
	assert(SparkKvCacheArenaReleaseUnassignedResidentBlocks(&fixture.arena,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestKvUnassignedOwnershipEvictsReusableResident(void)
{
	SparkTestKvFixture fixture;
	uint32_t block;
	SparkTestKvInitialize(&fixture);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaReserveUnassignedResidentBlocks(&fixture.arena,
		SPARK_TEST_RESIDENT_SLOT_COUNT) == SPARK_STATUS_OK);
	assert(fixture.arena.resident_block_count == 0u);
	assert(fixture.evict_count == 1u);
	assert(fixture.evicted_logical_block == block);
	assert(SparkKvCacheArenaUnassignedResidentBlockCount(&fixture.arena) ==
		SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(SparkKvCacheArenaReleaseUnassignedResidentBlocks(&fixture.arena,
		SPARK_TEST_RESIDENT_SLOT_COUNT) == SPARK_STATUS_OK);
}

#define SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT 8u

typedef struct SparkTestKvOwnershipThread
{
	SparkKvCacheArena *arena;
	atomic_uint *ready_count;
	atomic_uint *start;
	SparkStatus status;
}
SparkTestKvOwnershipThread;

static void *SparkTestKvReserveOwnershipThread(void *context)
{
	SparkTestKvOwnershipThread *thread;
	thread = (SparkTestKvOwnershipThread *)context;
	atomic_fetch_add(thread->ready_count,1u);
	while ( atomic_load(thread->start) == 0u )
		(void)sched_yield();
	thread->status = SparkKvCacheArenaReserveUnassignedResidentBlocks(
		thread->arena,1u);
	return(0);
}

static void SparkTestKvConcurrentOwnershipSaturatesExactly(void)
{
	SparkTestKvFixture fixture;
	SparkTestKvOwnershipThread threads[SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT];
	pthread_t handles[SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT];
	atomic_uint ready_count,start;
	uint32_t capacity_count,index;
	SparkTestKvInitialize(&fixture);
	atomic_init(&ready_count,0u);
	atomic_init(&start,0u);
	memset(threads,0,sizeof(threads));
	for (index=0u; index<SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT; index++)
	{
		threads[index].arena = &fixture.arena;
		threads[index].ready_count = &ready_count;
		threads[index].start = &start;
		assert(pthread_create(&handles[index],0,
			SparkTestKvReserveOwnershipThread,&threads[index]) == 0);
	}
	while ( atomic_load(&ready_count) != SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT )
		(void)sched_yield();
	atomic_store(&start,1u);
	capacity_count = 0u;
	for (index=0u; index<SPARK_TEST_KV_OWNERSHIP_THREAD_COUNT; index++)
	{
		assert(pthread_join(handles[index],0) == 0);
		if ( threads[index].status == SPARK_STATUS_OK )
			capacity_count++;
		else
			assert(threads[index].status == SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	assert(capacity_count == SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(SparkKvCacheArenaUnassignedResidentBlockCount(&fixture.arena) ==
		SPARK_TEST_RESIDENT_SLOT_COUNT);
	assert(SparkKvCacheArenaReleaseUnassignedResidentBlocks(&fixture.arena,
		SPARK_TEST_RESIDENT_SLOT_COUNT) == SPARK_STATUS_OK);
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
	SparkKvCacheBlockView view;
	char path[] = "/tmp/sparkpipe-kv-page-store-XXXXXX";
	uint8_t staging[SPARK_TEST_BLOCK_BYTES],expected[SPARK_TEST_BLOCK_BYTES];
	uint32_t block0,block1,block2,index,slot;
	struct stat file_status;
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
	for (;;)
	{
		(void)sched_yield();
		status = SparkKvPageStoreProgress(&store,&fixture.arena,1u);
		assert(status == SPARK_STATUS_OK);
		status = SparkKvPageStorePrefetch(&store,&fixture.arena,block0);
		assert(status == SPARK_STATUS_OK || status == SPARK_STATUS_BUSY);
		assert(SparkKvCacheArenaResolveBlock(&fixture.arena,block0,&view) ==
			SPARK_STATUS_OK);
		if ( (view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
			break;
	}
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
	assert(SparkKvPageStoreInvalidate(&store,block0,
		fixture.blocks[block0].generation + 1u) == SPARK_STATUS_NOT_FOUND);
	assert(SparkKvPageStoreInvalidate(&store,block0,
		fixture.blocks[block0].generation) == SPARK_STATUS_OK);
	assert(store.backing_page_count == 1u);
	status = SparkKvPageStoreWriteback(&store,block2,
		fixture.blocks[block2].resident_slot_index,
		fixture.blocks[block2].generation,
		fixture.blocks[block2].key_device_address,SPARK_TEST_BLOCK_BYTES,
		0u,0u);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvPageStoreWriteback(&store,block2,
			fixture.blocks[block2].resident_slot_index,
			fixture.blocks[block2].generation,
			fixture.blocks[block2].key_device_address,SPARK_TEST_BLOCK_BYTES,
			0u,0u);
	}
	assert(status == SPARK_STATUS_OK);
	assert(store.backing_page_count == 2u);
	assert(fstat(store.file_descriptor,&file_status) == 0);
	assert((uint64_t)file_status.st_size <= configuration.maximum_backing_bytes);
	SparkKvPageStoreDestroy(&store);
	assert(unlink(path) == 0);
}

typedef struct SparkTestKvBlockedCopy
{
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	uint32_t active;
	uint32_t release;
}
SparkTestKvBlockedCopy;

static SparkStatus SparkTestKvBlockedPageCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	SparkTestKvBlockedCopy *copy;
	copy = (SparkTestKvBlockedCopy *)context;
	assert(copy != 0);
	assert(direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST);
	memcpy(host_address,(const void *)device_address,(size_t)bytes);
	assert(pthread_mutex_lock(&copy->mutex) == 0);
	copy->active = 1u;
	assert(pthread_cond_broadcast(&copy->condition) == 0);
	while ( copy->release == 0u )
		assert(pthread_cond_wait(&copy->condition,&copy->mutex) == 0);
	assert(pthread_mutex_unlock(&copy->mutex) == 0);
	return(SPARK_STATUS_OK);
}

static void SparkTestKvPageStoreInvalidationWaitsForTransfer(void)
{
	SparkTestKvFixture fixture;
	SparkKvPageStoreConfiguration configuration;
	SparkKvPageStore store;
	SparkTestKvBlockedCopy copy;
	char path[] = "/tmp/sparkpipe-kv-page-store-busy-XXXXXX";
	uint8_t staging[SPARK_TEST_BLOCK_BYTES];
	uint32_t block;
	int32_t descriptor;
	SparkStatus status;
	SparkTestKvInitialize(&fixture);
	memset(&copy,0,sizeof(copy));
	assert(pthread_mutex_init(&copy.mutex,0) == 0);
	assert(pthread_cond_init(&copy.condition,0) == 0);
	descriptor = mkstemp(path);
	assert(descriptor >= 0);
	assert(close(descriptor) == 0);
	assert(unlink(path) == 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = SPARK_TEST_LOGICAL_BLOCK_COUNT;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.maximum_backing_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.staging_bytes = sizeof(staging);
	configuration.copy_function = SparkTestKvBlockedPageCopy;
	configuration.copy_context = &copy;
	assert(SparkKvPageStoreInitialize(&store,&configuration) == SPARK_STATUS_OK);
	block = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block) ==
		SPARK_STATUS_OK);
	assert(SparkKvPageStoreWriteback(&store,block,
		fixture.blocks[block].resident_slot_index,
		fixture.blocks[block].generation,
		fixture.blocks[block].key_device_address,SPARK_TEST_BLOCK_BYTES,0u,0u) ==
		SPARK_STATUS_BUSY);
	assert(pthread_mutex_lock(&copy.mutex) == 0);
	while ( copy.active == 0u )
		assert(pthread_cond_wait(&copy.condition,&copy.mutex) == 0);
	assert(SparkKvPageStoreInvalidate(&store,block,
		fixture.blocks[block].generation) == SPARK_STATUS_BUSY);
	copy.release = 1u;
	assert(pthread_cond_broadcast(&copy.condition) == 0);
	assert(pthread_mutex_unlock(&copy.mutex) == 0);
	status = SPARK_STATUS_BUSY;
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvPageStoreWriteback(&store,block,
			fixture.blocks[block].resident_slot_index,
			fixture.blocks[block].generation,
			fixture.blocks[block].key_device_address,SPARK_TEST_BLOCK_BYTES,0u,0u);
	}
	assert(status == SPARK_STATUS_OK);
	assert(SparkKvPageStoreInvalidate(&store,block,
		fixture.blocks[block].generation) == SPARK_STATUS_OK);
	SparkKvPageStoreDestroy(&store);
	assert(unlink(path) == 0);
	assert(pthread_cond_destroy(&copy.condition) == 0);
	assert(pthread_mutex_destroy(&copy.mutex) == 0);
}

static void SparkTestKvPageStoreDirectIoContract(void)
{
	SparkKvPageStoreConfiguration configuration;
	SparkKvPageStore store;
	void *staging;
	SparkStatus status;
	assert(posix_memalign(&staging,
		(size_t)SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT,
		(size_t)SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT) == 0);
	memset(&configuration,0,sizeof(configuration));
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_ANONYMOUS |
		SPARK_KV_PAGE_STORE_FLAG_DIRECT_IO;
	configuration.logical_page_capacity = 1u;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT;
	configuration.maximum_backing_bytes =
		SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT;
	configuration.backing_path = "/tmp";
	configuration.staging_address = (uint8_t *)staging + 1u;
	configuration.staging_bytes = SPARK_KV_PAGE_STORE_DIRECT_IO_ALIGNMENT;
	assert(SparkKvPageStoreInitialize(&store,&configuration) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	configuration.staging_address = staging;
	status = SparkKvPageStoreInitialize(&store,&configuration);
	assert(status == SPARK_STATUS_OK || status == SPARK_STATUS_UNSUPPORTED);
	if ( status == SPARK_STATUS_OK )
		SparkKvPageStoreDestroy(&store);
	free(staging);
}

static SparkStatus SparkTestKvFailingPrefetchCopy(
	void *context,
	uint32_t direction,
	uintptr_t device_address,
	void *host_address,
	uint64_t bytes)
{
	(void)context;
	if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		return(SPARK_STATUS_IO_ERROR);
	if ( direction != SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memcpy(host_address,(const void *)device_address,(size_t)bytes);
	return(SPARK_STATUS_OK);
}

static void SparkTestKvPageStoreFailedPrefetchCancelsReservation(void)
{
	SparkTestKvFixture fixture;
	SparkKvPageStoreConfiguration configuration;
	SparkKvPageStore store;
	char path[] = "/tmp/sparkpipe-kv-page-store-fail-XXXXXX";
	uint8_t staging[SPARK_TEST_BLOCK_BYTES];
	uint32_t attempts,block0;
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
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.maximum_backing_bytes = SPARK_TEST_BLOCK_BYTES;
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.staging_bytes = sizeof(staging);
	configuration.copy_function = SparkTestKvFailingPrefetchCopy;
	assert(SparkKvPageStoreInitialize(&store,&configuration) == SPARK_STATUS_OK);
	fixture.arena.evict_function = SparkKvPageStoreWriteback;
	fixture.arena.evict_context = &store;
	block0 = SparkTestKvAcquire(&fixture);
	assert(SparkKvCacheArenaMarkBlockResident(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	assert(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,block0) ==
		SPARK_STATUS_OK);
	status = SparkKvCacheArenaMarkBlockNonResident(&fixture.arena,block0);
	while ( status == SPARK_STATUS_BUSY )
	{
		(void)sched_yield();
		status = SparkKvCacheArenaMarkBlockNonResident(&fixture.arena,block0);
	}
	assert(status == SPARK_STATUS_OK);
	assert(SparkKvPageStorePrefetch(&store,&fixture.arena,block0) ==
		SPARK_STATUS_BUSY);
	status = SPARK_STATUS_OK;
	for (attempts=0u; attempts<100000u && status==SPARK_STATUS_OK; attempts++)
	{
		(void)sched_yield();
		status = SparkKvPageStoreProgress(&store,&fixture.arena,1u);
		if ( status == SPARK_STATUS_OK )
		{
			status = SparkKvPageStorePrefetch(&store,&fixture.arena,block0);
			if ( status == SPARK_STATUS_BUSY )
				status = SPARK_STATUS_OK;
		}
	}
	assert(status == SPARK_STATUS_IO_ERROR);
	assert(fixture.arena.reserved_block_count == 0u);
	assert((fixture.blocks[block0].flags &
		SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u);
	SparkKvPageStoreDestroy(&store);
	assert(unlink(path) == 0);
}

static SparkStatus SparkTestKvRecordWrite(SparkKvPageStore *store,uint64_t generation,const uint8_t *source)
{
	SparkStatus status = SPARK_STATUS_BUSY;
	uint32_t attempts;
	for (attempts=0u; attempts<100000u && status==SPARK_STATUS_BUSY; attempts++)
	{
		status = SparkKvPageStoreWriteback(store,1u,0u,generation,(uintptr_t)source,SPARK_TEST_BLOCK_BYTES,0u,0u);
		(void)sched_yield();
	}
	return(status);
}

static int32_t SparkTestKvRecordRead(SparkKvPageStore *store,uint8_t *source,uint8_t *output,uint8_t *other)
{
	SparkTestKvFixture fixture;
	uint32_t index,attempts;
	SparkStatus status = SPARK_STATUS_BUSY;
	SparkTestKvInitialize(&fixture);
	for (index=0u; index<SPARK_TEST_BLOCK_BYTES; index++)
		source[index] = (uint8_t)(index + 7u);
	memset(output,0,SPARK_TEST_BLOCK_BYTES);
	memset(other,0,SPARK_TEST_BLOCK_BYTES);
	if ( SparkTestKvRecordWrite(store,7u,source) != SPARK_STATUS_OK || SparkKvPageStoreReadback(store,1u,6u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES) != SPARK_STATUS_NOT_FOUND || SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)output,1u) != SPARK_STATUS_INVALID_ARGUMENT )
		return(-50);
	if ( SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES) != SPARK_STATUS_BUSY || SparkKvPageStoreInvalidate(store,1u,7u) != SPARK_STATUS_BUSY || SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)other,SPARK_TEST_BLOCK_BYTES) != SPARK_STATUS_BUSY )
		return(-51);
	for (attempts=0u; attempts<100000u && status==SPARK_STATUS_BUSY; attempts++)
	{
		if ( SparkKvPageStoreProgress(store,&fixture.arena,1u) != SPARK_STATUS_OK )
			return(-52);
		status = SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES);
		(void)sched_yield();
	}
	if ( status != SPARK_STATUS_OK || memcmp(output,source,SPARK_TEST_BLOCK_BYTES) != 0 || fixture.arena.resident_block_count != 0u || fixture.arena.reserved_block_count != 0u )
		return(-53);
	for (index=0u; index<SPARK_TEST_BLOCK_BYTES; index++)
		if ( other[index] != 0u )
			return(-54);
	if ( SparkKvPageStoreInvalidate(store,1u,7u) != SPARK_STATUS_OK || SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES) != SPARK_STATUS_NOT_FOUND || SparkTestKvRecordWrite(store,8u,source) != SPARK_STATUS_OK || SparkKvPageStoreReadback(store,1u,7u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES) != SPARK_STATUS_NOT_FOUND )
		return(-55);
	store->copy_function = SparkTestKvFailingPrefetchCopy;
	status = SPARK_STATUS_BUSY;
	for (attempts=0u; attempts<100000u && status==SPARK_STATUS_BUSY; attempts++)
	{
		status = SparkKvPageStoreReadback(store,1u,8u,(uintptr_t)output,SPARK_TEST_BLOCK_BYTES);
		(void)sched_yield();
	}
	return(status == SPARK_STATUS_IO_ERROR ? 0 : -56);
}

static int32_t SparkTestKvPageStoreReadback(void)
{
	SparkKvPageStore store;
	SparkKvPageStoreConfiguration configuration = {0};
	uint8_t staging[SPARK_TEST_BLOCK_BYTES],source[SPARK_TEST_BLOCK_BYTES],output[SPARK_TEST_BLOCK_BYTES],other[SPARK_TEST_BLOCK_BYTES];
	char path[] = "/tmp/sparkpipe-kv-readback-XXXXXX";
	int32_t descriptor,status;
	descriptor = mkstemp(path);
	if ( descriptor < 0 || close(descriptor) != 0 || unlink(path) != 0 )
		return(-57);
	configuration.abi_version = SPARK_KV_PAGE_STORE_ABI_VERSION;
	configuration.descriptor_bytes = SPARK_KV_PAGE_STORE_CONFIGURATION_BYTES;
	configuration.flags = SPARK_KV_PAGE_STORE_FLAG_CREATE_EXCLUSIVE;
	configuration.logical_page_capacity = 2u;
	configuration.transfer_capacity = 1u;
	configuration.page_bytes = configuration.maximum_backing_bytes = sizeof(staging);
	configuration.backing_path = path;
	configuration.staging_address = staging;
	configuration.staging_bytes = sizeof(staging);
	if ( SparkKvPageStoreInitialize(&store,&configuration) != SPARK_STATUS_OK )
		return(-58);
	status = SparkTestKvRecordRead(&store,source,output,other);
	SparkKvPageStoreDestroy(&store);
	if ( unlink(path) != 0 )
		return(-59);
	return(status);
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
	lane->request_generation = 1u;
	lane->step_generation = 1u;
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

static int32_t SparkTestKvPagePinnedTransaction(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t logical[4],physical[4],count,mutations,prefix,mutable_page;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,1u,0u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,71u);
	if ( SparkKvPageCacheBeginPinnedLaneTransaction(&fixture.cache,&lane,logical,physical,4u,&count,&mutations) != SPARK_STATUS_OK || count != 1u )
		return(-16);
	prefix = logical[0];
	if ( SparkKvCacheArenaUnpinResidentTable(&fixture.kv.arena,logical,count) != SPARK_STATUS_OK || SparkKvPageCacheCompleteLane(&fixture.cache,&lane) != SPARK_STATUS_OK || SparkKvPageCacheReleaseLane(&fixture.cache,0u,1u) != SPARK_STATUS_OK )
		return(-17);
	SparkTestKvPageLane(&lane,2u,1u,4u,5u);
	SparkTestKvPagePrefix(&lane,4u,71u);
	if ( SparkKvPageCacheBeginPinnedLaneTransaction(&fixture.cache,&lane,logical,physical,1u,&count,&mutations) != SPARK_STATUS_CAPACITY_EXCEEDED )
		return(-18);
	if ( count != 0u || mutations != 0u || fixture.cache.sequences[1].sequence_id != 0u || fixture.kv.blocks[prefix].residency_reference_count != 0u )
		return(-19);
	if ( SparkKvPageCacheBeginPinnedLaneTransaction(&fixture.cache,&lane,logical,physical,4u,&count,&mutations) != SPARK_STATUS_OK || count != 2u || logical[0] != prefix || logical[1] == prefix )
		return(-20);
	mutable_page = logical[1];
	if ( fixture.kv.blocks[prefix].residency_reference_count != 1u || fixture.kv.blocks[mutable_page].residency_reference_count != 1u || physical[0] == physical[1] )
		return(-21);
	if ( SparkKvCacheArenaUnpinResidentTable(&fixture.kv.arena,logical,count) != SPARK_STATUS_OK || SparkKvPageCacheRollbackLaneTransaction(&fixture.cache,&lane,mutations) != SPARK_STATUS_OK )
		return(-22);
	if ( fixture.cache.sequences[1].sequence_id != 0u || fixture.kv.blocks[prefix].residency_reference_count != 0u || (fixture.kv.blocks[mutable_page].flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u )
		return(-23);
	SparkTestKvPageLane(&lane,3u,2u,0u,4u);
	SparkTestKvPagePublish(&lane,4u,71u);
	if ( SparkKvPageCacheBeginPinnedLaneTransaction(&fixture.cache,&lane,logical,physical,4u,&count,&mutations) != SPARK_STATUS_OK )
		return(-24);
	if ( SparkKvCacheArenaUnpinResidentTable(&fixture.kv.arena,logical,count) != SPARK_STATUS_OK || SparkKvPageCacheCompleteLane(&fixture.cache,&lane) != SPARK_STATUS_OK || fixture.cache.deduplicated_page_count != 1u )
		return(-25);
	return(0);
}

typedef struct SparkTestKvTransactions
{
	SparkTestKvPageFixture pages;
	SparkKvLaneTransaction owners[4];
	uint32_t logical[16],physical[16];
	SparkKvLaneTransactions transactions;
	SparkModelDriverCacheLane lanes[3];
	SparkModelDriverAdmissionRequest request;
} SparkTestKvTransactions;

static void SparkTestKvTransactionsInitialize(SparkTestKvTransactions *fixture,uint32_t count)
{
	uint32_t index;
	memset(fixture,0,sizeof(*fixture));
	SparkTestKvPageInitialize(&fixture->pages);
	fixture->transactions.cache = &fixture->pages.cache;
	fixture->transactions.lanes = fixture->owners;
	fixture->transactions.logical_pages = fixture->logical;
	fixture->transactions.physical_pages = fixture->physical;
	fixture->transactions.page_capacity = 4u;
	for (index=0u; index<count; index++)
		SparkTestKvPageLane(&fixture->lanes[index],index + 1u,index,0u,1u);
	fixture->request.descriptor_bytes = sizeof(fixture->request);
	fixture->request.program_id = 1u;
	fixture->request.request_id = 1u;
	fixture->request.submission_id = 1u;
	fixture->request.control_generation = 2u;
	fixture->request.transaction_id = 3u;
	fixture->request.request_generation = 4u;
	fixture->request.step_generation = 5u;
	fixture->request.active_slot_count = count;
	fixture->request.new_token_count = count;
	fixture->request.cache_lane_count = count;
	fixture->request.cache_lanes = fixture->lanes;
	fixture->request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
}

static SparkModelDriverFrame SparkTestKvTransactionFrame(const SparkModelDriverAdmissionRequest *request)
{
	SparkModelDriverFrame frame = {0};
	frame.program_id = request->program_id;
	frame.request_id = request->request_id;
	frame.active_slot_count = request->active_slot_count;
	frame.new_token_count = request->new_token_count;
	frame.cache_lane_count = request->cache_lane_count;
	frame.cache_lanes = request->cache_lanes;
	frame.driver_dispatch_generation = request->control_generation;
	frame.driver_dispatch_cookie0 = request->transaction_id;
	frame.driver_dispatch_cookie1 = request->submission_id;
	frame.flags = SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID;
	return(frame);
}

static int32_t SparkTestKvTransactionsRollbackBatch(void)
{
	SparkTestKvTransactions fixture;
	uint32_t index;
	SparkTestKvTransactionsInitialize(&fixture,3u);
	fixture.lanes[2].resident_sequence_slot = 1u;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_INVALID_ARGUMENT )
		return(-26);
	fixture.lanes[2].resident_sequence_slot = 2u;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_CAPACITY_EXCEEDED )
		return(-27);
	for (index=0u; index<3u; index++)
		if ( fixture.owners[index].phase != SPARK_KV_LANE_TRANSACTION_EMPTY || fixture.pages.cache.sequences[index].sequence_id != 0u )
			return(-28);
	for (index=0u; index<SPARK_TEST_LOGICAL_BLOCK_COUNT; index++)
		if ( fixture.pages.kv.blocks[index].residency_reference_count != 0u )
			return(-29);
	return(0);
}

static int32_t SparkTestKvTransactionsRejectStaleAndInFlight(void)
{
	SparkTestKvTransactions fixture;
	SparkModelDriverFrame frame;
	SparkTestKvTransactionsInitialize(&fixture,2u);
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK || SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-30);
	fixture.request.transaction_id++;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_VALIDATION_FAILED || fixture.owners[0].request.transaction_id != 3u )
		return(-31);
	fixture.request.transaction_id--;
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-32);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	fixture.request.step_generation++;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_VALIDATION_FAILED )
		return(-33);
	fixture.request.step_generation--;
	frame = SparkTestKvTransactionFrame(&fixture.request);
	frame.driver_dispatch_cookie0++;
	if ( SparkKvLaneTransactionsClaim(&fixture.transactions,&frame) != SPARK_STATUS_VALIDATION_FAILED )
		return(-34);
	frame.driver_dispatch_cookie0--;
	if ( SparkKvLaneTransactionsClaim(&fixture.transactions,&frame) != SPARK_STATUS_OK || SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_BUSY )
		return(-35);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_BUSY )
		return(-36);
	if ( SparkKvLaneTransactionsFinish(&fixture.transactions,(uint32_t[]){0u},1u,SPARK_STATUS_OK,0u) != SPARK_STATUS_INVALID_ARGUMENT )
		return(-37);
	if ( SparkKvLaneTransactionsFinish(&fixture.transactions,(uint32_t[]){0u,1u},2u,SPARK_STATUS_OK,0u) != SPARK_STATUS_OK || fixture.pages.cache.sequences[0].next_token_position != 1u || fixture.pages.cache.sequences[1].next_token_position != 1u )
		return(-38);
	fixture.request.cache_lane_count = 1u;
	fixture.request.active_slot_count = 1u;
	fixture.request.new_token_count = 1u;
	fixture.request.step_generation++;
	fixture.lanes[0].sequence_position = 1u;
	fixture.lanes[0].context_token_count = 2u;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK || fixture.owners[0].mutation_flags != 0u )
		return(-39);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-40);
	frame = SparkTestKvTransactionFrame(&fixture.request);
	if ( SparkKvLaneTransactionsClaim(&fixture.transactions,&frame) != SPARK_STATUS_OK || SparkKvLaneTransactionsFinish(&fixture.transactions,(uint32_t[]){0u},1u,SPARK_STATUS_IO_ERROR,0u) != SPARK_STATUS_IO_ERROR || fixture.pages.cache.sequences[0].sequence_id != 0u )
		return(-41);
	return(0);
}

static int32_t SparkTestKvTransactionsPartialCompletionFailure(void)
{
	SparkTestKvTransactions fixture;
	SparkModelDriverFrame frame;
	SparkTestKvTransactionsInitialize(&fixture,2u);
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-42);
	fixture.request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT;
	if ( SparkKvLaneTransactionsAdmit(&fixture.transactions,&fixture.request) != SPARK_STATUS_OK )
		return(-43);
	frame = SparkTestKvTransactionFrame(&fixture.request);
	if ( SparkKvLaneTransactionsClaim(&fixture.transactions,&frame) != SPARK_STATUS_OK )
		return(-44);
	fixture.pages.cache.sequences[1].next_token_position++;
	if ( SparkKvLaneTransactionsFinish(&fixture.transactions,(uint32_t[]){0u,1u},2u,SPARK_STATUS_OK,0u) != SPARK_STATUS_INVALID_ARGUMENT )
		return(-45);
	if ( fixture.pages.cache.sequences[0].sequence_id != 0u || fixture.pages.cache.sequences[1].sequence_id != 0u || fixture.owners[0].phase != SPARK_KV_LANE_TRANSACTION_EMPTY || fixture.owners[1].phase != SPARK_KV_LANE_TRANSACTION_EMPTY )
		return(-46);
	return(0);
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

static void SparkTestKvPageCacheNonMutatingResolutionAndDemand(void)
{
	SparkTestKvPageFixture fixture;
	SparkModelDriverCacheLane lane;
	uint32_t count,demand,mutable_page,pages[4u];
	uint64_t epoch,hit_count,miss_count;
	SparkTestKvPageInitialize(&fixture);
	SparkTestKvPageLane(&lane,1u,0u,0u,1u);
	epoch = fixture.cache.epoch;
	hit_count = fixture.cache.prefix_hit_count;
	miss_count = fixture.cache.prefix_miss_count;
	assert(SparkKvPageCacheResolveLanePages(&fixture.cache,&lane,pages,4u,
		&count) == SPARK_STATUS_OK);
	assert(count == 0u);
	assert(SparkKvPageCacheGetLaneMutablePageDemand(&fixture.cache,&lane,
		&demand) == SPARK_STATUS_OK);
	assert(demand == 1u);
	assert(fixture.cache.epoch == epoch);
	assert(fixture.cache.prefix_hit_count == hit_count);
	assert(fixture.cache.prefix_miss_count == miss_count);
	assert(SparkKvPageCacheBeginLane(&fixture.cache,&lane,&mutable_page) ==
		SPARK_STATUS_OK);
	assert(SparkKvPageCacheGetLaneMutablePageDemand(&fixture.cache,&lane,
		&demand) == SPARK_STATUS_OK);
	assert(demand == 0u);
	assert(SparkKvPageCacheResolveLanePages(&fixture.cache,&lane,pages,4u,
		&count) == SPARK_STATUS_OK);
	assert(count == 1u && pages[0u] == mutable_page);
	SparkTestKvPageLane(&lane,2u,1u,4u,5u);
	SparkTestKvPagePrefix(&lane,4u,99u);
	epoch = fixture.cache.epoch;
	miss_count = fixture.cache.prefix_miss_count;
	assert(SparkKvPageCacheResolveLanePages(&fixture.cache,&lane,pages,4u,
		&count) == SPARK_STATUS_NOT_FOUND);
	assert(fixture.cache.epoch == epoch);
	assert(fixture.cache.prefix_miss_count == miss_count);
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
	int32_t status;
	status = SparkTestKvPageStoreReadback();
	if ( status == 0 )
		status = SparkTestKvPinnedTableUsesPhysicalMapping();
	if ( status == 0 )
		status = SparkTestKvPinnedTableFailurePreservesOtherOwners();
	if ( status == 0 )
		status = SparkTestKvPagePinnedTransaction();
	if ( status == 0 )
		status = SparkTestKvTransactionsRollbackBatch();
	if ( status == 0 )
		status = SparkTestKvTransactionsRejectStaleAndInFlight();
	if ( status == 0 )
		status = SparkTestKvTransactionsPartialCompletionFailure();
	if ( status != 0 )
		return(-status);
	SparkTestKvLogicalBlocksReuseBoundedResidentSlots();
	SparkTestKvEvictionBackpressurePreservesResidentOwner();
	SparkTestKvEvictionIoErrorDegradesInsteadOfWedging();
	SparkTestKvEvictionInternalErrorStaysLoud();
	SparkTestKvPageStoreFullDiskDegradesAndServingContinues();
	SparkTestKvLogicalBlockFreeListReusesReleasedHead();
	SparkTestKvFramePinProtectsResidentBlock();
	SparkTestKvUnassignedResidentCapacityOwnership();
	SparkTestKvUnassignedOwnershipEvictsReusableResident();
	SparkTestKvConcurrentOwnershipSaturatesExactly();
	SparkTestKvPrefetchUsesReservedDeviceSlot();
	SparkTestKvOverlappingPrefetchReservationsAreReferenceCounted();
	SparkTestKvCancelledPrefetchReleasesDeviceSlot();
	SparkTestKvPrefetchCursorOwnsPlanChunking();
	SparkTestKvPageStoreWritesDirtyOnceAndRestores();
	SparkTestKvPageStoreInvalidationWaitsForTransfer();
	SparkTestKvPageStoreDirectIoContract();
	SparkTestKvPageStoreFailedPrefetchCancelsReservation();
	SparkTestPrefixCacheReusesCommittedLogicalBlocks();
	SparkTestKvPageCacheSharesImmutableChains();
	SparkTestKvPageCacheDeduplicatesAndRequiresRelease();
	SparkTestKvPageCacheRejectsMissingPrefix();
	SparkTestKvPageCacheNonMutatingResolutionAndDemand();
	SparkTestKvPageCachePrefetchAndBeginAreTransactional();
	SparkTestKvPageCacheReclaimsColdPrefixUnderPressure();
	return(0);
}
