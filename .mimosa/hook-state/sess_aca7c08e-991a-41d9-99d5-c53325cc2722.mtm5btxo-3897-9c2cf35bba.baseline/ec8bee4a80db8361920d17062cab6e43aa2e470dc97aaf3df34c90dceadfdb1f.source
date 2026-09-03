#include "sparkpipe/spark_kv_pager.h"
#include "sparkpipe/spark_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n", label);
	if ( !condition )
		++failures;
}

#define SLICE_BLOCK_BYTES 4096u
#define SLICE_KEY_BYTES 2048u
#define SLICE_VALUE_BYTES 2048u
#define SLICE_TIER_BASE (1u << 20)
#define SLICE_MAX_READS 16u

typedef struct SliceRead
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint8_t active;
}
SliceRead;

typedef struct SliceDevice
{
	SliceRead reads[SLICE_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;
	uint32_t submits;
	const uint8_t *drive;
	uint64_t drive_base;
	uint64_t drive_bytes;
}
SliceDevice;

static void SliceDeviceReset(SliceDevice *device, uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

static SparkStatus SliceSubmitRead(
	void *context, uint64_t device_offset, void *destination,
	uint32_t bytes, uint64_t *ticket_out)
{
	SliceDevice *device = (SliceDevice *)context;
	uint32_t index;
	if ( bytes != SLICE_BLOCK_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < SLICE_MAX_READS; ++index )
	{
		if ( device->reads[index].active )
			continue;
		device->reads[index].active = 1u;
		device->reads[index].ticket = device->next_ticket++;
		device->reads[index].offset = device_offset;
		device->reads[index].destination = (uint8_t *)destination;
		device->reads[index].polls_left = device->polls_per_read;
		device->submits++;
		*ticket_out = device->reads[index].ticket;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus SlicePollRead(void *context, uint64_t ticket)
{
	SliceDevice *device = (SliceDevice *)context;
	uint32_t index;
	for ( index = 0u; index < SLICE_MAX_READS; ++index )
	{
		SliceRead *read = &device->reads[index];
		if ( !read->active || read->ticket != ticket )
			continue;
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
			memcpy(read->destination,
				device->drive + (read->offset - device->drive_base),
				SLICE_BLOCK_BYTES);
		}
		read->active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

static SparkStatus SliceCancelRead(void *context, uint64_t ticket)
{
	SliceDevice *device = (SliceDevice *)context;
	uint32_t index;
	for ( index = 0u; index < SLICE_MAX_READS; ++index )
	{
		if ( !device->reads[index].active || device->reads[index].ticket != ticket )
			continue;
		device->reads[index].active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

typedef struct SliceModule
{
	uint32_t save_count;
	uint32_t restore_count;
	uint32_t last_block;
}
SliceModule;

static SparkStatus SliceModuleSave(
	void *context, const SparkKvPagerBlockView *view)
{
	SliceModule *module = (SliceModule *)context;
	memcpy(view->host_staging,(void *)(uintptr_t)view->key_device_address,
		view->key_bytes);
	if ( view->value_bytes != 0u )
		memcpy((uint8_t *)view->host_staging + view->key_bytes,
			(void *)(uintptr_t)view->value_device_address,view->value_bytes);
	module->save_count++;
	module->last_block = view->first_block;
	return(SPARK_STATUS_OK);
}

static SparkStatus SliceModuleRestore(
	void *context, const SparkKvPagerBlockView *view)
{
	SliceModule *module = (SliceModule *)context;
	memcpy((void *)(uintptr_t)view->key_device_address,view->host_staging,
		view->key_bytes);
	if ( view->value_bytes != 0u )
		memcpy((void *)(uintptr_t)view->value_device_address,
			(uint8_t *)view->host_staging + view->key_bytes,view->value_bytes);
	module->restore_count++;
	module->last_block = view->first_block;
	return(SPARK_STATUS_OK);
}

#define SLICE_MAX_LOGICAL_BLOCKS 16u
#define SLICE_MAX_RESIDENT_SLOTS 4u

typedef struct SliceFixture
{
	SliceDevice device;
	SparkNvmeTierDevice vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration tier_configuration;
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[SLICE_MAX_LOGICAL_BLOCKS];
	uint32_t resident_slots[SLICE_MAX_RESIDENT_SLOTS];
	uint8_t key_device[SLICE_MAX_RESIDENT_SLOTS * SLICE_KEY_BYTES];
	uint8_t value_device[SLICE_MAX_RESIDENT_SLOTS * SLICE_VALUE_BYTES];
	SparkKvPager pager;
	SparkKvPagerConfiguration pager_configuration;
	SliceModule module;
	_Alignas(64) uint8_t tier_tables[96u * 1024u];
	_Alignas(SLICE_BLOCK_BYTES) uint8_t tier_staging[4u * SLICE_BLOCK_BYTES];
	_Alignas(SLICE_BLOCK_BYTES) uint8_t pager_staging[2u * SLICE_BLOCK_BYTES];
	_Alignas(SLICE_BLOCK_BYTES) uint8_t drive[12u * SLICE_BLOCK_BYTES];
	uint8_t golden[SLICE_MAX_LOGICAL_BLOCKS][SLICE_BLOCK_BYTES];
	uint32_t logical_block_count;
	uint32_t resident_capacity;
	uint32_t tier_slot_count;
	uint32_t park_budget_blocks;
	uint32_t backing_writes;
}
SliceFixture;

static uint64_t SliceFoldDigest(const uint8_t *digest)
{
	uint64_t hash = 0u;
	uint32_t index;
	for ( index = 0u; index < (uint32_t)sizeof(hash); ++index )
		hash |= (uint64_t)digest[index] << (8u * index);
	return(hash | 1u);
}

static void SliceContentFill(uint32_t block_index, uint8_t *block)
{
	uint32_t index;
	for ( index = 0u; index < SLICE_BLOCK_BYTES; ++index )
		block[index] = (uint8_t)(block_index * 0x9Eu + index * 7u + 1u);
}

static void SliceDigest(uint32_t block_index, uint8_t *digest_out)
{
	uint8_t block[SLICE_BLOCK_BYTES];
	SparkSha256Context context;
	SliceContentFill(block_index,block);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,block,SLICE_BLOCK_BYTES);
	SparkSha256Finalize(&context,digest_out);
}

static SparkStatus SliceBackingWrite(
	void *context, uint64_t device_offset, const void *host_staging,
	uint64_t bytes)
{
	SliceFixture *fixture = (SliceFixture *)context;
	if ( device_offset < SLICE_TIER_BASE ||
		device_offset + bytes > SLICE_TIER_BASE + sizeof(fixture->drive) )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memcpy(fixture->drive + (device_offset - SLICE_TIER_BASE),host_staging,
		bytes);
	fixture->backing_writes++;
	return(SPARK_STATUS_OK);
}

static void SliceCheckBudget(SliceFixture *fixture, const char *where)
{
	SparkKvCacheArena *arena = &fixture->arena;
	uint32_t unassigned = atomic_load(&arena->unassigned_resident_block_count);
	uint64_t resident_bytes;
	int ok = arena->resident_block_count <= arena->resident_block_capacity &&
		arena->resident_block_count + arena->reserved_block_count + unassigned <=
			arena->resident_block_capacity &&
		fixture->tier.slots_in_use <= fixture->tier.slot_count;
	resident_bytes = (uint64_t)arena->resident_block_count * SLICE_BLOCK_BYTES;
	ok = ok && resident_bytes <=
		fixture->pager_configuration.device_budget_bytes;
	ok = ok && SparkKvPagerAssertDeviceBudget(&fixture->pager) ==
		SPARK_STATUS_OK;
	if ( !ok )
	{
		printf("  FAIL budget law violated: %s (resident %llu / capacity %u,"
			" tier %llu / %llu)\n",where,
			(unsigned long long)arena->resident_block_count,
			arena->resident_block_capacity,
			(unsigned long long)fixture->tier.slots_in_use,
			(unsigned long long)fixture->tier.slot_count);
		++failures;
	}
}

static SparkStatus SliceOpen(
	SliceFixture *fixture,
	uint32_t logical_block_count,
	uint32_t resident_capacity,
	uint32_t tier_slot_count,
	uint32_t park_budget_blocks)
{
	SparkKvCacheConfiguration arena_configuration;
	SparkNvmeTierConfiguration *tier_configuration;
	SparkKvPagerConfiguration *pager_configuration;
	uint64_t table_bytes;

	memset(fixture,0,sizeof(*fixture));
	fixture->logical_block_count = logical_block_count;
	fixture->resident_capacity = resident_capacity;
	fixture->tier_slot_count = tier_slot_count;
	fixture->park_budget_blocks = park_budget_blocks;
	SliceDeviceReset(&fixture->device,2u);
	fixture->device.drive = fixture->drive;
	fixture->device.drive_base = SLICE_TIER_BASE;
	fixture->device.drive_bytes = sizeof(fixture->drive);
	fixture->vtable.context = &fixture->device;
	fixture->vtable.submit_read = SliceSubmitRead;
	fixture->vtable.poll_read = SlicePollRead;
	fixture->vtable.cancel_read = SliceCancelRead;
	tier_configuration = &fixture->tier_configuration;
	tier_configuration->abi_version = SPARK_NVME_TIER_ABI_VERSION;
	tier_configuration->descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	tier_configuration->budget_bytes =
		(uint64_t)tier_slot_count * SLICE_BLOCK_BYTES;
	tier_configuration->base_offset = SLICE_TIER_BASE;
	tier_configuration->block_bytes = SLICE_BLOCK_BYTES;
	tier_configuration->hash_bucket_count = 32u;
	tier_configuration->staging_buffer_count = 4u;
	tier_configuration->demand_reserve_buffers = 1u;
	tier_configuration->pending_capacity = 16u;
	tier_configuration->device_bytes_per_second = 2048u;
	tier_configuration->step_time_microseconds = 1000000u;
	table_bytes = SparkNvmeTierTableBytes(tier_configuration);
	if ( table_bytes == 0u || table_bytes > sizeof(fixture->tier_tables) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( SparkNvmeTierInitialize(&fixture->tier,tier_configuration,
		&fixture->vtable,fixture->tier_tables,sizeof(fixture->tier_tables),
		fixture->tier_staging,sizeof(fixture->tier_staging)) !=
		SPARK_STATUS_OK )
	{
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	memset(&arena_configuration,0,sizeof(arena_configuration));
	arena_configuration.abi_version = SPARK_KV_CACHE_ABI_VERSION;
	arena_configuration.descriptor_bytes =
		SPARK_KV_CACHE_CONFIGURATION_DESCRIPTOR_BYTES;
	arena_configuration.logical_block_count = logical_block_count;
	arena_configuration.block_token_count = 1u;
	arena_configuration.resident_block_capacity = resident_capacity;
	arena_configuration.layer_count = 1u;
	arena_configuration.kv_head_count = 1u;
	arena_configuration.head_dim = 1024u;
	arena_configuration.bytes_per_scalar = 2u;
	arena_configuration.key_block_stride_bytes = SLICE_KEY_BYTES;
	arena_configuration.value_block_stride_bytes = SLICE_VALUE_BYTES;
	arena_configuration.key_device_base = fixture->key_device;
	arena_configuration.value_device_base = fixture->value_device;
	arena_configuration.blocks = fixture->blocks;
	arena_configuration.resident_slot_logical_block_indices =
		fixture->resident_slots;
	if ( SparkKvCacheArenaInitialize(&fixture->arena,&arena_configuration) !=
		SPARK_STATUS_OK )
	{
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	pager_configuration = &fixture->pager_configuration;
	pager_configuration->abi_version = SPARK_KV_PAGER_ABI_VERSION;
	pager_configuration->descriptor_bytes =
		SPARK_KV_PAGER_CONFIGURATION_DESCRIPTOR_BYTES;
	pager_configuration->arena = &fixture->arena;
	pager_configuration->tier = &fixture->tier;
	pager_configuration->park_budget_blocks = park_budget_blocks;
	pager_configuration->device_budget_bytes =
		(uint64_t)resident_capacity * SLICE_BLOCK_BYTES;
	pager_configuration->staging = fixture->pager_staging;
	pager_configuration->staging_bytes = sizeof(fixture->pager_staging);
	pager_configuration->module_context = &fixture->module;
	pager_configuration->module_save = SliceModuleSave;
	pager_configuration->module_restore = SliceModuleRestore;
	pager_configuration->backing_context = fixture;
	pager_configuration->backing_write = SliceBackingWrite;
	return(SparkKvPagerInitialize(&fixture->pager,pager_configuration));
}

static int32_t SliceFillBlock(SliceFixture *fixture, uint32_t block_index)
{
	SparkKvCacheArena *arena = &fixture->arena;
	SparkKvCacheBlockView view;
	uint32_t acquired;
	SparkStatus status;

	status = SparkKvPagerCommitAdmission(&fixture->pager,1u);
	if ( status != SPARK_STATUS_OK )
		return(0);
	status = SparkKvCacheArenaAcquireBlock(arena,&acquired);
	if ( status != SPARK_STATUS_OK || acquired != block_index )
		return(0);
	if ( SparkKvCacheArenaRetainBlock(arena,acquired) != SPARK_STATUS_OK )
		return(0);
	if ( SparkKvCacheArenaMarkBlockResident(arena,acquired) !=
		SPARK_STATUS_OK )
		return(0);
	if ( SparkKvCacheArenaResolveBlock(arena,acquired,&view) !=
		SPARK_STATUS_OK )
		return(0);
	SliceContentFill(block_index,fixture->golden[block_index]);
	memcpy((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],SLICE_KEY_BYTES);
	memcpy((void *)(uintptr_t)view.value_device_address,
		fixture->golden[block_index] + SLICE_KEY_BYTES,SLICE_VALUE_BYTES);
	if ( SparkKvCacheArenaMarkBlockDirty(arena,acquired) != SPARK_STATUS_OK )
		return(0);
	SliceCheckBudget(fixture,"fill");
	return(1);
}

static int32_t SliceAdmitDemand(
	SliceFixture *fixture, uint32_t demand,
	SparkKvPagerAdmissionDecision *decision_out)
{
	SparkKvPagerAdmission admission;
	SparkStatus status;

	memset(&admission,0,sizeof(admission));
	admission.abi_version = SPARK_KV_PAGER_ADMISSION_ABI_VERSION;
	admission.descriptor_bytes = SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES;
	admission.block_demand = demand;
	status = SparkKvPagerAdmit(&fixture->pager,&admission,decision_out);
	SliceCheckBudget(fixture,"admit");
	return(status == SPARK_STATUS_OK);
}

static int32_t SliceBlockIsResident(SliceFixture *fixture, uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}

static int32_t SliceFlagIsSet(SliceFixture *fixture, uint32_t block_index,
	uint32_t flag)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return((view.flags & flag) != 0u ? 1 : 0);
}

static int32_t SlicePlanesMatchGolden(SliceFixture *fixture,
	uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return memcmp((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],SLICE_KEY_BYTES) == 0 &&
		memcmp((void *)(uintptr_t)view.value_device_address,
			fixture->golden[block_index] + SLICE_KEY_BYTES,
			SLICE_VALUE_BYTES) == 0;
}

static void SlicePageOutHistoryIs(SliceFixture *fixture,
	const uint32_t *expected, uint32_t count, const char *label)
{
	uint32_t index;
	int ok = fixture->pager.statistics.page_out_history_count >= count;
	for ( index = 0u; index < count && ok; ++index )
		ok = fixture->pager.statistics.page_out_history[
			(fixture->pager.statistics.page_out_history_count - count + index) %
			SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY] == expected[index];
	expect(ok,label);
}

int main(void)
{
	setvbuf(stdout,0,_IONBF,0);
	printf("JIT-KV vertical slice\n\nconfiguration fences: the device law binds\n");
	{
		SliceFixture fixture;
		SparkKvPagerConfiguration *configuration;
		expect(SliceOpen(&fixture,16u,4u,12u,12u) == SPARK_STATUS_OK,
			"the slice fixture opens: arena + tier + pager");
		expect(SparkKvPagerBlockBytes(&fixture.pager) == SLICE_BLOCK_BYTES,
			"the pager's block payload is the arena strides == the tier record");
		configuration = &fixture.pager_configuration;
		configuration->device_budget_bytes = SPARK_KV_PAGER_DEVICE_LAW_BYTES +
			(uint64_t)SLICE_BLOCK_BYTES;
		expect(SparkKvPagerInitialize(&fixture.pager,configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a device budget above the 110 GiB law is refused at init");
		configuration->device_budget_bytes =
			(uint64_t)fixture.resident_capacity * SLICE_BLOCK_BYTES;
		configuration->staging_bytes = 2u * SLICE_BLOCK_BYTES - 1u;
		expect(SparkKvPagerInitialize(&fixture.pager,configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"staging smaller than the save + landing planes is refused");
		configuration->staging_bytes = sizeof(fixture.pager_staging);
		expect(SparkKvPagerInitialize(&fixture.pager,configuration) ==
			SPARK_STATUS_OK,
			"re-init over the pager's own arena is accepted (same owner)");
	}

	printf("\nscenario 1: fill beyond the resident budget, LRU paging OUT\n");
	{
		SliceFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		uint32_t lru_order[4] = {0u,2u,3u,1u};
		uint32_t index;
		expect(SliceOpen(&fixture,16u,4u,12u,12u) == SPARK_STATUS_OK,
			"the fixture opens");
		expect(SliceAdmitDemand(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.reservation_held == 4u && decision.park_evictions == 0u,
			"lane A: demand 4 admits against the empty budget");
		for ( index = 0u; index < 4u; ++index )
			expect(SliceFillBlock(&fixture,index),
				"lane A block fills, resident within budget");
		expect(fixture.arena.resident_block_count == 4u,
			"the device budget is exactly filled by lane A");
		expect(fixture.pager.statistics.page_out_count == 0u,
			"nothing parked while the load fits");
		expect(SparkKvCacheArenaRetainBlock(&fixture.arena,1u) ==
			SPARK_STATUS_OK && SparkKvCacheArenaReleaseBlockReference(
				&fixture.arena,1u) == SPARK_STATUS_OK,
			"lane A touches block 1 (recency bump)");
		expect(SliceAdmitDemand(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 4u,
			"lane B: demand 4 beyond the budget admits BY PARKING");
		SlicePageOutHistoryIs(&fixture,lru_order,4u,
			"victims are the LRU set {0,2,3,1}: touched 1 evicted LAST");
		expect(fixture.pager.statistics.page_out_count == 4u,
			"every eviction was a pager page-out");
		expect(fixture.tier.statistics.slots_in_use == 4u,
			"the tier holds exactly the four parked records");
		expect(fixture.tier.statistics.publishes == 4u &&
			fixture.tier.statistics.digest_mismatches == 0u,
			"every write-back committed under its payload digest");
		expect(fixture.backing_writes == 4u,
			"every new record went through the backing write leg");
		for ( index = 0u; index < 4u; ++index )
		{
			expect(SliceFlagIsSet(&fixture,index,
				SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) &&
				!SliceBlockIsResident(&fixture,index),
				"a parked block is backing-valid and non-resident");
			expect(fixture.arena.blocks[index].reference_count == 1u,
				"lane A still owns its parked blocks");
		}
		for ( index = 4u; index < 8u; ++index )
			expect(SliceFillBlock(&fixture,index),
				"lane B block fills into the freed budget");
		expect(fixture.arena.resident_block_count == 4u,
			"the budget is full again with lane B");
		SliceCheckBudget(&fixture,"scenario 1 end");
	}

	printf("\nscenario 2: rewind - the evicted block pages IN, bit-exact\n");
	{
		SliceFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkNvmeTierStatistics tier_statistics;
		uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
		uint32_t index,status_ok = 1u;
		expect(SliceOpen(&fixture,16u,4u,12u,12u) == SPARK_STATUS_OK,
			"the fixture opens");
		expect(SliceAdmitDemand(&fixture,4u,&decision), "lane A admits");
		for ( index = 0u; index < 4u; ++index )
			SliceFillBlock(&fixture,index);
		expect(SparkKvCacheArenaRetainBlock(&fixture.arena,1u) ==
			SPARK_STATUS_OK && SparkKvCacheArenaReleaseBlockReference(
				&fixture.arena,1u) == SPARK_STATUS_OK, "touch block 1");
		expect(SliceAdmitDemand(&fixture,4u,&decision),
			"lane B admits by parking lane A");
		for ( index = 4u; index < 8u; ++index )
			SliceFillBlock(&fixture,index);
		SliceDigest(1u,digest);
		{
			uint32_t attempt;
			SparkStatus status = SPARK_STATUS_INTERNAL_ERROR;
			for ( attempt = 0u; attempt < 4u; ++attempt )
			{
				status = SparkKvPagerRestoreBlock(&fixture.pager,1u,digest);
				if ( status == SPARK_STATUS_OK )
					break;
				if ( status != SPARK_STATUS_BUSY )
					break;
			}
			expect(status == SPARK_STATUS_OK,
				"the parked block restores on request");
			status_ok = status == SPARK_STATUS_OK;
		}
		expect(fixture.pager.statistics.page_in_count == 1u && status_ok,
			"exactly one page-in landed");
		expect(SliceBlockIsResident(&fixture,1u),
			"the rewound block is resident again");
		expect(SliceFlagIsSet(&fixture,1u,
			SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID),
			"the restored block keeps its backing copy (re-park is free)");
		expect(SlicePlanesMatchGolden(&fixture,1u),
			"BIT-EXACT: the device planes equal the true payload");
		expect(fixture.module.restore_count == 1u,
			"the bytes arrived through the module restore op");
		SparkNvmeTierGetStatistics(&fixture.tier,&tier_statistics);
		expect(tier_statistics.digest_verifications >= 1u &&
			tier_statistics.digest_mismatches == 0u,
			"the tier landing was digest-verified, zero mismatches");
		expect(tier_statistics.demand_loads >= 1u,
			"the restore issued a real demand read");
		expect(fixture.pager.statistics.page_out_count == 5u,
			"making room paged the LRU resident out (lane B's oldest)");
		expect(fixture.tier.statistics.slots_in_use == 5u,
			"the tier now holds five records");
		expect(SparkKvCacheArenaMarkBlockDirty(&fixture.arena,1u) ==
			SPARK_STATUS_OK,
			"the restored block is re-dirtied (bytes unchanged)");
		expect(SliceAdmitDemand(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane C admits by parking");
		expect(fixture.pager.statistics.page_out_deduplicated == 1u,
			"the re-park found identical bytes on the tier: no write");
		expect(fixture.backing_writes == 8u,
			"only genuinely new payloads crossed the backing write leg");
		expect(fixture.tier.statistics.slots_in_use == 8u,
			"the dedup reused the existing tier record");
		for ( index = 8u; index < 12u; ++index )
			SliceFillBlock(&fixture,index);
		SliceCheckBudget(&fixture,"scenario 2 end");
	}

	printf("\nscenario 3: backpressure - saturated device refuses, never wedges\n");
	{
		SliceFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		uint32_t index;
		expect(SliceOpen(&fixture,8u,3u,4u,4u) == SPARK_STATUS_OK,
			"the fixture opens (capacity 3, park horizon 4)");
		expect(SliceAdmitDemand(&fixture,3u,&decision), "lane P admits");
		for ( index = 0u; index < 3u; ++index )
			SliceFillBlock(&fixture,index);
		for ( index = 0u; index < 3u; ++index )
			expect(SparkKvCacheArenaPinResidentBlock(&fixture.arena,index) ==
				SPARK_STATUS_OK,
				"lane P's residents are pinned (an active set)");
		expect(SliceAdmitDemand(&fixture,1u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED,
			"the pinned budget QUEUES the offer: no evictable resident");
		expect(fixture.pager.statistics.admission_queued_device == 1u,
			"the refusal is named: device full of protected residents");
		for ( index = 0u; index < 10u; ++index )
		{
			expect(SliceAdmitDemand(&fixture,1u,&decision) &&
				decision.outcome == SPARK_KV_PAGER_QUEUED,
				"repeated offers keep queueing - the queue, not a wedge");
		}
		expect(fixture.pager.statistics.page_out_count == 0u,
			"nothing was parked: backpressure never thrashes");
		expect(fixture.arena.resident_block_count == 3u,
			"the pinned set is untouched");
		expect(SparkKvPagerReleaseAdmission(&fixture.pager,1u) ==
			SPARK_STATUS_INVALID_ARGUMENT &&
			SparkKvCacheArenaUnassignedResidentBlockCount(&fixture.arena) == 0u,
			"a queued offer holds no reservation to leak");
		expect(SparkKvCacheArenaUnpinResidentBlock(&fixture.arena,2u) ==
			SPARK_STATUS_OK, "lane P releases one block (pressure drops)");
		expect(SliceAdmitDemand(&fixture,1u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 1u,
			"the same offer now admits by parking the released block");
		expect(SliceBlockIsResident(&fixture,0u) &&
			SliceBlockIsResident(&fixture,1u),
			"the pinned blocks survived: protected residents were never victims");
		expect(SliceFillBlock(&fixture,3u), "the admitted lane fills its block");
		SliceCheckBudget(&fixture,"scenario 3 end");
	}

	printf("\nscenario 4: backing horizon + tier BUSY stay retryable, no drop\n");
	{
		SliceFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkNvmeTierStatistics tier_statistics;
		uint8_t digest0[SPARK_KV_PAGER_DIGEST_BYTES];
		uint32_t index;
		expect(SliceOpen(&fixture,6u,2u,2u,2u) == SPARK_STATUS_OK,
			"the fixture opens (capacity 2, tier 2 records, horizon 2)");
		expect(SliceAdmitDemand(&fixture,2u,&decision), "lane R admits");
		for ( index = 0u; index < 2u; ++index )
			SliceFillBlock(&fixture,index);
		expect(SliceAdmitDemand(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane T admits by parking lane R entirely");
		for ( index = 2u; index < 4u; ++index )
			SliceFillBlock(&fixture,index);
		expect(fixture.tier.statistics.slots_in_use == 2u,
			"the park horizon is exactly full");
		expect(SliceAdmitDemand(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED,
			"a full park horizon QUEUES: backing horizon reached");
		expect(fixture.pager.statistics.admission_queued_backing == 1u,
			"the refusal is named: backing horizon");
		expect(SliceAdmitDemand(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED,
			"it stays queued - the horizon does not lie or thrash");
		SliceDigest(0u,digest0);
		{
			uint8_t digest1[SPARK_KV_PAGER_DIGEST_BYTES];
			SliceDigest(1u,digest1);
			expect(SparkNvmeTierPin(&fixture.tier,SliceFoldDigest(digest0),
				digest0,1) == SPARK_STATUS_OK &&
				SparkNvmeTierPin(&fixture.tier,SliceFoldDigest(digest1),
					digest1,1) == SPARK_STATUS_OK,
				"both tier records pin");
		}
		{
			SparkStatus status = SparkKvPagerRestoreBlock(&fixture.pager,0u,
				digest0);
			expect(status == SPARK_STATUS_BUSY,
				"the rewind answers BUSY while the tier is pinned full");
			expect(!SliceBlockIsResident(&fixture,0u),
				"the block stays parked: no half-restored residency");
			expect(fixture.arena.write_back_degraded_block_count == 0u,
				"nothing was degraded: BUSY is backpressure, not failure");
		}
		expect(SparkNvmeTierPin(&fixture.tier,SliceFoldDigest(digest0),
			digest0,0) == SPARK_STATUS_OK, "one record unpins");
		{
			SparkStatus status = SparkKvPagerRestoreBlock(&fixture.pager,0u,
				digest0);
			expect(status == SPARK_STATUS_OK,
				"the retry restores the block once the tier yields");
			expect(SlicePlanesMatchGolden(&fixture,0u),
				"BIT-EXACT after the retry: digest-verified, never stale");
		}
		SparkNvmeTierGetStatistics(&fixture.tier,&tier_statistics);
		expect(tier_statistics.digest_mismatches == 0u,
			"zero digest mismatches across the whole fight");
		SliceCheckBudget(&fixture,"scenario 4 end");
	}

	printf("\nscenario 5: the park horizon recovers with the load\n");
	{
		SliceFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		uint8_t digest1[SPARK_KV_PAGER_DIGEST_BYTES];
		uint32_t index;
		expect(SliceOpen(&fixture,16u,4u,12u,12u) == SPARK_STATUS_OK,
			"the fixture opens");
		expect(SliceAdmitDemand(&fixture,4u,&decision), "lane A admits");
		for ( index = 0u; index < 4u; ++index )
			SliceFillBlock(&fixture,index);
		expect(SparkKvCacheArenaRetainBlock(&fixture.arena,1u) ==
			SPARK_STATUS_OK && SparkKvCacheArenaReleaseBlockReference(
				&fixture.arena,1u) == SPARK_STATUS_OK, "touch block 1");
		expect(SliceAdmitDemand(&fixture,4u,&decision), "lane B parks lane A");
		for ( index = 4u; index < 8u; ++index )
			SliceFillBlock(&fixture,index);
		SliceDigest(1u,digest1);
		expect(SparkKvPagerRestoreBlock(&fixture.pager,1u,digest1) ==
			SPARK_STATUS_OK,
			"lane A rewinds block 1: the tier read lands digest-verified");
		expect(SlicePlanesMatchGolden(&fixture,1u),
			"the rewound block is bit-exact");
		expect(SliceAdmitDemand(&fixture,4u,&decision), "lane C parks; block 1 re-parks as a dedup");
		for ( index = 8u; index < 12u; ++index )
			SliceFillBlock(&fixture,index);
		expect(SliceAdmitDemand(&fixture,4u,&decision), "lane D parks lane C");
		for ( index = 12u; index < 16u; ++index )
			SliceFillBlock(&fixture,index);
		expect(fixture.tier.statistics.slots_in_use == fixture.tier.slot_count,
			"the horizon is exactly full of parked records");
		expect(SliceAdmitDemand(&fixture,1u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED,
			"the horizon full, the offer queues");
		for ( index = 0u; index < 4u; ++index )
		{
			expect(SparkKvCacheArenaReleaseBlockReference(&fixture.arena,
				16u - 1u - index) == SPARK_STATUS_OK,
				"lane D completes: references drop");
			expect(SparkKvCacheArenaFreeBlock(&fixture.arena,
				16u - 1u - index) == SPARK_STATUS_OK,
				"lane D's blocks return to the free list");
		}
		SliceCheckBudget(&fixture,"after lane D completes");
		expect(fixture.arena.resident_block_count == 0u,
			"the device budget drains with the completed lane");
		expect(SliceAdmitDemand(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 0u,
			"lane E admits WITHOUT parking: free budget, no thrash");
		expect(fixture.pager.statistics.page_out_count == 12u,
			"no new page-outs: the admitted load fit the freed budget"
			" (the restored block re-parks silently: clean + backed)");
		for ( index = 0u; index < 2u; ++index )
			expect(SliceFillBlock(&fixture,12u + index),
				"lane E fills recycled blocks");
		SliceCheckBudget(&fixture,"scenario 5 end");
	}

	printf("\n");
	if ( failures != 0 )
		printf("FAIL: %d check(s) failed\n",failures);
	else
		printf("JIT-KV SLICE: all proofs green\n");
	return(failures != 0 ? 1 : 0);
}
