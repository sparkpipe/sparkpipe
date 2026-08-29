// The JIT-KV family wiring, end to end on a host (docs/JIT_KV_DESIGN.md
// step 2, docs/JIT_KV_RESPONSE.md W1 + C2): the decode stage module's
// KV_BLOCKS_SAVE_OUT / KV_BLOCKS_RESTORE_IN frame ops behind the pager's
// module seam, the deployment parkability condition over the arena
// selector's shared predicate, and the C2 dispatch gate - a parked block's
// dispatch waits for restore completion.
//
// The proofs:
//   1. the frame ops run the real park/rewind loop (TERM copies against
//      host-mapped device planes), staging exactly the op codes and plane
//      byte counts the design names, bit-exact through the seam;
//   2. the device-plane backend is spark-gated: without the spark-side
//      module receipt the copy is STAGED and refused UNSUPPORTED - loud,
//      nothing run, staging untouched, nothing degraded; with the receipt
//      the staged op reaches the frame-submit seam;
//   3. parkability is ONE predicate: the deployment's active set is
//      protected exactly as the selector protects it, admission's parkable
//      pool is the same count, and a parked-but-still-listed-active block
//      fails the deployment condition until its lane re-pins it;
//   4. C2: dispatch answers READY only on a verified resident block,
//      QUEUES (healthily, repeatedly) while the tier is saturated, and
//      answers RECOMPUTE for a degraded block - never a wedge, never a
//      dispatch on partial state.
#include "spark_dsv4_jit_kv.h"
#include "sparkpipe/spark_sha256.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n",label);
	if ( !condition )
		++failures;
}

#define WIRE_BLOCK_BYTES 4096u
#define WIRE_KEY_BYTES 2048u
#define WIRE_VALUE_BYTES 2048u
#define WIRE_TIER_BASE (1u << 20)
#define WIRE_MAX_READS 16u

typedef struct WireRead
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint8_t active;
}
WireRead;

typedef struct WireDevice
{
	WireRead reads[WIRE_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;
	const uint8_t *drive;
	uint64_t drive_base;
}
WireDevice;

static void WireDeviceReset(WireDevice *device,uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

static SparkStatus WireSubmitRead(
	void *context,uint64_t device_offset,void *destination,
	uint32_t bytes,uint64_t *ticket_out)
{
	WireDevice *device = (WireDevice *)context;
	uint32_t index;
	if ( bytes != WIRE_BLOCK_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < WIRE_MAX_READS; ++index )
	{
		if ( device->reads[index].active )
			continue;
		device->reads[index].active = 1u;
		device->reads[index].ticket = device->next_ticket++;
		device->reads[index].offset = device_offset;
		device->reads[index].destination = (uint8_t *)destination;
		device->reads[index].polls_left = device->polls_per_read;
		*ticket_out = device->reads[index].ticket;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus WirePollRead(void *context,uint64_t ticket)
{
	WireDevice *device = (WireDevice *)context;
	uint32_t index;
	for ( index = 0u; index < WIRE_MAX_READS; ++index )
	{
		WireRead *read = &device->reads[index];
		if ( !read->active || read->ticket != ticket )
			continue;
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
			memcpy(read->destination,
				device->drive + (read->offset - device->drive_base),
				WIRE_BLOCK_BYTES);
		}
		read->active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

static SparkStatus WireCancelRead(void *context,uint64_t ticket)
{
	WireDevice *device = (WireDevice *)context;
	uint32_t index;
	for ( index = 0u; index < WIRE_MAX_READS; ++index )
	{
		if ( !device->reads[index].active ||
			device->reads[index].ticket != ticket )
		{
			continue;
		}
		device->reads[index].active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

/* Host stand-in for the frame-context submission of the spark-gated
   backend: the staged descriptor is recorded, and the SAME byte movement
   runs against the host mappings, because under the stub the device plane
   IS host memory. The real device-plane copy executes only behind the
   module receipt on a spark; nothing here claims to be it. */
typedef struct WireSpark
{
	uint64_t submissions;
	uint32_t last_op_code;
}
WireSpark;

static SparkStatus WireSparkSubmit(void *context,const SparkDsv4KvFrameOp *op)
{
	WireSpark *spark = (WireSpark *)context;
	spark->submissions += 1u;
	spark->last_op_code = op->op_code;
	return SparkDsv4KvFramesSubmitHostCopy(0,op);
}

#define WIRE_MAX_LOGICAL_BLOCKS 16u
#define WIRE_MAX_RESIDENT_SLOTS 4u

typedef struct WireFixture
{
	WireDevice device;
	WireSpark spark;
	SparkNvmeTierDevice vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration tier_configuration;
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[WIRE_MAX_LOGICAL_BLOCKS];
	uint32_t resident_slots[WIRE_MAX_RESIDENT_SLOTS];
	uint8_t key_device[WIRE_MAX_RESIDENT_SLOTS * WIRE_KEY_BYTES];
	uint8_t value_device[WIRE_MAX_RESIDENT_SLOTS * WIRE_VALUE_BYTES];
	SparkKvPager pager;
	SparkKvPagerConfiguration pager_configuration;
	SparkDsv4KvFrames frames;
	SparkDsv4KvFramesConfiguration frames_configuration;
	uint32_t backing_failures_left;
	uint64_t backing_writes;
	_Alignas(64) uint8_t tier_tables[96u * 1024u];
	_Alignas(WIRE_BLOCK_BYTES) uint8_t tier_staging[4u * WIRE_BLOCK_BYTES];
	_Alignas(WIRE_BLOCK_BYTES) uint8_t pager_staging[2u * WIRE_BLOCK_BYTES];
	_Alignas(WIRE_BLOCK_BYTES) uint8_t drive[12u * WIRE_BLOCK_BYTES];
	uint8_t golden[WIRE_MAX_LOGICAL_BLOCKS][WIRE_BLOCK_BYTES];
	uint32_t logical_block_count;
	uint32_t resident_capacity;
}
WireFixture;

static void WireContentFill(uint32_t block_index,uint8_t *block)
{
	uint32_t index;
	for ( index = 0u; index < WIRE_BLOCK_BYTES; ++index )
		block[index] = (uint8_t)(block_index * 0x9Eu + index * 7u + 1u);
}

static void WireDigest(uint32_t block_index,uint8_t *digest_out)
{
	uint8_t block[WIRE_BLOCK_BYTES];
	SparkSha256Context context;
	WireContentFill(block_index,block);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,block,WIRE_BLOCK_BYTES);
	SparkSha256Finalize(&context,digest_out);
}

static SparkStatus WireBackingWrite(
	void *context,uint64_t device_offset,const void *host_staging,
	uint64_t bytes)
{
	WireFixture *fixture = (WireFixture *)context;
	if ( device_offset < WIRE_TIER_BASE ||
		device_offset + bytes > WIRE_TIER_BASE + sizeof(fixture->drive) )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( fixture->backing_failures_left != 0u )
	{
		fixture->backing_failures_left -= 1u;
		return(SPARK_STATUS_IO_ERROR);
	}
	memcpy(fixture->drive + (device_offset - WIRE_TIER_BASE),host_staging,
		bytes);
	fixture->backing_writes++;
	return(SPARK_STATUS_OK);
}

/* The budget law, asserted after every state change in every scenario. */
static void WireCheckBudget(WireFixture *fixture,const char *where)
{
	SparkKvCacheArena *arena = &fixture->arena;
	uint32_t unassigned = atomic_load(&arena->unassigned_resident_block_count);
	uint64_t resident_bytes;
	int ok = arena->resident_block_count <= arena->resident_block_capacity &&
		arena->resident_block_count + arena->reserved_block_count +
			unassigned <= arena->resident_block_capacity &&
		fixture->tier.slots_in_use <= fixture->tier.slot_count;
	resident_bytes = (uint64_t)arena->resident_block_count * WIRE_BLOCK_BYTES;
	ok = ok && resident_bytes <=
		fixture->pager_configuration.device_budget_bytes;
	ok = ok && SparkKvPagerAssertDeviceBudget(&fixture->pager) ==
		SPARK_STATUS_OK;
	if ( !ok )
	{
		printf("  FAIL budget law violated: %s\n",where);
		++failures;
	}
}

static SparkStatus WireOpen(
	WireFixture *fixture,
	uint32_t logical_block_count,
	uint32_t resident_capacity,
	uint32_t tier_slot_count,
	uint32_t park_budget_blocks,
	uint32_t frames_backend)
{
	SparkKvCacheConfiguration arena_configuration;
	SparkNvmeTierConfiguration *tier_configuration;
	SparkKvPagerConfiguration *pager_configuration;
	SparkDsv4KvFramesConfiguration *frames_configuration;
	uint64_t table_bytes;

	memset(fixture,0,sizeof(*fixture));
	fixture->logical_block_count = logical_block_count;
	fixture->resident_capacity = resident_capacity;
	WireDeviceReset(&fixture->device,2u);
	fixture->device.drive = fixture->drive;
	fixture->device.drive_base = WIRE_TIER_BASE;
	fixture->vtable.context = &fixture->device;
	fixture->vtable.submit_read = WireSubmitRead;
	fixture->vtable.poll_read = WirePollRead;
	fixture->vtable.cancel_read = WireCancelRead;
	tier_configuration = &fixture->tier_configuration;
	tier_configuration->abi_version = SPARK_NVME_TIER_ABI_VERSION;
	tier_configuration->descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	tier_configuration->budget_bytes =
		(uint64_t)tier_slot_count * WIRE_BLOCK_BYTES;
	tier_configuration->base_offset = WIRE_TIER_BASE;
	tier_configuration->block_bytes = WIRE_BLOCK_BYTES;
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
	arena_configuration.key_block_stride_bytes = WIRE_KEY_BYTES;
	arena_configuration.value_block_stride_bytes = WIRE_VALUE_BYTES;
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
	frames_configuration = &fixture->frames_configuration;
	frames_configuration->abi_version = SPARK_DSV4_JIT_KV_ABI_VERSION;
	frames_configuration->descriptor_bytes =
		SPARK_DSV4_KV_FRAMES_CONFIGURATION_DESCRIPTOR_BYTES;
	frames_configuration->backend = frames_backend;
	frames_configuration->key_block_stride_bytes = WIRE_KEY_BYTES;
	frames_configuration->value_block_stride_bytes = WIRE_VALUE_BYTES;
	if ( frames_backend == SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS )
	{
		frames_configuration->submit_context = &fixture->spark;
		frames_configuration->submit = WireSparkSubmit;
	}
	if ( SparkDsv4KvFramesInitialize(&fixture->frames,
		frames_configuration) != SPARK_STATUS_OK )
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
		(uint64_t)resident_capacity * WIRE_BLOCK_BYTES;
	pager_configuration->staging = fixture->pager_staging;
	pager_configuration->staging_bytes = sizeof(fixture->pager_staging);
	pager_configuration->module_context = &fixture->frames;
	pager_configuration->module_save = SparkDsv4KvFramesSave;
	pager_configuration->module_restore = SparkDsv4KvFramesRestore;
	pager_configuration->backing_context = fixture;
	pager_configuration->backing_write = WireBackingWrite;
	return(SparkKvPagerInitialize(&fixture->pager,pager_configuration));
}

/* Admit + fill one block: commit the reservation immediately before the
   block becomes resident, acquire, mark residency, write the planes
   through the block view, dirty. */
static int32_t WireFillBlock(WireFixture *fixture,uint32_t block_index)
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
	WireContentFill(block_index,fixture->golden[block_index]);
	memcpy((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],WIRE_KEY_BYTES);
	memcpy((void *)(uintptr_t)view.value_device_address,
		fixture->golden[block_index] + WIRE_KEY_BYTES,WIRE_VALUE_BYTES);
	if ( SparkKvCacheArenaMarkBlockDirty(arena,acquired) != SPARK_STATUS_OK )
		return(0);
	WireCheckBudget(fixture,"fill");
	return(1);
}

static int32_t WireAdmit(
	WireFixture *fixture,uint32_t demand,
	SparkKvPagerAdmissionDecision *decision_out)
{
	SparkKvPagerAdmission admission;
	memset(&admission,0,sizeof(admission));
	admission.abi_version = SPARK_KV_PAGER_ADMISSION_ABI_VERSION;
	admission.descriptor_bytes = SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES;
	admission.block_demand = demand;
	WireCheckBudget(fixture,"admit");
	return(SparkKvPagerAdmit(&fixture->pager,&admission,decision_out) ==
		SPARK_STATUS_OK);
}

static int32_t WireBlockIsResident(WireFixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}

static int32_t WirePlanesMatchGolden(WireFixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return memcmp((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],WIRE_KEY_BYTES) == 0 &&
		memcmp((void *)(uintptr_t)view.value_device_address,
			fixture->golden[block_index] + WIRE_KEY_BYTES,
			WIRE_VALUE_BYTES) == 0;
}

/* Same fold as the pager: digest bytes -> the tier's bucket key, so the
   test can pin exactly the records the pager parked. */
static uint64_t WireFoldDigest(const uint8_t *digest)
{
	uint64_t hash = 0u;
	uint32_t index;
	for ( index = 0u; index < (uint32_t)sizeof(hash); ++index )
		hash |= (uint64_t)digest[index] << (8u * index);
	return(hash | 1u);
}

static int32_t WireDispatchOffer(
	WireFixture *fixture,
	uint32_t block_index,
	SparkKvPagerDispatchDecision *decision_out)
{
	SparkKvPagerDispatch dispatch;

	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_KV_PAGER_DISPATCH_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_KV_PAGER_DISPATCH_DESCRIPTOR_BYTES;
	dispatch.logical_block_index = block_index;
	WireDigest(block_index,dispatch.content_digest);
	memset(decision_out,0,sizeof(*decision_out));
	if ( SparkKvPagerDispatchBlock(&fixture->pager,&dispatch,
		decision_out) != SPARK_STATUS_OK )
	{
		return(0);
	}
	WireCheckBudget(fixture,"dispatch");
	return(1);
}

/* The dispatch queue's re-offer: C2's QUEUED means "restore not complete
   yet" (a transient tier BUSY inside the gate); the dispatcher offers the
   work item again. Bounded, and every non-READY answer must be QUEUED -
   never a drop, never a wedge, never a dispatch on partial state. */
static int32_t WireDispatchUntilReady(
	WireFixture *fixture,
	uint32_t block_index,
	SparkKvPagerDispatchDecision *decision_out)
{
	uint32_t attempt;
	for ( attempt = 0u; attempt < 8u; ++attempt )
	{
		if ( !WireDispatchOffer(fixture,block_index,decision_out) )
			return(0);
		if ( decision_out->outcome == SPARK_KV_PAGER_DISPATCH_READY )
			return(1);
		if ( decision_out->outcome !=
			SPARK_KV_PAGER_DISPATCH_QUEUED )
			return(0);
	}
	return(0);
}

static void WirePageOutHistoryIs(WireFixture *fixture,
	const uint32_t *expected,uint32_t count,const char *label)
{
	uint32_t index;
	int ok = fixture->pager.statistics.page_out_history_count >= count;
	for ( index = 0u; index < count && ok; ++index )
		ok = fixture->pager.statistics.page_out_history[
			(fixture->pager.statistics.page_out_history_count - count +
				index) % SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY] ==
			expected[index];
	expect(ok,label);
}

int main(void)
{
	setvbuf(stdout,0,_IONBF,0);
	printf("JIT-KV family wiring\n\nconfiguration fences: the frame ops bind\n");
	{
		WireFixture fixture;
		SparkDsv4KvFramesConfiguration configuration;
		uint8_t key_plane[WIRE_KEY_BYTES];
		uint8_t value_plane[WIRE_VALUE_BYTES];
		uint8_t staging[WIRE_BLOCK_BYTES];
		SparkKvPagerBlockView view;
		expect(WireOpen(&fixture,16u,4u,12u,12u,
			SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY) == SPARK_STATUS_OK,
			"the wire fixture opens: arena + tier + pager + frame ops");
		expect(fixture.frames.submit ==
			SparkDsv4KvFramesSubmitHostCopy,
			"the TERM backend installs the host copy primitive");
		configuration = fixture.frames_configuration;
		configuration.abi_version = 0u;
		expect(SparkDsv4KvFramesInitialize(&fixture.frames,&configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a frame-ops configuration with a wrong abi is refused");
		configuration = fixture.frames_configuration;
		configuration.backend =
			SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS;
		configuration.submit = 0;
		expect(SparkDsv4KvFramesInitialize(&fixture.frames,&configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"the device-plane backend refuses without its submission:"
			" that installs at the spark-side module open");
		configuration.reserved0 = 1u;
		expect(SparkDsv4KvFramesInitialize(&fixture.frames,&configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a non-zero reserved field is refused");
		configuration = fixture.frames_configuration;
		configuration.key_block_stride_bytes = 0u;
		expect(SparkDsv4KvFramesInitialize(&fixture.frames,&configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a zero key stride is refused");
		/* the view fence: the seam validates the pager's block view */
		memset(&view,0,sizeof(view));
		view.abi_version = SPARK_KV_PAGER_ABI_VERSION;
		view.descriptor_bytes = SPARK_KV_PAGER_BLOCK_VIEW_DESCRIPTOR_BYTES;
		view.block_count = 1u;
		view.key_bytes = WIRE_KEY_BYTES;
		view.value_bytes = WIRE_VALUE_BYTES;
		view.key_device_address = (uintptr_t)key_plane;
		view.value_device_address = (uintptr_t)value_plane;
		view.host_staging = staging;
		memset(key_plane,0xAB,sizeof(key_plane));
		memset(value_plane,0xCD,sizeof(value_plane));
		memset(staging,0,sizeof(staging));
		expect(SparkDsv4KvFramesSave(&fixture.frames,&view) ==
			SPARK_STATUS_OK && staging[0] == 0xABu &&
			staging[WIRE_KEY_BYTES] == 0xCDu &&
			fixture.frames.host_copy_count == 1u &&
			fixture.frames.staged.op_code == 0x00001000u,
			"SAVE_OUT streams key plane then value plane under 0x1000");
		view.key_bytes = WIRE_KEY_BYTES - 1u;
		expect(SparkDsv4KvFramesSave(&fixture.frames,&view) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a view whose plane bytes break the stride contract is refused");
		expect(SparkDsv4KvFramesSave(0,&view) == SPARK_STATUS_INVALID_ARGUMENT,
			"a null module context is refused");
	}

	printf("\nscenario 1: the frame ops run the real park/rewind loop\n");
	{
		WireFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		uint32_t lru_order[4] = {0u,2u,3u,1u};
		uint32_t index;
		expect(WireOpen(&fixture,16u,4u,12u,12u,
			SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY) == SPARK_STATUS_OK,
			"the fixture opens");
		expect(WireAdmit(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane A admits against the empty budget");
		for ( index = 0u; index < 4u; ++index )
			expect(WireFillBlock(&fixture,index),"lane A block fills");
		expect(SparkKvCacheArenaRetainBlock(&fixture.arena,1u) ==
			SPARK_STATUS_OK && SparkKvCacheArenaReleaseBlockReference(
				&fixture.arena,1u) == SPARK_STATUS_OK,
			"lane A touches block 1 (recency bump)");
		expect(WireAdmit(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 4u,
			"lane B admits BY PARKING through the frame ops");
		WirePageOutHistoryIs(&fixture,lru_order,4u,
			"victims are the LRU set {0,2,3,1}");
		expect(fixture.frames.staged_count == 4u &&
			fixture.frames.host_copy_count == 4u &&
			fixture.frames.device_run_count == 0u &&
			fixture.frames.refused_count == 0u,
			"four SAVE_OUT ops staged and copied, nothing else");
		expect(fixture.frames.staged.op_code == 0x00001000u &&
			fixture.frames.staged.block_count == 1u &&
			fixture.frames.staged.key_bytes == WIRE_KEY_BYTES &&
			fixture.frames.staged.value_bytes == WIRE_VALUE_BYTES,
			"the staged op is exactly the design's SAVE_OUT frame");
		expect(fixture.tier.statistics.slots_in_use == 4u &&
			fixture.tier.statistics.digest_mismatches == 0u,
			"every write-back committed digest-verified at the tier");
		for ( index = 4u; index < 8u; ++index )
			expect(WireFillBlock(&fixture,index),
				"lane B block fills into the freed budget");
		expect(SparkKvCacheArenaUnassignedResidentBlockCount(
				&fixture.arena) == 0u,
			"lane B committed its whole reservation");
		expect(WireDispatchUntilReady(&fixture,1u,&dispatch) &&
			dispatch.resident == 1u,
			"the rewind dispatches: restore completes inside the gate");
		expect(WirePlanesMatchGolden(&fixture,1u),
			"BIT-EXACT: the planes equal the payload through the seam");
		expect(fixture.frames.staged_count == 6u &&
			fixture.frames.staged.op_code == 0x00002000u,
			"the rewind's make-room staged a fifth SAVE_OUT first;"
			" the last staged op is RESTORE_IN (0x2000)");
		expect(fixture.pager.statistics.dispatch_ready == 1u,
			"the gate answered READY exactly once");
		expect(fixture.pager.statistics.dispatch_requests ==
			fixture.pager.statistics.dispatch_ready +
			fixture.pager.statistics.dispatch_queued,
			"every offer is accounted: ready + queued == requests");
		WireCheckBudget(&fixture,"scenario 1 end");
	}

	printf("\nscenario 2: the device-plane backend is spark-gated\n");
	{
		WireFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		uint32_t index;
		int staging_untouched = 1;
		expect(WireOpen(&fixture,16u,4u,12u,12u,
			SPARK_DSV4_KV_FRAMES_BACKEND_SPARK_FRAME_OPS) ==
			SPARK_STATUS_OK,
			"the fixture opens on the device-plane backend (receipt pending)");
		expect(WireAdmit(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane A admits");
		for ( index = 0u; index < 4u; ++index )
			WireFillBlock(&fixture,index);
		expect(WireAdmit(&fixture,4u,&decision) == 0,
			"without the receipt, lane B's admission FAILS LOUD");
		expect(fixture.frames.refused_count == 1u &&
			fixture.frames.staged_count == 1u &&
			fixture.frames.device_run_count == 0u &&
			fixture.spark.submissions == 0u,
			"the copy was STAGED and refused: nothing reached the frame"
			" seam, nothing ran");
		for ( index = 0u; index < sizeof(fixture.pager_staging); ++index )
		{
			if ( fixture.pager_staging[index] != 0u )
			{
				staging_untouched = 0;
				break;
			}
		}
		expect(staging_untouched,
			"the pager staging is untouched: the refused op never copied");
		expect(fixture.arena.write_back_degraded_block_count == 0u &&
			fixture.pager.statistics.page_out_count == 0u &&
			fixture.tier.statistics.slots_in_use == 0u,
			"UNSUPPORTED is not degradation: nothing dropped, nothing"
			" parked, nothing wedged");
		expect(fixture.arena.resident_block_count == 4u,
			"lane A's residents survived the refusal");
		SparkDsv4KvFramesAcceptSparkReceipt(&fixture.frames);
		expect(WireAdmit(&fixture,4u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 4u,
			"with the spark receipt, the SAME admission parks");
		expect(fixture.spark.submissions == 4u &&
			fixture.frames.device_run_count == 4u &&
			fixture.spark.last_op_code == 0x00001000u,
			"the staged ops reached the frame-submit seam under the receipt");
		expect(fixture.tier.statistics.slots_in_use == 4u,
			"the four parked records committed at the tier");
		for ( index = 4u; index < 8u; ++index )
			WireFillBlock(&fixture,index);
		expect(WireDispatchUntilReady(&fixture,0u,&dispatch),
			"the rewind restores through the received op");
		expect(WirePlanesMatchGolden(&fixture,0u),
			"BIT-EXACT through the spark-gated seam");
		WireCheckBudget(&fixture,"scenario 2 end");
	}

	printf("\nscenario 3: parkability is one predicate, adapter and pager\n");
	{
		WireFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		SparkDsv4JitKvParkability parkability;
		uint32_t active[4] = {0u,1u,2u,3u};
		uint32_t shrunk[3] = {0u,1u,3u};
		uint32_t index;
		expect(WireOpen(&fixture,8u,4u,8u,8u,
			SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY) == SPARK_STATUS_OK,
			"the fixture opens (capacity 4)");
		expect(WireAdmit(&fixture,4u,&decision),"lane P admits");
		for ( index = 0u; index < 4u; ++index )
			WireFillBlock(&fixture,index);
		for ( index = 0u; index < 4u; ++index )
			expect(SparkKvCacheArenaPinResidentBlock(&fixture.arena,index) ==
				SPARK_STATUS_OK,
				"lane P's dispatch window pins its active set");
		expect(SparkDsv4JitKvDecideParkability(&fixture.arena,active,4u,
			&parkability) == SPARK_STATUS_OK &&
			parkability.parkable_block_count == 0u &&
			parkability.pinned_block_count == 4u,
			"the deployment condition: active set pinned, parkable 0");
		expect(SparkKvCacheArenaBlockIsParkable(&fixture.arena,0u) == 0u &&
			SparkKvCacheArenaBlockIsParkable(&fixture.arena,3u) == 0u,
			"the shared predicate refuses the pinned blocks");
		expect(WireAdmit(&fixture,1u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED &&
			fixture.pager.statistics.admission_queued_device == 1u,
			"admission QUEUES: its parkable pool is the same zero");
		expect(SparkKvCacheArenaUnpinResidentBlock(&fixture.arena,2u) ==
			SPARK_STATUS_OK,
			"block 2 unpins while the lane still lists it active");
		expect(SparkDsv4JitKvDecideParkability(&fixture.arena,active,4u,
			&parkability) == SPARK_STATUS_BUSY &&
			parkability.unprotected_active_count == 1u,
			"an unpinned ACTIVE block FAILS the deployment condition:"
			" pin it or shrink the window first");
		expect(SparkDsv4JitKvDecideParkability(&fixture.arena,shrunk,3u,
			&parkability) == SPARK_STATUS_OK &&
			parkability.parkable_block_count == 1u &&
			parkability.pinned_block_count == 3u,
			"the shrunk window counts exactly one parkable block");
		expect(SparkKvCacheArenaBlockIsParkable(&fixture.arena,2u) == 1u &&
			SparkKvCacheArenaBlockIsParkable(&fixture.arena,0u) == 0u &&
			SparkKvCacheArenaBlockIsParkable(&fixture.arena,1u) == 0u &&
			SparkKvCacheArenaBlockIsParkable(&fixture.arena,3u) == 0u,
			"the predicate names block 2 and only block 2");
		expect(WireAdmit(&fixture,1u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 1u,
			"admission parks exactly the predicate's block");
		{
			uint32_t newest[1] = {2u};
			WirePageOutHistoryIs(&fixture,newest,1u,
				"the paged-out block is the unpinned one");
		}
		expect(WireBlockIsResident(&fixture,0u) &&
			WireBlockIsResident(&fixture,1u) &&
			WireBlockIsResident(&fixture,3u) && WirePlanesMatchGolden(
				&fixture,0u) && WirePlanesMatchGolden(&fixture,1u) &&
			WirePlanesMatchGolden(&fixture,3u),
			"the pinned actives survived: never victims");
		expect(SparkDsv4JitKvDecideParkability(&fixture.arena,active,4u,
			&parkability) == SPARK_STATUS_OK &&
			parkability.parkable_block_count == 0u &&
			parkability.non_resident_block_count == 1u,
			"once parked, the active-listed block is the DISPATCH"
			" gate's jurisdiction, not the park condition's");
		expect(WireFillBlock(&fixture,4u),
			"the admitted lane fills its block into the freed budget");
		expect(WireDispatchUntilReady(&fixture,2u,&dispatch),
			"lane P rewinds block 2: restore completes inside the gate");
		expect(WirePlanesMatchGolden(&fixture,2u),
			"the rewound block is bit-exact");
		expect(SparkKvCacheArenaPinResidentBlock(&fixture.arena,2u) ==
			SPARK_STATUS_OK,
			"the lane re-pins its restored block (window reopens)");
		expect(SparkDsv4JitKvDecideParkability(&fixture.arena,active,4u,
			&parkability) == SPARK_STATUS_OK &&
			parkability.parkable_block_count == 0u,
			"the deployment condition holds again: parkable 0");
		WireCheckBudget(&fixture,"scenario 3 end");
	}

	printf("\nscenario 4: C2 - dispatch gates on restore complete\n");
	{
		WireFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		uint8_t digest0[SPARK_KV_PAGER_DIGEST_BYTES];
		uint8_t digest1[SPARK_KV_PAGER_DIGEST_BYTES];
		uint32_t index;
		expect(WireOpen(&fixture,6u,2u,2u,2u,
			SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY) == SPARK_STATUS_OK,
			"the fixture opens (capacity 2, tier 2 records)");
		expect(WireAdmit(&fixture,2u,&decision),"lane R admits");
		for ( index = 0u; index < 2u; ++index )
			WireFillBlock(&fixture,index);
		expect(WireAdmit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane T admits by parking lane R entirely");
		for ( index = 2u; index < 4u; ++index )
			WireFillBlock(&fixture,index);
		expect(fixture.tier.statistics.slots_in_use == 2u,
			"the park horizon is exactly full");
		WireDigest(0u,digest0);
		WireDigest(1u,digest1);
		expect(SparkNvmeTierPin(&fixture.tier,WireFoldDigest(digest0),
			digest0,1) == SPARK_STATUS_OK &&
			SparkNvmeTierPin(&fixture.tier,WireFoldDigest(digest1),
				digest1,1) == SPARK_STATUS_OK,
			"both tier records pin (the tier saturates)");
		expect(WireDispatchOffer(&fixture,0u,&dispatch) &&
			dispatch.outcome == SPARK_KV_PAGER_DISPATCH_QUEUED &&
			dispatch.resident == 0u,
			"the dispatch of a parked block QUEUES while the tier is"
			" saturated - it does not run ahead of the restore");
		expect(!WireBlockIsResident(&fixture,0u),
			"no half-restored residency behind the QUEUED answer");
		expect(fixture.pager.statistics.dispatch_queued == 1u &&
			fixture.pager.statistics.dispatch_ready == 0u &&
			fixture.pager.statistics.page_in_count == 0u,
			"the gate counted the wait, nothing was dispatched");
		for ( index = 0u; index < 10u; ++index )
		{
			expect(WireDispatchOffer(&fixture,0u,&dispatch) &&
				dispatch.outcome == SPARK_KV_PAGER_DISPATCH_QUEUED,
				"repeated offers keep queueing - the queue, not a wedge");
		}
		expect(fixture.pager.statistics.dispatch_queued == 11u &&
			fixture.pager.statistics.page_in_count == 0u &&
			fixture.arena.write_back_degraded_block_count == 0u,
			"eleven waits, zero page-ins, zero drops: nothing poisoned");
		expect(SparkNvmeTierPin(&fixture.tier,WireFoldDigest(digest0),
			digest0,0) == SPARK_STATUS_OK,"one record unpins");
		expect(WireDispatchUntilReady(&fixture,0u,&dispatch) &&
			dispatch.resident == 1u,
			"the SAME dispatch offer completes once the tier yields");
		expect(WirePlanesMatchGolden(&fixture,0u),
			"BIT-EXACT: the dispatch ran only on the restored block");
		expect(fixture.pager.statistics.dispatch_ready == 1u &&
			fixture.pager.statistics.page_in_count == 1u,
			"exactly one restore fed the dispatch");
		expect(WireDispatchUntilReady(&fixture,0u,&dispatch),
			"dispatching the already-resident block is READY");
		expect(fixture.pager.statistics.page_in_count == 1u &&
			fixture.pager.statistics.dispatch_ready == 2u,
			"a resident dispatch re-gates without a second restore");
		WireCheckBudget(&fixture,"scenario 4 end");
	}

	printf("\nscenario 5: C2 - a degraded block answers RECOMPUTE\n");
	{
		WireFixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		uint32_t index;
		expect(WireOpen(&fixture,6u,2u,4u,4u,
			SPARK_DSV4_KV_FRAMES_BACKEND_TERM_COPY) == SPARK_STATUS_OK,
			"the fixture opens (capacity 2, horizon 4)");
		expect(WireAdmit(&fixture,2u,&decision),"lane R admits");
		for ( index = 0u; index < 2u; ++index )
			WireFillBlock(&fixture,index);
		fixture.backing_failures_left = 1u;
		expect(WireAdmit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"lane T admits; the armed backing failure hits the park");
		for ( index = 2u; index < 4u; ++index )
			WireFillBlock(&fixture,index);
		expect(fixture.arena.write_back_degraded_block_count == 1u,
			"the IO-class failure DEGRADED its block (B1: drop +"
			" recompute, never a wedge)");
		expect(fixture.tier.statistics.slots_in_use == 1u,
			"only the healthy block holds a tier record");
		expect(WireDispatchOffer(&fixture,0u,&dispatch) &&
			dispatch.outcome == SPARK_KV_PAGER_DISPATCH_RECOMPUTE &&
			dispatch.resident == 0u,
			"the degraded block's dispatch answers RECOMPUTE");
		expect(!WireBlockIsResident(&fixture,0u) &&
			fixture.pager.statistics.dispatch_recompute == 1u &&
			fixture.pager.statistics.page_in_count == 0u,
			"dispatch never runs on partial state: the caller recomputes");
		expect(WireDispatchUntilReady(&fixture,1u,&dispatch),
			"the healthy parked block still dispatches READY");
		expect(WirePlanesMatchGolden(&fixture,1u),
			"BIT-EXACT on the healthy path");
		WireCheckBudget(&fixture,"scenario 5 end");
	}

	printf("\n");
	if ( failures != 0 )
		printf("FAIL: %d check(s) failed\n",failures);
	else
		printf("JIT-KV WIRING: all proofs green\n");
	return(failures != 0 ? 1 : 0);
}
