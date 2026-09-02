#include "sparkpipe/spark_kv_pager.h"
#include "sparkpipe/spark_sha256.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n",label);
	if ( !condition )
		++failures;
}

#define C34_BLOCK_BYTES 4096u
#define C34_KEY_BYTES 2048u
#define C34_VALUE_BYTES 2048u
#define C34_TIER_BASE (1u << 20)
#define C34_MAX_READS 16u
#define C34_MAX_LOGICAL_BLOCKS 16u
#define C34_MAX_RESIDENT_SLOTS 4u
#define C34_PARK_QUEUE 4u

typedef struct C34Clock
{
	uint64_t now_us;
	uint64_t advance_per_call_us;
}
C34Clock;

static uint64_t C34ClockNow(void *context)
{
	C34Clock *clock = (C34Clock *)context;
	uint64_t now = clock->now_us;
	clock->now_us += clock->advance_per_call_us;
	return(now);
}

typedef struct C34Read
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint8_t active;
}
C34Read;

typedef struct C34Device
{
	C34Read reads[C34_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;
	const uint8_t *drive;
	uint64_t drive_base;
}
C34Device;

static void C34DeviceReset(C34Device *device,uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

static SparkStatus C34SubmitRead(
	void *context,uint64_t device_offset,void *destination,
	uint32_t bytes,uint64_t *ticket_out)
{
	C34Device *device = (C34Device *)context;
	uint32_t index;
	if ( bytes != C34_BLOCK_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < C34_MAX_READS; ++index )
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

static SparkStatus C34PollRead(void *context,uint64_t ticket)
{
	C34Device *device = (C34Device *)context;
	uint32_t index;
	for ( index = 0u; index < C34_MAX_READS; ++index )
	{
		C34Read *read = &device->reads[index];
		if ( !read->active || read->ticket != ticket )
			continue;
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
			memcpy(read->destination,
				device->drive + (read->offset - device->drive_base),
				C34_BLOCK_BYTES);
		}
		read->active = 0u;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

static SparkStatus C34CancelRead(void *context,uint64_t ticket)
{
	C34Device *device = (C34Device *)context;
	uint32_t index;
	for ( index = 0u; index < C34_MAX_READS; ++index )
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

typedef struct C34Backing
{
	uint32_t failures_left;
	uint32_t gate_armed;
	uint32_t gate_skip_writes;
	atomic_uint writes_entered;
	atomic_uint writes_completed;
	uint64_t worker_writes;
	uint64_t main_writes;
	pthread_t main_thread;
	uint8_t *drive;
	uint64_t drive_base;
}
C34Backing;

static void C34Sleep(uint32_t microseconds)
{
	struct timespec pause;
	pause.tv_sec = (time_t)0;
	pause.tv_nsec = (long)microseconds * 1000l;
	nanosleep(&pause,0);
}

static SparkStatus C34BackingWrite(
	void *context,uint64_t device_offset,const void *host_staging,
	uint64_t bytes)
{
	C34Backing *backing = (C34Backing *)context;
	uint32_t ordinal = atomic_fetch_add_explicit(&backing->writes_entered,
		1u,memory_order_relaxed);

	if ( device_offset < C34_TIER_BASE ||
		device_offset + bytes > C34_TIER_BASE + 16u * C34_BLOCK_BYTES )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( backing->gate_armed != 0u && ordinal >= backing->gate_skip_writes )
	{
		while ( backing->gate_armed != 0u )
			C34Sleep(50u);
	}
	if ( backing->failures_left != 0u )
	{
		backing->failures_left -= 1u;
		atomic_fetch_add_explicit(&backing->writes_completed,1u,
			memory_order_relaxed);
		return(SPARK_STATUS_IO_ERROR);
	}
	memcpy(backing->drive + (device_offset - backing->drive_base),
		host_staging,bytes);
	if ( pthread_equal(pthread_self(),backing->main_thread) != 0 )
		backing->main_writes += 1u;
	else
		backing->worker_writes += 1u;
	atomic_fetch_add_explicit(&backing->writes_completed,1u,
		memory_order_relaxed);
	return(SPARK_STATUS_OK);
}

typedef struct C34Fixture
{
	C34Device device;
	C34Backing backing;
	C34Clock clock;
	SparkNvmeTierDevice vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration tier_configuration;
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[C34_MAX_LOGICAL_BLOCKS];
	uint32_t resident_slots[C34_MAX_RESIDENT_SLOTS];
	uint8_t key_device[C34_MAX_RESIDENT_SLOTS * C34_KEY_BYTES];
	uint8_t value_device[C34_MAX_RESIDENT_SLOTS * C34_VALUE_BYTES];
	SparkKvPager pager;
	SparkKvPagerConfiguration pager_configuration;
	_Alignas(64) uint8_t tier_tables[96u * 1024u];
	_Alignas(C34_BLOCK_BYTES) uint8_t tier_staging[4u * C34_BLOCK_BYTES];
	_Alignas(C34_BLOCK_BYTES) uint8_t pager_staging[2u * C34_BLOCK_BYTES];
	_Alignas(C34_BLOCK_BYTES) uint8_t park_staging[8u * C34_BLOCK_BYTES];
	_Alignas(C34_BLOCK_BYTES) uint8_t drive[16u * C34_BLOCK_BYTES];
	uint8_t golden[C34_MAX_LOGICAL_BLOCKS][C34_BLOCK_BYTES];
	uint32_t logical_block_count;
	uint32_t resident_capacity;
}
C34Fixture;

static void C34ContentFill(uint32_t block_index,uint8_t *block)
{
	uint32_t index;
	for ( index = 0u; index < C34_BLOCK_BYTES; ++index )
		block[index] = (uint8_t)(block_index * 0x9Eu + index * 7u + 1u);
}

static void C34Digest(uint32_t block_index,uint8_t *digest_out)
{
	uint8_t block[C34_BLOCK_BYTES];
	SparkSha256Context context;
	C34ContentFill(block_index,block);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,block,C34_BLOCK_BYTES);
	SparkSha256Finalize(&context,digest_out);
}

static SparkStatus C34ModuleSave(void *module_context,
	const SparkKvPagerBlockView *view)
{
	(void)module_context;
	memcpy(view->host_staging,(const void *)view->key_device_address,
		view->key_bytes);
	memcpy((uint8_t *)view->host_staging + view->key_bytes,
		(const void *)view->value_device_address,view->value_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus C34ModuleRestore(void *module_context,
	const SparkKvPagerBlockView *view)
{
	(void)module_context;
	memcpy((void *)view->key_device_address,view->host_staging,
		view->key_bytes);
	memcpy((void *)view->value_device_address,
		(const uint8_t *)view->host_staging + view->key_bytes,
		view->value_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus C34BackingWriteEntry(
	void *context,uint64_t device_offset,const void *host_staging,
	uint64_t bytes)
{
	C34Fixture *fixture = (C34Fixture *)context;
	return(C34BackingWrite(&fixture->backing,device_offset,host_staging,
		bytes));
}

static void C34CheckBudget(C34Fixture *fixture,const char *where)
{
	SparkKvCacheArena *arena = &fixture->arena;
	uint32_t unassigned = atomic_load(&arena->unassigned_resident_block_count);
	uint64_t resident_bytes;
	int ok = arena->resident_block_count <= arena->resident_block_capacity &&
		arena->resident_block_count + arena->reserved_block_count +
			unassigned <= arena->resident_block_capacity &&
		fixture->tier.slots_in_use <= fixture->tier.slot_count;
	resident_bytes = (uint64_t)arena->resident_block_count * C34_BLOCK_BYTES;
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

static SparkStatus C34Open(
	C34Fixture *fixture,
	uint32_t logical_block_count,
	uint32_t resident_capacity,
	uint32_t tier_slot_count,
	uint32_t park_budget_blocks,
	uint32_t polls_per_read,
	uint64_t clock_advance_us,
	uint32_t park_queue_blocks,
	uint64_t nominal_bytes_per_second)
{
	SparkKvCacheConfiguration arena_configuration;
	SparkNvmeTierConfiguration *tier_configuration;
	SparkKvPagerConfiguration *pager_configuration;
	uint64_t table_bytes;

	memset(fixture,0,sizeof(*fixture));
	fixture->logical_block_count = logical_block_count;
	fixture->resident_capacity = resident_capacity;
	fixture->backing.main_thread = pthread_self();
	fixture->backing.drive = fixture->drive;
	fixture->backing.drive_base = C34_TIER_BASE;
	fixture->clock.advance_per_call_us = clock_advance_us;
	C34DeviceReset(&fixture->device,polls_per_read);
	fixture->device.drive = fixture->drive;
	fixture->device.drive_base = C34_TIER_BASE;
	fixture->vtable.context = &fixture->device;
	fixture->vtable.submit_read = C34SubmitRead;
	fixture->vtable.poll_read = C34PollRead;
	fixture->vtable.cancel_read = C34CancelRead;
	tier_configuration = &fixture->tier_configuration;
	tier_configuration->abi_version = SPARK_NVME_TIER_ABI_VERSION;
	tier_configuration->descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	tier_configuration->budget_bytes =
		(uint64_t)tier_slot_count * C34_BLOCK_BYTES;
	tier_configuration->base_offset = C34_TIER_BASE;
	tier_configuration->block_bytes = C34_BLOCK_BYTES;
	tier_configuration->hash_bucket_count = 32u;
	tier_configuration->staging_buffer_count = 4u;
	tier_configuration->demand_reserve_buffers = 1u;
	tier_configuration->pending_capacity = 16u;
	tier_configuration->device_bytes_per_second = nominal_bytes_per_second;
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
	arena_configuration.key_block_stride_bytes = C34_KEY_BYTES;
	arena_configuration.value_block_stride_bytes = C34_VALUE_BYTES;
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
		(uint64_t)resident_capacity * C34_BLOCK_BYTES;
	pager_configuration->staging = fixture->pager_staging;
	pager_configuration->staging_bytes = sizeof(fixture->pager_staging);
	pager_configuration->module_context = fixture;
	pager_configuration->module_save = C34ModuleSave;
	pager_configuration->module_restore = C34ModuleRestore;
	pager_configuration->backing_context = fixture;
	pager_configuration->backing_write = C34BackingWriteEntry;
	if ( clock_advance_us != 0u )
	{
		pager_configuration->clock_context = &fixture->clock;
		pager_configuration->clock_function = C34ClockNow;
	}
	if ( park_queue_blocks != 0u )
	{
		pager_configuration->park_queue_blocks = park_queue_blocks;
		pager_configuration->park_staging = fixture->park_staging;
		pager_configuration->park_staging_bytes =
			sizeof(fixture->park_staging);
	}
	return(SparkKvPagerInitialize(&fixture->pager,pager_configuration));
}

static int32_t C34FillBlock(C34Fixture *fixture,uint32_t block_index)
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
	C34ContentFill(block_index,fixture->golden[block_index]);
	memcpy((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],C34_KEY_BYTES);
	memcpy((void *)(uintptr_t)view.value_device_address,
		fixture->golden[block_index] + C34_KEY_BYTES,C34_VALUE_BYTES);
	if ( SparkKvCacheArenaMarkBlockDirty(arena,acquired) != SPARK_STATUS_OK )
		return(0);
	C34CheckBudget(fixture,"fill");
	return(1);
}

static int32_t C34AdmitEx(
	C34Fixture *fixture,uint32_t demand,uint32_t slack_us,
	SparkKvPagerAdmissionDecision *decision_out)
{
	SparkKvPagerAdmission admission;
	memset(&admission,0,sizeof(admission));
	admission.abi_version = SPARK_KV_PAGER_ADMISSION_ABI_VERSION;
	admission.descriptor_bytes = SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES;
	admission.block_demand = demand;
	admission.restore_slack_microseconds = slack_us;
	C34CheckBudget(fixture,"admit");
	return(SparkKvPagerAdmit(&fixture->pager,&admission,decision_out) ==
		SPARK_STATUS_OK);
}

static int32_t C34Admit(C34Fixture *fixture,uint32_t demand,
	SparkKvPagerAdmissionDecision *decision_out)
{
	return(C34AdmitEx(fixture,demand,0u,decision_out));
}

static int32_t C34BlockIsResident(C34Fixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}

static int32_t C34PlanesMatchGolden(C34Fixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return memcmp((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],C34_KEY_BYTES) == 0 &&
		memcmp((void *)(uintptr_t)view.value_device_address,
			fixture->golden[block_index] + C34_KEY_BYTES,
			C34_VALUE_BYTES) == 0;
}

static int32_t C34DispatchOffer(
	C34Fixture *fixture,
	uint32_t block_index,
	SparkKvPagerDispatchDecision *decision_out)
{
	SparkKvPagerDispatch dispatch;

	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_KV_PAGER_DISPATCH_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_KV_PAGER_DISPATCH_DESCRIPTOR_BYTES;
	dispatch.logical_block_index = block_index;
	{
		uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
		C34Digest(block_index,digest);
		memcpy(dispatch.content_digest,digest,
			SPARK_KV_PAGER_DIGEST_BYTES);
	}
	memset(decision_out,0,sizeof(*decision_out));
	if ( SparkKvPagerDispatchBlock(&fixture->pager,&dispatch,
		decision_out) != SPARK_STATUS_OK )
	{
		return(0);
	}
	C34CheckBudget(fixture,"dispatch");
	return(1);
}

static uint64_t C34RestoreDebtBlocks(C34Fixture *fixture)
{
	SparkKvCacheArena *arena = &fixture->arena;
	uint64_t debt_blocks = 0u;
	uint32_t index;
	for ( index = 0u; index < arena->logical_block_count; ++index )
	{
		SparkKvCacheBlockView view;
		if ( SparkKvCacheArenaResolveBlock(arena,index,&view) ==
			SPARK_STATUS_OK &&
			(view.flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u &&
			(view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u &&
			(view.flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u )
		{
			debt_blocks += 1u;
		}
	}
	return(debt_blocks);
}

static int32_t C34DispatchUntilReady(
	C34Fixture *fixture,
	uint32_t block_index,
	SparkKvPagerDispatchDecision *decision_out)
{
	uint32_t attempt;
	for ( attempt = 0u; attempt < 16u; ++attempt )
	{
		if ( !C34DispatchOffer(fixture,block_index,decision_out) )
			return(0);
		if ( decision_out->outcome == SPARK_KV_PAGER_DISPATCH_READY )
			return(1);
		if ( decision_out->outcome !=
			SPARK_KV_PAGER_DISPATCH_QUEUED )
			return(0);
		C34Sleep(100u);
	}
	return(0);
}

static int32_t C34WaitFor(atomic_uint *counter,uint32_t target)
{
	uint32_t attempt;
	for ( attempt = 0u; attempt < 20000u; ++attempt )
	{
		if ( atomic_load_explicit(counter,memory_order_relaxed) >= target )
			return(1);
		C34Sleep(50u);
	}
	return(0);
}

static int32_t C34DrainUntil(C34Fixture *fixture,uint32_t target)
{
	uint32_t attempt;
	for ( attempt = 0u; attempt < 20000u; ++attempt )
	{
		(void)SparkKvPagerPollParkCompletions(&fixture->pager);
		if ( fixture->pager.statistics.park_completions_published >= target )
			return(1);
		C34Sleep(50u);
	}
	return(0);
}

int main(void)
{
	setvbuf(stdout,0,_IONBF,0);
	printf("JIT-KV C3+C4: measured bandwidth in admission, the async park"
		" worker\n");
	printf("\nconfiguration fences\n");
	{
		C34Fixture fixture;
		SparkKvPager probe;
		SparkKvPagerConfiguration probe_configuration;
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,0u,5000000000ull) ==
			SPARK_STATUS_OK,
			"the inline fixture opens (worker off, no clock)");
		expect(fixture.pager.park_worker_active == 0u &&
			SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"shutdown is OK on a pager that never started a worker");
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,
			SPARK_KV_PAGER_PARK_QUEUE_CAPACITY + 1u,5000000000ull) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"a park queue deeper than the pager's ring is refused");
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,
			"the async fixture opens (worker on)");
		expect(fixture.pager.park_worker_active == 1u,
			"the worker thread is running");
		probe_configuration = fixture.pager_configuration;
		probe_configuration.park_staging = 0;
		probe_configuration.park_staging_bytes = 0;
		expect(SparkKvPagerInitialize(&probe,&probe_configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"an async pager without park staging is refused");
		probe_configuration = fixture.pager_configuration;
		probe_configuration.park_staging_bytes =
			C34_PARK_QUEUE * C34_BLOCK_BYTES - 1u;
		expect(SparkKvPagerInitialize(&probe,&probe_configuration) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"park staging that cannot hold every entry is refused");
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK &&
			fixture.pager.park_worker_active == 0u,
			"shutdown stops the worker");
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"shutdown is idempotent");
	}

	printf("\nscenario 1: C3 - the admission answer follows the MEASURED"
		" bandwidth\n");
	{
		C34Fixture slow;
		C34Fixture fast;
		C34Fixture nominal;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		uint64_t measured_slow,measured_fast,prediction_slow,prediction_fast;
		uint64_t debt_bytes,slack_us;

		expect(C34Open(&slow,8u,2u,8u,8u,4u,2000000u,0u,4000000000ull) ==
			SPARK_STATUS_OK,"the slow-tier fixture opens (4 polls/read)");
		expect(C34Open(&fast,8u,2u,8u,8u,1u,2000000u,0u,4000000000ull) ==
			SPARK_STATUS_OK,"the fast-tier fixture opens (1 poll/read)");
		expect(C34Admit(&slow,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&slow,0u) && C34FillBlock(&slow,1u),
			"slow: lane A fills the budget");
		expect(C34Admit(&slow,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 2u &&
			C34FillBlock(&slow,2u) && C34FillBlock(&slow,3u),
			"slow: lane B admits by parking both blocks, then commits");
		expect(C34DispatchUntilReady(&slow,0u,&dispatch) &&
			C34PlanesMatchGolden(&slow,0u),
			"slow: block 0 rewinds bit-exact (the page-in is measured)");
		expect(C34Admit(&fast,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&fast,0u) && C34FillBlock(&fast,1u) &&
			C34Admit(&fast,2u,&decision) &&
			C34FillBlock(&fast,2u) && C34FillBlock(&fast,3u) &&
			C34DispatchUntilReady(&fast,0u,&dispatch),
			"fast: the same history runs to the same state");
		measured_slow = slow.pager.statistics.measured_bytes_per_second;
		measured_fast = fast.pager.statistics.measured_bytes_per_second;
		expect(measured_slow != 0u && measured_fast != 0u &&
			measured_slow < measured_fast,
			"MEASURED: the slow tier's EMA is strictly below the fast"
			" tier's");
		expect(slow.pager.statistics.measured_bandwidth_samples >= 4u &&
			fast.pager.statistics.measured_bandwidth_samples >= 4u,
			"page-out and page-in samples both folded");
		debt_bytes = C34RestoreDebtBlocks(&slow) * C34_BLOCK_BYTES;
		expect(debt_bytes == 2u * C34_BLOCK_BYTES &&
			C34RestoreDebtBlocks(&fast) == 2u,
			"both fixtures owe the same two-block restore debt");
		prediction_slow = (debt_bytes + (uint64_t)C34_BLOCK_BYTES) *
			1000000ull / measured_slow;
		prediction_fast = (debt_bytes + (uint64_t)C34_BLOCK_BYTES) *
			1000000ull / measured_fast;
		slack_us = prediction_slow / 2u + prediction_fast / 2u;
		expect(prediction_fast < slack_us && slack_us < prediction_slow,
			"the slack separates the two predictions");
		expect(C34AdmitEx(&slow,1u,(uint32_t)slack_us,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED &&
			slow.pager.statistics.admission_queued_bandwidth == 1u,
			"SLOW tier: the identical admission QUEUES on bandwidth");
		expect(slow.arena.resident_block_count == 2u &&
			atomic_load(&slow.arena.unassigned_resident_block_count) == 0u,
			"the refusal touched nothing (queue, never evict)");
		expect(C34AdmitEx(&slow,1u,0u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"without a slack the same admission is the pre-C3 answer:"
			" ADMITTED (the check is slack-gated)");
		expect(C34AdmitEx(&fast,1u,(uint32_t)slack_us,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			fast.pager.statistics.admission_queued_bandwidth == 0u,
			"FAST tier: the identical admission is ADMITTED");
		C34CheckBudget(&slow,"scenario 1 slow end");
		C34CheckBudget(&fast,"scenario 1 fast end");

		expect(C34Open(&nominal,8u,2u,8u,8u,2u,0u,0u,1024ull) ==
			SPARK_STATUS_OK,
			"the no-clock fixture opens (nominal 1024 B/s)");
		expect(C34Admit(&nominal,2u,&decision) &&
			C34FillBlock(&nominal,0u) && C34FillBlock(&nominal,1u) &&
			C34Admit(&nominal,2u,&decision) &&
			C34FillBlock(&nominal,2u) && C34FillBlock(&nominal,3u),
			"nominal: the same history runs, committed");
		expect(nominal.pager.statistics.measured_bytes_per_second == 0u,
			"nothing was measured without a clock");
		expect(C34AdmitEx(&nominal,1u,1000000u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_QUEUED &&
			nominal.pager.statistics.admission_queued_bandwidth == 1u,
			"the NOMINAL fallback queues the slow figure");
		expect(C34AdmitEx(&nominal,1u,0u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED,
			"slack 0 keeps the capacity-only answer");
	}

	printf("\nscenario 2: C4 - a park mid-write does not block an"
		" admission that fits\n");
	{
		C34Fixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,"the async fixture opens");
		expect(C34Admit(&fixture,2u,&decision) &&
			C34FillBlock(&fixture,0u) && C34FillBlock(&fixture,1u),
			"lane A fills the budget");
		fixture.backing.gate_armed = 1u;
		expect(C34Admit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			decision.park_evictions == 2u &&
			C34FillBlock(&fixture,2u) && C34FillBlock(&fixture,3u),
			"lane B admits (parks staged, not run inline) and commits");
		expect(C34WaitFor(&fixture.backing.writes_entered,1u) &&
			atomic_load_explicit(&fixture.backing.writes_completed,
				memory_order_relaxed) == 0u,
			"the worker is STUCK INSIDE the first write leg, the"
			" second still queued, none completed");
		expect(fixture.tier.slots_in_use == 2u,
			"the reserved records hold their tier slots while in"
			" flight (the accounting admission reads)");
		expect(C34Admit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&fixture,4u) && C34FillBlock(&fixture,5u),
			"THE PROOF: an admission that fits does not wait for the"
			" in-flight parks - it even parks THROUGH them");
		fixture.backing.gate_armed = 0u;
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"TERM: shutdown stops the worker and drains");
		expect(C34WaitFor(&fixture.backing.writes_completed,4u),
			"all four staged writes completed");
		expect(fixture.pager.statistics.park_completions_published == 4u &&
			fixture.pager.statistics.park_write_failures == 0u,
			"every completion published, none failed");
		expect(fixture.pager.statistics.page_out_count == 4u &&
			fixture.tier.slots_in_use == 4u,
			"the page-out receipts and the tier slots agree");
		expect(C34BlockIsResident(&fixture,5u) &&
			C34PlanesMatchGolden(&fixture,5u) &&
			C34DispatchUntilReady(&fixture,0u,&dispatch) &&
			C34PlanesMatchGolden(&fixture,0u),
			"the parked bytes are bit-exact after the async round trip");
		C34CheckBudget(&fixture,"scenario 2 end");
	}

	printf("\nscenario 3: C4+C2 - a mid-write park dispatches QUEUED,"
		" never partial\n");
	{
		C34Fixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,"the async fixture opens");
		expect(C34Admit(&fixture,2u,&decision) &&
			C34FillBlock(&fixture,0u) && C34FillBlock(&fixture,1u),
			"lane A fills the budget");
		fixture.backing.gate_armed = 1u;
		fixture.backing.gate_skip_writes = 1u;
		expect(C34Admit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&fixture,2u) && C34FillBlock(&fixture,3u),
			"lane B admits and commits; block 0's write sails through,"
			" block 1's is the slow one");
		expect(C34WaitFor(&fixture.backing.writes_entered,2u),
			"the worker is INSIDE block 1's write leg");
		expect(C34DispatchOffer(&fixture,1u,&dispatch) &&
			dispatch.outcome == SPARK_KV_PAGER_DISPATCH_QUEUED &&
			dispatch.resident == 0u,
			"a MID-WRITE park answers QUEUED - the pager holds the"
			" bytes, the dispatch waits");
		expect(fixture.pager.statistics.dispatch_queued == 1u &&
			fixture.pager.statistics.dispatch_recompute == 0u &&
			fixture.pager.statistics.page_in_count == 0u,
			"never a recompute of bytes the pager is about to publish");
		fixture.backing.gate_armed = 0u;
		expect(C34DispatchUntilReady(&fixture,1u,&dispatch) &&
			C34PlanesMatchGolden(&fixture,1u),
			"the SAME offer completes bit-exact once the write lands");
		C34CheckBudget(&fixture,"scenario 3 end");
	}

	printf("\nscenario 4: C4 - an IO-class write DEGRADES (B1, async"
		" leg)\n");
	{
		C34Fixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,"the async fixture opens");
		expect(C34Admit(&fixture,2u,&decision) &&
			C34FillBlock(&fixture,0u) && C34FillBlock(&fixture,1u),
			"lane A fills the budget");
		fixture.backing.failures_left = 1u;
		expect(C34Admit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&fixture,2u) && C34FillBlock(&fixture,3u),
			"lane B admits and commits; the armed failure hits the"
			" first park");
		expect(C34WaitFor(&fixture.backing.writes_completed,2u),
			"both write legs ran (one of them failed)");
		expect(C34DrainUntil(&fixture,2u),
			"both completions published");
		expect(fixture.pager.statistics.park_write_failures == 1u &&
			fixture.pager.statistics.park_completions_published == 2u,
			"one failure published beside one healthy publish");
		expect(fixture.arena.write_back_degraded_block_count == 1u,
			"the B1 degrade applied at publish: drop + recompute,"
			" never a wedge");
		expect(fixture.tier.slots_in_use == 1u,
			"the failed reservation aborted: its tier slot returned");
		expect(C34DispatchOffer(&fixture,0u,&dispatch) &&
			dispatch.outcome == SPARK_KV_PAGER_DISPATCH_RECOMPUTE,
			"the degraded block's dispatch answers RECOMPUTE");
		expect(C34DispatchUntilReady(&fixture,1u,&dispatch) &&
			C34PlanesMatchGolden(&fixture,1u),
			"the healthy park dispatches READY, bit-exact");
		C34CheckBudget(&fixture,"scenario 4 end");
	}

	printf("\nscenario 5: C4 - TERM mid-park leaves a consistent arena\n");
	{
		C34Fixture fixture;
		SparkKvPagerAdmissionDecision decision;
		SparkKvPagerDispatchDecision dispatch;
		expect(C34Open(&fixture,8u,3u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,
			"the async fixture opens (capacity 3)");
		expect(C34Admit(&fixture,3u,&decision) &&
			C34FillBlock(&fixture,0u) && C34FillBlock(&fixture,1u) &&
			C34FillBlock(&fixture,2u),
			"lane A fills the budget");
		fixture.backing.gate_armed = 1u;
		fixture.backing.gate_skip_writes = 1u;
		expect(C34Admit(&fixture,3u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			C34FillBlock(&fixture,3u) && C34FillBlock(&fixture,4u) &&
			C34FillBlock(&fixture,5u),
			"the trim parks all three blocks through the worker; lane"
			" B commits");
		expect(C34WaitFor(&fixture.backing.writes_entered,2u),
			"write 1 done, write 2 STUCK MID-WRITE, one still queued");
		fixture.backing.gate_armed = 0u;
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"TERM lands mid-park: stop flag stored, worker joined");
		expect(fixture.pager.statistics.page_out_count == 3u &&
			fixture.pager.statistics.park_completions_published == 3u &&
			fixture.pager.statistics.park_write_failures == 0u,
			"every staged park resolved committed - none dropped");
		expect(fixture.tier.slots_in_use == 3u,
			"every reservation resolved: no leaks at the tier");
		expect(C34DispatchUntilReady(&fixture,1u,&dispatch) &&
			C34PlanesMatchGolden(&fixture,1u),
			"the mid-TERM park's bytes restore bit-exact");
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"shutdown stays idempotent after TERM");

		expect(C34Open(&fixture,8u,2u,8u,8u,2u,0u,C34_PARK_QUEUE,
			5000000000ull) == SPARK_STATUS_OK,
			"a fresh async fixture opens");
		expect(C34Admit(&fixture,2u,&decision) &&
			C34FillBlock(&fixture,0u) && C34FillBlock(&fixture,1u) &&
			C34Admit(&fixture,2u,&decision) &&
			C34FillBlock(&fixture,2u) && C34FillBlock(&fixture,3u),
			"lane B admits, commits; both parks go to the live worker");
		expect(C34WaitFor(&fixture.backing.writes_entered,2u),
			"the worker took the writes");
		expect(SparkKvPagerShutdown(&fixture.pager) == SPARK_STATUS_OK,
			"TERM: the worker stops");
		expect(fixture.pager.statistics.park_completions_published == 2u &&
			fixture.backing.worker_writes >= 1u,
			"the worker ran the write legs while alive");
		expect(C34Admit(&fixture,2u,&decision) &&
			decision.outcome == SPARK_KV_PAGER_ADMITTED &&
			fixture.backing.main_writes >= 1u,
			"a park AFTER shutdown completes INLINE on the arena's"
			" clock (worker off) - slower, never dropped");
		expect(fixture.pager.statistics.page_out_count == 4u &&
			fixture.pager.statistics.park_completions_published == 2u,
			"the inline parks recorded their receipts directly");
		C34CheckBudget(&fixture,"scenario 5 end");
	}

	printf("\n");
	if ( failures != 0 )
		printf("FAIL: %d check(s) failed\n",failures);
	else
		printf("JIT-KV C3+C4: all proofs green\n");
	return(failures != 0 ? 1 : 0);
}
