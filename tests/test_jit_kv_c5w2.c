// The JIT-KV pager's last two named remainders (docs/JIT_KV_RESPONSE.md
// C5 + W2), on a host, over the read-vtable fake backing - no GPU, no
// reservations, no fleet.
//
// C5 REUSE-VALUE PARK POLICY: the victim choice becomes a selectable
// admission-time knob on the pager configuration (park_policy). LRU stays
// the default and the historical behavior byte for byte. REUSE_VALUE ranks
// victims by keepness - restored-again history first (a block restored
// twice came back on purpose: hot), then dirtiness (a clean block is cheap
// to drop; a dirty one costs a write on park and endurance is a line
// item), then recency. The proofs: a twice-restored block that LRU evicts
// SURVIVES under the reuse policy (the policy flips the victim); a clean
// cold block drops before a dirty warm one under both policies, and a
// dirty old block survives a clean younger one only under the reuse policy
// (the dirtiness factor decides); and the policy choice never changes
// budget accounting - identical op sequences hold identical admissions,
// reservations, resident counts, tier slots, and pager budget statistics,
// differing ONLY in which block parked.
//
// W2 DEADLINE LOOKAHEAD AS THE C2 GATE'S ENGINE: the dispatch offer
// carries an optional deadline hint (the struct's former reserved0); the
// gate rides it into the tier's read path (RequestDemandDeadline), and a
// tier saturated in cancel-pending staging ORDERs the gated block in its
// pending debt at that deadline - earliest deadline first - instead of
// yanking it out of the queue to spin. The proofs, one fixture, two runs:
// with the hint the gate answers QUEUED (healthy backpressure) and the
// tier's very next freed staging buffer carries the GATED block's read,
// which completes before its FIFO-older peer. The debt-lane follow-up made
// the queue-not-wedge answer UNIVERSAL: the hintless gate used to burn its
// poll budget to a hard IO_ERROR while the backlog churned; it now answers
// the same QUEUED (the spin itself, the tier's yank and stall accounting,
// stay byte for byte - only the terminal answer changed), and the gated
// block still completes after the peer. Same state, same device, only the
// hint differs, and the hint's residue is the ORDER, never an error.
#include "sparkpipe/spark_kv_pager.h"
#include "sparkpipe/spark_sha256.h"

#include <stdio.h>
#include <string.h>

static int32_t failures = 0;

static void expect(int condition, const char *label)
{
	printf(condition ? "  ok   %s\n" : "  FAIL %s\n",label);
	if ( !condition )
		++failures;
}

#define CW2_BLOCK_BYTES 4096u
#define CW2_KEY_BYTES 2048u
#define CW2_VALUE_BYTES 2048u
#define CW2_TIER_BASE (1u << 20)
#define CW2_MAX_READS 8u
#define CW2_MAX_LOGICAL_BLOCKS 8u
#define CW2_MAX_RESIDENT_SLOTS 4u
#define CW2_MAX_TIER_SLOTS 8u

/* the fake drive: every read takes a fixed number of polls, and every
   completed poll is recorded - the completion ORDER is the W2 proof. */
typedef struct CW2Read
{
	uint64_t ticket;
	uint64_t offset;
	uint8_t *destination;
	uint32_t polls_left;
	uint8_t active;
}
CW2Read;

typedef struct CW2Device
{
	CW2Read reads[CW2_MAX_READS];
	uint64_t next_ticket;
	uint32_t polls_per_read;
	uint8_t *drive;
	uint64_t drive_base;
	uint64_t completions;
	uint64_t first_completion[CW2_MAX_TIER_SLOTS]; /* by slot, 0 = none */
}
CW2Device;

static void CW2DeviceReset(CW2Device *device,uint32_t polls_per_read)
{
	memset(device,0,sizeof(*device));
	device->polls_per_read = polls_per_read;
	device->next_ticket = 1u;
}

static SparkStatus CW2SubmitRead(
	void *context,uint64_t device_offset,void *destination,
	uint32_t bytes,uint64_t *ticket_out)
{
	CW2Device *device = (CW2Device *)context;
	uint32_t index;
	if ( bytes != CW2_BLOCK_BYTES )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for ( index = 0u; index < CW2_MAX_READS; ++index )
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

static SparkStatus CW2PollRead(void *context,uint64_t ticket)
{
	CW2Device *device = (CW2Device *)context;
	uint32_t index;
	for ( index = 0u; index < CW2_MAX_READS; ++index )
	{
		CW2Read *read = &device->reads[index];
		if ( !read->active || read->ticket != ticket )
			continue;
		if ( read->polls_left != 0u )
		{
			read->polls_left--;
			if ( read->polls_left != 0u )
				return(SPARK_STATUS_BUSY);
			memcpy(read->destination,
				device->drive + (read->offset - device->drive_base),
				CW2_BLOCK_BYTES);
		}
		read->active = 0u;
		device->completions += 1u;
		{
			uint64_t slot = (read->offset - device->drive_base) /
				CW2_BLOCK_BYTES;
			if ( slot < CW2_MAX_TIER_SLOTS &&
				device->first_completion[slot] == 0u )
			{
				device->first_completion[slot] = device->completions;
			}
		}
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_NOT_FOUND);
}

/* the saturated drive: cancellation is ALWAYS still in flight, so a staged
   read is unstealable until it lands - the tier must order work in its
   pending debt instead of taking a buffer. */
static SparkStatus CW2CancelReadPending(void *context,uint64_t ticket)
{
	(void)context;
	(void)ticket;
	return(SPARK_STATUS_PENDING);
}

static SparkStatus CW2BackingWrite(
	void *context,uint64_t device_offset,const void *host_staging,
	uint64_t bytes)
{
	CW2Device *device = (CW2Device *)context;
	if ( device_offset < CW2_TIER_BASE ||
		device_offset + bytes > CW2_TIER_BASE +
			CW2_MAX_TIER_SLOTS * CW2_BLOCK_BYTES )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memcpy(device->drive + (device_offset - device->drive_base),
		host_staging,bytes);
	return(SPARK_STATUS_OK);
}

typedef struct CW2Fixture
{
	CW2Device device;
	SparkNvmeTierDevice vtable;
	SparkNvmeTier tier;
	SparkNvmeTierConfiguration tier_configuration;
	SparkKvCacheArena arena;
	SparkKvCacheBlock blocks[CW2_MAX_LOGICAL_BLOCKS];
	uint32_t resident_slots[CW2_MAX_RESIDENT_SLOTS];
	uint8_t key_device[CW2_MAX_RESIDENT_SLOTS * CW2_KEY_BYTES];
	uint8_t value_device[CW2_MAX_RESIDENT_SLOTS * CW2_VALUE_BYTES];
	SparkKvPager pager;
	SparkKvPagerConfiguration pager_configuration;
	_Alignas(64) uint8_t tier_tables[96u * 1024u];
	_Alignas(CW2_BLOCK_BYTES) uint8_t tier_staging[4u * CW2_BLOCK_BYTES];
	_Alignas(CW2_BLOCK_BYTES) uint8_t pager_staging[2u * CW2_BLOCK_BYTES];
	_Alignas(CW2_BLOCK_BYTES) uint8_t drive
		[CW2_MAX_TIER_SLOTS * CW2_BLOCK_BYTES];
	uint8_t golden[CW2_MAX_LOGICAL_BLOCKS][CW2_BLOCK_BYTES];
	uint32_t logical_block_count;
	uint32_t resident_capacity;
}
CW2Fixture;

static void CW2ContentFill(uint32_t block_index,uint8_t *block)
{
	uint32_t index;
	for ( index = 0u; index < CW2_BLOCK_BYTES; ++index )
		block[index] = (uint8_t)(block_index * 0xA7u + index * 11u + 3u);
}

static void CW2Digest(uint32_t block_index,uint8_t *digest_out)
{
	uint8_t block[CW2_BLOCK_BYTES];
	SparkSha256Context context;
	CW2ContentFill(block_index,block);
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,block,CW2_BLOCK_BYTES);
	SparkSha256Finalize(&context,digest_out);
}

/* the pager's 64-bit tier key fold, mirrored for tier-side calls */
static uint64_t CW2HashFromDigest(const uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES])
{
	uint64_t hash = 0u;
	uint32_t index;
	for ( index = 0u; index < (uint32_t)sizeof(hash); ++index )
		hash |= (uint64_t)digest[index] << (8u * index);
	return(hash | 1u);
}

static SparkStatus CW2ModuleSave(void *module_context,
	const SparkKvPagerBlockView *view)
{
	(void)module_context;
	memcpy(view->host_staging,(const void *)view->key_device_address,
		view->key_bytes);
	memcpy((uint8_t *)view->host_staging + view->key_bytes,
		(const void *)view->value_device_address,view->value_bytes);
	return(SPARK_STATUS_OK);
}

static SparkStatus CW2ModuleRestore(void *module_context,
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

/* The budget law, asserted after every state change in every scenario. */
static void CW2CheckBudget(CW2Fixture *fixture,const char *where)
{
	SparkKvCacheArena *arena = &fixture->arena;
	uint32_t unassigned = atomic_load(&arena->unassigned_resident_block_count);
	uint64_t resident_bytes;
	int ok = arena->resident_block_count <= arena->resident_block_capacity &&
		arena->resident_block_count + arena->reserved_block_count +
			unassigned <= arena->resident_block_capacity &&
		fixture->tier.slots_in_use <= fixture->tier.slot_count;
	resident_bytes = (uint64_t)arena->resident_block_count * CW2_BLOCK_BYTES;
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

static SparkStatus CW2Open(
	CW2Fixture *fixture,
	uint32_t logical_block_count,
	uint32_t resident_capacity,
	uint32_t tier_slot_count,
	uint32_t park_budget_blocks,
	uint32_t polls_per_read,
	uint32_t staging_buffer_count,
	uint32_t demand_reserve_buffers,
	uint32_t park_policy,
	uint32_t saturate_cancel)
{
	SparkKvCacheConfiguration arena_configuration;
	SparkNvmeTierConfiguration *tier_configuration;
	SparkKvPagerConfiguration *pager_configuration;
	uint64_t table_bytes;

	memset(fixture,0,sizeof(*fixture));
	fixture->logical_block_count = logical_block_count;
	fixture->resident_capacity = resident_capacity;
	CW2DeviceReset(&fixture->device,polls_per_read);
	fixture->device.drive = fixture->drive;
	fixture->device.drive_base = CW2_TIER_BASE;
	fixture->vtable.context = &fixture->device;
	fixture->vtable.submit_read = CW2SubmitRead;
	fixture->vtable.poll_read = CW2PollRead;
	fixture->vtable.cancel_read =
		(saturate_cancel != 0u ? CW2CancelReadPending : 0);
	tier_configuration = &fixture->tier_configuration;
	tier_configuration->abi_version = SPARK_NVME_TIER_ABI_VERSION;
	tier_configuration->descriptor_bytes = SPARK_NVME_TIER_CONFIGURATION_BYTES;
	tier_configuration->budget_bytes =
		(uint64_t)tier_slot_count * CW2_BLOCK_BYTES;
	tier_configuration->base_offset = CW2_TIER_BASE;
	tier_configuration->block_bytes = CW2_BLOCK_BYTES;
	tier_configuration->hash_bucket_count = 32u;
	tier_configuration->staging_buffer_count = staging_buffer_count;
	tier_configuration->demand_reserve_buffers = demand_reserve_buffers;
	tier_configuration->pending_capacity = 16u;
	tier_configuration->device_bytes_per_second = 5000000000ull;
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
	arena_configuration.key_block_stride_bytes = CW2_KEY_BYTES;
	arena_configuration.value_block_stride_bytes = CW2_VALUE_BYTES;
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
		(uint64_t)resident_capacity * CW2_BLOCK_BYTES;
	pager_configuration->staging = fixture->pager_staging;
	pager_configuration->staging_bytes = sizeof(fixture->pager_staging);
	pager_configuration->module_context = fixture;
	pager_configuration->module_save = CW2ModuleSave;
	pager_configuration->module_restore = CW2ModuleRestore;
	pager_configuration->backing_context = &fixture->device;
	pager_configuration->backing_write = CW2BackingWrite;
	pager_configuration->park_policy = park_policy;
	return(SparkKvPagerInitialize(&fixture->pager,pager_configuration));
}

/* Admit + fill one block (the lane's ordinary residency path). */
static int32_t CW2FillBlock(CW2Fixture *fixture,uint32_t block_index)
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
	CW2ContentFill(block_index,fixture->golden[block_index]);
	memcpy((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],CW2_KEY_BYTES);
	memcpy((void *)(uintptr_t)view.value_device_address,
		fixture->golden[block_index] + CW2_KEY_BYTES,CW2_VALUE_BYTES);
	if ( SparkKvCacheArenaMarkBlockDirty(arena,acquired) != SPARK_STATUS_OK )
		return(0);
	CW2CheckBudget(fixture,"fill");
	return(1);
}

static int32_t CW2Admit(CW2Fixture *fixture,uint32_t demand,
	SparkKvPagerAdmissionDecision *decision_out)
{
	SparkKvPagerAdmission admission;
	memset(&admission,0,sizeof(admission));
	admission.abi_version = SPARK_KV_PAGER_ADMISSION_ABI_VERSION;
	admission.descriptor_bytes = SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES;
	admission.block_demand = demand;
	CW2CheckBudget(fixture,"admit");
	return(SparkKvPagerAdmit(&fixture->pager,&admission,decision_out) ==
		SPARK_STATUS_OK);
}

static int32_t CW2BlockIsResident(CW2Fixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return((view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u);
}

static int32_t CW2PlanesMatchGolden(CW2Fixture *fixture,uint32_t block_index)
{
	SparkKvCacheBlockView view;
	if ( SparkKvCacheArenaResolveBlock(&fixture->arena,block_index,&view) !=
		SPARK_STATUS_OK )
	{
		return(0);
	}
	return memcmp((void *)(uintptr_t)view.key_device_address,
		fixture->golden[block_index],CW2_KEY_BYTES) == 0 &&
		memcmp((void *)(uintptr_t)view.value_device_address,
			fixture->golden[block_index] + CW2_KEY_BYTES,
			CW2_VALUE_BYTES) == 0;
}

static int32_t CW2Restore(CW2Fixture *fixture,uint32_t block_index)
{
	uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
	CW2Digest(block_index,digest);
	return(SparkKvPagerRestoreBlock(&fixture->pager,block_index,digest) ==
		SPARK_STATUS_OK);
}

/* A dispatch offer. Returns the RAW status: under the universal
   queue-not-wedge discipline saturation answers OK/QUEUED; only a hard
   failure surfaces as an error status. */
static SparkStatus CW2Dispatch(CW2Fixture *fixture,uint32_t block_index,
	uint32_t deadline_step,SparkKvPagerDispatchDecision *decision_out)
{
	SparkKvPagerDispatch dispatch;
	uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];

	memset(&dispatch,0,sizeof(dispatch));
	dispatch.abi_version = SPARK_KV_PAGER_DISPATCH_ABI_VERSION;
	dispatch.descriptor_bytes = SPARK_KV_PAGER_DISPATCH_DESCRIPTOR_BYTES;
	dispatch.logical_block_index = block_index;
	dispatch.deadline_step = deadline_step;
	CW2Digest(block_index,digest);
	memcpy(dispatch.content_digest,digest,SPARK_KV_PAGER_DIGEST_BYTES);
	memset(decision_out,0,sizeof(*decision_out));
	return(SparkKvPagerDispatchBlock(&fixture->pager,&dispatch,
		decision_out));
}

static int32_t CW2DispatchUntilReady(CW2Fixture *fixture,
	uint32_t block_index,uint32_t deadline_step,
	SparkKvPagerDispatchDecision *decision_out)
{
	uint32_t attempt;
	for ( attempt = 0u; attempt < 16u; ++attempt )
	{
		if ( CW2Dispatch(fixture,block_index,deadline_step,
			decision_out) != SPARK_STATUS_OK )
		{
			return(0);
		}
		if ( decision_out->outcome == SPARK_KV_PAGER_DISPATCH_READY )
			return(1);
		if ( decision_out->outcome != SPARK_KV_PAGER_DISPATCH_QUEUED )
			return(0);
	}
	return(0);
}

/* ---- scenario 1 sequence: the reuse-value victim decision ----------- */

/* Drives one fixture through the identical op sequence and reports the
   residency of blocks 0 and 1 after THE victim decision: one park-out
   with two candidates - block 0 (restored twice, LRU-oldest, clean) and
   block 1 (restored once, newer, clean). */
static void CW2ReuseSequence(CW2Fixture *fixture,
	uint32_t *zero_resident_out,uint32_t *one_resident_out)
{
	SparkKvPagerAdmissionDecision decision;

	expect(CW2Admit(fixture,2u,&decision) &&
		decision.outcome == SPARK_KV_PAGER_ADMITTED &&
		CW2FillBlock(fixture,0u) && CW2FillBlock(fixture,1u),
		"lane A fills the budget");
	/* step 2: one park-out, both dirty, no history: the LRU-oldest
	   (block 0) is the victim under BOTH policies */
	expect(CW2Admit(fixture,2u,&decision) &&
		decision.outcome == SPARK_KV_PAGER_ADMITTED &&
		decision.park_evictions == 1u,
		"the first park-out takes the LRU-oldest");
	expect(SparkKvPagerReleaseAdmission(&fixture->pager,2u) == SPARK_STATUS_OK,
		"the unused reservation releases");
	expect(CW2Restore(fixture,0u),"block 0 restores (history 1)");
	/* step 4: the never-restored newer block 1 is the victim under BOTH
	   policies (block 0 now carries one restore) */
	expect(CW2Admit(fixture,2u,&decision) &&
		decision.outcome == SPARK_KV_PAGER_ADMITTED &&
		decision.park_evictions == 1u,
		"the second park-out takes the never-restored block");
	expect(SparkKvPagerReleaseAdmission(&fixture->pager,2u) == SPARK_STATUS_OK,
		"the unused reservation releases");
	/* step 5: block 0 parks once more and comes back a SECOND time */
	expect(CW2Admit(fixture,3u,&decision) &&
		decision.outcome == SPARK_KV_PAGER_ADMITTED &&
		decision.park_evictions == 1u,
		"the third park-out takes block 0 (its only resident)");
	expect(SparkKvPagerReleaseAdmission(&fixture->pager,3u) == SPARK_STATUS_OK,
		"the unused reservation releases");
	expect(CW2Restore(fixture,0u),"block 0 restores AGAIN (history 2)");
	expect(CW2Restore(fixture,1u),"block 1 restores (history 1)");
	/* THE VICTIM DECISION: one park-out, two candidates - block 0
	   (twice-restored, LRU-oldest) and block 1 (once-restored, newer).
	   Both are re-dirtied first (the oldest write-back receipt records
	   the victim), so the choice lands in the page-out receipt as well
	   as residency. */
	expect(SparkKvCacheArenaMarkBlockDirty(&fixture->arena,0u) ==
			SPARK_STATUS_OK &&
		SparkKvCacheArenaMarkBlockDirty(&fixture->arena,1u) ==
			SPARK_STATUS_OK,
		"both candidates are re-dirtied (0 stays the LRU-oldest)");
	expect(CW2Admit(fixture,2u,&decision) &&
		decision.outcome == SPARK_KV_PAGER_ADMITTED &&
		decision.park_evictions == 1u,
		"the final park-out chooses ONE victim of the two");
	expect(SparkKvPagerReleaseAdmission(&fixture->pager,1u) ==
			SPARK_STATUS_OK &&
		CW2FillBlock(fixture,2u),
		"block 2 fills the freed capacity");
	CW2CheckBudget(fixture,"sequence end");
	*zero_resident_out = CW2BlockIsResident(fixture,0u);
	*one_resident_out = CW2BlockIsResident(fixture,1u);
}

int main(void)
{
	setvbuf(stdout,0,_IONBF,0);
	printf("JIT-KV C5+W2: reuse-value park policy, the dispatch gate's"
		" deadline engine\n");

	printf("\nconfiguration fences\n");
	{
		CW2Fixture fixture;
		expect(CW2Open(&fixture,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE,0u) == SPARK_STATUS_OK,
			"the reuse-value fixture opens");
		expect(fixture.arena.eviction_policy ==
			SPARK_KV_CACHE_EVICTION_POLICY_REUSE_VALUE,
			"the pager installed the policy on its arena");
		expect(CW2Open(&fixture,3u,3u,8u,8u,1u,4u,1u,2u,0u) ==
			SPARK_STATUS_INVALID_ARGUMENT,
			"an unknown park policy is refused at Initialize");
		expect(CW2Open(&fixture,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_LRU,0u) == SPARK_STATUS_OK,
			"the LRU fixture opens (policy 0, the default)");
		expect(fixture.arena.eviction_policy ==
			SPARK_KV_CACHE_EVICTION_POLICY_LRU,
			"the default policy reads LRU");
	}

	printf("\nscenario 1: C5 - a twice-restored block SURVIVES the park"
		" that LRU would have made it\n");
	{
		CW2Fixture lru;
		CW2Fixture reuse;
		uint32_t zero_lru,one_lru,zero_reuse,one_reuse;

		expect(CW2Open(&lru,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_LRU,0u) == SPARK_STATUS_OK,
			"the LRU fixture opens");
		expect(CW2Open(&reuse,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE,0u) == SPARK_STATUS_OK,
			"the reuse-value fixture opens");
		CW2ReuseSequence(&lru,&zero_lru,&one_lru);
		expect(zero_lru == 0u && one_lru != 0u,
			"LRU: the TWICE-RESTORED block is the victim (it is the"
			" oldest)");
		expect(!CW2BlockIsResident(&lru,0u) && CW2BlockIsResident(&lru,1u) &&
			CW2BlockIsResident(&lru,2u),
			"LRU: blocks 1 and 2 hold the residency");
		CW2ReuseSequence(&reuse,&zero_reuse,&one_reuse);
		expect(zero_reuse != 0u && one_reuse == 0u,
			"THE PROOF: REUSE keeps the twice-restored block and parks"
			" the once-restored newer one - restored-again history"
			" outranks recency");
		expect(CW2BlockIsResident(&reuse,0u) &&
			!CW2BlockIsResident(&reuse,1u) &&
			CW2BlockIsResident(&reuse,2u),
			"REUSE: blocks 0 and 2 hold the residency");
		expect(reuse.arena.blocks[0].restore_count == 2u &&
			reuse.arena.blocks[1].restore_count == 1u &&
			lru.arena.blocks[0].restore_count == 2u,
			"the restored-again counters read 2 and 1 (the history is"
			" policy-independent)");
		/* C5 accounting invariance: the identical sequences must hold
		   identical budget state; only the victim moved. */
		expect(lru.pager.statistics.page_out_count ==
				reuse.pager.statistics.page_out_count &&
			lru.pager.statistics.page_out_bytes ==
				reuse.pager.statistics.page_out_bytes &&
			lru.pager.statistics.page_in_count ==
				reuse.pager.statistics.page_in_count &&
			lru.pager.statistics.page_in_misses ==
				reuse.pager.statistics.page_in_misses &&
			lru.pager.statistics.admission_requests ==
				reuse.pager.statistics.admission_requests &&
			lru.pager.statistics.admission_accepted ==
				reuse.pager.statistics.admission_accepted &&
			lru.pager.statistics.admission_queued_device ==
				reuse.pager.statistics.admission_queued_device &&
			lru.pager.statistics.admission_queued_backing ==
				reuse.pager.statistics.admission_queued_backing &&
			lru.pager.statistics.park_completions_published ==
				reuse.pager.statistics.park_completions_published &&
			lru.pager.statistics.park_write_failures ==
				reuse.pager.statistics.park_write_failures &&
			lru.arena.resident_block_count ==
				reuse.arena.resident_block_count &&
			lru.arena.reserved_block_count ==
				reuse.arena.reserved_block_count &&
			atomic_load(&lru.arena.unassigned_resident_block_count) ==
				atomic_load(&reuse.arena.unassigned_resident_block_count) &&
			lru.tier.slots_in_use == reuse.tier.slots_in_use,
			"ACCOUNTING: the policy changed NOTHING the budget reads");
		expect(lru.pager.statistics.page_out_history
				[(lru.pager.statistics.page_out_history_count - 1u) %
					SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY] == 0u &&
			reuse.pager.statistics.page_out_history
				[(reuse.pager.statistics.page_out_history_count - 1u) %
					SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY] == 1u,
			"...only WHICH block parked differs (the receipt agrees)");
	}

	printf("\nscenario 2: C5 - a clean cold block drops before a dirty"
		" warm one; dirtiness flips an LRU decision\n");
	{
		CW2Fixture lru;
		CW2Fixture reuse;
		SparkKvPagerAdmissionDecision decision;

		/* B1: clean-cold (0) vs dirty-warm (1) - both policies drop the
		   clean cold block. */
		expect(CW2Open(&lru,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_LRU,0u) == SPARK_STATUS_OK &&
			CW2Open(&reuse,3u,3u,8u,8u,1u,4u,1u,
				SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE,0u) ==
				SPARK_STATUS_OK,
			"the B1 fixtures open");
		{
			CW2Fixture *fixture;
			uint32_t round;
			for ( round = 0u; round < 2u; ++round )
			{
				fixture = (round == 0u) ? &lru : &reuse;
				expect(CW2Admit(fixture,2u,&decision) &&
					CW2FillBlock(fixture,0u) && CW2FillBlock(fixture,1u),
					"B1: lane A fills the budget");
				expect(CW2Admit(fixture,3u,&decision) &&
					decision.park_evictions == 2u,
					"B1: both blocks park");
				expect(SparkKvPagerReleaseAdmission(&fixture->pager,3u) ==
					SPARK_STATUS_OK,"B1: the reservation releases");
				expect(CW2Restore(fixture,0u) && CW2Restore(fixture,1u),
					"B1: both blocks restore (clean, 0 older)");
				expect(SparkKvCacheArenaMarkBlockDirty(&fixture->arena,1u) ==
					SPARK_STATUS_OK,
					"B1: block 1 is marked dirty (the warm one)");
				expect(CW2Admit(fixture,2u,&decision) &&
					decision.park_evictions == 1u,
					"B1: the park-out chooses a victim");
				expect(SparkKvPagerReleaseAdmission(&fixture->pager,1u) ==
						SPARK_STATUS_OK &&
					CW2FillBlock(fixture,2u),
					"B1: block 2 fills the freed capacity");
				expect(!CW2BlockIsResident(fixture,0u) &&
					CW2BlockIsResident(fixture,1u) &&
					CW2BlockIsResident(fixture,2u),
					round == 0u ?
						"B1 LRU: the clean cold block drops" :
						"B1 REUSE: the clean cold block STILL drops"
						" before the dirty warm one");
			}
		}

		/* B2: dirty-OLD (0) vs clean-NEW (1) - LRU ignores dirtiness and
		   evicts the dirty block (a write-back); the reuse policy keeps
		   it and drops the clean block (free to drop). */
		expect(CW2Open(&lru,3u,3u,8u,8u,1u,4u,1u,
			SPARK_KV_PAGER_PARK_POLICY_LRU,0u) == SPARK_STATUS_OK &&
			CW2Open(&reuse,3u,3u,8u,8u,1u,4u,1u,
				SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE,0u) ==
				SPARK_STATUS_OK,
			"the B2 fixtures open");
		{
			CW2Fixture *fixture;
			uint32_t round;
			for ( round = 0u; round < 2u; ++round )
			{
				fixture = (round == 0u) ? &lru : &reuse;
				expect(CW2Admit(fixture,2u,&decision) &&
					CW2FillBlock(fixture,0u) && CW2FillBlock(fixture,1u),
					"B2: lane A fills the budget");
				expect(CW2Admit(fixture,3u,&decision) &&
					decision.park_evictions == 2u,
					"B2: both blocks park");
				expect(SparkKvPagerReleaseAdmission(&fixture->pager,3u) ==
					SPARK_STATUS_OK,"B2: the reservation releases");
				expect(CW2Restore(fixture,0u) && CW2Restore(fixture,1u),
					"B2: both blocks restore (0 older)");
				expect(SparkKvCacheArenaMarkBlockDirty(&fixture->arena,0u) ==
					SPARK_STATUS_OK,
					"B2: the OLD block is marked dirty");
				expect(SparkKvCacheArenaMarkBlockResident(&fixture->arena,1u) ==
					SPARK_STATUS_OK,
					"B2: block 1 is touched (the LRU order keeps 0"
					" oldest)");
				expect(CW2Admit(fixture,2u,&decision) &&
					decision.park_evictions == 1u,
					"B2: the park-out chooses a victim");
				expect(SparkKvPagerReleaseAdmission(&fixture->pager,1u) ==
						SPARK_STATUS_OK &&
					CW2FillBlock(fixture,2u),
					"B2: block 2 fills the freed capacity");
				if ( round == 0u )
				{
					expect(!CW2BlockIsResident(fixture,0u) &&
						CW2BlockIsResident(fixture,1u),
						"B2 LRU: the DIRTY older block is the victim"
						" (recency alone)");
				}
				else
				{
					SparkKvCacheBlockView view;
					expect(CW2BlockIsResident(fixture,0u) &&
						!CW2BlockIsResident(fixture,1u),
						"THE PROOF: REUSE keeps the dirty old block and"
						" drops the clean younger one (a park of the"
						" dirty block would cost a write)");
					expect(SparkKvCacheArenaResolveBlock(&fixture->arena,
						0u,&view) == SPARK_STATUS_OK &&
						(view.flags &
							SPARK_KV_CACHE_BLOCK_FLAG_DIRTY) != 0u,
						"the survivor still holds its dirty flag");
				}
			}
		}
	}

	printf("\nscenario 3: W2 - under saturation, the deadline hint makes"
		" the tier serve the block the gate is waiting on\n");
	{
		CW2Fixture hinted;
		CW2Fixture baseline;
		SparkKvPagerDispatchDecision decision;
		SparkNvmeTierNeed needs[CW2_MAX_LOGICAL_BLOCKS];
		uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
		uint64_t offsets[CW2_MAX_LOGICAL_BLOCKS];
		uint64_t gated_slot,peer_slot;
		uint32_t run,block_index;
		SparkNvmeTierPlanReport report;

		/* one shared prologue: four parked blocks, the lookahead queueing
		   all of them at the same far deadline (FIFO inside it), and one
		   pump putting the first two backlog reads in flight on a drive
		   whose cancellations never land (staging saturated and
		   unstealable). The two fixtures are bit-identical up to the
		   gated dispatch. */
		for ( run = 0u; run < 2u; ++run )
		{
			CW2Fixture *fixture = (run == 0u) ? &hinted : &baseline;
			SparkKvPagerAdmissionDecision admission;
			expect(CW2Open(fixture,4u,4u,8u,8u,2u,2u,0u,
				SPARK_KV_PAGER_PARK_POLICY_LRU,1u) == SPARK_STATUS_OK,
				run == 0u ? "the hinted fixture opens (saturated"
					" cancellations)" :
				"the baseline fixture opens (same, hintless)");
			expect(CW2Admit(fixture,4u,&admission) &&
				CW2FillBlock(fixture,0u) && CW2FillBlock(fixture,1u) &&
				CW2FillBlock(fixture,2u) && CW2FillBlock(fixture,3u),
				"lane A fills the budget");
			expect(CW2Admit(fixture,4u,&admission) &&
				admission.park_evictions == 4u &&
				SparkKvPagerReleaseAdmission(&fixture->pager,4u) ==
					SPARK_STATUS_OK,
				"all four blocks park");
			for ( block_index = 0u; block_index < 4u; ++block_index )
			{
				CW2Digest(block_index,digest);
				memcpy(needs[block_index].content_digest,digest,
					SPARK_KV_PAGER_DIGEST_BYTES);
				needs[block_index].content_hash =
					CW2HashFromDigest(digest);
				needs[block_index].need_by_step = 100u;
				needs[block_index].reserved0 = 0u;
				expect(SparkNvmeTierOffsetOf(&fixture->tier,
					needs[block_index].content_hash,digest,
					&offsets[block_index]) == SPARK_STATUS_OK,
					"the parked record's offset resolves");
			}
			expect(SparkNvmeTierPlanLookahead(&fixture->tier,needs,4u,1u,
				&report) == SPARK_STATUS_OK && report.queued_count == 4u,
				"the lookahead queues all four restores at deadline 100");
			expect(SparkNvmeTierPump(&fixture->tier,1u) ==
				SPARK_STATUS_OK && fixture->tier.statistics.
				prefetch_issues == 2u,
				"one pump: two backlog reads in flight, the debt queue"
				" holds the rest");
		}
		/* the two fixtures are geometrically identical: the gated block 3
		   and its FIFO peer block 2 land on the same device slots */
		gated_slot = (offsets[3u] - CW2_TIER_BASE) / CW2_BLOCK_BYTES;
		peer_slot = (offsets[2u] - CW2_TIER_BASE) / CW2_BLOCK_BYTES;

		/* HINTED: the gate's offer ORDERs the gated block at its deadline;
		   the offer answers QUEUED, and the pump inside it serves the debt
		   queue head - the gated block, not the FIFO peer. */
		expect(CW2Dispatch(&hinted,3u,2u,&decision) == SPARK_STATUS_OK &&
			decision.outcome == SPARK_KV_PAGER_DISPATCH_QUEUED,
			"HINTED: the saturated gate answers QUEUED (never a hard"
			" error)");
		expect(hinted.tier.statistics.demand_deadline_orders == 1u,
			"the tier pulled the gated block forward in the debt queue");
		expect(CW2DispatchUntilReady(&hinted,3u,2u,&decision) &&
			CW2PlanesMatchGolden(&hinted,3u),
			"the same offer completes the gated block bit-exact");
		expect(CW2DispatchUntilReady(&hinted,2u,0u,&decision) &&
			CW2PlanesMatchGolden(&hinted,2u),
			"the FIFO peer completes after it");
		expect(hinted.device.first_completion[gated_slot] != 0u &&
			hinted.device.first_completion[peer_slot] != 0u &&
			hinted.device.first_completion[gated_slot] <
				hinted.device.first_completion[peer_slot],
			"THE PROOF: EDF - the gated block's read completed BEFORE"
			" its FIFO-older peer's");

		/* BASELINE: no hint - the gate yanks the block out of the debt
		   queue and spins its poll budget on the saturated tier; the
		   terminal answer is the SAME QUEUED the hinted gate gives (the
		   universal queue-not-wedge follow-up), not the hard IO_ERROR it
		   used to surface. The block lost its place and completes after
		   the FIFO peer - the order, not an error, is what the hint buys. */
		expect(CW2Dispatch(&baseline,3u,0u,&decision) ==
				SPARK_STATUS_OK &&
			decision.outcome == SPARK_KV_PAGER_DISPATCH_QUEUED,
			"BASELINE: the hintless gate spins to its poll limit and"
			" answers the SAME QUEUED - never a hard error");
		expect(baseline.tier.statistics.demand_deadline_orders == 0u,
			"no deadline order was ever stated");
		expect(baseline.tier.statistics.demand_stalls >= 1u,
			"the hintless path stalled instead of ordering");
		expect(CW2DispatchUntilReady(&baseline,2u,0u,&decision) &&
			CW2PlanesMatchGolden(&baseline,2u),
			"the FIFO peer completes (its own gate joins its read)");
		expect(CW2DispatchUntilReady(&baseline,3u,0u,&decision) &&
			CW2PlanesMatchGolden(&baseline,3u),
			"the gated block completes LAST (still no wedge)");
		expect(baseline.device.first_completion[gated_slot] != 0u &&
			baseline.device.first_completion[gated_slot] >
				baseline.device.first_completion[peer_slot],
			"...FIFO: the gated block's read completed AFTER its"
			" peer's");
		expect(hinted.pager.statistics.dispatch_queued >= 1u &&
			baseline.pager.statistics.dispatch_queued >= 1u,
			"the queue-not-wedge answer is UNIVERSAL now (both gates"
			" queue under the same saturation)");
		expect(hinted.tier.statistics.demand_deadline_orders ==
				baseline.tier.statistics.demand_deadline_orders + 1u,
			"...and what the hint alone buys is the ORDER (one deadline"
			" order, gated block first), not a different answer class");
	}

	printf("\n");
	if ( failures != 0 )
		printf("FAIL: %d check(s) failed\n",failures);
	else
		printf("JIT-KV C5+W2: all proofs green\n");
	return(failures != 0 ? 1 : 0);
}
