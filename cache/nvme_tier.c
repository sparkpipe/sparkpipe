#include "sparkpipe/spark_nvme_tier.h"

#include <string.h>

// The mechanics behind the contract in spark_nvme_tier.h. Three tables, all
// carved from one caller-provided blob at init (runtime/arena.h pattern: one
// allocation, no malloc afterwards):
//
//   SLOTS     one per block the budget buys; slot index IS the device record,
//             so the offset is multiplication and there is no second allocator
//             to disagree with the first about what is free.
//   STAGING   pinned DMA buffers, double-buffered at minimum: one fills from
//             the drive while the other's contents are consumed upstairs, so
//             the drive never waits on consumption and consumption never waits
//             on the drive.
//   PENDING   the lookahead min-heap, ordered by earliest deadline and FIFO
//             inside equal deadlines, so Pump always issues the most urgent
//             valid read while making queue mutation O(log n).
//
// EVICTION IS A CLOCK, and that is a load-bearing choice. A full LRU scan
// over a budget of millions of records costs microseconds-to-milliseconds on
// the path that admits a write-back, and an exact ordering buys nothing when
// the evicted bytes are re-fetchable: the cost of a wrong victim is one
// re-read, not a stall. Second-chance gives recency approximated by a single
// bit, O(1) amortised, and the generation counter makes eviction safe against
// DMA already in flight - a read landing in a recycled slot is waste to be
// discarded, never data to be believed.

#define NVME_TIER_SLOT_EMPTY 0u
#define NVME_TIER_SLOT_WRITING 1u    /* reserved device offset, not readable */
#define NVME_TIER_SLOT_PRESENT 2u    /* committed on the drive, nothing moving */
#define NVME_TIER_SLOT_FILLING 3u    /* a read into staging is in flight */
#define NVME_TIER_SLOT_READY 4u      /* landed in staging, awaiting Consume */

#define NVME_TIER_STAGING_FREE 0u
#define NVME_TIER_STAGING_FILLING 1u
#define NVME_TIER_STAGING_READY 2u
#define NVME_TIER_STAGING_CANCEL_PENDING 3u

#define NVME_TIER_HOLDER_NONE 0u
#define NVME_TIER_HOLDER_PREFETCH 1u
#define NVME_TIER_HOLDER_DEMAND 2u

typedef struct NvmeTierSlot
{
	uint64_t content_hash;         /* 0 until first write reservation */
	uint64_t last_use;             /* monotonic tick, for observability */
	uint32_t next_in_bucket;
	uint32_t next_free;            /* free-list link while EMPTY */
	uint32_t generation;           /* bumped on every recycle */
	uint32_t need_by_step;         /* earliest deadline anyone has stated */
	uint32_t issued_step;          /* when the in-flight read was submitted */
	uint32_t pin_count;
	uint32_t staging_index;        /* while FILLING or READY */
	uint8_t state;
	uint8_t referenced;            /* the clock's second-chance bit */
	uint8_t queued;                /* sits in the pending queue */
	uint8_t reserved0;
	uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
	/* B3 TIER INTEGRITY: SHA-256 of the payload this record stands for,
	 * presented by the writer at ReserveWrite and demanded back by every
	 * reader and every landing. The 64-bit content_hash buckets; THIS is
	 * the identity. A digest that does not match is a collision or drive
	 * corruption: fail loud (HASH_MISMATCH), quarantine the record, never
	 * hand back wrong-KV. */
}
NvmeTierSlot;

typedef struct NvmeTierStagingState
{
	uint64_t ticket;
	uint32_t slot;
	uint32_t generation;           /* of the slot at issue time */
	uint32_t need_by_step;
	uint32_t issued_step;
	uint8_t state;
	uint8_t holder;                /* PREFETCH is preemptible, DEMAND is not */
	uint8_t reserved0;
	uint8_t reserved1;
}
NvmeTierStagingState;

typedef struct NvmeTierPendingEntry
{
	uint32_t slot;
	uint32_t generation;           /* of the slot at enqueue time */
	uint32_t need_by_step;
	uint32_t reserved0;
	uint64_t order;                /* FIFO tie-break inside one deadline */
}
NvmeTierPendingEntry;

typedef struct NvmeTierPendingQueue
{
	NvmeTierPendingEntry *entries;
	uint32_t count;
	uint32_t capacity;
	uint64_t next_order;
}
NvmeTierPendingQueue;

static uint32_t NvmeTierSlotCountForBudget(
	const SparkNvmeTierConfiguration *configuration)
{
	uint64_t count;
	if ( configuration->block_bytes == 0u )
		return(0u);
	count = configuration->budget_bytes / configuration->block_bytes;
	if ( count > 0xfffffffeULL )
		count = 0xfffffffeULL;
	return((uint32_t)count);
}

static uint64_t NvmeTierAlignU64(uint64_t value, uint64_t alignment)
{
	if ( alignment == 0u || ( alignment & ( alignment - 1u ) ) != 0u )
		return(UINT64_MAX);
	if ( value > UINT64_MAX - ( alignment - 1u ) )
		return(UINT64_MAX);
	return((value + alignment - 1u) & ~(alignment - 1u));
}


static uint64_t NvmeTierSaturatingAddU64(uint64_t left, uint64_t right)
{
	if ( left > UINT64_MAX - right )
		return(UINT64_MAX);
	return(left + right);
}

static uint64_t NvmeTierSaturatingMultiplyU64(uint64_t left, uint64_t right)
{
	if ( left != 0u && right > UINT64_MAX / left )
		return(UINT64_MAX);
	return(left * right);
}

// Exact floor(left * right / divisor) unless the mathematical result exceeds
// uint64_t, in which case it saturates. Decomposing both operands around the
// divisor prevents the intermediate product from wrapping before division.
static uint64_t NvmeTierSaturatingMultiplyDivideU64(
	uint64_t left,
	uint64_t right,
	uint64_t divisor)
{
	uint64_t left_quotient;
	uint64_t left_remainder;
	uint64_t right_quotient;
	uint64_t right_remainder;
	uint64_t result;
	uint64_t term;

	if ( divisor == 0u )
		return(UINT64_MAX);
	left_quotient = left / divisor;
	left_remainder = left % divisor;
	right_quotient = right / divisor;
	right_remainder = right % divisor;

	result = NvmeTierSaturatingMultiplyU64(left_quotient,right);
	term = NvmeTierSaturatingMultiplyU64(left_remainder,right_quotient);
	result = NvmeTierSaturatingAddU64(result,term);
	term = ( left_remainder * right_remainder ) / divisor;
	return(NvmeTierSaturatingAddU64(result,term));
}

static uint32_t NvmeTierAppendTableRegion(
	uint64_t *offset,
	uint64_t count,
	uint64_t element_bytes,
	uint64_t alignment)
{
	uint64_t bytes;
	uint64_t aligned;
	if ( offset == 0 || count == 0u || element_bytes == 0u )
		return(0u);
	if ( count > UINT64_MAX / element_bytes )
		return(0u);
	bytes = count * element_bytes;
	if ( *offset > UINT64_MAX - bytes )
		return(0u);
	aligned = NvmeTierAlignU64(*offset + bytes,alignment);
	if ( aligned == UINT64_MAX )
		return(0u);
	*offset = aligned;
	return(1u);
}

uint64_t SparkNvmeTierTableBytes(
	const SparkNvmeTierConfiguration *configuration)
{
	uint64_t total;
	uint32_t slot_count;
	if ( configuration == 0 )
		return(0u);
	slot_count = NvmeTierSlotCountForBudget(configuration);
	if ( slot_count == 0u || configuration->hash_bucket_count == 0u
		|| configuration->pending_capacity == 0u
		|| configuration->staging_buffer_count == 0u )
		return(0u);
	total = 0u;
	if ( NvmeTierAppendTableRegion(&total,slot_count,sizeof(NvmeTierSlot),8u) == 0u
		|| NvmeTierAppendTableRegion(&total,configuration->hash_bucket_count,
			sizeof(uint32_t),8u) == 0u
		|| NvmeTierAppendTableRegion(&total,1u,sizeof(NvmeTierPendingQueue),8u) == 0u
		|| NvmeTierAppendTableRegion(&total,configuration->pending_capacity,
			sizeof(NvmeTierPendingEntry),8u) == 0u
		|| NvmeTierAppendTableRegion(&total,configuration->staging_buffer_count,
			sizeof(NvmeTierStagingState),8u) == 0u )
		return(0u);
	return(total);
}

static uint32_t NvmeTierLookup(
	const SparkNvmeTier *tier,
	uint64_t content_hash)
{
	uint32_t walk;
	const NvmeTierSlot *slots = (const NvmeTierSlot *)tier->slots;
	if ( content_hash == 0u )
		return(SPARK_NVME_TIER_NO_SLOT);
	walk = tier->buckets[content_hash % tier->configuration.hash_bucket_count];
	while ( walk != SPARK_NVME_TIER_NO_SLOT )
	{
		if ( slots[walk].state != NVME_TIER_SLOT_EMPTY
			&& slots[walk].content_hash == content_hash )
			return(walk);
		walk = slots[walk].next_in_bucket;
	}
	return(SPARK_NVME_TIER_NO_SLOT);
}

// B3 TIER INTEGRITY. The 64-bit content_hash is a bucket key, not an
// identity: two tenants' blocks can collide on it, and serving either
// caller the other's bytes is silent cross-tenant KV corruption. Every
// record stores the SHA-256 its writer presented; every reader must present
// the digest it expects and every restored buffer must match the record's
// digest. Mismatch is SPARK_STATUS_HASH_MISMATCH - loud, never wrong-KV.
static uint32_t NvmeTierDigestIsUsable(
	const uint8_t *digest)
{
	uint32_t index;
	if ( digest == 0 )
		return(0u);
	for ( index = 0u; index < SPARK_NVME_TIER_DIGEST_BYTES; ++index )
		if ( digest[index] != 0u )
			return(1u);
	return(0u);
}

static uint32_t NvmeTierDigestMatches(
	const uint8_t *stored,
	const uint8_t *presented)
{
	return( memcmp(stored,presented,SPARK_NVME_TIER_DIGEST_BYTES) == 0 ?
		1u : 0u );
}

// Resolve hash -> slot AND bind the presented digest to it when one is
// given. Returns the slot index, or SPARK_NVME_TIER_NO_SLOT with a status
// explaining: NOT_FOUND (absent), HASH_MISMATCH (collision or corruption -
// never alias), OK. An unusable (NULL/zero) digest is KEY-ONLY bookkeeping
// access: OffsetOf, Pin and the planning paths classify records they never
// hand bytes from, so they may pass NULL. The data-carrying APIs
// (ReserveWrite, RequestDemand, Consume) reject an unusable digest before
// reaching here - bytes never move on a key-only match.
static uint32_t NvmeTierLookupVerified(
	const SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t *content_digest,
	SparkStatus *status_out)
{
	uint32_t slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
	{
		if ( status_out != 0 )
			*status_out = SPARK_STATUS_NOT_FOUND;
		return(SPARK_NVME_TIER_NO_SLOT);
	}
	if ( NvmeTierDigestIsUsable(content_digest) != 0u )
	{
		const NvmeTierSlot *slots = (const NvmeTierSlot *)tier->slots;
		if ( NvmeTierDigestMatches(slots[slot_index].digest,
				content_digest) == 0u )
		{
			if ( status_out != 0 )
				*status_out = SPARK_STATUS_HASH_MISMATCH;
			return(SPARK_NVME_TIER_NO_SLOT);
		}
	}
	if ( status_out != 0 )
		*status_out = SPARK_STATUS_OK;
	return(slot_index);
}

static void NvmeTierBucketInsert(SparkNvmeTier *tier, uint32_t slot_index)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t bucket = (uint32_t)(slots[slot_index].content_hash
		% tier->configuration.hash_bucket_count);
	slots[slot_index].next_in_bucket = tier->buckets[bucket];
	tier->buckets[bucket] = slot_index;
}

static void NvmeTierBucketRemove(SparkNvmeTier *tier, uint32_t slot_index)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t bucket = (uint32_t)(slots[slot_index].content_hash
		% tier->configuration.hash_bucket_count);
	uint32_t walk = tier->buckets[bucket];
	uint32_t previous = SPARK_NVME_TIER_NO_SLOT;
	while ( walk != SPARK_NVME_TIER_NO_SLOT && walk != slot_index )
	{
		previous = walk;
		walk = slots[walk].next_in_bucket;
	}
	if ( walk == slot_index )
	{
		if ( previous == SPARK_NVME_TIER_NO_SLOT )
			tier->buckets[bucket] = slots[slot_index].next_in_bucket;
		else
			slots[previous].next_in_bucket = slots[slot_index].next_in_bucket;
	}
	slots[slot_index].next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
}

SparkStatus SparkNvmeTierInitialize(
	SparkNvmeTier *tier,
	const SparkNvmeTierConfiguration *configuration,
	const SparkNvmeTierDevice *device,
	void *tables,
	uint64_t tables_bytes,
	void *staging,
	uint64_t staging_bytes)
{
	uint64_t bytes_per_step;
	uint64_t offset;
	uint64_t required_staging_bytes;
	uint64_t required_table_bytes;
	uint32_t slot_count;
	uint32_t index;
	uint8_t *table_bytes;
	NvmeTierSlot *slots;
	NvmeTierStagingState *staging_states;
	NvmeTierPendingQueue *queue;

	if ( tier == 0 || configuration == 0 || device == 0
		|| tables == 0 || staging == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_NVME_TIER_ABI_VERSION
		|| configuration->descriptor_bytes != SPARK_NVME_TIER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);

	slot_count = NvmeTierSlotCountForBudget(configuration);
	if ( slot_count == 0u
		|| configuration->hash_bucket_count == 0u
		|| configuration->pending_capacity == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ( configuration->block_bytes % SPARK_NVME_TIER_IO_ALIGNMENT_BYTES ) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->staging_buffer_count < 2u
		|| configuration->staging_buffer_count > SPARK_NVME_TIER_MAX_STAGING_BUFFERS
		|| configuration->demand_reserve_buffers >= configuration->staging_buffer_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->device_bytes_per_second == 0u
		|| configuration->step_time_microseconds == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( device->submit_read == 0 || device->poll_read == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( ( (uintptr_t)tables % 8u ) != 0u
		|| ( (uintptr_t)staging % SPARK_NVME_TIER_IO_ALIGNMENT_BYTES ) != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);

	required_table_bytes = SparkNvmeTierTableBytes(configuration);
	if ( required_table_bytes == 0u || tables_bytes < required_table_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( configuration->staging_buffer_count >
		UINT64_MAX / configuration->block_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	required_staging_bytes =
		(uint64_t)configuration->staging_buffer_count * configuration->block_bytes;
	if ( staging_bytes < required_staging_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);

	memset(tier,0,sizeof(*tier));
	memset(tables,0,(size_t)required_table_bytes);
	tier->configuration = *configuration;
	tier->device = *device;
	tier->slot_count = slot_count;
	tier->staging = (uint8_t *)staging;
	table_bytes = (uint8_t *)tables;
	offset = 0u;

	tier->slots = table_bytes + offset;
	offset = NvmeTierAlignU64(
		offset + (uint64_t)slot_count * sizeof(NvmeTierSlot),8u);
	tier->buckets = (uint32_t *)(void *)(table_bytes + offset);
	offset = NvmeTierAlignU64(
		offset + (uint64_t)configuration->hash_bucket_count * sizeof(uint32_t),8u);
	tier->pending = table_bytes + offset;
	offset = NvmeTierAlignU64(offset + sizeof(NvmeTierPendingQueue),8u);
	queue = (NvmeTierPendingQueue *)tier->pending;
	queue->entries = (NvmeTierPendingEntry *)(void *)(table_bytes + offset);
	offset = NvmeTierAlignU64(
		offset + (uint64_t)configuration->pending_capacity * sizeof(NvmeTierPendingEntry),8u);
	tier->staging_state = table_bytes + offset;
	offset = NvmeTierAlignU64(
		offset + (uint64_t)configuration->staging_buffer_count * sizeof(NvmeTierStagingState),8u);
	if ( offset != required_table_bytes )
	{
		memset(tier,0,sizeof(*tier));
		return(SPARK_STATUS_INTERNAL_ERROR);
	}

	slots = (NvmeTierSlot *)tier->slots;
	for ( index = 0u; index < slot_count; ++index )
	{
		slots[index].content_hash = 0u;
		slots[index].state = NVME_TIER_SLOT_EMPTY;
		slots[index].generation = 0u;
		slots[index].need_by_step = 0xffffffffu;
		slots[index].next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
		slots[index].next_free = index + 1u;
		slots[index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	}
	slots[slot_count - 1u].next_free = SPARK_NVME_TIER_NO_SLOT;
	tier->free_head = 0u;
	for ( index = 0u; index < configuration->hash_bucket_count; ++index )
		tier->buckets[index] = SPARK_NVME_TIER_NO_SLOT;

	staging_states = (NvmeTierStagingState *)tier->staging_state;
	for ( index = 0u; index < configuration->staging_buffer_count; ++index )
	{
		staging_states[index].state = NVME_TIER_STAGING_FREE;
		staging_states[index].holder = NVME_TIER_HOLDER_NONE;
		staging_states[index].slot = SPARK_NVME_TIER_NO_SLOT;
	}

	queue->count = 0u;
	queue->capacity = configuration->pending_capacity;
	queue->next_order = 0u;
	tier->tick = 1u;
	tier->clock_hand = 0u;

	bytes_per_step = NvmeTierSaturatingMultiplyDivideU64(
		configuration->device_bytes_per_second,
		configuration->step_time_microseconds,
		1000000ULL);
	if ( bytes_per_step == 0u )
		bytes_per_step = 1u;
	if ( bytes_per_step > 0xffffffffULL )
		bytes_per_step = 0xffffffffULL;
	tier->bytes_per_step = (uint32_t)bytes_per_step;
	tier->transfer_steps = (uint32_t)(
		( configuration->block_bytes / bytes_per_step )
		+ ( ( configuration->block_bytes % bytes_per_step ) != 0u ? 1u : 0u ) );
	if ( tier->transfer_steps == 0u )
		tier->transfer_steps = 1u;
	tier->statistics.slot_count = slot_count;
	return(SPARK_STATUS_OK);
}

// Release a staging buffer back to FREE, detaching it from its slot. The slot
// drops to PRESENT when it still names this buffer: the on-drive record
// outlives every staging cycle, which is what makes dropping a prefetch cheap.
static void NvmeTierStagingRelease(SparkNvmeTier *tier, uint32_t staging_index)
{
	NvmeTierStagingState *staging_states;
	NvmeTierSlot *slots;
	uint32_t slot_index;

	staging_states = (NvmeTierStagingState *)tier->staging_state;
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = staging_states[staging_index].slot;
	if ( slot_index != SPARK_NVME_TIER_NO_SLOT
		&& slot_index < tier->slot_count
		&& slots[slot_index].staging_index == staging_index
		&& ( slots[slot_index].state == NVME_TIER_SLOT_FILLING
			|| slots[slot_index].state == NVME_TIER_SLOT_READY ) )
	{
		slots[slot_index].state = NVME_TIER_SLOT_PRESENT;
		slots[slot_index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	}
	memset(&staging_states[staging_index],0,sizeof(staging_states[staging_index]));
	staging_states[staging_index].state = NVME_TIER_STAGING_FREE;
	staging_states[staging_index].holder = NVME_TIER_HOLDER_NONE;
	staging_states[staging_index].slot = SPARK_NVME_TIER_NO_SLOT;
}

// The eviction clock. Two revolutions at worst: the first clears second-chance
// bits, the second is guaranteed to meet a victim it cleared - unless every
// slot is one it must not touch:
//
//   pinned        - admission promised this block to a scheduled sequence.
//   demand-held   - its staging buffer is the decode path's data; evicting it
//                   under the reader is the one eviction that WOULD stall a
//                   step, which is the thing this tier exists to prevent.
//   filling, no cancel - the device owns the staging buffer until the read
//                   lands; recycling the slot now would let late DMA land in a
//                   buffer already reused for someone else.
//
// A single revolution cannot evict anything once every slot has its grace
// bit set - found by the test's ninth publish into an 8-record tier. Two
// revolutions with no victim means the tier is genuinely wedged (every record
// pinned or demand-held) and the caller hears BUSY - loud, instead of the
// silent alternative of evicting a block a scheduled sequence is about to
// read.
static uint32_t NvmeTierClockEvict(SparkNvmeTier *tier)
{
	NvmeTierSlot *slots;
	NvmeTierStagingState *staging_states;
	uint32_t probe;

	slots = (NvmeTierSlot *)tier->slots;
	staging_states = (NvmeTierStagingState *)tier->staging_state;
	for ( probe = 0u; probe < 2u * tier->slot_count; ++probe )
	{
		NvmeTierSlot *slot;
		NvmeTierStagingState *held;
		uint32_t index;

		index = tier->clock_hand;
		slot = &slots[index];
		held = 0;
		tier->clock_hand = ( tier->clock_hand + 1u ) % tier->slot_count;
		if ( slot->state == NVME_TIER_SLOT_EMPTY
			|| slot->state == NVME_TIER_SLOT_WRITING )
			continue;
		if ( slot->pin_count != 0u )
		{
			tier->statistics.pinned_eviction_skips++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING
			|| slot->state == NVME_TIER_SLOT_READY )
		{
			if ( slot->staging_index >=
				tier->configuration.staging_buffer_count )
				continue;
			held = &staging_states[slot->staging_index];
			if ( held->holder == NVME_TIER_HOLDER_DEMAND )
				continue;
			if ( slot->state == NVME_TIER_SLOT_FILLING
				&& ( tier->device.cancel_read == 0
					|| held->state == NVME_TIER_STAGING_CANCEL_PENDING ) )
				continue;
		}
		if ( slot->referenced != 0u )
		{
			slot->referenced = 0u;
			continue;
		}

		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			SparkStatus cancel_status;

			cancel_status = tier->device.cancel_read(
				tier->device.context,held->ticket);
			if ( cancel_status == SPARK_STATUS_OK )
			{
				NvmeTierStagingRelease(tier,slot->staging_index);
			}
			else if ( cancel_status == SPARK_STATUS_BUSY
				|| cancel_status == SPARK_STATUS_PENDING )
			{
				held->state = NVME_TIER_STAGING_CANCEL_PENDING;
				tier->statistics.cancel_pending_count++;
				continue;
			}
			else
			{
				tier->statistics.io_errors++;
				NvmeTierStagingRelease(tier,slot->staging_index);
			}
		}
		else if ( slot->state == NVME_TIER_SLOT_READY )
		{
			if ( held->holder == NVME_TIER_HOLDER_PREFETCH )
				tier->statistics.prefetch_dropped++;
			NvmeTierStagingRelease(tier,slot->staging_index);
		}

		slot->queued = 0u;
		NvmeTierBucketRemove(tier,index);
		slot->content_hash = 0u;
		memset(slot->digest,0,sizeof(slot->digest));
		slot->state = NVME_TIER_SLOT_EMPTY;
		slot->generation++;
		if ( slot->generation == 0u )
			slot->generation = 1u;
		slot->pin_count = 0u;
		slot->next_free = tier->free_head;
		tier->free_head = index;
		tier->slots_in_use--;
		tier->statistics.evictions++;
		tier->statistics.slots_in_use = tier->slots_in_use;
		return(index);
	}
	return(SPARK_NVME_TIER_NO_SLOT);
}

static uint32_t NvmeTierSlotAcquire(SparkNvmeTier *tier)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	uint32_t index;
	if ( tier->free_head != SPARK_NVME_TIER_NO_SLOT )
	{
		index = tier->free_head;
		tier->free_head = slots[index].next_free;
		slots[index].next_free = SPARK_NVME_TIER_NO_SLOT;
		return(index);
	}
	if ( NvmeTierClockEvict(tier) == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_NVME_TIER_NO_SLOT);
	// The clock pushed its victim onto the free list.
	index = tier->free_head;
	tier->free_head = slots[index].next_free;
	slots[index].next_free = SPARK_NVME_TIER_NO_SLOT;
	return(index);
}

static void NvmeTierReleaseReservedSlot(
	SparkNvmeTier *tier,
	uint32_t slot_index)
{
	NvmeTierSlot *slots;
	NvmeTierSlot *slot;

	slots = (NvmeTierSlot *)tier->slots;
	slot = &slots[slot_index];
	slot->content_hash = 0u;
	memset(slot->digest,0,sizeof(slot->digest));
	slot->state = NVME_TIER_SLOT_EMPTY;
	slot->generation++;
	if ( slot->generation == 0u )
		slot->generation = 1u;
	slot->need_by_step = 0xffffffffu;
	slot->staging_index = SPARK_NVME_TIER_NO_SLOT;
	slot->next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
	slot->next_free = tier->free_head;
	tier->free_head = slot_index;
	if ( tier->slots_in_use != 0u )
		tier->slots_in_use--;
	tier->statistics.slots_in_use = tier->slots_in_use;
}

SparkStatus SparkNvmeTierReserveWrite(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	SparkNvmeTierWriteReservation *reservation_out)
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	uint32_t index;

	if ( tier == 0 || content_hash == 0u || reservation_out == 0 ||
		NvmeTierDigestIsUsable(content_digest) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(reservation_out,0,sizeof(*reservation_out));
	memcpy(reservation_out->content_digest,content_digest,
		SPARK_NVME_TIER_DIGEST_BYTES);
	slots = (NvmeTierSlot *)tier->slots;

	slot_index = NvmeTierLookup(tier,content_hash);
	if ( slot_index != SPARK_NVME_TIER_NO_SLOT )
	{
		/* B3: same 64-bit hash, DIFFERENT digest = collision. Fail loud;
		 * treating it as already_present would alias two tenants' KV. */
		if ( NvmeTierDigestMatches(slots[slot_index].digest,
				content_digest) == 0u )
		{
			tier->statistics.digest_mismatches++;
			return(SPARK_STATUS_HASH_MISMATCH);
		}
		slots[slot_index].last_use = tier->tick++;
		slots[slot_index].referenced = 1u;
		reservation_out->content_hash = content_hash;
		reservation_out->device_offset = tier->configuration.base_offset
			+ (uint64_t)slot_index * tier->configuration.block_bytes;
		reservation_out->slot_index = slot_index;
		reservation_out->generation = slots[slot_index].generation;
		reservation_out->already_present = 1u;
		return(SPARK_STATUS_OK);
	}

	for ( index = 0u; index < tier->slot_count; ++index )
	{
		if ( slots[index].state == NVME_TIER_SLOT_WRITING
			&& slots[index].content_hash == content_hash )
		{
			if ( NvmeTierDigestMatches(slots[index].digest,
					content_digest) == 0u )
			{
				tier->statistics.digest_mismatches++;
				return(SPARK_STATUS_HASH_MISMATCH);
			}
			return(SPARK_STATUS_BUSY);
		}
	}

	slot_index = NvmeTierSlotAcquire(tier);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(SPARK_STATUS_BUSY);

	slots[slot_index].generation++;
	if ( slots[slot_index].generation == 0u )
		slots[slot_index].generation = 1u;
	slots[slot_index].content_hash = content_hash;
	memcpy(slots[slot_index].digest,content_digest,
		SPARK_NVME_TIER_DIGEST_BYTES);
	slots[slot_index].state = NVME_TIER_SLOT_WRITING;
	slots[slot_index].last_use = tier->tick++;
	slots[slot_index].referenced = 1u;
	slots[slot_index].pin_count = 0u;
	slots[slot_index].queued = 0u;
	slots[slot_index].need_by_step = 0xffffffffu;
	slots[slot_index].staging_index = SPARK_NVME_TIER_NO_SLOT;
	slots[slot_index].next_in_bucket = SPARK_NVME_TIER_NO_SLOT;
	tier->slots_in_use++;
	tier->statistics.write_reservations++;
	tier->statistics.slots_in_use = tier->slots_in_use;

	reservation_out->content_hash = content_hash;
	reservation_out->device_offset = tier->configuration.base_offset
		+ (uint64_t)slot_index * tier->configuration.block_bytes;
	reservation_out->slot_index = slot_index;
	reservation_out->generation = slots[slot_index].generation;
	reservation_out->already_present = 0u;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierCommitWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation)
{
	NvmeTierSlot *slots;
	NvmeTierSlot *slot;
	uint32_t existing;

	if ( tier == 0 || reservation == 0 || reservation->content_hash == 0u ||
		NvmeTierDigestIsUsable(reservation->content_digest) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	if ( reservation->already_present != 0u )
	{
		existing = NvmeTierLookup(tier,reservation->content_hash);
		if ( existing != reservation->slot_index
			|| existing >= tier->slot_count
			|| slots[existing].generation != reservation->generation )
			return(SPARK_STATUS_VALIDATION_FAILED);
		return(SPARK_STATUS_OK);
	}
	if ( reservation->slot_index >= tier->slot_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = &slots[reservation->slot_index];
	if ( slot->state != NVME_TIER_SLOT_WRITING
		|| slot->content_hash != reservation->content_hash
		|| slot->generation != reservation->generation
		|| NvmeTierDigestMatches(slot->digest,
			reservation->content_digest) == 0u
		|| reservation->device_offset != tier->configuration.base_offset
			+ (uint64_t)reservation->slot_index * tier->configuration.block_bytes )
		return(SPARK_STATUS_VALIDATION_FAILED);

	slot->state = NVME_TIER_SLOT_PRESENT;
	NvmeTierBucketInsert(tier,reservation->slot_index);
	tier->statistics.publishes++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierAbortWrite(
	SparkNvmeTier *tier,
	const SparkNvmeTierWriteReservation *reservation)
{
	NvmeTierSlot *slots;
	NvmeTierSlot *slot;

	if ( tier == 0 || reservation == 0 || reservation->content_hash == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( reservation->already_present != 0u )
		return(SPARK_STATUS_OK);
	if ( reservation->slot_index >= tier->slot_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot = &slots[reservation->slot_index];
	if ( slot->state != NVME_TIER_SLOT_WRITING
		|| slot->content_hash != reservation->content_hash
		|| slot->generation != reservation->generation )
		return(SPARK_STATUS_VALIDATION_FAILED);
	NvmeTierReleaseReservedSlot(tier,reservation->slot_index);
	tier->statistics.write_aborts++;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierOffsetOf(
	const SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	uint64_t *device_offset_out)
{
	uint32_t slot_index;
	SparkStatus status;
	if ( tier == 0 || device_offset_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot_index = NvmeTierLookupVerified(tier,content_hash,content_digest,
		&status);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		return(status);
	*device_offset_out = tier->configuration.base_offset
		+ (uint64_t)slot_index * tier->configuration.block_bytes;
	return(SPARK_STATUS_OK);
}

// -- the pending queue ------------------------------------------------------------
//
// Bounded binary min-heap ordered by (need_by_step, insertion order). The old
// sorted array shifted every later entry on insertion, removal, and deadline
// tightening. At the lookahead capacities used for long-context scheduling that
// turned a burst into repeated O(n) memory traffic on the owner thread. A heap
// retains exact FIFO tie ordering while making each mutation O(log n). The
// occasional slot lookup remains a bounded linear search; it reads only compact
// metadata and never moves the queue payload.

static uint32_t NvmeTierPendingLess(
	const NvmeTierPendingEntry *left,
	const NvmeTierPendingEntry *right)
{
	if ( left->need_by_step != right->need_by_step )
		return(left->need_by_step < right->need_by_step);
	return(left->order < right->order);
}

static void NvmeTierPendingSwap(
	NvmeTierPendingEntry *left,
	NvmeTierPendingEntry *right)
{
	NvmeTierPendingEntry temporary;

	temporary = *left;
	*left = *right;
	*right = temporary;
}

static void NvmeTierPendingSiftUp(
	NvmeTierPendingQueue *queue,
	uint32_t position)
{
	while ( position != 0u )
	{
		uint32_t parent;

		parent = ( position - 1u ) / 2u;
		if ( NvmeTierPendingLess(
				&queue->entries[parent],
				&queue->entries[position]) != 0u )
			break;
		NvmeTierPendingSwap(
			&queue->entries[parent],
			&queue->entries[position]);
		position = parent;
	}
}

static void NvmeTierPendingSiftDown(
	NvmeTierPendingQueue *queue,
	uint32_t position)
{
	for ( ;; )
	{
		uint32_t left;
		uint32_t right;
		uint32_t smallest;

		left = ( position * 2u ) + 1u;
		if ( left >= queue->count )
			break;
		right = left + 1u;
		smallest = left;
		if ( right < queue->count
			&& NvmeTierPendingLess(
				&queue->entries[right],
				&queue->entries[left]) != 0u )
			smallest = right;
		if ( NvmeTierPendingLess(
				&queue->entries[position],
				&queue->entries[smallest]) != 0u )
			break;
		NvmeTierPendingSwap(
			&queue->entries[position],
			&queue->entries[smallest]);
		position = smallest;
	}
}

static int32_t NvmeTierPendingFind(
	const NvmeTierPendingQueue *queue,
	uint32_t slot_index)
{
	uint32_t index;

	for ( index = 0u; index < queue->count; ++index )
		if ( queue->entries[index].slot == slot_index )
			return((int32_t)index);
	return(-1);
}

static void NvmeTierPendingInsert(
	NvmeTierPendingQueue *queue,
	uint32_t slot_index,
	uint32_t generation,
	uint32_t need_by_step)
{
	uint32_t position;

	position = queue->count;
	queue->entries[position].slot = slot_index;
	queue->entries[position].generation = generation;
	queue->entries[position].need_by_step = need_by_step;
	queue->entries[position].order = queue->next_order++;
	queue->count++;
	NvmeTierPendingSiftUp(queue,position);
}

static void NvmeTierPendingRemoveAt(
	NvmeTierPendingQueue *queue,
	uint32_t position)
{
	if ( position >= queue->count )
		return;
	queue->count--;
	if ( position == queue->count )
		return;
	queue->entries[position] = queue->entries[queue->count];
	if ( position != 0u
		&& NvmeTierPendingLess(
			&queue->entries[position],
			&queue->entries[( position - 1u ) / 2u]) != 0u )
		NvmeTierPendingSiftUp(queue,position);
	else
		NvmeTierPendingSiftDown(queue,position);
}

// An earlier deadline decreases the heap key. The original insertion order is
// retained so equal deadlines remain FIFO after tightening.
static void NvmeTierPendingTighten(
	NvmeTierPendingQueue *queue,
	uint32_t position,
	uint32_t need_by_step)
{
	if ( position >= queue->count
		|| queue->entries[position].need_by_step <= need_by_step )
		return;
	queue->entries[position].need_by_step = need_by_step;
	NvmeTierPendingSiftUp(queue,position);
}

SparkStatus SparkNvmeTierPlanLookahead(
	SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	SparkNvmeTierPlanReport *report_out)
{
	NvmeTierSlot *slots;
	NvmeTierPendingQueue *queue;
	SparkNvmeTierPlanReport report;
	uint32_t index;
	if ( tier == 0 || ( needs == 0 && need_count != 0u ) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&report,0,sizeof(report));
	slots = (NvmeTierSlot *)tier->slots;
	queue = (NvmeTierPendingQueue *)tier->pending;
	for ( index = 0u; index < need_count; ++index )
	{
		uint64_t hash = needs[index].content_hash;
		uint32_t need_by = needs[index].need_by_step;
		SparkStatus lookup_status;
		uint32_t slot_index = NvmeTierLookupVerified(tier,hash,
			needs[index].content_digest,&lookup_status);
		NvmeTierSlot *slot;
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT &&
			lookup_status == SPARK_STATUS_HASH_MISMATCH )
		{
			// The hash resolves but the digest contradicts it: admitting
			// against this record would serve the wrong tenant's KV.
			return(SPARK_STATUS_HASH_MISMATCH);
		}
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		{
			// Not on the drive. Admission must see this: planning around an
			// absent block as if it were queued is how a sequence gets admitted
			// warm and starts cold.
			report.absent_count++;
			continue;
		}
		slot = &slots[slot_index];
		slot->last_use = tier->tick++;
		slot->referenced = 1u;      /* someone will need it: the clock should know */
		if ( need_by < slot->need_by_step || slot->state == NVME_TIER_SLOT_PRESENT )
			slot->need_by_step = need_by;
		if ( slot->state == NVME_TIER_SLOT_READY )
		{
			report.already_ready_count++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			NvmeTierStagingState *held
				= &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
			if ( need_by < held->need_by_step )
				held->need_by_step = need_by;
			report.already_inflight_count++;
			continue;
		}
		// PRESENT. Queued already: an earlier deadline pulls it forward.
		{
			int32_t position = NvmeTierPendingFind(queue,slot_index);
			if ( position >= 0 )
			{
				NvmeTierPendingTighten(queue,(uint32_t)position,need_by);
				report.queued_count++;
				continue;
			}
		}
		// Late risk is reported, not hidden: a need closer than the transfer
		// time cannot arrive on schedule, and the caller prefers to know
		// before admission rather than at the stalled step. The read is still
		// queued - late bytes beat no bytes by exactly the recompute time.
		if ( need_by <= step_now
			|| need_by - step_now < tier->transfer_steps )
			report.late_risk_count++;
		if ( queue->count >= queue->capacity )
		{
			report.queue_full_count++;
			continue;
		}
		slot->queued = 1u;
		NvmeTierPendingInsert(queue,slot_index,slot->generation,need_by);
		report.queued_count++;
	}
	if ( report_out != 0 )
		*report_out = report;
	return(SPARK_STATUS_OK);
}

// Staging for a demand load. The ordering is the priority contract in code:
//
//   1. a FREE buffer;
//   2. a buffer holding a landed PREFETCH nobody has consumed - the bytes are
//      on the drive, so dropping them costs one future re-read at most;
//   3. a buffer a PREFETCH read is still filling, cancelled - the drive loses
//      some positioning, the decode step keeps its budget;
//   4. nothing: every buffer holds DEMAND data awaiting consumption, which is
//      a sizing bug and is counted as one (demand_stalls) rather than hidden
//      as latency.
//
// Pinned slots are skipped in 2 and 3: admission pinned them precisely because
// a scheduled sequence cannot afford to re-fetch them.
// A displaced prefetch is re-queued, not forgotten: the need that motivated it
// is still in the schedule, and dropping it outright would convert a cheap
// buffer hand-off into a demand load later - exactly the critical-path read
// the prefetch existed to prevent.
static void NvmeTierPrefetchRequeue(SparkNvmeTier *tier, uint32_t slot_index, uint32_t need_by_step)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierPendingQueue *queue = (NvmeTierPendingQueue *)tier->pending;
	if ( slots[slot_index].queued != 0u || queue->count >= queue->capacity )
		return;
	slots[slot_index].queued = 1u;
	NvmeTierPendingInsert(queue,slot_index,slots[slot_index].generation,need_by_step);
}

static uint32_t NvmeTierStagingAcquireForDemand(SparkNvmeTier *tier)
{
	NvmeTierStagingState *staging_states;
	NvmeTierSlot *slots;
	uint32_t count;
	uint32_t index;
	uint32_t attempt;

	staging_states = (NvmeTierStagingState *)tier->staging_state;
	slots = (NvmeTierSlot *)tier->slots;
	count = tier->configuration.staging_buffer_count;
	for ( index = 0u; index < count; ++index )
	{
		if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			return(index);
	}
	for ( index = 0u; index < count; ++index )
	{
		uint32_t slot_index;
		uint32_t deadline;

		if ( staging_states[index].state != NVME_TIER_STAGING_READY
			|| staging_states[index].holder != NVME_TIER_HOLDER_PREFETCH )
			continue;
		slot_index = staging_states[index].slot;
		if ( slot_index >= tier->slot_count || slots[slot_index].pin_count != 0u )
			continue;
		deadline = staging_states[index].need_by_step;
		tier->statistics.prefetch_preemptions++;
		tier->statistics.prefetch_dropped++;
		NvmeTierStagingRelease(tier,index);
		NvmeTierPrefetchRequeue(tier,slot_index,deadline);
		return(index);
	}

	if ( tier->device.cancel_read != 0 )
	{
		for ( attempt = 0u; attempt < count; ++attempt )
		{
			SparkStatus cancel_status;
			uint32_t best;
			uint32_t best_deadline;
			uint32_t slot_index;
			uint32_t deadline;

			best = SPARK_NVME_TIER_NO_SLOT;
			best_deadline = 0u;
			for ( index = 0u; index < count; ++index )
			{
				if ( staging_states[index].state != NVME_TIER_STAGING_FILLING
					|| staging_states[index].holder != NVME_TIER_HOLDER_PREFETCH )
					continue;
				if ( staging_states[index].slot >= tier->slot_count
					|| slots[staging_states[index].slot].pin_count != 0u )
					continue;
				if ( best == SPARK_NVME_TIER_NO_SLOT
					|| staging_states[index].need_by_step > best_deadline )
				{
					best = index;
					best_deadline = staging_states[index].need_by_step;
				}
			}
			if ( best == SPARK_NVME_TIER_NO_SLOT )
				break;

			slot_index = staging_states[best].slot;
			deadline = staging_states[best].need_by_step;
			cancel_status = tier->device.cancel_read(
				tier->device.context,staging_states[best].ticket);
			tier->statistics.prefetch_preemptions++;
			if ( cancel_status == SPARK_STATUS_OK )
			{
				tier->statistics.prefetch_dropped++;
				NvmeTierStagingRelease(tier,best);
				NvmeTierPrefetchRequeue(tier,slot_index,deadline);
				return(best);
			}
			if ( cancel_status == SPARK_STATUS_BUSY
				|| cancel_status == SPARK_STATUS_PENDING )
			{
				staging_states[best].state =
					NVME_TIER_STAGING_CANCEL_PENDING;
				tier->statistics.cancel_pending_count++;
				continue;
			}

			tier->statistics.io_errors++;
			tier->statistics.prefetch_dropped++;
			NvmeTierStagingRelease(tier,best);
			NvmeTierPrefetchRequeue(tier,slot_index,deadline);
			return(best);
		}
	}

	tier->statistics.demand_stalls++;
	return(SPARK_NVME_TIER_NO_SLOT);
}

static SparkStatus NvmeTierIssueRead(
	SparkNvmeTier *tier,
	uint32_t slot_index,
	uint32_t staging_index,
	uint8_t holder,
	uint32_t need_by_step,
	uint32_t step_now)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierStagingState *staging_states = (NvmeTierStagingState *)tier->staging_state;
	uint64_t ticket = 0u;
	SparkStatus status;
	status = tier->device.submit_read(tier->device.context,
		tier->configuration.base_offset
			+ (uint64_t)slot_index * tier->configuration.block_bytes,
		tier->staging + (uint64_t)staging_index * tier->configuration.block_bytes,
		tier->configuration.block_bytes,&ticket);
	if ( status != SPARK_STATUS_OK )
	{
		// The device refused; leave the slot PRESENT so a later pump retries,
		// and the buffer free so nothing else pays for the refusal.
		tier->statistics.io_errors++;
		NvmeTierStagingRelease(tier,staging_index);
		return(status);
	}
	staging_states[staging_index].ticket = ticket;
	staging_states[staging_index].slot = slot_index;
	staging_states[staging_index].generation = slots[slot_index].generation;
	staging_states[staging_index].need_by_step = need_by_step;
	staging_states[staging_index].issued_step = step_now;
	staging_states[staging_index].state = NVME_TIER_STAGING_FILLING;
	staging_states[staging_index].holder = holder;
	slots[slot_index].state = NVME_TIER_SLOT_FILLING;
	slots[slot_index].staging_index = staging_index;
	slots[slot_index].issued_step = step_now;
	tier->statistics.read_bytes += tier->configuration.block_bytes;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierRequestDemand(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	uint32_t step_now,
	SparkNvmeTierDemandResult *result_out)
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	NvmeTierSlot *slot;
	SparkStatus status;
	if ( tier == 0 || result_out == 0 || content_hash == 0u ||
		NvmeTierDigestIsUsable(content_digest) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(result_out,0,sizeof(*result_out));
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookupVerified(tier,content_hash,content_digest,
		&status);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT &&
		status == SPARK_STATUS_HASH_MISMATCH )
	{
		// A hash collision against a committed record: the caller would be
		// handed another tenant's bytes. MISS would silently recompute from
		// the wrong sequence's tokens; this fails loud instead.
		tier->statistics.digest_mismatches++;
		result_out->state = SPARK_NVME_TIER_DEMAND_MISS;
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
	{
		// The genuine miss, and the only one: the bytes are not on this tier
		// at all, so somebody recomputes. Counted, because a rising line here
		// is the write-back path falling behind, not bad luck.
		tier->statistics.demand_misses++;
		result_out->state = SPARK_NVME_TIER_DEMAND_MISS;
		return(SPARK_STATUS_OK);
	}
	slot = &slots[slot_index];
	slot->last_use = tier->tick++;
	slot->referenced = 1u;
	if ( slot->state == NVME_TIER_SLOT_READY )
	{
		NvmeTierStagingState *held
			= &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
		// Promote to DEMAND-held. Without this, a second demand load could
		// preempt-steal this buffer between the hit and the caller's Consume,
		// handing back a pointer whose bytes are about to be overwritten.
		held->holder = NVME_TIER_HOLDER_DEMAND;
		tier->statistics.demand_hits++;
		result_out->state = SPARK_NVME_TIER_DEMAND_READY;
		result_out->staging_pointer = tier->staging
			+ (uint64_t)slot->staging_index * tier->configuration.block_bytes;
		return(SPARK_STATUS_OK);
	}
	if ( slot->state == NVME_TIER_SLOT_FILLING )
	{
		NvmeTierStagingState *held;

		if ( slot->staging_index >= tier->configuration.staging_buffer_count )
			return(SPARK_STATUS_INTERNAL_ERROR);
		held = &((NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
		if ( held->state == NVME_TIER_STAGING_CANCEL_PENDING )
		{
			slot->need_by_step = step_now;
			tier->statistics.demand_stalls++;
			return(SPARK_STATUS_BUSY);
		}
		held->holder = NVME_TIER_HOLDER_DEMAND;
		tier->statistics.demand_joins++;
		result_out->state = SPARK_NVME_TIER_DEMAND_IN_FLIGHT;
		return(SPARK_STATUS_OK);
	}
	// PRESENT. If the lookahead queued it, the queue loses it: demand is the
	// deadline now, and a queued prefetch that a demand load duplicates is
	// the inversion this whole design exists to prevent.
	if ( slot->queued != 0u )
	{
		NvmeTierPendingQueue *queue = (NvmeTierPendingQueue *)tier->pending;
		int32_t position = NvmeTierPendingFind(queue,slot_index);
		if ( position >= 0 )
			NvmeTierPendingRemoveAt(queue,(uint32_t)position);
		slot->queued = 0u;
	}
	{
		uint32_t staging_index = NvmeTierStagingAcquireForDemand(tier);
		SparkStatus status;
		if ( staging_index == SPARK_NVME_TIER_NO_SLOT )
			return(SPARK_STATUS_BUSY);
		status = NvmeTierIssueRead(tier,slot_index,staging_index,
			NVME_TIER_HOLDER_DEMAND,step_now,step_now);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	tier->statistics.demand_loads++;
	result_out->state = SPARK_NVME_TIER_DEMAND_STARTED;
	return(SPARK_STATUS_OK);
}

// A record whose bytes failed their digest is gone, not served: drop it
// from the index, recycle the slot, and let the caller's recompute path
// rebuild the block. Quarantine keeps a corrupt on-drive record from being
// re-read forever while keeping the failure LOUD (Pump answers
// HASH_MISMATCH for the call that found it).
static void NvmeTierQuarantineSlot(SparkNvmeTier *tier, uint32_t slot_index)
{
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	NvmeTierSlot *slot = &slots[slot_index];
	slot->queued = 0u;
	NvmeTierBucketRemove(tier,slot_index);
	slot->content_hash = 0u;
	memset(slot->digest,0,sizeof(slot->digest));
	slot->state = NVME_TIER_SLOT_EMPTY;
	slot->generation++;
	if ( slot->generation == 0u )
		slot->generation = 1u;
	slot->pin_count = 0u;
	slot->need_by_step = 0xffffffffu;
	slot->staging_index = SPARK_NVME_TIER_NO_SLOT;
	slot->next_free = tier->free_head;
	tier->free_head = slot_index;
	if ( tier->slots_in_use != 0u )
		tier->slots_in_use--;
	tier->statistics.slots_in_use = tier->slots_in_use;
}

// Verify a landed staging buffer against its record's digest. The buffer
// becomes READY only on a match; a mismatch quarantines the record and
// answers HASH_MISMATCH - restored bytes are verified-correct or never
// handed to a caller.
static SparkStatus NvmeTierVerifyLanding(
	SparkNvmeTier *tier,
	uint32_t slot_index,
	uint32_t staging_index)
{
	SparkSha256Context context;
	uint8_t digest[SPARK_NVME_TIER_DIGEST_BYTES];
	NvmeTierSlot *slots = (NvmeTierSlot *)tier->slots;
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,
		tier->staging + (uint64_t)staging_index *
			tier->configuration.block_bytes,
		tier->configuration.block_bytes);
	SparkSha256Finalize(&context,digest);
	tier->statistics.digest_verifications++;
	if ( NvmeTierDigestMatches(slots[slot_index].digest,digest) == 0u )
	{
		tier->statistics.digest_mismatches++;
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierPump(SparkNvmeTier *tier, uint32_t step_now)
{
	NvmeTierSlot *slots;
	NvmeTierStagingState *staging_states;
	NvmeTierPendingQueue *queue;
	uint32_t index,free_count;
	if ( tier == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	staging_states = (NvmeTierStagingState *)tier->staging_state;
	queue = (NvmeTierPendingQueue *)tier->pending;
	// Completions first, issues second: a buffer freed by a landing can carry
	// the next read in the same pump, which is the double-buffer doing its job.
	for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
	{
		NvmeTierStagingState *held;
		SparkStatus status;
		uint32_t slot_index;

		held = &staging_states[index];
		if ( held->state != NVME_TIER_STAGING_FILLING
			&& held->state != NVME_TIER_STAGING_CANCEL_PENDING )
			continue;
		status = tier->device.poll_read(tier->device.context,held->ticket);
		if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
			continue;
		slot_index = held->slot;
		if ( held->state == NVME_TIER_STAGING_CANCEL_PENDING )
		{
			uint32_t deadline;
			uint32_t generation;

			deadline = held->need_by_step;
			generation = held->generation;
			if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND )
				tier->statistics.io_errors++;
			tier->statistics.prefetch_dropped++;
			NvmeTierStagingRelease(tier,index);
			if ( slot_index < tier->slot_count
				&& slots[slot_index].state == NVME_TIER_SLOT_PRESENT
				&& slots[slot_index].generation == generation )
				NvmeTierPrefetchRequeue(tier,slot_index,deadline);
			continue;
		}
		if ( status != SPARK_STATUS_OK )
		{
			tier->statistics.io_errors++;
			NvmeTierStagingRelease(tier,index);
			continue;
		}
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT
			|| slot_index >= tier->slot_count
			|| slots[slot_index].state != NVME_TIER_SLOT_FILLING
			|| slots[slot_index].staging_index != index
			|| slots[slot_index].generation != held->generation )
		{
			tier->statistics.stale_completions++;
			NvmeTierStagingRelease(tier,index);
			continue;
		}
		/* B3: the bytes just landed are checked against the record's
		 * digest BEFORE they become readable. On mismatch the record is
		 * quarantined (the next demand is a MISS -> recompute) and the
		 * failure is loud. */
		if ( NvmeTierVerifyLanding(tier,slot_index,index) !=
			SPARK_STATUS_OK )
		{
			NvmeTierStagingRelease(tier,index);
			NvmeTierQuarantineSlot(tier,slot_index);
			return(SPARK_STATUS_HASH_MISMATCH);
		}
		held->state = NVME_TIER_STAGING_READY;
		slots[slot_index].state = NVME_TIER_SLOT_READY;
		if ( held->holder == NVME_TIER_HOLDER_PREFETCH )
		{
			tier->statistics.prefetch_landings++;
			if ( step_now > held->need_by_step )
				tier->statistics.prefetch_late_landings++;
		}
	}
	free_count = 0u;
	for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
		if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			free_count++;
	// Prefetches issue only into staging ABOVE the demand reserve. The reserve
	// is the mechanism behind "prefetch never starves demand": a lookahead
	// queue deep enough to fill every buffer would turn the next miss into a
	// stall, so the last buffers are simply not prefetch's to take.
	while ( queue->count != 0u
		&& free_count > tier->configuration.demand_reserve_buffers )
	{
		NvmeTierPendingEntry head = queue->entries[0];
		uint32_t slot_index = head.slot;
		uint32_t staging_index = SPARK_NVME_TIER_NO_SLOT;
		NvmeTierPendingRemoveAt(queue,0u);
		if ( slot_index >= tier->slot_count
			|| slots[slot_index].generation != head.generation
			|| slots[slot_index].state != NVME_TIER_SLOT_PRESENT
			|| slots[slot_index].queued == 0u )
			continue;   /* evicted or already claimed while it waited */
		slots[slot_index].queued = 0u;
		for ( index = 0u; index < tier->configuration.staging_buffer_count; ++index )
			if ( staging_states[index].state == NVME_TIER_STAGING_FREE )
			{
				staging_index = index;
				break;
			}
		if ( staging_index == SPARK_NVME_TIER_NO_SLOT )
			break;      /* reserve miscounted: stop rather than take it */
		staging_states[staging_index].state = NVME_TIER_STAGING_FILLING;  /* claim */
		staging_states[staging_index].holder = NVME_TIER_HOLDER_PREFETCH;
		staging_states[staging_index].slot = SPARK_NVME_TIER_NO_SLOT;
		free_count--;
		if ( NvmeTierIssueRead(tier,slot_index,staging_index,
				NVME_TIER_HOLDER_PREFETCH,head.need_by_step,step_now) != SPARK_STATUS_OK )
			continue;
		tier->statistics.prefetch_issues++;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierConsume(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES])
{
	NvmeTierSlot *slots;
	uint32_t slot_index;
	SparkStatus status;
	if ( tier == 0 || content_hash == 0u ||
		NvmeTierDigestIsUsable(content_digest) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookupVerified(tier,content_hash,content_digest,
		&status);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
	{
		if ( status == SPARK_STATUS_HASH_MISMATCH )
			tier->statistics.digest_mismatches++;
		return(status);
	}
	if ( slots[slot_index].state != NVME_TIER_SLOT_READY )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots[slot_index].last_use = tier->tick++;
	slots[slot_index].referenced = 1u;
	// The record stays PRESENT: consumption copies the bytes upstairs, it does
	// not move them. Eviction, and only eviction, reclaims the drive.
	NvmeTierStagingRelease(tier,slots[slot_index].staging_index);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierPin(
	SparkNvmeTier *tier,
	uint64_t content_hash,
	const uint8_t content_digest[SPARK_NVME_TIER_DIGEST_BYTES],
	int32_t pin)
{
	uint32_t slot_index;
	NvmeTierSlot *slots;
	SparkStatus status;
	if ( tier == 0 || content_hash == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slots = (NvmeTierSlot *)tier->slots;
	slot_index = NvmeTierLookupVerified(tier,content_hash,content_digest,
		&status);
	if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
	{
		if ( status == SPARK_STATUS_HASH_MISMATCH )
			tier->statistics.digest_mismatches++;
		return(status);
	}
	if ( pin )
		slots[slot_index].pin_count++;
	else if ( slots[slot_index].pin_count != 0u )
		slots[slot_index].pin_count--;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkNvmeTierWillBeResidentBy(
	const SparkNvmeTier *tier,
	const SparkNvmeTierNeed *needs,
	uint32_t need_count,
	uint32_t step_now,
	uint32_t step_deadline,
	SparkNvmeTierResidencyAssessment *assessment_out)
{
	const NvmeTierSlot *slots;
	uint32_t index,confident;
	if ( tier == 0 || ( needs == 0 && need_count != 0u )
		|| assessment_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(assessment_out,0,sizeof(*assessment_out));
	slots = (const NvmeTierSlot *)tier->slots;
	for ( index = 0u; index < need_count; ++index )
	{
		SparkStatus lookup_status;
		uint32_t slot_index = NvmeTierLookupVerified(tier,
			needs[index].content_hash,needs[index].content_digest,
			&lookup_status);
		const NvmeTierSlot *slot;
		uint64_t eta_steps;
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT &&
			lookup_status == SPARK_STATUS_HASH_MISMATCH )
			return(SPARK_STATUS_HASH_MISMATCH);
		if ( slot_index == SPARK_NVME_TIER_NO_SLOT )
		{
			assessment_out->absent_count++;
			continue;
		}
		slot = &slots[slot_index];
		if ( slot->state == NVME_TIER_SLOT_READY )
		{
			assessment_out->ready_count++;
			continue;
		}
		if ( slot->state == NVME_TIER_SLOT_FILLING )
		{
			const NvmeTierStagingState *held;

			if ( slot->staging_index >= tier->configuration.staging_buffer_count )
			{
				assessment_out->late_count++;
				continue;
			}
			held = &((const NvmeTierStagingState *)tier->staging_state)[slot->staging_index];
			if ( held->state == NVME_TIER_STAGING_CANCEL_PENDING )
			{
				assessment_out->late_count++;
				continue;
			}
			eta_steps = tier->transfer_steps;
			if ( (uint64_t)step_now + eta_steps <= step_deadline )
				assessment_out->inflight_confident_count++;
			else
				assessment_out->late_count++;
			continue;
		}
		// PRESENT: the read still has to be issued, so the queue in front of
		// it is part of the ETA. One staging buffer's worth of queue drains
		// per landing, so queue depth approximates the wait in steps.
		{
			const NvmeTierPendingQueue *queue = (const NvmeTierPendingQueue *)tier->pending;
			eta_steps = (uint64_t)tier->transfer_steps * ( 1u + queue->count );
		}
		if ( (uint64_t)step_now + eta_steps <= step_deadline )
			assessment_out->planned_confident_count++;
		else
			assessment_out->late_count++;
	}
	confident = assessment_out->ready_count
		+ assessment_out->inflight_confident_count
		+ assessment_out->planned_confident_count;
	if ( assessment_out->absent_count == 0u && assessment_out->late_count == 0u )
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_ALL;
	else if ( confident == 0u )
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_NONE;
	else
		assessment_out->confidence = SPARK_NVME_TIER_CONFIDENCE_PARTIAL;
	return(SPARK_STATUS_OK);
}

void SparkNvmeTierGetStatistics(
	const SparkNvmeTier *tier,
	SparkNvmeTierStatistics *statistics_out)
{
	if ( tier == 0 || statistics_out == 0 )
		return;
	*statistics_out = tier->statistics;
	statistics_out->slot_count = tier->slot_count;
	statistics_out->slots_in_use = tier->slots_in_use;
}
