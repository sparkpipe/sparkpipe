// The JIT-KV pager (docs/JIT_KV_DESIGN.md adapter layer): park, restore,
// and admission backpressure joining the resident arena to the backing tier.
// The contract lives in sparkpipe/spark_kv_pager.h; this file is the policy -
// LRU page-out through the arena's evict function, digest-verified page-in,
// and the queue-not-wedge admission rule of docs/JIT_KV_RESPONSE.md C1.

#include "sparkpipe/spark_kv_pager.h"

#include <string.h>

static uint32_t SparkKvPagerDigestIsUsable(
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES])
{
	uint32_t index,usable = 0u;
	if ( content_digest == 0 )
		return(0u);
	for ( index = 0u; index < SPARK_KV_PAGER_DIGEST_BYTES; ++index )
		usable |= content_digest[index];
	return(usable != 0u);
}

/* The tier's bucket key, folded from the payload digest (see header). */
static uint64_t SparkKvPagerHashFromDigest(
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES])
{
	uint64_t hash = 0u;
	uint32_t index;
	for ( index = 0u; index < (uint32_t)sizeof(hash); ++index )
		hash |= (uint64_t)content_digest[index] << (8u * index);
	return(hash | 1u);
}

static void SparkKvPagerDigestOf(
    const void *payload,
    uint64_t payload_bytes,
    uint8_t digest_out[SPARK_KV_PAGER_DIGEST_BYTES])
{
	SparkSha256Context context;
	SparkSha256Initialize(&context);
	SparkSha256Update(&context,payload,(size_t)payload_bytes);
	SparkSha256Finalize(&context,digest_out);
}

static uint32_t SparkKvPagerIsValid(const SparkKvPager *pager)
{
	return pager != 0 &&
		pager->abi_version == SPARK_KV_PAGER_ABI_VERSION &&
		pager->descriptor_bytes == SPARK_KV_PAGER_DESCRIPTOR_BYTES &&
		pager->configuration.arena != 0 &&
		pager->configuration.tier != 0;
}

static void SparkKvPagerRecordPageOut(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    uint32_t already_present)
{
	SparkKvPagerStatistics *statistics = &pager->statistics;
	uint32_t position;

	statistics->page_out_count += 1u;
	statistics->page_out_bytes += pager->block_bytes;
	statistics->page_out_deduplicated += already_present;
	position = (uint32_t)(statistics->page_out_history_count %
		SPARK_KV_PAGER_PAGE_OUT_HISTORY_CAPACITY);
	statistics->page_out_history[position] = logical_block_index;
	statistics->page_out_history_count += 1u;
}

SparkStatus SparkKvPagerInitialize(
    SparkKvPager *pager,
    const SparkKvPagerConfiguration *configuration)
{
	uint64_t block_bytes,resident_bytes;

	if ( pager == 0 || configuration == 0 ||
		configuration->abi_version != SPARK_KV_PAGER_ABI_VERSION ||
		configuration->descriptor_bytes !=
			SPARK_KV_PAGER_CONFIGURATION_DESCRIPTOR_BYTES ||
		configuration->reserved0 != 0u || configuration->reserved1 != 0u ||
		configuration->reserved2 != 0u ||
		configuration->arena == 0 || configuration->tier == 0 ||
		configuration->arena->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
		configuration->arena->descriptor_bytes !=
			SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES ||
		configuration->staging == 0 ||
		configuration->module_context == 0 ||
		configuration->module_save == 0 ||
		configuration->module_restore == 0 ||
		configuration->backing_write == 0 )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	block_bytes = configuration->arena->key_block_stride_bytes +
		configuration->arena->value_block_stride_bytes;
	if ( block_bytes == 0u || block_bytes > UINT32_MAX ||
		block_bytes != configuration->tier->configuration.block_bytes ||
		configuration->staging_bytes <
			block_bytes * SPARK_KV_PAGER_STAGING_BLOCK_COUNT )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* The device law: the configured budget is nonzero, at most the law, and
	 * large enough to hold the arena's ENTIRE resident capacity - a pager
	 * whose own arena can break its budget is a wedge on a timer. */
	resident_bytes = (uint64_t)configuration->arena->resident_block_capacity *
		block_bytes;
	if ( configuration->device_budget_bytes == 0u ||
		configuration->device_budget_bytes > SPARK_KV_PAGER_DEVICE_LAW_BYTES ||
		resident_bytes > configuration->device_budget_bytes )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	/* One pager owns the arena's eviction path; a second install would
	 * silently detach the first owner's write-back. */
	if ( configuration->arena->evict_function != 0 &&
		configuration->arena->evict_function != SparkKvPagerEvictWriteback )
	{
		return(SPARK_STATUS_BUSY);
	}
	memset(pager,0,sizeof(*pager));
	pager->abi_version = SPARK_KV_PAGER_ABI_VERSION;
	pager->descriptor_bytes = SPARK_KV_PAGER_DESCRIPTOR_BYTES;
	pager->block_bytes = (uint32_t)block_bytes;
	pager->park_budget_blocks = configuration->park_budget_blocks;
	pager->configuration = *configuration;
	pager->landing_staging =
		(uint8_t *)configuration->staging + block_bytes;
	configuration->arena->evict_function = SparkKvPagerEvictWriteback;
	configuration->arena->evict_context = pager;
	return(SPARK_STATUS_OK);
}

uint64_t SparkKvPagerBlockBytes(const SparkKvPager *pager)
{
	return(SparkKvPagerIsValid(pager) != 0u ? (uint64_t)pager->block_bytes : 0u);
}

SparkStatus SparkKvPagerAdmit(
    SparkKvPager *pager,
    const SparkKvPagerAdmission *admission,
    SparkKvPagerAdmissionDecision *decision_out)
{
	SparkKvPagerAdmissionDecision decision;
	SparkKvCacheArena *arena;
	SparkNvmeTier *tier;
	uint32_t unassigned,occupied,free_capacity,deficit;
	uint32_t block_index,parkable,park_headroom;
	SparkStatus status;

	if ( SparkKvPagerIsValid(pager) == 0u || admission == 0 || decision_out == 0 ||
		admission->abi_version != SPARK_KV_PAGER_ADMISSION_ABI_VERSION ||
		admission->descriptor_bytes !=
			SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES ||
		admission->reserved0 != 0u )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memset(decision_out,0,sizeof(*decision_out));
	decision.abi_version = SPARK_KV_PAGER_ADMISSION_ABI_VERSION;
	decision.descriptor_bytes = SPARK_KV_PAGER_ADMISSION_DECISION_DESCRIPTOR_BYTES;
	decision.block_demand = admission->block_demand;
	if ( admission->block_demand == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	arena = pager->configuration.arena;
	tier = pager->configuration.tier;
	pager->statistics.admission_requests += 1u;

	/* The exact overflow arithmetic of the arena's own residency grant
	 * (SparkKvCacheArenaMakeRoomForResidentBlocks): resident + reserved +
	 * unassigned is the occupied device budget. */
	unassigned = atomic_load(&arena->unassigned_resident_block_count);
	occupied = (uint32_t)(arena->resident_block_count +
		arena->reserved_block_count) + unassigned;
	free_capacity = occupied >= arena->resident_block_capacity ? 0u :
		arena->resident_block_capacity - occupied;
	deficit = admission->block_demand > free_capacity ?
		admission->block_demand - free_capacity : 0u;
	if ( deficit != 0u )
	{
		/* C1: name the refusal BEFORE touching anything. Parkable is the
		 * exact victim pool of the arena's own selector: residency-evictable
		 * blocks (residency references and reservations are protected - that
		 * is the "never evict an active lane for a younger requester" rule
		 * in its enforceable form). Backing headroom is the park budget:
		 * when it cannot hold the deficit, the request QUEUES - parking
		 * into a full horizon would degrade blocks the tier could have
		 * kept, which is the thrash the design forbids. */
		parkable = 0u;
		for ( block_index = 0u; block_index < arena->logical_block_count;
			++block_index )
		{
			if ( (arena->blocks[block_index].flags &
					SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u ||
				(arena->blocks[block_index].flags &
					SPARK_KV_CACHE_BLOCK_FLAG_RESIDENCY_RESERVED) != 0u ||
				arena->blocks[block_index].residency_reference_count != 0u )
			{
				continue;
			}
			parkable += 1u;
		}
		if ( parkable < deficit )
		{
			decision.outcome = SPARK_KV_PAGER_QUEUED;
			pager->statistics.admission_queued_device += 1u;
			*decision_out = decision;
			return(SPARK_STATUS_OK);
		}
		park_headroom = pager->park_budget_blocks > tier->slots_in_use ?
			pager->park_budget_blocks - tier->slots_in_use : 0u;
		if ( park_headroom < deficit )
		{
			decision.outcome = SPARK_KV_PAGER_QUEUED;
			pager->statistics.admission_queued_backing += 1u;
			*decision_out = decision;
			return(SPARK_STATUS_OK);
		}
	}
	/* The grant itself, which under the pager trims to fit: every block the
	 * trim evicts is a page-out through SparkKvPagerEvictWriteback. */
	status = SparkKvCacheArenaReserveUnassignedResidentBlocks(arena,
		admission->block_demand);
	if ( status == SPARK_STATUS_OK )
	{
		decision.outcome = SPARK_KV_PAGER_ADMITTED;
		decision.park_evictions = deficit;
		decision.reservation_held = admission->block_demand;
		pager->statistics.admission_accepted += 1u;
		pager->statistics.park_evictions = deficit;
	}
	else if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
	{
		/* The trim stalled on protected residents: queue, never evict them. */
		decision.outcome = SPARK_KV_PAGER_QUEUED;
		pager->statistics.admission_queued_device += 1u;
	}
	else if ( status == SPARK_STATUS_BUSY )
	{
		/* A write-back in flight at the tier (or a pinned-full horizon):
		 * backpressure, not a wedge and never a silent drop. */
		decision.outcome = SPARK_KV_PAGER_QUEUED;
		pager->statistics.admission_queued_backing += 1u;
	}
	else
	{
		return(status);
	}
	*decision_out = decision;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPagerCommitAdmission(
    SparkKvPager *pager,
    uint32_t block_count)
{
	if ( SparkKvPagerIsValid(pager) == 0u || block_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvCacheArenaConsumeUnassignedResidentBlocks(
		pager->configuration.arena,block_count));
}

SparkStatus SparkKvPagerReleaseAdmission(
    SparkKvPager *pager,
    uint32_t block_count)
{
	if ( SparkKvPagerIsValid(pager) == 0u || block_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkKvCacheArenaReleaseUnassignedResidentBlocks(
		pager->configuration.arena,block_count));
}

SparkStatus SparkKvPagerEvictWriteback(
    void *context,
    uint32_t logical_block_index,
    uint32_t resident_slot_index,
    uint64_t generation,
    uintptr_t key_device_address,
    uint64_t key_bytes,
    uintptr_t value_device_address,
    uint64_t value_bytes)
{
	SparkKvPager *pager = (SparkKvPager *)context;
	SparkKvPagerBlockView view;
	SparkNvmeTierWriteReservation reservation;
	uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
	SparkStatus status;

	if ( SparkKvPagerIsValid(pager) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	(void)resident_slot_index;
	(void)generation;
	if ( key_bytes + value_bytes != (uint64_t)pager->block_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	view.abi_version = SPARK_KV_PAGER_ABI_VERSION;
	view.descriptor_bytes = SPARK_KV_PAGER_BLOCK_VIEW_DESCRIPTOR_BYTES;
	view.lane_index = 0u;
	view.first_block = logical_block_index;
	view.block_count = 1u;
	view.reserved0 = 0u;
	view.key_bytes = key_bytes;
	view.value_bytes = value_bytes;
	view.key_device_address = key_device_address;
	view.value_device_address = value_device_address;
	view.host_staging = pager->configuration.staging;
	/* 1. the module seam: the block's planes land in the pager staging. */
	status = pager->configuration.module_save(pager->configuration.module_context,
		&view);
	if ( status != SPARK_STATUS_OK )
		return(status);
	/* 2. the digest is the identity; the folded key is only a bucket. */
	SparkKvPagerDigestOf(pager->configuration.staging,pager->block_bytes,digest);
	status = SparkNvmeTierReserveWrite(pager->configuration.tier,
		SparkKvPagerHashFromDigest(digest),digest,&reservation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( reservation.already_present == 0u )
	{
		/* 3. the backing write leg at the offset the tier reserved. An
		 * IO-class failure aborts the reservation and propagates: the
		 * arena's B1 degrade owns it (drop + recompute, never a wedge). */
		status = pager->configuration.backing_write(
			pager->configuration.backing_context,reservation.device_offset,
			pager->configuration.staging,(uint64_t)pager->block_bytes);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkNvmeTierAbortWrite(pager->configuration.tier,&reservation);
			return(status);
		}
	}
	status = SparkNvmeTierCommitWrite(pager->configuration.tier,&reservation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkKvPagerRecordPageOut(pager,logical_block_index,
		reservation.already_present);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPagerRestoreBlock(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES])
{
	SparkKvCacheArena *arena;
	SparkKvCacheBlockView block_view;
	SparkKvPagerBlockView view;
	SparkNvmeTierDemandResult demand;
	uint8_t landed_digest[SPARK_KV_PAGER_DIGEST_BYTES];
	SparkStatus status;
	uint32_t attempt;

	if ( SparkKvPagerIsValid(pager) == 0u ||
		SparkKvPagerDigestIsUsable(content_digest) == 0u )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	arena = pager->configuration.arena;
	if ( logical_block_index >= arena->logical_block_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	{
		SparkKvCacheBlock *block = &arena->blocks[logical_block_index];
		if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) == 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) != 0u )
			return(SPARK_STATUS_OK);
		if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) == 0u )
			return(SPARK_STATUS_NOT_FOUND);
	}
	/* Bring the record's bytes up digest-verified. STARTED/IN_FLIGHT pump;
	 * MISS hands the recompute path back; anything else stays loud. */
	for ( attempt = 0u; attempt < SPARK_KV_PAGER_RESTORE_POLL_LIMIT; ++attempt )
	{
		status = SparkNvmeTierRequestDemand(pager->configuration.tier,
			SparkKvPagerHashFromDigest(content_digest),content_digest,attempt,
			&demand);
		if ( status == SPARK_STATUS_OK &&
			demand.state == SPARK_NVME_TIER_DEMAND_MISS )
		{
			pager->statistics.page_in_misses += 1u;
			return(SPARK_STATUS_NOT_FOUND);
		}
		if ( status == SPARK_STATUS_OK &&
			demand.state == SPARK_NVME_TIER_DEMAND_READY )
		{
			break;
		}
		if ( status != SPARK_STATUS_OK && status != SPARK_STATUS_BUSY )
			return(status);
		status = SparkNvmeTierPump(pager->configuration.tier,attempt);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	if ( attempt == SPARK_KV_PAGER_RESTORE_POLL_LIMIT )
		return(SPARK_STATUS_IO_ERROR);
	/* The tier verified the landing; the pager re-verifies the buffer it is
	 * about to copy from, because the staging leg is the pager's own
	 * responsibility once the pointer is handed over. */
	SparkKvPagerDigestOf(demand.staging_pointer,(uint64_t)pager->block_bytes,
		landed_digest);
	if ( memcmp(landed_digest,content_digest,SPARK_KV_PAGER_DIGEST_BYTES) != 0 )
	{
		(void)SparkNvmeTierConsume(pager->configuration.tier,
			SparkKvPagerHashFromDigest(content_digest),content_digest);
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	/* Land the verified bytes into the pager's OWN staging plane and release
	 * the tier's buffer BEFORE make-room. The residency grant below may page
	 * the LRU victim out, and those write-backs need the tier free to evict
	 * - including the record these bytes just came from. Holding the tier's
	 * demand staging across make-room would protect that record from its own
	 * clock and turn a busy tier into a deadlock. */
	memcpy(pager->landing_staging,demand.staging_pointer,
		pager->block_bytes);
	(void)SparkNvmeTierConsume(pager->configuration.tier,
		SparkKvPagerHashFromDigest(content_digest),content_digest);
	/* Re-attach residency (it fixes the device addresses, and its make-room
	 * may page the LRU victim out); on saturation answer BUSY - the caller
	 * retries, nothing was dropped. */
	status = SparkKvCacheArenaMarkParkedBlockResident(arena,logical_block_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkKvCacheArenaResolveBlock(arena,logical_block_index,&block_view);
	if ( status == SPARK_STATUS_OK )
	{
		view.abi_version = SPARK_KV_PAGER_ABI_VERSION;
		view.descriptor_bytes = SPARK_KV_PAGER_BLOCK_VIEW_DESCRIPTOR_BYTES;
		view.lane_index = 0u;
		view.first_block = logical_block_index;
		view.block_count = 1u;
		view.reserved0 = 0u;
		view.key_bytes = (uint64_t)arena->key_block_stride_bytes;
		view.value_bytes = (uint64_t)arena->value_block_stride_bytes;
		view.key_device_address = block_view.key_device_address;
		view.value_device_address = block_view.value_device_address;
		view.host_staging = pager->landing_staging;
		/* the module seam, reversed: landing copy into the device planes. A
		 * failure here leaves the block holding unverified bytes, so it is
		 * parked straight back out - never resident-and-trusted. */
		status = pager->configuration.module_restore(
			pager->configuration.module_context,&view);
		if ( status != SPARK_STATUS_OK )
		{
			SparkStatus rollback = SparkKvCacheArenaMarkBlockNonResident(
				arena,logical_block_index);
			if ( rollback != SPARK_STATUS_OK )
				status = rollback;
		}
		else
		{
			pager->statistics.page_in_count += 1u;
			pager->statistics.page_in_bytes += pager->block_bytes;
		}
	}
	return(status);
}

SparkStatus SparkKvPagerAssertDeviceBudget(const SparkKvPager *pager)
{
	const SparkKvCacheArena *arena;
	uint32_t unassigned;
	uint64_t resident_bytes;

	if ( SparkKvPagerIsValid(pager) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	arena = pager->configuration.arena;
	unassigned = atomic_load(&arena->unassigned_resident_block_count);
	if ( arena->resident_block_count > arena->resident_block_capacity ||
		arena->resident_block_count + arena->reserved_block_count + unassigned >
			arena->resident_block_capacity ||
		pager->configuration.tier->slots_in_use >
			pager->configuration.tier->slot_count )
	{
		return(SPARK_STATUS_INTERNAL_ERROR);
	}
	resident_bytes = arena->resident_block_count * (uint64_t)pager->block_bytes;
	if ( resident_bytes > pager->configuration.device_budget_bytes )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

void SparkKvPagerGetStatistics(
    const SparkKvPager *pager,
    SparkKvPagerStatistics *statistics_out)
{
	if ( pager == 0 || statistics_out == 0 )
		return;
	*statistics_out = pager->statistics;
}
