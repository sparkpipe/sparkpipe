
#include "sparkpipe/spark_kv_pager.h"

#include <pthread.h>
#include <string.h>
#include <time.h>

_Static_assert(sizeof(pthread_t) <= SPARK_KV_PAGER_PARK_WORKER_HANDLE_BYTES,
	"the pager's worker-handle storage must hold a pthread_t");

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


static uint64_t SparkKvPagerClockNow(const SparkKvPager *pager)
{
	return(pager->configuration.clock_function != 0 ?
		pager->configuration.clock_function(
			pager->configuration.clock_context) : 0u);
}

static void SparkKvPagerFoldBandwidthSample(
	SparkKvPager *pager,uint64_t payload_bytes,uint64_t elapsed_microseconds)
{
	uint64_t average,sample;

	if ( pager->configuration.clock_function == 0 ||
		payload_bytes == 0u || elapsed_microseconds == 0u )
		return;
	sample = payload_bytes * 1000000ull / elapsed_microseconds;
	average = pager->statistics.measured_bytes_per_second;
	if ( average == 0u )
		pager->statistics.measured_bytes_per_second = sample;
	else
		pager->statistics.measured_bytes_per_second =
			average - average / 2u + sample / 2u;
	pager->statistics.measured_bandwidth_samples += 1u;
}


static uint32_t SparkKvPagerParkQueuePush(SparkKvPager *pager,
	const SparkKvPagerParkEntry *entry)
{
	uint32_t head = atomic_load_explicit(&pager->park_head,
		memory_order_relaxed);
	uint32_t tail = atomic_load_explicit(&pager->park_tail,
		memory_order_acquire);

	if ( head - tail >= pager->park_queue_blocks )
		return(0u);
	pager->park_queue[head % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY] = *entry;
	atomic_store_explicit(&pager->park_head,head + 1u,memory_order_release);
	return(1u);
}

static uint32_t SparkKvPagerParkQueuePop(SparkKvPager *pager,
	SparkKvPagerParkEntry *entry_out)
{
	uint32_t tail = atomic_load_explicit(&pager->park_tail,
		memory_order_relaxed);
	uint32_t head = atomic_load_explicit(&pager->park_head,
		memory_order_acquire);

	if ( tail == head )
		return(0u);
	*entry_out = pager->park_queue[tail % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
	atomic_store_explicit(&pager->park_tail,tail + 1u,memory_order_release);
	return(1u);
}

static uint32_t SparkKvPagerParkCompletionPush(SparkKvPager *pager,
	const SparkKvPagerParkCompletion *completion)
{
	uint32_t head = atomic_load_explicit(&pager->park_completion_head,
		memory_order_relaxed);
	uint32_t tail = atomic_load_explicit(&pager->park_completion_tail,
		memory_order_acquire);

	if ( head - tail >= SPARK_KV_PAGER_PARK_QUEUE_CAPACITY )
		return(0u);
	pager->park_completions[head % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY] =
		*completion;
	atomic_store_explicit(&pager->park_completion_head,head + 1u,
		memory_order_release);
	return(1u);
}

static uint32_t SparkKvPagerParkCompletionPop(SparkKvPager *pager,
	SparkKvPagerParkCompletion *completion_out)
{
	uint32_t tail = atomic_load_explicit(&pager->park_completion_tail,
		memory_order_relaxed);
	uint32_t head = atomic_load_explicit(&pager->park_completion_head,
		memory_order_acquire);

	if ( tail == head )
		return(0u);
	*completion_out =
		pager->park_completions[tail % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
	atomic_store_explicit(&pager->park_completion_tail,tail + 1u,
		memory_order_release);
	return(1u);
}

static uint32_t SparkKvPagerParkIsInFlight(const SparkKvPager *pager,
	uint32_t logical_block_index,
	const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES])
{
	uint32_t tail = atomic_load_explicit(&pager->park_tail,
		memory_order_acquire);
	uint32_t head = atomic_load_explicit(&pager->park_head,
		memory_order_relaxed);

	for ( ; tail != head; ++tail )
	{
		const SparkKvPagerParkEntry *entry =
			&pager->park_queue[tail % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
		if ( entry->logical_block_index == logical_block_index &&
			memcmp(entry->reservation.content_digest,content_digest,
				SPARK_KV_PAGER_DIGEST_BYTES) == 0 )
		{
			return(1u);
		}
	}
	if ( atomic_load_explicit(&pager->park_write_inflight_valid,
		memory_order_acquire) != 0u &&
		pager->park_write_inflight.logical_block_index ==
			logical_block_index &&
		memcmp(pager->park_write_inflight.reservation.content_digest,
			content_digest,SPARK_KV_PAGER_DIGEST_BYTES) == 0 )
	{
		return(1u);
	}
	return(0u);
}

static void SparkKvPagerParkExecuteWrite(SparkKvPager *pager,
	const SparkKvPagerParkEntry *entry,
	SparkKvPagerParkCompletion *completion_out)
{
	uint64_t write_started = SparkKvPagerClockNow(pager);
	SparkStatus status;

	completion_out->logical_block_index = entry->logical_block_index;
	completion_out->reservation = entry->reservation;
	status = pager->configuration.backing_write(
		pager->configuration.backing_context,
		entry->reservation.device_offset,entry->staging,
		entry->payload_bytes);
	completion_out->status = (uint32_t)status;
	completion_out->write_elapsed_microseconds =
		SparkKvPagerClockNow(pager) - write_started;
}

static void SparkKvPagerPublishCompletion(SparkKvPager *pager,
	const SparkKvPagerParkCompletion *completion)
{
	SparkKvCacheBlock *block =
		&pager->configuration.arena->blocks[completion->logical_block_index];
	SparkStatus status = (SparkStatus)completion->status;

	if ( status == SPARK_STATUS_OK )
	{
		status = SparkNvmeTierCommitWrite(pager->configuration.tier,
			&completion->reservation);
		if ( status == SPARK_STATUS_OK )
		{
			block->flags |= SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID;
			block->flags &=
				(uint32_t)~SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
			SparkKvPagerRecordPageOut(pager,
				completion->logical_block_index,
				completion->reservation.already_present);
		}
	}
	if ( status != SPARK_STATUS_OK )
	{
		(void)SparkNvmeTierAbortWrite(pager->configuration.tier,
			&completion->reservation);
		block->flags &= (uint32_t)~SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID;
		block->flags &= (uint32_t)~SPARK_KV_CACHE_BLOCK_FLAG_DIRTY;
		if ( pager->configuration.arena->write_back_degraded_block_count !=
			UINT32_MAX )
		{
			pager->configuration.arena->write_back_degraded_block_count +=
				1u;
		}
		pager->statistics.park_write_failures += 1u;
	}
	if ( completion->write_elapsed_microseconds != 0u )
	{
		SparkKvPagerFoldBandwidthSample(pager,
			(uint64_t)pager->block_bytes,
			completion->write_elapsed_microseconds);
	}
	pager->statistics.park_completions_published += 1u;
}

static void SparkKvPagerParkWorkerSleep(void)
{
	struct timespec quantum;

	quantum.tv_sec = (time_t)0;
	quantum.tv_nsec =
		(long)SPARK_KV_PAGER_PARK_POLL_QUANTUM_MICROSECONDS * 1000l;
	nanosleep(&quantum,0);
}

static void *SparkKvPagerParkWorkerMain(void *argument)
{
	SparkKvPager *pager = (SparkKvPager *)argument;
	SparkKvPagerParkEntry entry;
	SparkKvPagerParkCompletion completion;
	uint32_t tail;

	while ( atomic_load_explicit(&pager->park_worker_stop,
		memory_order_seq_cst) == 0 )
	{
		tail = atomic_load_explicit(&pager->park_tail,
			memory_order_relaxed);
		if ( tail == atomic_load_explicit(&pager->park_head,
			memory_order_acquire) )
		{
			SparkKvPagerParkWorkerSleep();
			continue;
		}
		entry = pager->park_queue[tail % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY];
		pager->park_write_inflight = entry;
		atomic_store_explicit(&pager->park_write_inflight_valid,1u,
			memory_order_release);
		atomic_store_explicit(&pager->park_tail,tail + 1u,
			memory_order_release);
		SparkKvPagerParkExecuteWrite(pager,&entry,&completion);
		while ( SparkKvPagerParkCompletionPush(pager,&completion) == 0u )
			SparkKvPagerParkWorkerSleep();
		atomic_store_explicit(&pager->park_write_inflight_valid,0u,
			memory_order_release);
	}
	return(0);
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
		configuration->arena == 0 || configuration->tier == 0 ||
		configuration->arena->abi_version != SPARK_KV_CACHE_ABI_VERSION ||
		configuration->arena->descriptor_bytes !=
			SPARK_KV_CACHE_ARENA_DESCRIPTOR_BYTES ||
		configuration->staging == 0 ||
		configuration->module_context == 0 ||
		configuration->module_save == 0 ||
		configuration->module_restore == 0 ||
		configuration->backing_write == 0 ||
		configuration->park_policy >
			SPARK_KV_PAGER_PARK_POLICY_REUSE_VALUE)
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
	resident_bytes = (uint64_t)configuration->arena->resident_block_capacity *
		block_bytes;
	if ( configuration->device_budget_bytes == 0u ||
		configuration->device_budget_bytes > SPARK_KV_PAGER_DEVICE_LAW_BYTES ||
		resident_bytes > configuration->device_budget_bytes )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	if ( configuration->arena->evict_function != 0 &&
		configuration->arena->evict_function != SparkKvPagerEvictWriteback )
	{
		return(SPARK_STATUS_BUSY);
	}
	if ( configuration->park_queue_blocks >
		SPARK_KV_PAGER_PARK_QUEUE_CAPACITY ||
		(configuration->park_queue_blocks != 0u &&
			(configuration->park_staging == 0 ||
				configuration->park_staging_bytes <
					(uint64_t)configuration->park_queue_blocks *
						block_bytes)) )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memset(pager,0,sizeof(*pager));
	pager->abi_version = SPARK_KV_PAGER_ABI_VERSION;
	pager->descriptor_bytes = SPARK_KV_PAGER_DESCRIPTOR_BYTES;
	pager->block_bytes = (uint32_t)block_bytes;
	pager->park_budget_blocks = configuration->park_budget_blocks;
	pager->configuration = *configuration;
	pager->landing_staging =
		(uint8_t *)configuration->staging + block_bytes;
	pager->park_queue_blocks = configuration->park_queue_blocks;
	atomic_init(&pager->park_head,0u);
	atomic_init(&pager->park_tail,0u);
	atomic_init(&pager->park_completion_head,0u);
	atomic_init(&pager->park_completion_tail,0u);
	atomic_init(&pager->park_worker_stop,0u);
	atomic_init(&pager->park_write_inflight_valid,0u);
	configuration->arena->eviction_policy = configuration->park_policy;
	configuration->arena->evict_function = SparkKvPagerEvictWriteback;
	configuration->arena->evict_context = pager;
	if ( pager->park_queue_blocks != 0u )
	{
		pthread_t worker;

		if ( pthread_create(&worker,0,SparkKvPagerParkWorkerMain,pager) != 0 )
		{
			configuration->arena->eviction_policy =
				SPARK_KV_PAGER_PARK_POLICY_LRU;
			configuration->arena->evict_function = 0;
			configuration->arena->evict_context = 0;
			return(SPARK_STATUS_IO_ERROR);
		}
		memset(pager->park_worker_handle,0,sizeof(pager->park_worker_handle));
		memcpy(pager->park_worker_handle,&worker,sizeof(worker));
		pager->park_worker_active = 1u;
	}
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
	uint32_t block_index,parkable;
	uint64_t queued_restore_bytes;
	SparkStatus status;

	if ( SparkKvPagerIsValid(pager) == 0u || admission == 0 || decision_out == 0 ||
		admission->abi_version != SPARK_KV_PAGER_ADMISSION_ABI_VERSION ||
		admission->descriptor_bytes !=
			SPARK_KV_PAGER_ADMISSION_DESCRIPTOR_BYTES )
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
	(void)SparkKvPagerPollParkCompletions(pager);
	pager->statistics.admission_requests += 1u;

	unassigned = atomic_load(&arena->unassigned_resident_block_count);
	occupied = (uint32_t)(arena->resident_block_count +
		arena->reserved_block_count) + unassigned;
	free_capacity = occupied >= arena->resident_block_capacity ? 0u :
		arena->resident_block_capacity - occupied;
	deficit = admission->block_demand > free_capacity ?
		admission->block_demand - free_capacity : 0u;
	parkable = 0u;
	queued_restore_bytes = 0u;
	if ( deficit != 0u || admission->restore_slack_microseconds != 0u )
	{
		for ( block_index = 0u; block_index < arena->logical_block_count;
			++block_index )
		{
			const SparkKvCacheBlock *block = &arena->blocks[block_index];
			parkable += SparkKvCacheArenaBlockIsParkable(arena,block_index);
			if ( (block->flags & SPARK_KV_CACHE_BLOCK_FLAG_ALLOCATED) != 0u &&
				(block->flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u &&
				(block->flags & SPARK_KV_CACHE_BLOCK_FLAG_BACKING_VALID) != 0u )
			{
				queued_restore_bytes += (uint64_t)pager->block_bytes;
			}
		}
	}
	if ( deficit != 0u )
	{
		if ( parkable < deficit )
		{
			decision.outcome = SPARK_KV_PAGER_QUEUED;
			pager->statistics.admission_queued_device += 1u;
			*decision_out = decision;
			return(SPARK_STATUS_OK);
		}
		{
			uint32_t park_headroom =
				pager->park_budget_blocks > tier->slots_in_use ?
				pager->park_budget_blocks - tier->slots_in_use : 0u;
			if ( park_headroom < deficit )
			{
				decision.outcome = SPARK_KV_PAGER_QUEUED;
				pager->statistics.admission_queued_backing += 1u;
				*decision_out = decision;
				return(SPARK_STATUS_OK);
			}
		}
	}
	if ( admission->restore_slack_microseconds != 0u )
	{
		uint64_t bandwidth = pager->statistics.measured_bytes_per_second;
		if ( bandwidth == 0u )
			bandwidth = tier->configuration.device_bytes_per_second;
		if ( bandwidth != 0u )
		{
			uint64_t debt_bytes = queued_restore_bytes +
				(uint64_t)deficit * (uint64_t)pager->block_bytes;
			if ( debt_bytes > UINT64_MAX / 1000000ull ||
				debt_bytes * 1000000ull / bandwidth >
					(uint64_t)admission->restore_slack_microseconds )
			{
				decision.outcome = SPARK_KV_PAGER_QUEUED;
				pager->statistics.admission_queued_bandwidth += 1u;
				*decision_out = decision;
				return(SPARK_STATUS_OK);
			}
		}
	}
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
		decision.outcome = SPARK_KV_PAGER_QUEUED;
		pager->statistics.admission_queued_device += 1u;
	}
	else if ( status == SPARK_STATUS_BUSY )
	{
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
	SparkKvPagerParkEntry entry;
	uint8_t digest[SPARK_KV_PAGER_DIGEST_BYTES];
	uint8_t *staging;
	uint32_t asynchronous,queue_head;
	uint64_t write_started;
	SparkStatus status;

	if ( SparkKvPagerIsValid(pager) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	(void)resident_slot_index;
	(void)generation;
	if ( key_bytes + value_bytes != (uint64_t)pager->block_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	asynchronous = pager->park_worker_active != 0u &&
		pager->park_queue_blocks != 0u;
	if ( pager->park_worker_active != 0u )
		(void)SparkKvPagerPollParkCompletions(pager);
	if ( asynchronous != 0u )
	{
		queue_head = atomic_load_explicit(&pager->park_head,
			memory_order_relaxed);
		staging = (uint8_t *)pager->configuration.park_staging +
			(uint64_t)(queue_head % SPARK_KV_PAGER_PARK_QUEUE_CAPACITY) *
				(uint64_t)pager->block_bytes;
	}
	else
	{
		staging = (uint8_t *)pager->configuration.staging;
	}
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
	view.host_staging = staging;
	status = pager->configuration.module_save(pager->configuration.module_context,
		&view);
	if ( status != SPARK_STATUS_OK )
		return(status);
	SparkKvPagerDigestOf(staging,pager->block_bytes,digest);
	status = SparkNvmeTierReserveWrite(pager->configuration.tier,
		SparkKvPagerHashFromDigest(digest),digest,&reservation);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( asynchronous != 0u && reservation.already_present == 0u )
	{
		entry.logical_block_index = logical_block_index;
		entry.reserved0 = 0u;
		entry.reservation = reservation;
		entry.staging = staging;
		entry.payload_bytes = (uint64_t)pager->block_bytes;
		if ( SparkKvPagerParkQueuePush(pager,&entry) != 0u )
		{
			return(SPARK_STATUS_OK);
		}
	}
	if ( reservation.already_present == 0u )
	{
		write_started = SparkKvPagerClockNow(pager);
		status = pager->configuration.backing_write(
			pager->configuration.backing_context,reservation.device_offset,
			staging,(uint64_t)pager->block_bytes);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkNvmeTierAbortWrite(pager->configuration.tier,&reservation);
			return(status);
		}
		SparkKvPagerFoldBandwidthSample(pager,(uint64_t)pager->block_bytes,
			SparkKvPagerClockNow(pager) - write_started);
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
    return(SparkKvPagerRestoreBlockDeadline(pager,logical_block_index,
        content_digest,SPARK_KV_PAGER_DISPATCH_NO_DEADLINE_HINT));
}

static SparkStatus SparkKvPagerRestoreBlockDeadlineEx(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES],
    uint32_t deadline_step,
    uint32_t *poll_budget_exhausted)
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
    *poll_budget_exhausted = 0u;
	(void)SparkKvPagerPollParkCompletions(pager);
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
	if ( SparkKvPagerParkIsInFlight(pager,logical_block_index,
		content_digest) != 0u )
	{
		return(SPARK_STATUS_BUSY);
	}
	{
		uint64_t read_started = 0u;
		uint32_t read_issued = 0u;
		for ( attempt = 0u; attempt < SPARK_KV_PAGER_RESTORE_POLL_LIMIT; ++attempt )
		{
			status = SparkNvmeTierRequestDemandDeadline(pager->configuration.tier,
				SparkKvPagerHashFromDigest(content_digest),content_digest,attempt,
				deadline_step,&demand);
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
			if ( status == SPARK_STATUS_BUSY && deadline_step != 0u &&
				demand.ordered != 0u )
			{
				(void)SparkNvmeTierPump(pager->configuration.tier,attempt);
				return(SPARK_STATUS_BUSY);
			}
			if ( read_issued == 0u )
			{
				read_issued = 1u;
				read_started = SparkKvPagerClockNow(pager);
			}
			else
			{
				(void)SparkKvPagerClockNow(pager);
			}
			status = SparkNvmeTierPump(pager->configuration.tier,attempt);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
		if ( attempt == SPARK_KV_PAGER_RESTORE_POLL_LIMIT )
		{
			*poll_budget_exhausted = 1u;
			return(SPARK_STATUS_IO_ERROR);
		}
		if ( read_issued != 0u )
		{
			SparkKvPagerFoldBandwidthSample(pager,
				(uint64_t)pager->block_bytes,
				SparkKvPagerClockNow(pager) - read_started);
		}
	}
	SparkKvPagerDigestOf(demand.staging_pointer,(uint64_t)pager->block_bytes,
		landed_digest);
	if ( memcmp(landed_digest,content_digest,SPARK_KV_PAGER_DIGEST_BYTES) != 0 )
	{
		(void)SparkNvmeTierConsume(pager->configuration.tier,
			SparkKvPagerHashFromDigest(content_digest),content_digest);
		return(SPARK_STATUS_HASH_MISMATCH);
	}
	memcpy(pager->landing_staging,demand.staging_pointer,
		pager->block_bytes);
	(void)SparkNvmeTierConsume(pager->configuration.tier,
		SparkKvPagerHashFromDigest(content_digest),content_digest);
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

SparkStatus SparkKvPagerRestoreBlockDeadline(
    SparkKvPager *pager,
    uint32_t logical_block_index,
    const uint8_t content_digest[SPARK_KV_PAGER_DIGEST_BYTES],
    uint32_t deadline_step)
{
	uint32_t poll_budget_exhausted;

	return(SparkKvPagerRestoreBlockDeadlineEx(pager,logical_block_index,
		content_digest,deadline_step,&poll_budget_exhausted));
}

SparkStatus SparkKvPagerDispatchBlock(
	SparkKvPager *pager,
	const SparkKvPagerDispatch *dispatch,
	SparkKvPagerDispatchDecision *decision_out)
{
	SparkKvPagerDispatchDecision decision;
	SparkKvCacheBlockView block_view;
	uint32_t poll_budget_exhausted;
	SparkStatus status;

	if ( SparkKvPagerIsValid(pager) == 0u || dispatch == 0 ||
		decision_out == 0 ||
		dispatch->abi_version != SPARK_KV_PAGER_DISPATCH_ABI_VERSION ||
		dispatch->descriptor_bytes !=
			SPARK_KV_PAGER_DISPATCH_DESCRIPTOR_BYTES ||
		SparkKvPagerDigestIsUsable(dispatch->content_digest) == 0u )
	{
		return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	memset(decision_out,0,sizeof(*decision_out));
	memset(&decision,0,sizeof(decision));
	decision.abi_version = SPARK_KV_PAGER_DISPATCH_ABI_VERSION;
	decision.descriptor_bytes =
		SPARK_KV_PAGER_DISPATCH_DECISION_DESCRIPTOR_BYTES;
	decision.logical_block_index = dispatch->logical_block_index;
	pager->statistics.dispatch_requests += 1u;
	status = SparkKvPagerRestoreBlockDeadlineEx(pager,
		dispatch->logical_block_index,dispatch->content_digest,
		dispatch->deadline_step,&poll_budget_exhausted);
	if ( status == SPARK_STATUS_IO_ERROR && poll_budget_exhausted != 0u )
	{
		status = SPARK_STATUS_BUSY;
	}
	if ( status == SPARK_STATUS_OK )
	{
		status = SparkKvCacheArenaResolveBlock(pager->configuration.arena,
			dispatch->logical_block_index,&block_view);
		if ( status == SPARK_STATUS_OK &&
			(block_view.flags & SPARK_KV_CACHE_BLOCK_FLAG_RESIDENT) == 0u )
		{
			status = SPARK_STATUS_INTERNAL_ERROR;
		}
		if ( status == SPARK_STATUS_OK )
		{
			decision.outcome = SPARK_KV_PAGER_DISPATCH_READY;
			decision.resident = 1u;
			pager->statistics.dispatch_ready += 1u;
		}
	}
	else if ( status == SPARK_STATUS_BUSY )
	{
		decision.outcome = SPARK_KV_PAGER_DISPATCH_QUEUED;
		pager->statistics.dispatch_queued += 1u;
		status = SPARK_STATUS_OK;
	}
	else if ( status == SPARK_STATUS_NOT_FOUND )
	{
		decision.outcome = SPARK_KV_PAGER_DISPATCH_RECOMPUTE;
		pager->statistics.dispatch_recompute += 1u;
		status = SPARK_STATUS_OK;
	}
	if ( status == SPARK_STATUS_OK )
		*decision_out = decision;
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

SparkStatus SparkKvPagerPollParkCompletions(SparkKvPager *pager)
{
	SparkKvPagerParkCompletion completion;

	if ( SparkKvPagerIsValid(pager) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	while ( SparkKvPagerParkCompletionPop(pager,&completion) != 0u )
		SparkKvPagerPublishCompletion(pager,&completion);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkKvPagerShutdown(SparkKvPager *pager)
{
	SparkKvPagerParkEntry entry;
	SparkKvPagerParkCompletion completion;
	pthread_t worker;

	if ( SparkKvPagerIsValid(pager) == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( pager->park_worker_active == 0u )
		return(SPARK_STATUS_OK);
	atomic_store_explicit(&pager->park_worker_stop,1u,memory_order_seq_cst);
	memcpy(&worker,pager->park_worker_handle,sizeof(worker));
	pthread_join(worker,0);
	pager->park_worker_active = 0u;
	atomic_store_explicit(&pager->park_write_inflight_valid,0u,
		memory_order_release);
	while ( SparkKvPagerParkQueuePop(pager,&entry) != 0u )
	{
		SparkKvPagerParkExecuteWrite(pager,&entry,&completion);
		SparkKvPagerPublishCompletion(pager,&completion);
	}
	return(SparkKvPagerPollParkCompletions(pager));
}
