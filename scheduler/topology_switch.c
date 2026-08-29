#include "sparkpipe/spark_topology_switch.h"

#include "sparkpipe/spark_sha256.h"

#include <string.h>

// The mechanics behind the contract in spark_topology_switch.h. Three
// tables carved from the caller's blob at init, the arena pattern: one
// allocation, nothing afterwards:
//
//   SEQUENCES   one record per sequence the scheduler may have in flight.
//               Fixed slots, no free list: slot index is identity while a
//               sequence lives, and Track/Complete mark it used or free.
//   BLOCK KEYS  max_blocks_per_sequence u64s per slot, so a sequence's tier
//               keys are one contiguous run and a manifest is a memcpy away
//               from serialised.
//   MANIFEST    one staging buffer; manifests are written one at a time, so
//               one buffer is not a bottleneck, it is the honest shape.
//
// WHY PHASES RETRY INSTEAD OF FAILING. Every phase's work is bounded
// bookkeeping plus non-blocking vtable calls, and every vtable call can say
// BUSY for reasons that resolve themselves (the tier's clock finding no
// unpinned victim this pass, the swap stream not yet drained). A switch is
// an operator action, not a request-path event: the right response to BUSY
// is to finish the current decode step and try again, so Advance carries a
// per-phase cursor and last_error instead of an abort path. There is no
// CANCEL. Cancelling mid-swap would leave residency split between recipes,
// which is the one state worse than either recipe alone; the protocol is
// crash-only, and the manifest records on the tier are what a restarted rank
// rebuilds from.

#define TOPOLOGY_SWITCH_SEQUENCE_FREE 0u
#define TOPOLOGY_SWITCH_SEQUENCE_ACTIVE 1u
#define TOPOLOGY_SWITCH_SEQUENCE_AT_BOUNDARY 2u

// Manifest keys live in the same tier as block keys, so they must not
// collide with one: the domain constant is hashed into every manifest's
// content side, which a token-run hash cannot produce by construction
// (cache/cache.h hashes token ids; this hashes a fixed magic over a
// sequence id).
#define TOPOLOGY_SWITCH_MANIFEST_DOMAIN 0x9e3779b97f4a7c15ULL

// Resume plans a sequence's blocks into the tier lookahead in chunks: the
// needs array is stack, not heap, and a chunk this size keeps the stack
// frame small without turning planning into a per-block call.
#define TOPOLOGY_SWITCH_PLAN_CHUNK 16u

typedef struct TopologySwitchSequence
{
	uint64_t sequence_id;
	uint64_t recipe_id;          /* the recipe it was admitted under */
	uint32_t position_tokens;
	uint32_t block_count;
	uint8_t state;
	uint8_t resume_class;        /* SparkTopologySwitchResumeClass */
	uint8_t pins_done;           /* block pins taken; write reservation may still retry */
	uint8_t checkpointed;        /* pins down, manifest committed */
	uint32_t reserved1;
}
TopologySwitchSequence;

static uint64_t TopologySwitchAlignU64(uint64_t value, uint64_t alignment)
{
	return((value + alignment - 1u) & ~(alignment - 1u));
}

uint64_t SparkTopologySwitchTableBytes(
	const SparkTopologySwitchConfiguration *configuration)
{
	uint64_t total;
	if ( configuration == 0 )
		return(0u);
	total = 0u;
	total = TopologySwitchAlignU64(total
		+ (uint64_t)configuration->max_sequences * sizeof(TopologySwitchSequence),8u);
	total = TopologySwitchAlignU64(total
		+ (uint64_t)configuration->max_sequences
			* configuration->max_blocks_per_sequence * sizeof(uint64_t),8u);
	total = TopologySwitchAlignU64(total + configuration->manifest_block_bytes,8u);
	return(total);
}

uint64_t SparkTopologySwitchKvKey(
	uint64_t kv_namespace,
	uint64_t content_hash)
{
	// SparkHashBytes is order-sensitive FNV-1a (spark_status.h), so the
	// namespace commits first and the content hash second - the same content
	// under two models diverges, the same content under two strategies of
	// one model does not, which is exactly the partition the tier needs.
	uint64_t key = SparkHashBytes(kv_namespace,&content_hash,sizeof(content_hash));
	return(key | 1u);   /* never zero; zero means unhashed on the tier */
}

// A manifest's tier key: the sequence id behind the domain magic, inside
// the same namespace as the blocks it names.
static uint64_t TopologySwitchManifestKey(
	uint64_t kv_namespace,
	uint64_t sequence_id)
{
	uint64_t content = SparkHashBytes(
		TOPOLOGY_SWITCH_MANIFEST_DOMAIN,&sequence_id,sizeof(sequence_id)) | 1u;
	return(SparkTopologySwitchKvKey(kv_namespace,content));
}

static TopologySwitchSequence *TopologySwitchSequences(
	const SparkTopologySwitch *sw)
{
	return((TopologySwitchSequence *)sw->sequences);
}

static uint64_t *TopologySwitchBlockKeysOf(
	const SparkTopologySwitch *sw,
	uint32_t sequence_slot)
{
	return(sw->block_keys
		+ (uint64_t)sequence_slot * sw->configuration.max_blocks_per_sequence);
}

static int32_t TopologySwitchFindSequence(
	const SparkTopologySwitch *sw,
	uint64_t sequence_id)
{
	const TopologySwitchSequence *sequences = TopologySwitchSequences(sw);
	uint32_t index;
	for ( index = 0u; index < sw->configuration.max_sequences; ++index )
		if ( sequences[index].state != TOPOLOGY_SWITCH_SEQUENCE_FREE
			&& sequences[index].sequence_id == sequence_id )
			return((int32_t)index);
	return(-1);
}

SparkStatus SparkTopologySwitchInitialize(
	SparkTopologySwitch *sw,
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologySwitchSwapDevice *swap_device,
	SparkTopologySwitchWriteBlock write_block,
	void *write_block_context,
	void *tables)
{
	uint8_t *cursor;
	TopologySwitchSequence *sequences;
	uint32_t index;
	uint64_t manifest_capacity;
	if ( sw == 0 || configuration == 0 || swap_device == 0 || tables == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_TOPOLOGY_SWITCH_ABI_VERSION
		|| configuration->descriptor_bytes
			!= SPARK_TOPOLOGY_SWITCH_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	if ( configuration->tier == 0
		|| configuration->max_sequences == 0u
		|| configuration->max_blocks_per_sequence == 0u
		|| configuration->step_time_microseconds == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// Zero bandwidth makes the budget silently zero-cost, which is the one
	// estimate this module must never produce: refuse it at init instead.
	if ( configuration->nvme_read_bytes_per_second == 0u
		|| configuration->nvme_write_bytes_per_second == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// A manifest that cannot hold its own block list is a checkpoint that
	// truncates silently. Checked here, where the mistake is free.
	manifest_capacity = 24u
		+ (uint64_t)configuration->max_blocks_per_sequence * sizeof(uint64_t);
	if ( manifest_capacity > configuration->manifest_block_bytes )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( swap_device->begin_swap == 0 || swap_device->poll_swap == 0
		|| write_block == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(sw,0,sizeof(*sw));
	sw->configuration = *configuration;
	sw->swap_device = *swap_device;
	sw->write_block = write_block;
	sw->write_block_context = write_block_context;
	cursor = (uint8_t *)tables;
	sw->sequences = cursor;
	cursor += TopologySwitchAlignU64(
		(uint64_t)configuration->max_sequences * sizeof(TopologySwitchSequence),8u);
	sw->block_keys = (uint64_t *)(void *)cursor;
	cursor += TopologySwitchAlignU64(
		(uint64_t)configuration->max_sequences
			* configuration->max_blocks_per_sequence * sizeof(uint64_t),8u);
	sw->manifest_buffer = cursor;
	sequences = TopologySwitchSequences(sw);
	for ( index = 0u; index < configuration->max_sequences; ++index )
	{
		sequences[index].state = TOPOLOGY_SWITCH_SEQUENCE_FREE;
		sequences[index].resume_class = SPARK_TOPOLOGY_SWITCH_RESUME_PENDING;
	}
	sw->current_recipe = configuration->initial_recipe;
	sw->target_recipe = configuration->initial_recipe;
	sw->state = SPARK_TOPOLOGY_SWITCH_STEADY;
	sw->last_error = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

uint32_t SparkTopologySwitchAdmissionsOpen(const SparkTopologySwitch *sw)
{
	return(sw != 0 && sw->state == SPARK_TOPOLOGY_SWITCH_STEADY ? 1u : 0u);
}

SparkTopologySwitchState SparkTopologySwitchStateOf(
	const SparkTopologySwitch *sw)
{
	return((SparkTopologySwitchState)sw->state);
}

const SparkTopologyRecipe *SparkTopologySwitchCurrentRecipe(
	const SparkTopologySwitch *sw)
{
	return(&sw->current_recipe);
}

SparkStatus SparkTopologySwitchLastError(const SparkTopologySwitch *sw)
{
	return(sw->last_error);
}

SparkStatus SparkTopologySwitchTrackSequence(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint64_t recipe_id)
{
	TopologySwitchSequence *sequences;
	uint32_t index;
	if ( sw == 0 || sequence_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	// Admission is the scheduler's gate; this is the backstop. A sequence
	// tracked mid-switch would bind to the recipe being unloaded.
	if ( sw->state != SPARK_TOPOLOGY_SWITCH_STEADY )
		return(SPARK_STATUS_BUSY);
	if ( recipe_id != sw->current_recipe.recipe_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( TopologySwitchFindSequence(sw,sequence_id) >= 0 )
		return(SPARK_STATUS_DUPLICATE);
	sequences = TopologySwitchSequences(sw);
	for ( index = 0u; index < sw->configuration.max_sequences; ++index )
	{
		if ( sequences[index].state != TOPOLOGY_SWITCH_SEQUENCE_FREE )
			continue;
		sequences[index].sequence_id = sequence_id;
		sequences[index].recipe_id = recipe_id;
		sequences[index].position_tokens = 0u;
		sequences[index].block_count = 0u;
		sequences[index].state = TOPOLOGY_SWITCH_SEQUENCE_ACTIVE;
		sequences[index].resume_class = SPARK_TOPOLOGY_SWITCH_RESUME_PENDING;
		sequences[index].pins_done = 0u;
		sequences[index].checkpointed = 0u;
		sw->sequence_count++;
		return(SPARK_STATUS_OK);
	}
	return(SPARK_STATUS_CAPACITY_EXCEEDED);
}

SparkStatus SparkTopologySwitchSetSequenceKv(
	SparkTopologySwitch *sw,
	uint64_t sequence_id,
	uint32_t position_tokens,
	const uint64_t *content_hashes,
	uint32_t hash_count)
{
	int32_t slot;
	TopologySwitchSequence *sequence;
	uint64_t *keys;
	uint32_t index;
	if ( sw == 0 || ( content_hashes == 0 && hash_count != 0u ) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( hash_count > sw->configuration.max_blocks_per_sequence )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	slot = TopologySwitchFindSequence(sw,sequence_id);
	if ( slot < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	sequence = &TopologySwitchSequences(sw)[slot];
	// After checkpoint the block set is frozen on the tier; updating it now
	// would commit a manifest that names the wrong keys.
	if ( sequence->checkpointed != 0u )
		return(SPARK_STATUS_BUSY);
	keys = TopologySwitchBlockKeysOf(sw,(uint32_t)slot);
	for ( index = 0u; index < hash_count; ++index )
		keys[index] = SparkTopologySwitchKvKey(
			sw->configuration.kv_namespace,content_hashes[index]);
	sequence->block_count = hash_count;
	sequence->position_tokens = position_tokens;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTopologySwitchSequenceAtBoundary(
	SparkTopologySwitch *sw,
	uint64_t sequence_id)
{
	int32_t slot;
	if ( sw == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = TopologySwitchFindSequence(sw,sequence_id);
	if ( slot < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	TopologySwitchSequences(sw)[slot].state = TOPOLOGY_SWITCH_SEQUENCE_AT_BOUNDARY;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTopologySwitchSequenceComplete(
	SparkTopologySwitch *sw,
	uint64_t sequence_id)
{
	int32_t slot;
	TopologySwitchSequence *sequence;
	if ( sw == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	slot = TopologySwitchFindSequence(sw,sequence_id);
	if ( slot < 0 )
		return(SPARK_STATUS_NOT_FOUND);
	sequence = &TopologySwitchSequences(sw)[slot];
	// Checkpointed and now completing before resume: the pins came down at
	// checkpoint and resume will skip this freed slot, so they drop here.
	if ( sequence->pins_done != 0u )
	{
		const uint64_t *keys = TopologySwitchBlockKeysOf(sw,(uint32_t)slot);
		uint32_t index;
		for ( index = 0u; index < sequence->block_count; ++index )
			(void)SparkNvmeTierPin(sw->configuration.tier,keys[index],0,0);
	}
	if ( sw->state != SPARK_TOPOLOGY_SWITCH_STEADY
		&& sequence->checkpointed == 0u )
		sw->statistics.sequences_completed_mid_switch++;
	sequence->state = TOPOLOGY_SWITCH_SEQUENCE_FREE;
	sequence->resume_class = SPARK_TOPOLOGY_SWITCH_RESUME_PENDING;
	sequence->block_count = 0u;
	sw->sequence_count--;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkTopologySwitchBegin(
	SparkTopologySwitch *sw,
	const SparkTopologyRecipe *target)
{
	if ( sw == 0 || target == 0 || target->recipe_id == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( sw->state != SPARK_TOPOLOGY_SWITCH_STEADY )
		return(SPARK_STATUS_BUSY);
	// A switch to the running recipe is a misconfiguration wearing a fast
	// path's clothes; refuse it loudly.
	if ( target->recipe_id == sw->current_recipe.recipe_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sw->target_recipe = *target;
	sw->state = SPARK_TOPOLOGY_SWITCH_QUIESCE;
	sw->phase_cursor = 0u;
	sw->swap_started = 0u;
	sw->last_error = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

// -- quiesce ---------------------------------------------------------------
//
// Every tracked sequence must report a token boundary (or complete). The
// bound is one decode step: a step is atomic at the boundary by design, so
// "stop issuing and let what is in flight land" can never wait longer.

static uint32_t TopologySwitchQuiesceDrained(const SparkTopologySwitch *sw)
{
	const TopologySwitchSequence *sequences = TopologySwitchSequences(sw);
	uint32_t index;
	for ( index = 0u; index < sw->configuration.max_sequences; ++index )
		if ( sequences[index].state == TOPOLOGY_SWITCH_SEQUENCE_ACTIVE )
			return(0u);
	return(1u);
}

// -- checkpoint --------------------------------------------------------------
//
// Per sequence: pin every block still on the tier (BEFORE any manifest
// reserve, because ReserveWrite is the tier's only eviction path and an unpinned
// block is fair game for it), count the absent ones, then write one
// manifest. The manifest is the crash-recovery record: a restarted rank
// finds it by key and rebuilds exactly this table.

static SparkStatus TopologySwitchCheckpointOne(
	SparkTopologySwitch *sw,
	uint32_t slot)
{
	TopologySwitchSequence *sequence = &TopologySwitchSequences(sw)[slot];
	const uint64_t *keys = TopologySwitchBlockKeysOf(sw,slot);
	uint8_t *manifest = sw->manifest_buffer;
	SparkNvmeTierWriteReservation reservation;
	uint64_t manifest_key,device_offset = 0u;
	uint32_t index;
	SparkStatus status;
	// Pins first, reserves second: ReserveWrite is the tier's only eviction
	// path, so an unpinned block named by a manifest is fair game for the
	// manifest's own slot acquisition. pins_done makes the retry exact - a
	// failed reservation retries the reservation, never the pins, because a double
	// pin leaks one count that resume's single unpin never returns.
	if ( sequence->pins_done == 0u )
	{
		for ( index = 0u; index < sequence->block_count; ++index )
		{
			status = SparkNvmeTierPin(sw->configuration.tier,keys[index],0,1);
			if ( status == SPARK_STATUS_NOT_FOUND )
			{
				// Never written back during serving. Counted, not repaired:
				// the bytes live in device memory and the checkpoint is not
				// a copy engine - resume will classify RECOMPUTE.
				sw->statistics.tier_blocks_absent++;
				continue;
			}
			if ( status != SPARK_STATUS_OK )
				return(status);
			sw->statistics.tier_blocks_found++;
			sw->statistics.blocks_pinned++;
		}
		sequence->pins_done = 1u;
	}
	// Serialise: id, recipe, position, count, then the key run.
	memset(manifest,0,sw->configuration.manifest_block_bytes);
	memcpy(manifest,&sequence->sequence_id,sizeof(uint64_t));
	memcpy(manifest + 8u,&sequence->recipe_id,sizeof(uint64_t));
	memcpy(manifest + 16u,&sequence->position_tokens,sizeof(uint32_t));
	memcpy(manifest + 20u,&sequence->block_count,sizeof(uint32_t));
	memcpy(manifest + 24u,keys,(uint64_t)sequence->block_count * sizeof(uint64_t));
	manifest_key = TopologySwitchManifestKey(
		sw->configuration.kv_namespace,sequence->sequence_id);
	{
		/* B3: the tier's records carry the SHA-256 of their payload; the
		 * switch wrote the manifest bytes, so it presents their digest. */
		uint8_t manifest_digest[SPARK_NVME_TIER_DIGEST_BYTES];
		SparkSha256Context digest_context;
		SparkSha256Initialize(&digest_context);
		SparkSha256Update(&digest_context,manifest,
			sw->configuration.manifest_block_bytes);
		SparkSha256Finalize(&digest_context,manifest_digest);
		status = SparkNvmeTierReserveWrite(
			sw->configuration.tier,manifest_key,manifest_digest,
			&reservation);
	}
	if ( status != SPARK_STATUS_OK )
		return(status);
	device_offset = reservation.device_offset;
	if ( reservation.already_present == 0u )
	{
		status = sw->write_block(sw->write_block_context,device_offset,
			manifest,sw->configuration.manifest_block_bytes);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SparkNvmeTierAbortWrite(
				sw->configuration.tier,&reservation);
			return(status);
		}
		status = SparkNvmeTierCommitWrite(
			sw->configuration.tier,&reservation);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	sequence->checkpointed = 1u;
	sw->statistics.sequences_checkpointed++;
	sw->statistics.manifest_writes++;
	sw->statistics.manifest_bytes += sw->configuration.manifest_block_bytes;
	return(SPARK_STATUS_OK);
}

static uint32_t TopologySwitchCheckpointRun(SparkTopologySwitch *sw)
{
	TopologySwitchSequence *sequences = TopologySwitchSequences(sw);
	while ( sw->phase_cursor < sw->configuration.max_sequences )
	{
		uint32_t slot = sw->phase_cursor;
		SparkStatus status;
		sw->phase_cursor++;
		if ( sequences[slot].state == TOPOLOGY_SWITCH_SEQUENCE_FREE
			|| sequences[slot].checkpointed != 0u )
			continue;
		status = TopologySwitchCheckpointOne(sw,slot);
		if ( status != SPARK_STATUS_OK )
		{
			// Retry this sequence next Advance; the cursor does not advance
			// past it, so nothing is skipped, and pins_done keeps the retry
			// from double-pinning what the first attempt already pinned.
			sw->phase_cursor--;
			sw->last_error = status;
			return(0u);
		}
	}
	return(1u);
}

// -- resume ------------------------------------------------------------------
//
// Classification is per sequence and total: every pinned block still on the
// tier is WARM and goes into the lookahead so the first steps fetch ahead;
// one absent block makes the whole sequence RECOMPUTE, because resuming
// decode from position N requires the whole prefix, not most of it. The
// partial prefix is not wasted - the recompute's own demand path hits the
// tier for whatever survived - but the skip-prefill win is all-or-nothing.

static void TopologySwitchResumeOne(
	SparkTopologySwitch *sw,
	uint32_t slot,
	uint32_t step_now)
{
	TopologySwitchSequence *sequence = &TopologySwitchSequences(sw)[slot];
	const uint64_t *keys = TopologySwitchBlockKeysOf(sw,slot);
	uint32_t index,warm = 1u;
	uint64_t device_offset;
	for ( index = 0u; index < sequence->block_count; ++index )
	{
		if ( SparkNvmeTierOffsetOf(sw->configuration.tier,keys[index],0,
				&device_offset) != SPARK_STATUS_OK )
		{
			warm = 0u;
		}
		// The pin drops either way: its job was to protect the block between
		// checkpoint and this moment, and this moment has arrived.
		(void)SparkNvmeTierPin(sw->configuration.tier,keys[index],0,0);
	}
	if ( warm != 0u && sequence->block_count != 0u )
	{
		SparkNvmeTierNeed needs[TOPOLOGY_SWITCH_PLAN_CHUNK];
		uint32_t base;
		for ( base = 0u; base < sequence->block_count;
			base += TOPOLOGY_SWITCH_PLAN_CHUNK )
		{
			uint32_t count = sequence->block_count - base;
			uint32_t fill;
			if ( count > TOPOLOGY_SWITCH_PLAN_CHUNK )
				count = TOPOLOGY_SWITCH_PLAN_CHUNK;
			for ( fill = 0u; fill < count; ++fill )
			{
				needs[fill].content_hash = keys[base + fill];
				// The sequence re-issues as soon as admissions open; its
				// first layer needs block zero immediately, and layer i
				// block i one step-ish later. One deadline for all is the
				// honest version of "now".
				needs[fill].need_by_step = step_now + 1u;
				needs[fill].reserved0 = 0u;
				/* Key-only planning: the switch classifies, it never hands
				 * bytes over - landing still verifies against each record's
				 * stored digest before anything becomes readable. */
				memset(needs[fill].content_digest,0,
					sizeof(needs[fill].content_digest));
			}
			// A plan failure costs prefetch, never correctness: the demand
			// path is the fallback, so the result is deliberately unchecked.
			(void)SparkNvmeTierPlanLookahead(
				sw->configuration.tier,needs,count,step_now,0);
		}
		sequence->resume_class = SPARK_TOPOLOGY_SWITCH_RESUME_WARM;
		sw->statistics.sequences_resumed_warm++;
	}
	else
	{
		sequence->resume_class = SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE;
		sw->statistics.sequences_resumed_recompute++;
	}
}

SparkTopologySwitchState SparkTopologySwitchAdvance(
	SparkTopologySwitch *sw,
	uint32_t step_now)
{
	if ( sw == 0 )
		return(SPARK_TOPOLOGY_SWITCH_STEADY);
	switch ( sw->state )
	{
	case SPARK_TOPOLOGY_SWITCH_QUIESCE:
		if ( TopologySwitchQuiesceDrained(sw) == 0u )
			break;
		sw->state = SPARK_TOPOLOGY_SWITCH_CHECKPOINT;
		sw->phase_cursor = 0u;
		/* The drain's last boundary IS this step, and the checkpoint is
		   bookkeeping, not I/O waits - same call, same step. */
		/* fall through */
	case SPARK_TOPOLOGY_SWITCH_CHECKPOINT:
		if ( TopologySwitchCheckpointRun(sw) == 0u )
			break;
		sw->state = SPARK_TOPOLOGY_SWITCH_SWAP;
		sw->swap_started = 0u;
		/* Issuing the swap is one non-blocking vtable call. */
		/* fall through */
	case SPARK_TOPOLOGY_SWITCH_SWAP:
		if ( sw->swap_started == 0u )
		{
			SparkStatus status = sw->swap_device.begin_swap(
				sw->swap_device.context,&sw->current_recipe,&sw->target_recipe);
			if ( status != SPARK_STATUS_OK )
			{
				sw->last_error = status;
				break;
			}
			sw->swap_started = 1u;
			sw->statistics.swaps_started++;
		}
		{
			SparkStatus status = sw->swap_device.poll_swap(sw->swap_device.context);
			if ( status == SPARK_STATUS_BUSY )
				break;
			if ( status != SPARK_STATUS_OK )
			{
				sw->last_error = status;
				break;
			}
		}
		sw->state = SPARK_TOPOLOGY_SWITCH_RESUME;
		sw->phase_cursor = 0u;
		/* fall through */
	case SPARK_TOPOLOGY_SWITCH_RESUME:
	{
		TopologySwitchSequence *sequences = TopologySwitchSequences(sw);
		while ( sw->phase_cursor < sw->configuration.max_sequences )
		{
			uint32_t slot = sw->phase_cursor;
			sw->phase_cursor++;
			if ( sequences[slot].state == TOPOLOGY_SWITCH_SEQUENCE_FREE )
				continue;
			TopologySwitchResumeOne(sw,slot,step_now);
		}
		sw->current_recipe = sw->target_recipe;
		sw->state = SPARK_TOPOLOGY_SWITCH_STEADY;
		sw->last_error = SPARK_STATUS_OK;
		sw->statistics.switches_completed++;
		break;
	}
	default:
		break;
	}
	return((SparkTopologySwitchState)sw->state);
}

SparkTopologySwitchResumeClass SparkTopologySwitchResumeClassOf(
	const SparkTopologySwitch *sw,
	uint64_t sequence_id)
{
	int32_t slot;
	if ( sw == 0 )
		return(SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE);
	slot = TopologySwitchFindSequence(sw,sequence_id);
	if ( slot < 0 )
		return(SPARK_TOPOLOGY_SWITCH_RESUME_RECOMPUTE);
	return((SparkTopologySwitchResumeClass)
		TopologySwitchSequences(sw)[slot].resume_class);
}

SparkStatus SparkTopologySwitchEstimateBudget(
	const SparkTopologySwitchConfiguration *configuration,
	const SparkTopologyRecipe *target,
	uint32_t active_sequence_count,
	uint64_t warm_kv_bytes,
	SparkTopologySwitchBudget *budget_out)
{
	SparkTopologySwitchBudget budget;
	if ( configuration == 0 || target == 0 || budget_out == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->nvme_read_bytes_per_second == 0u
		|| configuration->nvme_write_bytes_per_second == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&budget,0,sizeof(budget));
	budget.quiesce_us = configuration->step_time_microseconds;
	{
		uint64_t manifest_bytes = (uint64_t)active_sequence_count
			* configuration->manifest_block_bytes;
		budget.checkpoint_us = ( manifest_bytes * 1000000ULL )
			/ configuration->nvme_write_bytes_per_second;
	}
	budget.swap_fixed_us = configuration->swap_fixed_microseconds;
	budget.swap_stream_us = ( target->weight_pack_bytes * 1000000ULL )
		/ configuration->nvme_read_bytes_per_second;
	budget.resume_warm_us = ( warm_kv_bytes * 1000000ULL )
		/ configuration->nvme_read_bytes_per_second;
	budget.total_us = budget.quiesce_us + budget.checkpoint_us
		+ budget.swap_fixed_us + budget.swap_stream_us;
	*budget_out = budget;
	return(SPARK_STATUS_OK);
}

void SparkTopologySwitchGetStatistics(
	const SparkTopologySwitch *sw,
	SparkTopologySwitchStatistics *statistics_out)
{
	if ( statistics_out == 0 )
		return;
	*statistics_out = sw->statistics;
}
