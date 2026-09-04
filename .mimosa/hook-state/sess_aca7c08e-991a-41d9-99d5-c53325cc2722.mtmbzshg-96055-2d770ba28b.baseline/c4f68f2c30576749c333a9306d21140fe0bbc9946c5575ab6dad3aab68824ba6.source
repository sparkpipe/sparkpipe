#include "sparkpipe/spark_topology_switch.h"

#include "sparkpipe/spark_sha256.h"

#include <string.h>


#define TOPOLOGY_SWITCH_SEQUENCE_FREE 0u
#define TOPOLOGY_SWITCH_SEQUENCE_ACTIVE 1u
#define TOPOLOGY_SWITCH_SEQUENCE_AT_BOUNDARY 2u

#define TOPOLOGY_SWITCH_MANIFEST_DOMAIN 0x9e3779b97f4a7c15ULL

#define TOPOLOGY_SWITCH_PLAN_CHUNK 16u

typedef struct TopologySwitchSequence
{
	uint64_t sequence_id;
	uint64_t recipe_id;
	uint32_t position_tokens;
	uint32_t block_count;
	uint8_t state;
	uint8_t resume_class;
	uint8_t pins_done;
	uint8_t checkpointed;
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
	uint64_t key = SparkHashBytes(kv_namespace,&content_hash,sizeof(content_hash));
	return(key | 1u);
}

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
	if ( configuration->nvme_read_bytes_per_second == 0u
		|| configuration->nvme_write_bytes_per_second == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
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
	if ( target->recipe_id == sw->current_recipe.recipe_id )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	sw->target_recipe = *target;
	sw->state = SPARK_TOPOLOGY_SWITCH_QUIESCE;
	sw->phase_cursor = 0u;
	sw->swap_started = 0u;
	sw->last_error = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}


static uint32_t TopologySwitchQuiesceDrained(const SparkTopologySwitch *sw)
{
	const TopologySwitchSequence *sequences = TopologySwitchSequences(sw);
	uint32_t index;
	for ( index = 0u; index < sw->configuration.max_sequences; ++index )
		if ( sequences[index].state == TOPOLOGY_SWITCH_SEQUENCE_ACTIVE )
			return(0u);
	return(1u);
}


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
	if ( sequence->pins_done == 0u )
	{
		for ( index = 0u; index < sequence->block_count; ++index )
		{
			status = SparkNvmeTierPin(sw->configuration.tier,keys[index],0,1);
			if ( status == SPARK_STATUS_NOT_FOUND )
			{
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
	memset(manifest,0,sw->configuration.manifest_block_bytes);
	memcpy(manifest,&sequence->sequence_id,sizeof(uint64_t));
	memcpy(manifest + 8u,&sequence->recipe_id,sizeof(uint64_t));
	memcpy(manifest + 16u,&sequence->position_tokens,sizeof(uint32_t));
	memcpy(manifest + 20u,&sequence->block_count,sizeof(uint32_t));
	memcpy(manifest + 24u,keys,(uint64_t)sequence->block_count * sizeof(uint64_t));
	manifest_key = TopologySwitchManifestKey(
		sw->configuration.kv_namespace,sequence->sequence_id);
	{
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
			sw->phase_cursor--;
			sw->last_error = status;
			return(0u);
		}
	}
	return(1u);
}


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
				needs[fill].need_by_step = step_now + 1u;
				needs[fill].reserved0 = 0u;
				memset(needs[fill].content_digest,0,
					sizeof(needs[fill].content_digest));
			}
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
	case SPARK_TOPOLOGY_SWITCH_CHECKPOINT:
		if ( TopologySwitchCheckpointRun(sw) == 0u )
			break;
		sw->state = SPARK_TOPOLOGY_SWITCH_SWAP;
		sw->swap_started = 0u;
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
