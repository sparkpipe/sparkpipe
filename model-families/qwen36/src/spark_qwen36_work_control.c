#include "sparkpipe/spark_qwen36_work_control.h"

#include <string.h>

uint32_t SparkQwen36WorkControlGdnBlockEquivalents(uint32_t gdn_record_bytes, uint32_t block_record_bytes)
{
	if ( gdn_record_bytes == 0u || block_record_bytes == 0u )
		return(0u);
	return((gdn_record_bytes + block_record_bytes - 1u) / block_record_bytes);
}

// One lane's transfer cost in block equivalents: its nonresident attention
// blocks plus the GDN record's staging charge when it must come back too.
static uint32_t SparkQwen36WorkControlLaneCost(const SparkQwen36WorkControlPendingLane *lane, uint32_t gdn_block_equivalents)
{
	return(lane->nonresident_block_count + (lane->gdn_nonresident != 0u ? gdn_block_equivalents : 0u));
}

SparkStatus SparkQwen36WorkControlCumulativeNonresident(const SparkQwen36WorkControlPendingLane *pending, uint32_t lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, uint32_t gdn_block_equivalents, uint32_t *cumulative_out)
{
	uint32_t packet,lane = 0u,within,cumulative = 0u;
	if ( pending == 0 || packet_lane_counts == 0 || cumulative_out == 0 || packet_count == 0u || packet_count > SPARK_KV_STORE_MAX_LOOKAHEAD_PACKETS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (packet = 0; packet < packet_count; packet++)
	{
		for (within = 0; within < packet_lane_counts[packet]; within++)
		{
			if ( lane >= lane_count )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			cumulative += SparkQwen36WorkControlLaneCost(&pending[lane],gdn_block_equivalents);
			lane++;
		}
		cumulative_out[packet] = cumulative;
	}
	if ( lane != lane_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

uint32_t SparkQwen36WorkControlSelectRestorePackets(const SparkQwen36WorkControlKvPlanConfig *config, uint32_t packet_count, const uint32_t *cumulative_nonresident)
{
	if ( config == 0 || packet_count == 0u )
		return(0u);
	return(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(config->lookahead_packet_count,packet_count,config->physical_block_capacity,config->allocated_physical_block_count,config->staging_block_capacity,cumulative_nonresident));
}

// All of one lane's records into the batch, or none of them: the GDN
// recurrence record first when nonresident, then every nonresident
// attention block, each drawing its payload from the matching staging pool.
static SparkStatus SparkQwen36WorkControlAppendLaneRestore(const SparkQwen36WorkControlKvPlanConfig *config, const SparkQwen36WorkControlPendingLane *lane, void *block_staging, uint32_t block_staging_capacity, void *gdn_staging, uint32_t gdn_staging_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count, uint32_t *block_staging_used, uint32_t *gdn_staging_used)
{
	uint32_t needed = lane->nonresident_block_count + (lane->gdn_nonresident != 0u ? 1u : 0u),index;
	SparkKvStoreBlock *block;
	int32_t key_bytes;
	if ( needed == 0u )
		return(SPARK_STATUS_OK);
	if ( *block_count + needed > block_capacity || *block_staging_used + lane->nonresident_block_count > block_staging_capacity || (lane->gdn_nonresident != 0u && *gdn_staging_used + 1u > gdn_staging_capacity) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( lane->gdn_nonresident != 0u )
	{
		block = &blocks[*block_count];
		memset(block,0,sizeof(*block));
		block->operation = SPARK_KV_STORE_OPERATION_GET;
		block->payload_bytes = config->gdn_record_bytes;
		block->payload = (uint8_t *)gdn_staging + ((uint64_t)*gdn_staging_used * config->gdn_record_bytes);
		key_bytes = SparkStageKvClientFormatKey(block->key,SPARK_KV_STORE_MAX_KEY_BYTES,config->model_fingerprint,config->cache_layout_fingerprint,config->rank_index,lane->sequence_id,SPARK_QWEN36_WORK_CONTROL_GDN_RECORD_BLOCK);
		if ( key_bytes < 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		block->key_bytes = (uint32_t)key_bytes;
		*gdn_staging_used += 1u;
		*block_count += 1u;
	}
	for (index = 0; index < lane->nonresident_block_count; index++)
	{
		block = &blocks[*block_count];
		memset(block,0,sizeof(*block));
		block->operation = SPARK_KV_STORE_OPERATION_GET;
		block->payload_bytes = config->block_record_bytes;
		block->payload = (uint8_t *)block_staging + ((uint64_t)*block_staging_used * config->block_record_bytes);
		key_bytes = SparkStageKvClientFormatKey(block->key,SPARK_KV_STORE_MAX_KEY_BYTES,config->model_fingerprint,config->cache_layout_fingerprint,config->rank_index,lane->sequence_id,lane->nonresident_blocks[index]);
		if ( key_bytes < 0 )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		block->key_bytes = (uint32_t)key_bytes;
		*block_staging_used += 1u;
		*block_count += 1u;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen36WorkControlBuildRestoreBatch(const SparkQwen36WorkControlKvPlanConfig *config, const SparkQwen36WorkControlPendingLane *pending, uint32_t lane_count, const uint32_t *packet_lane_counts, uint32_t selected_packet_count, void *block_staging, uint32_t block_staging_capacity, void *gdn_staging, uint32_t gdn_staging_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count_out, uint32_t *lanes_built_out)
{
	uint32_t packet,lane = 0u,within,block_count = 0u,block_staging_used = 0u,gdn_staging_used = 0u;
	SparkStatus status;
	if ( config == 0 || pending == 0 || packet_lane_counts == 0 || blocks == 0 || block_count_out == 0 || lanes_built_out == 0 || selected_packet_count == 0u || block_capacity > SPARK_KV_STORE_MAX_BATCH_BLOCKS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*block_count_out = 0u;
	*lanes_built_out = 0u;
	for (packet = 0; packet < selected_packet_count; packet++)
		for (within = 0; within < packet_lane_counts[packet]; within++)
		{
			if ( lane >= lane_count )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			status = SparkQwen36WorkControlAppendLaneRestore(config,&pending[lane],block_staging,block_staging_capacity,gdn_staging,gdn_staging_capacity,blocks,block_capacity,&block_count,&block_staging_used,&gdn_staging_used);
			if ( status == SPARK_STATUS_CAPACITY_EXCEEDED )
			{
				*block_count_out = block_count;
				*lanes_built_out = lane;
				return(SPARK_STATUS_OK);
			}
			if ( status != SPARK_STATUS_OK )
				return(status);
			lane++;
		}
	*block_count_out = block_count;
	*lanes_built_out = lane;
	return(SPARK_STATUS_OK);
}

// Eviction payloads are the caller's already-staged copies of the lane, in
// the order of resident_blocks, with the GDN record leading when present -
// the exact mirror of a restore, so a stored lane restores byte-identical.
SparkStatus SparkQwen36WorkControlBuildEvictBatch(const SparkQwen36WorkControlKvPlanConfig *config, uint64_t sequence_id, const uint32_t *resident_blocks, uint32_t resident_block_count, uint32_t gdn_present, void *block_staging, void *gdn_staging, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count_out)
{
	SparkQwen36WorkControlPendingLane lane;
	uint32_t block_count = 0u,block_staging_used = 0u,gdn_staging_used = 0u,index;
	SparkStatus status;
	if ( config == 0 || blocks == 0 || block_count_out == 0 || (resident_block_count != 0u && resident_blocks == 0) || block_capacity > SPARK_KV_STORE_MAX_BATCH_BLOCKS )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	lane.sequence_id = sequence_id;
	lane.lane_index = 0u;
	lane.nonresident_block_count = resident_block_count;
	lane.nonresident_blocks = resident_blocks;
	lane.gdn_nonresident = gdn_present;
	lane.reserved0 = 0u;
	status = SparkQwen36WorkControlAppendLaneRestore(config,&lane,block_staging,resident_block_count,gdn_staging,gdn_present != 0u ? 1u : 0u,blocks,block_capacity,&block_count,&block_staging_used,&gdn_staging_used);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (index = 0; index < block_count; index++)
		blocks[index].operation = SPARK_KV_STORE_OPERATION_PUT;
	*block_count_out = block_count;
	return(SPARK_STATUS_OK);
}

// BUSY is the pressure valve: the provider's inflight window is full, the
// batch state stays IDLE, nothing was consumed, and the frame this batch
// serves simply remains queued until a later Progress frees the window.
SparkStatus SparkQwen36WorkControlSubmit(SparkStageKvClient *client, SparkQwen36WorkControlKvBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority)
{
	SparkStatus status;
	if ( client == 0 || batch_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( batch_state->state != SPARK_QWEN36_WORK_CONTROL_BATCH_IDLE )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvClientSubmit(client,operation,blocks,block_count,priority,&batch_state->batch_id);
	if ( status != SPARK_STATUS_OK )
		return(status);
	batch_state->state = SPARK_QWEN36_WORK_CONTROL_BATCH_WAIT;
	batch_state->status = SPARK_STATUS_BUSY;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkQwen36WorkControlPollBatch(SparkStageKvClient *client, SparkQwen36WorkControlKvBatchState *batch_state)
{
	SparkKvStoreCompletion completion;
	SparkStatus status;
	if ( batch_state->state != SPARK_QWEN36_WORK_CONTROL_BATCH_WAIT )
		return(SPARK_STATUS_OK);
	status = SparkStageKvClientPoll(client,batch_state->batch_id,&completion);
	if ( status == SPARK_STATUS_BUSY )
		return(SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
		return(status);
	batch_state->status = completion.status;
	batch_state->state = SPARK_QWEN36_WORK_CONTROL_BATCH_READY;
	return(SPARK_STATUS_OK);
}

SparkStatus SparkQwen36WorkControlProgress(SparkStageKvClient *client, SparkQwen36WorkControlKvState *kv_state)
{
	SparkStatus status;
	if ( client == 0 || kv_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkQwen36WorkControlPollBatch(client,&kv_state->restore);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkQwen36WorkControlPollBatch(client,&kv_state->evict));
}

SparkStatus SparkQwen36WorkControlAcknowledge(SparkQwen36WorkControlKvBatchState *batch_state)
{
	if ( batch_state == 0 || batch_state->state != SPARK_QWEN36_WORK_CONTROL_BATCH_READY )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	batch_state->state = SPARK_QWEN36_WORK_CONTROL_BATCH_IDLE;
	batch_state->batch_id = 0u;
	return(SPARK_STATUS_OK);
}
