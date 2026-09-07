#ifndef SPARKPIPE_SPARK_WORK_CONTROL_COMMON_H
#define SPARKPIPE_SPARK_WORK_CONTROL_COMMON_H

#include <limits.h>
#include <stddef.h>
#include <string.h>

static SparkStatus SPARK_WORK_CONTROL_FN(ValidatePlanConfiguration)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration)
{
	if ( configuration == 0
		|| configuration->model_fingerprint == 0u
		|| configuration->cache_layout_fingerprint == 0u
		|| configuration->block_record_bytes == 0u
		|| configuration->gdn_record_bytes == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_WORK_CONTROL_FN(PrepareBlock)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	uint64_t sequence_id,
	uint32_t logical_block,
	uint32_t operation,
	void *payload,
	uint32_t payload_bytes,
	SparkKvStoreBlock *block)
{
	int32_t key_bytes;
	if ( configuration == 0 || sequence_id == 0u || payload == 0
		|| payload_bytes == 0u || block == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(block,0,sizeof(*block));
	key_bytes = SparkStageKvClientFormatKey(
		block->key,
		SPARK_KV_STORE_MAX_KEY_BYTES,
		configuration->model_fingerprint,
		configuration->cache_layout_fingerprint,
		configuration->rank_index,
		sequence_id,
		logical_block);
	if ( key_bytes < 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	block->operation = operation;
	block->key_bytes = (uint32_t)key_bytes;
	block->payload_bytes = payload_bytes;
	block->payload = payload;
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_WORK_CONTROL_FN(ProgressBatch)(
	SparkStageKvClient *client,
	SPARK_WORK_CONTROL_TYPE(KvBatchState) *batch_state)
{
	SparkKvStoreCompletion completion;
	SparkStatus status;
	if ( batch_state->state != SPARK_WORK_CONTROL_CONST(BATCH_SUBMITTED) )
		return(SPARK_STATUS_OK);
	memset(&completion,0,sizeof(completion));
	status = SparkStageKvClientPoll(client,batch_state->batch_id,&completion);
	if ( status == SPARK_STATUS_BUSY || status == SPARK_STATUS_PENDING )
		return(SPARK_STATUS_OK);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( completion.abi_version != SPARK_KV_STORE_ABI_VERSION
		|| completion.descriptor_bytes != SPARK_KV_STORE_COMPLETION_BYTES
		|| completion.batch_id != batch_state->batch_id
		|| completion.completed_block_count > batch_state->submitted_block_count )
		return(SPARK_STATUS_ABI_MISMATCH);
	batch_state->status = completion.status;
	batch_state->state = SPARK_WORK_CONTROL_CONST(BATCH_READY);
	return(SPARK_STATUS_OK);
}

uint32_t SPARK_WORK_CONTROL_FN(GdnBlockEquivalents)(
	uint32_t gdn_record_bytes,
	uint32_t block_record_bytes)
{
	if ( gdn_record_bytes == 0u || block_record_bytes == 0u )
		return(0u);
	return((gdn_record_bytes + block_record_bytes - 1u) / block_record_bytes);
}

SparkStatus SPARK_WORK_CONTROL_FN(CumulativeNonresident)(
	const SPARK_WORK_CONTROL_TYPE(PendingLane) *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	uint32_t gdn_block_equivalents,
	uint32_t *cumulative_nonresident_block_counts)
{
	uint32_t cumulative_count,lane_index,packet_index;
	if ( pending_lanes == 0 || pending_lane_count == 0u
		|| packet_lane_counts == 0 || packet_count == 0u
		|| cumulative_nonresident_block_counts == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cumulative_count = 0u;
	lane_index = 0u;
	for ( packet_index = 0u; packet_index < packet_count; ++packet_index )
	{
		uint32_t packet_lane_index;
		if ( packet_lane_counts[packet_index] == 0u
			|| packet_lane_counts[packet_index] > pending_lane_count - lane_index )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		for ( packet_lane_index = 0u;
			packet_lane_index < packet_lane_counts[packet_index];
			++packet_lane_index )
		{
			const SPARK_WORK_CONTROL_TYPE(PendingLane) *lane;
			uint64_t lane_count;
			lane = &pending_lanes[lane_index++];
			if ( lane->sequence_id == 0u
				|| (lane->nonresident_block_count != 0u
					&& lane->nonresident_blocks == 0) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
			lane_count = lane->nonresident_block_count;
			if ( lane->gdn_nonresident != 0u )
				lane_count += gdn_block_equivalents;
			if ( lane_count > UINT32_MAX - cumulative_count )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			cumulative_count += (uint32_t)lane_count;
		}
		cumulative_nonresident_block_counts[packet_index] = cumulative_count;
	}
	if ( lane_index != pending_lane_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}

uint32_t SPARK_WORK_CONTROL_FN(SelectRestorePackets)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	uint32_t packet_count,
	const uint32_t *cumulative_nonresident_block_counts)
{
	if ( configuration == 0 )
		return(0u);
	return(SparkKvStoreSelectPressureLimitedLookaheadPacketCount(
		configuration->lookahead_packet_count,
		packet_count,
		configuration->physical_block_capacity,
		configuration->allocated_physical_block_count,
		configuration->staging_block_capacity,
		cumulative_nonresident_block_counts));
}

SparkStatus SPARK_WORK_CONTROL_FN(BuildRestoreBatch)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	const SPARK_WORK_CONTROL_TYPE(PendingLane) *pending_lanes,
	uint32_t pending_lane_count,
	const uint32_t *packet_lane_counts,
	uint32_t packet_count,
	void *block_staging,
	uint32_t block_staging_record_capacity,
	void *gdn_staging,
	uint32_t gdn_staging_record_capacity,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count,
	uint32_t *lanes_built)
{
	uint32_t block_record_index,gdn_record_index,lane_index;
	uint64_t declared_lane_count;
	SparkStatus status;
	status = SPARK_WORK_CONTROL_FN(ValidatePlanConfiguration)(configuration);
	if ( status != SPARK_STATUS_OK || pending_lanes == 0
		|| pending_lane_count == 0u || packet_lane_counts == 0
		|| packet_count == 0u || block_staging == 0 || gdn_staging == 0
		|| blocks == 0 || block_capacity == 0u || block_count == 0
		|| lanes_built == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	declared_lane_count = 0u;
	for ( lane_index = 0u; lane_index < packet_count; ++lane_index )
		declared_lane_count += packet_lane_counts[lane_index];
	if ( declared_lane_count != pending_lane_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*block_count = 0u;
	*lanes_built = 0u;
	block_record_index = 0u;
	gdn_record_index = 0u;
	for ( lane_index = 0u; lane_index < pending_lane_count; ++lane_index )
	{
		const SPARK_WORK_CONTROL_TYPE(PendingLane) *lane;
		uint32_t lane_output_count,lane_block_index;
		lane = &pending_lanes[lane_index];
		if ( lane->sequence_id == 0u
			|| (lane->nonresident_block_count != 0u
				&& lane->nonresident_blocks == 0) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		lane_output_count = lane->nonresident_block_count
			+ (lane->gdn_nonresident != 0u ? 1u : 0u);
		if ( lane_output_count > block_capacity - *block_count
			|| lane->nonresident_block_count
				> block_staging_record_capacity - block_record_index
			|| (lane->gdn_nonresident != 0u
				&& gdn_record_index >= gdn_staging_record_capacity) )
		{
			if ( *lanes_built == 0u )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
			break;
		}
		if ( lane->gdn_nonresident != 0u )
		{
			status = SPARK_WORK_CONTROL_FN(PrepareBlock)(
				configuration,lane->sequence_id,UINT32_MAX,
				SPARK_KV_STORE_OPERATION_GET,
				(uint8_t *)gdn_staging
					+ ((uint64_t)gdn_record_index * configuration->gdn_record_bytes),
				configuration->gdn_record_bytes,&blocks[*block_count]);
			if ( status != SPARK_STATUS_OK )
				return(status);
			gdn_record_index++;
			(*block_count)++;
		}
		for ( lane_block_index = 0u;
			lane_block_index < lane->nonresident_block_count;
			++lane_block_index )
		{
			status = SPARK_WORK_CONTROL_FN(PrepareBlock)(
				configuration,lane->sequence_id,
				lane->nonresident_blocks[lane_block_index],
				SPARK_KV_STORE_OPERATION_GET,
				(uint8_t *)block_staging
					+ ((uint64_t)block_record_index * configuration->block_record_bytes),
				configuration->block_record_bytes,&blocks[*block_count]);
			if ( status != SPARK_STATUS_OK )
				return(status);
			block_record_index++;
			(*block_count)++;
		}
		(*lanes_built)++;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SPARK_WORK_CONTROL_FN(BuildEvictBatch)(
	const SPARK_WORK_CONTROL_TYPE(KvPlanConfig) *configuration,
	uint64_t sequence_id,
	const uint32_t *resident_blocks,
	uint32_t resident_block_count,
	uint32_t include_gdn_state,
	const void *block_staging,
	const void *gdn_staging,
	SparkKvStoreBlock *blocks,
	uint32_t block_capacity,
	uint32_t *block_count)
{
	uint32_t output_count,resident_block_index;
	SparkStatus status;
	status = SPARK_WORK_CONTROL_FN(ValidatePlanConfiguration)(configuration);
	if ( status != SPARK_STATUS_OK || sequence_id == 0u
		|| (resident_block_count != 0u && resident_blocks == 0)
		|| block_staging == 0 || gdn_staging == 0 || blocks == 0
		|| block_count == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	output_count = resident_block_count + (include_gdn_state != 0u ? 1u : 0u);
	if ( output_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( output_count > block_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*block_count = 0u;
	if ( include_gdn_state != 0u )
	{
		status = SPARK_WORK_CONTROL_FN(PrepareBlock)(
			configuration,sequence_id,UINT32_MAX,SPARK_KV_STORE_OPERATION_PUT,
			(void *)gdn_staging,configuration->gdn_record_bytes,
			&blocks[*block_count]);
		if ( status != SPARK_STATUS_OK )
			return(status);
		(*block_count)++;
	}
	for ( resident_block_index = 0u;
		resident_block_index < resident_block_count;
		++resident_block_index )
	{
		status = SPARK_WORK_CONTROL_FN(PrepareBlock)(
			configuration,sequence_id,resident_blocks[resident_block_index],
			SPARK_KV_STORE_OPERATION_PUT,
			(uint8_t *)block_staging
				+ ((uint64_t)resident_block_index * configuration->block_record_bytes),
			configuration->block_record_bytes,&blocks[*block_count]);
		if ( status != SPARK_STATUS_OK )
			return(status);
		(*block_count)++;
	}
	return(SPARK_STATUS_OK);
}

SparkStatus SPARK_WORK_CONTROL_FN(Submit)(
	SparkStageKvClient *client,
	SPARK_WORK_CONTROL_TYPE(KvBatchState) *batch_state,
	uint32_t operation,
	const SparkKvStoreBlock *blocks,
	uint32_t block_count,
	uint32_t priority)
{
	SparkStatus status;
	uint64_t batch_id;
	if ( client == 0 || batch_state == 0 || blocks == 0
		|| block_count == 0u || block_count > SPARK_KV_STORE_MAX_BATCH_BLOCKS
		|| (operation != SPARK_KV_STORE_OPERATION_GET
			&& operation != SPARK_KV_STORE_OPERATION_PUT)
		|| priority > SPARK_WORK_CONTROL_CONST(RESTORE_PRIORITY_SPECULATIVE) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( batch_state->state != SPARK_WORK_CONTROL_CONST(BATCH_IDLE) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkStageKvClientSubmit(
		client,operation,blocks,block_count,priority,&batch_id);
	if ( status != SPARK_STATUS_OK )
		return(status);
	batch_state->batch_id = batch_id;
	batch_state->status = SPARK_STATUS_PENDING;
	batch_state->state = SPARK_WORK_CONTROL_CONST(BATCH_SUBMITTED);
	batch_state->submitted_block_count = block_count;
	return(SPARK_STATUS_OK);
}

SparkStatus SPARK_WORK_CONTROL_FN(Progress)(
	SparkStageKvClient *client,
	SPARK_WORK_CONTROL_TYPE(KvState) *state)
{
	SparkStatus status;
	if ( client == 0 || state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_WORK_CONTROL_FN(ProgressBatch)(client,&state->restore);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SPARK_WORK_CONTROL_FN(ProgressBatch)(client,&state->evict));
}

SparkStatus SPARK_WORK_CONTROL_FN(Acknowledge)(
	SPARK_WORK_CONTROL_TYPE(KvBatchState) *batch_state)
{
	if ( batch_state == 0
		|| batch_state->state != SPARK_WORK_CONTROL_CONST(BATCH_READY) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(batch_state,0,sizeof(*batch_state));
	return(SPARK_STATUS_OK);
}

#endif
