#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <cuda_runtime.h>

#include "sparkpipe/llm_defines.h"
#include "sparkpipe/spark_module_abi.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_kv_client.h"
#include "sparkpipe/spark_stage_module_common.h"

typedef struct LmKvFrameBatchState
{
	uint64_t batch_id;
	SparkStatus status;
	uint32_t state;
	uint32_t submitted_block_count;
} LmKvFrameBatchState;

typedef struct LmKvFrameWorkState
{
	LmKvFrameBatchState restore;
	LmKvFrameBatchState evict;
} LmKvFrameWorkState;

typedef struct LmKvFramePendingLane
{
	uint64_t sequence_id;
	const uint32_t *nonresident_blocks;
	uint32_t nonresident_block_count;
	uint32_t gdn_nonresident;
} LmKvFramePendingLane;

typedef struct LmKvFramePlanConfig
{
	uint64_t model_fingerprint;
	uint64_t cache_layout_fingerprint;
	uint32_t rank_index;
	uint32_t block_record_bytes;
	uint32_t gdn_record_bytes;
	uint32_t lookahead_packet_count;
	uint32_t physical_block_capacity;
	uint32_t allocated_physical_block_count;
	uint32_t staging_block_capacity;
} LmKvFramePlanConfig;

typedef struct LmKvFrameOps
{
	const char *tag;
	SparkStatus (*build_restore_batch)(const LmKvFramePlanConfig *configuration, const LmKvFramePendingLane *pending_lanes, uint32_t pending_lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, void *block_staging, uint32_t block_staging_record_capacity, void *gdn_staging, uint32_t gdn_staging_record_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count, uint32_t *lanes_built);
	SparkStatus (*build_evict_batch)(const LmKvFramePlanConfig *configuration, uint64_t sequence_id, const uint32_t *resident_blocks, uint32_t resident_block_count, uint32_t include_gdn_state, const void *block_staging, const void *gdn_staging, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count);
	SparkStatus (*submit)(SparkStageKvClient *client, LmKvFrameBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority);
	SparkStatus (*progress)(SparkStageKvClient *client, LmKvFrameWorkState *work);
	SparkStatus (*acknowledge)(LmKvFrameBatchState *batch_state);
} LmKvFrameOps;

typedef struct LmKvFrameState
{
	LmKvFrameOps ops;
	SparkStageKvClient client;
	LmKvFramePlanConfig plan;
	LmKvFrameWorkState work;
	uint32_t tier_active;
	uint32_t block_count;
	uint32_t *logical_to_slot;
	uint64_t logical_to_slot_capacity;
	uint32_t logical_stride;
	uint32_t *table_indices_device;
	uint32_t *table_counts_device;
	uint32_t *table_indices_host;
	uint32_t *slot_lane;
	uint32_t *slot_logical;
	uint64_t *slot_sequence;
	uint8_t *slot_dirty;
	uint8_t *slot_pinned;
	uint32_t *slot_free_stack;
	uint32_t slot_free_count;
	uint32_t evict_cursor;
	void *block_staging;
	void *gdn_staging;
	void *cache_bf16;
} LmKvFrameState;

typedef struct LmKvFrameSlot
{
	void *cuda_stream;
	const uint32_t *host_row_lane_indices;
	const uint64_t *host_row_positions;
	const uint32_t *host_context_lengths;
	uint32_t *host_slot_mapping;
	uint32_t *slot_mapping;
} LmKvFrameSlot;

typedef struct LmKvFrameTable
{
	uint32_t lane_count;
	uint32_t lane_stride;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_physical_block_counts;
	const uint32_t *physical_block_indices;
	const uint32_t *lane_physical_block_counts;
} LmKvFrameTable;

static inline SparkStatus LmKvFrameWaitBatch(LmKvFrameState *state, LmKvFrameBatchState *batch)
{
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t polls = 0u;
	struct timespec pause;
	pause.tv_sec = 0;
	pause.tv_nsec = 500000;
	while ( batch->state == SPARK_LLM_KV_BATCH_SUBMITTED )
	{
		status = state->ops.progress(&state->client,&state->work);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		if ( batch->state == SPARK_LLM_KV_BATCH_READY )
			break;
		if ( ++polls >= SPARK_LLM_KV_POLL_BOUND )
		{
			fprintf(stderr,"%s kv_store_stall\n",state->ops.tag);
			SPARK_FAIL(SPARK_STATUS_IO_ERROR);
		}
		nanosleep(&pause,0);
	}
	if ( batch->state != SPARK_LLM_KV_BATCH_READY || batch->status != SPARK_STATUS_OK )
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	return(state->ops.acknowledge(batch));
}
static inline SparkStatus LmKvFrameEvictSlot(LmKvFrameState *state, uint32_t slot)
{
	LmKvFrameBatchState *batch = &state->work.evict;
	SparkKvStoreBlock blocks[1];
	uint32_t block_count = 0u,logical;
	uint64_t sequence_id;
	cudaError_t error;
	SparkStatus status;
	if ( state->slot_dirty[slot] != 0u )
	{
		error = cudaMemcpy(state->block_staging,(const uint8_t *)state->cache_bf16 + (uint64_t)slot * state->plan.block_record_bytes,(size_t)state->plan.block_record_bytes,cudaMemcpyDeviceToHost);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(state->ops.tag,error,"kv_evict_copy"));
		logical = state->slot_logical[slot];
		sequence_id = state->slot_sequence[slot];
		status = state->ops.build_evict_batch(&state->plan,sequence_id,&logical,1u,0u,state->block_staging,state->gdn_staging,blocks,1u,&block_count);
		if ( status == SPARK_STATUS_OK )
			status = state->ops.submit(&state->client,batch,SPARK_KV_STORE_OPERATION_PUT,blocks,block_count,SPARK_LLM_KV_RESTORE_PRIORITY_SPECULATIVE);
		if ( status == SPARK_STATUS_OK )
			status = LmKvFrameWaitBatch(state,batch);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	if ( state->logical_to_slot != 0 && state->slot_lane[slot] != UINT32_MAX &&
		state->slot_logical[slot] != UINT32_MAX && state->logical_stride != 0u )
	{
		uint64_t evict_index = (uint64_t)state->slot_lane[slot] * state->logical_stride + state->slot_logical[slot];
		if ( evict_index < state->logical_to_slot_capacity &&
			state->logical_to_slot[evict_index] == slot + 1u )
			state->logical_to_slot[evict_index] = 0u;
	}
	state->slot_dirty[slot] = 0u;
	state->slot_pinned[slot] = 0u;
	state->slot_lane[slot] = UINT32_MAX;
	state->slot_logical[slot] = UINT32_MAX;
	state->slot_sequence[slot] = 0u;
	state->slot_free_stack[state->slot_free_count++] = slot;
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFramePrepareFrame(LmKvFrameState *state, const LmKvFrameSlot *slot, const uint64_t *row_sequence_ids, LmKvFrameTable *table, uint32_t rows)
{
	LmKvFrameBatchState *restore_batch = &state->work.restore;
	SparkKvStoreBlock blocks[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t packet_lane_counts[1],block_count,lanes_built;
	uint32_t lane_required[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t lane_sequence[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_list[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t lane_count = 0u,row,lane_index,logical,slot_index;
	uint64_t logical_capacity;
	SparkStatus status;
	cudaError_t error;
	uint32_t uncommitted[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t uncommitted_count = 0u;
	uint32_t unwind_index;
	SparkStatus fail_status;
	if ( state->tier_active == 0u )
		return(SPARK_STATUS_OK);
	if ( row_sequence_ids == 0 || table == 0 || table->host_physical_block_indices == 0 || table->host_lane_physical_block_counts == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	logical_capacity = (uint64_t)table->lane_count * table->lane_stride;
	if ( logical_capacity == 0u || table->lane_count > SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT || table->lane_stride > SPARK_LLM_KV_MAX_BLOCKS_PER_LANE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( logical_capacity > state->logical_to_slot_capacity )
	{
		uint32_t *grown = (uint32_t *)realloc(state->logical_to_slot,(size_t)logical_capacity * sizeof(uint32_t));
		if ( grown == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(grown + state->logical_to_slot_capacity,0,(size_t)(logical_capacity - state->logical_to_slot_capacity) * sizeof(uint32_t));
		state->logical_to_slot = grown;
		state->logical_to_slot_capacity = logical_capacity;
		state->logical_stride = table->lane_stride;
	}
	state->logical_stride = table->lane_stride;
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint32_t required_for_row = (slot->host_context_lengths[row] + SPARK_LLM_KV_BLOCK_TOKENS - 1u) / SPARK_LLM_KV_BLOCK_TOKENS;
		uint64_t sequence_id = row_sequence_ids[row];
		if ( lane >= table->lane_count || required_for_row > table->lane_stride || sequence_id == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
			if ( lane_list[lane_index] == lane )
				break;
		if ( lane_index == lane_count )
		{
			lane_list[lane_count] = lane;
			lane_required[lane_count] = 0u;
			lane_sequence[lane_count] = sequence_id;
			lane_count++;
		}
		if ( required_for_row > lane_required[lane_index] )
			lane_required[lane_index] = required_for_row;
	}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->slot_pinned[slot_index - 1u] = 1u;
		}
	}
	{
		LmKvFramePendingLane pending_lanes[SPARK_LLM_KV_STAGING_RECORDS];
		uint32_t pending_slots[SPARK_LLM_KV_STAGING_RECORDS];
		uint32_t pending_logical[SPARK_LLM_KV_STAGING_RECORDS];
		uint64_t pending_lane_index[SPARK_LLM_KV_STAGING_RECORDS];
		uint32_t batch_block_count = 0u,batch_index;
		memset(pending_lanes,0,sizeof(pending_lanes));
		for (lane_index = 0u; lane_index < lane_count; lane_index++)
		{
			uint32_t lane = lane_list[lane_index];
			for (logical = 0u; logical < lane_required[lane_index]; logical++)
			{
				uint32_t *residency = &state->logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
				if ( *residency != 0u )
					continue;
				if ( state->slot_free_count == 0u )
				{
					uint32_t scans = 0u;
					while ( state->slot_pinned[state->evict_cursor] != 0u )
					{
						state->evict_cursor = (state->evict_cursor + 1u) % state->block_count;
						if ( ++scans > state->block_count )
							{
								fail_status = SPARK_STATUS_CAPACITY_EXCEEDED;
								goto fail;
							}
					}
					status = LmKvFrameEvictSlot(state,state->evict_cursor);
					if ( status != SPARK_STATUS_OK )
						{
							fail_status = status;
							goto fail;
						}
				}
			slot_index = state->slot_free_stack[--state->slot_free_count];
			uncommitted[uncommitted_count++] = slot_index;
			pending_lanes[batch_block_count].sequence_id = lane_sequence[lane_index];
			pending_lanes[batch_block_count].nonresident_blocks = &pending_logical[batch_block_count];
			pending_lanes[batch_block_count].nonresident_block_count = 1u;
			pending_lanes[batch_block_count].gdn_nonresident = 0u;
			pending_logical[batch_block_count] = logical;
			pending_slots[batch_block_count] = slot_index;
			pending_lane_index[batch_block_count] = (uint64_t)lane_index;
			batch_block_count++;
			if ( batch_block_count == SPARK_LLM_KV_STAGING_RECORDS )
			{
				packet_lane_counts[0] = batch_block_count;
				block_count = 0u;
				lanes_built = 0u;
				status = state->ops.build_restore_batch(&state->plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->block_staging,SPARK_LLM_KV_STAGING_RECORDS,state->gdn_staging,1u,blocks,SPARK_LLM_KV_STAGING_RECORDS,&block_count,&lanes_built);
				if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
					status = SPARK_STATUS_CAPACITY_EXCEEDED;
				if ( status == SPARK_STATUS_OK )
					status = state->ops.submit(&state->client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_LLM_KV_RESTORE_PRIORITY_IMMEDIATE);
				if ( status == SPARK_STATUS_OK )
					status = LmKvFrameWaitBatch(state,restore_batch);
				if ( status != SPARK_STATUS_OK )
					{
						fail_status = status;
						goto fail;
					}
				for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
				{
					error = cudaMemcpyAsync((uint8_t *)state->cache_bf16 + (uint64_t)pending_slots[batch_index] * state->plan.block_record_bytes,(const uint8_t *)state->block_staging + ((uint64_t)batch_index * state->plan.block_record_bytes),(size_t)state->plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
					if ( error != cudaSuccess )
						{
							fail_status = SparkStageModuleCudaStatus(state->ops.tag,error,"kv_restore_copy");
							goto fail;
						}
					state->slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
					state->slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
					state->slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
					state->slot_dirty[pending_slots[batch_index]] = 0u;
					state->logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
					for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
						if ( uncommitted[unwind_index] == pending_slots[batch_index] )
						{
							uncommitted[unwind_index] = uncommitted[--uncommitted_count];
							break;
						}
				}
				batch_block_count = 0u;
			}
		}
	}
	if ( batch_block_count != 0u )
	{
			packet_lane_counts[0] = batch_block_count;
			block_count = 0u;
			lanes_built = 0u;
			status = state->ops.build_restore_batch(&state->plan,pending_lanes,batch_block_count,packet_lane_counts,1u,state->block_staging,SPARK_LLM_KV_STAGING_RECORDS,state->gdn_staging,1u,blocks,SPARK_LLM_KV_STAGING_RECORDS,&block_count,&lanes_built);
			if ( status == SPARK_STATUS_OK && lanes_built != batch_block_count )
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
			if ( status == SPARK_STATUS_OK )
				status = state->ops.submit(&state->client,restore_batch,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_LLM_KV_RESTORE_PRIORITY_IMMEDIATE);
			if ( status == SPARK_STATUS_OK )
				status = LmKvFrameWaitBatch(state,restore_batch);
			if ( status != SPARK_STATUS_OK )
				{
					fail_status = status;
					goto fail;
				}
			for (batch_index = 0u; batch_index < batch_block_count; batch_index++)
			{
				error = cudaMemcpyAsync((uint8_t *)state->cache_bf16 + (uint64_t)pending_slots[batch_index] * state->plan.block_record_bytes,(const uint8_t *)state->block_staging + ((uint64_t)batch_index * state->plan.block_record_bytes),(size_t)state->plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
				if ( error != cudaSuccess )
					{
						fail_status = SparkStageModuleCudaStatus(state->ops.tag,error,"kv_restore_copy");
						goto fail;
					}
				state->slot_lane[pending_slots[batch_index]] = lane_list[pending_lane_index[batch_index]];
				state->slot_logical[pending_slots[batch_index]] = pending_logical[batch_index];
				state->slot_sequence[pending_slots[batch_index]] = pending_lanes[batch_index].sequence_id;
				state->slot_dirty[pending_slots[batch_index]] = 0u;
				for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
					if ( uncommitted[unwind_index] == pending_slots[batch_index] )
					{
						uncommitted[unwind_index] = uncommitted[--uncommitted_count];
						break;
					}
				state->logical_to_slot[((uint64_t)lane_list[pending_lane_index[batch_index]] * table->lane_stride) + pending_logical[batch_index]] = pending_slots[batch_index] + 1u;
			}
		}
	}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		uint64_t lane_slice = (uint64_t)lane * table->lane_stride;
		memcpy(state->table_indices_host + lane_slice,table->host_physical_block_indices + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t));
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
			state->table_indices_host[lane_slice + logical] = state->logical_to_slot[lane_slice + logical] - 1u;
		error = cudaMemcpyAsync((uint8_t *)state->table_indices_device + (lane_slice * sizeof(uint32_t)),state->table_indices_host + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			{
				fail_status = SparkStageModuleCudaStatus(state->ops.tag,error,"kv_table_upload");
				goto fail;
			}
	}
	table->physical_block_indices = state->table_indices_device;
	table->lane_physical_block_counts = state->table_counts_device;
	error = cudaMemcpyAsync((void *)state->table_counts_device,(const void *)table->host_lane_physical_block_counts,(size_t)table->lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(state->ops.tag,error,"kv_table_upload");
			goto fail;
		}
	for (row = 0u; row < rows; row++)
	{
		uint32_t lane = slot->host_row_lane_indices[row];
		uint64_t position = slot->host_row_positions[row];
		slot_index = state->logical_to_slot[((uint64_t)lane * table->lane_stride) + (uint32_t)(position / SPARK_LLM_KV_BLOCK_TOKENS)] - 1u;
		slot->host_slot_mapping[row] = slot_index * SPARK_LLM_KV_BLOCK_TOKENS + (uint32_t)(position % SPARK_LLM_KV_BLOCK_TOKENS);
	}
	error = cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		{
			fail_status = SparkStageModuleCudaStatus(state->ops.tag,error,"kv_slot_upload");
			goto fail;
		}
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->logical_to_slot[((uint64_t)lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(SPARK_STATUS_OK);
fail:
	for (unwind_index = 0u; unwind_index < uncommitted_count; unwind_index++)
		state->slot_free_stack[state->slot_free_count++] = uncommitted[unwind_index];
	for (lane_index = 0u; lane_index < lane_count; lane_index++)
	{
		uint32_t fail_lane = lane_list[lane_index];
		for (logical = 0u; logical < lane_required[lane_index]; logical++)
		{
			slot_index = state->logical_to_slot[((uint64_t)fail_lane * table->lane_stride) + logical];
			if ( slot_index != 0u )
				state->slot_pinned[slot_index - 1u] = 0u;
		}
	}
	return(fail_status);
}
static inline void LmKvFrameMarkWritten(LmKvFrameState *state, const LmKvFrameSlot *slot, uint32_t rows)
{
	uint32_t row,slot_index;
	if ( state->tier_active == 0u )
		return;
	for (row = 0u; row < rows; row++)
	{
		slot_index = slot->host_slot_mapping[row] / SPARK_LLM_KV_BLOCK_TOKENS;
		if ( slot_index < state->block_count )
			state->slot_dirty[slot_index] = 1u;
	}
}