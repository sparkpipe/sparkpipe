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

typedef struct LmKvFrameLanes
{
	uint32_t count;
	uint32_t lane[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t required[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint64_t sequence[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t index_of[SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT];
} LmKvFrameLanes;

typedef struct LmKvFrameRestore
{
	uint32_t count;
	uint32_t committed;
	LmKvFramePendingLane pending[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t logical[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t slot[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t lane[SPARK_LLM_KV_STAGING_RECORDS];
} LmKvFrameRestore;

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
	uint64_t sequence_id,evict_index;
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
	if ( state->logical_to_slot != 0 && state->slot_lane[slot] != UINT32_MAX && state->slot_logical[slot] != UINT32_MAX && state->logical_stride != 0u )
	{
		evict_index = (uint64_t)state->slot_lane[slot] * state->logical_stride + state->slot_logical[slot];
		if ( evict_index < state->logical_to_slot_capacity && state->logical_to_slot[evict_index] == slot + 1u )
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
static inline SparkStatus LmKvFrameCheckTable(LmKvFrameState *state, const uint64_t *row_sequence_ids, const LmKvFrameTable *table)
{
	uint64_t logical_capacity;
	uint32_t *grown;
	if ( row_sequence_ids == 0 || table == 0 || table->host_physical_block_indices == 0 || table->host_lane_physical_block_counts == 0 || table->physical_block_indices == 0 || table->lane_physical_block_counts == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	logical_capacity = (uint64_t)table->lane_count * table->lane_stride;
	if ( logical_capacity == 0u || table->lane_count > SPARK_LLM_MAX_ACTIVE_SEQUENCE_COUNT || table->lane_stride > SPARK_LLM_KV_MAX_BLOCKS_PER_LANE )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( logical_capacity > state->logical_to_slot_capacity )
	{
		grown = (uint32_t *)realloc(state->logical_to_slot,(size_t)logical_capacity * sizeof(uint32_t));
		if ( grown == 0 )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		memset(grown + state->logical_to_slot_capacity,0,(size_t)(logical_capacity - state->logical_to_slot_capacity) * sizeof(uint32_t));
		state->logical_to_slot = grown;
		state->logical_to_slot_capacity = logical_capacity;
	}
	state->logical_stride = table->lane_stride;
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameCollectLanes(LmKvFrameLanes *lanes, const LmKvFrameSlot *slot, const uint64_t *row_sequence_ids, const LmKvFrameTable *table, uint32_t rows)
{
	uint32_t row,lane,required,index;
	memset(lanes->index_of,0xff,(size_t)table->lane_count * sizeof(uint32_t));
	lanes->count = 0u;
	for (row = 0u; row < rows; row++)
	{
		lane = slot->host_row_lane_indices[row];
		required = (slot->host_context_lengths[row] + SPARK_LLM_KV_BLOCK_TOKENS - 1u) / SPARK_LLM_KV_BLOCK_TOKENS;
		if ( lane >= table->lane_count || required > table->lane_stride || row_sequence_ids[row] == 0u )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
		index = lanes->index_of[lane];
		if ( index == UINT32_MAX )
		{
			index = lanes->count++;
			lanes->index_of[lane] = index;
			lanes->lane[index] = lane;
			lanes->required[index] = 0u;
			lanes->sequence[index] = row_sequence_ids[row];
		}
		if ( required > lanes->required[index] )
			lanes->required[index] = required;
	}
	return(SPARK_STATUS_OK);
}
static inline void LmKvFrameSetPins(LmKvFrameState *state, const LmKvFrameLanes *lanes, uint32_t lane_stride, uint8_t pinned)
{
	uint32_t index,logical,slot_index;
	const uint32_t *residency;
	for (index = 0u; index < lanes->count; index++)
	{
		residency = state->logical_to_slot + (uint64_t)lanes->lane[index] * lane_stride;
		for (logical = 0u; logical < lanes->required[index]; logical++)
		{
			slot_index = residency[logical];
			if ( slot_index != 0u )
				state->slot_pinned[slot_index - 1u] = pinned;
		}
	}
}
static inline SparkStatus LmKvFrameClaimSlot(LmKvFrameState *state, uint32_t *slot_index)
{
	uint32_t scans = 0u;
	SparkStatus status;
	if ( state->slot_free_count == 0u )
	{
		while ( state->slot_pinned[state->evict_cursor] != 0u )
		{
			state->evict_cursor = (state->evict_cursor + 1u) % state->block_count;
			if ( ++scans > state->block_count )
				return(SPARK_STATUS_CAPACITY_EXCEEDED);
		}
		status = LmKvFrameEvictSlot(state,state->evict_cursor);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	*slot_index = state->slot_free_stack[--state->slot_free_count];
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameFlushRestore(LmKvFrameState *state, LmKvFrameRestore *restore, const LmKvFrameSlot *slot, uint32_t lane_stride)
{
	SparkKvStoreBlock blocks[SPARK_LLM_KV_STAGING_RECORDS];
	uint32_t packet_lane_counts[1],block_count = 0u,lanes_built = 0u,index,slot_index;
	SparkStatus status;
	cudaError_t error;
	packet_lane_counts[0] = restore->count;
	status = state->ops.build_restore_batch(&state->plan,restore->pending,restore->count,packet_lane_counts,1u,state->block_staging,SPARK_LLM_KV_STAGING_RECORDS,state->gdn_staging,1u,blocks,SPARK_LLM_KV_STAGING_RECORDS,&block_count,&lanes_built);
	if ( status == SPARK_STATUS_OK && lanes_built != restore->count )
		status = SPARK_STATUS_CAPACITY_EXCEEDED;
	if ( status == SPARK_STATUS_OK )
		status = state->ops.submit(&state->client,&state->work.restore,SPARK_KV_STORE_OPERATION_GET,blocks,block_count,SPARK_LLM_KV_RESTORE_PRIORITY_IMMEDIATE);
	if ( status == SPARK_STATUS_OK )
		status = LmKvFrameWaitBatch(state,&state->work.restore);
	if ( status != SPARK_STATUS_OK )
		return(status);
	for (index = 0u; index < restore->count; index++)
	{
		slot_index = restore->slot[index];
		error = cudaMemcpyAsync((uint8_t *)state->cache_bf16 + (uint64_t)slot_index * state->plan.block_record_bytes,(const uint8_t *)state->block_staging + (uint64_t)index * state->plan.block_record_bytes,(size_t)state->plan.block_record_bytes,cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(state->ops.tag,error,"kv_restore_copy"));
		state->slot_lane[slot_index] = restore->lane[index];
		state->slot_logical[slot_index] = restore->logical[index];
		state->slot_sequence[slot_index] = restore->pending[index].sequence_id;
		state->slot_dirty[slot_index] = 0u;
		state->logical_to_slot[(uint64_t)restore->lane[index] * lane_stride + restore->logical[index]] = slot_index + 1u;
		restore->committed++;
	}
	restore->count = 0u;
	restore->committed = 0u;
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameQueueRestore(LmKvFrameState *state, LmKvFrameRestore *restore, const LmKvFrameSlot *slot, uint32_t lane, uint64_t sequence_id, uint32_t logical, uint32_t lane_stride)
{
	uint32_t slot_index,entry;
	SparkStatus status;
	status = LmKvFrameClaimSlot(state,&slot_index);
	if ( status != SPARK_STATUS_OK )
		return(status);
	entry = restore->count++;
	restore->pending[entry].sequence_id = sequence_id;
	restore->pending[entry].nonresident_blocks = &restore->logical[entry];
	restore->pending[entry].nonresident_block_count = 1u;
	restore->pending[entry].gdn_nonresident = 0u;
	restore->logical[entry] = logical;
	restore->slot[entry] = slot_index;
	restore->lane[entry] = lane;
	if ( restore->count == SPARK_LLM_KV_STAGING_RECORDS )
		return(LmKvFrameFlushRestore(state,restore,slot,lane_stride));
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameRestoreMissing(LmKvFrameState *state, const LmKvFrameLanes *lanes, LmKvFrameRestore *restore, const LmKvFrameSlot *slot, uint32_t lane_stride)
{
	uint32_t index,logical;
	const uint32_t *residency;
	SparkStatus status;
	for (index = 0u; index < lanes->count; index++)
	{
		residency = state->logical_to_slot + (uint64_t)lanes->lane[index] * lane_stride;
		for (logical = 0u; logical < lanes->required[index]; logical++)
		{
			if ( residency[logical] != 0u )
				continue;
			status = LmKvFrameQueueRestore(state,restore,slot,lanes->lane[index],lanes->sequence[index],logical,lane_stride);
			if ( status != SPARK_STATUS_OK )
				return(status);
		}
	}
	if ( restore->count != 0u )
		return(LmKvFrameFlushRestore(state,restore,slot,lane_stride));
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameUploadTables(LmKvFrameState *state, const LmKvFrameLanes *lanes, const LmKvFrameSlot *slot, LmKvFrameTable *table)
{
	uint32_t index,logical;
	uint64_t lane_slice;
	cudaError_t error;
	for (index = 0u; index < lanes->count; index++)
	{
		lane_slice = (uint64_t)lanes->lane[index] * table->lane_stride;
		memcpy(state->table_indices_host + lane_slice,table->host_physical_block_indices + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t));
		for (logical = 0u; logical < lanes->required[index]; logical++)
			state->table_indices_host[lane_slice + logical] = state->logical_to_slot[lane_slice + logical] - 1u;
		error = cudaMemcpyAsync((uint8_t *)state->table_indices_device + (lane_slice * sizeof(uint32_t)),state->table_indices_host + lane_slice,(size_t)table->lane_stride * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
		if ( error != cudaSuccess )
			return(SparkStageModuleCudaStatus(state->ops.tag,error,"kv_table_upload"));
	}
	table->physical_block_indices = state->table_indices_device;
	table->lane_physical_block_counts = state->table_counts_device;
	error = cudaMemcpyAsync((void *)state->table_counts_device,(const void *)table->host_lane_physical_block_counts,(size_t)table->lane_count * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(state->ops.tag,error,"kv_table_upload"));
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameMapRows(LmKvFrameState *state, const LmKvFrameSlot *slot, uint32_t lane_stride, uint32_t rows)
{
	uint32_t row,slot_index;
	uint64_t position;
	cudaError_t error;
	for (row = 0u; row < rows; row++)
	{
		position = slot->host_row_positions[row];
		slot_index = state->logical_to_slot[(uint64_t)slot->host_row_lane_indices[row] * lane_stride + (uint32_t)(position / SPARK_LLM_KV_BLOCK_TOKENS)] - 1u;
		slot->host_slot_mapping[row] = slot_index * SPARK_LLM_KV_BLOCK_TOKENS + (uint32_t)(position % SPARK_LLM_KV_BLOCK_TOKENS);
	}
	error = slot->slot_mapping != 0 ? cudaMemcpyAsync(slot->slot_mapping,slot->host_slot_mapping,(size_t)rows * sizeof(uint32_t),cudaMemcpyHostToDevice,(cudaStream_t)slot->cuda_stream) : cudaSuccess;
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(state->ops.tag,error,"kv_slot_upload"));
	return(SPARK_STATUS_OK);
}
static inline SparkStatus LmKvFrameUnwind(LmKvFrameState *state, const LmKvFrameLanes *lanes, const LmKvFrameRestore *restore, uint32_t lane_stride, SparkStatus status)
{
	uint32_t index;
	for (index = restore->committed; index < restore->count; index++)
		state->slot_free_stack[state->slot_free_count++] = restore->slot[index];
	LmKvFrameSetPins(state,lanes,lane_stride,0u);
	return(status);
}
static inline SparkStatus LmKvFramePrepareFrame(LmKvFrameState *state, const LmKvFrameSlot *slot, const uint64_t *row_sequence_ids, LmKvFrameTable *table, uint32_t rows)
{
	LmKvFrameLanes lanes;
	LmKvFrameRestore restore;
	SparkStatus status;
	if ( state->tier_active == 0u )
		return(SPARK_STATUS_OK);
	status = LmKvFrameCheckTable(state,row_sequence_ids,table);
	if ( status == SPARK_STATUS_OK )
		status = LmKvFrameCollectLanes(&lanes,slot,row_sequence_ids,table,rows);
	if ( status != SPARK_STATUS_OK )
		return(status);
	restore.count = 0u;
	restore.committed = 0u;
	LmKvFrameSetPins(state,&lanes,table->lane_stride,1u);
	status = LmKvFrameRestoreMissing(state,&lanes,&restore,slot,table->lane_stride);
	if ( status == SPARK_STATUS_OK )
		status = LmKvFrameUploadTables(state,&lanes,slot,table);
	if ( status == SPARK_STATUS_OK )
		status = LmKvFrameMapRows(state,slot,table->lane_stride,rows);
	if ( status != SPARK_STATUS_OK )
		return(LmKvFrameUnwind(state,&lanes,&restore,table->lane_stride,status));
	LmKvFrameSetPins(state,&lanes,table->lane_stride,0u);
	return(SPARK_STATUS_OK);
}
static inline void LmKvFrameRelease(LmKvFrameState *state)
{
	SparkStageKvClientClose(&state->client);
	free(state->logical_to_slot);
	free(state->slot_lane);
	free(state->slot_logical);
	free(state->slot_sequence);
	free(state->slot_dirty);
	free(state->slot_pinned);
	free(state->slot_free_stack);
	free(state->block_staging);
	free(state->gdn_staging);
	free(state->table_indices_host);
	if ( state->table_indices_device != 0 )
		cudaFree(state->table_indices_device);
	if ( state->table_counts_device != 0 )
		cudaFree(state->table_counts_device);
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