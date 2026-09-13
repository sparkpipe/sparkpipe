#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/llm_defines.h"
#include "common/common_kv_frame.h"

static int g_failures = 0;

static void Check(int condition, const char *name)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",name);
		g_failures++;
	}
}

static SparkStatus FakeBuildRestoreBatch(const LmKvFramePlanConfig *configuration, const LmKvFramePendingLane *pending_lanes, uint32_t pending_lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, void *block_staging, uint32_t block_staging_record_capacity, void *gdn_staging, uint32_t gdn_staging_record_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count, uint32_t *lanes_built)
{
	(void)configuration;
	(void)pending_lanes;
	(void)packet_lane_counts;
	(void)packet_count;
	(void)block_staging;
	(void)block_staging_record_capacity;
	(void)gdn_staging;
	(void)gdn_staging_record_capacity;
	(void)blocks;
	(void)block_capacity;
	*block_count = pending_lane_count;
	*lanes_built = pending_lane_count;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeBuildEvictBatch(const LmKvFramePlanConfig *configuration, uint64_t sequence_id, const uint32_t *resident_blocks, uint32_t resident_block_count, uint32_t include_gdn_state, const void *block_staging, const void *gdn_staging, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count)
{
	(void)configuration;
	(void)sequence_id;
	(void)resident_blocks;
	(void)resident_block_count;
	(void)include_gdn_state;
	(void)block_staging;
	(void)gdn_staging;
	(void)blocks;
	(void)block_capacity;
	*block_count = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeSubmit(SparkStageKvClient *client, LmKvFrameBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority)
{
	(void)client;
	(void)operation;
	(void)blocks;
	(void)block_count;
	(void)priority;
	batch_state->state = SPARK_LLM_KV_BATCH_SUBMITTED;
	batch_state->status = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeProgress(SparkStageKvClient *client, LmKvFrameWorkState *work)
{
	(void)client;
	work->restore.state = SPARK_LLM_KV_BATCH_READY;
	work->restore.status = SPARK_STATUS_OK;
	work->evict.state = SPARK_LLM_KV_BATCH_READY;
	work->evict.status = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeAcknowledge(LmKvFrameBatchState *batch_state)
{
	(void)batch_state;
	return(SPARK_STATUS_OK);
}

int LlmModuleNegativeControlRun(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[2] = {0u,1u};
	uint64_t positions[2] = {0u,1u};
	uint32_t contexts[2] = {100u,1u};
	uint64_t sequence_ids[2] = {101u,102u};
	uint32_t host_mapping[2] = {0u,0u};
	uint32_t device_mapping[2] = {0u,0u};
	uint32_t host_blocks[8] = {0u};
	uint32_t host_counts[2] = {2u,1u};
	uint32_t stale_device[8] = {0u};
	SparkStatus status;
	uint32_t index;

	Check(SPARK_LLM_KV_BLOCK_TOKENS != 64u,"negative control flip applied (SPARK_LLM_KV_BLOCK_TOKENS)");
	memset(&state,0,sizeof(state));
	state.ops.tag = "negative_stage";
	state.ops.build_restore_batch = FakeBuildRestoreBatch;
	state.ops.build_evict_batch = FakeBuildEvictBatch;
	state.ops.submit = FakeSubmit;
	state.ops.progress = FakeProgress;
	state.ops.acknowledge = FakeAcknowledge;
	state.block_count = 8u;
	state.logical_to_slot = (uint32_t *)calloc(8u,sizeof(uint32_t));
	state.logical_to_slot_capacity = 8u;
	state.logical_stride = 4u;
	state.table_indices_device = (uint32_t *)malloc(8u * sizeof(uint32_t));
	state.table_counts_device = (uint32_t *)malloc(8u * sizeof(uint32_t));
	state.table_indices_host = (uint32_t *)calloc(8u,sizeof(uint32_t));
	state.slot_lane = (uint32_t *)calloc(8u,sizeof(uint32_t));
	state.slot_logical = (uint32_t *)calloc(8u,sizeof(uint32_t));
	state.slot_sequence = (uint64_t *)calloc(8u,sizeof(uint64_t));
	state.slot_dirty = (uint8_t *)calloc(8u,sizeof(uint8_t));
	state.slot_pinned = (uint8_t *)calloc(8u,sizeof(uint8_t));
	state.slot_free_stack = (uint32_t *)malloc(8u * sizeof(uint32_t));
	state.block_staging = malloc(64u);
	state.gdn_staging = malloc(64u);
	state.cache_bf16 = malloc(4096u);
	for (index = 0u; index < 8u; index++)
		state.slot_free_stack[index] = index;
	state.slot_free_count = 8u;
	state.evict_cursor = 0u;
	state.tier_active = 1u;
	view.cuda_stream = 0;
	view.host_row_lane_indices = lane_indices;
	view.host_row_positions = positions;
	view.host_context_lengths = contexts;
	view.host_slot_mapping = host_mapping;
	view.slot_mapping = device_mapping;
	table.lane_count = 2u;
	table.lane_stride = 4u;
	table.host_physical_block_indices = host_blocks;
	table.host_lane_physical_block_counts = host_counts;
	table.physical_block_indices = stale_device;
	table.lane_physical_block_counts = host_counts;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_OK,"negative control prepare status");
	Check(host_mapping[0] == 7u * SPARK_LLM_KV_BLOCK_TOKENS + (uint32_t)(positions[0] % SPARK_LLM_KV_BLOCK_TOKENS),"negative control module uses flipped value");
	Check(host_mapping[0] != 7u * 64u,"negative control diverges from contract vector (SPARK_LLM_KV_BLOCK_TOKENS)");
	Check(host_mapping[1] != 5u * 64u + 1u,"negative control offset diverges (SPARK_LLM_KV_BLOCK_TOKENS)");
	if ( g_failures == 0 )
		fprintf(stderr,"negative control detected: SPARK_LLM_KV_BLOCK_TOKENS flipped to %u changes module output\n",(uint32_t)SPARK_LLM_KV_BLOCK_TOKENS);
	free(state.logical_to_slot);
	free(state.table_indices_device);
	free(state.table_counts_device);
	free(state.table_indices_host);
	free(state.slot_lane);
	free(state.slot_logical);
	free(state.slot_sequence);
	free(state.slot_dirty);
	free(state.slot_pinned);
	free(state.slot_free_stack);
	free(state.block_staging);
	free(state.gdn_staging);
	free(state.cache_bf16);
	return(g_failures);
}
