#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/llm_defines.h"
#include "common/common_kv_frame.h"

extern int LlmModuleNegativeControlRun(void);

static int failures = 0;

SparkStatus SparkStageModuleCudaStatus(const char *tag, cudaError_t error, const char *what)
{
	fprintf(stderr,"%s %s cuda_error=%d\n",tag,what,(int)error);
	return(SPARK_STATUS_IO_ERROR);
}

static void Check(int condition, const char *name)
{
	if ( condition == 0 )
	{
		fprintf(stderr,"FAIL %s\n",name);
		failures++;
	}
}

static int g_restore_calls;
static int g_evict_calls;
static int g_submit_calls;
static int g_progress_calls;
static int g_fail_submit_at;
static uint32_t g_last_submit_operation;
static uint32_t g_last_evicted_word;

static uint32_t KvFrameWord(uint64_t sequence_id, uint32_t logical)
{
	return((uint32_t)(sequence_id * 1000u + logical));
}

static SparkStatus FakeBuildRestoreBatch(const LmKvFramePlanConfig *configuration, const LmKvFramePendingLane *pending_lanes, uint32_t pending_lane_count, const uint32_t *packet_lane_counts, uint32_t packet_count, void *block_staging, uint32_t block_staging_record_capacity, void *gdn_staging, uint32_t gdn_staging_record_capacity, SparkKvStoreBlock *blocks, uint32_t block_capacity, uint32_t *block_count, uint32_t *lanes_built)
{
	uint32_t index;
	(void)configuration;
	(void)packet_lane_counts;
	(void)packet_count;
	(void)block_staging_record_capacity;
	(void)gdn_staging;
	(void)gdn_staging_record_capacity;
	(void)blocks;
	(void)block_capacity;
	g_restore_calls++;
	for (index = 0u; index < pending_lane_count; index++)
		((uint32_t *)block_staging)[index] = KvFrameWord(pending_lanes[index].sequence_id,pending_lanes[index].nonresident_blocks[0]);
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
	(void)gdn_staging;
	(void)blocks;
	(void)block_capacity;
	g_evict_calls++;
	g_last_evicted_word = ((const uint32_t *)block_staging)[0];
	*block_count = 1u;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeSubmit(SparkStageKvClient *client, LmKvFrameBatchState *batch_state, uint32_t operation, const SparkKvStoreBlock *blocks, uint32_t block_count, uint32_t priority)
{
	(void)client;
	(void)blocks;
	(void)block_count;
	(void)priority;
	g_submit_calls++;
	g_last_submit_operation = operation;
	if ( g_fail_submit_at != 0 && g_submit_calls >= g_fail_submit_at )
		return(SPARK_STATUS_IO_ERROR);
	batch_state->state = SPARK_LLM_KV_BATCH_SUBMITTED;
	batch_state->status = SPARK_STATUS_OK;
	return(SPARK_STATUS_OK);
}

static SparkStatus FakeProgress(SparkStageKvClient *client, LmKvFrameWorkState *work)
{
	(void)client;
	g_progress_calls++;
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

static const LmKvFrameOps g_ops =
{
	"test_stage",
	FakeBuildRestoreBatch,
	FakeBuildEvictBatch,
	FakeSubmit,
	FakeProgress,
	FakeAcknowledge
};

static void MakeStateSized(LmKvFrameState *state, uint32_t block_count, uint32_t lanes, uint32_t stride)
{
	uint32_t index;
	memset(state,0,sizeof(*state));
	state->ops = g_ops;
	state->block_count = block_count;
	state->plan.block_record_bytes = 4u;
	state->logical_to_slot = (uint32_t *)calloc((size_t)lanes * stride,sizeof(uint32_t));
	state->logical_to_slot_capacity = (uint64_t)lanes * stride;
	state->logical_stride = stride;
	state->table_indices_device = (uint32_t *)calloc((size_t)lanes * stride,sizeof(uint32_t));
	state->table_counts_device = (uint32_t *)calloc(lanes,sizeof(uint32_t));
	state->table_indices_host = (uint32_t *)calloc((size_t)lanes * stride,sizeof(uint32_t));
	state->slot_lane = (uint32_t *)malloc(block_count * sizeof(uint32_t));
	state->slot_logical = (uint32_t *)malloc(block_count * sizeof(uint32_t));
	state->slot_sequence = (uint64_t *)calloc(block_count,sizeof(uint64_t));
	state->slot_dirty = (uint8_t *)calloc(block_count,sizeof(uint8_t));
	state->slot_pinned = (uint8_t *)calloc(block_count,sizeof(uint8_t));
	state->slot_free_stack = (uint32_t *)malloc(block_count * sizeof(uint32_t));
	state->block_staging = calloc(SPARK_LLM_KV_STAGING_RECORDS,4u);
	state->gdn_staging = malloc(64u);
	state->cache_bf16 = calloc(block_count,4u);
	for (index = 0u; index < block_count; index++)
	{
		state->slot_free_stack[index] = index;
		state->slot_lane[index] = UINT32_MAX;
		state->slot_logical[index] = UINT32_MAX;
	}
	state->slot_free_count = block_count;
	state->evict_cursor = 0u;
	state->tier_active = 1u;
}

static void MakeState(LmKvFrameState *state, uint32_t block_count)
{
	MakeStateSized(state,block_count,2u,4u);
}

static void FreeState(LmKvFrameState *state)
{
	free(state->logical_to_slot);
	free(state->table_indices_device);
	free(state->table_counts_device);
	free(state->table_indices_host);
	free(state->slot_lane);
	free(state->slot_logical);
	free(state->slot_sequence);
	free(state->slot_dirty);
	free(state->slot_pinned);
	free(state->slot_free_stack);
	free(state->block_staging);
	free(state->gdn_staging);
	free(state->cache_bf16);
}

static void FillSlotView(LmKvFrameSlot *view, uint32_t *lane_indices, uint64_t *positions, uint32_t *contexts, uint32_t *host_mapping, uint32_t *device_mapping)
{
	view->cuda_stream = 0;
	view->host_row_lane_indices = lane_indices;
	view->host_row_positions = positions;
	view->host_context_lengths = contexts;
	view->host_slot_mapping = host_mapping;
	view->slot_mapping = device_mapping;
}

static void FillTable(LmKvFrameTable *table, uint32_t lanes, uint32_t stride, const uint32_t *host_blocks, const uint32_t *host_counts, const uint32_t *stale_device)
{
	table->lane_count = lanes;
	table->lane_stride = stride;
	table->host_physical_block_indices = host_blocks;
	table->host_lane_physical_block_counts = host_counts;
	table->physical_block_indices = stale_device;
	table->lane_physical_block_counts = host_counts;
}

static int KvFrameResidencyConsistent(const LmKvFrameState *state, uint32_t stride)
{
	uint8_t *seen = (uint8_t *)calloc(state->block_count,1u);
	uint64_t index;
	uint32_t slot,resident = 0u;
	int consistent = 1;
	for (index = 0u; index < state->logical_to_slot_capacity; index++)
	{
		slot = state->logical_to_slot[index];
		if ( slot == 0u )
			continue;
		resident++;
		if ( slot > state->block_count || seen[slot - 1u] != 0u || state->slot_lane[slot - 1u] != index / stride || state->slot_logical[slot - 1u] != index % stride || ((const uint32_t *)state->cache_bf16)[slot - 1u] != KvFrameWord(state->slot_sequence[slot - 1u],(uint32_t)(index % stride)) )
			consistent = 0;
		else
			seen[slot - 1u] = 1u;
	}
	if ( resident + state->slot_free_count != state->block_count )
		consistent = 0;
	free(seen);
	return(consistent);
}

static void TestLayerPartition(void)
{
	uint32_t layer,gdn_total = 0u,full_total = 0u;
	for (layer = 0u; layer < SPARK_LLM_LAYER_COUNT; layer++)
	{
		if ( SPARK_LLM_LAYER_IS_GDN(layer) != 0u )
			gdn_total++;
		else
			full_total++;
	}
	Check(SPARK_LLM_GDN_LAYER_COUNT + SPARK_LLM_FULL_ATTENTION_LAYER_COUNT == SPARK_LLM_LAYER_COUNT,"SPARK_LLM_LAYER_COUNT partition");
	Check(gdn_total == SPARK_LLM_GDN_LAYER_COUNT,"SPARK_LLM_LAYER_IS_GDN count");
	Check(full_total == SPARK_LLM_FULL_ATTENTION_LAYER_COUNT,"SPARK_LLM_FULL_ATTENTION_PHASE count");
}

static void TestTpShardVectors(void)
{
	const uint32_t degrees[] = {1u,2u,4u,8u,16u};
	uint32_t index,degree,rank;
	for (index = 0u; index < sizeof(degrees) / sizeof(degrees[0]); index++)
	{
		degree = degrees[index];
		Check(SPARK_LLM_ATTN_KV_SHARD_COUNT(degree) * SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT(degree) == SPARK_LLM_KV_HEAD_COUNT,"SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT shard coverage");
		for (rank = 0u; rank < degree; rank++)
		{
			uint32_t base = SPARK_LLM_ATTN_RANK_KV_HEAD_BASE(degree,rank);
			Check(base < SPARK_LLM_ATTN_KV_HEAD_COUNT,"SPARK_LLM_ATTN_RANK_KV_HEAD_BASE bound");
			Check(base + SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT(degree) <= SPARK_LLM_ATTN_KV_HEAD_COUNT,"SPARK_LLM_ATTN_RANK_KV_HEAD_BASE span");
		}
		Check(SPARK_LLM_ATTN_LOCAL_KV_DIMENSION(degree) == SPARK_LLM_ATTN_LOCAL_KV_HEAD_COUNT(degree) * SPARK_LLM_HEAD_DIMENSION,"SPARK_LLM_ATTN_LOCAL_KV_DIMENSION");
	}
	Check(SPARK_LLM_ATTN_QUERY_HEAD_COUNT % SPARK_LLM_ATTN_HEADS_PER_CTA == 0u,"SPARK_LLM_ATTN_HEADS_PER_CTA query divisibility");
	Check((SPARK_LLM_ATTN_QUERY_HEAD_COUNT / SPARK_LLM_KV_HEAD_COUNT) % SPARK_LLM_ATTN_HEADS_PER_CTA == 0u,"SPARK_LLM_ATTN_HEADS_PER_CTA kv ownership");
}

static float ReferenceRopeFrequency(float pair,float dimension,float theta)
{
	return(exp2f(-(pair / dimension) * log2f(theta)));
}

static void TestRopeTable(void)
{
	const uint32_t pairs = SPARK_LLM_ROPE_DIMENSION / 2u;
	uint32_t pair;
	float previous = 1.0f;
	for (pair = 0u; pair < pairs; pair++)
	{
		float frequency = exp2f(-((float)(2u * pair) / (float)SPARK_LLM_ROPE_DIMENSION) * log2f((float)SPARK_LLM_ROPE_THETA));
		float reference = ReferenceRopeFrequency((float)(2u * pair),(float)SPARK_LLM_ROPE_DIMENSION,(float)SPARK_LLM_ROPE_THETA);
		Check(fabsf(frequency - reference) < 1.0e-6f,"SPARK_LLM_ROPE_THETA frequency law");
		Check(frequency <= previous + 1.0e-9f,"rope frequency monotone decreasing");
		previous = frequency;
	}
	Check(previous < 1.0f,"rope highest pair decays");
}

static void TestSharedMemoryBudgets(void)
{
	const uint32_t state_elements = SPARK_LLM_GDN_HEAD_KEY_DIMENSION * SPARK_LLM_GDN_HEAD_VALUE_DIMENSION;
	const uint32_t decode_bytes = state_elements * (uint32_t)sizeof(float);
	const uint32_t qk_bytes = 2u * SPARK_LLM_GDN_CHUNK_TOKENS * SPARK_LLM_GDN_HEAD_KEY_DIMENSION * (uint32_t)sizeof(float);
	const uint32_t chunk_bytes = (state_elements + (SPARK_LLM_GDN_CHUNK_TOKENS * SPARK_LLM_GDN_HEAD_VALUE_DIMENSION) + (2u * SPARK_LLM_GDN_CHUNK_TOKENS)) * (uint32_t)sizeof(float);
	Check(decode_bytes == 65536u,"SPARK_LLM_GDN_HEAD_KEY_DIMENSION decode shared budget");
	Check(qk_bytes == 65536u,"SPARK_LLM_GDN_CHUNK_TOKENS qk shared budget");
	Check(chunk_bytes == 98816u,"sm_121a chunk shared budget (SPARK_LLM_GDN_CHUNK_TOKENS x SPARK_LLM_GDN_HEAD_VALUE_DIMENSION)");
	Check(decode_bytes <= 99u * 1024u && chunk_bytes <= 99u * 1024u,"sm_121a block shared limit");
}

static void TestRouterCapacity(void)
{
	Check(SPARK_LLM_ROUTED_EXPERT_COUNT <= SPARK_LLM_ROUTER_SORT_CAPACITY,"SPARK_LLM_ROUTER_SORT_CAPACITY covers SPARK_LLM_ROUTED_EXPERT_COUNT");
	Check(SPARK_LLM_EXPERTS_PER_TOKEN <= SPARK_LLM_ROUTED_EXPERT_COUNT,"SPARK_LLM_EXPERTS_PER_TOKEN bound");
}

static void TestKvFrameRestore(void)
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
	uint32_t host_blocks[8] = {0u,0u,0u,0u,0u,0u,0u,0u};
	uint32_t host_counts[2] = {2u,1u};
	uint32_t stale_device[8] = {0u};
	SparkStatus status;
	uint32_t expected_row0,expected_row1;

	MakeState(&state,8u);
	g_restore_calls = 0;
	g_submit_calls = 0;
	g_progress_calls = 0;
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	table.lane_count = 2u;
	table.lane_stride = 4u;
	table.host_physical_block_indices = host_blocks;
	table.host_lane_physical_block_counts = host_counts;
	table.physical_block_indices = stale_device;
	table.lane_physical_block_counts = host_counts;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_OK,"kv_frame_prepare status");
	Check(g_submit_calls == 1,"kv_frame_restore one batch per staging packet");
	Check(g_last_submit_operation == SPARK_KV_STORE_OPERATION_GET,"kv_frame_restore operation");
	expected_row0 = 7u * SPARK_LLM_KV_BLOCK_TOKENS + (uint32_t)(positions[0] % SPARK_LLM_KV_BLOCK_TOKENS);
	expected_row1 = 5u * SPARK_LLM_KV_BLOCK_TOKENS + (uint32_t)(positions[1] % SPARK_LLM_KV_BLOCK_TOKENS);
	Check(host_mapping[0] == expected_row0,"kv_frame slot mapping row0 (SPARK_LLM_KV_BLOCK_TOKENS)");
	Check(host_mapping[1] == expected_row1,"kv_frame slot mapping row1 (SPARK_LLM_KV_BLOCK_TOKENS)");
	Check(table.physical_block_indices == state.table_indices_device,"kv_frame table device bind");
	Check(table.lane_physical_block_counts == state.table_counts_device,"kv_frame table counts bind");
	Check(state.slot_pinned[7u] == 0u && state.slot_pinned[6u] == 0u && state.slot_pinned[5u] == 0u,"kv_frame pins released");
	Check(state.slot_free_count == 5u,"kv_frame free stack drained");

	g_submit_calls = 0;
	g_restore_calls = 0;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_OK,"kv_frame second prepare status");
	Check(g_submit_calls == 0,"kv_frame resident blocks skip restore");
	Check(host_mapping[0] == expected_row0 && host_mapping[1] == expected_row1,"kv_frame resident mapping stable");

	LmKvFrameMarkWritten(&state,&view,2u);
	Check(state.slot_dirty[7u] == 1u && state.slot_dirty[5u] == 1u && state.slot_dirty[6u] == 0u,"kv_frame mark written blocks (SPARK_LLM_KV_BLOCK_TOKENS)");
	host_mapping[0] = 0u;
	host_mapping[1] = 0u;
	view.slot_mapping = 0;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_OK,"kv_frame prepare without a device slot mapping");
	Check(host_mapping[0] == expected_row0 && host_mapping[1] == expected_row1,"kv_frame host mapping without a device slot mapping");
	FreeState(&state);
}

static void TestKvFrameUnwind(void)
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

	MakeState(&state,8u);
	g_fail_submit_at = 1;
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	table.lane_count = 2u;
	table.lane_stride = 4u;
	table.host_physical_block_indices = host_blocks;
	table.host_lane_physical_block_counts = host_counts;
	table.physical_block_indices = stale_device;
	table.lane_physical_block_counts = host_counts;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_IO_ERROR,"kv_frame submit failure propagates");
	Check(state.slot_free_count == 8u,"kv_frame unwind returns uncommitted slots");
	Check(state.logical_to_slot[0] == 0u && state.logical_to_slot[4] == 0u,"kv_frame unwind clears residency");
	g_fail_submit_at = 0;
	FreeState(&state);
}

static void TestKvFrameNegativeGeometry(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[2] = {0u,1u};
	uint64_t positions[2] = {0u,0u};
	uint32_t contexts[2] = {1u,1u};
	uint64_t sequence_ids[2] = {101u,102u};
	uint32_t host_mapping[2] = {0u,0u};
	uint32_t device_mapping[2] = {0u,0u};
	uint32_t host_blocks[8] = {0u};
	uint32_t host_counts[2] = {1u,1u};
	uint32_t stale_device[8] = {0u};
	SparkStatus status;

	MakeState(&state,8u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	table.lane_count = 2u;
	table.lane_stride = SPARK_LLM_KV_MAX_BLOCKS_PER_LANE + 1u;
	table.host_physical_block_indices = host_blocks;
	table.host_lane_physical_block_counts = host_counts;
	table.physical_block_indices = stale_device;
	table.lane_physical_block_counts = host_counts;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_INVALID_ARGUMENT,"SPARK_LLM_KV_MAX_BLOCKS_PER_LANE guard");

	table.lane_stride = 4u;
	lane_indices[0] = 5u;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_INVALID_ARGUMENT,"kv_frame lane bound guard");
	lane_indices[0] = 0u;
	contexts[0] = 5u * SPARK_LLM_KV_BLOCK_TOKENS;
	status = LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,2u);
	Check(status == SPARK_STATUS_INVALID_ARGUMENT,"kv_frame context exceeds lane stride guard");
	FreeState(&state);
}

static void TestKvFrameMultiBatch(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[1] = {0u},contexts[1] = {21u * SPARK_LLM_KV_BLOCK_TOKENS},host_mapping[1] = {0u},device_mapping[1] = {0u},host_counts[1] = {21u};
	uint32_t host_blocks[32],stale_device[32] = {0u},logical;
	uint64_t positions[1] = {21u * SPARK_LLM_KV_BLOCK_TOKENS - 1u},sequence_ids[1] = {301u};
	int placed = 1;
	for (logical = 0u; logical < 32u; logical++)
		host_blocks[logical] = 1000u + logical;
	MakeStateSized(&state,32u,1u,32u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,1u,32u,host_blocks,host_counts,stale_device);
	g_submit_calls = 0;
	g_restore_calls = 0;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK,"kv_frame multi-batch prepare status");
	Check(g_restore_calls == 2 && g_submit_calls == 2,"kv_frame restores 21 blocks in two staging batches");
	Check(state.slot_free_count == 11u,"kv_frame multi-batch claims 21 slots");
	for (logical = 0u; logical < 21u; logical++)
		placed &= state.logical_to_slot[logical] == 32u - logical && state.table_indices_host[logical] == 31u - logical && state.table_indices_device[logical] == 31u - logical;
	Check(placed,"kv_frame multi-batch table maps each restored block to its slot");
	Check(state.table_indices_host[21] == 1021u && state.table_indices_device[31] == 1031u,"kv_frame table keeps the host entries past the frame's blocks");
	Check(KvFrameResidencyConsistent(&state,32u),"kv_frame multi-batch residency and data consistent");
	Check(host_mapping[0] == 11u * SPARK_LLM_KV_BLOCK_TOKENS + SPARK_LLM_KV_BLOCK_TOKENS - 1u && device_mapping[0] == host_mapping[0],"kv_frame multi-batch row mapping");
	FreeState(&state);
}

static void TestKvFrameLaneOrder(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[3] = {1u,0u,1u},contexts[3] = {SPARK_LLM_KV_BLOCK_TOKENS,2u * SPARK_LLM_KV_BLOCK_TOKENS,3u * SPARK_LLM_KV_BLOCK_TOKENS},host_mapping[3] = {0u},device_mapping[3] = {0u};
	uint32_t host_blocks[8] = {0u},host_counts[2] = {2u,3u},stale_device[8] = {0u};
	uint64_t positions[3] = {SPARK_LLM_KV_BLOCK_TOKENS - 1u,2u * SPARK_LLM_KV_BLOCK_TOKENS - 1u,3u * SPARK_LLM_KV_BLOCK_TOKENS - 1u},sequence_ids[3] = {601u,602u,601u};
	MakeStateSized(&state,8u,2u,4u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,2u,4u,host_blocks,host_counts,stale_device);
	g_submit_calls = 0;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,3u) == SPARK_STATUS_OK && g_submit_calls == 1,"kv_frame restores two lanes in one batch");
	Check(state.logical_to_slot[4] == 8u && state.logical_to_slot[5] == 7u && state.logical_to_slot[6] == 6u && state.logical_to_slot[0] == 5u && state.logical_to_slot[1] == 4u,"kv_frame restores lanes in first-row order, each to its longest context");
	Check(host_mapping[0] == 7u * SPARK_LLM_KV_BLOCK_TOKENS + SPARK_LLM_KV_BLOCK_TOKENS - 1u && host_mapping[1] == 3u * SPARK_LLM_KV_BLOCK_TOKENS + SPARK_LLM_KV_BLOCK_TOKENS - 1u && host_mapping[2] == 5u * SPARK_LLM_KV_BLOCK_TOKENS + SPARK_LLM_KV_BLOCK_TOKENS - 1u,"kv_frame maps rows that share a lane");
	Check(state.slot_sequence[7] == 601u && state.slot_sequence[4] == 602u,"kv_frame tags slots with the lane's sequence");
	Check(KvFrameResidencyConsistent(&state,4u),"kv_frame lane order residency consistent");
	FreeState(&state);
}

static void TestKvFrameEvictDirty(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[1] = {0u},contexts[1] = {2u * SPARK_LLM_KV_BLOCK_TOKENS},host_mapping[1] = {0u},device_mapping[1] = {0u};
	uint32_t host_blocks[8] = {0u},host_counts[2] = {2u,1u},stale_device[8] = {0u};
	uint64_t positions[1] = {2u * SPARK_LLM_KV_BLOCK_TOKENS - 1u},sequence_ids[1] = {401u};
	MakeStateSized(&state,2u,2u,4u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,2u,4u,host_blocks,host_counts,stale_device);
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK && state.slot_free_count == 0u,"kv_frame fills the pool");
	LmKvFrameMarkWritten(&state,&view,1u);
	lane_indices[0] = 1u;
	contexts[0] = 1u;
	positions[0] = 0u;
	sequence_ids[0] = 402u;
	g_evict_calls = 0;
	g_submit_calls = 0;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK,"kv_frame evicts for a new lane");
	Check(g_evict_calls == 1 && g_submit_calls == 2 && g_last_submit_operation == SPARK_KV_STORE_OPERATION_GET,"kv_frame writes back the dirty block before restoring");
	Check(g_last_evicted_word == KvFrameWord(401u,1u),"kv_frame writes back the evicted block's data");
	Check(state.logical_to_slot[1] == 0u && state.logical_to_slot[0] == 2u && state.logical_to_slot[4] == 1u,"kv_frame eviction moves the slot to the new lane");
	Check(state.slot_dirty[0] == 0u && state.slot_lane[0] == 1u && state.slot_sequence[0] == 402u,"kv_frame restored slot metadata");
	Check(KvFrameResidencyConsistent(&state,4u),"kv_frame residency consistent after eviction");
	FreeState(&state);
}

static void TestKvFrameTwoEvictions(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[1] = {0u},contexts[1] = {3u * SPARK_LLM_KV_BLOCK_TOKENS},host_mapping[1] = {0u},device_mapping[1] = {0u};
	uint32_t host_blocks[8] = {0u},host_counts[2] = {3u,3u},stale_device[8] = {0u};
	uint64_t positions[1] = {3u * SPARK_LLM_KV_BLOCK_TOKENS - 1u},sequence_ids[1] = {501u};
	MakeStateSized(&state,4u,2u,4u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,2u,4u,host_blocks,host_counts,stale_device);
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK && state.slot_free_count == 1u,"kv_frame holds three blocks of the first lane");
	lane_indices[0] = 1u;
	sequence_ids[0] = 502u;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK,"kv_frame evicts twice in one frame");
	Check(state.logical_to_slot[4] != 0u && state.logical_to_slot[5] != 0u && state.logical_to_slot[6] != 0u && state.logical_to_slot[4] != state.logical_to_slot[5] && state.logical_to_slot[5] != state.logical_to_slot[6] && state.logical_to_slot[4] != state.logical_to_slot[6],"kv_frame gives each block of a frame its own slot");
	Check(state.table_indices_host[4] < 4u && state.table_indices_host[5] < 4u && state.table_indices_host[6] < 4u,"kv_frame table holds valid slots after two evictions");
	Check(KvFrameResidencyConsistent(&state,4u),"kv_frame residency and data consistent after two evictions");
	FreeState(&state);
}

static void TestKvFrameOverDemand(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[1] = {0u},contexts[1] = {3u * SPARK_LLM_KV_BLOCK_TOKENS},host_mapping[1] = {0u},device_mapping[1] = {0u};
	uint32_t host_blocks[4] = {0u},host_counts[1] = {3u},stale_device[4] = {0u};
	uint64_t positions[1] = {3u * SPARK_LLM_KV_BLOCK_TOKENS - 1u},sequence_ids[1] = {701u};
	MakeStateSized(&state,2u,1u,4u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,1u,4u,host_blocks,host_counts,stale_device);
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_CAPACITY_EXCEEDED,"kv_frame refuses a frame that needs more blocks than the pool");
	Check(state.slot_free_count == 2u && state.logical_to_slot[0] == 0u && state.logical_to_slot[1] == 0u && state.logical_to_slot[2] == 0u,"kv_frame over-demand returns every claimed slot");
	Check(state.slot_pinned[0] == 0u && state.slot_pinned[1] == 0u,"kv_frame over-demand releases its pins");
	FreeState(&state);
}

static void TestKvFrameEvictionOrder(void)
{
	LmKvFrameState state;
	LmKvFrameSlot view;
	LmKvFrameTable table;
	uint32_t lane_indices[3] = {0u,1u,2u},contexts[3] = {1u,1u,1u},host_mapping[3] = {0u},device_mapping[3] = {0u};
	uint32_t host_blocks[10] = {0u},host_counts[5] = {1u,1u,1u,1u,1u},stale_device[10] = {0u};
	uint64_t positions[3] = {0u,0u,0u},sequence_ids[3] = {801u,802u,803u};
	MakeStateSized(&state,3u,5u,2u);
	FillSlotView(&view,lane_indices,positions,contexts,host_mapping,device_mapping);
	FillTable(&table,5u,2u,host_blocks,host_counts,stale_device);
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,3u) == SPARK_STATUS_OK && state.slot_free_count == 0u,"kv_frame fills the pool with three lanes");
	lane_indices[0] = 3u;
	sequence_ids[0] = 804u;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK && state.logical_to_slot[6] != 0u,"kv_frame restores a fourth lane");
	lane_indices[0] = 4u;
	sequence_ids[0] = 805u;
	Check(LmKvFramePrepareFrame(&state,&view,sequence_ids,&table,1u) == SPARK_STATUS_OK && state.logical_to_slot[8] != 0u,"kv_frame restores a fifth lane");
	Check(state.logical_to_slot[6] != 0u,"kv_frame does not evict the block the previous frame restored");
	Check(KvFrameResidencyConsistent(&state,2u),"kv_frame eviction order residency consistent");
	FreeState(&state);
}

int LlmModuleNegativeControlRun(void);

#define SPARK_FAMILY_CAMEL KvProbe
#define SPARK_FAMILY_UPPER KV_PROBE
#define SPARK_FAMILY_LOWER kv_probe
#include "sparkpipe/family/spark_family.h"

typedef struct SparkKvProbeModuleState
{
	LmKvFrameState kv;
} SparkKvProbeModuleState;

typedef struct SparkKvProbeModuleSlot
{
	void *cuda_stream;
	uint32_t *host_row_lane_indices;
	uint64_t *host_row_positions;
	uint32_t *host_context_lengths;
	uint32_t *host_slot_mapping;
	uint32_t *slot_mapping;
} SparkKvProbeModuleSlot;

typedef struct SparkKvProbeDecodeBatchView
{
	const uint64_t *row_sequence_ids;
} SparkKvProbeDecodeBatchView;

typedef struct SparkKvProbeResidentDecodeStageFrameContext
{
	const SparkKvProbeDecodeBatchView *decode_batch;
} SparkKvProbeResidentDecodeStageFrameContext;

typedef struct SparkKvProbeKvBlockTableView
{
	uint32_t lane_count;
	uint32_t lane_stride;
	const uint32_t *host_physical_block_indices;
	const uint32_t *host_lane_physical_block_counts;
	const uint32_t *physical_block_indices;
	const uint32_t *lane_physical_block_counts;
} SparkKvProbeKvBlockTableView;

#define SPARK_KV_PROBE_MODULE_KV_DEVICE_SLOT_MAPPING(slot) ((slot)->slot_mapping)
#include "sparkpipe/family/module/spark_module_kv_prepare_frame.h"

static void TestKvFrameTemplateTierOff(void)
{
	SparkKvProbeModuleState state;
	SparkKvProbeModuleSlot slot;
	SparkKvProbeResidentDecodeStageFrameContext prefill_context;
	SparkKvProbeKvBlockTableView table;
	memset(&state,0,sizeof(state));
	memset(&slot,0,sizeof(slot));
	memset(&prefill_context,0,sizeof(prefill_context));
	memset(&table,0,sizeof(table));
	Check(SparkKvProbeModuleKvPrepareFrame(&state,&slot,&prefill_context,&table,1u) == SPARK_STATUS_OK,"kv_frame template leaves a prefill frame alone when the tier is off");
	Check(SparkKvProbeModuleKvPrepareFrame(&state,&slot,0,&table,1u) == SPARK_STATUS_OK,"kv_frame template leaves a frame without context alone when the tier is off");
	SparkKvProbeModuleKvMarkWritten(&state,&slot,1u);
	state.kv.tier_active = 1u;
	Check(SparkKvProbeModuleKvPrepareFrame(&state,&slot,&prefill_context,&table,1u) == SPARK_STATUS_INVALID_ARGUMENT,"kv_frame template refuses a frame without a decode batch when the tier is on");
}

int main(void)
{
	TestLayerPartition();
	TestTpShardVectors();
	TestRopeTable();
	TestSharedMemoryBudgets();
	TestRouterCapacity();
	TestKvFrameRestore();
	TestKvFrameUnwind();
	TestKvFrameNegativeGeometry();
	TestKvFrameMultiBatch();
	TestKvFrameLaneOrder();
	TestKvFrameEvictDirty();
	TestKvFrameTwoEvictions();
	TestKvFrameOverDemand();
	TestKvFrameEvictionOrder();
	TestKvFrameTemplateTierOff();
	failures += LlmModuleNegativeControlRun();
	if ( failures == 0 )
		printf("test_llm_module_contract PASS\n");
	else
		printf("test_llm_module_contract FAIL %d\n",failures);
	return(failures == 0 ? 0 : 1);
}
