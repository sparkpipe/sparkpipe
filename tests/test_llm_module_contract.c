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
	g_restore_calls++;
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
	g_evict_calls++;
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

static void MakeState(LmKvFrameState *state, uint32_t block_count)
{
	uint32_t index;
	memset(state,0,sizeof(*state));
	state->ops = g_ops;
	state->block_count = block_count;
	state->logical_to_slot = (uint32_t *)calloc(2u * 4u,sizeof(uint32_t));
	state->logical_to_slot_capacity = 2u * 4u;
	state->logical_stride = 4u;
	state->table_indices_device = (uint32_t *)malloc(8u * sizeof(uint32_t));
	state->table_counts_device = (uint32_t *)malloc(8u * sizeof(uint32_t));
	state->table_indices_host = (uint32_t *)calloc(2u * 4u,sizeof(uint32_t));
	state->slot_lane = (uint32_t *)calloc(block_count,sizeof(uint32_t));
	state->slot_logical = (uint32_t *)calloc(block_count,sizeof(uint32_t));
	state->slot_sequence = (uint64_t *)calloc(block_count,sizeof(uint64_t));
	state->slot_dirty = (uint8_t *)calloc(block_count,sizeof(uint8_t));
	state->slot_pinned = (uint8_t *)calloc(block_count,sizeof(uint8_t));
	state->slot_free_stack = (uint32_t *)malloc(block_count * sizeof(uint32_t));
	state->block_staging = malloc(64u);
	state->gdn_staging = malloc(64u);
	state->cache_bf16 = malloc(4096u);
	for (index = 0u; index < block_count; index++)
		state->slot_free_stack[index] = index;
	state->slot_free_count = block_count;
	state->evict_cursor = 0u;
	state->tier_active = 1u;
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

int LlmModuleNegativeControlRun(void);

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
	failures += LlmModuleNegativeControlRun();
	if ( failures == 0 )
		printf("test_llm_module_contract PASS\n");
	else
		printf("test_llm_module_contract FAIL %d\n",failures);
	return(failures == 0 ? 0 : 1);
}
