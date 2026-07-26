#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_glm52_cuda_resident_ipc.h"
#include "sparkpipe/spark_glm52_pp13_work_control.h"

#define SPARK_TEST_KV_LANE_CAPACITY 16u
#define SPARK_TEST_KV_LANE_STRIDE 4096u
#define SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY 1024u
#define SPARK_TEST_KV_DIRECTORY_CAPACITY 2048u

typedef struct SparkTestWorkControlKvStorage
{
	uint32_t physical_blocks[
		SPARK_TEST_KV_LANE_CAPACITY * SPARK_TEST_KV_LANE_STRIDE];
	uint32_t lane_counts[SPARK_TEST_KV_LANE_CAPACITY];
	uint8_t block_states[SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY];
	SparkGlm52Pp13KvKey block_keys[SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY];
	uint64_t block_last_used_epochs[SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY];
	uint32_t block_pin_counts[SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY];
	SparkGlm52Pp13WorkControlKvDirectoryEntry
		directory_entries[SPARK_TEST_KV_DIRECTORY_CAPACITY];
	SparkGlm52Pp13WorkControlKvBlockEntry
		block_entries[SPARK_TEST_KV_DIRECTORY_CAPACITY];
} SparkTestWorkControlKvStorage;

typedef struct SparkTestWorkControlSwapStorage
{
	uint32_t physical_values[2u];
	uint32_t backing_values[8u];
	uint64_t backing_sequence_ids[8u];
	uint32_t backing_logical_indices[8u];
	uint32_t store_count;
	uint32_t load_count;
} SparkTestWorkControlSwapStorage;

static SparkStatus SparkTestWorkControlSwapStore(
	void *context,
	SparkGlm52Pp13KvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index)
{
	SparkTestWorkControlSwapStorage *storage;
	storage = (SparkTestWorkControlSwapStorage *)context;
	if (storage == 0 || physical_block_index >= 2u ||
		backing_block_index >= 8u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	storage->backing_values[backing_block_index] =
		storage->physical_values[physical_block_index];
	storage->backing_sequence_ids[backing_block_index] = key.low;
	storage->backing_logical_indices[backing_block_index] = (uint32_t)key.high;
	storage->store_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestWorkControlSwapLoad(
	void *context,
	SparkGlm52Pp13KvKey key,
	uint32_t physical_block_index,
	uint32_t backing_block_index)
{
	SparkTestWorkControlSwapStorage *storage;
	storage = (SparkTestWorkControlSwapStorage *)context;
	if (storage == 0 || physical_block_index >= 2u ||
		backing_block_index >= 8u ||
		storage->backing_sequence_ids[backing_block_index] != key.low ||
		storage->backing_logical_indices[backing_block_index] !=
			(uint32_t)key.high)
		return SPARK_STATUS_VALIDATION_FAILED;
	storage->physical_values[physical_block_index] =
		storage->backing_values[backing_block_index];
	storage->load_count += 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkTestInitializeKvState(
	SparkGlm52Pp13WorkControlKvState *state,
	SparkTestWorkControlKvStorage *storage,
	uint32_t lane_capacity)
{
	SparkStatus status;
	memset(storage,0,sizeof(*storage));
	status = SparkGlm52Pp13WorkControlInitializeKvState(
		state,
		lane_capacity,
		SPARK_TEST_KV_LANE_STRIDE,
		256u,
		SPARK_TEST_KV_PHYSICAL_BLOCK_CAPACITY,
		SPARK_TEST_KV_DIRECTORY_CAPACITY,
		SPARK_TEST_KV_DIRECTORY_CAPACITY,
		storage->physical_blocks,
		storage->lane_counts,
		storage->block_states,
		storage->block_keys,
		storage->block_last_used_epochs,
		storage->directory_entries,
		storage->block_entries);
	if (status != SPARK_STATUS_OK)
		return status;
	return SparkGlm52Pp13WorkControlConfigureKvPins(
		state,storage->block_pin_counts);
}

static void SparkTestInitializeWorkPacket(
	SparkGlm52Pp13WorkControlPacket *packet)
{

	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->control_generation =
		SPARK_GLM52_PP13_WORK_CONTROL_STANDALONE_GENERATION;
	packet->active_sequence_count = 4u;
	packet->descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(
			packet->active_sequence_count);
	packet->request_id = 7u;
	packet->sequence_id = 11u;
	packet->sequence_position = 128u;
	packet->new_token_count = 1u;
	packet->pipeline_slot = 0u;
	packet->block_token_count = 256u;
	packet->kv_block_table_token_count = 32769u;
	packet->max_blocks_per_sequence = 4096u;
	packet->lane_count = packet->active_sequence_count;
	packet->rows_per_lane = 1u;
	packet->execution_row_count = packet->lane_count;
	packet->execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
	for (uint32_t lane_index = 0u;
		 lane_index < packet->lane_count;
		 ++lane_index)
	{
		packet->lanes[lane_index].request_id = packet->request_id + lane_index;
		packet->lanes[lane_index].sequence_id = packet->sequence_id + lane_index;
		packet->lanes[lane_index].sequence_position =
			packet->sequence_position + lane_index;
		packet->lanes[lane_index].request_slot_index = lane_index;
		packet->lanes[lane_index].context_token_count =
			packet->kv_block_table_token_count - lane_index;
		packet->lanes[lane_index].input_token_id = packet->input_token_id;
	}
}

static void SparkTestPrefillPacketLanes(
	const SparkGlm52Pp13WorkControlPacket *source_packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView view;
	uint32_t lane_index;

	assert(source_packet != 0);
	for (lane_index = 0u; lane_index < source_packet->lane_count; ++lane_index)
	{
		memset(&packet,0,sizeof(packet));
		packet = *source_packet;
		packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
		packet.active_sequence_count = 1u;
		packet.lane_count = 1u;
		packet.rows_per_lane = 1u;
		packet.execution_row_count = 1u;
		packet.descriptor_bytes =
			SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
		packet.lanes[0u] = source_packet->lanes[lane_index];
		packet.lanes[0u].sequence_position =
			packet.lanes[0u].context_token_count - 1u;
		packet.request_id = packet.lanes[0u].request_id;
		packet.sequence_id = packet.lanes[0u].sequence_id;
		packet.sequence_position = packet.lanes[0u].sequence_position;
		packet.input_token_id = packet.lanes[0u].input_token_id;
		packet.prefill_token_ids[0u] = packet.input_token_id;
		packet.kv_block_table_token_count =
			packet.lanes[0u].context_token_count;
		assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
			&packet,state,&view) == SPARK_STATUS_OK);
		assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
			&packet,state) == SPARK_STATUS_OK);
	}
}

static void SparkTestGlm52Pp13WorkControlPacket(void)
{
	SparkGlm52Pp13WorkControlPacket packet;

	SparkTestInitializeWorkPacket(&packet);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.execution_batch_bucket = 0u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
	packet.new_token_count = 9u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	SparkTestInitializeWorkPacket(&packet);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.execution_row_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.kv_block_table_token_count = packet.sequence_position + 1u;
	packet.lanes[0u].context_token_count =
		packet.kv_block_table_token_count;
	packet.prefill_token_ids[0u] = packet.lanes[0u].input_token_id;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.new_token_count =
		SPARK_GLM52_PP13_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET + 1u;
	packet.rows_per_lane = packet.new_token_count;
	packet.execution_row_count = packet.new_token_count;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.new_token_count =
		SPARK_GLM52_PP13_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET;
	packet.rows_per_lane = packet.new_token_count;
	packet.execution_row_count = packet.new_token_count;
	packet.active_sequence_count = 1025u;
	packet.lane_count = 1025u;
	packet.execution_row_count = 1025u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlExecutionChunks(void)
{
	uint32_t chunk_count;
	uint32_t maximum_lanes_per_chunk;

	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		1024u,1u,1024u,&maximum_lanes_per_chunk,&chunk_count) ==
		SPARK_STATUS_OK);
	assert(maximum_lanes_per_chunk == 1024u);
	assert(chunk_count == 1u);
	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		1024u,7u,1024u,&maximum_lanes_per_chunk,&chunk_count) ==
		SPARK_STATUS_OK);
	assert(maximum_lanes_per_chunk == 146u);
	assert(chunk_count == 8u);
	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		1024u,7u,7168u,&maximum_lanes_per_chunk,&chunk_count) ==
		SPARK_STATUS_OK);
	assert(maximum_lanes_per_chunk == 146u);
	assert(chunk_count == 8u);
	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		SPARK_GLM52_STAGE_PLAN_BUCKET_B1024,
		SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
		SPARK_GLM52_STAGE_PLAN_BUCKET_B1024 *
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT,
		&maximum_lanes_per_chunk,&chunk_count) == SPARK_STATUS_OK);
	assert(maximum_lanes_per_chunk == 170u);
	assert(chunk_count == 7u);
	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		146u,7u,1024u,&maximum_lanes_per_chunk,&chunk_count) ==
		SPARK_STATUS_OK);
	assert(maximum_lanes_per_chunk == 146u);
	assert(chunk_count == 1u);
	assert(SparkGlm52Pp13WorkControlPlanExecutionChunks(
		1u,7u,6u,&maximum_lanes_per_chunk,&chunk_count) ==
		SPARK_STATUS_CAPACITY_EXCEEDED);
}

static void SparkTestGlm52Pp13WorkControlHostBlockTable(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	SparkTestWorkControlKvStorage storage;

	SparkTestInitializeWorkPacket(&packet);
	assert(SparkTestInitializeKvState(&state,&storage,4u) == SPARK_STATUS_OK);
	SparkTestPrefillPacketLanes(&packet,&state);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(view.abi_version == SPARK_GLM52_KV_CACHE_ABI_VERSION);
	assert(view.block_token_count == packet.block_token_count);
	assert(view.lane_count == 4u);
	assert(view.lane_stride == 4096u);
	assert(view.lane_capacity == view.lane_stride);
	assert(view.physical_block_indices == storage.physical_blocks);
	assert(view.host_physical_block_indices == storage.physical_blocks);
	assert(storage.lane_counts[0] == 129u);
	assert(storage.lane_counts[3] == 128u);
	assert(storage.physical_blocks[0] == 0u);
	assert(storage.physical_blocks[128] == 128u);
	assert(storage.physical_blocks[4096] == 129u);
	assert(storage.physical_blocks[(3u * 4096u) + 127u] == 512u);
	assert(state.allocated_physical_block_count == 513u);
}

static void SparkTestGlm52Pp13WorkControlTracksKvReadiness(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	SparkTestWorkControlKvStorage storage;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 2u;
	packet.lane_count = 2u;
	packet.execution_row_count = 2u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(2u);
	assert(SparkTestInitializeKvState(&state,&storage,2u) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_BUSY);
	assert(state.missing_block_count == 1u);
	SparkTestPrefillPacketLanes(&packet,&state);
	assert(storage.block_states[0] == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,
		&state,
		&view) == SPARK_STATUS_OK);
	assert(state.resident_block_count != 0u);
	assert(SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
		&packet,
		&state) == SPARK_STATUS_OK);
	assert(storage.block_states[0] == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
}

static void SparkTestGlm52Pp13WorkControlKeepsStableBlocksAcrossLaneReorder(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	SparkTestWorkControlKvStorage storage;
	SparkGlm52Pp13WorkControlLane lane0;
	uint32_t sequence11_block0;
	uint32_t sequence12_block0;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 2u;
	packet.lane_count = 2u;
	packet.execution_row_count = 2u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(2u);
	assert(SparkTestInitializeKvState(&state,&storage,2u) == SPARK_STATUS_OK);
	SparkTestPrefillPacketLanes(&packet,&state);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	sequence11_block0 = storage.physical_blocks[0u];
	sequence12_block0 = storage.physical_blocks[SPARK_TEST_KV_LANE_STRIDE];
	assert(sequence11_block0 != sequence12_block0);

	lane0 = packet.lanes[0u];
	packet.lanes[0u] = packet.lanes[1u];
	packet.lanes[1u] = lane0;
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	packet.flags = 0u;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(storage.physical_blocks[0u] == sequence12_block0);
	assert(storage.physical_blocks[SPARK_TEST_KV_LANE_STRIDE] ==
		sequence11_block0);
}

static void SparkTestGlm52Pp13WorkControlDsparkVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	packet.speculative_token_count =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	packet.speculative_token_index =
		0u;
	packet.rows_per_lane = packet.speculative_token_count + 1u;
	packet.execution_row_count = packet.rows_per_lane;
	packet.new_token_count = packet.rows_per_lane;
	packet.input_token_id = 101u;
	packet.lanes[0u].input_token_id = packet.input_token_id;
	for (token_index = 0u;
		 token_index < SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 200u + token_index;
	packet.lanes[0u].speculative_token_count = packet.speculative_token_count;
	memcpy(packet.lanes[0u].speculative_draft_token_ids,
		packet.speculative_draft_token_ids,
		sizeof(packet.speculative_draft_token_ids));
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.flags &=
		~SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.flags |=
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	packet.speculative_token_index = 1u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlMtpVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
	packet.speculative_token_count = SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
	packet.speculative_token_index = 0u;
	packet.rows_per_lane = packet.speculative_token_count + 1u;
	packet.execution_row_count = packet.rows_per_lane;
	packet.new_token_count = packet.rows_per_lane;
	packet.input_token_id = 101u;
	packet.lanes[0u].input_token_id = packet.input_token_id;
	for (token_index = 0u;
		 token_index < SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 300u + token_index;
	packet.lanes[0u].speculative_token_count = packet.speculative_token_count;
	memcpy(packet.lanes[0u].speculative_draft_token_ids,
		packet.speculative_draft_token_ids,
		sizeof(packet.speculative_draft_token_ids));
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.flags |= SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
	packet.rows_per_lane = 1u;
	packet.execution_row_count = 1u;
	packet.new_token_count = 1u;
	packet.speculative_token_index = 3u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlB1024MtpBatch(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t lane_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT;
	packet.lane_count = packet.active_sequence_count;
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(packet.lane_count);
	packet.execution_row_count = packet.lane_count;
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT;
	packet.mtp_draft_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	packet.new_token_count = packet.mtp_draft_token_count + 1u;
	for (lane_index = 0u; lane_index < packet.lane_count; ++lane_index)
	{
		packet.lanes[lane_index].request_id = 1000u + lane_index;
		packet.lanes[lane_index].sequence_id = 2000u + lane_index;
		packet.lanes[lane_index].sequence_position = 4096u;
		packet.lanes[lane_index].request_slot_index = lane_index;
		packet.lanes[lane_index].context_token_count =
			4097u + SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
		packet.lanes[lane_index].input_token_id = 100u + (lane_index % 100u);
		packet.lanes[lane_index].mtp_draft_token_count =
			packet.mtp_draft_token_count;
	}
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	packet.kv_block_table_token_count =
		4097u + SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_OK);
	packet.execution_row_count -= 1u;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,1024u,4u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlB1024LayerMajorMtpVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t lane_index;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_TREE_VERIFY;
	packet.active_sequence_count = 170u;
	packet.lane_count = packet.active_sequence_count;
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(packet.lane_count);
	packet.speculative_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	packet.rows_per_lane = SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
	packet.execution_row_count = packet.lane_count * packet.rows_per_lane;
	packet.new_token_count = packet.rows_per_lane;
	packet.kv_block_table_token_count = 4103u;
	packet.mtp_draft_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	for (token_index = 0u;
		 token_index < packet.speculative_token_count;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 300u + token_index;
	for (lane_index = 0u; lane_index < packet.lane_count; ++lane_index)
	{
		SparkGlm52Pp13WorkControlLane *lane;
		lane = &packet.lanes[lane_index];
		lane->request_id = 1000u + lane_index;
		lane->sequence_id = 2000u + lane_index;
		lane->sequence_position = 4096u;
		lane->request_slot_index = lane_index;
		lane->context_token_count = 4103u;
		lane->input_token_id = 100u + (lane_index % 100u);
		lane->mtp_draft_token_count = packet.mtp_draft_token_count;
		lane->speculative_token_count = packet.speculative_token_count;
		memcpy(lane->speculative_draft_token_ids,
			packet.speculative_draft_token_ids,
			sizeof(lane->speculative_draft_token_ids));
	}
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,1020u,4u) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,1019u,4u) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlB1024LayerMajorDsparkVerify(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	uint32_t lane_index;
	uint32_t token_index;

	SparkTestInitializeWorkPacket(&packet);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	packet.active_sequence_count =
		SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT;
	packet.lane_count = packet.active_sequence_count;
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(packet.lane_count);
	packet.speculative_token_count =
		SPARK_GLM52_DSPARK_MAX_SPECULATIVE_TOKEN_COUNT;
	packet.rows_per_lane = packet.speculative_token_count + 1u;
	packet.execution_row_count = packet.lane_count * packet.rows_per_lane;
	packet.new_token_count = packet.rows_per_lane;
	packet.kv_block_table_token_count = 4104u;
	for (token_index = 0u;
		 token_index < packet.speculative_token_count;
		 ++token_index)
		packet.speculative_draft_token_ids[token_index] = 400u + token_index;
	for (lane_index = 0u; lane_index < packet.lane_count; ++lane_index)
	{
		SparkGlm52Pp13WorkControlLane *lane;
		lane = &packet.lanes[lane_index];
		lane->request_id = 3000u + lane_index;
		lane->sequence_id = 4000u + lane_index;
		lane->sequence_position = 4096u;
		lane->request_slot_index = lane_index;
		lane->context_token_count = 4104u;
		lane->input_token_id = 200u + (lane_index % 100u);
		lane->speculative_token_count = packet.speculative_token_count;
		memcpy(lane->speculative_draft_token_ids,
			packet.speculative_draft_token_ids,
			sizeof(lane->speculative_draft_token_ids));
	}
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,8192u,4u) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,8191u,4u) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlCommitsTreePositions(void)
{
	static SparkTestWorkControlKvStorage storage;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView view;
	uint32_t token_index;
	assert(SparkTestInitializeKvState(
		&state,&storage,SPARK_TEST_KV_LANE_CAPACITY) == SPARK_STATUS_OK);
	SparkTestInitializeWorkPacket(&packet);
	packet.flags =
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY |
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_TREE_VERIFY;
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
	packet.speculative_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	packet.rows_per_lane = SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT;
	packet.execution_row_count = packet.rows_per_lane;
	packet.new_token_count = packet.rows_per_lane;
	packet.mtp_draft_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT;
	packet.kv_block_table_token_count =
		SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION + 1u;
	packet.sequence_position = 0u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	packet.lanes[0u].sequence_position = packet.sequence_position;
	packet.lanes[0u].request_slot_index = 0u;
	packet.lanes[0u].context_token_count = packet.kv_block_table_token_count;
	packet.lanes[0u].input_token_id = 101u;
	packet.lanes[0u].mtp_draft_token_count = packet.mtp_draft_token_count;
	packet.lanes[0u].speculative_token_count =
		packet.speculative_token_count;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	for (token_index = 0u;
		 token_index < packet.speculative_token_count;
		 ++token_index)
	{
		packet.speculative_draft_token_ids[token_index] = 300u + token_index;
		packet.lanes[0u].speculative_draft_token_ids[token_index] =
			packet.speculative_draft_token_ids[token_index];
	}
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,SPARK_GLM52_STAGE_PLAN_BUCKET_B16,1u) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
		&packet,&state) == SPARK_STATUS_OK);
	assert(storage.block_states[view.host_physical_block_indices[0u]] ==
		SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
}

static void SparkTestGlm52Pp13WorkControlB1024PhysicalDirectory(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52KvBlockTableView view;
	SparkGlm52Pp13WorkControlKvDirectoryEntry *directory_entries;
	uint32_t *physical_blocks;
	uint32_t *lane_counts;
	uint8_t *block_states;
	SparkGlm52Pp13KvKey *block_keys;
	uint64_t *block_last_used_epochs;
	SparkGlm52Pp13WorkControlKvBlockEntry *block_entries;
	uint32_t lane_index;
	uint32_t first_physical_block;
	uint32_t last_physical_block;
	SparkGlm52Pp13WorkControlLane lane;

	physical_blocks = (uint32_t *)calloc(1024u,sizeof(*physical_blocks));
	lane_counts = (uint32_t *)calloc(1024u,sizeof(*lane_counts));
	block_states = (uint8_t *)calloc(1024u,sizeof(*block_states));
	block_keys =
		(SparkGlm52Pp13KvKey *)calloc(1024u,sizeof(*block_keys));
	block_entries =
		(SparkGlm52Pp13WorkControlKvBlockEntry *)calloc(
			2048u,sizeof(*block_entries));
	block_last_used_epochs =
		(uint64_t *)calloc(1024u,sizeof(*block_last_used_epochs));
	directory_entries =
		(SparkGlm52Pp13WorkControlKvDirectoryEntry *)calloc(
			2048u,sizeof(*directory_entries));
	assert(physical_blocks != 0 && lane_counts != 0 && block_states != 0 &&
		block_keys != 0 && block_entries != 0 &&
		block_last_used_epochs != 0 && directory_entries != 0);
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,1024u,1u,256u,1024u,2048u,2048u,physical_blocks,lane_counts,
		block_states,block_keys,block_last_used_epochs,directory_entries,
		block_entries) == SPARK_STATUS_OK);

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1024u;
	packet.lane_count = 1024u;
	packet.execution_batch_bucket = SPARK_GLM52_STAGE_PLAN_BUCKET_B1024;
	packet.execution_row_count = 1024u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(packet.lane_count);
	packet.block_token_count = 256u;
	packet.kv_block_table_token_count = 1u;
	packet.max_blocks_per_sequence = 1u;
	for (lane_index = 0u; lane_index < packet.lane_count; ++lane_index)
	{
		packet.lanes[lane_index].request_id = 10000u + lane_index;
		packet.lanes[lane_index].sequence_id = 20000u + lane_index;
		packet.lanes[lane_index].sequence_position = 0u;
		packet.lanes[lane_index].request_slot_index = lane_index;
		packet.lanes[lane_index].context_token_count = 1u;
		packet.lanes[lane_index].input_token_id = lane_index;
	}
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	SparkTestPrefillPacketLanes(&packet,&state);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(state.allocated_physical_block_count == 1024u);
	for (lane_index = 0u; lane_index < packet.lane_count; ++lane_index)
	{
		assert(lane_counts[lane_index] == 1u);
		assert(physical_blocks[lane_index] < 1024u);
	}
	assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
		&packet,&state) == SPARK_STATUS_OK);
	first_physical_block = physical_blocks[0u];
	last_physical_block = physical_blocks[1023u];
	lane = packet.lanes[0u];
	packet.lanes[0u] = packet.lanes[1023u];
	packet.lanes[1023u] = lane;
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	packet.input_token_id = packet.lanes[0u].input_token_id;
	packet.flags = 0u;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(physical_blocks[0u] == last_physical_block);
	assert(physical_blocks[1023u] == first_physical_block);

	free(directory_entries);
	free(block_last_used_epochs);
	free(block_states);
	free(lane_counts);
	free(physical_blocks);
	free(block_keys);
	free(block_entries);
}

static void SparkTestGlm52Pp13WorkControlNvmeSwapAndRelease(void)
{
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52Pp13WorkControlPacket prefetch_packets[2u];
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52Pp13WorkControlKvPrefetchEntry prefetch_entries[2u];
	SparkGlm52KvBlockTableView view;
	SparkGlm52Pp13WorkControlKvDirectoryEntry directory_entries[16u];
	SparkGlm52Pp13WorkControlKvBlockEntry block_entries[16u];
	SparkTestWorkControlSwapStorage swap_storage;
	uint32_t physical_blocks[8u];
	uint32_t lane_counts[2u];
	uint8_t block_states[2u];
	SparkGlm52Pp13KvKey block_keys[2u];
	uint64_t block_last_used_epochs[2u];
	uint32_t backing_free_next[8u];
	uint32_t sequence_index;
	uint32_t physical_block_index;
	uint32_t prefetch_entry_count;

	memset(&swap_storage,0,sizeof(swap_storage));
	assert(SparkGlm52Pp13WorkControlInitializeKvState(
		&state,2u,4u,64u,2u,16u,16u,physical_blocks,lane_counts,
		block_states,block_keys,block_last_used_epochs,directory_entries,
		block_entries) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlConfigureKvSwap(
		&state,8u,backing_free_next,SparkTestWorkControlSwapStore,
		SparkTestWorkControlSwapLoad,&swap_storage) == SPARK_STATUS_OK);

	SparkTestInitializeWorkPacket(&packet);
	packet.flags = 0u;
	packet.active_sequence_count = 2u;
	packet.lane_count = 2u;
	packet.execution_row_count = 2u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(2u);
	packet.block_token_count = 64u;
	packet.kv_block_table_token_count = 65u;
	packet.max_blocks_per_sequence = 4u;
	for (sequence_index = 0u; sequence_index < 2u; ++sequence_index)
	{
		packet.lanes[sequence_index].request_id = 50u + sequence_index;
		packet.lanes[sequence_index].sequence_id = 60u + sequence_index;
		packet.lanes[sequence_index].sequence_position = 64u;
		packet.lanes[sequence_index].request_slot_index = sequence_index;
		packet.lanes[sequence_index].context_token_count = 65u;
	}
	packet.request_id = packet.lanes[0u].request_id;
	packet.sequence_id = packet.lanes[0u].sequence_id;
	packet.sequence_position = packet.lanes[0u].sequence_position;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_CAPACITY_EXCEEDED);
	assert(state.directory_entry_count == 0u);
	assert(swap_storage.store_count == 0u);
	assert(swap_storage.load_count == 0u);

	for (sequence_index = 0u; sequence_index < 3u; ++sequence_index)
	{
		SparkTestInitializeWorkPacket(&packet);
		packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
		packet.active_sequence_count = 1u;
		packet.lane_count = 1u;
		packet.execution_row_count = 1u;
		packet.descriptor_bytes =
			SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
		packet.block_token_count = 64u;
		packet.kv_block_table_token_count = 1u;
		packet.max_blocks_per_sequence = 4u;
		packet.request_id = 100u + sequence_index;
		packet.sequence_id = 200u + sequence_index;
		packet.sequence_position = 0u;
		packet.lanes[0u].request_id = packet.request_id;
		packet.lanes[0u].sequence_id = packet.sequence_id;
		packet.lanes[0u].sequence_position = 0u;
		packet.lanes[0u].request_slot_index = sequence_index;
		packet.lanes[0u].context_token_count = 1u;
		assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
			&packet,&state,&view) == SPARK_STATUS_OK);
		physical_block_index = physical_blocks[0u];
		assert(physical_block_index < 2u);
		swap_storage.physical_values[physical_block_index] =
			1000u + sequence_index;
		assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
			&packet,&state) == SPARK_STATUS_OK);
	}
	assert(state.directory_entry_count == 3u);
	assert(state.allocated_physical_block_count == 2u);
	assert(state.swapped_block_count == 1u);
	assert(state.swap_store_count == 1u);
	assert(swap_storage.store_count == 1u);

	SparkTestInitializeWorkPacket(&packet);
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.execution_row_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.block_token_count = 64u;
	packet.kv_block_table_token_count = 1u;
	packet.max_blocks_per_sequence = 4u;
	packet.request_id = 100u;
	packet.sequence_id = 200u;
	packet.sequence_position = 0u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	packet.lanes[0u].sequence_position = 0u;
	packet.lanes[0u].request_slot_index = 0u;
	packet.lanes[0u].context_token_count = 1u;
	prefetch_packets[0u] = packet;
	prefetch_packets[1u] = packet;
	assert(SparkGlm52Pp13WorkControlCollectKvPrefetchEntries(
		prefetch_packets,2u,&state,prefetch_entries,2u,
		&prefetch_entry_count) == SPARK_STATUS_OK);
	assert(prefetch_entry_count == 1u);
	assert(SparkGlm52Pp13WorkControlKvKeyEqual(prefetch_entries[0u].key,
		SparkGlm52Pp13WorkControlPrivateKey(200u,0u)) != 0u);
	assert(prefetch_entries[0u].backing_block_index < 8u);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	physical_block_index = physical_blocks[0u];
	assert(swap_storage.physical_values[physical_block_index] == 1000u);
	assert(state.swap_load_count == 1u);
	assert(swap_storage.load_count == 1u);
	assert(state.swap_store_count == 2u);

	packet.request_id = 101u;
	packet.sequence_id = 201u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	packet.request_id = 102u;
	packet.sequence_id = 202u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(state.swap_load_count == 3u);
	assert(swap_storage.load_count == 3u);
	assert(state.swap_store_count == 3u);
	assert(swap_storage.store_count == 3u);
	assert(state.clean_evict_count == 1u);

	assert(SparkGlm52Pp13WorkControlReleaseSequence(
		&state,200u,1u) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 2u);
	assert(SparkGlm52Pp13WorkControlReleaseSequence(
		&state,200u,1u) == SPARK_STATUS_OK);

	memset(&packet,0,sizeof(packet));
	packet.magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet.abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet.control_generation =
		SPARK_GLM52_PP13_WORK_CONTROL_STANDALONE_GENERATION;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(2u);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES;
	packet.request_id = 101u;
	packet.sequence_id = 201u;
	packet.active_sequence_count = 2u;
	packet.lane_count = 2u;
	packet.block_token_count = 64u;
	packet.kv_block_table_token_count = 1u;
	packet.max_blocks_per_sequence = 4u;
	for (sequence_index = 0u; sequence_index < 2u; ++sequence_index)
	{
		packet.lanes[sequence_index].request_id = 101u + sequence_index;
		packet.lanes[sequence_index].sequence_id = 201u + sequence_index;
		packet.lanes[sequence_index].context_token_count = 1u;
	}
	assert(SparkGlm52Pp13WorkControlValidatePacket(
		&packet,1024u,4u) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlReleasePacketSequences(
		&packet,&state) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 0u);
	assert(state.allocated_physical_block_count == 0u);
	assert(state.swapped_block_count == 0u);
}

static void SparkTestGlm52Pp13WorkControlBuildDecodeBatch(void)
{
	static SparkGlm52RequestApiDispatch request_dispatch;
	static SparkGlm52RequestApiDecodeDispatchView decode_view;
	static SparkGlm52ServingDecodeDispatch decode_dispatch;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView kv_view;
	uint32_t block_indices[4u][2u];
	uint32_t block_counts[4u];
	uint32_t lane_index;

	memset(&request_dispatch,0,sizeof(request_dispatch));
	request_dispatch.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_dispatch.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
	request_dispatch.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	request_dispatch.request_count = 4u;
	request_dispatch.highest_priority = 77u;
	request_dispatch.decode_batch_decision.batch_bucket =
		SPARK_GLM52_STAGE_PLAN_BUCKET_B64;
	memset(&decode_view,0,sizeof(decode_view));
	decode_view.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	decode_view.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DECODE_DISPATCH_VIEW_DESCRIPTOR_BYTES;
	decode_view.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	decode_view.active_sequence_count = 4u;
	decode_view.lane_count = 4u;
	memset(&kv_view,0,sizeof(kv_view));
	kv_view.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_view.descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	kv_view.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	kv_view.lane_count = 4u;
	kv_view.lane_stride = 2u;
	kv_view.lane_capacity = 4u;
	kv_view.physical_block_indices = &block_indices[0u][0u];
	kv_view.lane_physical_block_counts = block_counts;
	kv_view.host_physical_block_indices = &block_indices[0u][0u];
	kv_view.host_lane_physical_block_counts = block_counts;
	memset(&decode_dispatch,0,sizeof(decode_dispatch));
	decode_dispatch.abi_version = SPARK_GLM52_SERVING_ENGINE_ABI_VERSION;
	decode_dispatch.descriptor_bytes =
		SPARK_GLM52_SERVING_DECODE_DISPATCH_DESCRIPTOR_BYTES;
	decode_dispatch.dispatch_kind =
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH;
	decode_dispatch.request_count = 4u;
	decode_dispatch.active_sequence_count = 4u;
	decode_dispatch.request_dispatch = &request_dispatch;
	decode_dispatch.kv_block_table_view = &kv_view;
	decode_dispatch.decode_view = &decode_view;
	for (lane_index = 0u; lane_index < 4u; ++lane_index)
	{
		request_dispatch.request_ids[lane_index] = 100u + lane_index;
		request_dispatch.sequence_ids[lane_index] = 200u + lane_index;
		decode_view.lanes[lane_index].request_index = lane_index;
		decode_view.lanes[lane_index].request_id =
			request_dispatch.request_ids[lane_index];
		decode_view.lanes[lane_index].sequence_id =
			request_dispatch.sequence_ids[lane_index];
		decode_view.lanes[lane_index].sequence_position = 31u;
		decode_view.lanes[lane_index].context_token_count = 32u;
		decode_view.lanes[lane_index].request_slot_index = lane_index;
		decode_dispatch.input_token_ids[lane_index] = 300u + lane_index;
		block_counts[lane_index] = 1u;
		block_indices[lane_index][0u] = 40u + lane_index;
	}
	decode_view.lanes[0u].mtp_resolution_base_position = 29u;
	decode_view.lanes[0u].mtp_resolution_proposed_token_count = 3u;
	decode_view.lanes[0u].mtp_resolution_accepted_token_count = 1u;
	decode_view.lanes[0u].mtp_resolution_committed_token_count = 2u;
	assert(SparkGlm52Pp13WorkControlBuildDecodePacket(
		&decode_dispatch,0u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 4u);
	assert(packet.execution_batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B64);
	assert(packet.descriptor_bytes ==
		SparkGlm52Pp13WorkControlCalculatePacketBytes(4u));
	assert(packet.lanes[3u].request_id == 103u);
	assert(packet.lanes[3u].input_token_id == 303u);
	assert(packet.lanes[3u].request_slot_index == 3u);
	assert((packet.flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE) != 0u);
	assert(packet.lanes[0u].mtp_resolution_proposed_token_count == 3u);
	assert(packet.lanes[0u].mtp_resolution_accepted_token_count == 1u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_OK);
	packet.flags &= ~SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	assert(SparkGlm52Pp13WorkControlBuildDecodePacketRange(
		&decode_dispatch,2u,2u,0u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 2u);
	assert(packet.lanes[0u].request_id == 102u);
	assert(packet.lanes[1u].request_id == 103u);
	assert(packet.descriptor_bytes ==
		SparkGlm52Pp13WorkControlCalculatePacketBytes(2u));
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_OK);
}

static void SparkTestGlm52Pp13WorkControlBuildPackedMtpVerify(void)
{
	static SparkGlm52RequestApiDispatch request_dispatch;
	static SparkGlm52RequestApiDecodeDispatchView decode_view;
	static SparkGlm52ServingDecodeDispatch decode_dispatch;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView kv_view;
	uint32_t token_index;
	memset(&request_dispatch,0,sizeof(request_dispatch));
	request_dispatch.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_dispatch.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
	request_dispatch.kind =
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
	request_dispatch.flags =
		SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY;
	request_dispatch.request_count = 1u;
	request_dispatch.decode_batch_decision.batch_bucket =
		SPARK_GLM52_STAGE_PLAN_BUCKET_B16;
	request_dispatch.request_ids[0u] = 100u;
	request_dispatch.sequence_ids[0u] = 200u;
	memset(&decode_view,0,sizeof(decode_view));
	decode_view.lane_count = 1u;
	decode_view.lanes[0u].request_index = 0u;
	decode_view.lanes[0u].request_id = request_dispatch.request_ids[0u];
	decode_view.lanes[0u].sequence_id = request_dispatch.sequence_ids[0u];
	decode_view.lanes[0u].sequence_position = 32u;
	decode_view.lanes[0u].context_token_count = 33u;
	decode_view.lanes[0u].request_slot_index = 7u;
	decode_view.lanes[0u].mtp_resolution_proposed_token_count = 2u;
	decode_view.lanes[0u].mtp_resolution_accepted_token_count = 1u;
	memset(&kv_view,0,sizeof(kv_view));
	memset(&decode_dispatch,0,sizeof(decode_dispatch));
	decode_dispatch.dispatch_kind =
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
	decode_dispatch.request_count = 1u;
	decode_dispatch.active_sequence_count = 1u;
	decode_dispatch.request_dispatch = &request_dispatch;
	decode_dispatch.decode_view = &decode_view;
	decode_dispatch.kv_block_table_view = &kv_view;
	decode_dispatch.input_token_ids[0u] = 300u;
	decode_dispatch.speculative_token_count = 3u;
	for (token_index = 0u; token_index < 3u; ++token_index)
		decode_dispatch.speculative_draft_token_ids[0u][token_index] =
			400u + token_index;
	assert(SparkGlm52Pp13WorkControlBuildDecodePacket(
		&decode_dispatch,0u,&packet) == SPARK_STATUS_OK);
	assert(packet.rows_per_lane == 4u && packet.execution_row_count == 4u);
	assert(packet.execution_batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B16);
	assert(packet.new_token_count == 4u);
	assert(packet.sequence_position == 32u && packet.input_token_id == 300u);
	assert((packet.flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE) != 0u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,7u,1u) ==
		SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlBuildDecodePacket(
		&decode_dispatch,1u,&packet) == SPARK_STATUS_INVALID_ARGUMENT);
}

static void SparkTestGlm52Pp13WorkControlBuildPrefillBatch(void)
{
	static SparkGlm52RequestApiDispatch request_dispatch;
	static SparkGlm52RequestApiPrefillDispatchView prefill_view;
	SparkGlm52PromptPipelinePrefillDispatch prefill_dispatch;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView kv_view;
	uint32_t token_ids[4u][2u];
	uint32_t block_indices[4u];
	uint32_t block_counts[4u];
	uint32_t lane_index;
	uint32_t token_count;

	memset(&request_dispatch,0,sizeof(request_dispatch));
	request_dispatch.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	request_dispatch.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_DISPATCH_DESCRIPTOR_BYTES;
	request_dispatch.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH;
	request_dispatch.request_count = 4u;
	request_dispatch.highest_priority = 81u;
	request_dispatch.prefill_batch_decision.batch_bucket =
		SPARK_GLM52_STAGE_PLAN_BUCKET_B32;
	memset(&prefill_view,0,sizeof(prefill_view));
	prefill_view.abi_version = SPARK_GLM52_REQUEST_API_ABI_VERSION;
	prefill_view.descriptor_bytes =
		SPARK_GLM52_REQUEST_API_PREFILL_DISPATCH_VIEW_DESCRIPTOR_BYTES;
	prefill_view.kind = SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH;
	prefill_view.active_sequence_count = 4u;
	prefill_view.lane_count = 4u;
	prefill_view.prompt_token_count = 2u;
	prefill_view.prompt_token_stride = 2u;
	memset(&kv_view,0,sizeof(kv_view));
	kv_view.abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	kv_view.descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	kv_view.block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	kv_view.lane_count = 4u;
	kv_view.lane_stride = 1u;
	kv_view.lane_capacity = 4u;
	kv_view.physical_block_indices = block_indices;
	kv_view.lane_physical_block_counts = block_counts;
	kv_view.host_physical_block_indices = block_indices;
	kv_view.host_lane_physical_block_counts = block_counts;
	memset(&prefill_dispatch,0,sizeof(prefill_dispatch));
	prefill_dispatch.abi_version = SPARK_GLM52_PROMPT_PIPELINE_ABI_VERSION;
	prefill_dispatch.descriptor_bytes =
		SPARK_GLM52_PROMPT_PIPELINE_PREFILL_DISPATCH_DESCRIPTOR_BYTES;
	prefill_dispatch.dispatch_kind =
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH;
	prefill_dispatch.active_sequence_count = 4u;
	prefill_dispatch.lane_count = 4u;
	prefill_dispatch.prompt_token_count = 2u;
	prefill_dispatch.prompt_token_stride = 2u;
	prefill_dispatch.host_token_stride = 2u;
	prefill_dispatch.request_dispatch = &request_dispatch;
	prefill_dispatch.prefill_view = &prefill_view;
	prefill_dispatch.host_token_ids = &token_ids[0u][0u];
	prefill_dispatch.kv_block_table_view = &kv_view;
	for (lane_index = 0u; lane_index < 4u; ++lane_index)
	{
		request_dispatch.request_ids[lane_index] = 100u + lane_index;
		request_dispatch.sequence_ids[lane_index] = 200u + lane_index;
		prefill_view.lanes[lane_index].request_index = lane_index;
		prefill_view.lanes[lane_index].request_slot_index = lane_index;
		prefill_view.lanes[lane_index].prompt_token_offset = 4u;
		prefill_view.lanes[lane_index].prompt_token_count = 2u;
		prefill_view.lanes[lane_index].request_id = 100u + lane_index;
		prefill_view.lanes[lane_index].sequence_id = 200u + lane_index;
		token_ids[lane_index][0u] = 300u + lane_index;
		token_ids[lane_index][1u] = 400u + lane_index;
		block_indices[lane_index] = 40u + lane_index;
		block_counts[lane_index] = 1u;
	}
	prefill_view.lanes[1u].prompt_token_count = 1u;
	assert(SparkGlm52Pp13WorkControlSelectPrefillChunk(
		&prefill_dispatch,0u,8u,&token_count) == SPARK_STATUS_OK);
	assert(token_count == 1u);
	assert(SparkGlm52Pp13WorkControlBuildPrefillPacket(
		&prefill_dispatch,0u,1u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 4u);
	assert(packet.rows_per_lane == 1u);
	assert(packet.new_token_count == 1u);
	assert(packet.execution_row_count == 4u);
	assert(packet.lanes[0u].input_token_id == 300u);
	assert(packet.lanes[1u].input_token_id == 301u);
	assert(packet.lanes[1u].context_token_count == 5u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_OK);
	packet.lanes[0u].input_token_id = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	packet.lanes[0u].input_token_id = 300u;
	assert(SparkGlm52Pp13WorkControlBuildPrefillPacket(
		&prefill_dispatch,0u,2u,&packet) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	prefill_view.lanes[1u].prompt_token_count = 2u;
	assert(SparkGlm52Pp13WorkControlSelectPrefillChunk(
		&prefill_dispatch,0u,8u,&token_count) == SPARK_STATUS_OK);
	assert(token_count == 2u);
	assert(SparkGlm52Pp13WorkControlBuildPrefillPacket(
		&prefill_dispatch,0u,2u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 4u);
	assert(packet.rows_per_lane == 2u);
	assert(packet.new_token_count == 2u);
	assert(packet.execution_row_count == 8u);
	assert(packet.prefill_token_ids[0u] == 300u);
	assert(packet.prefill_token_ids[1u] == 400u);
	assert(packet.prefill_token_ids[2u] == 301u);
	assert(packet.prefill_token_ids[7u] == 403u);
	assert(packet.lanes[0u].input_token_id == 400u);
	assert(packet.lanes[0u].context_token_count == 6u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,8u,1u) ==
		SPARK_STATUS_OK);
	packet.prefill_token_ids[7u] = SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT;
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,8u,1u) ==
		SPARK_STATUS_INVALID_ARGUMENT);
	prefill_view.lanes[1u].prompt_token_count = 1u;
	assert(SparkGlm52Pp13WorkControlSelectPrefillChunk(
		&prefill_dispatch,1u,8u,&token_count) == SPARK_STATUS_OK);
	assert(token_count == 1u);
	assert(SparkGlm52Pp13WorkControlBuildPrefillPacket(
		&prefill_dispatch,1u,1u,&packet) == SPARK_STATUS_OK);
	assert(packet.active_sequence_count == 3u);
	assert(packet.execution_batch_bucket == SPARK_GLM52_STAGE_PLAN_BUCKET_B32);
	assert(packet.descriptor_bytes ==
		SparkGlm52Pp13WorkControlCalculatePacketBytes(3u));
	assert(SparkGlm52CudaResidentIpcCalculateSubmitPrefillBytes(&packet) ==
		SPARK_GLM52_CUDA_RESIDENT_IPC_SUBMIT_PREFILL_PREFIX_BYTES +
		packet.descriptor_bytes);
	assert(packet.lanes[0u].request_id == 100u);
	assert(packet.lanes[1u].request_id == 102u);
	assert(packet.lanes[1u].input_token_id == 402u);
	assert(packet.lanes[1u].sequence_position == 5u);
	assert(packet.lanes[1u].request_slot_index == 2u);
	assert(SparkGlm52Pp13WorkControlValidatePacket(&packet,4u,1u) ==
		SPARK_STATUS_OK);
}

static void SparkTestGlm52Pp13WorkControlResetsOlderGeneration(void)
{
	static SparkTestWorkControlKvStorage storage;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView view;

	assert(SparkTestInitializeKvState(&state,&storage,1u) == SPARK_STATUS_OK);
	SparkTestInitializeWorkPacket(&packet);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	packet.control_generation = 100u;
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.execution_row_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.request_id = 101u;
	packet.sequence_id = 201u;
	packet.sequence_position = 0u;
	packet.kv_block_table_token_count = 1u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	packet.lanes[0u].sequence_position = 0u;
	packet.lanes[0u].context_token_count = 1u;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
		&packet,&state) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 1u);
	assert(state.control_generation == 100u);
	assert(state.control_generation_reset_count == 1u);
	assert(SparkGlm52Pp13WorkControlAdvanceKvGeneration(
		&state,200u) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 0u);
	assert(state.allocated_physical_block_count == 0u);
	assert(state.control_generation == 200u);
	assert(state.control_generation_reset_count == 2u);
	packet.control_generation = 200u;
	packet.request_id = 102u;
	packet.sequence_id = 202u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 1u);
	assert(state.allocated_physical_block_count == 1u);
	assert(state.control_generation == 200u);
	assert(state.control_generation_reset_count == 2u);
	packet.control_generation = 100u;
	assert(SparkGlm52Pp13WorkControlAdvanceKvGeneration(
		&state,100u) == SPARK_STATUS_VALIDATION_FAILED);
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_NOT_FOUND);
	assert(state.directory_entry_count == 1u);
	assert(state.control_generation == 200u);
}

static void SparkTestGlm52Pp13WorkControlPinsSpeculativeBlocks(void)
{
	static SparkTestWorkControlKvStorage storage;
	SparkGlm52Pp13WorkControlKvState state;
	SparkGlm52Pp13WorkControlPacket packet;
	SparkGlm52KvBlockTableView view;
	uint32_t physical_block_index;
	assert(SparkTestInitializeKvState(&state,&storage,1u) == SPARK_STATUS_OK);
	SparkTestInitializeWorkPacket(&packet);
	packet.flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	packet.active_sequence_count = 1u;
	packet.lane_count = 1u;
	packet.execution_row_count = 1u;
	packet.descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(1u);
	packet.sequence_position = 0u;
	packet.kv_block_table_token_count = 1u;
	packet.lanes[0u].request_id = packet.request_id;
	packet.lanes[0u].sequence_id = packet.sequence_id;
	packet.lanes[0u].sequence_position = 0u;
	packet.lanes[0u].request_slot_index = 0u;
	packet.lanes[0u].context_token_count = 1u;
	assert(SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
		&packet,&state,&view) == SPARK_STATUS_OK);
	physical_block_index = view.host_physical_block_indices[0u];
	assert(SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
		&packet,&state) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlPinPhysicalBlock(
		&state,physical_block_index) == SPARK_STATUS_OK);
	assert(state.physical_block_pin_counts[physical_block_index] == 1u);
	assert(SparkGlm52Pp13WorkControlReleaseSequence(
		&state,packet.sequence_id,1u) == SPARK_STATUS_BUSY);
	assert(state.directory_entry_count == 1u);
	assert(SparkGlm52Pp13WorkControlUnpinPhysicalBlock(
		&state,physical_block_index) == SPARK_STATUS_OK);
	assert(SparkGlm52Pp13WorkControlReleaseSequence(
		&state,packet.sequence_id,1u) == SPARK_STATUS_OK);
	assert(state.directory_entry_count == 0u);
}

int main(void)
{
	SparkTestGlm52Pp13WorkControlExecutionChunks();
	SparkTestGlm52Pp13WorkControlPacket();
	SparkTestGlm52Pp13WorkControlHostBlockTable();
	SparkTestGlm52Pp13WorkControlTracksKvReadiness();
	SparkTestGlm52Pp13WorkControlKeepsStableBlocksAcrossLaneReorder();
	SparkTestGlm52Pp13WorkControlDsparkVerify();
	SparkTestGlm52Pp13WorkControlMtpVerify();
	SparkTestGlm52Pp13WorkControlBuildDecodeBatch();
	SparkTestGlm52Pp13WorkControlBuildPackedMtpVerify();
	SparkTestGlm52Pp13WorkControlBuildPrefillBatch();
	SparkTestGlm52Pp13WorkControlB1024MtpBatch();
	SparkTestGlm52Pp13WorkControlB1024LayerMajorMtpVerify();
	SparkTestGlm52Pp13WorkControlB1024LayerMajorDsparkVerify();

	SparkTestGlm52Pp13WorkControlCommitsTreePositions();
	SparkTestGlm52Pp13WorkControlB1024PhysicalDirectory();
	SparkTestGlm52Pp13WorkControlNvmeSwapAndRelease();
	SparkTestGlm52Pp13WorkControlResetsOlderGeneration();
	SparkTestGlm52Pp13WorkControlPinsSpeculativeBlocks();
	return 0;
}
