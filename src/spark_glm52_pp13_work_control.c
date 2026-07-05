#include "sparkpipe/spark_glm52_pp13_work_control.h"

#include <string.h>

uint32_t SparkGlm52Pp13WorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count)
{
	if (token_count == 0u || block_token_count == 0u)
		return 0u;
	return (token_count + block_token_count - 1u) / block_token_count;
}

SparkStatus SparkGlm52Pp13WorkControlValidatePacket(
	const SparkGlm52Pp13WorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count)
{
	if (packet == 0 ||
		packet->magic != SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC ||
		packet->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		packet->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_PACKET_BYTES)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & ~SPARK_GLM52_PP13_WORK_CONTROL_KNOWN_FLAGS) != 0u ||
		packet->request_id == 0u ||
		packet->sequence_id == 0u ||
		packet->active_sequence_count == 0u ||
		packet->active_sequence_count > max_active_sequence_count ||
		packet->new_token_count == 0u ||
		packet->pipeline_slot >= max_pipeline_slot_count ||
		packet->block_token_count == 0u ||
		packet->kv_block_table_token_count == 0u ||
		packet->max_blocks_per_sequence == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u &&
		packet->new_token_count > 8u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlInitializeKvState(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts)
{
	uint64_t physical_block_capacity;

	if (state == 0 ||
		lane_capacity == 0u ||
		lane_stride == 0u ||
		block_token_count == 0u ||
		physical_block_indices == 0 ||
		lane_physical_block_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	physical_block_capacity = (uint64_t)lane_capacity * (uint64_t)lane_stride;
	if (physical_block_capacity > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(state,0,sizeof(*state));
	state->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	state->descriptor_bytes = SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES;
	state->lane_capacity = lane_capacity;
	state->lane_stride = lane_stride;
	state->block_token_count = block_token_count;
	state->physical_block_capacity = (uint32_t)physical_block_capacity;
	state->physical_block_indices = physical_block_indices;
	state->lane_physical_block_counts = lane_physical_block_counts;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlBuildHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state,
	SparkGlm52KvBlockTableView *view)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint64_t base_block_index;
	uint64_t physical_block_index;

	if (packet == 0 || state == 0 || view == 0 ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_indices == 0 ||
		state->lane_physical_block_counts == 0 ||
		state->lane_capacity == 0u ||
		state->lane_stride == 0u ||
		state->block_token_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkGlm52Pp13WorkControlValidatePacket(
			packet,
			state->lane_capacity,
			UINT32_MAX) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->block_token_count != state->block_token_count ||
		packet->max_blocks_per_sequence > state->lane_stride ||
		packet->active_sequence_count > state->lane_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	block_count = SparkGlm52Pp13WorkControlBlockCount(
		packet->kv_block_table_token_count,
		packet->block_token_count);
	if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(
		state->physical_block_indices,
		0xff,
		state->physical_block_capacity * sizeof(state->physical_block_indices[0]));
	memset(
		state->lane_physical_block_counts,
		0,
		state->lane_capacity * sizeof(state->lane_physical_block_counts[0]));
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		base_block_index = (uint64_t)lane_index * (uint64_t)state->lane_stride;
		state->lane_physical_block_counts[lane_index] = block_count;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			physical_block_index = base_block_index + (uint64_t)block_index;
			if (physical_block_index > UINT32_MAX)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			state->physical_block_indices[base_block_index + block_index] =
				(uint32_t)physical_block_index;
		}
	}
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	view->block_token_count = packet->block_token_count;
	view->lane_count = packet->active_sequence_count;
	view->lane_stride = state->lane_stride;
	view->lane_capacity = state->lane_stride;
	view->physical_block_indices = state->physical_block_indices;
	view->lane_physical_block_counts = state->lane_physical_block_counts;
	view->host_physical_block_indices = state->physical_block_indices;
	return SPARK_STATUS_OK;
}
