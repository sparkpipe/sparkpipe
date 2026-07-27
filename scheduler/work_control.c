#include "sparkpipe/spark_glm52_ring_work_control.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"

#include <string.h>

uint32_t SparkGlm52RingWorkControlCalculatePacketBytes(
	uint32_t active_sequence_count)
{
	uint64_t packet_bytes;

	if (active_sequence_count == 0u ||
		active_sequence_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT)
		return 0u;
	packet_bytes =
		(uint64_t)SPARK_GLM52_RING_WORK_CONTROL_PACKET_PREFIX_BYTES +
		((uint64_t)active_sequence_count *
		 (uint64_t)SPARK_GLM52_RING_WORK_CONTROL_LANE_BYTES);
	return packet_bytes <= UINT32_MAX ? (uint32_t)packet_bytes : 0u;
}

SparkStatus SparkGlm52RingWorkControlSelectExecutionBatchBucket(
	const SparkGlm52RequestApiDispatch *request_dispatch,
	uint32_t batch_lane_or_row_count,
	uint32_t *batch_bucket_out)
{
	uint32_t batch_bucket;
	if (request_dispatch == 0 || batch_bucket_out == 0 ||
		batch_lane_or_row_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request_dispatch->kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL)
		batch_bucket = request_dispatch->prefill_decision.batch_bucket;
	else if (request_dispatch->kind ==
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_PREFILL_BATCH)
		batch_bucket = request_dispatch->prefill_batch_decision.batch_bucket;
	else if (request_dispatch->kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH ||
		 request_dispatch->kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH)
		batch_bucket = request_dispatch->decode_batch_decision.batch_bucket;
	else
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkGlm52StagePlanBatchBucketIsSupported(batch_bucket) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (batch_bucket < batch_lane_or_row_count)
	{
		SparkStatus status;

		status = SparkGlm52StagePlanSelectBatchBucket(
			batch_lane_or_row_count,
			&batch_bucket);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	*batch_bucket_out = batch_bucket;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlSelectMtpDraftBudget(
	uint32_t dispatch_kind,
	uint32_t request_flags,
	uint32_t requested_budget,
	uint32_t *mtp_budget_out)
{
	uint32_t producer;
	uint32_t tree_verify;
	if (mtp_budget_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	producer =
		dispatch_kind == SPARK_GLM52_REQUEST_API_DISPATCH_KIND_DECODE_BATCH &&
		(request_flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_COMMIT) != 0u;
	tree_verify =
		dispatch_kind ==
			SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH &&
		(request_flags &
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY)) ==
			(SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY |
			 SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY);
	if (producer == 0u && tree_verify == 0u)
	{
		if (requested_budget != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		*mtp_budget_out = 0u;
		return SPARK_STATUS_OK;
	}
	if (requested_budget != SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	*mtp_budget_out = requested_budget;
	return SPARK_STATUS_OK;
}

static uint32_t SparkGlm52RingWorkControlWrittenPositionCount(
	const SparkGlm52RingWorkControlPacket *packet)
{
	if (packet != 0 &&
		(packet->flags &
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY) != 0u)
		return SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION + 1u;
	return packet == 0 ? 0u : packet->new_token_count;
}

static void SparkGlm52RingWorkControlSetDecodeFlags(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t mtp_budget,
	SparkGlm52RingWorkControlPacket *packet)
{
	if (mtp_budget != 0u)
		packet->flags |= SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY;
}

static void SparkGlm52RingWorkControlSetMtpResolutionFlag(
	SparkGlm52RingWorkControlPacket *packet)
{
	uint32_t lane_index;

	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		if (packet->lanes[lane_index].
			mtp_resolution_proposed_token_count != 0u)
		{
			packet->flags |= SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_RESOLVE;
			break;
		}
	}
}

static SparkStatus SparkGlm52RingWorkControlBuildDecodeLanes(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_verify,
	uint32_t mtp_budget,
	SparkGlm52RingWorkControlPacket *packet)
{
	const SparkGlm52RequestApiDecodeDispatchLaneView *source_lane;
	SparkGlm52RingWorkControlLane *lane;
	uint32_t lane_index;
	uint32_t mtp_tree_verify;
	uint32_t source_index;
	uint32_t token_index;

	mtp_tree_verify = (decode_dispatch->request_dispatch->flags &
		SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u;
	for (lane_index = 0u; lane_index < lane_count; ++lane_index)
	{
		source_index = lane_offset + lane_index;
		source_lane = &decode_dispatch->decode_view->lanes[source_index];
		lane = &packet->lanes[lane_index];
		if (source_lane->request_index != source_index ||
			source_lane->request_id !=
				decode_dispatch->request_dispatch->request_ids[source_index] ||
			source_lane->sequence_id !=
				decode_dispatch->request_dispatch->sequence_ids[source_index])
			return SPARK_STATUS_INVALID_ARGUMENT;
		lane->request_id =
			decode_dispatch->request_dispatch->request_ids[source_index];
		lane->sequence_id =
			decode_dispatch->request_dispatch->sequence_ids[source_index];
		lane->sequence_position = source_lane->sequence_position;
		lane->request_slot_index = source_lane->request_slot_index;
		lane->context_token_count = source_lane->context_token_count;
		lane->input_token_id = decode_dispatch->input_token_ids[source_index];
		lane->mtp_draft_token_count = mtp_budget;
		lane->mtp_resolution_proposed_token_count =
			(uint8_t)source_lane->mtp_resolution_proposed_token_count;
		lane->mtp_resolution_accepted_token_count =
			(uint8_t)source_lane->mtp_resolution_accepted_token_count;
		lane->mtp_resolution_path_id =
			(uint16_t)source_lane->mtp_resolution_path_id;
		if (speculative_verify != 0u)
		{
			uint32_t context_extension;
			context_extension = mtp_tree_verify != 0u
				? SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION
				: decode_dispatch->speculative_token_count;
			if (lane->context_token_count >
				UINT32_MAX - context_extension)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			lane->context_token_count += context_extension;
			lane->speculative_token_count =
				decode_dispatch->speculative_token_count;
			for (token_index = 0u;
				 token_index < lane->speculative_token_count;
				 ++token_index)
				lane->speculative_draft_token_ids[token_index] =
					decode_dispatch->speculative_draft_token_ids[
						source_index][token_index];
		}
		else if (mtp_budget != 0u)
		{
			if (lane->context_token_count > UINT32_MAX - mtp_budget)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			lane->context_token_count += mtp_budget;
		}
		if (lane->context_token_count > packet->kv_block_table_token_count)
			packet->kv_block_table_token_count = lane->context_token_count;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlBuildDecodePacketRange(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet)
{
	uint32_t speculative_verify;
	uint32_t mtp_tree_verify;
	uint32_t mtp_budget;
	uint32_t packet_bytes;
	SparkStatus status;

	if (decode_dispatch == 0 || packet == 0 ||
		decode_dispatch->request_dispatch == 0 ||
		decode_dispatch->decode_view == 0 ||
		decode_dispatch->kv_block_table_view == 0 ||
		decode_dispatch->request_count == 0u ||
		decode_dispatch->request_count != decode_dispatch->active_sequence_count ||
		decode_dispatch->decode_view->lane_count !=
			decode_dispatch->active_sequence_count ||
		decode_dispatch->active_sequence_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		lane_count == 0u ||
		lane_offset >= decode_dispatch->active_sequence_count ||
		lane_count > decode_dispatch->active_sequence_count - lane_offset)
		return SPARK_STATUS_INVALID_ARGUMENT;
	speculative_verify = decode_dispatch->dispatch_kind ==
		SPARK_GLM52_REQUEST_API_DISPATCH_KIND_SPECULATIVE_VERIFY_BATCH;
	mtp_tree_verify = (decode_dispatch->request_dispatch->flags &
		SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u;
	if ((speculative_verify != 0u &&
		 (decode_dispatch->speculative_token_count == 0u ||
		  decode_dispatch->speculative_token_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
		  speculative_token_index != 0u)) ||
		(speculative_verify == 0u && speculative_token_index != 0u))
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingWorkControlSelectMtpDraftBudget(
		decode_dispatch->dispatch_kind,
		decode_dispatch->request_dispatch->flags,
		decode_dispatch->request_dispatch->mtp_draft_token_budget,
		&mtp_budget);
	if (status != SPARK_STATUS_OK)
		return status;
	packet_bytes = SparkGlm52RingWorkControlCalculatePacketBytes(lane_count);
	if (packet_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(packet,0,packet_bytes);
	packet->magic = SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
	packet->control_generation =
		SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
	packet->active_sequence_count = lane_count;
	packet->lane_count = packet->active_sequence_count;
	packet->descriptor_bytes = SparkGlm52RingWorkControlCalculatePacketBytes(
		packet->active_sequence_count);
	packet->rows_per_lane = speculative_verify != 0u
		? mtp_tree_verify != 0u
			? decode_dispatch->request_dispatch->
				speculative_verifier_token_count
			: decode_dispatch->speculative_token_count + 1u
		: 1u;
	if (mtp_tree_verify != 0u &&
		(packet->rows_per_lane !=
				SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT ||
		 decode_dispatch->speculative_token_count !=
				SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT))
		return SPARK_STATUS_MODULE_NOT_VALIDATED;
	if ((uint64_t)packet->lane_count * packet->rows_per_lane > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	packet->execution_row_count =
		packet->lane_count * packet->rows_per_lane;
	status = SparkGlm52RingWorkControlSelectExecutionBatchBucket(
		decode_dispatch->request_dispatch,
		packet->execution_row_count,
		&packet->execution_batch_bucket);
	if (status != SPARK_STATUS_OK)
		return status;
	packet->new_token_count = speculative_verify != 0u
		? packet->rows_per_lane : mtp_budget + 1u;
	packet->priority = decode_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_RING_WORK_CONTROL_KV_BLOCK_CAPACITY;
	packet->mtp_draft_token_count = mtp_budget;
	SparkGlm52RingWorkControlSetDecodeFlags(decode_dispatch,mtp_budget,packet);
	status = SparkGlm52RingWorkControlBuildDecodeLanes(
		decode_dispatch,lane_offset,lane_count,speculative_verify,
		mtp_budget,packet);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52RingWorkControlSetMtpResolutionFlag(packet);
	packet->request_id = packet->lanes[0u].request_id;
	packet->sequence_id = packet->lanes[0u].sequence_id;
	packet->sequence_position = packet->lanes[0u].sequence_position;
	packet->input_token_id = packet->lanes[0u].input_token_id;
	if (speculative_verify != 0u)
	{
		packet->speculative_token_count =
			decode_dispatch->speculative_token_count;
		packet->speculative_token_index = speculative_token_index;
		memcpy(packet->speculative_draft_token_ids,
			decode_dispatch->speculative_draft_token_ids[lane_offset],
			sizeof(packet->speculative_draft_token_ids));
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlBuildDecodePacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkGlm52RingWorkControlPacket *packet)
{
	if (decode_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52RingWorkControlBuildDecodePacketRange(
		decode_dispatch,0u,decode_dispatch->active_sequence_count,
		speculative_token_index,packet);
}

SparkStatus SparkGlm52RingWorkControlSelectPrefillChunk(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t maximum_execution_row_count,
	uint32_t *token_count_out)
{
	const SparkGlm52RequestApiPrefillDispatchLaneView *lane;
	uint32_t active_lane_count;
	uint32_t lane_index;
	uint32_t minimum_remaining_token_count;
	uint32_t remaining_token_count;
	uint32_t token_count;
	if (prefill_dispatch == 0 || token_count_out == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->lane_count == 0u ||
		prefill_dispatch->lane_count != prefill_dispatch->active_sequence_count ||
		prefill_dispatch->lane_count !=
			prefill_dispatch->prefill_view->lane_count ||
		token_offset >= prefill_dispatch->prompt_token_count ||
		maximum_execution_row_count == 0u ||
		maximum_execution_row_count >
			SPARK_GLM52_STAGE_PLAN_MAX_BATCH_BUCKET)
		return SPARK_STATUS_INVALID_ARGUMENT;
	active_lane_count = 0u;
	minimum_remaining_token_count = UINT32_MAX;
	for (lane_index = 0u; lane_index < prefill_dispatch->lane_count; ++lane_index)
	{
		lane = &prefill_dispatch->prefill_view->lanes[lane_index];
		if (token_offset >= lane->prompt_token_count)
			continue;
		remaining_token_count = lane->prompt_token_count - token_offset;
		if (minimum_remaining_token_count > remaining_token_count)
			minimum_remaining_token_count = remaining_token_count;
		active_lane_count += 1u;
	}
	if (active_lane_count == 0u ||
		active_lane_count > maximum_execution_row_count)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	token_count = maximum_execution_row_count / active_lane_count;
	if (token_count > minimum_remaining_token_count)
		token_count = minimum_remaining_token_count;
	if (token_count >
		SPARK_GLM52_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET)
		token_count =
			SPARK_GLM52_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET;
	if (token_count > prefill_dispatch->prompt_token_count - token_offset)
		token_count = prefill_dispatch->prompt_token_count - token_offset;
	if (token_count == 0u)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	*token_count_out = token_count;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlBuildPrefillPacket(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkGlm52RingWorkControlPacket *packet)
{
	const SparkGlm52RequestApiPrefillDispatchLaneView *source_lane;
	SparkGlm52RingWorkControlLane *destination_lane;
	uint32_t active_lane_count;
	uint32_t execution_row_index;
	uint32_t packet_bytes;
	uint32_t row_offset;
	uint32_t source_lane_index;
	uint32_t position;

	if (prefill_dispatch == 0 || packet == 0 ||
		prefill_dispatch->request_dispatch == 0 ||
		prefill_dispatch->prefill_view == 0 ||
		prefill_dispatch->kv_block_table_view == 0 ||
		prefill_dispatch->lane_count == 0u ||
		prefill_dispatch->lane_count !=
			prefill_dispatch->active_sequence_count ||
		prefill_dispatch->lane_count !=
			prefill_dispatch->prefill_view->lane_count ||
		prefill_dispatch->lane_count !=
			prefill_dispatch->kv_block_table_view->lane_count ||
		prefill_dispatch->lane_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		prefill_dispatch->host_token_ids == 0 ||
		prefill_dispatch->host_token_stride == 0u ||
		token_offset >= prefill_dispatch->prompt_token_count ||
		token_count == 0u ||
		token_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET ||
		token_count > prefill_dispatch->prompt_token_count - token_offset)
		return SPARK_STATUS_INVALID_ARGUMENT;
	active_lane_count = 0u;
	for (source_lane_index = 0u;
		 source_lane_index < prefill_dispatch->lane_count;
		 ++source_lane_index)
	{
		source_lane = &prefill_dispatch->prefill_view->lanes[source_lane_index];
		if (token_offset < source_lane->prompt_token_count)
		{
			if (token_count >
				source_lane->prompt_token_count - token_offset)
				return SPARK_STATUS_INVALID_ARGUMENT;
			active_lane_count += 1u;
		}
	}
	if ((uint64_t)active_lane_count * token_count >
		SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	packet_bytes =
		SparkGlm52RingWorkControlCalculatePacketBytes(active_lane_count);
	if (packet_bytes == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(packet,0,packet_bytes);
	packet->magic = SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
	packet->control_generation =
		SPARK_GLM52_RING_WORK_CONTROL_STANDALONE_GENERATION;
	packet->flags = SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL;
	if ((prefill_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	packet->new_token_count = token_count;
	packet->priority = prefill_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_RING_WORK_CONTROL_KV_BLOCK_CAPACITY;
	active_lane_count = 0u;
	for (source_lane_index = 0u;
		 source_lane_index < prefill_dispatch->lane_count;
		 ++source_lane_index)
	{
		source_lane = &prefill_dispatch->prefill_view->lanes[source_lane_index];
		if (token_offset >= source_lane->prompt_token_count)
			continue;
		if (source_lane->prompt_token_count >
			prefill_dispatch->host_token_stride)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		if (source_lane->prompt_token_offset > UINT32_MAX - token_offset)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		position = source_lane->prompt_token_offset + token_offset;
		if (position > UINT32_MAX - token_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		destination_lane = &packet->lanes[active_lane_count];
		destination_lane->request_id = source_lane->request_id;
		destination_lane->sequence_id = source_lane->sequence_id;
		destination_lane->sequence_position = position;
		destination_lane->request_slot_index =
			source_lane->request_slot_index;
		destination_lane->context_token_count = position + token_count;
		execution_row_index = active_lane_count * token_count;
		for (row_offset = 0u; row_offset < token_count; ++row_offset)
		{
			packet->prefill_token_ids[execution_row_index + row_offset] =
				prefill_dispatch->host_token_ids[
					((uint64_t)source_lane_index *
					 prefill_dispatch->host_token_stride) +
					token_offset + row_offset];
		}
		destination_lane->input_token_id =
			packet->prefill_token_ids[execution_row_index + token_count - 1u];
		if (destination_lane->context_token_count >
			packet->kv_block_table_token_count)
			packet->kv_block_table_token_count =
				destination_lane->context_token_count;
		active_lane_count += 1u;
	}
	if (active_lane_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	packet->active_sequence_count = active_lane_count;
	packet->lane_count = active_lane_count;
	packet->rows_per_lane = token_count;
	packet->execution_row_count = active_lane_count * token_count;
	if (SparkGlm52RingWorkControlSelectExecutionBatchBucket(
			prefill_dispatch->request_dispatch,
			packet->execution_row_count,
			&packet->execution_batch_bucket) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	packet->descriptor_bytes =
		SparkGlm52RingWorkControlCalculatePacketBytes(active_lane_count);
	packet->request_id = packet->lanes[0u].request_id;
	packet->sequence_id = packet->lanes[0u].sequence_id;
	packet->sequence_position = packet->lanes[0u].sequence_position;
	packet->input_token_id = packet->lanes[0u].input_token_id;
	return SPARK_STATUS_OK;
}

uint32_t SparkGlm52RingWorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count)
{
	if (token_count == 0u || block_token_count == 0u)
		return 0u;
	return (token_count + block_token_count - 1u) / block_token_count;
}

SparkStatus SparkGlm52RingWorkControlPlanExecutionChunks(
	uint32_t logical_lane_count,
	uint32_t rows_per_lane,
	uint32_t execution_row_capacity,
	uint32_t *maximum_lanes_per_chunk_out,
	uint32_t *chunk_count_out)
{
	if (logical_lane_count == 0u ||
		logical_lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
		rows_per_lane == 0u || execution_row_capacity == 0u ||
		maximum_lanes_per_chunk_out == 0 || chunk_count_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52StagePlanExecutionChunkShape(
		logical_lane_count,
		rows_per_lane,
		execution_row_capacity,
		maximum_lanes_per_chunk_out,
		chunk_count_out);
}

SparkStatus SparkGlm52RingWorkControlValidatePacket(
	const SparkGlm52RingWorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count)
{
	uint32_t token_index;
	uint32_t dspark_verify;
	uint32_t mtp_verify;
	uint32_t mtp_tree_verify;
	uint32_t speculative_verify;
	uint32_t lane_index;
	uint32_t execution_row_index;
	uint32_t expected_rows_per_lane;
	uint64_t expected_execution_row_count;
	uint32_t release_sequences;
	uint32_t mtp_resolution_lane_count;

	if (packet == 0 ||
		packet->magic != SPARK_GLM52_RING_WORK_CONTROL_PACKET_MAGIC ||
		packet->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & ~SPARK_GLM52_RING_WORK_CONTROL_KNOWN_FLAGS) != 0u ||
		packet->request_id == 0u ||
		packet->sequence_id == 0u ||
		packet->control_generation == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	release_sequences = (packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u;
	if (release_sequences != 0u)
	{
		if (packet->flags !=
				SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES ||
			packet->descriptor_bytes !=
				SparkGlm52RingWorkControlCalculatePacketBytes(packet->lane_count) ||
			packet->active_sequence_count == 0u ||
			packet->active_sequence_count >
				SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
			packet->lane_count != packet->active_sequence_count ||
			packet->lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
			packet->new_token_count != 0u ||
			packet->pipeline_slot >= max_pipeline_slot_count ||
			packet->block_token_count == 0u ||
			packet->kv_block_table_token_count == 0u ||
			packet->max_blocks_per_sequence == 0u ||
			packet->mtp_draft_token_count != 0u ||
			packet->input_token_id != 0u ||
			packet->speculative_token_count != 0u ||
			packet->speculative_token_index != 0u ||
			packet->rows_per_lane != 0u ||
			packet->execution_row_count != 0u ||
			packet->execution_batch_bucket != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (token_index = 0u;
			 token_index < SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
		{
			const SparkGlm52RingWorkControlLane *lane;
			lane = &packet->lanes[lane_index];
			if (lane->request_id == 0u || lane->sequence_id == 0u ||
				lane->context_token_count == 0u ||
				lane->context_token_count > packet->kv_block_table_token_count ||
				SparkGlm52RingWorkControlBlockCount(
					lane->context_token_count,packet->block_token_count) >
					packet->max_blocks_per_sequence ||
				lane->input_token_id != 0u ||
				lane->mtp_draft_token_count != 0u ||
				lane->speculative_token_count != 0u ||
				lane->mtp_resolution_proposed_token_count != 0u ||
				lane->mtp_resolution_accepted_token_count != 0u ||
				lane->mtp_resolution_path_id !=
					SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)
				return SPARK_STATUS_INVALID_ARGUMENT;
			for (token_index = 0u;
				 token_index < SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
				 ++token_index)
			{
				if (lane->speculative_draft_token_ids[token_index] != 0u)
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
		}
		if (packet->request_id != packet->lanes[0u].request_id ||
			packet->sequence_id != packet->lanes[0u].sequence_id)
			return SPARK_STATUS_INVALID_ARGUMENT;
		return SPARK_STATUS_OK;
	}
	if (
		packet->active_sequence_count == 0u ||
		packet->active_sequence_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
		packet->lane_count == 0u ||
		packet->lane_count > SPARK_GLM52_RING_WORK_CONTROL_MAX_LANE_COUNT ||
		packet->lane_count != packet->active_sequence_count ||
		packet->new_token_count == 0u ||
		packet->pipeline_slot >= max_pipeline_slot_count ||
		packet->block_token_count == 0u ||
		packet->kv_block_table_token_count == 0u ||
		packet->max_blocks_per_sequence == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->active_sequence_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		packet->descriptor_bytes != SparkGlm52RingWorkControlCalculatePacketBytes(
			packet->active_sequence_count) ||
		packet->max_blocks_per_sequence >
			SPARK_GLM52_RING_WORK_CONTROL_KV_BLOCK_CAPACITY)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) == 0u &&
		packet->new_token_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u &&
		(packet->new_token_count == 0u ||
		 packet->new_token_count >
			SPARK_GLM52_RING_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET))
		return SPARK_STATUS_INVALID_ARGUMENT;
	dspark_verify = (packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
	mtp_verify = (packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
	mtp_tree_verify = (packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_TREE_VERIFY) != 0u;
	speculative_verify = dspark_verify | mtp_verify;
	expected_rows_per_lane = (packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u
		? packet->new_token_count
		: speculative_verify != 0u
			? mtp_tree_verify != 0u
				? SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT
				: packet->speculative_token_count + 1u
			: 1u;
	expected_execution_row_count =
		(uint64_t)packet->lane_count * expected_rows_per_lane;
	if (packet->rows_per_lane != expected_rows_per_lane ||
		expected_execution_row_count > UINT32_MAX ||
		expected_execution_row_count > max_active_sequence_count ||
		packet->execution_row_count != (uint32_t)expected_execution_row_count ||
		SparkGlm52StagePlanBatchBucketIsSupported(
			packet->execution_batch_bucket) == 0u ||
		((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u
			? packet->execution_row_count : packet->lane_count) >
			packet->execution_batch_bucket)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->input_token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
	{
		for (execution_row_index = 0u;
			 execution_row_index < packet->execution_row_count;
			 ++execution_row_index)
		{
			if (packet->prefill_token_ids[execution_row_index] >=
				SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	if (dspark_verify != 0u && mtp_verify != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (mtp_tree_verify != 0u &&
		(mtp_verify == 0u ||
		 packet->speculative_token_count !=
			SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT ||
		 packet->rows_per_lane !=
			SPARK_GLM52_MODEL_MTP_TREE_VERIFIER_ROW_COUNT))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->mtp_draft_token_count >
		SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT ||
		(((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u) !=
		 (packet->mtp_draft_token_count != 0u)))
		return SPARK_STATUS_INVALID_ARGUMENT;
	mtp_resolution_lane_count = 0u;
	if (speculative_verify != 0u)
	{
		if ((packet->flags &
				SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u ||
			packet->new_token_count != packet->rows_per_lane ||
			packet->speculative_token_count == 0u ||
			packet->speculative_token_count >
				SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
			packet->speculative_token_index != 0u ||
			(dspark_verify != 0u &&
			 (packet->flags &
				SPARK_GLM52_RING_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE) == 0u))
			return SPARK_STATUS_INVALID_ARGUMENT;
		if (mtp_tree_verify != 0u)
		{
			if ((packet->flags &
					SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT) == 0u ||
				packet->mtp_draft_token_count !=
					SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
				return SPARK_STATUS_MODULE_NOT_VALIDATED;
		}
		else if ((packet->flags &
					SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u ||
			packet->mtp_draft_token_count != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (token_index = 0u;
			 token_index < packet->speculative_token_count;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] >=
				SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		for (token_index = packet->speculative_token_count;
			 token_index <
				SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	else
	{
		if (packet->speculative_token_count != 0u ||
			packet->speculative_token_index != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
		for (token_index = 0u;
			 token_index <
				SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		const SparkGlm52RingWorkControlLane *lane;
		uint32_t lane_position_count;
		lane = &packet->lanes[lane_index];
		lane_position_count =
			SparkGlm52RingWorkControlWrittenPositionCount(packet);
		if (lane->request_id == 0u || lane->sequence_id == 0u ||
			lane->request_slot_index == SPARK_GLM52_RING_WORK_CONTROL_INVALID_REQUEST_SLOT ||
			lane->context_token_count == 0u ||
			lane->sequence_position + (uint64_t)lane_position_count >
				lane->context_token_count ||
			lane->context_token_count > packet->kv_block_table_token_count ||
			SparkGlm52RingWorkControlBlockCount(
				lane->context_token_count,packet->block_token_count) >
				packet->max_blocks_per_sequence ||
			lane->input_token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
		if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
		{
			if (lane->mtp_draft_token_count != 0u ||
				lane->speculative_token_count != 0u ||
				lane->context_token_count !=
					lane->sequence_position + packet->new_token_count ||
				lane->input_token_id != packet->prefill_token_ids[
					(lane_index * packet->rows_per_lane) +
					packet->rows_per_lane - 1u])
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		if (lane->mtp_resolution_proposed_token_count == 0u)
		{
			if (lane->mtp_resolution_accepted_token_count != 0u ||
				lane->mtp_resolution_path_id !=
					SPARK_GLM52_MODEL_MTP_TREE_RESOLUTION_NONE)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		else
		{
			if ((packet->flags &
					SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_RESOLVE) == 0u ||
				lane->mtp_resolution_proposed_token_count >
					SPARK_GLM52_MODEL_MTP_DRAFT_TOKEN_COUNT ||
				SparkGlm52MtpTreeResolutionIsValid(
					lane->mtp_resolution_proposed_token_count,
					lane->mtp_resolution_accepted_token_count,
					lane->mtp_resolution_path_id) == 0u ||
				lane->sequence_position <
					(uint64_t)lane->mtp_resolution_accepted_token_count + 1u)
				return SPARK_STATUS_INVALID_ARGUMENT;
			mtp_resolution_lane_count += 1u;
		}
		if (speculative_verify != 0u)
		{
			if (lane->mtp_draft_token_count != packet->mtp_draft_token_count ||
				lane->speculative_token_count != packet->speculative_token_count)
				return SPARK_STATUS_INVALID_ARGUMENT;
			for (token_index = 0u;
				 token_index < lane->speculative_token_count;
				 ++token_index)
			{
				if (lane->speculative_draft_token_ids[token_index] >=
					SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
			for (token_index = lane->speculative_token_count;
				 token_index <
					SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
				 ++token_index)
			{
				if (lane->speculative_draft_token_ids[token_index] != 0u)
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
		}
		else
		{
			if (lane->speculative_token_count != 0u ||
				lane->mtp_draft_token_count != packet->mtp_draft_token_count)
				return SPARK_STATUS_INVALID_ARGUMENT;
			for (token_index = 0u;
				 token_index < SPARK_GLM52_RING_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
				 ++token_index)
			{
				if (lane->speculative_draft_token_ids[token_index] != 0u)
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
		}
	}
	if (((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_MTP_RESOLVE) != 0u) !=
		(mtp_resolution_lane_count != 0u))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->request_id != packet->lanes[0u].request_id ||
		packet->sequence_id != packet->lanes[0u].sequence_id ||
		packet->sequence_position != packet->lanes[0u].sequence_position ||
		packet->input_token_id != packet->lanes[0u].input_token_id ||
		(speculative_verify != 0u &&
		 memcmp(packet->speculative_draft_token_ids,
			packet->lanes[0u].speculative_draft_token_ids,
			sizeof(packet->speculative_draft_token_ids)) != 0))
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

static uint64_t SparkGlm52RingWorkControlKvMix(uint64_t value)
{
	value ^= (value >> 30u);
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= (value >> 27u);
	value *= 0x94d049bb133111ebull;
	return value ^ (value >> 31u);
}

uint32_t SparkGlm52RingWorkControlKvKeyEqual(SparkGlm52RingKvKey left,SparkGlm52RingKvKey right) { return (left.low == right.low && left.high == right.high) ? 1u : 0u; }

static uint32_t SparkGlm52RingWorkControlKvKeyEmpty(SparkGlm52RingKvKey key) { return (key.low == 0u && key.high == 0u) ? 1u : 0u; }

SparkGlm52RingKvKey SparkGlm52RingWorkControlPrivateKey(uint64_t sequence_id,uint32_t logical_block_index)
{
	SparkGlm52RingKvKey key;
	key.low = SparkGlm52RingWorkControlKvMix(sequence_id ^ ((uint64_t)logical_block_index * 0x9e3779b97f4a7c15ull));
	key.high = SparkGlm52RingWorkControlKvMix(key.low ^ 0xd6e8feb86659fd93ull) | 1ull;
	return key;
}

SparkGlm52RingKvKey SparkGlm52RingWorkControlContentKey(uint64_t digest_low,uint64_t digest_high)
{
	SparkGlm52RingKvKey key;
	key.low = digest_low;
	key.high = (digest_high | 2ull) & ~1ull;
	return key;
}

static SparkGlm52RingKvKey *SparkGlm52RingWorkControlKvIndexKeyAt(void *entries,uint32_t entry_bytes,uint32_t slot)
{
	return (SparkGlm52RingKvKey *)((uint8_t *)entries + ((size_t)slot * (size_t)entry_bytes));
}

// Both key domains carry an avalanched low half - private keys are a splitmix
// output, content keys a caller digest - so the home slot is a mask, not another
// mix. This runs on every probe of every block of every lane.
static uint32_t SparkGlm52RingWorkControlKvIndexHome(SparkGlm52RingKvKey key,uint32_t mask)
{
	return (uint32_t)(key.low & (uint64_t)mask);
}

// Linear probe over a power-of-two open-addressed table whose entries begin with
// a SparkGlm52RingKvKey. Returns the slot to occupy when the key is absent, so
// one implementation serves both the sequence directory and the block table.
static uint32_t SparkGlm52RingWorkControlKvIndexProbe(void *entries,uint32_t entry_bytes,uint32_t capacity,SparkGlm52RingKvKey key,uint32_t *found_out)
{
	const SparkGlm52RingKvKey *slot_key;
	uint32_t mask,slot,probes;
	mask = capacity - 1u;
	slot = SparkGlm52RingWorkControlKvIndexHome(key,mask);
	*found_out = 0u;
	for (probes = 0u; probes <= mask; ++probes)
	{
		slot_key = SparkGlm52RingWorkControlKvIndexKeyAt(entries,entry_bytes,slot);
		if (SparkGlm52RingWorkControlKvKeyEmpty(*slot_key) != 0u)
			return slot;
		if (SparkGlm52RingWorkControlKvKeyEqual(*slot_key,key) != 0u)
		{
			*found_out = 1u;
			return slot;
		}
		slot = (slot + 1u) & mask;
	}
	return UINT32_MAX;
}

// Backward-shift deletion keeps every probe chain contiguous, so no tombstone
// state is needed and probe length stays governed only by load factor. The
// scan is bounded by a full revolution so a saturated table cannot spin.
static void SparkGlm52RingWorkControlKvIndexErase(void *entries,uint32_t entry_bytes,uint32_t capacity,uint32_t slot)
{
	const SparkGlm52RingKvKey *scan_key;
	uint8_t *base;
	uint32_t mask,hole,scan,home,hole_distance,scan_distance;
	base = (uint8_t *)entries;
	mask = capacity - 1u;
	hole = slot;
	memset(base + ((size_t)hole * entry_bytes),0,entry_bytes);
	for (scan = (hole + 1u) & mask; scan != hole; scan = (scan + 1u) & mask)
	{
		scan_key = SparkGlm52RingWorkControlKvIndexKeyAt(entries,entry_bytes,scan);
		if (SparkGlm52RingWorkControlKvKeyEmpty(*scan_key) != 0u)
			return;
		home = SparkGlm52RingWorkControlKvIndexHome(*scan_key,mask);
		hole_distance = (hole - home) & mask;
		scan_distance = (scan - home) & mask;
		if (hole_distance >= scan_distance)
			continue;
		memcpy(base + ((size_t)hole * entry_bytes),base + ((size_t)scan * entry_bytes),entry_bytes);
		memset(base + ((size_t)scan * entry_bytes),0,entry_bytes);
		hole = scan;
	}
}

// Slot of an existing key, or UINT32_MAX. Load factor is held at one half, so
// an exhausted probe and an absent key are the same answer to every caller.
static uint32_t SparkGlm52RingWorkControlKvIndexFind(void *entries,uint32_t entry_bytes,uint32_t capacity,SparkGlm52RingKvKey key)
{
	uint32_t slot,found;
	slot = SparkGlm52RingWorkControlKvIndexProbe(entries,entry_bytes,capacity,key,&found);
	return found != 0u ? slot : UINT32_MAX;
}

#define SPARK_GLM52_KV_FIND_DIRECTORY(state,sequence_id,logical_block_index) \
	SparkGlm52RingWorkControlKvIndexFind((state)->directory_entries,(uint32_t)sizeof((state)->directory_entries[0]),(state)->directory_capacity,SparkGlm52RingWorkControlPrivateKey((sequence_id),(logical_block_index)))

// The block record a sequence slot names, or 0 when the slot is not held.
// Every caller that walks a packet wants this, not the two lookups separately.
static SparkGlm52RingWorkControlKvBlockEntry *SparkGlm52RingWorkControlKvSequenceBlock(const SparkGlm52RingWorkControlKvState *state,uint64_t sequence_id,uint32_t logical_block_index);

#define SPARK_GLM52_KV_FIND_BLOCK(state,block_key) \
	SparkGlm52RingWorkControlKvIndexFind((state)->block_entries,(uint32_t)sizeof((state)->block_entries[0]),(state)->block_entry_capacity,(block_key))

static SparkGlm52RingWorkControlKvBlockEntry *SparkGlm52RingWorkControlKvSequenceBlock(const SparkGlm52RingWorkControlKvState *state,uint64_t sequence_id,uint32_t logical_block_index)
{
	uint32_t slot;
	slot = SPARK_GLM52_KV_FIND_DIRECTORY(state,sequence_id,logical_block_index);
	if (slot == UINT32_MAX)
		return 0;
	return &state->block_entries[SPARK_GLM52_KV_FIND_BLOCK(state,state->directory_entries[slot].block_key)];
}

SparkStatus SparkGlm52RingWorkControlInitializeKvState(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t physical_block_capacity,
	uint32_t directory_capacity,
	uint32_t block_entry_capacity,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts,
	uint8_t *physical_block_states,
	SparkGlm52RingKvKey *physical_block_keys,
	uint64_t *physical_block_last_used_epochs,
	SparkGlm52RingWorkControlKvDirectoryEntry *directory_entries,
	SparkGlm52RingWorkControlKvBlockEntry *block_entries)
{
	uint64_t table_entry_capacity;

	if (state == 0 ||
		lane_capacity == 0u ||
		lane_stride == 0u ||
		block_token_count == 0u ||
		physical_block_capacity == 0u ||
		physical_block_capacity > UINT32_MAX / 2u ||
		directory_capacity < physical_block_capacity * 2u ||
		(directory_capacity & (directory_capacity - 1u)) != 0u ||
		block_entry_capacity < physical_block_capacity * 2u ||
		(block_entry_capacity & (block_entry_capacity - 1u)) != 0u ||
		physical_block_indices == 0 ||
		lane_physical_block_counts == 0 ||
		physical_block_states == 0 ||
		physical_block_keys == 0 ||
		physical_block_last_used_epochs == 0 ||
		directory_entries == 0 ||
		block_entries == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	table_entry_capacity = (uint64_t)lane_capacity * (uint64_t)lane_stride;
	if (table_entry_capacity > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(state,0,sizeof(*state));
	state->abi_version = SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION;
	state->descriptor_bytes = SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES;
	state->lane_capacity = lane_capacity;
	state->lane_stride = lane_stride;
	state->block_token_count = block_token_count;
	state->table_entry_capacity = (uint32_t)table_entry_capacity;
	state->physical_block_capacity = physical_block_capacity;
	state->directory_capacity = directory_capacity;
	state->block_entry_capacity = block_entry_capacity;
	state->free_backing_block_head = SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
	state->physical_block_indices = physical_block_indices;
	state->lane_physical_block_counts = lane_physical_block_counts;
	state->physical_block_states = physical_block_states;
	state->physical_block_keys = physical_block_keys;
	state->physical_block_last_used_epochs = physical_block_last_used_epochs;
	state->directory_entries = directory_entries;
	state->block_entries = block_entries;
	memset(state->physical_block_states,SPARK_GLM52_RING_KV_ENTRY_MISSING,
		state->physical_block_capacity * sizeof(state->physical_block_states[0]));
	memset(state->physical_block_keys,0,
		state->physical_block_capacity * sizeof(state->physical_block_keys[0]));
	memset(state->physical_block_last_used_epochs,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_last_used_epochs[0]));
	memset(state->directory_entries,0,
		state->directory_capacity * sizeof(state->directory_entries[0]));
	memset(state->block_entries,0,
		state->block_entry_capacity * sizeof(state->block_entries[0]));
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlConfigureKvSwap(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t backing_block_capacity,
	uint32_t *backing_block_free_next,
	SparkGlm52RingWorkControlKvSwapStoreFunction swap_store_function,
	SparkGlm52RingWorkControlKvSwapLoadFunction swap_load_function,
	void *swap_context)
{
	uint32_t backing_block_index;

	if (state == 0 ||
		state->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES ||
		backing_block_capacity == 0u ||
		backing_block_capacity > state->directory_capacity / 2u ||
		backing_block_free_next == 0 ||
		swap_store_function == 0 || swap_load_function == 0 ||
		swap_context == 0 || state->directory_entry_count != 0u ||
		state->allocated_physical_block_count != 0u ||
		state->backing_block_capacity != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (backing_block_index = 0u;
		 backing_block_index < backing_block_capacity;
		 ++backing_block_index)
	{
		backing_block_free_next[backing_block_index] =
			backing_block_index + 1u < backing_block_capacity
				? backing_block_index + 1u
				: SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
	}
	state->backing_block_capacity = backing_block_capacity;
	state->free_backing_block_head = 0u;
	state->backing_block_free_next = backing_block_free_next;
	state->swap_store_function = swap_store_function;
	state->swap_load_function = swap_load_function;
	state->swap_context = swap_context;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlConfigureKvPins(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t *physical_block_pin_counts)
{
	if (state == 0 || physical_block_pin_counts == 0 ||
		state->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_pin_counts != 0 ||
		state->directory_entry_count != 0u ||
		state->allocated_physical_block_count != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(physical_block_pin_counts,0,
		state->physical_block_capacity * sizeof(physical_block_pin_counts[0]));
	state->physical_block_pin_counts = physical_block_pin_counts;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlConfigureKvSharing(
	SparkGlm52RingWorkControlKvState *state,
	const SparkGlm52RingKvKey *lane_block_keys,
	uint32_t lane_block_key_stride)
{
	if (state == 0 ||
		state->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION ||
		(lane_block_keys == 0) != (lane_block_key_stride == 0u) ||
		lane_block_key_stride > state->lane_stride)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->lane_block_keys = lane_block_keys;
	state->lane_block_key_stride = lane_block_key_stride;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlPinPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		SparkGlm52RingWorkControlKvKeyEmpty(
			state->physical_block_keys[physical_block_index]) != 0u ||
		state->physical_block_pin_counts[physical_block_index] == UINT32_MAX)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] += 1u;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlUnpinPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		state->physical_block_pin_counts[physical_block_index] == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] -= 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlValidateKvState(
	const SparkGlm52RingWorkControlPacket *packet,
	const SparkGlm52RingWorkControlKvState *state)
{
	if (packet == 0 || state == 0 ||
		state->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_indices == 0 ||
		state->lane_physical_block_counts == 0 ||
		state->physical_block_states == 0 ||
		state->physical_block_keys == 0 ||
		state->physical_block_last_used_epochs == 0 ||
		state->directory_entries == 0 ||
		state->block_entries == 0 ||
		state->lane_capacity == 0u ||
		state->lane_stride == 0u ||
		state->block_token_count == 0u ||
		(uint64_t)state->table_entry_capacity !=
			(uint64_t)state->lane_capacity * state->lane_stride ||
		state->physical_block_capacity == 0u ||
		state->physical_block_capacity > UINT32_MAX / 2u ||
		state->directory_capacity < state->physical_block_capacity * 2u ||
		(state->directory_capacity & (state->directory_capacity - 1u)) != 0u ||
		state->block_entry_capacity < state->physical_block_capacity * 2u ||
		(state->block_entry_capacity & (state->block_entry_capacity - 1u)) != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->backing_block_capacity == 0u) !=
		(state->backing_block_free_next == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_store_function == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_load_function == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_context == 0) ||
		state->backing_block_capacity > state->block_entry_capacity / 2u ||
		state->block_entry_count > (state->backing_block_capacity != 0u
			? state->backing_block_capacity : state->physical_block_capacity) ||
		state->directory_entry_count > state->directory_capacity / 2u ||
		state->allocated_physical_block_count > state->physical_block_capacity ||
		state->swapped_block_count > state->block_entry_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkGlm52RingWorkControlValidatePacket(
			packet,
			state->lane_capacity,
			UINT32_MAX) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->block_token_count != state->block_token_count ||
		packet->max_blocks_per_sequence > state->lane_stride ||
		packet->active_sequence_count > state->lane_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingWorkControlResetBackingBlocks(
	SparkGlm52RingWorkControlKvState *state)
{
	uint32_t backing_block_index;

	if (state->backing_block_capacity == 0u)
	{
		state->free_backing_block_head =
			SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
		return;
	}
	for (backing_block_index = 0u;
		 backing_block_index < state->backing_block_capacity;
		 ++backing_block_index)
	{
		state->backing_block_free_next[backing_block_index] =
			backing_block_index + 1u < state->backing_block_capacity
				? backing_block_index + 1u
				: SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
	}
	state->free_backing_block_head = 0u;
}

static void SparkGlm52RingWorkControlResetKvGeneration(
	SparkGlm52RingWorkControlKvState *state,
	uint64_t control_generation)
{
	memset(state->physical_block_indices,0xff,
		state->table_entry_capacity * sizeof(state->physical_block_indices[0]));
	memset(state->lane_physical_block_counts,0,
		state->lane_capacity * sizeof(state->lane_physical_block_counts[0]));
	memset(state->physical_block_states,SPARK_GLM52_RING_KV_ENTRY_MISSING,
		state->physical_block_capacity * sizeof(state->physical_block_states[0]));
	memset(state->physical_block_keys,0,
		state->physical_block_capacity * sizeof(state->physical_block_keys[0]));
	memset(state->physical_block_last_used_epochs,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_last_used_epochs[0]));
	if (state->physical_block_pin_counts != 0)
		memset(state->physical_block_pin_counts,0,
			state->physical_block_capacity *
				sizeof(state->physical_block_pin_counts[0]));
	memset(state->directory_entries,0,
		state->directory_capacity * sizeof(state->directory_entries[0]));
	memset(state->block_entries,0,
		state->block_entry_capacity * sizeof(state->block_entries[0]));
	SparkGlm52RingWorkControlResetBackingBlocks(state);
	state->next_physical_block_index = 0u;
	state->directory_entry_count = 0u;
	state->block_entry_count = 0u;
	state->swapped_block_count = 0u;
	state->epoch = 0u;
	state->missing_block_count = 0u;
	state->in_flight_block_count = 0u;
	state->resident_block_count = 0u;
	state->allocated_physical_block_count = 0u;
	state->control_generation = control_generation;
	state->control_generation_reset_count += 1u;
}

SparkStatus SparkGlm52RingWorkControlAdvanceKvGeneration(
	SparkGlm52RingWorkControlKvState *state,
	uint64_t control_generation)
{
	if (state == 0 || control_generation == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (control_generation < state->control_generation)
		return SPARK_STATUS_VALIDATION_FAILED;
	if (control_generation != state->control_generation)
		SparkGlm52RingWorkControlResetKvGeneration(state,control_generation);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlSelectKvGeneration(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state)
{
	SparkStatus status;
	status = SparkGlm52RingWorkControlAdvanceKvGeneration(
		state,packet->control_generation);
	return status == SPARK_STATUS_VALIDATION_FAILED
		? SPARK_STATUS_NOT_FOUND : status;
}

static void SparkGlm52RingWorkControlResetReadinessCounts(
	SparkGlm52RingWorkControlKvState *state)
{
	state->missing_block_count = 0u;
	state->in_flight_block_count = 0u;
	state->resident_block_count = 0u;
}

static void SparkGlm52RingWorkControlAccountReadiness(
	SparkGlm52RingWorkControlKvState *state,
	uint8_t entry_state)
{
	if (entry_state == SPARK_GLM52_RING_KV_ENTRY_RESIDENT)
		state->resident_block_count += 1u;
	else if (entry_state == SPARK_GLM52_RING_KV_ENTRY_IN_FLIGHT)
		state->in_flight_block_count += 1u;
	else
		state->missing_block_count += 1u;
}



SparkStatus SparkGlm52RingWorkControlCollectKvPrefetchEntries(
	const SparkGlm52RingWorkControlPacket *packets,
	uint32_t packet_count,
	SparkGlm52RingWorkControlKvState *state,
	SparkGlm52RingWorkControlKvPrefetchEntry *entries,
	uint32_t entry_capacity,
	uint32_t *entry_count_out)
{
	uint32_t packet_index,lane_index,block_index,entry_count,mark;
	if (packets == 0 || packet_count == 0u || state == 0 || entries == 0 ||
		entry_capacity == 0u || entry_count_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	entry_count = 0u;
	state->prefetch_generation += 1u;
	mark = state->prefetch_generation & 0x1FFFFFFFu;
	for (packet_index = 0u; packet_index < packet_count; ++packet_index)
	{
		const SparkGlm52RingWorkControlPacket *packet;
		SparkStatus status;
		packet = &packets[packet_index];
		status = SparkGlm52RingWorkControlValidateKvState(packet,state);
		if (status != SPARK_STATUS_OK)
			return status;
		if (packet->control_generation != state->control_generation)
			continue;
		if ((packet->flags &
			SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u)
			continue;
		for (lane_index = 0u; lane_index < packet->active_sequence_count;
			 ++lane_index)
		{
			uint32_t block_count;
			block_count = SparkGlm52RingWorkControlBlockCount(
				packet->lanes[lane_index].context_token_count,
				packet->block_token_count);
			for (block_index = 0u; block_index < block_count; ++block_index)
			{
				SparkGlm52RingWorkControlKvBlockEntry *block_entry;
				block_entry = SparkGlm52RingWorkControlKvSequenceBlock(state,packet->lanes[lane_index].sequence_id,block_index);
				if (block_entry == 0)
					continue;
				// Sharers reach one record many times. Marking the record dedupes
				// in constant time instead of rescanning what has been emitted.
				if (block_entry->residency_state !=
						SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_NVME ||
					block_entry->backing_valid == 0u ||
					block_entry->prefetch_mark == mark)
					continue;
				block_entry->prefetch_mark = mark;
				if (entry_count >= entry_capacity)
				{
					*entry_count_out = entry_count;
					return SPARK_STATUS_CAPACITY_EXCEEDED;
				}
				entries[entry_count].key = block_entry->key;
				entries[entry_count].backing_block_index =
					block_entry->backing_block_index;
				entry_count += 1u;
			}
		}
	}
	*entry_count_out = entry_count;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlKvBackingAcquire(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t *backing_block_index_out)
{
	uint32_t backing_block_index;

	if (state == 0 || backing_block_index_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->backing_block_capacity == 0u)
	{
		*backing_block_index_out = SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
		return SPARK_STATUS_OK;
	}
	backing_block_index = state->free_backing_block_head;
	if (backing_block_index >= state->backing_block_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->free_backing_block_head =
		state->backing_block_free_next[backing_block_index];
	state->backing_block_free_next[backing_block_index] =
		SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
	*backing_block_index_out = backing_block_index;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlKvBackingRelease(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t backing_block_index)
{
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->backing_block_capacity == 0u)
		return backing_block_index == SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX
			? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	if (backing_block_index >= state->backing_block_capacity ||
		state->backing_block_free_next[backing_block_index] !=
			SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX)
		return SPARK_STATUS_INTERNAL_ERROR;
	state->backing_block_free_next[backing_block_index] =
		state->free_backing_block_head;
	state->free_backing_block_head = backing_block_index;
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingWorkControlKvClearPhysicalBlock(SparkGlm52RingWorkControlKvState *state,uint32_t physical_block_index)
{
	state->physical_block_keys[physical_block_index].low = 0u;
	state->physical_block_keys[physical_block_index].high = 0u;
	state->physical_block_last_used_epochs[physical_block_index] = 0u;
	state->physical_block_states[physical_block_index] = SPARK_GLM52_RING_KV_ENTRY_MISSING;
}

static void SparkGlm52RingWorkControlKvAssignPhysicalBlock(SparkGlm52RingWorkControlKvState *state,uint32_t physical_block_index,SparkGlm52RingKvKey key,uint8_t physical_state)
{
	state->physical_block_keys[physical_block_index] = key;
	state->physical_block_last_used_epochs[physical_block_index] = state->epoch;
	state->physical_block_states[physical_block_index] = physical_state;
	state->allocated_physical_block_count += 1u;
}

// Write a resident block out to its backing slot and mark the record swapped.
static SparkStatus SparkGlm52RingWorkControlKvSpillBlock(SparkGlm52RingWorkControlKvState *state,SparkGlm52RingWorkControlKvBlockEntry *entry,uint32_t physical_block_index)
{
	SparkStatus status;
	if (entry->backing_valid == 0u)
	{
		status = state->swap_store_function(state->swap_context,entry->key,physical_block_index,entry->backing_block_index);
		if (status != SPARK_STATUS_OK)
			return status;
		entry->backing_valid = 1u;
		state->swap_store_count += 1u;
	}
	else
		state->clean_evict_count += 1u;
	entry->physical_block_index = SPARK_GLM52_RING_KV_INVALID_BLOCK_INDEX;
	entry->residency_state = SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_NVME;
	SparkGlm52RingWorkControlKvClearPhysicalBlock(state,physical_block_index);
	state->allocated_physical_block_count -= 1u;
	state->swapped_block_count += 1u;
	return SPARK_STATUS_OK;
}

// Clock sweep. A free slot wins outright, otherwise the first unpinned resident
// block outside the current epoch is spilled. The reverse map is the block key
// carried on the physical block, so the cost is one probe however many sequences
// share the block, and every sharer observes the spill because residency lives
// only on the block record.
static SparkStatus SparkGlm52RingWorkControlKvAcquirePhysicalBlock(SparkGlm52RingWorkControlKvState *state,uint32_t *physical_block_index_out)
{
	SparkGlm52RingWorkControlKvBlockEntry *entry;
	uint32_t physical_block_index,scan_count,slot;
	SparkStatus status;
	for (scan_count = 0u; scan_count < state->physical_block_capacity; ++scan_count)
	{
		physical_block_index = state->next_physical_block_index;
		state->next_physical_block_index = (physical_block_index + 1u) % state->physical_block_capacity;
		if (state->physical_block_pin_counts != 0 && state->physical_block_pin_counts[physical_block_index] != 0u)
			continue;
		if (SparkGlm52RingWorkControlKvKeyEmpty(state->physical_block_keys[physical_block_index]) != 0u)
		{
			*physical_block_index_out = physical_block_index;
			return SPARK_STATUS_OK;
		}
		if (state->backing_block_capacity == 0u ||
			state->physical_block_states[physical_block_index] != SPARK_GLM52_RING_KV_ENTRY_RESIDENT ||
			state->physical_block_last_used_epochs[physical_block_index] == state->epoch)
			continue;
		slot = SPARK_GLM52_KV_FIND_BLOCK(state,state->physical_block_keys[physical_block_index]);
		entry = &state->block_entries[slot];
		status = SparkGlm52RingWorkControlKvSpillBlock(state,entry,physical_block_index);
		if (status != SPARK_STATUS_OK)
			return status;
		*physical_block_index_out = physical_block_index;
		return SPARK_STATUS_OK;
	}
	return SPARK_STATUS_CAPACITY_EXCEEDED;
}

SparkStatus SparkGlm52RingWorkControlAcquireTransientPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t *physical_block_index_out)
{
	uint32_t physical_block_index;
	SparkStatus status;
	if (state == 0 || physical_block_index_out == 0 ||
		state->physical_block_pin_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingWorkControlKvAcquirePhysicalBlock(
		state,&physical_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52RingWorkControlKvAssignPhysicalBlock(state,physical_block_index,
		SparkGlm52RingWorkControlPrivateKey(UINT64_MAX,physical_block_index),
		SPARK_GLM52_RING_KV_ENTRY_TRANSIENT);
	state->physical_block_pin_counts[physical_block_index] = 1u;
	*physical_block_index_out = physical_block_index;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlReleaseTransientPhysicalBlock(
	SparkGlm52RingWorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		state->physical_block_states[physical_block_index] !=
			SPARK_GLM52_RING_KV_ENTRY_TRANSIENT ||
		SparkGlm52RingWorkControlKvKeyEqual(
			state->physical_block_keys[physical_block_index],
			SparkGlm52RingWorkControlPrivateKey(UINT64_MAX,physical_block_index)) == 0u ||
		state->physical_block_pin_counts[physical_block_index] != 1u ||
		state->allocated_physical_block_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] = 0u;
	SparkGlm52RingWorkControlKvClearPhysicalBlock(
		state,physical_block_index);
	state->allocated_physical_block_count -= 1u;
	return SPARK_STATUS_OK;
}

// Take a reference on the block a key names, admitting a physical block and a
// backing slot on the first reference and sharing them on every later one.
static SparkStatus SparkGlm52RingWorkControlKvBlockResolve(SparkGlm52RingWorkControlKvState *state,SparkGlm52RingKvKey block_key)
{
	SparkGlm52RingWorkControlKvBlockEntry *entry;
	uint32_t slot,found,physical_block_index,backing_block_index;
	SparkStatus status;
	slot = SparkGlm52RingWorkControlKvIndexProbe(state->block_entries,(uint32_t)sizeof(state->block_entries[0]),state->block_entry_capacity,block_key,&found);
	if (slot == UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	entry = &state->block_entries[slot];
	if (found != 0u)
	{
		entry->reference_count += 1u;
		state->share_hit_count += 1u;
		return SPARK_STATUS_OK;
	}
	if (state->block_entry_count >= (state->backing_block_capacity != 0u ? state->backing_block_capacity : state->physical_block_capacity))
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52RingWorkControlKvBackingAcquire(state,&backing_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingWorkControlKvAcquirePhysicalBlock(state,&physical_block_index);
	if (status != SPARK_STATUS_OK)
	{
		(void)SparkGlm52RingWorkControlKvBackingRelease(state,backing_block_index);
		return status;
	}
	entry = &state->block_entries[slot];
	entry->key = block_key;
	entry->physical_block_index = physical_block_index;
	entry->backing_block_index = backing_block_index;
	entry->reference_count = 1u;
	entry->residency_state = SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU;
	entry->backing_valid = 0u;
	state->block_entry_count += 1u;
	state->share_admit_count += 1u;
	SparkGlm52RingWorkControlKvAssignPhysicalBlock(state,physical_block_index,block_key,SPARK_GLM52_RING_KV_ENTRY_MISSING);
	return SPARK_STATUS_OK;
}

// Drop a reference, freeing the physical and backing blocks at zero. The
// decrement is guarded so an unbalanced release reports rather than wrapping to
// UINT32_MAX and stranding the block forever.
static SparkStatus SparkGlm52RingWorkControlKvBlockDeref(SparkGlm52RingWorkControlKvState *state,SparkGlm52RingKvKey block_key)
{
	SparkGlm52RingWorkControlKvBlockEntry *entry;
	uint32_t slot;
	SparkStatus status;
	slot = SPARK_GLM52_KV_FIND_BLOCK(state,block_key);
	entry = &state->block_entries[slot];
	if (entry->reference_count > 1u)
	{
		entry->reference_count -= 1u;
		return SPARK_STATUS_OK;
	}
	if (entry->residency_state == SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU)
	{
		if (state->physical_block_pin_counts != 0 && state->physical_block_pin_counts[entry->physical_block_index] != 0u)
			return SPARK_STATUS_BUSY;
		SparkGlm52RingWorkControlKvClearPhysicalBlock(state,entry->physical_block_index);
		state->allocated_physical_block_count -= 1u;
	}
	else
		state->swapped_block_count -= 1u;
	status = SparkGlm52RingWorkControlKvBackingRelease(state,entry->backing_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	entry->reference_count = 0u;
	state->block_entry_count -= 1u;
	SparkGlm52RingWorkControlKvIndexErase(state->block_entries,(uint32_t)sizeof(state->block_entries[0]),state->block_entry_capacity,slot);
	return SPARK_STATUS_OK;
}

// Change a block record's identity in place. Used when a private block becomes
// the first block to publish its content, so the bytes already computed are kept
// and no other sequence has to recompute them.
static void SparkGlm52RingWorkControlKvRekeyBlock(SparkGlm52RingWorkControlKvState *state,uint32_t source_slot,SparkGlm52RingKvKey block_key)
{
	SparkGlm52RingWorkControlKvBlockEntry record;
	uint32_t slot,found;
	record = state->block_entries[source_slot];
	SparkGlm52RingWorkControlKvIndexErase(state->block_entries,(uint32_t)sizeof(state->block_entries[0]),state->block_entry_capacity,source_slot);
	record.key = block_key;
	slot = SparkGlm52RingWorkControlKvIndexProbe(state->block_entries,(uint32_t)sizeof(state->block_entries[0]),state->block_entry_capacity,block_key,&found);
	state->block_entries[slot] = record;
	if (record.residency_state == SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU)
		state->physical_block_keys[record.physical_block_index] = block_key;
}

// Move a sequence slot from its private block onto the shared block holding the
// same content, or publish the private block under that content key when no
// sharer exists yet. Refused while the private block carries an in-flight
// speculative write, so a rejected draft can never reach another sequence.
static SparkStatus SparkGlm52RingWorkControlKvPromoteBlock(SparkGlm52RingWorkControlKvState *state,SparkGlm52RingWorkControlKvDirectoryEntry *entry,SparkGlm52RingKvKey block_key)
{
	SparkGlm52RingWorkControlKvBlockEntry *previous;
	uint32_t source_slot;
	SparkStatus status;
	source_slot = SPARK_GLM52_KV_FIND_BLOCK(state,entry->block_key);
	previous = &state->block_entries[source_slot];
	if (previous->residency_state == SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU &&
		state->physical_block_pin_counts != 0 &&
		state->physical_block_pin_counts[previous->physical_block_index] != 0u)
		return SPARK_STATUS_OK;
	if (SPARK_GLM52_KV_FIND_BLOCK(state,block_key) == UINT32_MAX && previous->reference_count == 1u)
	{
		SparkGlm52RingWorkControlKvRekeyBlock(state,source_slot,block_key);
		entry->block_key = block_key;
		return SPARK_STATUS_OK;
	}
	status = SparkGlm52RingWorkControlKvBlockResolve(state,block_key);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingWorkControlKvBlockDeref(state,entry->block_key);
	if (status != SPARK_STATUS_OK)
		return status;
	entry->block_key = block_key;
	return SPARK_STATUS_OK;
}

// Return the physical block a key resides in, faulting it back from backing
// storage when the clock sweep has spilled it.
static SparkStatus SparkGlm52RingWorkControlKvBlockResident(SparkGlm52RingWorkControlKvState *state,SparkGlm52RingKvKey block_key,uint32_t *physical_block_index_out)
{
	SparkGlm52RingWorkControlKvBlockEntry *entry;
	uint32_t slot,physical_block_index;
	SparkStatus status;
	slot = SPARK_GLM52_KV_FIND_BLOCK(state,block_key);
	entry = &state->block_entries[slot];
	if (entry->residency_state == SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU)
	{
		state->physical_block_last_used_epochs[entry->physical_block_index] = state->epoch;
		*physical_block_index_out = entry->physical_block_index;
		return SPARK_STATUS_OK;
	}
	status = SparkGlm52RingWorkControlKvAcquirePhysicalBlock(state,&physical_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	status = state->swap_load_function(state->swap_context,block_key,physical_block_index,entry->backing_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	state->swapped_block_count -= 1u;
	state->swap_load_count += 1u;
	entry->physical_block_index = physical_block_index;
	entry->residency_state = SPARK_GLM52_RING_KV_DIRECTORY_RESIDENCY_GPU;
	SparkGlm52RingWorkControlKvAssignPhysicalBlock(state,physical_block_index,block_key,SPARK_GLM52_RING_KV_ENTRY_RESIDENT);
	*physical_block_index_out = physical_block_index;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlKvDirectoryAcquire(SparkGlm52RingWorkControlKvState *state,uint64_t sequence_id,uint32_t logical_block_index,SparkGlm52RingKvKey block_key,uint32_t *physical_block_index_out)
{
	SparkGlm52RingWorkControlKvDirectoryEntry *entry;
	SparkGlm52RingKvKey directory_key;
	uint32_t slot,found;
	SparkStatus status;
	directory_key = SparkGlm52RingWorkControlPrivateKey(sequence_id,logical_block_index);
	slot = SparkGlm52RingWorkControlKvIndexProbe(state->directory_entries,(uint32_t)sizeof(state->directory_entries[0]),state->directory_capacity,directory_key,&found);
	if (slot == UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	entry = &state->directory_entries[slot];
	if (found == 0u)
	{
		if (state->directory_entry_count >= state->directory_capacity / 2u)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		status = SparkGlm52RingWorkControlKvBlockResolve(state,block_key);
		if (status != SPARK_STATUS_OK)
			return status;
		entry->key = directory_key;
		entry->block_key = block_key;
		state->directory_entry_count += 1u;
	}
	else if (SparkGlm52RingWorkControlKvKeyEqual(entry->block_key,block_key) == 0u)
	{
		status = SparkGlm52RingWorkControlKvPromoteBlock(state,entry,block_key);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SparkGlm52RingWorkControlKvBlockResident(state,entry->block_key,physical_block_index_out);
}


static SparkStatus SparkGlm52RingWorkControlKvDirectoryRelease(SparkGlm52RingWorkControlKvState *state,uint64_t sequence_id,uint32_t logical_block_index)
{
	SparkGlm52RingKvKey block_key;
	uint32_t slot;
	SparkStatus status;
	slot = SPARK_GLM52_KV_FIND_DIRECTORY(state,sequence_id,logical_block_index);
	if (slot == UINT32_MAX)
		return SPARK_STATUS_NOT_FOUND;
	block_key = state->directory_entries[slot].block_key;
	status = SparkGlm52RingWorkControlKvBlockDeref(state,block_key);
	if (status != SPARK_STATUS_OK)
		return status;
	state->directory_entry_count -= 1u;
	SparkGlm52RingWorkControlKvIndexErase(state->directory_entries,(uint32_t)sizeof(state->directory_entries[0]),state->directory_capacity,slot);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlMarkTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state,
	uint8_t entry_state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint32_t physical_block_index;
	SparkGlm52RingWorkControlKvBlockEntry *block_entry;
	SparkStatus status;

	status = SparkGlm52RingWorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if (packet->control_generation != state->control_generation)
		return SPARK_STATUS_NOT_FOUND;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52RingWorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			block_entry = SparkGlm52RingWorkControlKvSequenceBlock(state,packet->lanes[lane_index].sequence_id,block_index);
			if (block_entry == 0)
				return SPARK_STATUS_NOT_FOUND;
			physical_block_index = block_entry->physical_block_index;
			state->physical_block_states[physical_block_index] = entry_state;
			state->physical_block_last_used_epochs[physical_block_index] =
				state->epoch;
		}
	}
	return SPARK_STATUS_OK;
}

// Tokens of this lane that are committed: everything below the speculative and
// MTP draft tail. A block ending at or below this line may be shared. A block
// crossing it holds a draft that may yet be rejected, so it stays private and a
// rejected draft can never reach another sequence. An outstanding count above
// the context length yields zero, which shares nothing - the safe direction.
uint64_t SparkGlm52RingWorkControlKvCommittedFrontier(const SparkGlm52RingWorkControlLane *lane)
{
	uint64_t outstanding_token_count;
	outstanding_token_count = (uint64_t)lane->speculative_token_count + (uint64_t)lane->mtp_draft_token_count;
	return outstanding_token_count < (uint64_t)lane->context_token_count ? (uint64_t)lane->context_token_count - outstanding_token_count : 0u;
}

// Content key when the caller published one for a committed block, private key
// otherwise. The lane's key row and frontier are hoisted by the caller, so the
// per-block cost here is one compare.
static SparkGlm52RingKvKey SparkGlm52RingWorkControlKvSelectKey(const SparkGlm52RingKvKey *lane_keys,uint64_t sequence_id,uint32_t logical_block_index,uint64_t block_end_token,uint64_t committed_frontier)
{
	if (lane_keys != 0 && block_end_token <= committed_frontier &&
		SparkGlm52RingWorkControlKvKeyEmpty(lane_keys[logical_block_index]) == 0u)
		return SparkGlm52RingWorkControlContentKey(lane_keys[logical_block_index].low,lane_keys[logical_block_index].high);
	return SparkGlm52RingWorkControlPrivateKey(sequence_id,logical_block_index);
}

// The lane's published key row, or null when this lane cannot share.
static const SparkGlm52RingKvKey *SparkGlm52RingWorkControlKvLaneKeys(const SparkGlm52RingWorkControlKvState *state,uint32_t lane_index,uint32_t block_count)
{
	if (state->lane_block_keys == 0 || block_count > state->lane_block_key_stride)
		return 0;
	return &state->lane_block_keys[(size_t)lane_index * (size_t)state->lane_block_key_stride];
}

// Count what this packet would add, before anything is mutated. Directory
// entries are per sequence slot; physical blocks are per distinct block key, so
// sequences presenting a prefix that is already admitted cost no physical block
// at all. Lanes presenting the same brand new key in one packet are counted
// once each, which over-states the need and can only reject conservatively.
static SparkStatus SparkGlm52RingWorkControlKvCountAdmission(const SparkGlm52RingWorkControlKvState *state,const SparkGlm52RingWorkControlPacket *packet,uint32_t *new_entry_count_out,uint32_t *new_block_count_out)
{
	const SparkGlm52RingWorkControlLane *lane;
	const SparkGlm52RingKvKey *lane_keys;
	uint64_t block_end_token,committed_frontier;
	uint32_t lane_index,block_index,block_count,new_entry_count,new_block_count;
	new_entry_count = 0u;
	new_block_count = 0u;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		lane = &packet->lanes[lane_index];
		block_count = SparkGlm52RingWorkControlBlockCount(lane->context_token_count,packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence || block_count > state->physical_block_capacity)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		lane_keys = SparkGlm52RingWorkControlKvLaneKeys(state,lane_index,block_count);
		committed_frontier = SparkGlm52RingWorkControlKvCommittedFrontier(lane);
		block_end_token = 0u;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			block_end_token += (uint64_t)packet->block_token_count;
			new_entry_count += SPARK_GLM52_KV_FIND_DIRECTORY(state,lane->sequence_id,block_index) == UINT32_MAX ? 1u : 0u;
			new_block_count += SPARK_GLM52_KV_FIND_BLOCK(state,SparkGlm52RingWorkControlKvSelectKey(lane_keys,lane->sequence_id,block_index,block_end_token,committed_frontier)) == UINT32_MAX ? 1u : 0u;
		}
	}
	*new_entry_count_out = new_entry_count;
	*new_block_count_out = new_block_count;
	return SPARK_STATUS_OK;
}

// Reject a packet that cannot fit before anything is mutated. The cheap bound
// costs one multiply per lane and admits every packet that would fit even with
// no sharing at all; the probing count runs only when that bound is exceeded,
// which is exactly when sharing has to be measured to know whether it fits.
static SparkStatus SparkGlm52RingWorkControlKvAdmitPacket(const SparkGlm52RingWorkControlPacket *packet,SparkGlm52RingWorkControlKvState *state)
{
	uint32_t lane_index,block_count,total_block_count,new_entry_count,new_block_count;
	SparkStatus status;
	total_block_count = 0u;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52RingWorkControlBlockCount(packet->lanes[lane_index].context_token_count,packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		total_block_count += block_count;
	}
	if (state->directory_entry_count + total_block_count <= state->directory_capacity / 2u &&
		total_block_count <= state->physical_block_capacity)
		return SPARK_STATUS_OK;
	status = SparkGlm52RingWorkControlKvCountAdmission(state,packet,&new_entry_count,&new_block_count);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->directory_entry_count > state->directory_capacity / 2u ||
		new_entry_count > (state->directory_capacity / 2u) - state->directory_entry_count ||
		new_block_count > state->physical_block_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	return SPARK_STATUS_OK;
}

// Readiness state machine for one block of one lane. Prefill may start a missing
// block; decode requires every block but the last to be resident already.
static SparkStatus SparkGlm52RingWorkControlKvResolveEntryState(const SparkGlm52RingWorkControlPacket *packet,SparkGlm52RingWorkControlKvState *state,uint32_t block_index,uint32_t block_count,uint8_t *entry_state)
{
	if ((packet->flags & SPARK_GLM52_RING_WORK_CONTROL_FLAG_PREFILL) != 0u)
	{
		if (*entry_state == SPARK_GLM52_RING_KV_ENTRY_MISSING)
			*entry_state = SPARK_GLM52_RING_KV_ENTRY_IN_FLIGHT;
		return SPARK_STATUS_OK;
	}
	if (block_index + 1u < block_count && *entry_state != SPARK_GLM52_RING_KV_ENTRY_RESIDENT)
	{
		SparkGlm52RingWorkControlAccountReadiness(state,*entry_state);
		return SPARK_STATUS_BUSY;
	}
	if (block_index + 1u == block_count && *entry_state == SPARK_GLM52_RING_KV_ENTRY_MISSING)
		*entry_state = SPARK_GLM52_RING_KV_ENTRY_IN_FLIGHT;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52RingWorkControlKvBuildLane(const SparkGlm52RingWorkControlPacket *packet,SparkGlm52RingWorkControlKvState *state,uint32_t lane_index)
{
	const SparkGlm52RingWorkControlLane *lane;
	const SparkGlm52RingKvKey *lane_keys;
	uint64_t base_block_index,block_end_token,committed_frontier;
	uint32_t block_index,block_count,physical_block_index;
	uint8_t entry_state;
	SparkStatus status;
	lane = &packet->lanes[lane_index];
	block_count = SparkGlm52RingWorkControlBlockCount(lane->context_token_count,packet->block_token_count);
	if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	base_block_index = (uint64_t)lane_index * (uint64_t)state->lane_stride;
	if (base_block_index + block_count > state->table_entry_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	lane_keys = SparkGlm52RingWorkControlKvLaneKeys(state,lane_index,block_count);
	committed_frontier = SparkGlm52RingWorkControlKvCommittedFrontier(lane);
	state->lane_physical_block_counts[lane_index] = block_count;
	block_end_token = 0u;
	for (block_index = 0u; block_index < block_count; ++block_index)
	{
		block_end_token += (uint64_t)packet->block_token_count;
		status = SparkGlm52RingWorkControlKvDirectoryAcquire(state,lane->sequence_id,block_index,SparkGlm52RingWorkControlKvSelectKey(lane_keys,lane->sequence_id,block_index,block_end_token,committed_frontier),&physical_block_index);
		if (status != SPARK_STATUS_OK)
			return status;
		entry_state = state->physical_block_states[physical_block_index];
		status = SparkGlm52RingWorkControlKvResolveEntryState(packet,state,block_index,block_count,&entry_state);
		if (status != SPARK_STATUS_OK)
			return status;
		state->physical_block_states[physical_block_index] = entry_state;
		SparkGlm52RingWorkControlAccountReadiness(state,entry_state);
		state->physical_block_indices[base_block_index + block_index] = physical_block_index;
	}
	return SPARK_STATUS_OK;
}

static void SparkGlm52RingWorkControlKvFillBlockTableView(const SparkGlm52RingWorkControlPacket *packet,const SparkGlm52RingWorkControlKvState *state,SparkGlm52KvBlockTableView *view)
{
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
	view->host_lane_physical_block_counts = state->lane_physical_block_counts;
}

SparkStatus SparkGlm52RingWorkControlBuildHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state,
	SparkGlm52KvBlockTableView *view)
{
	uint32_t lane_index;
	SparkStatus status;
	if (view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52RingWorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingWorkControlSelectKvGeneration(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52RingWorkControlKvAdmitPacket(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(state->lane_physical_block_counts,0,packet->active_sequence_count * sizeof(state->lane_physical_block_counts[0]));
	state->epoch += 1u;
	if (state->epoch == 0u)
		state->epoch = 1u;
	SparkGlm52RingWorkControlResetReadinessCounts(state);
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		status = SparkGlm52RingWorkControlKvBuildLane(packet,state,lane_index);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	SparkGlm52RingWorkControlKvFillBlockTableView(packet,state,view);
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlCommitHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state)
{
	uint64_t write_end_token;
	uint32_t first_written_block;
	uint32_t last_written_block;
	SparkGlm52RingWorkControlKvBlockEntry *block_entry;
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t written_position_count;
	SparkStatus status;

	status = SparkGlm52RingWorkControlMarkTable(
		packet,
		state,
		SPARK_GLM52_RING_KV_ENTRY_RESIDENT);
	if (status != SPARK_STATUS_OK)
		return status;
	written_position_count =
		SparkGlm52RingWorkControlWrittenPositionCount(packet);
	if (written_position_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		write_end_token = packet->lanes[lane_index].sequence_position +
			(uint64_t)written_position_count - 1u;
		if (write_end_token >= packet->lanes[lane_index].context_token_count)
			return SPARK_STATUS_INVALID_ARGUMENT;
		first_written_block =
			(uint32_t)(packet->lanes[lane_index].sequence_position /
				packet->block_token_count);
		last_written_block =
			(uint32_t)(write_end_token / packet->block_token_count);
		for (block_index = first_written_block;
			 block_index <= last_written_block;
			 ++block_index)
		{
			block_entry = SparkGlm52RingWorkControlKvSequenceBlock(state,packet->lanes[lane_index].sequence_id,block_index);
			if (block_entry == 0)
				return SPARK_STATUS_NOT_FOUND;
			block_entry->backing_valid = 0u;
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlCancelHostKvBlockTable(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint32_t physical_block_index;
	SparkGlm52RingWorkControlKvBlockEntry *block_entry;
	SparkStatus status;

	status = SparkGlm52RingWorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if (packet->control_generation != state->control_generation)
		return SPARK_STATUS_NOT_FOUND;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52RingWorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			block_entry = SparkGlm52RingWorkControlKvSequenceBlock(state,packet->lanes[lane_index].sequence_id,block_index);
			if (block_entry == 0)
				continue;
			physical_block_index = block_entry->physical_block_index;
			if (state->physical_block_states[physical_block_index] ==
				SPARK_GLM52_RING_KV_ENTRY_IN_FLIGHT)
			{
				status = SparkGlm52RingWorkControlKvDirectoryRelease(
					state,packet->lanes[lane_index].sequence_id,block_index);
				if (status != SPARK_STATUS_OK)
					return status;
			}
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlReleaseSequence(
	SparkGlm52RingWorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_count)
{
	uint32_t logical_block_index;

	if (state == 0 || sequence_id == 0u || logical_block_count == 0u ||
		state->abi_version != SPARK_GLM52_RING_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_RING_WORK_CONTROL_KV_STATE_BYTES ||
		logical_block_count > state->lane_stride ||
		state->directory_entries == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (logical_block_index = 0u;
		 logical_block_index < logical_block_count;
		 ++logical_block_index)
	{
		SparkStatus status;
		status = SparkGlm52RingWorkControlKvDirectoryRelease(
			state,sequence_id,logical_block_index);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
			return status;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52RingWorkControlReleasePacketSequences(
	const SparkGlm52RingWorkControlPacket *packet,
	SparkGlm52RingWorkControlKvState *state)
{
	uint32_t lane_index;
	SparkStatus status;

	status = SparkGlm52RingWorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if ((packet->flags &
		SPARK_GLM52_RING_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->control_generation < state->control_generation)
		return SPARK_STATUS_OK;
	status = SparkGlm52RingWorkControlSelectKvGeneration(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		uint32_t logical_block_count;
		logical_block_count = SparkGlm52RingWorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		status = SparkGlm52RingWorkControlReleaseSequence(
			state,packet->lanes[lane_index].sequence_id,logical_block_count);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}
