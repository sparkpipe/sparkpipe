#include "sparkpipe/spark_glm52_pp13_work_control.h"
#include "sparkpipe/spark_glm52_mtp_tree.h"

#include <string.h>

uint32_t SparkGlm52Pp13WorkControlCalculatePacketBytes(
	uint32_t active_sequence_count)
{
	uint64_t packet_bytes;

	if (active_sequence_count == 0u ||
		active_sequence_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT)
		return 0u;
	packet_bytes =
		(uint64_t)SPARK_GLM52_PP13_WORK_CONTROL_PACKET_PREFIX_BYTES +
		((uint64_t)active_sequence_count *
		 (uint64_t)SPARK_GLM52_PP13_WORK_CONTROL_LANE_BYTES);
	return packet_bytes <= UINT32_MAX ? (uint32_t)packet_bytes : 0u;
}

SparkStatus SparkGlm52Pp13WorkControlSelectExecutionBatchBucket(
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
	if (SparkGlm52StagePlanBatchBucketIsSupported(batch_bucket) == 0u ||
		batch_bucket < batch_lane_or_row_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	*batch_bucket_out = batch_bucket;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlSelectMtpDraftBudget(
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

static void SparkGlm52Pp13WorkControlSetDecodeFlags(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t mtp_budget,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	if (mtp_budget != 0u)
		packet->flags |= SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_SPECULATIVE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY;
	if ((decode_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_MTP_TREE_VERIFY) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_TREE_VERIFY;
}

static void SparkGlm52Pp13WorkControlSetMtpResolutionFlag(
	SparkGlm52Pp13WorkControlPacket *packet)
{
	uint32_t lane_index;

	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		if (packet->lanes[lane_index].
			mtp_resolution_proposed_token_count != 0u)
		{
			packet->flags |= SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE;
			break;
		}
	}
}

static SparkStatus SparkGlm52Pp13WorkControlBuildDecodeLanes(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_verify,
	uint32_t mtp_budget,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	const SparkGlm52RequestApiDecodeDispatchLaneView *source_lane;
	SparkGlm52Pp13WorkControlLane *lane;
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

SparkStatus SparkGlm52Pp13WorkControlBuildDecodePacketRange(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t lane_offset,
	uint32_t lane_count,
	uint32_t speculative_token_index,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	uint32_t speculative_verify;
	uint32_t mtp_tree_verify;
	uint32_t mtp_budget;
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
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
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
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
		  speculative_token_index != 0u)) ||
		(speculative_verify == 0u && speculative_token_index != 0u))
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlSelectMtpDraftBudget(
		decode_dispatch->dispatch_kind,
		decode_dispatch->request_dispatch->flags,
		decode_dispatch->request_dispatch->mtp_draft_token_budget,
		&mtp_budget);
	if (status != SPARK_STATUS_OK)
		return status;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->control_generation =
		SPARK_GLM52_PP13_WORK_CONTROL_STANDALONE_GENERATION;
	packet->active_sequence_count = lane_count;
	packet->lane_count = packet->active_sequence_count;
	packet->descriptor_bytes = SparkGlm52Pp13WorkControlCalculatePacketBytes(
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
	status = SparkGlm52Pp13WorkControlSelectExecutionBatchBucket(
		decode_dispatch->request_dispatch,
		packet->lane_count,
		&packet->execution_batch_bucket);
	if (status != SPARK_STATUS_OK)
		return status;
	packet->new_token_count = speculative_verify != 0u
		? packet->rows_per_lane : mtp_budget + 1u;
	packet->priority = decode_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY;
	packet->mtp_draft_token_count = mtp_budget;
	SparkGlm52Pp13WorkControlSetDecodeFlags(decode_dispatch,mtp_budget,packet);
	status = SparkGlm52Pp13WorkControlBuildDecodeLanes(
		decode_dispatch,lane_offset,lane_count,speculative_verify,
		mtp_budget,packet);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13WorkControlSetMtpResolutionFlag(packet);
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

SparkStatus SparkGlm52Pp13WorkControlBuildDecodePacket(
	const SparkGlm52ServingDecodeDispatch *decode_dispatch,
	uint32_t speculative_token_index,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	if (decode_dispatch == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SparkGlm52Pp13WorkControlBuildDecodePacketRange(
		decode_dispatch,0u,decode_dispatch->active_sequence_count,
		speculative_token_index,packet);
}

SparkStatus SparkGlm52Pp13WorkControlBuildPrefillPacket(
	const SparkGlm52PromptPipelinePrefillDispatch *prefill_dispatch,
	uint32_t token_offset,
	uint32_t token_count,
	SparkGlm52Pp13WorkControlPacket *packet)
{
	const SparkGlm52RequestApiPrefillDispatchLaneView *source_lane;
	SparkGlm52Pp13WorkControlLane *destination_lane;
	uint32_t active_lane_count;
	uint32_t source_lane_index;
	uint32_t position;
	uint32_t lane_token_count;
	uint32_t execution_row_index;
	uint32_t row_offset;
	uint64_t execution_row_count;

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
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		prefill_dispatch->host_token_ids == 0 ||
		prefill_dispatch->host_token_stride == 0u ||
		token_offset >= prefill_dispatch->prompt_token_count ||
		token_count == 0u ||
		token_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET ||
		token_count > prefill_dispatch->prompt_token_count - token_offset)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(packet,0,sizeof(*packet));
	packet->magic = SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC;
	packet->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	packet->control_generation =
		SPARK_GLM52_PP13_WORK_CONTROL_STANDALONE_GENERATION;
	packet->flags = SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL;
	if ((prefill_dispatch->request_dispatch->flags &
			SPARK_GLM52_REQUEST_API_DISPATCH_FLAG_DSPARK_TAP_CAPTURE) != 0u)
		packet->flags |=
			SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE;
	packet->new_token_count = token_count;
	packet->priority = prefill_dispatch->request_dispatch->highest_priority;
	packet->block_token_count = SPARK_GLM52_KV_BLOCK_TOKENS;
	packet->max_blocks_per_sequence =
		SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY;
	active_lane_count = 0u;
	execution_row_index = 0u;
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
		lane_token_count = source_lane->prompt_token_count - token_offset;
		if (lane_token_count > token_count)
			lane_token_count = token_count;
		position = source_lane->prompt_token_offset + token_offset;
		destination_lane = &packet->lanes[active_lane_count];
		destination_lane->request_id = source_lane->request_id;
		destination_lane->sequence_id = source_lane->sequence_id;
		destination_lane->sequence_position = position;
		destination_lane->request_slot_index =
			source_lane->request_slot_index;
		destination_lane->context_token_count = position + lane_token_count;
		destination_lane->prefill_token_count = lane_token_count;
		destination_lane->input_token_id = prefill_dispatch->host_token_ids[
			((uint64_t)source_lane_index * prefill_dispatch->host_token_stride) +
			token_offset + lane_token_count - 1u];
		for (row_offset = 0u; row_offset < token_count; ++row_offset)
		{
			packet->prefill_token_ids[execution_row_index++] =
				row_offset < lane_token_count
					? prefill_dispatch->host_token_ids[
						((uint64_t)source_lane_index *
						 prefill_dispatch->host_token_stride) +
						token_offset + row_offset]
					: 0u;
		}
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
	execution_row_count = (uint64_t)active_lane_count * token_count;
	if (execution_row_count > UINT32_MAX ||
		execution_row_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		execution_row_index != (uint32_t)execution_row_count)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	packet->execution_row_count = (uint32_t)execution_row_count;
	if (SparkGlm52Pp13WorkControlSelectExecutionBatchBucket(
			prefill_dispatch->request_dispatch,
			packet->execution_row_count,
			&packet->execution_batch_bucket) != SPARK_STATUS_OK)
		return SPARK_STATUS_INVALID_ARGUMENT;
	packet->descriptor_bytes =
		SparkGlm52Pp13WorkControlCalculatePacketBytes(active_lane_count);
	packet->request_id = packet->lanes[0u].request_id;
	packet->sequence_id = packet->lanes[0u].sequence_id;
	packet->sequence_position = packet->lanes[0u].sequence_position;
	packet->input_token_id = packet->lanes[0u].input_token_id;
	return SPARK_STATUS_OK;
}

uint32_t SparkGlm52Pp13WorkControlBlockCount(
	uint32_t token_count,
	uint32_t block_token_count)
{
	if (token_count == 0u || block_token_count == 0u)
		return 0u;
	return (token_count + block_token_count - 1u) / block_token_count;
}

SparkStatus SparkGlm52Pp13WorkControlPlanExecutionChunks(
	uint32_t logical_lane_count,
	uint32_t rows_per_lane,
	uint32_t execution_row_capacity,
	uint32_t *maximum_lanes_per_chunk_out,
	uint32_t *chunk_count_out)
{
	uint32_t maximum_lanes_per_chunk;
	uint32_t chunk_count;

	if (logical_lane_count == 0u ||
		logical_lane_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT ||
		rows_per_lane == 0u || execution_row_capacity == 0u ||
		maximum_lanes_per_chunk_out == 0 || chunk_count_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	maximum_lanes_per_chunk = execution_row_capacity / rows_per_lane;
	if (maximum_lanes_per_chunk == 0u)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (maximum_lanes_per_chunk > logical_lane_count)
		maximum_lanes_per_chunk = logical_lane_count;
	chunk_count = logical_lane_count / maximum_lanes_per_chunk;
	if (logical_lane_count % maximum_lanes_per_chunk != 0u)
		chunk_count += 1u;
	*maximum_lanes_per_chunk_out = maximum_lanes_per_chunk;
	*chunk_count_out = chunk_count;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlValidatePacket(
	const SparkGlm52Pp13WorkControlPacket *packet,
	uint32_t max_active_sequence_count,
	uint32_t max_pipeline_slot_count)
{
	uint32_t token_index;
	uint32_t dspark_verify;
	uint32_t mtp_verify;
	uint32_t mtp_tree_verify;
	uint32_t speculative_verify;
	uint32_t lane_index;
	uint32_t row_offset;
	uint32_t token_id;
	uint32_t expected_rows_per_lane;
	uint64_t expected_execution_row_count;
	uint32_t release_sequences;
	uint32_t mtp_resolution_lane_count;

	if (packet == 0 ||
		packet->magic != SPARK_GLM52_PP13_WORK_CONTROL_PACKET_MAGIC ||
		packet->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & ~SPARK_GLM52_PP13_WORK_CONTROL_KNOWN_FLAGS) != 0u ||
		packet->request_id == 0u ||
		packet->sequence_id == 0u ||
		packet->control_generation == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	release_sequences = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) != 0u;
	if (release_sequences != 0u)
	{
		if (packet->flags !=
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES ||
			packet->descriptor_bytes !=
				SparkGlm52Pp13WorkControlCalculatePacketBytes(packet->lane_count) ||
			packet->active_sequence_count == 0u ||
			packet->active_sequence_count >
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT ||
			packet->lane_count != packet->active_sequence_count ||
			packet->lane_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT ||
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
			 token_index < SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
		for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
		{
			const SparkGlm52Pp13WorkControlLane *lane;
			lane = &packet->lanes[lane_index];
			if (lane->request_id == 0u || lane->sequence_id == 0u ||
				lane->context_token_count == 0u ||
				lane->context_token_count > packet->kv_block_table_token_count ||
				SparkGlm52Pp13WorkControlBlockCount(
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
				 token_index < SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
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
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT ||
		packet->lane_count == 0u ||
		packet->lane_count > SPARK_GLM52_PP13_WORK_CONTROL_MAX_LANE_COUNT ||
		packet->lane_count != packet->active_sequence_count ||
		packet->new_token_count == 0u ||
		packet->pipeline_slot >= max_pipeline_slot_count ||
		packet->block_token_count == 0u ||
		packet->kv_block_table_token_count == 0u ||
		packet->max_blocks_per_sequence == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->active_sequence_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_ACTIVE_SEQUENCE_COUNT ||
		packet->descriptor_bytes != SparkGlm52Pp13WorkControlCalculatePacketBytes(
			packet->active_sequence_count) ||
		packet->max_blocks_per_sequence >
			SPARK_GLM52_PP13_WORK_CONTROL_KV_BLOCK_CAPACITY)
		return SPARK_STATUS_ABI_MISMATCH;
	if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) == 0u &&
		packet->new_token_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT + 1u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u &&
		packet->new_token_count >
			SPARK_GLM52_PP13_WORK_CONTROL_MAX_PREFILL_TOKENS_PER_PACKET)
		return SPARK_STATUS_INVALID_ARGUMENT;
	dspark_verify = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_SPECULATIVE_VERIFY) != 0u;
	mtp_verify = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_SPECULATIVE_VERIFY) != 0u;
	mtp_tree_verify = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_TREE_VERIFY) != 0u;
	speculative_verify = dspark_verify | mtp_verify;
	expected_rows_per_lane = (packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
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
		((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
			? packet->execution_row_count : packet->lane_count) >
			packet->execution_batch_bucket)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->input_token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
		return SPARK_STATUS_INVALID_ARGUMENT;
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
		(((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u) !=
		 (packet->mtp_draft_token_count != 0u)))
		return SPARK_STATUS_INVALID_ARGUMENT;
	mtp_resolution_lane_count = 0u;
	if (speculative_verify != 0u)
	{
		if ((packet->flags &
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u ||
			packet->new_token_count != packet->rows_per_lane ||
			packet->speculative_token_count == 0u ||
			packet->speculative_token_count >
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT ||
			packet->speculative_token_index != 0u ||
			(dspark_verify != 0u &&
			 (packet->flags &
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_DSPARK_TAP_CAPTURE) == 0u))
			return SPARK_STATUS_INVALID_ARGUMENT;
		if (mtp_tree_verify != 0u)
		{
			if ((packet->flags &
					SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) == 0u ||
				packet->mtp_draft_token_count !=
					SPARK_GLM52_MODEL_MTP_TREE_CANDIDATE_COUNT)
				return SPARK_STATUS_MODULE_NOT_VALIDATED;
		}
		else if ((packet->flags &
					SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_DRAFT) != 0u ||
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
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
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
				SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
			 ++token_index)
		{
			if (packet->speculative_draft_token_ids[token_index] != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
		}
	}
	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		const SparkGlm52Pp13WorkControlLane *lane;
		uint32_t lane_position_count;
		lane = &packet->lanes[lane_index];
		lane_position_count =
			(packet->flags &
				SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u
			? lane->prefill_token_count
			: mtp_tree_verify != 0u
				? SPARK_GLM52_MODEL_MTP_TREE_CONTEXT_EXTENSION + 1u
				: packet->new_token_count;
		if (lane->request_id == 0u || lane->sequence_id == 0u ||
			lane->request_slot_index == SPARK_GLM52_PP13_WORK_CONTROL_INVALID_REQUEST_SLOT ||
			lane->context_token_count == 0u ||
			lane->sequence_position + (uint64_t)lane_position_count >
				lane->context_token_count ||
			lane->context_token_count > packet->kv_block_table_token_count ||
			SparkGlm52Pp13WorkControlBlockCount(
				lane->context_token_count,packet->block_token_count) >
				packet->max_blocks_per_sequence ||
			lane->input_token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT)
			return SPARK_STATUS_INVALID_ARGUMENT;
		if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
		{
			if (lane->prefill_token_count == 0u ||
				lane->prefill_token_count > packet->new_token_count ||
				lane->mtp_draft_token_count != 0u ||
				lane->speculative_token_count != 0u)
				return SPARK_STATUS_INVALID_ARGUMENT;
			for (row_offset = 0u; row_offset < packet->rows_per_lane; ++row_offset)
			{
				token_id = packet->prefill_token_ids[
					(lane_index * packet->rows_per_lane) + row_offset];
				if ((row_offset < lane->prefill_token_count &&
					 token_id >= SPARK_GLM52_MODEL_OUTPUT_VOCAB_COUNT) ||
					(row_offset >= lane->prefill_token_count && token_id != 0u))
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
		}
		else if (lane->prefill_token_count != 0u)
			return SPARK_STATUS_INVALID_ARGUMENT;
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
					SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE) == 0u ||
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
					SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
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
				 token_index < SPARK_GLM52_PP13_WORK_CONTROL_MAX_SPECULATIVE_TOKEN_COUNT;
				 ++token_index)
			{
				if (lane->speculative_draft_token_ids[token_index] != 0u)
					return SPARK_STATUS_INVALID_ARGUMENT;
			}
		}
	}
	if (((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_MTP_RESOLVE) != 0u) !=
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

SparkStatus SparkGlm52Pp13WorkControlInitializeKvState(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t lane_capacity,
	uint32_t lane_stride,
	uint32_t block_token_count,
	uint32_t physical_block_capacity,
	uint32_t directory_capacity,
	uint32_t *physical_block_indices,
	uint32_t *lane_physical_block_counts,
	uint8_t *physical_block_states,
	uint64_t *physical_block_sequence_ids,
	uint32_t *physical_block_logical_indices,
	uint64_t *physical_block_last_used_epochs,
	SparkGlm52Pp13WorkControlKvDirectoryEntry *directory_entries)
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
		physical_block_indices == 0 ||
		lane_physical_block_counts == 0 ||
		physical_block_states == 0 ||
		physical_block_sequence_ids == 0 ||
		physical_block_logical_indices == 0 ||
		physical_block_last_used_epochs == 0 ||
		directory_entries == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	table_entry_capacity = (uint64_t)lane_capacity * (uint64_t)lane_stride;
	if (table_entry_capacity > UINT32_MAX)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	memset(state,0,sizeof(*state));
	state->abi_version = SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION;
	state->descriptor_bytes = SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES;
	state->lane_capacity = lane_capacity;
	state->lane_stride = lane_stride;
	state->block_token_count = block_token_count;
	state->table_entry_capacity = (uint32_t)table_entry_capacity;
	state->physical_block_capacity = physical_block_capacity;
	state->directory_capacity = directory_capacity;
	state->free_backing_block_head = SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
	state->physical_block_indices = physical_block_indices;
	state->lane_physical_block_counts = lane_physical_block_counts;
	state->physical_block_states = physical_block_states;
	state->physical_block_sequence_ids = physical_block_sequence_ids;
	state->physical_block_logical_indices = physical_block_logical_indices;
	state->physical_block_last_used_epochs = physical_block_last_used_epochs;
	state->directory_entries = directory_entries;
	memset(state->physical_block_states,SPARK_GLM52_PP13_KV_ENTRY_MISSING,
		state->physical_block_capacity * sizeof(state->physical_block_states[0]));
	memset(state->physical_block_sequence_ids,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_sequence_ids[0]));
	memset(state->physical_block_logical_indices,0xff,
		state->physical_block_capacity *
			sizeof(state->physical_block_logical_indices[0]));
	memset(state->physical_block_last_used_epochs,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_last_used_epochs[0]));
	memset(state->directory_entries,0,
		state->directory_capacity * sizeof(state->directory_entries[0]));
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlConfigureKvSwap(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t backing_block_capacity,
	uint32_t *backing_block_free_next,
	SparkGlm52Pp13WorkControlKvSwapStoreFunction swap_store_function,
	SparkGlm52Pp13WorkControlKvSwapLoadFunction swap_load_function,
	void *swap_context)
{
	uint32_t backing_block_index;

	if (state == 0 ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
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
				: SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
	}
	state->backing_block_capacity = backing_block_capacity;
	state->free_backing_block_head = 0u;
	state->backing_block_free_next = backing_block_free_next;
	state->swap_store_function = swap_store_function;
	state->swap_load_function = swap_load_function;
	state->swap_context = swap_context;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlConfigureKvPins(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t *physical_block_pin_counts)
{
	if (state == 0 || physical_block_pin_counts == 0 ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_pin_counts != 0 ||
		state->directory_entry_count != 0u ||
		state->allocated_physical_block_count != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	memset(physical_block_pin_counts,0,
		state->physical_block_capacity * sizeof(physical_block_pin_counts[0]));
	state->physical_block_pin_counts = physical_block_pin_counts;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlPinPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		state->physical_block_sequence_ids[physical_block_index] == 0u ||
		state->physical_block_pin_counts[physical_block_index] == UINT32_MAX)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] += 1u;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlUnpinPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		state->physical_block_pin_counts[physical_block_index] == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] -= 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlValidateKvState(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	if (packet == 0 || state == 0 ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
		state->physical_block_indices == 0 ||
		state->lane_physical_block_counts == 0 ||
		state->physical_block_states == 0 ||
		state->physical_block_sequence_ids == 0 ||
		state->physical_block_logical_indices == 0 ||
		state->physical_block_last_used_epochs == 0 ||
		state->directory_entries == 0 ||
		state->lane_capacity == 0u ||
		state->lane_stride == 0u ||
		state->block_token_count == 0u ||
		(uint64_t)state->table_entry_capacity !=
			(uint64_t)state->lane_capacity * state->lane_stride ||
		state->physical_block_capacity == 0u ||
		state->physical_block_capacity > UINT32_MAX / 2u ||
		state->directory_capacity < state->physical_block_capacity * 2u ||
		(state->directory_capacity & (state->directory_capacity - 1u)) != 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if ((state->backing_block_capacity == 0u) !=
		(state->backing_block_free_next == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_store_function == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_load_function == 0) ||
		(state->backing_block_capacity == 0u) !=
		(state->swap_context == 0) ||
		state->backing_block_capacity > state->directory_capacity / 2u ||
		state->directory_entry_count > (state->backing_block_capacity != 0u
			? state->backing_block_capacity : state->physical_block_capacity) ||
		state->allocated_physical_block_count > state->physical_block_capacity ||
		state->swapped_block_count > state->directory_entry_count)
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
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13WorkControlResetBackingBlocks(
	SparkGlm52Pp13WorkControlKvState *state)
{
	uint32_t backing_block_index;

	if (state->backing_block_capacity == 0u)
	{
		state->free_backing_block_head =
			SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
		return;
	}
	for (backing_block_index = 0u;
		 backing_block_index < state->backing_block_capacity;
		 ++backing_block_index)
	{
		state->backing_block_free_next[backing_block_index] =
			backing_block_index + 1u < state->backing_block_capacity
				? backing_block_index + 1u
				: SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
	}
	state->free_backing_block_head = 0u;
}

static void SparkGlm52Pp13WorkControlResetKvGeneration(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t control_generation)
{
	memset(state->physical_block_indices,0xff,
		state->table_entry_capacity * sizeof(state->physical_block_indices[0]));
	memset(state->lane_physical_block_counts,0,
		state->lane_capacity * sizeof(state->lane_physical_block_counts[0]));
	memset(state->physical_block_states,SPARK_GLM52_PP13_KV_ENTRY_MISSING,
		state->physical_block_capacity * sizeof(state->physical_block_states[0]));
	memset(state->physical_block_sequence_ids,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_sequence_ids[0]));
	memset(state->physical_block_logical_indices,0xff,
		state->physical_block_capacity *
			sizeof(state->physical_block_logical_indices[0]));
	memset(state->physical_block_last_used_epochs,0,
		state->physical_block_capacity *
			sizeof(state->physical_block_last_used_epochs[0]));
	if (state->physical_block_pin_counts != 0)
		memset(state->physical_block_pin_counts,0,
			state->physical_block_capacity *
				sizeof(state->physical_block_pin_counts[0]));
	memset(state->directory_entries,0,
		state->directory_capacity * sizeof(state->directory_entries[0]));
	SparkGlm52Pp13WorkControlResetBackingBlocks(state);
	state->next_physical_block_index = 0u;
	state->directory_entry_count = 0u;
	state->swapped_block_count = 0u;
	state->epoch = 0u;
	state->missing_block_count = 0u;
	state->in_flight_block_count = 0u;
	state->resident_block_count = 0u;
	state->allocated_physical_block_count = 0u;
	state->control_generation = control_generation;
	state->control_generation_reset_count += 1u;
}

SparkStatus SparkGlm52Pp13WorkControlAdvanceKvGeneration(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t control_generation)
{
	if (state == 0 || control_generation == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (control_generation < state->control_generation)
		return SPARK_STATUS_VALIDATION_FAILED;
	if (control_generation != state->control_generation)
		SparkGlm52Pp13WorkControlResetKvGeneration(state,control_generation);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlSelectKvGeneration(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	SparkStatus status;
	status = SparkGlm52Pp13WorkControlAdvanceKvGeneration(
		state,packet->control_generation);
	return status == SPARK_STATUS_VALIDATION_FAILED
		? SPARK_STATUS_NOT_FOUND : status;
}

static void SparkGlm52Pp13WorkControlResetReadinessCounts(
	SparkGlm52Pp13WorkControlKvState *state)
{
	state->missing_block_count = 0u;
	state->in_flight_block_count = 0u;
	state->resident_block_count = 0u;
}

static void SparkGlm52Pp13WorkControlAccountReadiness(
	SparkGlm52Pp13WorkControlKvState *state,
	uint8_t entry_state)
{
	if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_RESIDENT)
		state->resident_block_count += 1u;
	else if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT)
		state->in_flight_block_count += 1u;
	else
		state->missing_block_count += 1u;
}

static uint64_t SparkGlm52Pp13WorkControlKvDirectoryHash(
	uint64_t sequence_id,
	uint32_t logical_block_index)
{
	uint64_t value;
	value = sequence_id ^
		((uint64_t)logical_block_index * 0x9e3779b97f4a7c15ull);
	value ^= value >> 30u;
	value *= 0xbf58476d1ce4e5b9ull;
	value ^= value >> 27u;
	value *= 0x94d049bb133111ebull;
	return value ^ (value >> 31u);
}

static SparkStatus SparkGlm52Pp13WorkControlKvDirectoryFind(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint32_t *directory_index_out,
	uint32_t *found_out)
{
	uint32_t first_tombstone;
	uint32_t probe_index;
	uint32_t directory_index;
	if (state == 0 || sequence_id == 0u || directory_index_out == 0 ||
		found_out == 0 || state->directory_capacity == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	first_tombstone = UINT32_MAX;
	directory_index = (uint32_t)(SparkGlm52Pp13WorkControlKvDirectoryHash(
		sequence_id,logical_block_index) & (state->directory_capacity - 1u));
	for (probe_index = 0u;
		 probe_index < state->directory_capacity;
		 ++probe_index)
	{
		SparkGlm52Pp13WorkControlKvDirectoryEntry *entry;
		entry = &state->directory_entries[directory_index];
		if (entry->state == SPARK_GLM52_PP13_KV_DIRECTORY_EMPTY)
		{
			*directory_index_out = first_tombstone != UINT32_MAX
				? first_tombstone : directory_index;
			*found_out = 0u;
			return SPARK_STATUS_OK;
		}
		if (entry->state == SPARK_GLM52_PP13_KV_DIRECTORY_TOMBSTONE)
		{
			if (first_tombstone == UINT32_MAX)
				first_tombstone = directory_index;
		}
		else if (entry->state != SPARK_GLM52_PP13_KV_DIRECTORY_OCCUPIED ||
			entry->backing_valid > 1u || entry->sequence_id == 0u ||
			(entry->residency_state !=
				SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU &&
			 entry->residency_state !=
				SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_NVME))
		{
			return SPARK_STATUS_INTERNAL_ERROR;
		}
		else if (entry->sequence_id == sequence_id &&
			entry->logical_block_index == logical_block_index)
		{
			*directory_index_out = directory_index;
			*found_out = 1u;
			return SPARK_STATUS_OK;
		}
		directory_index = (directory_index + 1u) &
			(state->directory_capacity - 1u);
	}
	if (first_tombstone != UINT32_MAX)
	{
		*directory_index_out = first_tombstone;
		*found_out = 0u;
		return SPARK_STATUS_OK;
	}
	return SPARK_STATUS_CAPACITY_EXCEEDED;
}

static SparkStatus SparkGlm52Pp13WorkControlKvBackingAcquire(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t *backing_block_index_out)
{
	uint32_t backing_block_index;

	if (state == 0 || backing_block_index_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->backing_block_capacity == 0u)
	{
		*backing_block_index_out = SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
		return SPARK_STATUS_OK;
	}
	backing_block_index = state->free_backing_block_head;
	if (backing_block_index >= state->backing_block_capacity)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	state->free_backing_block_head =
		state->backing_block_free_next[backing_block_index];
	state->backing_block_free_next[backing_block_index] =
		SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
	*backing_block_index_out = backing_block_index;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlKvBackingRelease(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t backing_block_index)
{
	if (state == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (state->backing_block_capacity == 0u)
		return backing_block_index == SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX
			? SPARK_STATUS_OK : SPARK_STATUS_INTERNAL_ERROR;
	if (backing_block_index >= state->backing_block_capacity ||
		state->backing_block_free_next[backing_block_index] !=
			SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX)
		return SPARK_STATUS_INTERNAL_ERROR;
	state->backing_block_free_next[backing_block_index] =
		state->free_backing_block_head;
	state->free_backing_block_head = backing_block_index;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13WorkControlKvClearPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t physical_block_index)
{
	state->physical_block_sequence_ids[physical_block_index] = 0u;
	state->physical_block_logical_indices[physical_block_index] =
		SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
	state->physical_block_last_used_epochs[physical_block_index] = 0u;
	state->physical_block_states[physical_block_index] =
		SPARK_GLM52_PP13_KV_ENTRY_MISSING;
}

static void SparkGlm52Pp13WorkControlKvAssignPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t physical_block_index,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint8_t physical_state)
{
	state->physical_block_sequence_ids[physical_block_index] = sequence_id;
	state->physical_block_logical_indices[physical_block_index] =
		logical_block_index;
	state->physical_block_last_used_epochs[physical_block_index] = state->epoch;
	state->physical_block_states[physical_block_index] = physical_state;
	state->allocated_physical_block_count += 1u;
}

static SparkStatus SparkGlm52Pp13WorkControlKvAcquirePhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t *physical_block_index_out)
{
	uint32_t physical_block_index;
	uint32_t scan_count;

	if (state == 0 || physical_block_index_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (scan_count = 0u;
		 scan_count < state->physical_block_capacity;
		 ++scan_count)
	{
		uint32_t directory_index;
		uint32_t found;
		SparkGlm52Pp13WorkControlKvDirectoryEntry *entry;
		SparkStatus status;

		physical_block_index = state->next_physical_block_index;
		state->next_physical_block_index =
			(physical_block_index + 1u) % state->physical_block_capacity;
		if (state->physical_block_pin_counts != 0 &&
			state->physical_block_pin_counts[physical_block_index] != 0u)
			continue;
		if (state->physical_block_sequence_ids[physical_block_index] == 0u)
		{
			*physical_block_index_out = physical_block_index;
			return SPARK_STATUS_OK;
		}
		if (state->backing_block_capacity == 0u ||
			state->physical_block_states[physical_block_index] !=
				SPARK_GLM52_PP13_KV_ENTRY_RESIDENT ||
			state->physical_block_last_used_epochs[physical_block_index] ==
				state->epoch)
			continue;
		status = SparkGlm52Pp13WorkControlKvDirectoryFind(
			state,
			state->physical_block_sequence_ids[physical_block_index],
			state->physical_block_logical_indices[physical_block_index],
			&directory_index,&found);
		if (status != SPARK_STATUS_OK || found == 0u)
			return status == SPARK_STATUS_OK
				? SPARK_STATUS_INTERNAL_ERROR : status;
		entry = &state->directory_entries[directory_index];
		if (entry->residency_state !=
				SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU ||
			entry->physical_block_index != physical_block_index ||
			entry->backing_block_index >= state->backing_block_capacity)
			return SPARK_STATUS_INTERNAL_ERROR;
		if (entry->backing_valid == 0u)
		{
			status = state->swap_store_function(
				state->swap_context,entry->sequence_id,entry->logical_block_index,
				physical_block_index,entry->backing_block_index);
			if (status != SPARK_STATUS_OK)
				return status;
			entry->backing_valid = 1u;
			state->swap_store_count += 1u;
		}
		else
		{
			state->clean_evict_count += 1u;
		}
		entry->physical_block_index = SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX;
		entry->residency_state =
			SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_NVME;
		SparkGlm52Pp13WorkControlKvClearPhysicalBlock(
			state,physical_block_index);
		if (state->allocated_physical_block_count == 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		state->allocated_physical_block_count -= 1u;
		state->swapped_block_count += 1u;
		*physical_block_index_out = physical_block_index;
		return SPARK_STATUS_OK;
	}
	return SPARK_STATUS_CAPACITY_EXCEEDED;
}

SparkStatus SparkGlm52Pp13WorkControlAcquireTransientPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t *physical_block_index_out)
{
	uint32_t physical_block_index;
	SparkStatus status;
	if (state == 0 || physical_block_index_out == 0 ||
		state->physical_block_pin_counts == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlKvAcquirePhysicalBlock(
		state,&physical_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	SparkGlm52Pp13WorkControlKvAssignPhysicalBlock(
		state,physical_block_index,UINT64_MAX,
		SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX,
		SPARK_GLM52_PP13_KV_ENTRY_TRANSIENT);
	state->physical_block_pin_counts[physical_block_index] = 1u;
	*physical_block_index_out = physical_block_index;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlReleaseTransientPhysicalBlock(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t physical_block_index)
{
	if (state == 0 || state->physical_block_pin_counts == 0 ||
		physical_block_index >= state->physical_block_capacity ||
		state->physical_block_states[physical_block_index] !=
			SPARK_GLM52_PP13_KV_ENTRY_TRANSIENT ||
		state->physical_block_sequence_ids[physical_block_index] != UINT64_MAX ||
		state->physical_block_logical_indices[physical_block_index] !=
			SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX ||
		state->physical_block_pin_counts[physical_block_index] != 1u ||
		state->allocated_physical_block_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	state->physical_block_pin_counts[physical_block_index] = 0u;
	SparkGlm52Pp13WorkControlKvClearPhysicalBlock(
		state,physical_block_index);
	state->allocated_physical_block_count -= 1u;
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlKvDirectoryAcquire(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_index,
	uint32_t *physical_block_index_out,
	uint32_t *allocated_out)
{
	SparkGlm52Pp13WorkControlKvDirectoryEntry *entry;
	uint32_t directory_index;
	uint32_t found;
	uint32_t physical_block_index;
	uint32_t backing_block_index;
	SparkStatus status;
	if (physical_block_index_out == 0 || allocated_out == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlKvDirectoryFind(
		state,sequence_id,logical_block_index,&directory_index,&found);
	if (status != SPARK_STATUS_OK)
		return status;
	entry = &state->directory_entries[directory_index];
	if (found != 0u)
	{
		if (entry->residency_state ==
			SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU)
		{
			physical_block_index = entry->physical_block_index;
			if (physical_block_index >= state->physical_block_capacity ||
				state->physical_block_sequence_ids[physical_block_index] !=
					sequence_id ||
				state->physical_block_logical_indices[physical_block_index] !=
					logical_block_index)
				return SPARK_STATUS_INTERNAL_ERROR;
			state->physical_block_last_used_epochs[physical_block_index] =
				state->epoch;
		}
		else if (entry->residency_state ==
			SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_NVME)
		{
			if (entry->backing_valid == 0u)
				return SPARK_STATUS_INTERNAL_ERROR;
			status = SparkGlm52Pp13WorkControlKvAcquirePhysicalBlock(
				state,&physical_block_index);
			if (status != SPARK_STATUS_OK)
				return status;
			status = state->swap_load_function(
				state->swap_context,sequence_id,logical_block_index,
				physical_block_index,entry->backing_block_index);
			if (status != SPARK_STATUS_OK)
				return status;
			if (state->swapped_block_count == 0u)
				return SPARK_STATUS_INTERNAL_ERROR;
			state->swapped_block_count -= 1u;
			state->swap_load_count += 1u;
			entry->physical_block_index = physical_block_index;
			entry->residency_state =
				SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU;
			SparkGlm52Pp13WorkControlKvAssignPhysicalBlock(
				state,physical_block_index,sequence_id,logical_block_index,
				SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
		}
		else
		{
			return SPARK_STATUS_INTERNAL_ERROR;
		}
		*physical_block_index_out = physical_block_index;
		*allocated_out = 0u;
		return SPARK_STATUS_OK;
	}
	if (state->directory_entry_count >=
		(state->backing_block_capacity != 0u
			? state->backing_block_capacity : state->physical_block_capacity))
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	status = SparkGlm52Pp13WorkControlKvBackingAcquire(
		state,&backing_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13WorkControlKvAcquirePhysicalBlock(
		state,&physical_block_index);
	if (status != SPARK_STATUS_OK)
	{
		(void)SparkGlm52Pp13WorkControlKvBackingRelease(
			state,backing_block_index);
		return status;
	}
	entry->sequence_id = sequence_id;
	entry->logical_block_index = logical_block_index;
	entry->physical_block_index = physical_block_index;
	entry->backing_block_index = backing_block_index;
	entry->residency_state = SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU;
	entry->state = SPARK_GLM52_PP13_KV_DIRECTORY_OCCUPIED;
	entry->backing_valid = 0u;
	SparkGlm52Pp13WorkControlKvAssignPhysicalBlock(
		state,physical_block_index,sequence_id,logical_block_index,
		SPARK_GLM52_PP13_KV_ENTRY_MISSING);
	state->directory_entry_count += 1u;
	*physical_block_index_out = physical_block_index;
	*allocated_out = 1u;
	return SPARK_STATUS_OK;
}

static void SparkGlm52Pp13WorkControlKvDirectoryEraseAt(
	SparkGlm52Pp13WorkControlKvState *state,
	uint32_t directory_index)
{
	uint32_t hole_index;
	uint32_t scan_index;
	uint32_t mask;

	mask = state->directory_capacity - 1u;
	hole_index = directory_index;
	memset(&state->directory_entries[hole_index],0,
		sizeof(state->directory_entries[hole_index]));
	scan_index = (hole_index + 1u) & mask;
	while (state->directory_entries[scan_index].state ==
		SPARK_GLM52_PP13_KV_DIRECTORY_OCCUPIED)
	{
		uint32_t home_index;
		uint32_t distance_to_hole;
		uint32_t distance_to_scan;
		home_index = (uint32_t)(SparkGlm52Pp13WorkControlKvDirectoryHash(
			state->directory_entries[scan_index].sequence_id,
			state->directory_entries[scan_index].logical_block_index) & mask);
		distance_to_hole = (hole_index - home_index) & mask;
		distance_to_scan = (scan_index - home_index) & mask;
		if (distance_to_hole < distance_to_scan)
		{
			state->directory_entries[hole_index] =
				state->directory_entries[scan_index];
			memset(&state->directory_entries[scan_index],0,
				sizeof(state->directory_entries[scan_index]));
			hole_index = scan_index;
		}
		scan_index = (scan_index + 1u) & mask;
	}
}

static SparkStatus SparkGlm52Pp13WorkControlKvDirectoryRelease(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_index)
{
	SparkGlm52Pp13WorkControlKvDirectoryEntry *entry;
	uint32_t directory_index;
	uint32_t found;
	SparkStatus status;
	status = SparkGlm52Pp13WorkControlKvDirectoryFind(
		state,sequence_id,logical_block_index,&directory_index,&found);
	if (status != SPARK_STATUS_OK)
		return status;
	if (found == 0u)
		return SPARK_STATUS_NOT_FOUND;
	entry = &state->directory_entries[directory_index];
	if (entry->residency_state ==
		SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU)
	{
		if (entry->physical_block_index < state->physical_block_capacity &&
			state->physical_block_pin_counts != 0 &&
			state->physical_block_pin_counts[entry->physical_block_index] != 0u)
			return SPARK_STATUS_BUSY;
		if (entry->physical_block_index >= state->physical_block_capacity ||
			state->physical_block_sequence_ids[entry->physical_block_index] !=
				sequence_id ||
			state->physical_block_logical_indices[entry->physical_block_index] !=
				logical_block_index ||
			state->allocated_physical_block_count == 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		SparkGlm52Pp13WorkControlKvClearPhysicalBlock(
			state,entry->physical_block_index);
		state->allocated_physical_block_count -= 1u;
	}
	else if (entry->residency_state ==
		SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_NVME)
	{
		if (entry->physical_block_index !=
				SPARK_GLM52_PP13_KV_INVALID_BLOCK_INDEX ||
			state->swapped_block_count == 0u)
			return SPARK_STATUS_INTERNAL_ERROR;
		state->swapped_block_count -= 1u;
	}
	else
	{
		return SPARK_STATUS_INTERNAL_ERROR;
	}
	status = SparkGlm52Pp13WorkControlKvBackingRelease(
		state,entry->backing_block_index);
	if (status != SPARK_STATUS_OK)
		return status;
	if (state->directory_entry_count == 0u)
		return SPARK_STATUS_INTERNAL_ERROR;
	state->directory_entry_count -= 1u;
	SparkGlm52Pp13WorkControlKvDirectoryEraseAt(state,directory_index);
	return SPARK_STATUS_OK;
}

static SparkStatus SparkGlm52Pp13WorkControlMarkTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state,
	uint8_t entry_state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint32_t directory_index;
	uint32_t found;
	uint32_t physical_block_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if (packet->control_generation != state->control_generation)
		return SPARK_STATUS_NOT_FOUND;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52Pp13WorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			status = SparkGlm52Pp13WorkControlKvDirectoryFind(
				state,packet->lanes[lane_index].sequence_id,block_index,
				&directory_index,&found);
			if (status != SPARK_STATUS_OK || found == 0u)
				return status == SPARK_STATUS_OK ? SPARK_STATUS_NOT_FOUND : status;
			physical_block_index =
				state->directory_entries[directory_index].physical_block_index;
			if (physical_block_index >= state->physical_block_capacity)
				return SPARK_STATUS_INTERNAL_ERROR;
			state->physical_block_states[physical_block_index] = entry_state;
			state->physical_block_last_used_epochs[physical_block_index] =
				state->epoch;
		}
	}
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
	uint32_t allocated;
	uint8_t entry_state;
	uint64_t base_block_index;
	uint64_t required_physical_block_count;
	uint32_t physical_block_index;
	SparkStatus status;

	if (view == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	status = SparkGlm52Pp13WorkControlSelectKvGeneration(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	required_physical_block_count = 0u;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52Pp13WorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence ||
			block_count > state->physical_block_capacity ||
			required_physical_block_count >
				state->physical_block_capacity - block_count)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		required_physical_block_count += block_count;
	}
	memset(
		state->lane_physical_block_counts,
		0,
		packet->active_sequence_count *
			sizeof(state->lane_physical_block_counts[0]));
	state->epoch += 1u;
	if (state->epoch == 0u)
		state->epoch = 1u;
	SparkGlm52Pp13WorkControlResetReadinessCounts(state);
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52Pp13WorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		base_block_index = (uint64_t)lane_index * (uint64_t)state->lane_stride;
		state->lane_physical_block_counts[lane_index] = block_count;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			if (base_block_index + block_index >= state->table_entry_capacity)
				return SPARK_STATUS_CAPACITY_EXCEEDED;
			status = SparkGlm52Pp13WorkControlKvDirectoryAcquire(
				state,
				packet->lanes[lane_index].sequence_id,
				block_index,
				&physical_block_index,
				&allocated);
			if (status != SPARK_STATUS_OK)
				return status;
			(void)allocated;
			entry_state = state->physical_block_states[physical_block_index];
			if ((packet->flags & SPARK_GLM52_PP13_WORK_CONTROL_FLAG_PREFILL) != 0u)
			{
				if (entry_state == SPARK_GLM52_PP13_KV_ENTRY_MISSING)
					entry_state = SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT;
			}
			else if (block_index + 1u < block_count &&
				entry_state != SPARK_GLM52_PP13_KV_ENTRY_RESIDENT)
			{
				SparkGlm52Pp13WorkControlAccountReadiness(state,entry_state);
				return SPARK_STATUS_BUSY;
			}
			else if (block_index + 1u == block_count &&
				entry_state == SPARK_GLM52_PP13_KV_ENTRY_MISSING)
			{
				entry_state = SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT;
			}
			state->physical_block_states[physical_block_index] = entry_state;
			SparkGlm52Pp13WorkControlAccountReadiness(state,entry_state);
			state->physical_block_indices[base_block_index + block_index] =
				physical_block_index;
		}
	}
	memset(view,0,sizeof(*view));
	view->abi_version = SPARK_GLM52_KV_CACHE_ABI_VERSION;
	view->descriptor_bytes = SPARK_GLM52_KV_BLOCK_TABLE_VIEW_DESCRIPTOR_BYTES;
	view->block_token_count = packet->block_token_count;
	view->lane_count = packet->active_sequence_count;
	view->lane_stride = state->lane_stride;
	view->lane_capacity = state->lane_capacity;
	view->physical_block_indices = state->physical_block_indices;
	view->lane_physical_block_counts = state->lane_physical_block_counts;
	view->host_physical_block_indices = state->physical_block_indices;
	view->host_lane_physical_block_counts = state->lane_physical_block_counts;
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlCommitHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	uint64_t write_end_token;
	uint32_t first_written_block;
	uint32_t last_written_block;
	uint32_t directory_index;
	uint32_t found;
	uint32_t lane_index;
	uint32_t block_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlMarkTable(
		packet,
		state,
		SPARK_GLM52_PP13_KV_ENTRY_RESIDENT);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		write_end_token = packet->lanes[lane_index].sequence_position +
			(uint64_t)packet->new_token_count - 1u;
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
			status = SparkGlm52Pp13WorkControlKvDirectoryFind(
				state,packet->lanes[lane_index].sequence_id,block_index,
				&directory_index,&found);
			if (status != SPARK_STATUS_OK || found == 0u)
				return status == SPARK_STATUS_OK
					? SPARK_STATUS_NOT_FOUND : status;
			if (state->directory_entries[directory_index].residency_state !=
					SPARK_GLM52_PP13_KV_DIRECTORY_RESIDENCY_GPU)
				return SPARK_STATUS_INTERNAL_ERROR;
			state->directory_entries[directory_index].backing_valid = 0u;
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlCancelHostKvBlockTable(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	uint32_t lane_index;
	uint32_t block_index;
	uint32_t block_count;
	uint32_t directory_index;
	uint32_t found;
	uint32_t physical_block_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if (packet->control_generation != state->control_generation)
		return SPARK_STATUS_NOT_FOUND;
	for (lane_index = 0u; lane_index < packet->active_sequence_count; ++lane_index)
	{
		block_count = SparkGlm52Pp13WorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		if (block_count == 0u || block_count > packet->max_blocks_per_sequence)
			return SPARK_STATUS_CAPACITY_EXCEEDED;
		for (block_index = 0u; block_index < block_count; ++block_index)
		{
			status = SparkGlm52Pp13WorkControlKvDirectoryFind(
				state,packet->lanes[lane_index].sequence_id,block_index,
				&directory_index,&found);
			if (status != SPARK_STATUS_OK)
				return status;
			if (found == 0u)
				continue;
			physical_block_index =
				state->directory_entries[directory_index].physical_block_index;
			if (physical_block_index >= state->physical_block_capacity)
				return SPARK_STATUS_INTERNAL_ERROR;
			if (state->physical_block_states[physical_block_index] ==
				SPARK_GLM52_PP13_KV_ENTRY_IN_FLIGHT)
			{
				status = SparkGlm52Pp13WorkControlKvDirectoryRelease(
					state,packet->lanes[lane_index].sequence_id,block_index);
				if (status != SPARK_STATUS_OK)
					return status;
			}
		}
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlReleaseSequence(
	SparkGlm52Pp13WorkControlKvState *state,
	uint64_t sequence_id,
	uint32_t logical_block_count)
{
	uint32_t logical_block_index;

	if (state == 0 || sequence_id == 0u || logical_block_count == 0u ||
		state->abi_version != SPARK_GLM52_PP13_WORK_CONTROL_ABI_VERSION ||
		state->descriptor_bytes != SPARK_GLM52_PP13_WORK_CONTROL_KV_STATE_BYTES ||
		logical_block_count > state->lane_stride ||
		state->directory_entries == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	for (logical_block_index = 0u;
		 logical_block_index < logical_block_count;
		 ++logical_block_index)
	{
		SparkStatus status;
		status = SparkGlm52Pp13WorkControlKvDirectoryRelease(
			state,sequence_id,logical_block_index);
		if (status != SPARK_STATUS_OK && status != SPARK_STATUS_NOT_FOUND)
			return status;
	}
	return SPARK_STATUS_OK;
}

SparkStatus SparkGlm52Pp13WorkControlReleasePacketSequences(
	const SparkGlm52Pp13WorkControlPacket *packet,
	SparkGlm52Pp13WorkControlKvState *state)
{
	uint32_t lane_index;
	SparkStatus status;

	status = SparkGlm52Pp13WorkControlValidateKvState(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	if ((packet->flags &
		SPARK_GLM52_PP13_WORK_CONTROL_FLAG_RELEASE_SEQUENCES) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (packet->control_generation < state->control_generation)
		return SPARK_STATUS_OK;
	status = SparkGlm52Pp13WorkControlSelectKvGeneration(packet,state);
	if (status != SPARK_STATUS_OK)
		return status;
	for (lane_index = 0u; lane_index < packet->lane_count; ++lane_index)
	{
		uint32_t logical_block_count;
		logical_block_count = SparkGlm52Pp13WorkControlBlockCount(
			packet->lanes[lane_index].context_token_count,
			packet->block_token_count);
		status = SparkGlm52Pp13WorkControlReleaseSequence(
			state,packet->lanes[lane_index].sequence_id,logical_block_count);
		if (status != SPARK_STATUS_OK)
			return status;
	}
	return SPARK_STATUS_OK;
}
