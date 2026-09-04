#ifndef SPARKPIPE_SPARK_QWEN38_PP_SERVING_ADAPTER_COMMON_H
#define SPARKPIPE_SPARK_QWEN38_PP_SERVING_ADAPTER_COMMON_H

#include "sparkpipe/spark_qwen38_serving_adapter_common.h"

#define SPARK_QWEN38_SERVING_ADAPTER_STRINGIZE_TOKEN(token) #token
#define SPARK_QWEN38_SERVING_ADAPTER_STRINGIZE(token) \
	SPARK_QWEN38_SERVING_ADAPTER_STRINGIZE_TOKEN(token)
#define SPARK_QWEN38_SERVING_ADAPTER_ENV(name) \
	SPARK_QWEN38_SERVING_ADAPTER_STRINGIZE(SPARK_QWEN38_SERVING_ADAPTER_CONST(name))

static const char *const SPARK_QWEN38_SERVING_ADAPTER_FN(ServingConfigurationMembers)[] =
{
	"schema_version",
	"model_revision",
	"stage_pack_path",
	"max_sequence_positions",
	"tp_degree"
};

static uint32_t SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state, uint32_t world_rank)
{
	return(state->tp_degree != 0u ? world_rank / state->tp_degree : world_rank / SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DEFAULT_TP_DEGREE));
}

static uint32_t SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state)
{
	return(state->pp_stage_count != 0u ? state->pp_stage_count : SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_COUNT) / SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DEFAULT_TP_DEGREE));
}

static uint32_t SPARK_QWEN38_SERVING_ADAPTER_FN(ServingFirstLayer)(const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state, uint32_t stage_index)
{
	uint32_t index,first_layer,pp_stage;
	first_layer = 0u;
	pp_stage = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,stage_index);
	for (index=0u; index<pp_stage; index++)
		first_layer += state->stage_layer_counts[index];
	return(first_layer);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingLoadConfiguration)(
	const char *path,
	const char *runtime_root,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	uint32_t *max_sequence_positions)
{
	SparkJsonDocument document;
	int32_t root,token;
	uint32_t schema_version;
	char *relative_stage_pack_path;
	SparkStatus status;
	relative_stage_pack_path = 0;
	SparkJsonDocumentReset(&document);
	status = SparkJsonLoadFile(path,&document);
	root = status == SPARK_STATUS_OK ? SparkJsonGetRootToken(&document) : -1;
	if ( status == SPARK_STATUS_OK && !SparkJsonTokenIsType(&document,root,SPARK_JSON_TOKEN_OBJECT) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkJsonValidateObjectMembersExact(&document,root,SPARK_QWEN38_SERVING_ADAPTER_FN(ServingConfigurationMembers),(uint32_t)(sizeof(SPARK_QWEN38_SERVING_ADAPTER_FN(ServingConfigurationMembers)) / sizeof(SPARK_QWEN38_SERVING_ADAPTER_FN(ServingConfigurationMembers)[0])));
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"schema_version",&schema_version);
	if ( status == SPARK_STATUS_OK && schema_version != SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_ADAPTER_CONFIGURATION_SCHEMA_VERSION) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"model_revision") : -1;
	if ( status == SPARK_STATUS_OK && (token < 0 || !SparkJsonStringEquals(&document,token,SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"tp_degree",&state->tp_degree);
	if ( status == SPARK_STATUS_OK && !SPARK_QWEN38_SERVING_ADAPTER_TP_DEGREE_VALID(state->tp_degree) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	token = status == SPARK_STATUS_OK ? SparkServingAdapterTemplateJsonMember(&document,root,"stage_pack_path") : -1;
	if ( status == SPARK_STATUS_OK )
		status = token < 0 ? SPARK_STATUS_SCHEMA_ERROR : SparkJsonCopyString(&document,token,&relative_stage_pack_path);
	if ( status == SPARK_STATUS_OK )
		status = SparkServingAdapterTemplateJsonUnsigned(&document,root,"max_sequence_positions",max_sequence_positions);
	SparkJsonDocumentDestroy(&document);
	if ( status == SPARK_STATUS_OK )
		status = SparkResolveRuntimePath(runtime_root,relative_stage_pack_path,state->stage_pack_path,sizeof(state->stage_pack_path));
	free(relative_stage_pack_path);
	return(status);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSetEnvironment)(
	const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state)
{
	char value[32];
#define SPARK_QWEN38_SERVING_SET_TEXT(name,text) \
	do { if ( setenv(name,text,1) != 0 ) return(SPARK_STATUS_INTERNAL_ERROR); } while (0)
#define SPARK_QWEN38_SERVING_SET_UNSIGNED(name,number) \
	do { snprintf(value,sizeof(value),"%u",(uint32_t)(number)); SPARK_QWEN38_SERVING_SET_TEXT(name,value); } while (0)
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(ALLOW_UNQUALIFIED_EXECUTION),"1");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_PACK_PATH),state->stage_pack_path);
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_COUNT),SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_COUNT(state));
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_INDEX),SPARK_QWEN38_SERVING_ADAPTER_ENV_STAGE_INDEX(state));
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_FIRST_LAYER),state->first_layer_index);
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_LAYER_COUNT),state->stage_layer_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_MAX_ACTIVE_SEQUENCES),state->max_active_sequence_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_PIPELINE_SLOTS),state->pipeline_slot_count);
	SPARK_QWEN38_SERVING_SET_UNSIGNED(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_BLOCKS),state->kv_block_count);
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_MTP),"0");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_GDN_SNAPSHOT_SLOTS),"0");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_STORE),"none");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_SERVICE),"none");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_SOCKET),"none");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_POOL_BYTES),"0");
	SPARK_QWEN38_SERVING_SET_TEXT(SPARK_QWEN38_SERVING_ADAPTER_ENV(STAGE_KV_WORKERS),"0");
#undef SPARK_QWEN38_SERVING_SET_TEXT
#undef SPARK_QWEN38_SERVING_SET_UNSIGNED
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateRowOrder)(
	const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint8_t slot_seen[SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint64_t last_position[SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->resident_sequence_capacity || slot_seen[slot] != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		slot_seen[slot] = 1u;
	}
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == submission->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateBoundaries)(
	const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	uint32_t pp_stage;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES);
	pp_stage = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index);
	if ( (pp_stage != 0u && (submission->hidden_input_address == 0 || submission->hidden_input_bytes < boundary_bytes)) || (pp_stage == 0u && (submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u)) || (pp_stage + 1u < SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) && (submission->hidden_output_address == 0 || submission->hidden_output_bytes < boundary_bytes)) || (pp_stage + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) && (submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmissionBase)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	SparkStatus status;
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->quiescing != 0u )
		return(SPARK_STATUS_BUSY);
	status = SparkModelServingAdapterValidateRuntimeSubmission(&SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDescriptor),&state->runtime_limits,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateRowOrder)(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( submission->model_extension_bytes != 0u )
		return(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmission)(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	uint32_t emit_count;
	SparkStatus status;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)adapter_state;
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmissionBase)(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	return(SparkModelServingAdapterSelectEmitRows(submission,0,0,0u,&emit_count));
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReleaseLane)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	uint32_t slot)
{
	uint32_t ordinal;
	for (ordinal=0u; ordinal<state->lane_block_counts[slot]; ordinal++)
		state->free_blocks[state->free_block_count++] = state->host_block_indices[((uint64_t)slot * state->blocks_per_lane) + ordinal];
	state->lane_block_counts[slot] = 0u;
	state->lane_context_tokens[slot] = 0u;
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCoverLane)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	uint32_t slot,
	uint64_t end_position)
{
	uint32_t required,ordinal;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	required = (uint32_t)((end_position + SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) - 1u) / SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS));
	if ( required > state->blocks_per_lane )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (ordinal=state->lane_block_counts[slot]; ordinal<required; ordinal++)
	{
		if ( state->free_block_count == 0u )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
		state->host_block_indices[((uint64_t)slot * state->blocks_per_lane) + ordinal] = state->free_blocks[--state->free_block_count];
		state->lane_block_counts[slot] = ordinal + 1u;
	}
	state->lane_block_counts[slot] = required;
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCoverSubmission)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	SparkStatus status;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		uint64_t first_position,end_position;
		slot = submission->lanes[lane].resident_sequence_slot;
		first_position = UINT64_MAX;
		end_position = 0u;
		for (row=0u; row<submission->row_count; row++)
		{
			if ( submission->row_lane_indices[row] != lane )
				continue;
			if ( submission->row_positions[row] < first_position )
				first_position = submission->row_positions[row];
			if ( submission->row_positions[row] + 1u > end_position )
				end_position = submission->row_positions[row] + 1u;
		}
		if ( first_position == 0u && state->lane_context_tokens[slot] != 0u )
			SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReleaseLane)(state,slot);
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCoverLane)(state,slot,end_position);
		if ( status != SPARK_STATUS_OK )
		{
			SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDropSubmission)(state,submission);
			return(status);
		}
	}
	return(SPARK_STATUS_OK);
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCommitSubmission)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint32_t lane,row;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
	{
		uint32_t slot;
		slot = submission->lanes[lane].resident_sequence_slot;
		for (row=0u; row<submission->row_count; row++)
			if ( submission->row_lane_indices[row] == lane && submission->row_positions[row] + 1u > state->lane_context_tokens[slot] )
				state->lane_context_tokens[slot] = submission->row_positions[row] + 1u;
	}
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingUploadBlockTable)(
	const SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	cudaError_t error = cudaSuccess;
	uint64_t lane_slice_bytes,counts_bytes;
	uint32_t lane,slot;
	if ( state->stage_attn_layer_count == 0u )
		return(SPARK_STATUS_OK);
	lane_slice_bytes = (uint64_t)state->blocks_per_lane * sizeof(uint32_t);
	counts_bytes = (uint64_t)state->max_active_sequence_count * sizeof(uint32_t);
	for (lane=0u; error == cudaSuccess && lane<submission->active_sequence_count; lane++)
	{
		slot = submission->lanes[lane].resident_sequence_slot;
		if ( slot >= state->max_active_sequence_count )
			continue;
		error = cudaMemcpyAsync((uint8_t *)state->device_block_indices + ((uint64_t)slot * lane_slice_bytes),(const uint8_t *)state->host_block_indices + ((uint64_t)slot * lane_slice_bytes),(size_t)lane_slice_bytes,cudaMemcpyHostToDevice,(cudaStream_t)state->execution_stream);
	}
	if ( error == cudaSuccess )
		error = cudaMemcpyAsync(state->device_block_counts,state->lane_block_counts,(size_t)counts_bytes,cudaMemcpyHostToDevice,(cudaStream_t)state->execution_stream);
	if ( error == cudaSuccess )
		error = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( error != cudaSuccess )
		return(SPARK_STATUS_IO_ERROR);
	return(SPARK_STATUS_OK);
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingBuildFrame)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(DecodeBatchView) *decode_batch,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(PrefillFrameView) *prefill_view,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ResidentDecodeStageFrameContext) *context,
	SparkModelDriverBuffer *buffers,
	SparkModelDriverFrame *frame)
{
	uint32_t slot;
	uint64_t base_position;
	uint32_t row;
	slot = prefill != 0u ? pending->resident_slots[lane] : 0u;
	base_position = 0u;
	memset(context,0,sizeof(*context));
	context->abi_version = SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION);
	context->descriptor_bytes = sizeof(*context);
	if ( state->stage_attn_layer_count != 0u )
	{
		context->flags |= SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_KV_BLOCK_TABLE);
		context->kv_block_table = &state->block_table;
	}
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) != 0u )
	{
		context->flags |= SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT);
		context->hidden_input_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_input_post_receive_function = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPostReceive);
	}
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u < SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) )
	{
		context->flags |= SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT);
		context->hidden_output_transport_session = (SparkHiddenTransportSession *)&state->shim;
		context->hidden_output_send_function = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSend);
	}
	state->shim.input_base = submission->hidden_input_address;
	state->shim.input_rows = frame_rows;
	state->shim.output_base = submission->hidden_output_address;
	if ( prefill != 0u )
	{
		uint32_t lane_row,flat;
		lane_row = 0u;
		for (flat=0u; flat<submission->row_count; flat++)
		{
			if ( submission->row_lane_indices[flat] != lane )
				continue;
			if ( lane_row >= wave_base && lane_row < wave_base + frame_rows )
			{
				pending->frame_row_flats[lane_row - wave_base] = flat;
				pending->frame_token_ids[lane_row - wave_base] = submission->token_ids[flat];
			}
			lane_row++;
		}
		state->shim.input_row_map = pending->frame_row_flats;
		state->shim.output_row_map = pending->frame_row_flats;
		base_position = submission->row_positions[pending->frame_row_flats[0]];
		prefill_view->abi_version = SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_PREFILL_FRAME_VIEW_ABI_VERSION);
		prefill_view->descriptor_bytes = sizeof(*prefill_view);
		prefill_view->lane_index = slot;
		prefill_view->token_count = frame_rows;
		prefill_view->base_position = base_position;
		prefill_view->sequence_id = submission->row_sequence_ids[pending->frame_row_flats[0]];
		context->flags |= SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW);
		context->prefill_frame = prefill_view;
	}
	else
	{
		for (row=0u; row<frame_rows; row++)
			pending->frame_row_slots[row] = pending->resident_slots[submission->row_lane_indices[row]];
		memcpy(pending->frame_token_ids,submission->token_ids,(size_t)frame_rows * sizeof(uint32_t));
		state->shim.input_row_map = 0;
		state->shim.output_row_map = 0;
		decode_batch->abi_version = SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_DECODE_BATCH_VIEW_ABI_VERSION);
		decode_batch->descriptor_bytes = sizeof(*decode_batch);
		decode_batch->row_count = frame_rows;
		decode_batch->row_lane_indices = pending->frame_row_slots;
		decode_batch->row_positions = submission->row_positions;
		decode_batch->row_sequence_ids = submission->row_sequence_ids;
		context->flags |= SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_DECODE_BATCH_VIEW);
		context->decode_batch = decode_batch;
	}
	memset(buffers,0,sizeof(SparkModelDriverBuffer[2]));
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) == 0u )
	{
		buffers[0].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_READ;
		buffers[0].address = pending->frame_token_ids;
		buffers[0].bytes = (uint64_t)frame_rows * sizeof(uint32_t);
	}
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) )
	{
		uint32_t out_index;
		out_index = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) == 0u ? 1u : 0u;
		buffers[out_index].slot = 1u;
		buffers[out_index].flags = SPARK_MODEL_DRIVER_BUFFER_FLAG_WRITE;
		buffers[out_index].address = pending->frame_output_ids;
		buffers[out_index].bytes = (uint64_t)(prefill != 0u ? 1u : frame_rows) * sizeof(uint32_t);
	}
	memset(frame,0,sizeof(*frame));
	frame->request_id = submission->request_id;
	frame->sequence_id = prefill != 0u ? prefill_view->sequence_id : submission->sequence_id;
	frame->sequence_position = prefill != 0u ? base_position : submission->sequence_position;
	frame->deadline_time_ns = submission->deadline_time_ns;
	frame->active_slot_count = prefill != 0u ? 1u : submission->active_sequence_count;
	frame->new_token_count = frame_rows;
	frame->tokens_per_sequence = submission->tokens_per_sequence;
	frame->priority = submission->priority;
	frame->flags = prefill != 0u ? SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL : 0u;
	frame->driver_dispatch_slot = SPARK_MODEL_DRIVER_INVALID_DISPATCH_SLOT;
	frame->program_id = state->program->program_id;
	frame->execution_stream = state->execution_stream;
	frame->buffers = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) == 0u || SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) ? buffers : 0;
	frame->buffer_count = (SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) == 0u ? 1u : 0u) + (SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) ? 1u : 0u);
	frame->residency = submission->residency;
	frame->user_context = context;
	frame->completion_function = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDriverCompletion);
	frame->completion_context = pending;
	pending->frame_sequence_id = frame->sequence_id;
	pending->frame_sequence_position = frame->sequence_position;
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingRunFrame)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingSubmission *submission,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending,
	uint32_t prefill,
	uint32_t lane,
	uint32_t wave_base,
	uint32_t frame_rows)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(DecodeBatchView) decode_batch;
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(PrefillFrameView) prefill_view;
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ResidentDecodeStageFrameContext) context;
	SparkModelDriverBuffer buffers[2];
	SparkModelDriverFrame frame;
	SparkStatus status;
	SPARK_QWEN38_SERVING_ADAPTER_FN(ServingBuildFrame)(state,submission,pending,prefill,lane,wave_base,frame_rows,&decode_batch,&prefill_view,&context,buffers,&frame);
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAdmit)(state,submission,&frame);
	if ( status == SPARK_STATUS_OK )
		status = state->program->submit(state->driver_instance,&frame);
	if ( status == SPARK_STATUS_OK )
		status = pending->frame_status;
	if ( status == SPARK_STATUS_OK && SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) )
	{
		if ( prefill != 0u )
			pending->output_token_ids[lane] = (submission->lanes[lane].flags & SPARK_MODEL_SERVING_LANE_FLAG_OUTPUT_TOKEN) != 0u ? pending->frame_output_ids[0] : 0u;
		else
		{
			uint32_t row;
			for (row=0u; row<frame_rows; row++)
				pending->output_token_ids[submission->row_lane_indices[row]] = pending->frame_output_ids[row];
		}
	}
	return(status);
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingComplete)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending,
	SparkStatus status)
{
	SparkModelServingCompletion completion;
	uint32_t index;
	memset(&completion,0,sizeof(completion));
	completion.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	completion.descriptor_bytes = SPARK_MODEL_SERVING_COMPLETION_BYTES;
	completion.status = (uint32_t)status;
	completion.submission_id = pending->common.submission_id;
	completion.request_id = pending->common.request_id;
	completion.sequence_id = pending->common.sequence_id;
	completion.sequence_position = pending->common.sequence_position;
	completion.control_generation = pending->common.control_generation;
	completion.transaction_id = pending->common.transaction_id;
	completion.dispatch_generation = pending->common.dispatch_generation;
	completion.request_generation = pending->common.request_generation;
	completion.step_generation = pending->common.step_generation;
	completion.residency = pending->residency;
	completion.accepted_token_count = (uint32_t)(pending->accepted_token_count > UINT32_MAX ? UINT32_MAX : pending->accepted_token_count);
	completion.queue_delay_ns = pending->queue_delay_ns;
	completion.service_time_ns = pending->service_time_ns;
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,state->stage_index) + 1u == SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageCount)(state) && status == SPARK_STATUS_OK )
	{
		completion.tokens_per_sequence = 1u;
		completion.token_count = pending->common.active_sequence_count;
		completion.completion_flags = SPARK_MODEL_SERVING_COMPLETION_FLAG_TOKEN_IDS;
		for (index=0u; index<completion.token_count; index++)
			completion.token_ids[index] = pending->output_token_ids[index];
	}
	pending->common.active = 0u;
	state->completion_function(state->completion_context,&completion);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSubmit)(
	void *adapter_state,
	const SparkModelServingSubmission *submission)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingPending) *pending;
	SparkStatus status;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)adapter_state;
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmissionBase)(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind != SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateBoundaries)(state,submission);
	if ( status != SPARK_STATUS_OK )
		return(status);
	pending = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingReservePending)(state,submission);
	if ( pending == 0 )
		return(SPARK_STATUS_BUSY);
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCoverSubmission)(state,submission);
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingUploadBlockTable)(state,submission);
	if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingRunFrame)(state,submission,pending,0u,0u,0u,submission->row_count);
	else if ( status == SPARK_STATUS_OK && submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_PREFILL )
	{
		uint32_t lane,wave,chunk_rows;
		for (lane=0u; status == SPARK_STATUS_OK && lane<submission->active_sequence_count; lane++)
		{
			uint32_t lane_rows;
			lane_rows = 0u;
			for (wave=0u; wave<submission->row_count; wave++)
				lane_rows += submission->row_lane_indices[wave] == lane ? 1u : 0u;
			for (wave=0u; status == SPARK_STATUS_OK && wave<lane_rows; wave+=chunk_rows)
			{
				chunk_rows = lane_rows - wave;
				if ( chunk_rows > state->max_active_sequence_count )
					chunk_rows = state->max_active_sequence_count;
				status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingRunFrame)(state,submission,pending,1u,lane,wave,chunk_rows);
			}
		}
	}
	else if ( status == SPARK_STATUS_OK )
		status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( status != SPARK_STATUS_OK )
	{
		SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDropSubmission)(state,submission);
		pending->common.active = 0u;
		return(status);
	}
	SPARK_QWEN38_SERVING_ADAPTER_FN(ServingCommitSubmission)(state,submission);
	SPARK_QWEN38_SERVING_ADAPTER_FN(ServingComplete)(state,pending,SPARK_STATUS_OK);
	return(SPARK_STATUS_OK);
}

static void SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDestroy)(void *adapter_state)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)adapter_state;
	if ( state == 0 )
		return;
	if ( SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAvailableSubmissionCount)(state) != state->pipeline_slot_count )
		return;
	if ( state->driver.interface != 0 && state->driver.interface->snapshot != 0 && state->driver_instance != 0 && state->program != 0 )
	{
		memset(&snapshot,0,sizeof(snapshot));
		if ( state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot) != SPARK_STATUS_OK || snapshot.active_submission_count != 0u )
			return;
	}
	if ( state->driver.interface != 0 && state->driver.interface->destroy != 0 && state->driver_instance != 0 )
		state->driver.interface->destroy(state->driver_instance);
	SparkUnloadModelDriver(&state->driver);
	if ( state->device_block_indices != 0 )
		(void)cudaFree(state->device_block_indices);
	if ( state->device_block_counts != 0 )
		(void)cudaFree(state->device_block_counts);
	if ( state->gather_scratch != 0 )
		(void)cudaFree(state->gather_scratch);
	free(state->host_block_indices);
	free(state->free_blocks);
	free(state);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAcceptsProgram)(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)accept_context;
	if ( (program->flags & SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_REQUIRED_PROGRAM_FLAGS)) != SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_REQUIRED_PROGRAM_FLAGS) || program->max_inflight < state->pipeline_slot_count || program->profile == 0 || program->profile->max_active_slots < state->max_active_sequence_count || program->profile->max_new_tokens < state->max_input_row_count )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingLoadDriver)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state,
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkServingAdapterDriverRequest request;
	const SparkModelDriverProgramDescriptor *program;
	SparkStatus status;
	request.contract.driver_model_id = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DRIVER_MODEL_ID);
	request.contract.driver_model_revision = SPARK_QWEN38_SERVING_ADAPTER_MODEL_REVISION;
	request.contract.driver_stage_name = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_NAME);
	request.contract.driver_target = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_TARGET);
	request.contract.model_description_sha256 = SPARK_QWEN38_SERVING_ADAPTER_CONTRACT_SHA256;
	request.node_context = 0;
	request.completion_context = state;
	request.completion_function = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingOrphanDriverCompletion);
	request.wake_function = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDriverWake);
	program = 0;
	status = SparkServingAdapterTemplateLoadDriver(&request,configuration,
		&state->driver,&program,SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAcceptsProgram),state,
		&state->driver_instance);
	state->program = program;
	return(status);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAllocatePools)(
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state)
{
	uint32_t block;
	uint64_t indices;
	indices = (uint64_t)state->max_active_sequence_count * state->blocks_per_lane;
	state->host_block_indices = (uint32_t *)malloc((size_t)indices * sizeof(uint32_t));
	state->free_blocks = (uint32_t *)malloc((size_t)state->kv_block_count * sizeof(uint32_t));
	if ( state->host_block_indices == 0 || state->free_blocks == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (block=0u; block<state->kv_block_count; block++)
		state->free_blocks[block] = state->kv_block_count - 1u - block;
	state->free_block_count = state->kv_block_count;
	if ( cudaMalloc(&state->gather_scratch,(size_t)((uint64_t)state->max_active_sequence_count * SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_HIDDEN_BF16_BYTES))) != cudaSuccess )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( state->stage_attn_layer_count != 0u )
	{
		if ( cudaMalloc((void **)&state->device_block_indices,(size_t)(indices * sizeof(uint32_t))) != cudaSuccess || cudaMalloc((void **)&state->device_block_counts,(size_t)(state->max_active_sequence_count * sizeof(uint32_t))) != cudaSuccess )
			return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	state->block_table.abi_version = SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TABLE_ABI_VERSION);
	state->block_table.descriptor_bytes = sizeof(state->block_table);
	state->block_table.block_token_count = SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
	state->block_table.lane_count = state->max_active_sequence_count;
	state->block_table.lane_stride = state->blocks_per_lane;
	state->block_table.lane_capacity = state->max_active_sequence_count;
	state->block_table.physical_block_indices = state->device_block_indices;
	state->block_table.lane_physical_block_counts = state->device_block_counts;
	state->block_table.host_physical_block_indices = state->host_block_indices;
	state->block_table.host_lane_physical_block_counts = state->lane_block_counts;
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_QWEN38_SERVING_ADAPTER_FN(ServingInitialize)(
	const SparkModelServingAdapterConfiguration *configuration,
	void **adapter_state)
{
	SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *state;
	uint32_t max_sequence_positions;
	SparkStatus status;
	if ( adapter_state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*adapter_state = 0;
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateConfiguration)(configuration);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = (SPARK_QWEN38_SERVING_ADAPTER_TYPE(ServingState) *)calloc(1u,sizeof(*state));
	if ( state == 0 )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	state->stage_index = configuration->stage_index;
	state->tp_degree = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_DEFAULT_TP_DEGREE);
	state->pipeline_slot_count = configuration->runtime_limits.max_inflight_submission_count;
	state->max_active_sequence_count = configuration->runtime_limits.max_active_sequence_count;
	state->max_input_row_count = configuration->runtime_limits.max_input_row_count;
	state->resident_sequence_capacity = configuration->runtime_limits.resident_sequence_capacity;
	state->runtime_limits = configuration->runtime_limits;
	state->completion_function = configuration->completion_function;
	state->completion_context = configuration->completion_context;
	state->wake_function = configuration->wake_function;
	state->wake_context = configuration->wake_context;
	state->execution_stream = configuration->execution_stream;
	state->shim.execution_stream = configuration->execution_stream;
	status = SPARK_QWEN38_SERVING_ADAPTER_BIND_FAMILY(state);
	if ( status != SPARK_STATUS_OK )
		{ free(state); return(status); }
	status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingLoadConfiguration)(configuration->adapter_configuration_path,configuration->runtime_root,state,&max_sequence_positions);
	if ( status == SPARK_STATUS_OK && (max_sequence_positions == 0u || max_sequence_positions > SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_MAX_SEQUENCE_POSITIONS_CAP)) )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( status == SPARK_STATUS_OK )
	{
		uint32_t pp,base,extra;
		state->pp_stage_count = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_STAGE_COUNT) / state->tp_degree;
		if ( state->pp_stage_count > SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_MAX_PP_STAGE_COUNT) )
			state->pp_stage_count = SPARK_QWEN38_SERVING_ADAPTER_CONST(SERVING_MAX_PP_STAGE_COUNT);
		base = SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_LAYER_COUNT) / state->pp_stage_count;
		extra = SPARK_QWEN38_SERVING_ADAPTER_CONST(MODEL_LAYER_COUNT) % state->pp_stage_count;
		for (pp = 0u; pp < state->pp_stage_count; pp++)
			state->stage_layer_counts[pp] = base + (pp < extra ? 1u : 0u);
		state->first_layer_index = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingFirstLayer)(state,configuration->stage_index);
		state->stage_layer_count = state->stage_layer_counts[SPARK_QWEN38_SERVING_ADAPTER_FN(ServingPpStageIndex)(state,configuration->stage_index)];
		state->stage_attn_layer_count = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingStageAttentionLayers)(state->first_layer_index,state->stage_layer_count);
		state->max_sequence_positions = max_sequence_positions;
		state->blocks_per_lane = (max_sequence_positions + SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS) - 1u) / SPARK_QWEN38_SERVING_ADAPTER_CONST(RESIDENT_DECODE_STAGE_KV_BLOCK_TOKENS);
		state->kv_block_count = state->resident_sequence_capacity * state->blocks_per_lane;
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingAllocatePools)(state);
		state->shim.input_scratch = state->gather_scratch;
	}
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSetEnvironment)(state);
	if ( status == SPARK_STATUS_OK )
		status = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingLoadDriver)(state,configuration);
	if ( status != SPARK_STATUS_OK )
	{
		SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDestroy)(state);
		return(status);
	}
	*adapter_state = state;
	return(SPARK_STATUS_OK);
}

static const SparkModelServingAdapterInterface SPARK_QWEN38_SERVING_ADAPTER_FN(ServingInterface) =
{
	.abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION,
	.interface_bytes = SPARK_MODEL_SERVING_ADAPTER_INTERFACE_BYTES,
	.descriptor = &SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDescriptor),
	.initialize = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingInitialize),
	.destroy = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingDestroy),
	.validate_submission = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingValidateSubmission),
	.submit = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSubmit),
	.progress = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingProgress),
	.quiesce = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingQuiesce),
	.snapshot = SPARK_QWEN38_SERVING_ADAPTER_FN(ServingSnapshot)
};

__attribute__((visibility("default")))
const SparkModelServingAdapterInterface *SparkModelServingAdapterGetInterface(void)
{
	return(&SPARK_QWEN38_SERVING_ADAPTER_FN(ServingInterface));
}
#endif
