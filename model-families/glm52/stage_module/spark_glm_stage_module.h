#pragma once

#include <stdint.h>

#include "sparkpipe/llm_defines.h"

static uint32_t SparkGlmStagePackRangesOverlap(const SPARK_GLM_STAGE_PACK_RANGE *left,const SPARK_GLM_STAGE_PACK_RANGE *right)
{
	return(left->bytes != 0u && right->bytes != 0u && left->offset < right->offset + right->bytes && right->offset < left->offset + left->bytes ? 1u : 0u);
}

static SparkStatus SparkGlmStagePackValidateRanges(
	const SPARK_GLM_STAGE_ENTRY *entries,
	uint32_t entry_count)
{
	SPARK_GLM_STAGE_PACK_RANGE left[2],right[2];
	uint32_t left_index,right_index,left_part,right_part;
	for (left_index=0u; left_index<entry_count; left_index++)
	{
		left[0].offset = entries[left_index].payload_offset;
		left[0].bytes = entries[left_index].payload_bytes;
		left[1].offset = entries[left_index].scale_offset;
		left[1].bytes = entries[left_index].scale_bytes;
		for (right_index=left_index + 1u; right_index<entry_count; right_index++)
		{
			right[0].offset = entries[right_index].payload_offset;
			right[0].bytes = entries[right_index].payload_bytes;
			right[1].offset = entries[right_index].scale_offset;
			right[1].bytes = entries[right_index].scale_bytes;
			for (left_part=0u; left_part<2u; left_part++)
				for (right_part=0u; right_part<2u; right_part++)
					if ( SparkGlmStagePackRangesOverlap(&left[left_part],&right[right_part]) != 0u )
						SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
		}
		if ( SparkGlmStagePackRangesOverlap(&left[0],&left[1]) != 0u )
			SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static uint64_t SparkGlmStageExpectedLayerMask(
	const SPARK_GLM_STAGE_STATE *state,
	uint32_t layer_index)
{
	SPARK_GLM_STAGE_TENSOR_SHAPE shape;
	uint64_t mask;
	uint32_t kind;
	mask = 0u;
	for (kind=SPARK_GLM_STAGE_TENSOR_FIRST; kind<SPARK_GLM_STAGE_TENSOR_KIND_COUNT; kind++)
		if ( SPARK_GLM_STAGE_EXPECTED_SHAPE(kind,layer_index,state->expert_weight_codec,state->tp_degree,&shape) == 0 )
			mask |= UINT64_C(1) << kind;
	return(mask);
}

static uint64_t SparkGlmStageExpectedGlobalMask(const SPARK_GLM_STAGE_STATE *state)
{
	uint64_t mask;
	mask = 0u;
	if ( state->owns_embedding != 0u )
		mask |= UINT64_C(1) << SPARK_GLM_STAGE_TENSOR_EMBEDDING;
	if ( state->owns_final_head != 0u )
		mask |= (UINT64_C(1) << SPARK_GLM_STAGE_TENSOR_FINAL_NORM) | (UINT64_C(1) << SPARK_GLM_STAGE_TENSOR_LM_HEAD);
	return(mask);
}

static SparkStatus SparkGlmStageAllocateBytes(
	SPARK_GLM_STAGE_STATE *state,
	uint64_t count,
	uint64_t width,
	uint64_t element_bytes,
	void **pointer)
{
	uint64_t bytes;
	if ( state == 0 || pointer == 0 || count == 0u || width == 0u || element_bytes == 0u || count > UINT64_MAX / width || count * width > UINT64_MAX / element_bytes )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	bytes = count * width * element_bytes;
	return(SparkStageModuleDeviceAllocate(&state->ledger,bytes,pointer));
}

static SparkStatus SparkGlmStageAllocateRows(
	SPARK_GLM_STAGE_STATE *state,
	uint64_t rows,
	uint64_t columns,
	void **pointer)
{
	return(SparkGlmStageAllocateBytes(state,rows,columns,sizeof(uint16_t),pointer));
}

static SparkStatus SparkGlmStageAllocateSlotHead(
	SPARK_GLM_STAGE_STATE *state,
	SPARK_GLM_STAGE_SLOT *slot)
{
	uint64_t rows,tiles;
	SparkStatus status;
	rows = state->execution_row_capacity;
	tiles = SparkCeilDivU64(SPARK_LLM_OUTPUT_VOCAB_COUNT,SPARK_LLM_HEAD_TILE);
	status = SparkGlmStageAllocateBytes(state,rows,tiles,sizeof(float),(void **)&slot->head_candidate_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,tiles,sizeof(uint32_t),(void **)&slot->head_candidate_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,1u,sizeof(uint32_t),(void **)&slot->output_token);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,1u,sizeof(float),(void **)&slot->output_score);
	if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateBytes(state,rows,1u,sizeof(uint64_t),(void **)&slot->head_maxloc_u64);
	if ( status == SPARK_STATUS_OK && state->owns_final_head != 0u )
	{
		uint64_t shard_rows = SPARK_LLM_OUTPUT_VOCAB_COUNT / state->tp_degree;
		status = SparkGlmStageAllocateBytes(state,1u,SparkHeadCertifiedFp8ScratchBytes(shard_rows,SPARK_LLM_HIDDEN_DIMENSION),1u,(void **)&slot->head_certified_scratch);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlmStageAllocateBytes(state,1u,SparkHeadCertifiedFp8CandidateBytes(shard_rows),1u,(void **)&slot->head_certified_candidates);
		if ( status == SPARK_STATUS_OK )
			status = SparkGlmStageAllocateBytes(state,1u,1u,sizeof(uint32_t),(void **)&slot->head_screened_count);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlmStageAllocateSlots(SPARK_GLM_STAGE_STATE *state)
{
	uint32_t index;
	SparkStatus status;
	status = SPARK_STATUS_OK;
	for (index=0u; status==SPARK_STATUS_OK && index<state->pipeline_slot_count; index++)
	{
		state->slots[index].stream = state->execution_stream;
		status = SPARK_GLM_STAGE_ALLOCATE_SLOT_HOST(&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SPARK_GLM_STAGE_ALLOCATE_SLOT_METADATA(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SPARK_GLM_STAGE_ALLOCATE_SLOT_HIDDEN(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SPARK_GLM_STAGE_ALLOCATE_SLOT_MLP(state,&state->slots[index]);
		if ( status == SPARK_STATUS_OK ) status = SparkGlmStageAllocateSlotHead(state,&state->slots[index]);
	}
	SPARK_RETURN(status);
}

static SparkStatus SparkGlmStageBuildHeadShadow(SPARK_GLM_STAGE_STATE *state)
{
	uint64_t head_rows,dim;
	SparkStatus status;
	if ( state->owns_final_head == 0u || state->lm_head_bf16 == 0 )
		return(SPARK_STATUS_OK);
	head_rows = SPARK_LLM_OUTPUT_VOCAB_COUNT / state->tp_degree;
	dim = SPARK_LLM_HIDDEN_DIMENSION;
	status = SparkGlmStageAllocateBytes(state,head_rows,dim,1u,(void **)&state->head_certified_fp8_payload);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_scale_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkGlmStageAllocateBytes(state,head_rows,dim / 32u,sizeof(float),(void **)&state->head_certified_fp8_norm_f32);
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM_STAGE_MODULE_TAG,SparkGlmLaunchHeadCertifiedQuantize(0,state->lm_head_bf16,state->head_certified_fp8_payload,state->head_certified_fp8_scale_f32,state->head_certified_fp8_norm_f32,(uint32_t)head_rows,(uint32_t)dim),"head_certified_quantize");
	if ( status == SPARK_STATUS_OK )
		status = SparkStageModuleCudaStatus(SPARK_GLM_STAGE_MODULE_TAG,cudaDeviceSynchronize(),"head_certified_sync");
	SPARK_RETURN(status);
}

static SparkStatus SparkGlmStageEnqueueAsyncCompletion(
	SPARK_GLM_STAGE_STATE *state,
	SPARK_GLM_STAGE_SLOT *slot,
	uint32_t slot_index)
{
	cudaStream_t stream;
	cudaError_t error;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->host_kv_access_error,slot->kv_access_error,SPARK_GLM_STAGE_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc(stream,SPARK_GLM_STAGE_COMPLETE_ASYNC,&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_GLM_STAGE_MODULE_TAG,error,"async_completion"));
}

static void SparkGlmStagePrepareAsyncCompletion(
	SPARK_GLM_STAGE_STATE *state,
	SparkModelDriverFrame *frame,
	const SPARK_GLM_STAGE_BATCH *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SPARK_GLM_STAGE_COMPLETION *async;
	uint32_t lane;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = batch->active_sequence_count;
	async->row_count = batch->row_count;
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[0].address : 0;
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT; lane++)
	{
		async->lane_indices[lane] = batch->row_resident_slots[lane];
		async->lane_bound[lane] = lane_bound[lane];
		async->lane_sequence_ids[lane] = lane_sequence_ids[lane];
		async->lane_next_positions[lane] = lane_next_positions[lane];
	}
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)batch->row_count * sizeof(uint32_t) * (3u + state->owns_final_head);
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes;
}

static uint32_t SparkGlmStageRoundMajorWaveRows(
	const SPARK_GLM_STAGE_STATE *state,
	const SPARK_GLM_STAGE_BATCH *batch,
	uint32_t first_row)
{
	SparkStageModuleClaimedLaneContext lanes;
	if ( state == 0 || batch == 0 || batch->active_sequence_count == 0u )
		return(0u);
	lanes.index_states = state->lane_states;
	lanes.index_capacity = state->resident_sequence_capacity;
	return(SparkRowLayoutRoundMajorWaveRowCount(first_row,batch->row_count,batch->row_resident_slots,SparkStageModuleClaimedLaneOrdinal,&lanes));
}

static SparkStatus SparkGlmStageStageHostBatch(
	const SPARK_GLM_STAGE_STATE *state,
	SPARK_GLM_STAGE_SLOT *slot,
	const SPARK_GLM_STAGE_BATCH *batch)
{
	uint32_t row;
	if ( state == 0 || slot == 0 || batch == 0 || batch->row_count > SPARK_GLM_STAGE_MAX_INPUT_ROW_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (row=0u; row<batch->row_count; row++)
	{
		if ( batch->row_positions[row] >= UINT32_MAX )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		slot->host_resident_slots[row] = batch->row_resident_slots[row];
		slot->host_positions[row] = (uint32_t)batch->row_positions[row];
		if ( state->owns_embedding != 0u )
			slot->host_token_ids[row] = batch->token_ids[row];
	}
	memset(slot->host_kv_access_error,0,SPARK_GLM_STAGE_KV_ACCESS_ERROR_WORD_COUNT * sizeof(uint32_t));
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkGlmStageValidateRoundMajor(
	const SPARK_GLM_STAGE_STATE *state,
	const SPARK_GLM_STAGE_BATCH *batch)
{
	uint32_t ordinals[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t counts[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	uint32_t last_rows[SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT];
	SparkRowLayoutDirectLaneContext lanes;
	SparkStatus status;
	if ( state == 0 || batch == 0 || batch->row_count < batch->active_sequence_count || state->resident_sequence_capacity > SPARK_GLM_STAGE_MAX_ACTIVE_SEQUENCE_COUNT )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkRowLayoutDirectLaneMapInitialize(&lanes,ordinals,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkRowLayoutValidateRoundMajor(batch->row_count,batch->active_sequence_count,batch->row_resident_slots,SparkRowLayoutDirectLaneOrdinal,&lanes,counts,last_rows));
}

static SparkStatus SparkGlmStageValidateFrame(
	const SPARK_GLM_STAGE_STATE *state,
	const SparkModelDriverFrame *frame,
	const SPARK_GLM_STAGE_FRAME **context_out)
{
	const SPARK_GLM_STAGE_FRAME *context;
	const SPARK_GLM_STAGE_BATCH *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes,sideband_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SPARK_GLM_STAGE_FRAME *)frame->user_context;
	if ( context->abi_version != SPARK_GLM_STAGE_FRAME_CONTEXT_ABI_VERSION || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_GLM_STAGE_FRAME_KNOWN_FLAGS) != 0u || context->batch == 0 )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_GLM_STAGE_BATCH_VIEW_ABI_VERSION || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > SPARK_GLM_STAGE_MAX_INPUT_ROW_COUNT || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_embedding != 0u && batch->token_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_GLM_STAGE_FRAME_FLAG_PREFILL : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_INPUT : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_GLM_STAGE_FRAME_FLAG_HIDDEN_OUTPUT : 0u;
	expected_flags |= SPARK_GLM_STAGE_REQUIRES_SIDEBAND_INPUT(state->stage_index) != 0u ? SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_INPUT : 0u;
	expected_flags |= SPARK_GLM_STAGE_REQUIRES_SIDEBAND_OUTPUT(state->stage_index) != 0u ? SPARK_GLM_STAGE_FRAME_FLAG_SIDEBAND_OUTPUT : 0u;
	if ( context->flags != expected_flags )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_GLM_STAGE_BOUNDARY_ELEMENT_COUNT * SPARK_GLM_STAGE_BOUNDARY_ELEMENT_BYTES;
	sideband_bytes = (uint64_t)batch->row_count * SPARK_GLM_STAGE_DSA_SIDEBAND_BYTES_PER_ROW;
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( (SPARK_GLM_STAGE_REQUIRES_SIDEBAND_INPUT(state->stage_index) != 0u && (context->sideband_input == 0 || context->sideband_input_bytes < sideband_bytes)) || (SPARK_GLM_STAGE_REQUIRES_SIDEBAND_INPUT(state->stage_index) == 0u && (context->sideband_input != 0 || context->sideband_input_bytes != 0u)) || (SPARK_GLM_STAGE_REQUIRES_SIDEBAND_OUTPUT(state->stage_index) != 0u && (context->sideband_output == 0 || context->sideband_output_bytes < sideband_bytes)) || (SPARK_GLM_STAGE_REQUIRES_SIDEBAND_OUTPUT(state->stage_index) == 0u && (context->sideband_output != 0 || context->sideband_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SparkGlmStageValidateRoundMajor(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SPARK_GLM_STAGE_VALIDATE_FRAME_BUFFERS(state,frame,batch->row_count);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	SPARK_RETURN(status);
}

static void SparkGlmStageLazyRetryRetained(void *context)
{
	SPARK_GLM_STAGE_STATE *state = (SPARK_GLM_STAGE_STATE *)context;
	SPARK_GLM_STAGE_CHAIN *chain;
	uint32_t slot;
	for (slot=0u; slot<state->pipeline_slot_count; slot++)
		if ( SPARK_GLM_STAGE_LAZY_RECOVER_LEASE(state,slot,&chain) == SPARK_STATUS_OK )
			SPARK_GLM_STAGE_TP_CHAIN_FAIL(chain,chain->retained_status);
}
