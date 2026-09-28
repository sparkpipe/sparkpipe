#pragma once

static SparkStatus SPARK_FAMILY(ValidateFrame)(const SPARK_FAMILY(ModuleState) *state,const SparkModelDriverFrame *frame,const SPARK_FAMILY(ResidentDecodeStageFrameContext) **context_out)
{
	const SPARK_FAMILY(ResidentDecodeStageFrameContext) *context;
	const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch;
	uint32_t expected_flags,prefill;
	uint64_t boundary_bytes;
	SparkStatus status;
	if ( state == 0 || frame == 0 || context_out == 0 || frame->user_context == 0 || frame->execution_stream != state->execution_stream || frame->completion_function == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	context = (const SPARK_FAMILY(ResidentDecodeStageFrameContext) *)frame->user_context;
	if ( context->abi_version != SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION) || context->descriptor_bytes != sizeof(*context) || context->reserved0 != 0u || (context->flags & ~SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_KNOWN_FLAGS)) != 0u || context->batch == 0 )
		SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
	batch = context->batch;
	if ( batch->abi_version != SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_BATCH_VIEW_ABI_VERSION) || batch->descriptor_bytes != sizeof(*batch) || batch->row_count == 0u || batch->row_count > SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT) || batch->active_sequence_count == 0u || batch->active_sequence_count > state->resident_sequence_capacity || batch->row_resident_slots == 0 || batch->row_positions == 0 || batch->row_sequence_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	prefill = (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_PREFILL) != 0u ? 1u : 0u;
	if ( prefill == 0u && batch->row_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( frame->active_slot_count != batch->active_sequence_count || frame->new_token_count != batch->row_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( state->owns_embedding != 0u && batch->token_ids == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	expected_flags = prefill != 0u ? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_FLAG_PREFILL) : 0u;
	expected_flags |= state->owns_embedding == 0u ? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_INPUT) : 0u;
	expected_flags |= state->owns_final_head == 0u ? SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_FLAG_HIDDEN_OUTPUT) : 0u;
	if ( context->flags != expected_flags )
		SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
	boundary_bytes = (uint64_t)batch->row_count * SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT) * SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES);
	if ( (state->owns_embedding == 0u && (context->hidden_input_bf16 == 0 || context->hidden_input_bytes < boundary_bytes)) || (state->owns_embedding != 0u && (context->hidden_input_bf16 != 0 || context->hidden_input_bytes != 0u)) || (state->owns_final_head == 0u && (context->hidden_output_bf16 == 0 || context->hidden_output_bytes < boundary_bytes)) || (state->owns_final_head != 0u && (context->hidden_output_bf16 != 0 || context->hidden_output_bytes != 0u)) )
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	status = SPARK_FAMILY(ValidateRoundMajor)(state,batch);
	if ( status == SPARK_STATUS_OK )
		status = SPARK_FAMILY(ValidateFrameBuffers)(state,frame,batch->row_count);
	*context_out = status == SPARK_STATUS_OK ? context : 0;
	return(status);
}
