#pragma once

static SparkStatus SPARK_FAMILY(ModuleValidateFrameContext)(SPARK_FAMILY(ModuleState) *state, const SPARK_FAMILY(ResidentDecodeStageFrameContext) *context)
{
	uint32_t wants_input,wants_output,has_input,has_output;
	wants_input = state->stage_index != 0u ? 1u : 0u;
	wants_output = state->stage_index + 1u < state->stage_count ? 1u : 0u;
	if ( context == 0 )
		return((wants_input == 0u && wants_output == 0u) || state->allow_unqualified_execution != 0u ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	if ( context->abi_version != SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_ABI_VERSION) || context->descriptor_bytes != sizeof(*context) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	has_input = (context->flags & SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_INPUT_TRANSPORT)) != 0u ? 1u : 0u;
	has_output = (context->flags & SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_HIDDEN_OUTPUT_TRANSPORT)) != 0u ? 1u : 0u;
	if ( has_input != wants_input || has_output != wants_output )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_input != 0u && (context->hidden_input_transport_session == 0 || context->hidden_input_post_receive_function == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( has_output != 0u && (context->hidden_output_transport_session == 0 || context->hidden_output_send_function == 0) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (context->flags & SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_FRAME_CONTEXT_FLAG_PREFILL_FRAME_VIEW)) != 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(ModuleConsumeHiddenInput)(SPARK_FAMILY(ModuleSlot) *slot, SPARK_FAMILY(ResidentDecodeStageFrameContext) *context, uint32_t rows)
{
	SparkHiddenTransportPacket *packet = &context->hidden_input_packet;
	cudaStream_t stream = (cudaStream_t)slot->cuda_stream;
	cudaError_t error;
	SparkStatus status;
	memset(packet,0,sizeof(*packet));
	status = context->hidden_input_post_receive_function(context->hidden_input_transport_session,packet);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( packet->hidden_bf16 == 0 || packet->active_sequence_count < rows || packet->hidden_dimension != SPARK_FAMILY_CONST(MODEL_HIDDEN_DIMENSION) || packet->bytes_per_sequence < SPARK_FAMILY_CONST(MODEL_HIDDEN_BF16_BYTES) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = cudaMemcpyAsync(slot->hidden_bf16,packet->hidden_bf16,(uint64_t)rows * SPARK_FAMILY_CONST(MODEL_HIDDEN_BF16_BYTES),cudaMemcpyDeviceToDevice,stream);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"hidden_input"));
}
