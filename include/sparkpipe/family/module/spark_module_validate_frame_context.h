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
	return(SPARK_STATUS_OK);
}
