#pragma once

static SparkStatus SPARK_FAMILY(ServingAcceptsProgram)(
	const SparkModelDriverProgramDescriptor *program,
	void *accept_context)
{
	SPARK_FAMILY(ServingState) *state;
	state = (SPARK_FAMILY(ServingState) *)accept_context;
	if ( SparkModelDriverProgramSupportsRuntimeLimits(program,SPARK_FAMILY_CONST(SERVING_REQUIRED_PROGRAM_FLAGS),state->pipeline_slot_count,state->max_active_sequence_count,state->max_input_row_count,state->resident_sequence_capacity) == 0u )
		return(SPARK_STATUS_TARGET_MISMATCH);
	return(SPARK_STATUS_OK);
}
