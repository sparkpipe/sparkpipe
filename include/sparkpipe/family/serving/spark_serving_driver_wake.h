#pragma once

static void SPARK_FAMILY(ServingDriverWake)(void *wake_context)
{
	SPARK_FAMILY(ServingState) *state;
	state = (SPARK_FAMILY(ServingState) *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}
