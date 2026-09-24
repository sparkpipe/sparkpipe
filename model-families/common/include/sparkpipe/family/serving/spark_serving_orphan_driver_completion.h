#pragma once

static void SPARK_FAMILY(ServingOrphanDriverCompletion)(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SPARK_FAMILY(ServingState) *state;
	(void)driver_completion;
	state = (SPARK_FAMILY(ServingState) *)completion_context;
	if ( state != 0 )
		state->orphan_completion_count++;
}

static void SPARK_FAMILY(ServingDriverWake)(void *wake_context)
{
	SPARK_FAMILY(ServingState) *state;
	state = (SPARK_FAMILY(ServingState) *)wake_context;
	if ( state != 0 && state->wake_function != 0 )
		state->wake_function(state->wake_context);
}
