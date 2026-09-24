#pragma once

static void SPARK_FAMILY(ServingOrphanDriverCompletion)(
	void *completion_context,
	const SparkModelDriverCompletion *driver_completion)
{
	SPARK_FAMILY(ServingState) *state;
	(void)driver_completion;
	state = (SPARK_FAMILY(ServingState) *)completion_context;
	if ( state != 0 )
		atomic_fetch_add_explicit(&state->orphan_completion_count,1u,memory_order_relaxed);
}
