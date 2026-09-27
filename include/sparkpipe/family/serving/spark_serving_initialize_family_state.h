#pragma once

static SparkStatus SPARK_FAMILY(ServingInitializeFamilyState)(SPARK_FAMILY(ServingState) *state)
{
	if ( state == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	atomic_init(&state->reset_active,0u);
	atomic_init(&state->reset_generation,0u);
	return(SPARK_STATUS_OK);
}
