#pragma once

static uint32_t SPARK_FAMILY(ServingSubmissionStale)(const SPARK_FAMILY(ServingState) *state,const SparkModelServingSubmission *submission)
{
	if ( submission == 0 )
		return(0u);
	return(submission->control_generation < atomic_load_explicit(&state->reset_generation,memory_order_acquire) ? 1u : 0u);
}
