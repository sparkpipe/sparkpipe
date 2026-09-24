#pragma once

static uint32_t SPARK_FAMILY(ServingAvailableSubmissionCount)(
	const SPARK_FAMILY(ServingState) *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += state->pending[index].common.active == 0u ? 1u : 0u;
	return(available);
}
