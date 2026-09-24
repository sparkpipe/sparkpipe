#pragma once

static SparkStatus SPARK_FAMILY(PrepareClaimedContinuity)(void *prepare_context)
{
	SPARK_FAMILY(ClaimedContinuityContext) *context;
	context = (SPARK_FAMILY(ClaimedContinuityContext) *)prepare_context;
	return(SPARK_FAMILY(ValidateSequenceContinuity)(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions));
}

static void SPARK_FAMILY(InvalidateClaimedLanes)(
	SPARK_FAMILY(ModuleState) *state,
	const uint32_t *lane_indices,
	uint32_t lane_count)
{
	uint32_t lane;
	for (lane=0u; lane<lane_count; lane++)
		atomic_store_explicit(&state->lane_bound[lane_indices[lane]],0u,memory_order_release);
}
