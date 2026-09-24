#pragma once

static SPARK_FAMILY(ServingPending) *SPARK_FAMILY(ServingReservePending)(
	SPARK_FAMILY(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	SPARK_FAMILY(ServingPending) *pending;
	uint32_t index,lane,row,expected;
	if ( atomic_load_explicit(&state->quiescing,memory_order_acquire) != 0u )
		return(0);
	for (index=0u; index<state->pipeline_slot_count; index++)
	{
		pending = &state->pending[index];
		expected = 0u;
		if ( atomic_compare_exchange_strong_explicit(&pending->active,&expected,1u,memory_order_acquire,memory_order_relaxed) != 0 )
		{
			if ( atomic_load_explicit(&state->quiescing,memory_order_acquire) != 0u )
			{
				atomic_store_explicit(&pending->active,0u,memory_order_release);
				return(0);
			}
			pending->owner = state;
			pending->row_count = submission->row_count;
			pending->lane_count = submission->lane_count;
			pending->active_sequence_count = submission->active_sequence_count;
			pending->work_kind = submission->work_kind;
			pending->submission_id = submission->submission_id;
			pending->request_id = submission->request_id;
			pending->sequence_id = submission->sequence_id;
			pending->sequence_position = submission->sequence_position;
			pending->control_generation = submission->control_generation;
			pending->transaction_id = submission->transaction_id;
			pending->dispatch_generation = submission->dispatch_generation;
			pending->request_generation = submission->request_generation;
			pending->step_generation = submission->step_generation;
			for (row=0u; row<submission->row_count; row++)
			{
				lane = submission->row_lane_indices[row];
				pending->last_row_by_lane[lane] = row;
				pending->resident_slots[row] = submission->lanes[lane].resident_sequence_slot;
				pending->input_token_ids[row] = submission->token_ids[row];
				pending->row_positions[row] = submission->row_positions[row];
				pending->row_sequence_ids[row] = submission->row_sequence_ids[row];
			}
			return(pending);
		}
	}
	return(0);
}

static uint32_t SPARK_FAMILY(ServingAvailableSubmissionCount)(
	const SPARK_FAMILY(ServingState) *state)
{
	uint32_t available,index;
	available = 0u;
	for (index=0u; index<state->pipeline_slot_count; index++)
		available += atomic_load_explicit(&state->pending[index].active,memory_order_acquire) == 0u ? 1u : 0u;
	return(available);
}
