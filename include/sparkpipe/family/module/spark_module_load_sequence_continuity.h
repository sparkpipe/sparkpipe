#pragma once

static SparkStatus SPARK_FAMILY(LoadSequenceContinuity)(const SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	const SparkKvLaneTransaction *owner;
	uint32_t lane,slot;
	if ( state->kv_lane_transactions == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		slot = batch->row_resident_slots[lane];
		if ( slot >= state->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&state->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&state->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&state->lane_next_positions[slot],memory_order_acquire);
		owner = &state->kv_lane_transactions[slot];
		if ( SPARK_FAMILY(PrefixRestorePending)(owner) != 0u )
		{
			if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != batch->row_sequence_ids[lane] || owner->lane.sequence_position != batch->row_positions[lane] )
				SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
			bound[lane] = 1u;
			sequence_ids[lane] = owner->lane.sequence_id;
			next_positions[lane] = owner->lane.sequence_position;
		}
	}
	return(SPARK_STATUS_OK);
}
