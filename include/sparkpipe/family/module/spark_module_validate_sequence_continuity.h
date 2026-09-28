#pragma once

static SparkStatus SPARK_FAMILY(ValidateSequenceContinuity)(const SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	uint8_t touched[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	for (lane=0u; lane<batch->active_sequence_count; lane++)
	{
		slot = batch->row_resident_slots[lane];
		if ( slot >= state->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		bound[lane] = atomic_load_explicit(&state->lane_bound[slot],memory_order_acquire);
		sequence_ids[lane] = atomic_load_explicit(&state->lane_sequence_ids[slot],memory_order_acquire);
		next_positions[lane] = atomic_load_explicit(&state->lane_next_positions[slot],memory_order_acquire);
	}
	for (row=0u; row<batch->row_count; row++)
	{
		slot = batch->row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,slot,&lane);
		position = batch->row_positions[row];
		sequence = batch->row_sequence_ids[row];
		if ( status != SPARK_STATUS_OK || lane >= batch->active_sequence_count || batch->row_resident_slots[lane] != slot || position >= state->max_sequence_positions )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( position == 0u )
		{
			if ( touched[lane] != 0u )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			bound[lane] = 1u;
			sequence_ids[lane] = sequence;
			next_positions[lane] = 1u;
		}
		else
		{
			if ( bound[lane] == 0u || sequence_ids[lane] != sequence || next_positions[lane] != position )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			next_positions[lane] = position + 1u;
		}
		touched[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}
