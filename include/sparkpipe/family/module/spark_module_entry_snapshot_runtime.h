#pragma once

SparkStatus SPARK_FAMILY(ResidentDecodeStageSnapshot)(void *module_state,uint32_t program_id,SparkModelDriverRuntimeSnapshot *snapshot)
{
	SPARK_FAMILY(ModuleState) *state;
	uint32_t index,resident_count;
	state = (SPARK_FAMILY(ModuleState) *)module_state;
	if ( state == 0 || snapshot == 0 || program_id == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	SparkStageModuleRuntimeSnapshotInitialize(snapshot,program_id,state->slot_states,state->pipeline_slot_count);
	snapshot->submitted_count = atomic_load_explicit(&state->submitted_count,memory_order_relaxed);
	snapshot->completed_count = atomic_load_explicit(&state->completed_count,memory_order_relaxed);
	snapshot->rejected_count = atomic_load_explicit(&state->rejected_count,memory_order_relaxed);
	snapshot->host_callback_completion_count = atomic_load_explicit(&state->host_callback_completion_count,memory_order_relaxed);
	resident_count = 0u;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		resident_count += atomic_load_explicit(&state->lane_bound[index],memory_order_acquire) != 0u ? 1u : 0u;
	snapshot->resident_sequence_count = resident_count;
	snapshot->kv_token_capacity = (uint64_t)state->resident_sequence_capacity * state->max_sequence_positions;
	return(SPARK_STATUS_OK);
}
