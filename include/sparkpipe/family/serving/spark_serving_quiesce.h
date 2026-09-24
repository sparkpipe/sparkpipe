#pragma once

static SparkStatus SPARK_FAMILY(ServingQuiesce)(
	void *adapter_state,
	uint64_t deadline_time_ns)
{
	SPARK_FAMILY(ServingState) *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	SparkStatus status;
	state = (SPARK_FAMILY(ServingState) *)adapter_state;
	if ( state == 0 || deadline_time_ns == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	state->quiescing = 1u;
	if ( SPARK_FAMILY(ServingAvailableSubmissionCount)(state) != state->pipeline_slot_count )
		return(SPARK_STATUS_BUSY);
	memset(&snapshot,0,sizeof(snapshot));
	status = state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(snapshot.active_submission_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_BUSY);
}
