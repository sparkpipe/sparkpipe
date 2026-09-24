#pragma once

static void SPARK_FAMILY(ServingDestroy)(void *adapter_state)
{
	SPARK_FAMILY(ServingState) *state;
	SparkModelDriverRuntimeSnapshot snapshot;
	state = (SPARK_FAMILY(ServingState) *)adapter_state;
	if ( state == 0 )
		return;
	if ( SPARK_FAMILY(ServingAvailableSubmissionCount)(state) != state->pipeline_slot_count )
		return;
	if ( state->driver.interface != 0 && state->driver.interface->snapshot != 0 && state->driver_instance != 0 && state->program != 0 )
	{
		memset(&snapshot,0,sizeof(snapshot));
		if ( state->driver.interface->snapshot(state->driver_instance,state->program->program_id,&snapshot) != SPARK_STATUS_OK || snapshot.active_submission_count != 0u )
			return;
	}
	if ( state->driver.interface != 0 && state->driver.interface->destroy != 0 && state->driver_instance != 0 )
		state->driver.interface->destroy(state->driver_instance);
	SparkUnloadModelDriver(&state->driver);
	free(state);
}
