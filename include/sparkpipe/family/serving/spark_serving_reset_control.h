#pragma once

static SparkStatus SPARK_FAMILY(ServingResetControl)(void *adapter_state,
	uint64_t control_generation)
{
	SPARK_FAMILY(ServingState) *state = (SPARK_FAMILY(ServingState) *)adapter_state;
	SparkModelDriverAdmissionRequest request = {0};
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	if ( state == 0 || control_generation == 0u ||
		control_generation <= atomic_load_explicit(&state->reset_generation,memory_order_acquire) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SPARK_FAMILY(ServingQuiesce)(state,UINT64_MAX);
	if ( status != SPARK_STATUS_OK )
		return(status);
	request.descriptor_bytes = (uint32_t)sizeof(request);
	request.program_id = state->program->program_id;
	request.control_generation = control_generation;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_RESET;
	SparkModelDriverInitializeAdmissionDecision(&decision);
	status = state->driver.interface->admit(state->driver_instance,&request,&decision);
	if ( status == SPARK_STATUS_OK && decision.accepted == 0u )
		status = SPARK_STATUS_VALIDATION_FAILED;
	if ( status == SPARK_STATUS_OK )
	{
		atomic_store_explicit(&state->reset_generation,control_generation,memory_order_release);
		state->quiescing = 0u;
	}
	return(status);
}
