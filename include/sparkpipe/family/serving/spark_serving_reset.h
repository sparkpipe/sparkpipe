#pragma once

static SparkStatus SPARK_FAMILY(ServingReset)(
	void *adapter_state,
	uint64_t control_generation)
{
	SPARK_FAMILY(ServingState) *state = (SPARK_FAMILY(ServingState) *)adapter_state;
	uint32_t expected = 0u;
	SparkStatus status;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( atomic_compare_exchange_strong_explicit(&state->reset_active,&expected,1u,memory_order_acquire,memory_order_relaxed) == 0 )
		return(SPARK_STATUS_BUSY);
	status = SPARK_FAMILY(ServingResetControl)(state,control_generation);
	atomic_store_explicit(&state->reset_active,0u,memory_order_release);
	return(status);
}
