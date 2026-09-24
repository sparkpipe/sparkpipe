#pragma once

static SparkServingCacheAdmission SPARK_FAMILY(ServingCacheContext)(SPARK_FAMILY(ServingState) *state,SparkModelDriverCacheLane *lanes)
{
	SparkServingCacheAdmission cache;
	cache.program_id = state->program->program_id;
	cache.lane_capacity = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT);
	cache.lanes = lanes;
	cache.driver = state->driver.interface;
	cache.driver_instance = state->driver_instance;
	cache.validate = SPARK_FAMILY(ServingValidateSubmission);
	cache.adapter_state = state;
	return(cache);
}
