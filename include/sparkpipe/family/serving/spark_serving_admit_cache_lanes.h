#pragma once

static SparkStatus SPARK_FAMILY(ServingAdmit)(SPARK_FAMILY(ServingState) *state,const SparkModelServingSubmission *submission,SPARK_FAMILY(ServingPending) *pending,SparkModelDriverFrame *frame)
{
	SparkServingCacheAdmission cache;
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	cache = SPARK_FAMILY(ServingCacheContext)(state,pending->cache_lanes);
	status = SparkServingCacheBuildRequest(&cache,submission,0u,&request);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	frame->cache_lanes = pending->cache_lanes;
	frame->cache_lane_count = request.cache_lane_count;
	return(SparkAdmissionEvaluateAndApply(state->driver.interface,state->driver_instance,&request,frame,&decision));
}
