#pragma once

#include "sparkpipe/spark_admission.h"

typedef struct SparkServingCacheAdmission
{
	uint32_t program_id;
	uint32_t lane_capacity;
	SparkModelDriverCacheLane *lanes;
	const SparkModelDriverInterface *driver;
	void *driver_instance;
	SparkModelServingAdapterValidateSubmissionFunction validate;
	void *adapter_state;
} SparkServingCacheAdmission;

static inline SparkStatus SparkServingCacheBuildRequest(const SparkServingCacheAdmission *cache,const SparkModelServingSubmission *submission,uint32_t flags,SparkModelDriverAdmissionRequest *request)
{
	uint32_t count;
	SparkStatus status;
	if ( cache == 0 || submission == 0 || request == 0 || cache->lanes == 0 || cache->lane_capacity == 0u || cache->program_id == 0u || submission->lane_count < submission->active_sequence_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkModelServingAdapterBuildDriverCacheLanes(submission,cache->lanes,cache->lane_capacity,&count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	if ( count != submission->active_sequence_count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	return(SparkAdmissionRequestFromSubmission(cache->program_id,submission,cache->lanes,flags,request));
}

// Prepare warms pages without acquiring execution ownership. Commit/abort act
// on one submission; the driver owns the corresponding lane transaction.
static inline SparkStatus SparkServingCacheAdmissionRun(const SparkServingCacheAdmission *cache,const SparkModelServingSubmission *submissions,uint32_t count,uint32_t flags)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkStatus status;
	uint32_t index;
	if ( cache == 0 || cache->validate == 0 || cache->driver == 0 || submissions == 0 || count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE && flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT && flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE && count != 1u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<count; index++)
	{
		status = cache->validate(cache->adapter_state,&submissions[index]);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	for (index=0u; index<count; index++)
	{
		if ( submissions[index].work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
			continue;
		status = SparkServingCacheBuildRequest(cache,&submissions[index],flags,&request);
		if ( status == SPARK_STATUS_OK )
			status = SparkAdmissionEvaluate(cache->driver,cache->driver_instance,&request,&decision);
		if ( status != SPARK_STATUS_OK )
			return(status);
	}
	return(SPARK_STATUS_OK);
}
