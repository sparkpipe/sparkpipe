#pragma once

#include <string.h>

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

static inline SparkStatus SparkServingCacheHintRun(const SparkServingCacheAdmission *cache,const SparkModelServingCacheIdentity *identity,uint32_t token_count)
{
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverAdmissionDecision decision;
	SparkModelDriverCacheLane lane;
	if ( cache == 0 || cache->driver == 0 || cache->program_id == 0u || identity == 0 || token_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(&lane,0,sizeof(lane));
	lane.flags = SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_PREFIX;
	lane.prefix_token_count = token_count;
	memcpy(lane.prefix_identity.sha256,identity->sha256,sizeof(lane.prefix_identity.sha256));
	memset(&request,0,sizeof(request));
	request.descriptor_bytes = (uint32_t)sizeof(request);
	request.program_id = cache->program_id;
	request.admission_flags = SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_HINT;
	request.cache_lanes = &lane;
	request.cache_lane_count = 1u;
	return(SparkAdmissionEvaluate(cache->driver,cache->driver_instance,&request,&decision));
}
