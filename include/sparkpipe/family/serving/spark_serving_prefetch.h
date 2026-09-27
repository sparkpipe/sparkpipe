#pragma once

static SparkStatus SPARK_FAMILY(ServingPrefetch)(void *adapter_state,const SparkModelServingSubmission *submissions,uint32_t count)
{
	SPARK_FAMILY(ServingState) *state;
	SparkServingCacheAdmission cache;
	state = (SPARK_FAMILY(ServingState) *)adapter_state;
	if ( state == 0 || state->program == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	cache = SPARK_FAMILY(ServingCacheContext)(state,SPARK_FAMILY(ServingCacheScratch));
	return(SparkServingCacheAdmissionRun(&cache,submissions,count,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE));
}

static SparkStatus SPARK_FAMILY(ServingResolvePrefetch)(void *adapter_state,const SparkModelServingSubmission *submission,uint32_t resolution)
{
	SPARK_FAMILY(ServingState) *state;
	SparkServingCacheAdmission cache;
	uint32_t flags;
	state = (SPARK_FAMILY(ServingState) *)adapter_state;
	if ( state == 0 || state->program == 0 || (resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT && resolution != SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	flags = resolution == SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_COMMIT ? SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT : SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT;
	cache = SPARK_FAMILY(ServingCacheContext)(state,SPARK_FAMILY(ServingCacheScratch));
	return(SparkServingCacheAdmissionRun(&cache,submission,1u,flags));
}

static SparkStatus SPARK_FAMILY(ServingAbortUnexecuted)(void *adapter_state,const SparkModelServingSubmission *submission,SparkStatus failure)
{
	SparkStatus status;
	if ( failure == SPARK_STATUS_OK || failure == SPARK_STATUS_BUSY || submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(failure);
	status = SPARK_FAMILY(ServingResolvePrefetch)(adapter_state,submission,SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT);
	fprintf(stderr,"SUBMIT-ABORT submission=%llu kind=%u failure=%d abort=%d\n",(unsigned long long)submission->submission_id,submission->work_kind,(int)failure,(int)status);
	return(failure);
}
