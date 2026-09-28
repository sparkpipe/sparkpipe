#pragma once

static SparkStatus SPARK_FAMILY(ServingAbortUnexecuted)(void *adapter_state,const SparkModelServingSubmission *submission,SparkStatus failure)
{
	SparkStatus status;
	if ( failure == SPARK_STATUS_OK || failure == SPARK_STATUS_BUSY || submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_RELEASE )
		return(failure);
	status = SPARK_FAMILY(ServingResolvePrefetch)(adapter_state,submission,SPARK_MODEL_SERVING_PREFETCH_RESOLUTION_ABORT);
	fprintf(stderr,"SUBMIT-ABORT submission=%llu kind=%u failure=%d abort=%d\n",(unsigned long long)submission->submission_id,submission->work_kind,(int)failure,(int)status);
	return(failure);
}
