#include <string.h>
#include <stdio.h>
#include "sparkpipe/spark_serving_cache_admission.h"

typedef struct TestState
{
	uint32_t validated,admitted,last_flags;
	uint64_t reject_submission;
	SparkStatus driver_status;
} TestState;

static SparkStatus TestValidate(void *context,const SparkModelServingSubmission *submission)
{
	TestState *state = (TestState *)context;
	state->validated++;
	return(submission->submission_id == state->reject_submission ? SPARK_STATUS_SCHEMA_ERROR : SPARK_STATUS_OK);
}

static SparkStatus TestAdmit(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	TestState *state = (TestState *)context;
	uint32_t lane;
	state->admitted++;
	state->last_flags = request->admission_flags;
	if ( request->cache_lane_count != 3u || request->transaction_id != 17u || request->control_generation != 9u )
	{
		fprintf(stderr,"request lanes=%u transaction=%llu control=%llu\n",request->cache_lane_count,(unsigned long long)request->transaction_id,(unsigned long long)request->control_generation);
		return(SPARK_STATUS_SCHEMA_ERROR);
	}
	for (lane=0u; lane<3u; lane++)
		if ( request->cache_lanes[lane].sequence_id != 100u + lane || request->cache_lanes[lane].resident_sequence_slot != 7u - lane || request->cache_lanes[lane].prefix_token_count != 64u || request->cache_lanes[lane].prefix_identity.sha256[0] != lane + 1u )
		{
			fprintf(stderr,"lane=%u sequence=%llu slot=%u prefix=%u identity=%u\n",lane,(unsigned long long)request->cache_lanes[lane].sequence_id,request->cache_lanes[lane].resident_sequence_slot,request->cache_lanes[lane].prefix_token_count,request->cache_lanes[lane].prefix_identity.sha256[0]);
			return(SPARK_STATUS_SCHEMA_ERROR);
		}
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(state->driver_status);
}

static void TestBuildSubmissions(SparkModelServingSubmission *submissions,SparkModelServingLane *lanes)
{
	uint32_t index;
	for (index=0u; index<3u; index++)
	{
		lanes[index].sequence_id = 100u + index;
		lanes[index].request_generation = lanes[index].step_generation = 1u;
		lanes[index].resident_sequence_slot = 7u - index;
		lanes[index].sequence_position = lanes[index].cache_prefix_token_count = 64u;
		lanes[index].context_token_count = 65u;
		lanes[index].flags = SPARK_MODEL_SERVING_LANE_FLAG_CACHE_PREFIX;
		lanes[index].cache_prefix_identity.sha256[0] = (uint8_t)(index + 1u);
	}
	submissions[0].abi_version = SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION;
	submissions[0].descriptor_bytes = SPARK_MODEL_SERVING_SUBMISSION_BYTES;
	submissions[0].submission_id = 1u;
	submissions[0].work_kind = SPARK_MODEL_SERVING_WORK_KIND_DECODE;
	submissions[0].active_sequence_count = submissions[0].lane_count = submissions[0].new_token_count = 3u;
	submissions[0].tokens_per_sequence = 1u;
	submissions[0].sequence_position = 64u;
	submissions[0].control_generation = 9u;
	submissions[0].request_generation = submissions[0].step_generation = 1u;
	submissions[0].transaction_id = 17u;
	submissions[0].lanes = lanes;
	submissions[1] = submissions[0];
	submissions[1].submission_id = 2u;
}

int main(void)
{
	SparkModelServingSubmission submissions[2] = {0};
	SparkModelServingLane lanes[3] = {0};
	SparkModelDriverCacheLane scratch[3];
	SparkModelDriverAdmissionRequest request;
	SparkModelDriverInterface driver = {0};
	TestState state = {0};
	SparkServingCacheAdmission cache = {1u,3u,scratch,&driver,&state,TestValidate,&state};
	SparkStatus status;
	driver.admit = TestAdmit;
	TestBuildSubmissions(submissions,lanes);
	status = SparkServingCacheAdmissionRun(&cache,submissions,2u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE);
	if ( status != SPARK_STATUS_OK || state.admitted != 2u || state.validated != 2u )
	{
		fprintf(stderr,"prepare status=%u admitted=%u validated=%u\n",status,state.admitted,state.validated);
		return(1);
	}
	if ( SparkServingCacheAdmissionRun(&cache,submissions,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) != SPARK_STATUS_OK || state.last_flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT )
		return(2);
	state.driver_status = SPARK_STATUS_IO_ERROR;
	if ( SparkServingCacheAdmissionRun(&cache,submissions,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) != SPARK_STATUS_IO_ERROR || state.last_flags != SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT )
		return(3);
	state.admitted = 0u;
	state.reject_submission = 2u;
	if ( SparkServingCacheAdmissionRun(&cache,submissions,2u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != SPARK_STATUS_SCHEMA_ERROR || state.admitted != 0u )
		return(4);
	if ( SparkServingCacheAdmissionRun(&cache,submissions,1u,0u) == SPARK_STATUS_OK || state.admitted != 0u )
		return(5);
	submissions[0].work_kind = SPARK_MODEL_SERVING_WORK_KIND_RELEASE;
	if ( SparkServingCacheAdmissionRun(&cache,submissions,1u,SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != SPARK_STATUS_OK || state.admitted != 0u )
		return(6);
	if ( SparkServingCacheBuildRequest(&cache,submissions,0u,&request) != SPARK_STATUS_OK || request.cache_lane_count != 3u || request.cache_lanes[0].flags != SPARK_MODEL_DRIVER_CACHE_LANE_FLAG_RELEASE )
		return(7);
	cache.lane_capacity = 2u;
	if ( SparkServingCacheBuildRequest(&cache,submissions,0u,&request) == SPARK_STATUS_OK )
		return(8);
	return(0);
}
