#pragma once

static SparkStatus SPARK_FAMILY(LaneRecordsRollback)(SPARK_FAMILY(ModuleState) *state,const SparkModelDriverAdmissionRequest *request,uint32_t lane_count)
{
	SparkKvLaneTransaction *owner;
	uint32_t lane_index;
	SparkStatus result,status;
	result = SPARK_STATUS_OK;
	for (lane_index=0u; lane_index<lane_count; lane_index++)
	{
		owner = &state->kv_lane_transactions[request->cache_lanes[lane_index].resident_sequence_slot];
		status = owner->phase == SPARK_KV_LANE_TRANSACTION_COMMITTED ? SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,&owner->lane,owner->mutation_flags) : SPARK_STATUS_OK;
		owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
		if ( status != SPARK_STATUS_OK && result == SPARK_STATUS_OK )
			result = status;
	}
	SPARK_RETURN(result);
}

static SparkStatus SPARK_FAMILY(LaneRecordsPrepare)(SPARK_FAMILY(ModuleState) *state,const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverCacheLane *lane;
	uint32_t lane_index,slot;
	SparkStatus status;
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
	{
		lane = &request->cache_lanes[lane_index];
		slot = lane->resident_sequence_slot;
		if ( state->kv_lane_transactions[slot].phase != SPARK_KV_LANE_TRANSACTION_COMMITTED )
			state->kv_lane_transactions[slot].mutation_flags = 0u;
		status = SparkKvPageCachePrepareLane(&state->kv_page_cache,lane,state->kv_lane_logical_pages + (uint64_t)slot * state->pages_per_sequence,state->pages_per_sequence,&state->kv_lane_page_count[slot]);
		if ( status != SPARK_STATUS_OK )
			SPARK_FAIL(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(LaneRecordsCommit)(SPARK_FAMILY(ModuleState) *state,const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverCacheLane *lane;
	SparkKvLaneTransaction *owner;
	uint32_t lane_index;
	SparkStatus status;
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
	{
		lane = &request->cache_lanes[lane_index];
		owner = &state->kv_lane_transactions[lane->resident_sequence_slot];
		owner->lane = *lane;
		owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
		status = SparkKvPageCacheBeginLaneTransaction(&state->kv_page_cache,lane,&state->kv_lane_mutable_page[lane->resident_sequence_slot],&owner->mutation_flags);
		if ( status != SPARK_STATUS_OK )
		{
			(void)SPARK_FAMILY(LaneRecordsRollback)(state,request,lane_index);
			SPARK_FAIL(status);
		}
		owner->phase = SPARK_KV_LANE_TRANSACTION_COMMITTED;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(LaneRecordsRelease)(SPARK_FAMILY(ModuleState) *state,const SparkModelDriverAdmissionRequest *request)
{
	const SparkModelDriverCacheLane *lane;
	uint32_t lane_index;
	SparkStatus status;
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
	{
		lane = &request->cache_lanes[lane_index];
		status = SparkKvPageCacheReleaseLane(&state->kv_page_cache,lane->resident_sequence_slot,lane->sequence_id);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(LaneRecordFinish)(SPARK_FAMILY(ModuleState) *state,uint32_t resident,uint64_t sequence_id,SparkStatus execution_status)
{
	SparkKvLaneTransaction *owner;
	SparkStatus status;
	owner = &state->kv_lane_transactions[resident];
	if ( owner->phase != SPARK_KV_LANE_TRANSACTION_COMMITTED || owner->lane.sequence_id != sequence_id )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	status = execution_status == SPARK_STATUS_OK ? SparkKvPageCacheCompleteLane(&state->kv_page_cache,&owner->lane) : SparkKvPageCacheRollbackLaneTransaction(&state->kv_page_cache,&owner->lane,owner->mutation_flags);
	owner->phase = SPARK_KV_LANE_TRANSACTION_EMPTY;
	SPARK_RETURN(status);
}

static SparkStatus SPARK_FAMILY(AdmissionPredicate)(void *context,const SparkModelDriverAdmissionRequest *request,SparkModelDriverAdmissionDecision *decision)
{
	SPARK_FAMILY(ModuleState) *state;
	uint32_t lane_index,admitting;
	SparkStatus status;
	state = (SPARK_FAMILY(ModuleState) *)context;
	if ( state == 0 || request == 0 || decision == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane_index=0u; lane_index<request->cache_lane_count; lane_index++)
		if ( request->cache_lanes[lane_index].resident_sequence_slot >= state->resident_sequence_capacity )
			SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	admitting = (request->frame_flags & SPARK_MODEL_DRIVER_FRAME_FLAG_CACHE_RELEASE) == 0u ? 1u : 0u;
	status = admitting == 0u ? SPARK_FAMILY(LaneRecordsRelease)(state,request) : SPARK_STATUS_OK;
	if ( status == SPARK_STATUS_OK && admitting != 0u && (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_PREPARE) != 0u )
		status = SPARK_FAMILY(LaneRecordsPrepare)(state,request);
	if ( status == SPARK_STATUS_OK && admitting != 0u && (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_COMMIT) != 0u )
		status = SPARK_FAMILY(LaneRecordsCommit)(state,request);
	if ( status == SPARK_STATUS_OK && admitting != 0u && (request->admission_flags & SPARK_MODEL_DRIVER_ADMISSION_FLAG_CACHE_ABORT) != 0u )
		status = SPARK_FAMILY(LaneRecordsRollback)(state,request,request->cache_lane_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	decision->accepted = 1u;
	decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	return(SPARK_STATUS_OK);
}
