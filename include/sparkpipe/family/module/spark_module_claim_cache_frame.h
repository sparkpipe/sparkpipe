#pragma once

static SparkStatus SPARK_FAMILY(ClaimCacheFrame)(SPARK_FAMILY(ModuleState) *state,const SparkModelDriverFrame *frame,const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch,const uint64_t *next_positions)
{
	uint32_t lane;
	SparkStatus status;
	if ( frame->cache_lanes == 0 || frame->cache_lane_count != batch->active_sequence_count )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( (frame->flags & SPARK_MODEL_DRIVER_FRAME_FLAG_DRIVER_DISPATCH_SLOT_VALID) == 0u || frame->driver_dispatch_slot != (uint32_t)(frame->request_id % state->pipeline_slot_count) )
		SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	for (lane=0u; lane<frame->cache_lane_count; lane++)
		if ( frame->cache_lanes[lane].resident_sequence_slot != batch->row_resident_slots[lane] || frame->cache_lanes[lane].sequence_id != batch->row_sequence_ids[lane] || frame->cache_lanes[lane].sequence_position != batch->row_positions[lane] || frame->cache_lanes[lane].context_token_count != next_positions[lane] )
			SPARK_FAIL(SPARK_STATUS_VALIDATION_FAILED);
	if ( pthread_mutex_lock(&state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SparkKvLaneTransactionsClaim(&state->kv_transactions,frame);
	(void)pthread_mutex_unlock(&state->kv_mutex);
	SPARK_RETURN(status);
}
