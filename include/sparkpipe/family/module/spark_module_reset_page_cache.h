#pragma once

static SparkStatus SPARK_FAMILY(ResetPageCacheClaimed)(SPARK_FAMILY(ModuleState) *state)
{
	cudaError_t drain;
	uint32_t lane;
	SparkStatus status;
	drain = cudaStreamSynchronize((cudaStream_t)state->execution_stream);
	if ( drain != cudaSuccess )
	{
		(void)SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),drain,"reset_stream_drain");
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	status = SparkKvPageCacheReleaseAll(&state->kv_page_cache);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (lane=0u; lane<state->resident_sequence_capacity; lane++)
	{
		atomic_store_explicit(&state->lane_bound[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_sequence_ids[lane],0u,memory_order_release);
		atomic_store_explicit(&state->lane_next_positions[lane],0u,memory_order_release);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SPARK_FAMILY(ResetPageCache)(
	SPARK_FAMILY(ModuleState) *state,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	uint32_t slots[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_PIPELINE_SLOT_COUNT)];
	uint32_t lanes[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)];
	uint32_t index;
	SparkStatus status;
	SparkModelDriverInitializeAdmissionDecision(decision);
	if ( SparkModelDriverAdmissionRequestIsValid(request) == 0u )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<state->pipeline_slot_count; index++)
		slots[index] = index;
	for (index=0u; index<state->resident_sequence_capacity; index++)
		lanes[index] = index;
	status = SparkStageModuleIndexSetClaim(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status != SPARK_STATUS_OK )
		return(status);
	status = SparkStageModuleIndexSetClaim(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	if ( status == SPARK_STATUS_OK )
	{
		status = SPARK_FAMILY(ResetPageCacheClaimed)(state);
		if ( status == SPARK_STATUS_IO_ERROR )
		{
			fprintf(stderr,"%s reset: execution stream failed; lanes and slots stay claimed\n",SPARK_FAMILY_CONST(MODULE_TAG));
			SPARK_RETURN(status);
		}
		SparkStageModuleIndexSetRelease(state->lane_states,state->resident_sequence_capacity,lanes,state->resident_sequence_capacity);
	}
	SparkStageModuleIndexSetRelease(state->slot_states,state->pipeline_slot_count,slots,state->pipeline_slot_count);
	if ( status == SPARK_STATUS_OK )
	{
		decision->accepted = 1u;
		decision->rejection_reason = SPARK_MODEL_DRIVER_ADMISSION_ACCEPTED;
	}
	return(status);
}
