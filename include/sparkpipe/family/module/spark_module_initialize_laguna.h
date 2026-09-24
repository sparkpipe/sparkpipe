#pragma once

static SparkStatus SPARK_FAMILY(AllocateSlotMetadata)(
	SPARK_FAMILY(ModuleState) *state,
	SPARK_FAMILY(ExecutionSlot) *slot)
{
	SparkStatus status;
	status = SPARK_FAMILY(AllocateBytes)(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->token_ids);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->resident_slots);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,state->execution_row_capacity,1u,sizeof(uint32_t),(void **)&slot->positions);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,state->resident_sequence_capacity,1u,sizeof(uint32_t),(void **)&slot->context_lengths);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_row_offset);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,2u,1u,sizeof(uint32_t),(void **)&slot->dense_tile_prefix);
	if ( status == SPARK_STATUS_OK ) status = SPARK_FAMILY(AllocateBytes)(state,1u,sizeof(uint32_t) * 6u,1u,&slot->kv_access_error);
	return(status);
}

static void SPARK_FAMILY(PrepareAsyncCompletion)(
	SPARK_FAMILY(ModuleState) *state,
	SparkModelDriverFrame *frame,
	const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch,
	const uint8_t *lane_bound,
	const uint64_t *lane_sequence_ids,
	const uint64_t *lane_next_positions,
	uint32_t slot_index)
{
	SPARK_FAMILY(AsyncCompletion) *async;
	uint32_t lane;
	async = &state->completions[slot_index];
	memset(async,0,sizeof(*async));
	async->state = state;
	async->completion_function = frame->completion_function;
	async->completion_context = frame->completion_context;
	async->slot_index = slot_index;
	async->lane_count = batch->active_sequence_count;
	async->row_count = batch->row_count;
	async->output_token_destination = state->owns_final_head != 0u ? (uint32_t *)frame->buffers[0].address : 0;
	for (lane=0u; lane<batch->active_sequence_count && lane<SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT); lane++)
	{
		async->lane_indices[lane] = batch->row_resident_slots[lane];
		async->lane_bound[lane] = lane_bound[lane];
		async->lane_sequence_ids[lane] = lane_sequence_ids[lane];
		async->lane_next_positions[lane] = lane_next_positions[lane];
	}
	async->completion.request_id = frame->request_id;
	async->completion.sequence_id = frame->sequence_id;
	async->completion.sequence_position = frame->sequence_position;
	async->completion.program_id = frame->program_id;
	async->completion.driver_dispatch_slot = frame->driver_dispatch_slot;
	async->completion.accepted_token_count = frame->new_token_count;
	async->completion.tokens_per_sequence = frame->tokens_per_sequence;
	async->completion.status = SPARK_STATUS_OK;
	async->completion.residency = frame->residency;
	async->completion.host_staging_bytes = (uint64_t)batch->row_count * sizeof(uint32_t) * (3u + state->owns_final_head);
	async->completion.device_memcpy_bytes = async->completion.host_staging_bytes;
}

SparkStatus SPARK_FAMILY(ResidentDecodeStageInitialize)(
	const SparkFirmwareModuleConfiguration *configuration,
	const SparkFirmwareModuleHostServices *host_services,
	void **module_state)
{
	SPARK_FAMILY(ModuleState) *state;
	SparkStatus status;
	status = SparkFirmwareModuleValidateInitialization(configuration,host_services,module_state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	state = 0;
	status = SPARK_FAMILY(InitializeState)(configuration,host_services,&state);
	if ( status != SPARK_STATUS_OK )
		return(status);
	*module_state = state;
	return(SPARK_STATUS_OK);
}
