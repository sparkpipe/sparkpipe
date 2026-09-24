#pragma once

static SparkStatus SPARK_FAMILY(ModuleKvPrepareFrame)(SPARK_FAMILY(ModuleState) *state, SPARK_FAMILY(ModuleSlot) *slot, SPARK_FAMILY(ResidentDecodeStageFrameContext) *context, SPARK_FAMILY(KvBlockTableView) *table, uint32_t rows)
{
	LmKvFrameSlot frame_slot;
	LmKvFrameTable frame_table;
	SparkStatus status;
	if ( context == 0 || context->decode_batch == 0 || table == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	frame_slot.cuda_stream = slot->cuda_stream;
	frame_slot.host_row_lane_indices = slot->host_row_lane_indices;
	frame_slot.host_row_positions = slot->host_row_positions;
	frame_slot.host_context_lengths = slot->host_context_lengths;
	frame_slot.host_slot_mapping = slot->host_slot_mapping;
	frame_slot.slot_mapping = slot->slot_mapping;
	frame_table.lane_count = table->lane_count;
	frame_table.lane_stride = table->lane_stride;
	frame_table.host_physical_block_indices = table->host_physical_block_indices;
	frame_table.host_lane_physical_block_counts = table->host_lane_physical_block_counts;
	frame_table.physical_block_indices = table->physical_block_indices;
	frame_table.lane_physical_block_counts = table->lane_physical_block_counts;
	status = LmKvFramePrepareFrame(&state->kv,&frame_slot,context->decode_batch->row_sequence_ids,&frame_table,rows);
	table->physical_block_indices = frame_table.physical_block_indices;
	table->lane_physical_block_counts = frame_table.lane_physical_block_counts;
	return(status);
}

static void SPARK_FAMILY(ModuleKvMarkWritten)(SPARK_FAMILY(ModuleState) *state, SPARK_FAMILY(ModuleSlot) *slot, uint32_t rows)
{
	LmKvFrameSlot frame_slot;
	frame_slot.cuda_stream = slot->cuda_stream;
	frame_slot.host_row_lane_indices = slot->host_row_lane_indices;
	frame_slot.host_row_positions = slot->host_row_positions;
	frame_slot.host_context_lengths = slot->host_context_lengths;
	frame_slot.host_slot_mapping = slot->host_slot_mapping;
	frame_slot.slot_mapping = slot->slot_mapping;
	LmKvFrameMarkWritten(&state->kv,&frame_slot,rows);
}
