#pragma once

static SparkStatus SPARK_FAMILY(ServingValidateRowOrder)(
	const SPARK_FAMILY(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint8_t seen[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint64_t last_position[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint32_t lane,row,wave,maximum;
	uint32_t counts[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	for (row=0u; row<submission->row_count; row++)
	{
		lane = submission->row_lane_indices[row];
		if ( lane >= submission->active_sequence_count || submission->row_positions[row] >= state->node_context.max_sequence_positions )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		if ( seen[lane] != 0u && submission->row_positions[row] != last_position[lane] + 1u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		seen[lane] = 1u;
		last_position[lane] = submission->row_positions[row];
		counts[lane]++;
	}
	if ( submission->work_kind == SPARK_MODEL_SERVING_WORK_KIND_DECODE )
		return(submission->row_count == submission->active_sequence_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
	maximum = 0u;
	for (lane=0u; lane<submission->active_sequence_count; lane++)
		if ( counts[lane] > maximum )
			maximum = counts[lane];
	row = 0u;
	for (wave=0u; wave<maximum; wave++)
		for (lane=0u; lane<submission->active_sequence_count; lane++)
			if ( counts[lane] > wave && (row >= submission->row_count || submission->row_lane_indices[row++] != lane) )
				return(SPARK_STATUS_INVALID_ARGUMENT);
	return(row == submission->row_count ? SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT);
}

static SparkStatus SPARK_FAMILY(ServingValidateBoundaries)(
	const SPARK_FAMILY(ServingState) *state,
	const SparkModelServingSubmission *submission)
{
	uint64_t boundary_bytes;
	boundary_bytes = (uint64_t)submission->row_count * SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_COUNT) * SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_BOUNDARY_ELEMENT_BYTES);
	if ( submission->hidden_input_address != 0 || submission->hidden_input_bytes != 0u || submission->hidden_output_address != 0 || submission->hidden_output_bytes != 0u || submission->boundary_sideband_input_address != 0 || submission->boundary_sideband_input_bytes != 0u || submission->boundary_sideband_output_address != 0 || submission->boundary_sideband_output_bytes != 0u )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	(void)boundary_bytes;
	(void)state;
	return(SPARK_STATUS_OK);
}
