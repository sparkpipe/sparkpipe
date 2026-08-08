#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

typedef SparkStatus (*SparkRowLayoutLaneOrdinalFunction)(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out);

typedef struct SparkRowLayoutDenseLaneContext
{
	uint32_t lane_count;
} SparkRowLayoutDenseLaneContext;

typedef struct SparkRowLayoutDirectLaneContext
{
	const uint32_t *ordinals;
	uint32_t capacity;
} SparkRowLayoutDirectLaneContext;

static inline SparkStatus SparkRowLayoutDenseLaneOrdinal(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out)
{
	const SparkRowLayoutDenseLaneContext *dense;
	dense = (const SparkRowLayoutDenseLaneContext *)context;
	if ( dense == 0 || ordinal_out == 0 || lane_id >= dense->lane_count )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*ordinal_out = lane_id;
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkRowLayoutDirectLaneMapInitialize(
	SparkRowLayoutDirectLaneContext *context,
	uint32_t *ordinals,
	uint32_t capacity,
	const uint32_t *lane_ids,
	uint32_t lane_count)
{
	uint32_t index,lane;
	if ( context == 0 || ordinals == 0 || capacity == 0u || lane_ids == 0 || lane_count == 0u || lane_count > capacity )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (index=0u; index<capacity; index++)
		ordinals[index] = UINT32_MAX;
	for (lane=0u; lane<lane_count; lane++)
	{
		index = lane_ids[lane];
		if ( index >= capacity || ordinals[index] != UINT32_MAX )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		ordinals[index] = lane;
	}
	context->ordinals = ordinals;
	context->capacity = capacity;
	return(SPARK_STATUS_OK);
}

static inline SparkStatus SparkRowLayoutDirectLaneOrdinal(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out)
{
	const SparkRowLayoutDirectLaneContext *direct;
	direct = (const SparkRowLayoutDirectLaneContext *)context;
	if ( direct == 0 || ordinal_out == 0 || lane_id >= direct->capacity || direct->ordinals[lane_id] == UINT32_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*ordinal_out = direct->ordinals[lane_id];
	return(SPARK_STATUS_OK);
}

/*
 * Validate one round-major frame in O(lanes + rows). The ordinal callback
 * must provide a stable O(1) mapping for every live lane. A valid frame is
 * ordered lexicographically by (zero-based lane occurrence, lane ordinal).
 */
static inline SparkStatus SparkRowLayoutValidateRoundMajor(
	uint32_t row_count,
	uint32_t lane_count,
	const uint32_t *row_lane_ids,
	SparkRowLayoutLaneOrdinalFunction ordinal_function,
	void *ordinal_context,
	uint32_t *occurrence_counts,
	uint32_t *last_rows)
{
	uint32_t lane,ordinal,previous_ordinal,previous_wave,row,wave;
	SparkStatus status;
	if ( row_count < lane_count || lane_count == 0u || row_lane_ids == 0 || ordinal_function == 0 || occurrence_counts == 0 || last_rows == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	for (lane=0u; lane<lane_count; lane++)
	{
		occurrence_counts[lane] = 0u;
		last_rows[lane] = UINT32_MAX;
		status = ordinal_function(ordinal_context,row_lane_ids[lane],&ordinal);
		if ( status != SPARK_STATUS_OK || ordinal != lane )
			return(SPARK_STATUS_INVALID_ARGUMENT);
	}
	previous_wave = 0u;
	previous_ordinal = 0u;
	for (row=0u; row<row_count; row++)
	{
		status = ordinal_function(ordinal_context,row_lane_ids[row],&ordinal);
		if ( status != SPARK_STATUS_OK || ordinal >= lane_count )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		wave = occurrence_counts[ordinal]++;
		if ( row != 0u && (wave < previous_wave || (wave == previous_wave && ordinal <= previous_ordinal)) )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		last_rows[ordinal] = row;
		previous_wave = wave;
		previous_ordinal = ordinal;
	}
	return(SPARK_STATUS_OK);
}

/* Return the next already-validated round-major wave without lane scans. */
static inline uint32_t SparkRowLayoutRoundMajorWaveRowCount(
	uint32_t first_row,
	uint32_t row_count,
	const uint32_t *row_lane_ids,
	SparkRowLayoutLaneOrdinalFunction ordinal_function,
	void *ordinal_context)
{
	uint32_t count,current,next;
	if ( first_row >= row_count || row_lane_ids == 0 || ordinal_function == 0 || ordinal_function(ordinal_context,row_lane_ids[first_row],&current) != SPARK_STATUS_OK )
		return(0u);
	count = 1u;
	while ( first_row + count < row_count )
	{
		if ( ordinal_function(ordinal_context,row_lane_ids[first_row + count],&next) != SPARK_STATUS_OK || next <= current )
			break;
		current = next;
		count++;
	}
	return(count);
}
