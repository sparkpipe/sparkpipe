#pragma once

#include <stdint.h>

#include "sparkpipe/spark_module_abi.h"

/* Position zero is the slot-reuse boundary even when sequence IDs repeat. */
static inline SparkStatus SparkDsv4AdvanceLaneContinuity(uint64_t sequence,uint64_t position,uint64_t *lane_sequence,uint64_t *lane_next_position,uint8_t *lane_touched,uint8_t *lane_requires_reset)
{
	if ( lane_sequence == 0 || lane_next_position == 0 || lane_touched == 0 || lane_requires_reset == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( position == 0u )
	{
		if ( *lane_touched != 0u )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		*lane_sequence = sequence;
		*lane_next_position = 1u;
		*lane_requires_reset = 1u;
	}
	else
	{
		if ( *lane_sequence != sequence || *lane_next_position != position )
			return(SPARK_STATUS_INVALID_ARGUMENT);
		*lane_next_position = position + 1u;
	}
	*lane_touched = 1u;
	return(SPARK_STATUS_OK);
}
