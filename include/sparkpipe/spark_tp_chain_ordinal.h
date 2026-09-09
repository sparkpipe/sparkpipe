#pragma once

#include "sparkpipe/spark_status.h"

// The transport stores its phase/status in the low 16 lifecycle bits.
#define SPARK_TP_CHAIN_MAX_GENERATION (UINT64_MAX >> 16u)

// Admission must claim chain_id % lane_count identically on every rank.
// Each live lane owns credits_per_lane credits until its chain completes.
// Chain IDs increase within a lane; capacity is fixed for the session.
static inline SparkStatus SparkTpChainOrdinal(uint64_t chain_id,uint32_t lane_count,uint32_t credits_per_lane,uint32_t operation_capacity,uint32_t operation_index,uint64_t *ordinal)
{
	uint64_t epoch,generation,credits,lane;
	if ( ordinal == 0 || chain_id == 0u || lane_count == 0u || credits_per_lane == 0u || operation_capacity == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( operation_index >= operation_capacity )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	epoch = (chain_id / lane_count);
	lane = (chain_id % lane_count);
	credits = ((uint64_t)lane_count * credits_per_lane);
	if ( epoch > ((SPARK_TP_CHAIN_MAX_GENERATION - 1u - operation_index) / operation_capacity) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	generation = ((epoch * operation_capacity) + operation_index);
	if ( credits > (UINT64_MAX / (generation + 1u)) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	*ordinal = ((generation * credits) + (lane * credits_per_lane) + (operation_index % credits_per_lane));
	return(SPARK_STATUS_OK);
}
