#pragma once

#include <stdatomic.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#define SPARK_STAGE_MODULE_SLOT_FREE 0u
#define SPARK_STAGE_MODULE_SLOT_CLAIMED 1u

typedef SparkStatus (*SparkStageModuleClaimedIndexPrepareFunction)(
    void *prepare_context);

SparkStatus SparkStageModuleSlotClaim(
    atomic_uint *slot_states,
    uint32_t slot_count,
    uint32_t *slot_index);
SparkStatus SparkStageModuleIndexSetClaim(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count);
SparkStatus SparkStageModuleIndexClaimOrdinal(
    const atomic_uint *index_states,
    uint32_t index_capacity,
    uint32_t index,
    uint32_t *ordinal_out);

typedef struct SparkStageModuleClaimedLaneContext
{
	const atomic_uint *index_states;
	uint32_t index_capacity;
} SparkStageModuleClaimedLaneContext;

static inline SparkStatus SparkStageModuleClaimedLaneOrdinal(
	void *context,
	uint32_t lane_id,
	uint32_t *ordinal_out)
{
	const SparkStageModuleClaimedLaneContext *lanes;
	lanes = (const SparkStageModuleClaimedLaneContext *)context;
	if ( lanes == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleIndexClaimOrdinal(lanes->index_states,lanes->index_capacity,lane_id,ordinal_out));
}

SparkStatus SparkStageModuleIndexSetClaimAndPrepare(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count,
    SparkStageModuleClaimedIndexPrepareFunction prepare_function,
    void *prepare_context);
void SparkStageModuleIndexSetRelease(
    atomic_uint *index_states,
    uint32_t index_capacity,
    const uint32_t *indices,
    uint32_t index_count);
void SparkStageModuleAtomicStateArrayInitialize(
    atomic_uint *states,
    uint32_t state_count);
uint32_t SparkStageModuleSlotAvailableCount(
    const atomic_uint *slot_states,
    uint32_t slot_count);
uint32_t SparkStageModuleSlotCountFree(
    const atomic_uint *slot_states,
    uint32_t slot_count);
void SparkStageModuleSlotRelease(
    atomic_uint *slot_states,
    uint32_t slot_index);
