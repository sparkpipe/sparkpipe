#pragma once

#include "sparkpipe/family/module/spark_module_validate_round_major.h"

static SparkStatus SPARK_FAMILY(ValidateSequenceContinuity)(const SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch,uint8_t *bound,uint64_t *sequence_ids,uint64_t *next_positions)
{
	uint8_t touched[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)] = {0u};
	uint64_t position,sequence;
	uint32_t lane,row,slot;
	SparkStatus status;
	status = SPARK_FAMILY(LoadSequenceContinuity)(state,batch,bound,sequence_ids,next_positions);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	for (row=0u; row<batch->row_count; row++)
	{
		slot = batch->row_resident_slots[row];
		status = SparkStageModuleIndexClaimOrdinal(state->lane_states,state->resident_sequence_capacity,slot,&lane);
		position = batch->row_positions[row];
		sequence = batch->row_sequence_ids[row];
		if ( status != SPARK_STATUS_OK || lane >= batch->active_sequence_count || batch->row_resident_slots[lane] != slot || position >= state->max_sequence_positions )
			SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
		if ( position == 0u )
		{
			if ( touched[lane] != 0u )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			bound[lane] = 1u;
			sequence_ids[lane] = sequence;
			next_positions[lane] = 1u;
		}
		else
		{
			if ( bound[lane] == 0u || sequence_ids[lane] != sequence || next_positions[lane] != position )
				SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
			next_positions[lane] = position + 1u;
		}
		touched[lane] = 1u;
	}
	return(SPARK_STATUS_OK);
}

typedef struct SPARK_FAMILY(ClaimedContinuityContext)
{
	SPARK_FAMILY(ModuleState) *state;
	const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch;
	uint8_t *bound;
	uint64_t *sequence_ids;
	uint64_t *next_positions;
} SPARK_FAMILY(ClaimedContinuityContext);

static SparkStatus SPARK_FAMILY(PrepareClaimedContinuity)(void *prepare_context)
{
	SPARK_FAMILY(ClaimedContinuityContext) *context;
	SparkStatus status;
	context = (SPARK_FAMILY(ClaimedContinuityContext) *)prepare_context;
	if ( pthread_mutex_lock(&context->state->kv_mutex) != 0 )
		SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
	status = SPARK_FAMILY(ValidateSequenceContinuity)(context->state,context->batch,context->bound,context->sequence_ids,context->next_positions);
	(void)pthread_mutex_unlock(&context->state->kv_mutex);
	SPARK_RETURN(status);
}
