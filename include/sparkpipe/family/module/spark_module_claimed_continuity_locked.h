#pragma once

static SparkStatus SPARK_FAMILY(ValidateRoundMajor)(const SPARK_FAMILY(ModuleState) *state,const SPARK_FAMILY(ResidentDecodeStageBatchView) *batch)
{
	uint32_t ordinals[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)];
	uint32_t counts[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)];
	uint32_t last_rows[SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT)];
	SparkRowLayoutDirectLaneContext lanes;
	SparkStatus status;
	if ( state == 0 || batch == 0 || batch->row_count < batch->active_sequence_count || state->resident_sequence_capacity > SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_ACTIVE_SEQUENCE_COUNT) )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkRowLayoutDirectLaneMapInitialize(&lanes,ordinals,state->resident_sequence_capacity,batch->row_resident_slots,batch->active_sequence_count);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	return(SparkRowLayoutValidateRoundMajor(batch->row_count,batch->active_sequence_count,batch->row_resident_slots,SparkRowLayoutDirectLaneOrdinal,&lanes,counts,last_rows));
}

#include "sparkpipe/family/module/spark_module_validate_sequence_continuity.h"

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
