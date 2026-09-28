#pragma once

static SparkStatus SPARK_FAMILY(AllocateSlotHost)(SPARK_FAMILY(ExecutionSlot) *slot)
{
	uint32_t *cursor;
	uint64_t rows,words,bytes;
	cudaError_t error;
	if ( slot == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	rows = SPARK_FAMILY_CONST(RESIDENT_DECODE_STAGE_MAX_INPUT_ROW_COUNT);
	words = (rows * 4u) + SPARK_FAMILY_CONST(KV_ACCESS_ERROR_WORD_COUNT);
	bytes = words * sizeof(uint32_t);
	error = cudaHostAlloc(&slot->host_staging,bytes,cudaHostAllocPortable);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"host_staging"));
	memset(slot->host_staging,0,bytes);
	cursor = (uint32_t *)slot->host_staging;
	slot->host_token_ids = cursor;
	cursor += rows;
	slot->host_resident_slots = cursor;
	cursor += rows;
	slot->host_positions = cursor;
	cursor += rows;
	slot->host_output_token_ids = cursor;
	cursor += rows;
	slot->host_kv_access_error = cursor;
	return(SPARK_STATUS_OK);
}
