#pragma once

static uint32_t SPARK_FAMILY(PackRangesOverlap)(const SPARK_FAMILY(PackRange) *left,const SPARK_FAMILY(PackRange) *right)
{
	return(left->bytes != 0u && right->bytes != 0u && left->offset < right->offset + right->bytes && right->offset < left->offset + left->bytes ? 1u : 0u);
}

static SparkStatus SPARK_FAMILY(AllocateRows)(
	SPARK_FAMILY(ModuleState) *state,
	uint64_t rows,
	uint64_t columns,
	void **pointer)
{
	return(SPARK_FAMILY(AllocateBytes)(state,rows,columns,sizeof(uint16_t),pointer));
}

static SparkStatus SPARK_FAMILY(EnqueueAsyncCompletion)(
	SPARK_FAMILY(ModuleState) *state,
	SPARK_FAMILY(ExecutionSlot) *slot,
	uint32_t slot_index)
{
	cudaStream_t stream;
	cudaError_t error;
	stream = (cudaStream_t)slot->stream;
	error = cudaMemcpyAsync(slot->host_kv_access_error,slot->kv_access_error,SPARK_FAMILY_CONST(KV_ACCESS_ERROR_WORD_COUNT) * sizeof(uint32_t),cudaMemcpyDeviceToHost,stream);
	if ( error == cudaSuccess )
		error = cudaLaunchHostFunc(stream,SPARK_FAMILY(CompleteAsync),&state->completions[slot_index]);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"async_completion"));
}
