#pragma once

static SparkStatus SPARK_FAMILY(ModuleTpReduceU64Max)(SPARK_FAMILY(ModuleState) *state, SPARK_FAMILY(ModuleSlot) *slot, uint64_t *device_u64, uint32_t count)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SPARK_FAMILY(ModuleTpSubmitOrdered)(state,device_u64,count,slot,1u));
}

static void SPARK_FAMILY(AdmissionCost)(
	void *context,
	const SparkModelDriverAdmissionRequest *request,
	SparkModelDriverAdmissionDecision *decision)
{
	SPARK_FAMILY(ModuleState) *state = (SPARK_FAMILY(ModuleState) *)context;
	decision->host_staging_bytes = (uint64_t)request->new_token_count *
		(sizeof(uint32_t) *
			 (uint64_t)(state->owns_embedding + state->owns_final_head + 3u) +
		 sizeof(uint64_t));
	decision->device_memcpy_bytes = decision->host_staging_bytes;
}
