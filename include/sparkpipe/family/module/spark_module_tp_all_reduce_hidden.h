#pragma once

static SparkStatus SPARK_FAMILY(ModuleTpAllReduceHidden)(SPARK_FAMILY(ModuleState) *state, SPARK_FAMILY(ModuleSlot) *slot, void *device_bf16, uint32_t rows)
{
	if ( state->tp_degree == 1u || state->tp_standalone != 0u )
		return(SPARK_STATUS_OK);
	return(SPARK_FAMILY(ModuleTpSubmitOrdered)(state,device_bf16,rows,slot,0u));
}
