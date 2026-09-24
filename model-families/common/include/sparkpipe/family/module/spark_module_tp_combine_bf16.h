#pragma once

static SparkStatus SPARK_FAMILY(ModuleTpCombineBf16)(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	(void)combine_context;
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),SPARK_FAMILY(LaunchTpCombineAdd)((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension),"tp_combine"));
}
