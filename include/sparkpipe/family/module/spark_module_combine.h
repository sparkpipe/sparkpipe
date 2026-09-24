#pragma once

static SparkStatus SPARK_FAMILY(ModuleCombineBf16)(
	void *combine_context,
	void *destination_device,
	const void *source_device,
	uint32_t active_sequence_count,
	uint32_t hidden_dimension,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SPARK_FAMILY(LaunchAccumAdd)((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_sum"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineU64Max)(
	void *combine_context,
	uint64_t *destination_device,
	const uint64_t *source_device,
	uint32_t element_count,
	void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SPARK_FAMILY(LaunchAccumU64Max)((cudaStream_t)cuda_stream,destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_max_u64"));
}
