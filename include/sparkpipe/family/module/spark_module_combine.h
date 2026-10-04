#pragma once

static SparkStatus SPARK_FAMILY(ModuleCombineFusedBf16)(void *combine_context, void *destination_device, const void *const *source_devices, uint32_t source_count, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchSumRanksF32((cudaStream_t)cuda_stream,destination_device,source_devices,source_count,(uint32_t)((uint64_t)active_sequence_count * hidden_dimension));
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_fused"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineF32Seed)(void *combine_context, void *destination_f32_device, const void *source_a_bf16_device, const void *source_b_bf16_device, uint32_t element_count, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchSeedF32((cudaStream_t)cuda_stream,(float *)destination_f32_device,source_a_bf16_device,source_b_bf16_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_f32_seed"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineF32Add)(void *combine_context, void *destination_f32_device, const void *source_bf16_device, uint32_t element_count, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchAddF32((cudaStream_t)cuda_stream,(float *)destination_f32_device,source_bf16_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_f32_add"));
}

static SparkStatus SPARK_FAMILY(ModuleRoundF32)(void *combine_context, void *destination_bf16_device, const void *source_f32_device, uint32_t element_count, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchRoundF32((cudaStream_t)cuda_stream,destination_bf16_device,(const float *)source_f32_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_f32_round"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineBf16)(void *combine_context, void *destination_device, const void *source_device, uint32_t active_sequence_count, uint32_t hidden_dimension, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchAccumAdd((cudaStream_t)cuda_stream,destination_device,source_device,active_sequence_count,hidden_dimension);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_sum"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineU64Max)(void *combine_context, uint64_t *destination_device, const uint64_t *source_device, uint32_t element_count, void *cuda_stream)
{
	cudaError_t error;
	(void)combine_context;
	error = SparkTpLaunchAccumU64Max((cudaStream_t)cuda_stream,destination_device,source_device,element_count);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_reduce_max_u64"));
}

static SparkStatus SPARK_FAMILY(ModuleCombineGatherBf16)(void *combine_context, void *destination_device, const void *const *source_devices, uint32_t source_count, uint32_t active_sequence_count, uint32_t row_elements, void *cuda_stream)
{
	cudaError_t error;
	uint64_t elements = (uint64_t)active_sequence_count * row_elements;
	(void)combine_context;
	if ( elements == 0u || elements > UINT32_MAX )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	error = SparkTpLaunchGatherRanks((cudaStream_t)cuda_stream,destination_device,source_devices,source_count,(uint32_t)elements);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"tp_all_gather"));
}

static inline void SPARK_FAMILY(ModuleRegisterCombines)(SparkTpDeviceCollectiveConfig *configuration)
{
	configuration->combine_fused_bf16_function = SPARK_FAMILY(ModuleCombineFusedBf16);
	configuration->combine_f32_seed_function = SPARK_FAMILY(ModuleCombineF32Seed);
	configuration->combine_f32_add_function = SPARK_FAMILY(ModuleCombineF32Add);
	configuration->round_f32_function = SPARK_FAMILY(ModuleRoundF32);
	configuration->combine_bf16_function = SPARK_FAMILY(ModuleCombineBf16);
	configuration->combine_u64_max_function = SPARK_FAMILY(ModuleCombineU64Max);
	configuration->combine_gather_bf16_function = SPARK_FAMILY(ModuleCombineGatherBf16);
}
