#pragma once

static SparkStatus SPARK_FAMILY(ServingValidateConfiguration)(
	const SparkModelServingAdapterConfiguration *configuration)
{
	SparkStatus status;
	if ( configuration == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( configuration->abi_version != SPARK_MODEL_SERVING_ADAPTER_ABI_VERSION || configuration->descriptor_bytes != SPARK_MODEL_SERVING_ADAPTER_CONFIGURATION_BYTES )
		return(SPARK_STATUS_ABI_MISMATCH);
	status = SparkModelServingAdapterValidateRuntimeLimits(&SPARK_FAMILY(ServingDescriptor),&configuration->runtime_limits);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( configuration->stage_index >= SPARK_FAMILY_CONST(SERVING_STAGE_COUNT) || configuration->runtime_root == 0 || configuration->node_id == 0 || configuration->node_target == 0 || configuration->adapter_configuration_path == 0 || configuration->driver_shared_object_path == 0 || configuration->driver_program_name == 0 || strcmp(configuration->driver_program_name,SPARK_FAMILY_CONST(SERVING_PROGRAM_NAME)) != 0 || configuration->execution_stream == 0 || configuration->completion_function == 0 )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	return(SPARK_STATUS_OK);
}
