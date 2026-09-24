#pragma once

static int32_t SPARK_FAMILY(RunLayers)(const SPARK_FAMILY(CudaWave) *wave)
{
	uint32_t local;
	int32_t status;
	for (local=0u; local<wave->layer_count; local++)
	{
		status = SPARK_FAMILY(RunLayerAttention)(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
		status = SPARK_FAMILY(RunLayerMlp)(wave,local);
		if ( status != LM_LAUNCH_OK )
			return(status);
	}
	return(LM_LAUNCH_OK);
}

extern "C" int32_t SPARK_FAMILY(LaunchCudaWaveBegin)(const SPARK_FAMILY(CudaWave) *wave)
{
	int32_t status;
	status = SPARK_FAMILY(ValidateWaveShape)(wave);
	if ( status == LM_LAUNCH_OK )
		status = SPARK_FAMILY(StageWaveMetadata)(wave);
	if ( status == LM_LAUNCH_OK )
		status = SPARK_FAMILY(StageWaveBoundary)(wave);
	return(status);
}

extern "C" int32_t SPARK_FAMILY(LaunchCudaLayerAttention)(const SPARK_FAMILY(CudaWave) *wave,uint32_t local_layer)
{
	int32_t status;
	status = SPARK_FAMILY(ValidateWaveShape)(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SPARK_FAMILY(RunLayerAttention)(wave,local_layer));
}

extern "C" int32_t SPARK_FAMILY(LaunchCudaLayerMlp)(const SPARK_FAMILY(CudaWave) *wave,uint32_t local_layer)
{
	int32_t status;
	status = SPARK_FAMILY(ValidateWaveShape)(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SPARK_FAMILY(RunLayerMlp)(wave,local_layer));
}

extern "C" int32_t SPARK_FAMILY(LaunchCudaWaveHead)(const SPARK_FAMILY(CudaWave) *wave)
{
	int32_t status;
	status = SPARK_FAMILY(ValidateWaveShape)(wave);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SPARK_FAMILY(RunHead)(wave));
}

extern "C" int32_t SPARK_FAMILY(LaunchCudaWave)(const SPARK_FAMILY(CudaWave) *wave)
{
	int32_t status;
	status = SPARK_FAMILY(LaunchCudaWaveBegin)(wave);
	if ( status == LM_LAUNCH_OK )
		status = SPARK_FAMILY(RunLayers)(wave);
	if ( status == LM_LAUNCH_OK )
		status = SPARK_FAMILY(RunHead)(wave);
	return(status);
}

extern "C" int32_t SPARK_FAMILY(ConfigureCudaModule)(uint32_t *multiprocessor_count)
{
	cudaDeviceProp properties;
	int32_t device;
	cudaError_t error;
	if ( multiprocessor_count == 0 )
		return(LM_LAUNCH_ERR_SHAPE);
	error = cudaGetDevice(&device);
	if ( error == cudaSuccess )
		error = cudaGetDeviceProperties(&properties,device);
	if ( error != cudaSuccess || properties.major != 12 || properties.minor != 1 || properties.multiProcessorCount <= 0 )
		return(LM_LAUNCH_ERR_LAUNCH);
	*multiprocessor_count = (uint32_t)properties.multiProcessorCount;
	return(LM_LAUNCH_OK);
}
