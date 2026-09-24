#pragma once

extern "C" int32_t SPARK_FAMILY(LaunchCudaLayerMlpExperts)(const SPARK_FAMILY(CudaWave) *wave,uint32_t local_layer)
{
	int32_t status;
	status = SPARK_FAMILY(ValidateWaveShape)(wave);
	if ( status != LM_LAUNCH_OK || local_layer >= wave->layer_count )
		return(LM_LAUNCH_ERR_SHAPE);
	return(SPARK_FAMILY(RunLayerMlpExperts)(wave,local_layer));
}
