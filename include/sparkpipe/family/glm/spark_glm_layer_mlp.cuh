#pragma once

static int32_t SPARK_FAMILY(RunLayerMlp)(const SPARK_FAMILY(CudaWave) *wave,uint32_t local_layer)
{
	int32_t status = SPARK_FAMILY(RunLayerMlpRoute)(wave,local_layer);
	if ( status != LM_LAUNCH_OK )
		return(status);
	return(SPARK_FAMILY(RunLayerMlpExperts)(wave,local_layer));
}
