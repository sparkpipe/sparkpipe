#include <stdint.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_stage_module_common.h"

#define SPARK_DSV41_FLASH_MODULE_TAG "dsv41_flash_stage"

SparkStatus SparkDsv41FlashCudaContextEnsure(void)
{
	cudaError_t error;
	error = cudaFree(0);
	if ( error != cudaSuccess )
		return(SparkStageModuleCudaStatus(SPARK_DSV41_FLASH_MODULE_TAG,error,"context_ensure"));
	return(SPARK_STATUS_OK);
}
