#pragma once

static SparkStatus SPARK_FAMILY(PageCopy)(void *context,uint32_t direction,uintptr_t device_address,void *host_address,uint64_t bytes)
{
	SPARK_FAMILY(ModuleState) *state;
	cudaError_t error;
	state = (SPARK_FAMILY(ModuleState) *)context;
	if ( state == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( direction == SPARK_KV_PAGE_STORE_COPY_DEVICE_TO_HOST )
		error = cudaMemcpy(host_address,(const void *)device_address,(size_t)bytes,cudaMemcpyDeviceToHost);
	else if ( direction == SPARK_KV_PAGE_STORE_COPY_HOST_TO_DEVICE )
		error = cudaMemcpy((void *)device_address,host_address,(size_t)bytes,cudaMemcpyHostToDevice);
	else
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	return(SparkStageModuleCudaStatus(SPARK_FAMILY_CONST(MODULE_TAG),error,"kv_page_copy"));
}
