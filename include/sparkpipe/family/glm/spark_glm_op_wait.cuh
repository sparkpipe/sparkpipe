#pragma once

__global__ void SPARK_FAMILY(OpWaitKernel)(
	volatile unsigned long long *flag,
	unsigned long long value)
{
	while (*flag < value)
		__nanosleep(100u);
}

extern "C" SparkStatus SPARK_FAMILY(LaunchOpWait)(
	cudaStream_t stream,
	void *flag_device,
	uint64_t wait_value)
{
	if ( stream == 0 || flag_device == 0 )
		return SPARK_STATUS_INVALID_ARGUMENT;
	SPARK_FAMILY(OpWaitKernel)<<<1,1,0,stream>>>(
		(volatile unsigned long long *)flag_device,
		(unsigned long long)wait_value);
	return cudaPeekAtLastError() == cudaSuccess ?
		SPARK_STATUS_OK : SPARK_STATUS_DRIVER_LOAD_ERROR;
}
