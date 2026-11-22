#ifndef SPARKPIPE_SPARK_WEIGHTD_DEVICE_H
#define SPARKPIPE_SPARK_WEIGHTD_DEVICE_H

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_gpu.h"

#define SPARK_WEIGHTD_DEVICE_READ SPARK_DEVICE_ACCESS_DEVICE_READ
#define SPARK_WEIGHTD_DEVICE_READ_WRITE (SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE)

static inline SparkStatus SparkWeightdDeviceOpen(void)
{
	const SparkStatus status = SparkDeviceInitialize(0);
	return(status == SPARK_STATUS_DUPLICATE ? SPARK_STATUS_OK : status);
}

static inline SparkDeviceMemoryHandle SparkWeightdMemoryOf(const void *handle)
{
	return((SparkDeviceMemoryHandle)(uintptr_t)handle);
}

static inline SparkStatus SparkWeightdDeviceWrite(void *device,const void *host,uint64_t bytes)
{
	SparkDeviceBuffer view;
	SparkDeviceRange range;
	SparkStatus status = SPARK_STATUS_OK;
	uint32_t adopted = 0u;
	if ( device == 0 || host == 0 )
		SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
	if ( bytes == 0u )
		return(SPARK_STATUS_OK);
	if ( SparkDeviceResolve((SparkDeviceAddress)(uintptr_t)device,bytes,&range) != SPARK_STATUS_OK )
	{
		status = SparkDeviceAdopt(&view,(SparkDeviceAddress)(uintptr_t)device,0,bytes,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,"weightd write");
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
		adopted = 1u;
	}
	if ( SparkGpuMemcpy(device,host,(size_t)bytes,SPARK_GPU_COPY_HOST_TO_DEVICE) != SPARK_GPU_SUCCESS )
		status = SPARK_STATUS_IO_ERROR;
	if ( adopted != 0u )
	{
		const SparkStatus freed = SparkDeviceFree(&view);
		if ( status == SPARK_STATUS_OK )
			status = freed;
	}
	SPARK_RETURN(status);
}

#endif
