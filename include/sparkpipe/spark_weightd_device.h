#ifndef SPARKPIPE_SPARK_WEIGHTD_DEVICE_H
#define SPARKPIPE_SPARK_WEIGHTD_DEVICE_H

#include "sparkpipe/spark_device.h"

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

#endif
