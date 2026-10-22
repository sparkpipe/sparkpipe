#include "sparkpipe/spark_device.h"

SparkStatus DeviceInstanceAllocate(SparkDeviceBuffer *buffer)
{
	return(SparkDeviceAllocate(buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"instance module"));
}

SparkStatus DeviceInstanceFree(SparkDeviceBuffer *buffer)
{
	return(SparkDeviceFree(buffer));
}

const SparkDeviceProperties *DeviceInstanceProperties(void)
{
	return(SparkDeviceGetProperties());
}
