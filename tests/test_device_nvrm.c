#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

int main(void)
{
	SparkDeviceConfiguration configuration;
	SparkDeviceBuffer buffer;
	SparkDeviceQueue *queue = 0;
	SparkStatus status;
	if ( access("/dev/nvidiactl",F_OK) == 0 )
	{
		printf("SKIP test_device_nvrm (the open kernel module's control device is present; the GPU gates cover this host)\n");
		return(0);
	}
	memset(&configuration,0,sizeof(configuration));
	memset(&buffer,0,sizeof(buffer));
	status = SparkDeviceInitialize(&configuration);
	CHECK(status != SPARK_STATUS_OK,"initialize refuses without a device");
	CHECK(SparkDeviceLastError()->status == status && strcmp(SparkDeviceLastError()->operation,"initialize") == 0,
		"the refusal is recorded against initialize");
	CHECK(strstr(SparkDeviceLastError()->text,"/dev/nvidiactl") != 0,"the refusal names the open kernel module's control device it could not open");
	CHECK(SparkDeviceInitialize(&configuration) == status,"a second initialize refuses the same way rather than claiming the device");
	CHECK(SparkDeviceAllocate(&buffer,SPARK_DEVICE_SPACE_DEVICE_PRIVATE,4096u,"after refusal") == SPARK_STATUS_INVALID_ARGUMENT &&
		strcmp(SparkDeviceLastError()->text,"device is not initialized") == 0,"the device stays closed after the refusal");
	CHECK(SparkDeviceQueueCreate(&queue,"after refusal") == SPARK_STATUS_INVALID_ARGUMENT && queue == 0,"no queue opens on a closed device");
	CHECK(SparkDeviceShutdown() == SPARK_STATUS_INVALID_ARGUMENT,"shutdown refuses a device that never opened");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the nvrm backend refuses to open without the open kernel module's device, naming the device\n");
	return(0);
}
