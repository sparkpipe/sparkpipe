#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"

int main(void)
{
	static const char *labels[4] = { "SparkMathFloatToBf16", "hardware bfloat(), non-NaN inputs", "hardware bfloat(), NaN inputs",
		"hardware bfloat() with NaN -> 0x7fff" };
	SparkDeviceQueue *queue = 0;
	SparkDeviceLibrary *library = 0;
	SparkDeviceKernel *kernel = 0;
	SparkDeviceBuffer counts, firsts;
	SparkDeviceBinding bindings[2];
	SparkDeviceLaunch launch;
	uint32_t index, constant = 0u;
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || strcmp(SparkDeviceGetProperties()->target,"metal.apple10") != 0 )
	{
		printf("SKIP test_metal_bf16_convert (needs an Apple family 10 GPU)\n");
		return(0);
	}
	if ( SparkDeviceQueueCreate(&queue,"bf16 sweep") != SPARK_STATUS_OK || SparkDeviceLibraryLoad(&library,SPARK_BF16_TEST_LIBRARY,"bf16 sweep") != SPARK_STATUS_OK ||
		SparkDeviceKernelCreate(&kernel,library,"spark_test_bf16_sweep") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&counts,SPARK_DEVICE_SPACE_SHARED_COHERENT,4u * sizeof(uint32_t),"bf16 sweep counts") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&firsts,SPARK_DEVICE_SPACE_SHARED_COHERENT,4u * sizeof(uint32_t),"bf16 sweep firsts") != SPARK_STATUS_OK )
	{
		printf("FAIL bf16 sweep setup: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	memset(counts.host,0,4u * sizeof(uint32_t));
	memset(firsts.host,0xff,4u * sizeof(uint32_t));
	bindings[0].range.buffer = &counts;
	bindings[0].range.offset = 0u;
	bindings[0].bytes = counts.bytes;
	bindings[0].access = SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE;
	bindings[1].range.buffer = &firsts;
	bindings[1].range.offset = 0u;
	bindings[1].bytes = firsts.bytes;
	bindings[1].access = SPARK_DEVICE_BIND_READ | SPARK_DEVICE_BIND_WRITE;
	memset(&launch,0,sizeof(launch));
	launch.groups.x = 1u << 16u;
	launch.groups.y = launch.groups.z = 1u;
	launch.threads.x = 256u;
	launch.threads.y = launch.threads.z = 1u;
	launch.constants = &constant;
	launch.constant_bytes = sizeof(constant);
	launch.bindings = bindings;
	launch.binding_count = 2u;
	if ( SparkDeviceKernelLaunch(queue,kernel,&launch) != SPARK_STATUS_OK || SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) != SPARK_STATUS_OK )
	{
		printf("FAIL bf16 sweep launch: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for ( index = 0u; index < 4u; index++ )
	{
		const uint32_t count = ((const uint32_t *)counts.host)[index], first = ((const uint32_t *)firsts.host)[index];
		if ( count == 0u )
			printf("%s: 0 of 4294967296 inputs differ from round-to-nearest-even with every NaN as 0x7fff\n",labels[index]);
		else
			printf("%s: %u of 4294967296 inputs differ from round-to-nearest-even with every NaN as 0x7fff (first 0x%08x)\n",labels[index],count,first);
	}
	if ( ((const uint32_t *)counts.host)[0] != 0u )
	{
		printf("FAIL the shared bf16 conversion breaks its contract on Metal\n");
		return(1);
	}
	printf("PASS the shared bf16 conversion matches its contract on all 2^32 FP32 inputs\n");
	return(0);
}
