#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "sparkpipe/spark_device.h"
#include "runtime/device/nvrm/spark_gpu_nvrm.h"
#include "runtime/device/nvrm/nvrm_occupancy.h"

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s\n",what); \
			failures++; \
		} \
	} while (0)

static void Unregistered(void)
{
}

static void *OtherThread(void *context)
{
	SparkGpuError *seen = (SparkGpuError *)context;
	seen[0] = SparkGpuPeekAtLastError();
	(void)SparkNvrmLaunchUnsupported((const void *)Unregistered,"a cluster launch");
	seen[1] = SparkGpuGetLastError();
	return(0);
}

static void TestOccupancy(void)
{
	const SparkNvrmOccupancyLimits limits = {48u, 65536u, 32u, 100u * 1024u, 256u, 128u};
	CHECK(SparkNvrmOccupancy(&limits,128u,32u,0u) == 12u,"128 threads at 32 registers are held to 12 blocks by 48 warps per multiprocessor");
	CHECK(SparkNvrmOccupancy(&limits,256u,64u,0u) == 4u,"256 threads at 64 registers are held to 4 blocks by the register file");
	CHECK(SparkNvrmOccupancy(&limits,256u,40u,0u) == 6u,"40 registers per thread round up to 1280 per warp: 51 warps fit, warps per SM still allow 6");
	CHECK(SparkNvrmOccupancy(&limits,32u,16u,0u) == 32u,"with 32 blocks per multiprocessor allowed, single-warp blocks stop there");
	CHECK(SparkNvrmOccupancy(&limits,128u,32u,33000u) == 3u,"33000 bytes of shared memory round to 33024 and allow 3 blocks in 100 KiB");
	CHECK(SparkNvrmOccupancy(&limits,64u,0u,0u) == 24u,"with no register count only warps and blocks limit the occupancy");
	CHECK(SparkNvrmOccupancy(&limits,2048u,16u,0u) == 0u && SparkNvrmOccupancy(&limits,128u,16u,110000u) == 0u && SparkNvrmOccupancy(&limits,0u,16u,0u) == 0u &&
		SparkNvrmOccupancy(0,128u,16u,0u) == 0u,"a block larger than a multiprocessor, more shared memory than it has, an empty block or no limits give 0");
}

int main(void)
{
	TestOccupancy();
	const uint32_t grid[3] = {1u, 1u, 1u},block[3] = {32u, 1u, 1u};
	SparkGpuError seen[2] = {SPARK_GPU_ERROR_UNKNOWN, SPARK_GPU_ERROR_UNKNOWN};
	pthread_t other;
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS,"a thread starts with no last error");
	CHECK(SparkNvrmLaunchPacked((const void *)Unregistered,grid,block,0u,0,0,0,0u,0u) == SPARK_GPU_ERROR_INVALID_VALUE &&
		strstr(SparkDeviceLastError()->text,"no module registered") != 0,"a launch of a function no module registered is refused, saying so");
	CHECK(SparkGpuPeekAtLastError() == SPARK_GPU_ERROR_INVALID_VALUE && SparkGpuPeekAtLastError() == SPARK_GPU_ERROR_INVALID_VALUE,
		"Peek reads the last error without clearing it");
	CHECK(pthread_create(&other,0,OtherThread,seen) == 0 && pthread_join(other,0) == 0 && seen[0] == SPARK_GPU_SUCCESS &&
		seen[1] == SPARK_GPU_ERROR_NOT_SUPPORTED,"each thread keeps its own last error");
	CHECK(SparkGpuGetLastError() == SPARK_GPU_ERROR_INVALID_VALUE && SparkGpuGetLastError() == SPARK_GPU_SUCCESS,
		"Get returns this thread's last error and clears it");
	CHECK(SparkNvrmLaunchPacked((const void *)Unregistered,0,block,0u,0,0,0,0u,0u) == SPARK_GPU_ERROR_INVALID_VALUE &&
		SparkNvrmLaunchPacked((const void *)Unregistered,grid,block,(size_t)UINT32_MAX + 1u,0,0,0,0u,0u) == SPARK_GPU_ERROR_INVALID_VALUE &&
		SparkNvrmLaunchPacked((const void *)Unregistered,grid,block,0u,0,0,0,1u,8u) == SPARK_GPU_ERROR_INVALID_VALUE,
		"a launch without its grid, with more than 4 GiB of shared memory or with parameters but no block is refused");
	CHECK(SparkNvrmLaunchUnsupported((const void *)Unregistered,"a programmatic dependent launch") == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		strstr(SparkDeviceLastError()->text,"dependent") != 0 && strstr(SparkDeviceLastError()->text,"unregistered") != 0,
		"an unsupported launch kind is refused, naming the kind and the function");
	CHECK(SparkNvrmSharedMemoryOptIn((const void *)Unregistered,1024u) == SPARK_GPU_ERROR_INVALID_VALUE,"shared memory is granted only on an initialized device");
	CHECK(strcmp(SparkGpuGetErrorString(SPARK_GPU_SUCCESS),"no error") == 0 && strcmp(SparkGpuGetErrorString(SPARK_GPU_ERROR_NOT_SUPPORTED),
		"operation not supported") == 0 && strcmp(SparkGpuGetErrorString(12345),"unknown error") == 0,"every error has its string");
	(void)SparkGpuGetLastError();
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS the hosted GPU front keeps a last error per thread with CUDA's peek and get semantics and refuses launches it cannot make\n");
	return(0);
}
