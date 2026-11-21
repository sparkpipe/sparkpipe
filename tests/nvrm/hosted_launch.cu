#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"
#include "runtime/launch.h"

#define HOSTED_THREADS 128u
#define HOSTED_GROUPS 4u
#define HOSTED_WORDS (HOSTED_THREADS * HOSTED_GROUPS)
#define HOSTED_ROWS 5u
#define HOSTED_STEP 7u

typedef struct HostedShape
{
	uint32_t rows;
	uint16_t lanes;
	uint8_t flag;
	double weight;
} HostedShape;

__global__ void HostedMix(uint8_t small, const uint64_t *table, uint16_t lanes, HostedShape shape, float scale, int8_t sign, double *output)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = (double)table[index % shape.rows] * scale * shape.weight + small + lanes + sign + shape.lanes + shape.flag;
}

template<uint32_t Step>
__global__ void HostedStep(uint32_t *output, uint32_t base)
{
	const uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	output[index] = base + Step * index;
}

__device__ uint32_t hosted_touched;

__global__ void HostedTouch(void)
{
	if ( threadIdx.x == 0u && blockIdx.x == 0u )
		hosted_touched = 1u;
}

static int failures;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s: %s\n",what,SparkDeviceLastError()->text); \
			failures++; \
		} \
	} while (0)

static double MixExpected(uint32_t index, const uint64_t *table)
{
	return((double)table[index % HOSTED_ROWS] * 0.5f * 1.25 + 3 + 9 + (-2) + 11 + 1);
}

static void TestLaunches(SparkGpuStream queue, SparkDeviceBuffer *table, SparkDeviceBuffer *doubles, SparkDeviceBuffer *words)
{
	const uint64_t *values = (const uint64_t *)table->host;
	HostedShape shape;
	SparkGpuStream default_stream = 0;
	uint32_t index,wrong = 0u;
	memset(&shape,0,sizeof(shape));
	shape.rows = HOSTED_ROWS;
	shape.lanes = 11u;
	shape.flag = 1u;
	shape.weight = 1.25;
	LM_LAUNCH((HostedMix),HOSTED_GROUPS,HOSTED_THREADS,0,0,3,(const uint64_t *)table->device,9,shape,0.5f,-2,(double *)doubles->device);
	CHECK(SparkGpuPeekAtLastError() == SPARK_GPU_SUCCESS && SparkNvrmGpuDefaultStream(&default_stream) == SPARK_GPU_SUCCESS &&
		SparkDeviceQueueWait(default_stream,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,"launch a mixed signature on the null stream, the process default queue");
	for (index = 0u; index < HOSTED_WORDS; index++)
		wrong += ((const double *)doubles->host)[index] != MixExpected(index,values);
	CHECK(wrong == 0u,"every argument reaches the kernel converted to its parameter's type and laid out as the PTX declares");
	LM_LAUNCH((HostedStep<HOSTED_STEP>),dim3(HOSTED_GROUPS),dim3(HOSTED_THREADS),0,queue,(uint32_t *)words->device,100u);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"launch a template kernel on a created queue");
	wrong = 0u;
	for (index = 0u; index < HOSTED_WORDS; index++)
		wrong += ((const uint32_t *)words->host)[index] != 100u + HOSTED_STEP * index;
	CHECK(wrong == 0u,"the template kernel ran on the queue it was given");
	LM_LAUNCH((HostedTouch),1,32,0,queue);
	CHECK(SparkGpuGetLastError() == SPARK_GPU_SUCCESS && SparkDeviceQueueWait(queue,SPARK_DEVICE_WAIT_FOREVER) == SPARK_STATUS_OK,
		"launch a kernel that takes no arguments");
}

static void TestErrors(SparkGpuStream queue, SparkDeviceBuffer *words)
{
	const uint32_t grid[3] = {1u, 1u, 1u},block[3] = {32u, 1u, 1u};
	SparkDeviceParameter layout[2];
	uint8_t packed[16];
	(void)SparkGpuGetLastError();
	LM_LAUNCH((HostedStep<HOSTED_STEP>),1,2048,0,queue,(uint32_t *)words->device,1u);
	CHECK(SparkGpuPeekAtLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED && SparkGpuPeekAtLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,
		"a refused launch leaves its error for Peek, which does not clear it");
	LM_LAUNCH((HostedStep<HOSTED_STEP>),1,32,0,queue,(uint32_t *)words->device,1u);
	CHECK(SparkGpuPeekAtLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,"a later good launch does not clear the last error");
	CHECK(SparkGpuGetLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED && SparkGpuGetLastError() == SPARK_GPU_SUCCESS,"Get returns the last error and clears it");
	CHECK(LM_LAUNCH_DEPENDENT((HostedStep<HOSTED_STEP>),1,32,0,queue,(uint32_t *)words->device,1u) == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		strstr(SparkDeviceLastError()->text,"dependent") != 0 && SparkGpuGetLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,
		"a programmatic dependent launch is refused, naming the kernel, until the descriptor carries it");
	CHECK(LmLaunchClusterKernel(HostedStep<HOSTED_STEP>,dim3(1),dim3(32),0,queue,2u,0u,(uint32_t *)words->device,1u) == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		strstr(SparkDeviceLastError()->text,"cluster") != 0 && SparkGpuGetLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,"a cluster launch is refused");
	memset(packed,0,sizeof(packed));
	layout[0].offset = 0u;
	layout[0].bytes = 8u;
	layout[0].align = 8u;
	layout[1].offset = 8u;
	layout[1].bytes = 8u;
	layout[1].align = 8u;
	CHECK(SparkNvrmLaunchPacked((const void *)HostedStep<HOSTED_STEP>,grid,block,0u,queue,packed,layout,2u,16u) == SPARK_GPU_ERROR_INVALID_VALUE &&
		strstr(SparkDeviceLastError()->text,"parameter") != 0,"a host layout that disagrees with the kernel's PTX is refused");
	CHECK(SparkNvrmLaunchPacked((const void *)TestErrors,grid,block,0u,queue,packed,layout,0u,0u) == SPARK_GPU_ERROR_INVALID_VALUE,
		"a function no module registered is not launched");
	CHECK(LmKernelSharedMemoryOptIn((const void *)HostedStep<HOSTED_STEP>,1024u) == LM_LAUNCH_OK &&
		LmKernelSharedMemoryOptIn((const void *)HostedStep<HOSTED_STEP>,SparkDeviceGetProperties()->max_shared_bytes + 1u) == LM_LAUNCH_ERR_ATTRIBUTE,
		"shared memory opt-in grants what the device has and refuses more");
	(void)SparkGpuGetLastError();
}

int main(void)
{
	SparkDeviceBuffer table,doubles,words;
	SparkDeviceQueue *queue = 0;
	uint32_t index;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_hosted_launch (no open kernel module on this host)\n");
		return(0);
	}
	if ( SparkDeviceInitialize(0) != SPARK_STATUS_OK || SparkDeviceQueueCreate(&queue,"hosted queue") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&table,SPARK_DEVICE_SPACE_SHARED_COHERENT,HOSTED_ROWS * sizeof(uint64_t),"hosted table") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&doubles,SPARK_DEVICE_SPACE_SHARED_COHERENT,HOSTED_WORDS * sizeof(double),"hosted doubles") != SPARK_STATUS_OK ||
		SparkDeviceAllocate(&words,SPARK_DEVICE_SPACE_SHARED_COHERENT,HOSTED_WORDS * sizeof(uint32_t),"hosted words") != SPARK_STATUS_OK )
	{
		printf("FAIL set up the device: %s\n",SparkDeviceLastError()->text);
		return(1);
	}
	for (index = 0u; index < HOSTED_ROWS; index++)
		((uint64_t *)table.host)[index] = 1000u + 17u * index;
	TestLaunches(queue,&table,&doubles,&words);
	TestErrors(queue,&words);
	CHECK(SparkDeviceFree(&table) == SPARK_STATUS_OK && SparkDeviceFree(&doubles) == SPARK_STATUS_OK && SparkDeviceFree(&words) == SPARK_STATUS_OK &&
		SparkDeviceQueueDestroy(queue) == SPARK_STATUS_OK,"free the buffers and the queue");
	CHECK(SparkNvrmGpuRelease() == SPARK_GPU_SUCCESS && SparkDeviceShutdown() == SPARK_STATUS_OK,
		"releasing the default queue and the registered kernels lets the device shut down");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS LM_LAUNCH packs arguments by parameter type, launches registered kernels on the default and created queues, and keeps CUDA's last-error semantics\n");
	return(0);
}
