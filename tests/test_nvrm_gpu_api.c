#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "sparkpipe/spark_device.h"
#include "sparkpipe/spark_gpu.h"

#define API_BYTES 65536u
#define API_LARGE_BYTES (20u << 20)
#define API_ROWS 64u
#define API_WIDTH 48u
#define API_PITCH 80u

static int failures;
static volatile int host_function_calls;

#define CHECK(condition, what) \
	do { \
		if ( !(condition) ) \
		{ \
			printf("FAIL %s: %s\n",what,SparkDeviceLastError()->text); \
			failures++; \
		} \
	} while (0)

static uint8_t Pattern(size_t index, uint32_t seed)
{
	return((uint8_t)((index * 2654435761u + seed) >> 13));
}

static void Fill(uint8_t *bytes, size_t count, uint32_t seed)
{
	size_t index;
	for (index = 0u; index < count; index++)
		bytes[index] = Pattern(index,seed);
}

static size_t Wrong(const uint8_t *bytes, size_t count, uint32_t seed)
{
	size_t index,wrong = 0u;
	for (index = 0u; index < count; index++)
		wrong += bytes[index] != Pattern(index,seed);
	return(wrong);
}

static void HostFunction(void *context)
{
	(void)context;
	host_function_calls++;
}

static void TestMemory(SparkGpuStream stream)
{
	uint8_t *device = 0,*pinned = 0,*mapped = 0,*managed = 0,*pageable = (uint8_t *)malloc(API_BYTES);
	void *alias = 0;
	size_t index,wrong = 0u;
	CHECK(SparkGpuMalloc((void **)&device,API_BYTES) == SPARK_GPU_SUCCESS && device != 0,"allocate device memory");
	CHECK(SparkGpuMallocHost((void **)&pinned,API_BYTES) == SPARK_GPU_SUCCESS && SparkGpuHostAlloc((void **)&mapped,API_BYTES,SPARK_GPU_HOST_ALLOC_MAPPED) ==
		SPARK_GPU_SUCCESS && SparkGpuMallocManaged((void **)&managed,API_BYTES,SPARK_GPU_MEM_ATTACH_GLOBAL) == SPARK_GPU_SUCCESS,"allocate pinned, mapped and managed memory");
	CHECK(SparkGpuHostGetDevicePointer(&alias,mapped,0u) == SPARK_GPU_SUCCESS && alias == mapped,"mapped host memory has one address for host and device");
	CHECK(SparkGpuMemset(device,0x5a,API_BYTES) == SPARK_GPU_SUCCESS && SparkGpuMemcpy(pageable,device,API_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST) ==
		SPARK_GPU_SUCCESS,"a synchronous fill and a copy to pageable memory");
	for (index = 0u; index < API_BYTES; index++)
		wrong += pageable[index] != 0x5a;
	CHECK(wrong == 0u,"the fill reached pageable host memory through staging");
	Fill(pinned,API_BYTES,7u);
	CHECK(SparkGpuMemcpyAsync(device,pinned,API_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(managed,device,API_BYTES,SPARK_GPU_COPY_DEVICE_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(mapped,managed,API_BYTES,SPARK_GPU_COPY_DEFAULT,stream) == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,
		"queue pinned to device to managed to mapped copies on a stream");
	CHECK(Wrong(mapped,API_BYTES,7u) == 0u && Wrong(managed,API_BYTES,7u) == 0u,"every byte arrives in queue order");
	CHECK(SparkGpuFree(device) == SPARK_GPU_SUCCESS && SparkGpuFreeHost(pinned) == SPARK_GPU_SUCCESS && SparkGpuFreeHost(mapped) == SPARK_GPU_SUCCESS &&
		SparkGpuFree(managed) == SPARK_GPU_SUCCESS && SparkGpuFree(0) == SPARK_GPU_SUCCESS,"free every allocation, and a null pointer");
	CHECK(SparkGpuFree(device) == SPARK_GPU_ERROR_INVALID_VALUE && SparkGpuGetLastError() == SPARK_GPU_ERROR_INVALID_VALUE,"a second free is refused");
	free(pageable);
}

static void TestPageable(SparkGpuStream stream)
{
	uint8_t *device = 0,*source = (uint8_t *)malloc(API_LARGE_BYTES),*target = (uint8_t *)malloc(API_LARGE_BYTES),*registered = 0;
	void *alias = 0;
	const long page = sysconf(_SC_PAGESIZE);
	CHECK(source != 0 && target != 0 && SparkGpuMalloc((void **)&device,API_LARGE_BYTES) == SPARK_GPU_SUCCESS,"allocate 20 MiB on both sides");
	Fill(source,API_LARGE_BYTES,11u);
	memset(target,0,API_LARGE_BYTES);
	CHECK(SparkGpuMemcpyAsync(device,source,API_LARGE_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(target,device,API_LARGE_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS,
		"copy 20 MiB of pageable memory to the device and back, past the staging buffer's size");
	CHECK(Wrong(target,API_LARGE_BYTES,11u) == 0u,"pageable copies arrive whole");
	CHECK(posix_memalign((void **)&registered,(size_t)page,API_BYTES) == 0 && SparkGpuHostRegister(registered,API_BYTES,SPARK_GPU_HOST_REGISTER_MAPPED) ==
		SPARK_GPU_SUCCESS && SparkGpuHostGetDevicePointer(&alias,registered,0u) == SPARK_GPU_SUCCESS && alias != 0,"register host memory and find its device address");
	CHECK(SparkGpuMemcpyAsync(registered,device,API_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS && Wrong(registered,API_BYTES,11u) == 0u,"registered host memory takes a device copy by its host pointer");
	CHECK(SparkGpuHostUnregister(registered) == SPARK_GPU_SUCCESS && SparkGpuFree(device) == SPARK_GPU_SUCCESS,"unregister and free");
	free(registered);
	free(target);
	free(source);
}

static void Test2D(SparkGpuStream stream)
{
	uint8_t source[API_ROWS * API_PITCH],target[API_ROWS * API_PITCH];
	uint8_t *device = 0;
	size_t row,column,wrong = 0u;
	Fill(source,sizeof(source),3u);
	memset(target,0xee,sizeof(target));
	CHECK(SparkGpuMalloc((void **)&device,API_ROWS * API_WIDTH) == SPARK_GPU_SUCCESS,"allocate a packed 2D buffer");
	CHECK(SparkGpuMemcpy2DAsync(device,API_WIDTH,source,API_PITCH,API_WIDTH,API_ROWS,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemset2DAsync(device,API_WIDTH,0x11,8u,API_ROWS,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy2DAsync(target,API_PITCH,device,API_WIDTH,API_WIDTH,API_ROWS,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS,"pitched 2D copies both ways around a pitched 2D fill");
	for (row = 0u; row < API_ROWS; row++)
		for (column = 0u; column < API_PITCH; column++)
		{
			const size_t at = row * API_PITCH + column;
			const uint8_t expected = column < 8u ? 0x11 : column < API_WIDTH ? source[at] : 0xee;
			wrong += target[at] != expected;
		}
	CHECK(wrong == 0u,"each row lands at its pitch and the bytes between rows stay untouched");
	CHECK(SparkGpuMemcpy2DAsync(device,8u,source,API_PITCH,API_WIDTH,API_ROWS,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_ERROR_INVALID_VALUE,
		"a width beyond a pitch is refused");
	(void)SparkGpuGetLastError();
	CHECK(SparkGpuFree(device) == SPARK_GPU_SUCCESS,"free the 2D buffer");
}

static void TestStreamsAndEvents(void)
{
	SparkGpuStream first = 0,second = 0,blocking = 0;
	SparkGpuEvent start = 0,stop = 0,ready = 0;
	SparkGpuCaptureStatus capture = 99;
	SparkGpuGraph graph = 0;
	SparkGpuGraphExec exec = 0;
	uint8_t *device = 0,*pinned = 0;
	int least = 7,greatest = 7;
	float milliseconds = -1.0f;
	CHECK(SparkGpuStreamCreateWithFlags(&blocking,0u) == SPARK_GPU_ERROR_NOT_SUPPORTED && SparkGpuGetLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,
		"a blocking stream is refused: there is no legacy default stream to synchronise it with");
	CHECK(SparkGpuDeviceGetStreamPriorityRange(&least,&greatest) == SPARK_GPU_SUCCESS && least == 0 && greatest == 0,
		"the device reports one stream priority: the range is 0 to 0");
	CHECK(SparkGpuStreamCreateWithPriority(&blocking,SPARK_GPU_STREAM_NON_BLOCKING,-1) == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		strstr(SparkDeviceLastError()->text,"one stream priority") != 0 && SparkGpuGetLastError() == SPARK_GPU_ERROR_NOT_SUPPORTED,
		"a priority outside the range is refused rather than ignored");
	CHECK(SparkGpuStreamCreateWithFlags(&first,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamCreateWithPriority(&second,SPARK_GPU_STREAM_NON_BLOCKING,0) == SPARK_GPU_SUCCESS,"create two non-blocking streams, one at the default priority");
	CHECK(SparkGpuEventCreate(&start) == SPARK_GPU_SUCCESS && SparkGpuEventCreate(&stop) == SPARK_GPU_SUCCESS &&
		SparkGpuEventCreateWithFlags(&ready,SPARK_GPU_EVENT_DISABLE_TIMING) == SPARK_GPU_SUCCESS,"create events");
	CHECK(SparkGpuMalloc((void **)&device,API_LARGE_BYTES) == SPARK_GPU_SUCCESS && SparkGpuMallocHost((void **)&pinned,API_BYTES) == SPARK_GPU_SUCCESS,
		"allocate for the stream tests");
	CHECK(SparkGpuEventRecord(start,first) == SPARK_GPU_SUCCESS && SparkGpuMemsetAsync(device,0x3c,API_LARGE_BYTES,first) == SPARK_GPU_SUCCESS &&
		SparkGpuEventRecord(ready,first) == SPARK_GPU_SUCCESS && SparkGpuEventRecord(stop,first) == SPARK_GPU_SUCCESS,"record events around a fill");
	CHECK(SparkGpuStreamWaitEvent(second,ready,0u) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(pinned,device + API_LARGE_BYTES - API_BYTES,API_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,second) == SPARK_GPU_SUCCESS &&
		SparkGpuLaunchHostFunc(second,HostFunction,0) == SPARK_GPU_SUCCESS && SparkGpuStreamSynchronize(second) == SPARK_GPU_SUCCESS,
		"a second stream waits on the first stream's event, copies and runs a host function");
	CHECK(pinned[0] == 0x3c && pinned[API_BYTES - 1u] == 0x3c && host_function_calls == 1,"the copy saw the fill it waited for, and the host function ran");
	CHECK(SparkGpuEventSynchronize(stop) == SPARK_GPU_SUCCESS && SparkGpuEventQuery(stop) == SPARK_GPU_SUCCESS && SparkGpuStreamQuery(first) == SPARK_GPU_SUCCESS &&
		SparkGpuEventElapsedTime(&milliseconds,start,stop) == SPARK_GPU_SUCCESS && milliseconds >= 0.0f,"events complete and time the fill");
	CHECK(SparkGpuStreamIsCapturing(first,&capture) == SPARK_GPU_SUCCESS && capture == SPARK_GPU_CAPTURE_STATUS_NONE,"no stream is capturing");
	CHECK(SparkGpuStreamBeginCapture(first,SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL) == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		SparkGpuGraphInstantiate(&exec,graph,0u) == SPARK_GPU_ERROR_NOT_SUPPORTED && SparkGpuGraphLaunch(exec,first) == SPARK_GPU_ERROR_NOT_SUPPORTED &&
		strstr(SparkDeviceLastError()->text,"command lists") != 0,"stream capture and graphs are refused until command lists exist");
	(void)SparkGpuGetLastError();
	CHECK(SparkGpuEventDestroy(start) == SPARK_GPU_SUCCESS && SparkGpuEventDestroy(stop) == SPARK_GPU_SUCCESS && SparkGpuEventDestroy(ready) == SPARK_GPU_SUCCESS &&
		SparkGpuFree(device) == SPARK_GPU_SUCCESS && SparkGpuFreeHost(pinned) == SPARK_GPU_SUCCESS,"destroy the events and free");
	CHECK(SparkGpuStreamDestroy(first) == SPARK_GPU_SUCCESS && SparkGpuStreamDestroy(second) == SPARK_GPU_SUCCESS,"destroy the streams");
}

static void TestDevice(void)
{
	size_t free_bytes = 0u,total_bytes = 0u;
	int device = -1,count = 0,value = 0;
	CHECK(SparkGpuGetDevice(&device) == SPARK_GPU_SUCCESS && device == 0 && SparkGpuSetDevice(0) == SPARK_GPU_SUCCESS,"the process uses device 0");
	CHECK(SparkGpuSetDevice(1) == SPARK_GPU_ERROR_INVALID_DEVICE && SparkGpuGetLastError() == SPARK_GPU_ERROR_INVALID_DEVICE,
		"another device is refused once the first is in use");
	CHECK(SparkGpuDeviceGetAttribute(&count,SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT,0) == SPARK_GPU_SUCCESS && count > 0 &&
		(uint32_t)count == SparkDeviceGetProperties()->compute_units,"the multiprocessor count is the device's");
	CHECK(SparkGpuDeviceGetAttribute(&value,SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS,0) == SPARK_GPU_ERROR_NOT_SUPPORTED,
		"an attribute the backend cannot answer yet is refused");
	(void)SparkGpuGetLastError();
	CHECK(SparkGpuMemGetInfo(&free_bytes,&total_bytes) == SPARK_GPU_SUCCESS && total_bytes > 0u && free_bytes <= total_bytes,
		"free and total memory are reported, free within total");
	CHECK(SparkGpuDeviceSynchronize() == SPARK_GPU_SUCCESS,"synchronize the device");
}

static void TestMappedCopies(SparkGpuStream stream)
{
	uint64_t granularity = 0u;
	SparkDeviceMemoryHandle memory = 0u;
	SparkDeviceAddress reserved = 0u;
	uint8_t *source = (uint8_t *)malloc(API_LARGE_BYTES),*target = (uint8_t *)malloc(API_LARGE_BYTES),*device = 0,*mapped = 0;
	CHECK(source != 0 && target != 0 && SparkDeviceMemoryGranularity(SPARK_DEVICE_GRANULARITY_RECOMMENDED,&granularity) == SPARK_STATUS_OK &&
		SparkDeviceMemoryCreate(&memory,32u << 20,"gpu mapped copies") == SPARK_STATUS_OK &&
		SparkDeviceAddressReserve(&reserved,(32u << 20) + granularity,"gpu mapped copies") == SPARK_STATUS_OK &&
		SparkDeviceMemoryMap(reserved,32u << 20,memory,0u,SPARK_DEVICE_ACCESS_DEVICE_READ | SPARK_DEVICE_ACCESS_DEVICE_WRITE) == SPARK_STATUS_OK &&
		SparkGpuMalloc((void **)&device,API_LARGE_BYTES) == SPARK_GPU_SUCCESS,
		"map 32 MiB of shared memory into a reservation one granule longer, without adopting it, and allocate device memory");
	mapped = (uint8_t *)(uintptr_t)reserved;
	Fill(source,API_LARGE_BYTES,17u);
	CHECK(SparkGpuMemcpy(mapped,source,API_LARGE_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(device,mapped,API_LARGE_BYTES,SPARK_GPU_COPY_DEVICE_TO_DEVICE) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpy(target,device,API_LARGE_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_SUCCESS,
		"pageable host memory copies into the mapping, the mapping into device memory, and that back to the host");
	CHECK(Wrong(target,API_LARGE_BYTES,17u) == 0u,"every byte arrives through the mapping");
	Fill(source,API_LARGE_BYTES,23u);
	CHECK(SparkGpuMemcpyAsync(device,source,API_LARGE_BYTES,SPARK_GPU_COPY_HOST_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(mapped + 4096u,device,API_LARGE_BYTES,SPARK_GPU_COPY_DEVICE_TO_DEVICE,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuMemcpyAsync(target,mapped + 4096u,API_LARGE_BYTES,SPARK_GPU_COPY_DEVICE_TO_HOST,stream) == SPARK_GPU_SUCCESS &&
		SparkGpuStreamSynchronize(stream) == SPARK_GPU_SUCCESS && Wrong(target,API_LARGE_BYTES,23u) == 0u,
		"queued copies through the mapping at an offset arrive in order");
	CHECK(SparkGpuMemcpy(target,mapped + (32u << 20),4096u,SPARK_GPU_COPY_DEVICE_TO_HOST) == SPARK_GPU_ERROR_INVALID_VALUE &&
		SparkGpuGetLastError() == SPARK_GPU_ERROR_INVALID_VALUE,
		"a copy from the reserved granule with nothing mapped is refused, not read by the CPU");
	CHECK(SparkGpuMemcpy(mapped + (32u << 20),source,4096u,SPARK_GPU_COPY_HOST_TO_DEVICE) == SPARK_GPU_ERROR_INVALID_VALUE &&
		SparkGpuGetLastError() == SPARK_GPU_ERROR_INVALID_VALUE,
		"a copy into the reserved granule with nothing mapped is refused, not written by the CPU");
	CHECK(SparkGpuFree(device) == SPARK_GPU_SUCCESS && SparkDeviceMemoryUnmap(reserved,32u << 20) == SPARK_STATUS_OK &&
		SparkDeviceAddressRelease(reserved,(32u << 20) + granularity) == SPARK_STATUS_OK && SparkDeviceMemoryRelease(memory) == SPARK_STATUS_OK,
		"free, unmap and release");
	free(source);
	free(target);
}

int main(void)
{
	SparkGpuStream stream = 0;
	if ( access("/dev/nvidiactl",F_OK) != 0 )
	{
		printf("SKIP test_nvrm_gpu_api (no open kernel module on this host)\n");
		return(0);
	}
	CHECK(SparkGpuStreamCreateWithFlags(&stream,SPARK_GPU_STREAM_NON_BLOCKING) == SPARK_GPU_SUCCESS && SparkDeviceGetProperties() != 0,
		"the first call initializes the device");
	TestMemory(stream);
	TestPageable(stream);
	Test2D(stream);
	TestMappedCopies(stream);
	TestStreamsAndEvents();
	TestDevice();
	CHECK(SparkGpuStreamDestroy(stream) == SPARK_GPU_SUCCESS,"destroy the stream");
	CHECK(SparkNvrmGpuRelease() == SPARK_GPU_SUCCESS && SparkDeviceShutdown() == SPARK_STATUS_OK,"release the default queue and staging, then shut down");
	if ( failures != 0 )
	{
		printf("\nFAIL (%d)\n",failures);
		return(1);
	}
	printf("PASS spark_gpu.h over nvrm: every space, staged pageable copies, 2D copies and fills, copies through device mappings, streams, events, host functions and device queries\n");
	return(0);
}
