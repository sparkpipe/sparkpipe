#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "runtime/launch.h"
#include "sparkpipe/spark_gpu.h"

#define GPU_API_SHARED_BYTES (64u * 1024u)

static int gpu_api_failures;

#define GPU_API_CHECK(call) do { SparkGpuError gpu_api_error = (call); if ( gpu_api_error != SPARK_GPU_SUCCESS ) { printf("FAIL %s: %s\n", #call, SparkGpuGetErrorString(gpu_api_error)); exit(1); } } while (0)
#define GPU_API_EXPECT(condition, ...) do { if ( !(condition) ) { printf("FAIL " __VA_ARGS__); printf("\n"); gpu_api_failures++; } } while (0)

__global__ void GpuApiFillKernel(uint32_t *values, uint32_t count, uint32_t base)
{
	uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
	if ( index < count )
		values[index] = base + index;
}

__global__ void GpuApiSharedKernel(uint32_t *values, uint32_t count)
{
	extern __shared__ uint32_t gpu_api_shared[];
	uint32_t index;
	for ( index = threadIdx.x; index < count; index += blockDim.x )
		gpu_api_shared[index] = index * 3u;
	__syncthreads();
	for ( index = threadIdx.x; index < count; index += blockDim.x )
		values[index] = gpu_api_shared[count - 1u - index];
}

__global__ void GpuApiSpinKernel(uint64_t cycles)
{
	const long long start = clock64();
	while ( (uint64_t)(clock64() - start) < cycles )
	{
	}
}

static void GpuApiHostCallback(void *context)
{
	*(volatile uint32_t *)context = 7u;
}

static void GpuApiDevice(void)
{
	SparkGpuDeviceProperties properties;
	int count = 0, device = -1, multiprocessors = 0;
	size_t free_bytes = 0, total_bytes = 0;
	GPU_API_CHECK(SparkGpuGetDeviceCount(&count));
	GPU_API_EXPECT(count >= 1, "device count %d", count);
	GPU_API_CHECK(SparkGpuSetDevice(0));
	GPU_API_CHECK(SparkGpuGetDevice(&device));
	GPU_API_EXPECT(device == 0, "current device %d after SparkGpuSetDevice(0)", device);
	GPU_API_CHECK(SparkGpuGetDeviceProperties(&properties, 0));
	GPU_API_CHECK(SparkGpuDeviceGetAttribute(&multiprocessors, SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0));
	GPU_API_EXPECT(multiprocessors > 0 && multiprocessors == properties.multiProcessorCount, "multiprocessors attribute %d properties %d", multiprocessors,
		properties.multiProcessorCount);
	GPU_API_CHECK(SparkGpuMemGetInfo(&free_bytes, &total_bytes));
	GPU_API_EXPECT(total_bytes > 0u && free_bytes <= total_bytes, "memory free %zu total %zu", free_bytes, total_bytes);
	GPU_API_CHECK(SparkGpuDeviceSynchronize());
}

static void GpuApiMemory(SparkGpuStream stream)
{
	enum { COUNT = 4096 };
	uint32_t *device_a, *device_b, *host_pinned, *host_mapped, *mapped_device, *managed;
	uint32_t *pageable = (uint32_t *)malloc(COUNT * sizeof(uint32_t)), *registered = (uint32_t *)malloc(COUNT * sizeof(uint32_t));
	uint32_t index, bad = 0u;
	if ( pageable == 0 || registered == 0 )
	{
		printf("FAIL host allocation\n");
		exit(1);
	}
	GPU_API_CHECK(SparkGpuMalloc((void **)&device_a, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuMalloc(&device_b, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuMallocHost((void **)&host_pinned, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuHostAlloc((void **)&host_mapped, COUNT * sizeof(uint32_t), SPARK_GPU_HOST_ALLOC_MAPPED | SPARK_GPU_HOST_ALLOC_PORTABLE));
	GPU_API_CHECK(SparkGpuHostGetDevicePointer((void **)&mapped_device, host_mapped, 0u));
	GPU_API_CHECK(SparkGpuMallocManaged(&managed, COUNT * sizeof(uint32_t)));
	for ( index = 0u; index < COUNT; index++ )
		pageable[index] = index * 5u + 1u;
	GPU_API_CHECK(SparkGpuMemcpy(device_a, pageable, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_HOST_TO_DEVICE));
	GPU_API_CHECK(SparkGpuMemcpy(device_b, device_a, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_DEVICE));
	GPU_API_CHECK(SparkGpuMemcpy(host_pinned, device_b, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST));
	for ( index = 0u; index < COUNT; index++ )
		bad += host_pinned[index] != index * 5u + 1u;
	GPU_API_EXPECT(bad == 0u, "synchronous copies: %u words differ", bad);
	GPU_API_CHECK(SparkGpuMemset(device_a, 0x5a, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuMemsetAsync(device_b, 0x00, COUNT * sizeof(uint32_t), stream));
	GPU_API_CHECK(SparkGpuMemset2DAsync(device_b, 64u * sizeof(uint32_t), 0x11, 16u * sizeof(uint32_t), COUNT / 64u, stream));
	GPU_API_CHECK(SparkGpuMemcpyAsync(host_pinned, device_b, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEFAULT, stream));
	GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
	bad = 0u;
	for ( index = 0u; index < COUNT; index++ )
		bad += host_pinned[index] != ((index % 64u) < 16u ? 0x11111111u : 0u);
	GPU_API_EXPECT(bad == 0u, "2D fill: %u words differ", bad);
	GPU_API_CHECK(SparkGpuMemcpy2DAsync(device_b, 64u * sizeof(uint32_t), device_a, 32u * sizeof(uint32_t), 8u * sizeof(uint32_t), COUNT / 64u,
		SPARK_GPU_COPY_DEVICE_TO_DEVICE, stream));
	GPU_API_CHECK(SparkGpuMemcpyAsync(host_pinned, device_b, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, stream));
	GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
	bad = 0u;
	for ( index = 0u; index < COUNT; index++ )
		bad += host_pinned[index] != ((index % 64u) < 8u ? 0x5a5a5a5au : (index % 64u) < 16u ? 0x11111111u : 0u);
	GPU_API_EXPECT(bad == 0u, "2D copy: %u words differ", bad);
	LM_LAUNCH((GpuApiFillKernel), (COUNT + 255u) / 256u, 256u, 0u, stream, mapped_device, (uint32_t)COUNT, 100u);
	GPU_API_CHECK(SparkGpuPeekAtLastError());
	LM_LAUNCH((GpuApiFillKernel), (COUNT + 255u) / 256u, 256u, 0u, stream, managed, (uint32_t)COUNT, 200u);
	GPU_API_CHECK(SparkGpuGetLastError());
	GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
	bad = 0u;
	for ( index = 0u; index < COUNT; index++ )
		bad += (host_mapped[index] != 100u + index) + (managed[index] != 200u + index);
	GPU_API_EXPECT(bad == 0u, "mapped and managed writes: %u words differ", bad);
	memset(registered, 0, COUNT * sizeof(uint32_t));
	GPU_API_CHECK(SparkGpuHostRegister(registered, COUNT * sizeof(uint32_t), SPARK_GPU_HOST_REGISTER_DEFAULT));
	GPU_API_CHECK(SparkGpuMemcpyAsync(registered, device_a, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, stream));
	GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
	GPU_API_CHECK(SparkGpuHostUnregister(registered));
	bad = 0u;
	for ( index = 0u; index < COUNT; index++ )
		bad += registered[index] != 0x5a5a5a5au;
	GPU_API_EXPECT(bad == 0u, "registered copy: %u words differ", bad);
	GPU_API_CHECK(SparkGpuFree(managed));
	GPU_API_CHECK(SparkGpuFreeHost(host_mapped));
	GPU_API_CHECK(SparkGpuFreeHost(host_pinned));
	GPU_API_CHECK(SparkGpuFree(device_b));
	GPU_API_CHECK(SparkGpuFree(device_a));
	free(registered);
	free(pageable);
}

static void GpuApiStreamsAndEvents(SparkGpuStream stream)
{
	SparkGpuStream side, urgent;
	SparkGpuEvent start, stop, fence;
	volatile uint32_t *flag;
	int least = 0, greatest = 0;
	float milliseconds = -1.0f;
	SparkGpuError query;
	GPU_API_CHECK(SparkGpuDeviceGetStreamPriorityRange(&least, &greatest));
	GPU_API_CHECK(SparkGpuStreamCreateWithPriority(&urgent, SPARK_GPU_STREAM_NON_BLOCKING, greatest));
	GPU_API_CHECK(SparkGpuStreamCreateWithFlags(&side, SPARK_GPU_STREAM_NON_BLOCKING));
	GPU_API_CHECK(SparkGpuEventCreate(&start));
	GPU_API_CHECK(SparkGpuEventCreate(&stop, 0u));
	GPU_API_CHECK(SparkGpuEventCreateWithFlags(&fence, SPARK_GPU_EVENT_DISABLE_TIMING));
	GPU_API_CHECK(SparkGpuMallocHost((void **)&flag, sizeof(uint32_t)));
	*flag = 0u;
	GPU_API_CHECK(SparkGpuEventRecord(start, side));
	LM_LAUNCH((GpuApiSpinKernel), 1u, 1u, 0u, side, (uint64_t)200000000u);
	GPU_API_CHECK(SparkGpuPeekAtLastError());
	GPU_API_CHECK(SparkGpuEventRecord(stop, side));
	query = SparkGpuStreamQuery(side);
	GPU_API_EXPECT(query == SPARK_GPU_ERROR_NOT_READY, "stream query while busy gave %s", SparkGpuGetErrorString(query));
	query = SparkGpuEventQuery(stop);
	GPU_API_EXPECT(query == SPARK_GPU_ERROR_NOT_READY, "event query while busy gave %s", SparkGpuGetErrorString(query));
	GPU_API_CHECK(SparkGpuEventRecord(fence, side));
	GPU_API_CHECK(SparkGpuStreamWaitEvent(urgent, fence, 0u));
	GPU_API_CHECK(SparkGpuLaunchHostFunc(urgent, GpuApiHostCallback, (void *)flag));
	GPU_API_CHECK(SparkGpuStreamSynchronize(urgent));
	GPU_API_EXPECT(*flag == 7u, "host function did not run before the stream completed");
	GPU_API_EXPECT(SparkGpuEventQuery(stop) == SPARK_GPU_SUCCESS, "the waiting stream finished before the event it waited on");
	GPU_API_CHECK(SparkGpuEventSynchronize(stop));
	GPU_API_CHECK(SparkGpuEventElapsedTime(&milliseconds, start, stop));
	GPU_API_EXPECT(milliseconds > 0.0f, "elapsed time %.6f ms", milliseconds);
	GPU_API_CHECK(SparkGpuStreamQuery(side));
	GPU_API_CHECK(SparkGpuFreeHost((void *)flag));
	GPU_API_CHECK(SparkGpuEventDestroy(fence));
	GPU_API_CHECK(SparkGpuEventDestroy(stop));
	GPU_API_CHECK(SparkGpuEventDestroy(start));
	GPU_API_CHECK(SparkGpuStreamDestroy(side));
	GPU_API_CHECK(SparkGpuStreamDestroy(urgent));
	(void)stream;
}

static void GpuApiGraph(SparkGpuStream stream)
{
	enum { COUNT = 1024 };
	SparkGpuGraph graph;
	SparkGpuGraphExec exec;
	SparkGpuCaptureStatus status;
	uint32_t *device, *host;
	size_t nodes = 0u;
	uint32_t index, bad = 0u, round;
	GPU_API_CHECK(SparkGpuMalloc(&device, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuMallocHost((void **)&host, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuStreamIsCapturing(stream, &status));
	GPU_API_EXPECT(status == SPARK_GPU_CAPTURE_STATUS_NONE, "stream captures before capture began");
	GPU_API_CHECK(SparkGpuStreamBeginCapture(stream, SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL));
	GPU_API_CHECK(SparkGpuStreamIsCapturing(stream, &status));
	GPU_API_EXPECT(status != SPARK_GPU_CAPTURE_STATUS_NONE, "stream does not capture after capture began");
	GPU_API_CHECK(SparkGpuMemsetAsync(device, 0, COUNT * sizeof(uint32_t), stream));
	LM_LAUNCH((GpuApiFillKernel), COUNT / 256u, 256u, 0u, stream, device, (uint32_t)COUNT, 9u);
	GPU_API_CHECK(SparkGpuPeekAtLastError());
	GPU_API_CHECK(SparkGpuMemcpyAsync(host, device, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, stream));
	GPU_API_CHECK(SparkGpuStreamEndCapture(stream, &graph));
	GPU_API_CHECK(SparkGpuGraphGetNodes(graph, 0, &nodes));
	GPU_API_EXPECT(nodes == 3u, "captured graph has %zu nodes, expected 3", nodes);
	GPU_API_CHECK(SparkGpuGraphInstantiate(&exec, graph, 0ull));
	GPU_API_CHECK(SparkGpuGraphUpload(exec, stream));
	for ( round = 0u; round < 3u; round++ )
	{
		memset(host, 0, COUNT * sizeof(uint32_t));
		GPU_API_CHECK(SparkGpuGraphLaunch(exec, stream));
		GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
		for ( index = 0u; index < COUNT; index++ )
			bad += host[index] != 9u + index;
	}
	GPU_API_EXPECT(bad == 0u, "graph replay: %u words differ", bad);
	GPU_API_CHECK(SparkGpuGraphExecDestroy(exec));
	GPU_API_CHECK(SparkGpuGraphDestroy(graph));
	GPU_API_CHECK(SparkGpuFreeHost(host));
	GPU_API_CHECK(SparkGpuFree(device));
}

static void GpuApiFunctions(SparkGpuStream stream)
{
	enum { COUNT = GPU_API_SHARED_BYTES / sizeof(uint32_t) };
	SparkGpuFunctionAttributes attributes;
	uint32_t *device, *host, *unreachable = 0;
	int blocks = 0, blocks_untyped = 0;
	uint32_t index, bad = 0u;
	SparkGpuError error;
	GPU_API_CHECK(SparkGpuMalloc(&device, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuMallocHost((void **)&host, COUNT * sizeof(uint32_t)));
	GPU_API_CHECK(SparkGpuFuncSetAttribute(GpuApiSharedKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)GPU_API_SHARED_BYTES));
	GPU_API_CHECK(SparkGpuFuncSetAttribute((const void *)GpuApiSharedKernel, SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES, (int)GPU_API_SHARED_BYTES));
	GPU_API_CHECK(SparkGpuFuncGetAttributes(&attributes, GpuApiSharedKernel));
	GPU_API_EXPECT(attributes.maxDynamicSharedSizeBytes == (int)GPU_API_SHARED_BYTES, "dynamic shared limit %d after opt-in to %u",
		attributes.maxDynamicSharedSizeBytes, GPU_API_SHARED_BYTES);
	GPU_API_CHECK(SparkGpuFuncGetAttributes(&attributes, (const void *)GpuApiFillKernel));
	GPU_API_EXPECT(attributes.maxThreadsPerBlock > 0, "fill kernel max threads %d", attributes.maxThreadsPerBlock);
	GPU_API_CHECK(SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, GpuApiFillKernel, 256, 0u));
	GPU_API_CHECK(SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_untyped, (const void *)GpuApiFillKernel, 256, 0u));
	GPU_API_EXPECT(blocks > 0 && blocks == blocks_untyped, "occupancy %d typed %d untyped", blocks, blocks_untyped);
	LM_LAUNCH((GpuApiSharedKernel), 1u, 256u, GPU_API_SHARED_BYTES, stream, device, (uint32_t)COUNT);
	GPU_API_CHECK(SparkGpuGetLastError());
	GPU_API_CHECK(SparkGpuMemcpyAsync(host, device, COUNT * sizeof(uint32_t), SPARK_GPU_COPY_DEVICE_TO_HOST, stream));
	GPU_API_CHECK(SparkGpuStreamSynchronize(stream));
	for ( index = 0u; index < COUNT; index++ )
		bad += host[index] != (uint32_t)(COUNT - 1u - index) * 3u;
	GPU_API_EXPECT(bad == 0u, "opted-in shared kernel: %u words differ", bad);
	LM_LAUNCH((GpuApiFillKernel), 1u, 4096u, 0u, stream, device, (uint32_t)COUNT, 0u);
	error = SparkGpuPeekAtLastError();
	GPU_API_EXPECT(error != SPARK_GPU_SUCCESS, "a 4096-thread block launched without an error");
	GPU_API_EXPECT(SparkGpuPeekAtLastError() == error, "peeking cleared the launch error");
	GPU_API_EXPECT(SparkGpuGetLastError() == error, "the launch error changed before it was taken");
	GPU_API_EXPECT(SparkGpuGetLastError() == SPARK_GPU_SUCCESS, "taking the launch error did not clear it");
	GPU_API_EXPECT(SparkGpuGetErrorString(error) != 0 && SparkGpuGetErrorString(error)[0] != '\0', "no text for error %d", (int)error);
	error = SparkGpuMalloc(&unreachable, (size_t)1 << 62);
	GPU_API_EXPECT(error == SPARK_GPU_ERROR_MEMORY_ALLOCATION, "an impossible allocation gave %s", SparkGpuGetErrorString(error));
	(void)SparkGpuGetLastError();
	error = SparkGpuMalloc((void **)0, 16u);
	GPU_API_EXPECT(error == SPARK_GPU_ERROR_INVALID_VALUE, "an allocation without a result pointer gave %s", SparkGpuGetErrorString(error));
	(void)SparkGpuGetLastError();
	error = SparkGpuSetDevice(1 << 20);
	GPU_API_EXPECT(error == SPARK_GPU_ERROR_INVALID_DEVICE, "an absent device gave %s", SparkGpuGetErrorString(error));
	(void)SparkGpuGetLastError();
	GPU_API_CHECK(SparkGpuSetDevice(0));
	GPU_API_CHECK(SparkGpuFreeHost(host));
	GPU_API_CHECK(SparkGpuFree(device));
}

int main(void)
{
	SparkGpuStream stream;
	GpuApiDevice();
	GPU_API_CHECK(SparkGpuStreamCreateWithFlags(&stream, SPARK_GPU_STREAM_NON_BLOCKING));
	GpuApiMemory(stream);
	GpuApiStreamsAndEvents(stream);
	GpuApiGraph(stream);
	GpuApiFunctions(stream);
	GPU_API_CHECK(SparkGpuStreamDestroy(stream));
	GPU_API_CHECK(SparkGpuDeviceSynchronize());
	if ( gpu_api_failures != 0 )
	{
		printf("FAIL gpu_api_cuda: %d checks failed\n", gpu_api_failures);
		return 1;
	}
	printf("PASS gpu_api_cuda: every spark_gpu.h function behaves as specified on the device\n");
	return 0;
}
