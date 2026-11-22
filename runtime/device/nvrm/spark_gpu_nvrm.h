#ifndef SPARKPIPE_RUNTIME_DEVICE_NVRM_SPARK_GPU_NVRM_H
#define SPARKPIPE_RUNTIME_DEVICE_NVRM_SPARK_GPU_NVRM_H

#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_device.h"

#ifdef __cplusplus
#define SPARK_GPU_DEFAULT(value) = value
extern "C" {
#else
#define SPARK_GPU_DEFAULT(value)
#endif

typedef int SparkGpuError;
typedef SparkDeviceQueue *SparkGpuStream;
typedef SparkDeviceEvent *SparkGpuEvent;
typedef struct SparkNvrmGpuGraph *SparkGpuGraph;
typedef struct SparkNvrmGpuGraphExec *SparkGpuGraphExec;
typedef struct SparkNvrmGpuGraphNode *SparkGpuGraphNode;
typedef SparkDeviceHostFunction SparkGpuHostFunction;
typedef int SparkGpuCopyKind;
typedef int SparkGpuCaptureStatus;
typedef int SparkGpuCaptureMode;
typedef int SparkGpuDeviceAttribute;
typedef int SparkGpuFunctionAttribute;

typedef struct SparkGpuFunctionAttributes
{
	int maxThreadsPerBlock;
	int numRegs;
	size_t sharedSizeBytes;
	size_t localSizeBytes;
	size_t constSizeBytes;
	int maxDynamicSharedSizeBytes;
} SparkGpuFunctionAttributes;

#define SPARK_GPU_SUCCESS 0
#define SPARK_GPU_ERROR_INVALID_VALUE 1
#define SPARK_GPU_ERROR_MEMORY_ALLOCATION 2
#define SPARK_GPU_ERROR_INVALID_DEVICE 101
#define SPARK_GPU_ERROR_NOT_READY 600
#define SPARK_GPU_ERROR_NOT_SUPPORTED 801
#define SPARK_GPU_ERROR_UNKNOWN 999

#define SPARK_GPU_COPY_HOST_TO_HOST 0
#define SPARK_GPU_COPY_HOST_TO_DEVICE 1
#define SPARK_GPU_COPY_DEVICE_TO_HOST 2
#define SPARK_GPU_COPY_DEVICE_TO_DEVICE 3
#define SPARK_GPU_COPY_DEFAULT 4

#define SPARK_GPU_STREAM_NON_BLOCKING 1u
#define SPARK_GPU_EVENT_DISABLE_TIMING 2u
#define SPARK_GPU_HOST_ALLOC_DEFAULT 0u
#define SPARK_GPU_HOST_ALLOC_PORTABLE 1u
#define SPARK_GPU_HOST_ALLOC_MAPPED 2u
#define SPARK_GPU_HOST_REGISTER_DEFAULT 0u
#define SPARK_GPU_HOST_REGISTER_PORTABLE 1u
#define SPARK_GPU_HOST_REGISTER_MAPPED 2u
#define SPARK_GPU_MEM_ATTACH_GLOBAL 1u
#define SPARK_GPU_CAPTURE_STATUS_NONE 0
#define SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL 1

#define SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT 16
#define SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS 88
#define SPARK_GPU_ATTRIBUTE_HOST_POINTER_FOR_REGISTERED_MEMORY 91
#define SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES 100
#define SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES 8

SparkGpuError SparkGpuGetLastError(void);
SparkGpuError SparkGpuPeekAtLastError(void);
const char *SparkGpuGetErrorString(SparkGpuError error);

SparkGpuError SparkGpuGetDevice(int *device);
SparkGpuError SparkGpuSetDevice(int device);
SparkGpuError SparkGpuDeviceGetAttribute(int *value, SparkGpuDeviceAttribute attribute, int device);
SparkGpuError SparkGpuDeviceGetStreamPriorityRange(int *least_priority, int *greatest_priority);
SparkGpuError SparkGpuDeviceSynchronize(void);
SparkGpuError SparkGpuMemGetInfo(size_t *free_bytes, size_t *total_bytes);

SparkGpuError SparkGpuMalloc(void **pointer, size_t bytes);
SparkGpuError SparkGpuMallocManaged(void **pointer, size_t bytes, unsigned int flags SPARK_GPU_DEFAULT(SPARK_GPU_MEM_ATTACH_GLOBAL));
SparkGpuError SparkGpuMallocHost(void **pointer, size_t bytes);
SparkGpuError SparkGpuHostAlloc(void **pointer, size_t bytes, unsigned int flags);
SparkGpuError SparkGpuHostGetDevicePointer(void **device_pointer, void *host_pointer, unsigned int flags);
SparkGpuError SparkGpuHostRegister(void *host_pointer, size_t bytes, unsigned int flags);
SparkGpuError SparkGpuHostUnregister(void *host_pointer);
SparkGpuError SparkGpuFree(void *pointer);
SparkGpuError SparkGpuFreeHost(void *pointer);

SparkGpuError SparkGpuMemcpy(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind);
SparkGpuError SparkGpuMemcpyAsync(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind, SparkGpuStream stream SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuMemcpy2DAsync(void *destination, size_t destination_pitch, const void *source, size_t source_pitch, size_t width, size_t height,
	SparkGpuCopyKind kind, SparkGpuStream stream SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuMemset(void *destination, int value, size_t bytes);
SparkGpuError SparkGpuMemsetAsync(void *destination, int value, size_t bytes, SparkGpuStream stream SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuMemset2DAsync(void *destination, size_t pitch, int value, size_t width, size_t height, SparkGpuStream stream SPARK_GPU_DEFAULT(0));

SparkGpuError SparkGpuStreamCreateWithFlags(SparkGpuStream *stream, unsigned int flags);
SparkGpuError SparkGpuStreamCreateWithPriority(SparkGpuStream *stream, unsigned int flags, int priority);
SparkGpuError SparkGpuStreamDestroy(SparkGpuStream stream);
SparkGpuError SparkGpuStreamSynchronize(SparkGpuStream stream);
SparkGpuError SparkGpuStreamQuery(SparkGpuStream stream);
SparkGpuError SparkGpuStreamWaitEvent(SparkGpuStream stream, SparkGpuEvent event, unsigned int flags SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuStreamIsCapturing(SparkGpuStream stream, SparkGpuCaptureStatus *status);
SparkGpuError SparkGpuStreamBeginCapture(SparkGpuStream stream, SparkGpuCaptureMode mode);
SparkGpuError SparkGpuStreamEndCapture(SparkGpuStream stream, SparkGpuGraph *graph);
SparkGpuError SparkGpuLaunchHostFunc(SparkGpuStream stream, SparkGpuHostFunction function, void *context);

SparkGpuError SparkGpuEventCreate(SparkGpuEvent *event);
SparkGpuError SparkGpuEventCreateWithFlags(SparkGpuEvent *event, unsigned int flags);
SparkGpuError SparkGpuEventDestroy(SparkGpuEvent event);
SparkGpuError SparkGpuEventRecord(SparkGpuEvent event, SparkGpuStream stream SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuEventQuery(SparkGpuEvent event);
SparkGpuError SparkGpuEventSynchronize(SparkGpuEvent event);
SparkGpuError SparkGpuEventElapsedTime(float *milliseconds, SparkGpuEvent start, SparkGpuEvent stop);

SparkGpuError SparkGpuGraphInstantiate(SparkGpuGraphExec *exec, SparkGpuGraph graph, unsigned long long flags SPARK_GPU_DEFAULT(0));
SparkGpuError SparkGpuGraphUpload(SparkGpuGraphExec exec, SparkGpuStream stream);
SparkGpuError SparkGpuGraphLaunch(SparkGpuGraphExec exec, SparkGpuStream stream);
SparkGpuError SparkGpuGraphGetNodes(SparkGpuGraph graph, SparkGpuGraphNode *nodes, size_t *count);
SparkGpuError SparkGpuGraphExecDestroy(SparkGpuGraphExec exec);
SparkGpuError SparkGpuGraphDestroy(SparkGpuGraph graph);

SparkGpuError SparkGpuFuncSetAttribute(const void *function, SparkGpuFunctionAttribute attribute, int value);
SparkGpuError SparkGpuFuncGetAttributes(SparkGpuFunctionAttributes *attributes, const void *function);
SparkGpuError SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *function, int block_threads, size_t shared_bytes);

SparkGpuError SparkNvrmGpuDefaultStream(SparkGpuStream *stream);
SparkGpuError SparkNvrmGpuRelease(void);
SparkGpuError SparkNvrmLaunchPacked(const void *function, const uint32_t grid[3], const uint32_t block[3], size_t shared_bytes, SparkGpuStream stream,
	const void *parameters, const SparkDeviceParameter *layout, uint32_t count, uint32_t bytes);
SparkGpuError SparkNvrmLaunchUnsupported(const void *function, const char *what);
SparkGpuError SparkNvrmSharedMemoryOptIn(const void *function, uint32_t shared_bytes);
SparkStatus SparkNvrmVideoMemoryInfo(uint64_t *free_bytes, uint64_t *total_bytes);

#ifdef __cplusplus
}

static inline SparkGpuError SparkGpuEventCreate(SparkGpuEvent *event, unsigned int flags)
{
	return(SparkGpuEventCreateWithFlags(event,flags));
}

template<typename T>
static inline SparkGpuError SparkGpuMalloc(T **pointer, size_t bytes)
{
	return(SparkGpuMalloc((void **)pointer,bytes));
}

template<typename T>
static inline SparkGpuError SparkGpuMallocManaged(T **pointer, size_t bytes, unsigned int flags = SPARK_GPU_MEM_ATTACH_GLOBAL)
{
	return(SparkGpuMallocManaged((void **)pointer,bytes,flags));
}

template<typename T>
static inline SparkGpuError SparkGpuMallocHost(T **pointer, size_t bytes)
{
	return(SparkGpuMallocHost((void **)pointer,bytes));
}

template<typename T>
static inline SparkGpuError SparkGpuHostAlloc(T **pointer, size_t bytes, unsigned int flags)
{
	return(SparkGpuHostAlloc((void **)pointer,bytes,flags));
}

template<typename T>
static inline SparkGpuError SparkGpuHostGetDevicePointer(T **device_pointer, void *host_pointer, unsigned int flags)
{
	return(SparkGpuHostGetDevicePointer((void **)device_pointer,host_pointer,flags));
}

template<typename T>
static inline SparkGpuError SparkGpuFuncSetAttribute(T *function, SparkGpuFunctionAttribute attribute, int value)
{
	return(SparkGpuFuncSetAttribute((const void *)function,attribute,value));
}

template<typename T>
static inline SparkGpuError SparkGpuFuncGetAttributes(SparkGpuFunctionAttributes *attributes, T *function)
{
	return(SparkGpuFuncGetAttributes(attributes,(const void *)function));
}

template<typename T>
static inline SparkGpuError SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, T function, int block_threads, size_t shared_bytes)
{
	return(SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(blocks,(const void *)function,block_threads,shared_bytes));
}
#endif

#endif
