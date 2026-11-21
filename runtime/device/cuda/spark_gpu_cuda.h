#ifndef SPARKPIPE_RUNTIME_DEVICE_CUDA_SPARK_GPU_CUDA_H
#define SPARKPIPE_RUNTIME_DEVICE_CUDA_SPARK_GPU_CUDA_H

#include <stddef.h>
#include <cuda_runtime_api.h>

#ifdef __cplusplus
#define SPARK_GPU_DEFAULT(value) = value
#else
#define SPARK_GPU_DEFAULT(value)
#endif

typedef cudaError_t SparkGpuError;
typedef cudaStream_t SparkGpuStream;
typedef cudaEvent_t SparkGpuEvent;
typedef cudaGraph_t SparkGpuGraph;
typedef cudaGraphExec_t SparkGpuGraphExec;
typedef cudaGraphNode_t SparkGpuGraphNode;
typedef cudaHostFn_t SparkGpuHostFunction;
typedef enum cudaMemcpyKind SparkGpuCopyKind;
typedef enum cudaStreamCaptureStatus SparkGpuCaptureStatus;
typedef enum cudaStreamCaptureMode SparkGpuCaptureMode;
typedef enum cudaDeviceAttr SparkGpuDeviceAttribute;
typedef enum cudaFuncAttribute SparkGpuFunctionAttribute;
typedef struct cudaDeviceProp SparkGpuDeviceProperties;
typedef struct cudaFuncAttributes SparkGpuFunctionAttributes;

#define SPARK_GPU_SUCCESS cudaSuccess
#define SPARK_GPU_ERROR_INVALID_VALUE cudaErrorInvalidValue
#define SPARK_GPU_ERROR_MEMORY_ALLOCATION cudaErrorMemoryAllocation
#define SPARK_GPU_ERROR_INVALID_DEVICE cudaErrorInvalidDevice
#define SPARK_GPU_ERROR_NOT_READY cudaErrorNotReady
#define SPARK_GPU_ERROR_NOT_SUPPORTED cudaErrorNotSupported
#define SPARK_GPU_ERROR_UNKNOWN cudaErrorUnknown

#define SPARK_GPU_COPY_HOST_TO_DEVICE cudaMemcpyHostToDevice
#define SPARK_GPU_COPY_DEVICE_TO_HOST cudaMemcpyDeviceToHost
#define SPARK_GPU_COPY_DEVICE_TO_DEVICE cudaMemcpyDeviceToDevice
#define SPARK_GPU_COPY_DEFAULT cudaMemcpyDefault

#define SPARK_GPU_STREAM_NON_BLOCKING cudaStreamNonBlocking
#define SPARK_GPU_EVENT_DISABLE_TIMING cudaEventDisableTiming
#define SPARK_GPU_HOST_ALLOC_DEFAULT cudaHostAllocDefault
#define SPARK_GPU_HOST_ALLOC_PORTABLE cudaHostAllocPortable
#define SPARK_GPU_HOST_ALLOC_MAPPED cudaHostAllocMapped
#define SPARK_GPU_HOST_REGISTER_DEFAULT cudaHostRegisterDefault
#define SPARK_GPU_CAPTURE_STATUS_NONE cudaStreamCaptureStatusNone
#define SPARK_GPU_CAPTURE_MODE_THREAD_LOCAL cudaStreamCaptureModeThreadLocal

#define SPARK_GPU_ATTRIBUTE_MULTIPROCESSOR_COUNT cudaDevAttrMultiProcessorCount
#define SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS cudaDevAttrPageableMemoryAccess
#define SPARK_GPU_ATTRIBUTE_PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES cudaDevAttrPageableMemoryAccessUsesHostPageTables
#define SPARK_GPU_ATTRIBUTE_HOST_POINTER_FOR_REGISTERED_MEMORY cudaDevAttrCanUseHostPointerForRegisteredMem
#define SPARK_GPU_FUNCTION_ATTRIBUTE_MAX_DYNAMIC_SHARED_BYTES cudaFuncAttributeMaxDynamicSharedMemorySize

static inline SparkGpuError SparkGpuGetLastError(void)
{
	return cudaGetLastError();
}

static inline SparkGpuError SparkGpuPeekAtLastError(void)
{
	return cudaPeekAtLastError();
}

static inline const char *SparkGpuGetErrorString(SparkGpuError error)
{
	return cudaGetErrorString(error);
}

static inline SparkGpuError SparkGpuGetDeviceCount(int *count)
{
	return cudaGetDeviceCount(count);
}

static inline SparkGpuError SparkGpuGetDevice(int *device)
{
	return cudaGetDevice(device);
}

static inline SparkGpuError SparkGpuSetDevice(int device)
{
	return cudaSetDevice(device);
}

static inline SparkGpuError SparkGpuGetDeviceProperties(SparkGpuDeviceProperties *properties, int device)
{
	return cudaGetDeviceProperties(properties, device);
}

static inline SparkGpuError SparkGpuDeviceGetAttribute(int *value, SparkGpuDeviceAttribute attribute, int device)
{
	return cudaDeviceGetAttribute(value, attribute, device);
}

static inline SparkGpuError SparkGpuDeviceGetStreamPriorityRange(int *least_priority, int *greatest_priority)
{
	return cudaDeviceGetStreamPriorityRange(least_priority, greatest_priority);
}

static inline SparkGpuError SparkGpuDeviceSynchronize(void)
{
	return cudaDeviceSynchronize();
}

static inline SparkGpuError SparkGpuMemGetInfo(size_t *free_bytes, size_t *total_bytes)
{
	return cudaMemGetInfo(free_bytes, total_bytes);
}

static inline SparkGpuError SparkGpuMalloc(void **pointer, size_t bytes)
{
	return cudaMalloc(pointer, bytes);
}

static inline SparkGpuError SparkGpuMallocManaged(void **pointer, size_t bytes, unsigned int flags SPARK_GPU_DEFAULT(cudaMemAttachGlobal))
{
	return cudaMallocManaged(pointer, bytes, flags);
}

static inline SparkGpuError SparkGpuMallocHost(void **pointer, size_t bytes)
{
	return cudaMallocHost(pointer, bytes);
}

static inline SparkGpuError SparkGpuHostAlloc(void **pointer, size_t bytes, unsigned int flags)
{
	return cudaHostAlloc(pointer, bytes, flags);
}

static inline SparkGpuError SparkGpuHostGetDevicePointer(void **device_pointer, void *host_pointer, unsigned int flags)
{
	return cudaHostGetDevicePointer(device_pointer, host_pointer, flags);
}

static inline SparkGpuError SparkGpuHostRegister(void *host_pointer, size_t bytes, unsigned int flags)
{
	return cudaHostRegister(host_pointer, bytes, flags);
}

static inline SparkGpuError SparkGpuHostUnregister(void *host_pointer)
{
	return cudaHostUnregister(host_pointer);
}

static inline SparkGpuError SparkGpuFree(void *pointer)
{
	return cudaFree(pointer);
}

static inline SparkGpuError SparkGpuFreeHost(void *pointer)
{
	return cudaFreeHost(pointer);
}

static inline SparkGpuError SparkGpuMemcpy(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind)
{
	return cudaMemcpy(destination, source, bytes, kind);
}

static inline SparkGpuError SparkGpuMemcpyAsync(void *destination, const void *source, size_t bytes, SparkGpuCopyKind kind,
	SparkGpuStream stream SPARK_GPU_DEFAULT(0))
{
	return cudaMemcpyAsync(destination, source, bytes, kind, stream);
}

static inline SparkGpuError SparkGpuMemcpy2DAsync(void *destination, size_t destination_pitch, const void *source, size_t source_pitch,
	size_t width, size_t height, SparkGpuCopyKind kind, SparkGpuStream stream SPARK_GPU_DEFAULT(0))
{
	return cudaMemcpy2DAsync(destination, destination_pitch, source, source_pitch, width, height, kind, stream);
}

static inline SparkGpuError SparkGpuMemset(void *destination, int value, size_t bytes)
{
	return cudaMemset(destination, value, bytes);
}

static inline SparkGpuError SparkGpuMemsetAsync(void *destination, int value, size_t bytes, SparkGpuStream stream SPARK_GPU_DEFAULT(0))
{
	return cudaMemsetAsync(destination, value, bytes, stream);
}

static inline SparkGpuError SparkGpuMemset2DAsync(void *destination, size_t pitch, int value, size_t width, size_t height,
	SparkGpuStream stream SPARK_GPU_DEFAULT(0))
{
	return cudaMemset2DAsync(destination, pitch, value, width, height, stream);
}

static inline SparkGpuError SparkGpuStreamCreateWithFlags(SparkGpuStream *stream, unsigned int flags)
{
	return cudaStreamCreateWithFlags(stream, flags);
}

static inline SparkGpuError SparkGpuStreamCreateWithPriority(SparkGpuStream *stream, unsigned int flags, int priority)
{
	return cudaStreamCreateWithPriority(stream, flags, priority);
}

static inline SparkGpuError SparkGpuStreamDestroy(SparkGpuStream stream)
{
	return cudaStreamDestroy(stream);
}

static inline SparkGpuError SparkGpuStreamSynchronize(SparkGpuStream stream)
{
	return cudaStreamSynchronize(stream);
}

static inline SparkGpuError SparkGpuStreamQuery(SparkGpuStream stream)
{
	return cudaStreamQuery(stream);
}

static inline SparkGpuError SparkGpuStreamWaitEvent(SparkGpuStream stream, SparkGpuEvent event, unsigned int flags SPARK_GPU_DEFAULT(0))
{
	return cudaStreamWaitEvent(stream, event, flags);
}

static inline SparkGpuError SparkGpuStreamIsCapturing(SparkGpuStream stream, SparkGpuCaptureStatus *status)
{
	return cudaStreamIsCapturing(stream, status);
}

static inline SparkGpuError SparkGpuStreamBeginCapture(SparkGpuStream stream, SparkGpuCaptureMode mode)
{
	return cudaStreamBeginCapture(stream, mode);
}

static inline SparkGpuError SparkGpuStreamEndCapture(SparkGpuStream stream, SparkGpuGraph *graph)
{
	return cudaStreamEndCapture(stream, graph);
}

static inline SparkGpuError SparkGpuLaunchHostFunc(SparkGpuStream stream, SparkGpuHostFunction function, void *context)
{
	return cudaLaunchHostFunc(stream, function, context);
}

static inline SparkGpuError SparkGpuEventCreate(SparkGpuEvent *event)
{
	return cudaEventCreate(event);
}

static inline SparkGpuError SparkGpuEventCreateWithFlags(SparkGpuEvent *event, unsigned int flags)
{
	return cudaEventCreateWithFlags(event, flags);
}

static inline SparkGpuError SparkGpuEventDestroy(SparkGpuEvent event)
{
	return cudaEventDestroy(event);
}

static inline SparkGpuError SparkGpuEventRecord(SparkGpuEvent event, SparkGpuStream stream SPARK_GPU_DEFAULT(0))
{
	return cudaEventRecord(event, stream);
}

static inline SparkGpuError SparkGpuEventQuery(SparkGpuEvent event)
{
	return cudaEventQuery(event);
}

static inline SparkGpuError SparkGpuEventSynchronize(SparkGpuEvent event)
{
	return cudaEventSynchronize(event);
}

static inline SparkGpuError SparkGpuEventElapsedTime(float *milliseconds, SparkGpuEvent start, SparkGpuEvent stop)
{
	return cudaEventElapsedTime(milliseconds, start, stop);
}

static inline SparkGpuError SparkGpuGraphInstantiate(SparkGpuGraphExec *exec, SparkGpuGraph graph, unsigned long long flags SPARK_GPU_DEFAULT(0))
{
	return cudaGraphInstantiate(exec, graph, flags);
}

static inline SparkGpuError SparkGpuGraphUpload(SparkGpuGraphExec exec, SparkGpuStream stream)
{
	return cudaGraphUpload(exec, stream);
}

static inline SparkGpuError SparkGpuGraphLaunch(SparkGpuGraphExec exec, SparkGpuStream stream)
{
	return cudaGraphLaunch(exec, stream);
}

static inline SparkGpuError SparkGpuGraphGetNodes(SparkGpuGraph graph, SparkGpuGraphNode *nodes, size_t *count)
{
	return cudaGraphGetNodes(graph, nodes, count);
}

static inline SparkGpuError SparkGpuGraphExecDestroy(SparkGpuGraphExec exec)
{
	return cudaGraphExecDestroy(exec);
}

static inline SparkGpuError SparkGpuGraphDestroy(SparkGpuGraph graph)
{
	return cudaGraphDestroy(graph);
}

static inline SparkGpuError SparkGpuFuncSetAttribute(const void *function, SparkGpuFunctionAttribute attribute, int value)
{
	return cudaFuncSetAttribute(function, attribute, value);
}

static inline SparkGpuError SparkGpuFuncGetAttributes(SparkGpuFunctionAttributes *attributes, const void *function)
{
	return cudaFuncGetAttributes(attributes, function);
}

static inline SparkGpuError SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, const void *function, int block_threads, size_t shared_bytes)
{
	return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, function, block_threads, shared_bytes);
}

#ifdef __cplusplus
static inline SparkGpuError SparkGpuEventCreate(SparkGpuEvent *event, unsigned int flags)
{
	return cudaEventCreateWithFlags(event, flags);
}

template<typename T>
static inline SparkGpuError SparkGpuMalloc(T **pointer, size_t bytes)
{
	return cudaMalloc((void **)pointer, bytes);
}

template<typename T>
static inline SparkGpuError SparkGpuMallocManaged(T **pointer, size_t bytes, unsigned int flags = cudaMemAttachGlobal)
{
	return cudaMallocManaged((void **)pointer, bytes, flags);
}

template<typename T>
static inline SparkGpuError SparkGpuMallocHost(T **pointer, size_t bytes)
{
	return cudaMallocHost((void **)pointer, bytes);
}

template<typename T>
static inline SparkGpuError SparkGpuHostAlloc(T **pointer, size_t bytes, unsigned int flags)
{
	return cudaHostAlloc((void **)pointer, bytes, flags);
}

template<typename T>
static inline SparkGpuError SparkGpuHostGetDevicePointer(T **device_pointer, void *host_pointer, unsigned int flags)
{
	return cudaHostGetDevicePointer((void **)device_pointer, host_pointer, flags);
}

template<typename T>
static inline SparkGpuError SparkGpuFuncSetAttribute(T *function, SparkGpuFunctionAttribute attribute, int value)
{
	return cudaFuncSetAttribute((const void *)function, attribute, value);
}

template<typename T>
static inline SparkGpuError SparkGpuFuncGetAttributes(SparkGpuFunctionAttributes *attributes, T *function)
{
	return cudaFuncGetAttributes(attributes, (const void *)function);
}

template<typename T>
static inline SparkGpuError SparkGpuOccupancyMaxActiveBlocksPerMultiprocessor(int *blocks, T function, int block_threads, size_t shared_bytes)
{
	return cudaOccupancyMaxActiveBlocksPerMultiprocessor(blocks, (const void *)function, block_threads, shared_bytes);
}
#endif

#endif
