#include "cuda_runtime_api.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t cuda_capture_depth;

cudaError_t cudaMalloc(void **pointer, size_t bytes)
{
    if (pointer == 0 || bytes == 0u)
    {
        return cudaErrorMemoryAllocation;
    }
    *pointer = malloc(bytes);
    return *pointer != 0 ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaFree(void *pointer)
{
    free(pointer);
    return cudaSuccess;
}

cudaError_t cudaMemset(void *pointer, int value, size_t bytes)
{
    if (pointer == 0 && bytes != 0u)
    {
        return cudaErrorMemoryAllocation;
    }
    memset(pointer, value, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(
    void *pointer,
    int value,
    size_t bytes,
    cudaStream_t stream)
{
    (void)stream;
    return cudaMemset(pointer, value, bytes);
}

cudaError_t cudaMemcpy(
    void *destination,
    const void *source,
    size_t bytes,
    cudaMemcpyKind kind)
{
    (void)kind;
    if ((destination == 0 || source == 0) && bytes != 0u)
    {
        return cudaErrorMemoryAllocation;
    }
    memcpy(destination, source, bytes);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(
    void *destination,
    const void *source,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
    (void)stream;
    return cudaMemcpy(destination, source, bytes, kind);
}

cudaError_t cudaMemcpy2DAsync(
    void *destination,
    size_t destination_pitch,
    const void *source,
    size_t source_pitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream)
{
    size_t row_index;

    (void)kind;
    (void)stream;
    if ((destination == 0 || source == 0) && width != 0u && height != 0u)
    {
        return cudaErrorMemoryAllocation;
    }
    for (row_index = 0u; row_index < height; ++row_index)
    {
        memcpy(
            (unsigned char *)destination + row_index * destination_pitch,
            (const unsigned char *)source + row_index * source_pitch,
            width);
    }
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t *stream)
{
    if (stream == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    *stream = malloc(1u);
    return *stream != 0 ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaStreamCreateWithFlags(
    cudaStream_t *stream,
    unsigned int flags)
{
    (void)flags;
    return cudaStreamCreate(stream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    free(stream);
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags)
{
    (void)stream;
    (void)flags;
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaDeviceSynchronize(void)
{
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    return cudaEventCreateWithFlags(event,cudaEventDefault);
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    (void)flags;
    if (event == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    *event = malloc(1u);
    return *event != 0 ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    free(event);
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    (void)stream;
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaLaunchHostFunc(
    cudaStream_t stream,
    cudaHostFn_t function,
    void *user_data)
{
    (void)stream;
    if (function == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    if (cuda_capture_depth == 0u)
    {
        function(user_data);
    }
    return cudaSuccess;
}

cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode)
{
    (void)stream;
    (void)mode;
    cuda_capture_depth += 1u;
    return cudaSuccess;
}

cudaError_t cudaStreamEndCapture(
    cudaStream_t stream,
    cudaGraph_t *graph)
{
    (void)stream;
    if (cuda_capture_depth == 0u)
    {
        return cudaErrorInvalidValue;
    }
    cuda_capture_depth -= 1u;
    if (graph == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    *graph = malloc(1u);
    return *graph != 0 ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t *graph_exec,
    cudaGraph_t graph,
    ...)
{
    (void)graph;
    if (graph_exec == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    *graph_exec = malloc(1u);
    return *graph_exec != 0 ? cudaSuccess : cudaErrorMemoryAllocation;
}

cudaError_t cudaGraphLaunch(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream)
{
    (void)graph_exec;
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec)
{
    free(graph_exec);
    return cudaSuccess;
}

cudaError_t cudaGraphDestroy(cudaGraph_t graph)
{
    free(graph);
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(
    void **pointer,
    size_t bytes,
    unsigned int flags)
{
    (void)flags;
    return cudaMalloc(pointer, bytes);
}

cudaError_t cudaFreeHost(void *pointer)
{
    return cudaFree(pointer);
}

cudaError_t cudaHostGetDevicePointer(
    void **device_pointer,
    void *host_pointer,
    unsigned int flags)
{
    (void)flags;
    if (device_pointer == 0 || host_pointer == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    *device_pointer = host_pointer;
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error)
{
    return error == cudaSuccess ? "cudaSuccess" : "cudaTestError";
}
