#ifndef SPARKPIPE_TEST_CUDA_RUNTIME_API_H
#define SPARKPIPE_TEST_CUDA_RUNTIME_API_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaGraph_t;
typedef void *cudaGraphExec_t;

typedef enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

typedef enum cudaStreamCaptureMode
{
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
} cudaStreamCaptureMode;

#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaStreamDefault 0u
#define cudaStreamNonBlocking 1u
#define cudaHostAllocDefault 0u
#define cudaHostAllocPortable 1u
#define cudaHostAllocMapped 2u

cudaError_t cudaMalloc(void **pointer, size_t bytes);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemset(void *pointer, int value, size_t bytes);
cudaError_t cudaMemsetAsync(
    void *pointer,
    int value,
    size_t bytes,
    cudaStream_t stream);
cudaError_t cudaMemcpy(
    void *destination,
    const void *source,
    size_t bytes,
    cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(
    void *destination,
    const void *source,
    size_t bytes,
    cudaMemcpyKind kind,
    cudaStream_t stream);
cudaError_t cudaMemcpy2DAsync(
    void *destination,
    size_t destination_pitch,
    const void *source,
    size_t source_pitch,
    size_t width,
    size_t height,
    cudaMemcpyKind kind,
    cudaStream_t stream);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(
    cudaStream_t *stream,
    unsigned int flags);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaStreamBeginCapture(
    cudaStream_t stream,
    cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(
    cudaStream_t stream,
    cudaGraph_t *graph);
cudaError_t cudaGraphInstantiate(
    cudaGraphExec_t *graph_exec,
    cudaGraph_t graph,
    ...);
cudaError_t cudaGraphLaunch(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaHostAlloc(
    void **pointer,
    size_t bytes,
    unsigned int flags);
cudaError_t cudaFreeHost(void *pointer);
cudaError_t cudaHostGetDevicePointer(
    void **device_pointer,
    void *host_pointer,
    unsigned int flags);
const char *cudaGetErrorString(cudaError_t error);

#ifdef __cplusplus
}
#endif

#endif
