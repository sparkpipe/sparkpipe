#pragma once

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
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
#define cudaErrorInvalidValue 1
#define cudaErrorMemoryAllocation 2
#define cudaErrorNotReady 600
#define cudaErrorHostMemoryAlreadyRegistered 712
#define cudaErrorHostMemoryNotRegistered 713
#define cudaErrorLaunchFailure 719
#define cudaErrorUnknown 999
#define cudaDevAttrMultiProcessorCount 16
#define cudaStreamDefault 0u
#define cudaStreamNonBlocking 1u
#define cudaEventDefault 0u
#define cudaEventDisableTiming 2u
#define cudaHostAllocDefault 0u
#define cudaHostAllocPortable 1u
#define cudaHostAllocMapped 2u
#define CUDART_CB
typedef void (*cudaHostFn_t)(void *user_data);

cudaError_t cudaMalloc(void **pointer, size_t bytes);
cudaError_t cudaHostAlloc(void **pointer, size_t bytes, unsigned int flags);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
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
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(
    cudaStream_t stream,
    cudaEvent_t event,
    unsigned int flags);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceGetAttribute(int *value, int attribute, int device);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start,
    cudaEvent_t stop);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaHostRegister(
    void *address,
    size_t bytes,
    unsigned int flags);
cudaError_t cudaHostUnregister(void *address);
cudaError_t cudaLaunchHostFunc(
    cudaStream_t stream,
    cudaHostFn_t function,
    void *user_data);
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
cudaError_t cudaGraphUpload(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream);
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
typedef void *cudaGraphNode_t;
typedef void *cudaArray_t;
typedef struct CUfunc_st *cudaFunction_t;

typedef enum cudaGraphNodeType
{
    cudaGraphNodeTypeKernel = 0x00,
    cudaGraphNodeTypeMemcpy = 0x01,
    cudaGraphNodeTypeMemset = 0x02,
    cudaGraphNodeTypeHost = 0x03,
    cudaGraphNodeTypeGraph = 0x04,
    cudaGraphNodeTypeEmpty = 0x05,
    cudaGraphNodeTypeWaitEvent = 0x06,
    cudaGraphNodeTypeEventRecord = 0x07,
    cudaGraphNodeTypeMemAlloc = 0x0a,
    cudaGraphNodeTypeMemFree = 0x0b
} cudaGraphNodeType;

typedef enum cudaMemoryType
{
    cudaMemoryTypeUnregistered = 0,
    cudaMemoryTypeHost = 1,
    cudaMemoryTypeDevice = 2,
    cudaMemoryTypeManaged = 3
} cudaMemoryType;

typedef struct dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
} dim3;

struct cudaKernelNodeParams
{
    void *func;
    dim3 gridDim;
    dim3 blockDim;
    unsigned int sharedMemBytes;
    void **kernelParams;
    void **extra;
};

struct cudaPos
{
    size_t x;
    size_t y;
    size_t z;
};

struct cudaPitchedPtr
{
    void *ptr;
    size_t pitch;
    size_t xsize;
    size_t ysize;
};

struct cudaExtent
{
    size_t width;
    size_t height;
    size_t depth;
};

struct cudaMemcpy3DParms
{
    cudaArray_t srcArray;
    struct cudaPos srcPos;
    struct cudaPitchedPtr srcPtr;
    cudaArray_t dstArray;
    struct cudaPos dstPos;
    struct cudaPitchedPtr dstPtr;
    struct cudaExtent extent;
    enum cudaMemcpyKind kind;
};

struct cudaMemsetParams
{
    void *dst;
    size_t pitch;
    unsigned int value;
    unsigned int elementSize;
    size_t width;
    size_t height;
};

struct cudaHostNodeParams
{
    cudaHostFn_t fn;
    void *userData;
};

struct cudaPointerAttributes
{
    enum cudaMemoryType type;
    int device;
    void *devicePointer;
    void *hostPointer;
};

cudaError_t cudaGraphGetNodes(cudaGraph_t graph,cudaGraphNode_t *nodes,size_t *count);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node,enum cudaGraphNodeType *type);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node,struct cudaKernelNodeParams *params);
cudaError_t cudaGraphMemcpyNodeGetParams(cudaGraphNode_t node,struct cudaMemcpy3DParms *params);
cudaError_t cudaGraphMemsetNodeGetParams(cudaGraphNode_t node,struct cudaMemsetParams *params);
cudaError_t cudaGraphHostNodeGetParams(cudaGraphNode_t node,struct cudaHostNodeParams *params);
cudaError_t cudaGraphEventRecordNodeGetEvent(cudaGraphNode_t node,cudaEvent_t *event);
cudaError_t cudaGraphEventWaitNodeGetEvent(cudaGraphNode_t node,cudaEvent_t *event);
cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaKernelNodeParams *params);
cudaError_t cudaGraphExecMemcpyNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaMemcpy3DParms *params);
cudaError_t cudaGraphExecMemsetNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaMemsetParams *params);
cudaError_t cudaGraphExecHostNodeSetParams(cudaGraphExec_t exec,cudaGraphNode_t node,const struct cudaHostNodeParams *params);
cudaError_t cudaGetFuncBySymbol(cudaFunction_t *function,const void *symbol);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes *attributes,const void *pointer);
cudaError_t cudaGetLastError(void);
#include <stdint.h>

void spark_stub_cuda_reset_faults(void);
typedef void (*spark_stub_cuda_copy_hook_t)(void *context, uint32_t asynchronous, cudaMemcpyKind kind, cudaStream_t stream);
void spark_stub_cuda_set_copy_hook(spark_stub_cuda_copy_hook_t hook, void *context);
uint32_t spark_stub_cuda_sync_copy_calls(void);
uint32_t spark_stub_cuda_async_copy_calls(void);
void spark_stub_cuda_reset_copy_calls(void);
uint32_t spark_stub_cuda_stream_wait_calls(void);
cudaStream_t spark_stub_cuda_last_wait_stream(void);
cudaEvent_t spark_stub_cuda_last_wait_event(void);
unsigned int spark_stub_cuda_stream_flags(cudaStream_t stream);
void spark_stub_cuda_event_pending(uint32_t pending);
void spark_stub_cuda_event_record_failure(uint32_t failure);
extern uint32_t cuda_stub_stream_sync_calls;
void spark_stub_cuda_fail_alloc_call(uint32_t one_based_call_index);
void spark_stub_cuda_fail_host_map_call(uint32_t one_based_call_index);
uint32_t spark_stub_cuda_outstanding_allocs(void);

#ifdef __cplusplus
}

#endif
