#ifndef SPARKPIPE_HARDWARE_CUDA_STUB_RUNTIME_H
#define SPARKPIPE_HARDWARE_CUDA_STUB_RUNTIME_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#ifndef __host__
#define __host__ __attribute__((host))
#endif
#ifndef __device__
#define __device__ __attribute__((device))
#endif
#ifndef __global__
#define __global__ __attribute__((global))
#endif
#ifndef __shared__
#define __shared__ __attribute__((shared))
#endif
#ifndef __constant__
#define __constant__ __attribute__((constant))
#endif
#ifndef __forceinline__
#define __forceinline__ inline __attribute__((always_inline))
#endif
#ifndef CUDART_CB
#define CUDART_CB
#endif

struct __attribute__((device_builtin)) uint3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct __attribute__((device_builtin)) dim3
{
    unsigned int x;
    unsigned int y;
    unsigned int z;

    __host__ __device__ constexpr dim3(
        unsigned int x_value = 1u,
        unsigned int y_value = 1u,
        unsigned int z_value = 1u)
        : x(x_value), y(y_value), z(z_value)
    {
    }

    __host__ __device__ constexpr dim3(uint3 value)
        : x(value.x), y(value.y), z(value.z)
    {
    }

    __host__ __device__ constexpr operator uint3() const
    {
        return uint3{x, y, z};
    }
};

struct uint4
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
};

__host__ __device__ inline uint4 make_uint4(
    unsigned int x,
    unsigned int y,
    unsigned int z,
    unsigned int w)
{
    return uint4{x, y, z, w};
}

extern __device__ const uint3 threadIdx;
extern __device__ const uint3 blockIdx;
extern __device__ const dim3 blockDim;
extern __device__ const dim3 gridDim;
extern __device__ const int warpSize;

template <typename ValueType>
__device__ inline ValueType __shfl_down_sync(
    unsigned int,
    ValueType value,
    unsigned int,
    int = 32)
{
    return value;
}

template <typename ValueType>
__device__ inline ValueType atomicAdd(ValueType *address, ValueType value)
{
    ValueType previous = *address;
    *address += value;
    return previous;
}

__device__ inline unsigned long long atomicXor(
    unsigned long long *address,
    unsigned long long value)
{
    unsigned long long previous = *address;
    *address ^= value;
    return previous;
}

typedef int cudaError_t;
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef void *cudaGraph_t;
typedef void *cudaGraphExec_t;
typedef void (*cudaHostFn_t)(void *);

enum cudaMemcpyKind
{
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
};

enum cudaStreamCaptureMode
{
    cudaStreamCaptureModeGlobal = 0,
    cudaStreamCaptureModeThreadLocal = 1,
    cudaStreamCaptureModeRelaxed = 2
};

enum cudaFuncAttribute
{
    cudaFuncAttributeMaxDynamicSharedMemorySize = 8
};

enum cudaDeviceAttribute
{
    cudaDevAttrMemoryClockRate = 36
};

struct cudaDeviceProp
{
    char name[256];
    std::size_t totalGlobalMem;
    int major;
    int minor;
    int multiProcessorCount;
    int memoryBusWidth;
    int maxThreadsPerMultiProcessor;
};

#define cudaSuccess 0
#define cudaErrorMemoryAllocation 2
#define cudaStreamDefault 0u
#define cudaStreamNonBlocking 1u
#define cudaEventDefault 0u
#define cudaEventDisableTiming 2u
#define cudaHostAllocDefault 0u
#define cudaHostAllocPortable 1u
#define cudaHostAllocMapped 2u
#define cudaHostRegisterDefault 0u
#define cudaHostRegisterPortable 1u
#define cudaDeviceMapHost 8u

extern "C" {
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *properties, int device);
cudaError_t cudaDeviceGetAttribute(int *value, int attribute, int device);
cudaError_t cudaDriverGetVersion(int *version);
cudaError_t cudaRuntimeGetVersion(int *version);
cudaError_t cudaMalloc(void **pointer, std::size_t bytes);
cudaError_t cudaFree(void *pointer);
cudaError_t cudaMemset(void *pointer, int value, std::size_t bytes);
cudaError_t cudaMemsetAsync(void *pointer, int value, std::size_t bytes, cudaStream_t stream);
cudaError_t cudaMemcpy(void *destination, const void *source, std::size_t bytes, cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void *destination, const void *source, std::size_t bytes, cudaMemcpyKind kind, cudaStream_t stream);
cudaError_t cudaHostAlloc(void **pointer, std::size_t bytes, unsigned int flags);
cudaError_t cudaFreeHost(void *pointer);
cudaError_t cudaHostGetDevicePointer(void **device_pointer, void *host_pointer, unsigned int flags);
cudaError_t cudaHostRegister(void *pointer, std::size_t bytes, unsigned int flags);
cudaError_t cudaHostUnregister(void *pointer);
cudaError_t cudaStreamCreate(cudaStream_t *stream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *stream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *stream, unsigned int flags, int priority);
cudaError_t cudaDeviceGetStreamPriorityRange(int *least_priority, int *greatest_priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start, cudaEvent_t finish);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t function, void *user_data);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, cudaStreamCaptureMode mode);
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t *graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *graph_exec, cudaGraph_t graph, ...);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graph_exec, cudaStream_t stream);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graph_exec);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGetLastError(void);
const char *cudaGetErrorString(cudaError_t error);

cudaError_t cudaConfigureCall(dim3 grid, dim3 block, std::size_t shared = 0u, cudaStream_t stream = nullptr);
cudaError_t cudaSetupArgument(const void *argument, std::size_t size, std::size_t offset);
cudaError_t cudaLaunch(const void *function);
}

template <typename PointerType>
inline cudaError_t cudaMalloc(PointerType **pointer, std::size_t bytes)
{
    return cudaMalloc(reinterpret_cast<void **>(pointer), bytes);
}

template <typename PointerType>
inline cudaError_t cudaHostAlloc(PointerType **pointer, std::size_t bytes, unsigned int flags)
{
    return cudaHostAlloc(reinterpret_cast<void **>(pointer), bytes, flags);
}

template <typename PointerType>
inline cudaError_t cudaHostGetDevicePointer(
    PointerType **device_pointer,
    void *host_pointer,
    unsigned int flags)
{
    return cudaHostGetDevicePointer(
        reinterpret_cast<void **>(device_pointer),
        host_pointer,
        flags);
}

template <typename FunctionType>
inline cudaError_t cudaFuncSetAttribute(
    FunctionType,
    cudaFuncAttribute,
    int)
{
    return cudaSuccess;
}

template <typename FunctionType>
inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    int *active_blocks,
    FunctionType,
    int,
    std::size_t)
{
    *active_blocks = 1;
    return cudaSuccess;
}

#endif
