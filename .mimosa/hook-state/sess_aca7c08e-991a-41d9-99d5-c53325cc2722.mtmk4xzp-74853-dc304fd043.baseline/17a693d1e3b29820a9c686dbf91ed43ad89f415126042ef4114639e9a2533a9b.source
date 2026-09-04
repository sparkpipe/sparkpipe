#pragma once

#include_next <cuda_runtime_api.h>

#ifndef CUDART_VERSION
#define CUDART_VERSION 12000
#endif

typedef struct cudaPointerAttributes
{
    int type;
    int memoryType;
    int device;
    void *devicePointer;
    void *hostPointer;
} cudaPointerAttributes;

#define cudaMemoryTypeHost 1
#define cudaMemoryTypeDevice 2
#define cudaDevAttrGPUDirectRDMASupported 1
#define cudaDevAttrGPUDirectRDMAFlushWritesOptions 2
#define cudaDevAttrGPUDirectRDMAWritesOrdering 3
#define cudaFlushGPUDirectRDMAWritesOptionHost 1
#define cudaGPUDirectRDMAWritesOrderingOwner 1
#define cudaFlushGPUDirectRDMAWritesTargetCurrentDevice 0
#define cudaFlushGPUDirectRDMAWritesToOwner 0

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaDeviceGetAttribute(int *value, int attribute, int device);
cudaError_t cudaPointerGetAttributes(
    cudaPointerAttributes *attributes,
    const void *pointer);
cudaError_t cudaDeviceFlushGPUDirectRDMAWrites(int target, int scope);

#ifdef __cplusplus
}
#endif
