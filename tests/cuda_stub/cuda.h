#pragma once

/* Minimal CUDA driver-API VMM surface (tests/cuda_stub). Builds without
 * CUDA_HOME compile runtime/spark_weightd.c against this header plus the
 * matching implementations in cuda_runtime_stub.c; GPU builds include the
 * real <cuda.h> from CUDA_HOME instead. Only the cuMem* virtual-memory-
 * management family the weightd daemon uses is declared - the names and
 * enum values mirror the real driver API so the same source compiles
 * against both. The POSIX shareable-handle type is the fd export the W2b
 * design stages for the consumer import+map tier; the stub declares it
 * alongside the rest so the surface is complete. */

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int CUresult;
typedef int CUdevice;
typedef unsigned long long CUdeviceptr;
typedef struct CUmemGenericAllocationHandle_st *CUmemGenericAllocationHandle;

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_INVALID_DEVICE 101

typedef enum CUmemAllocationType_enum
{
    CU_MEM_ALLOCATION_TYPE_INVALID = 0x0,
    CU_MEM_ALLOCATION_TYPE_PINNED = 0x1
} CUmemAllocationType;

typedef enum CUmemAllocationHandleType_enum
{
    CU_MEM_HANDLE_TYPE_NONE = 0x0,
    CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = 0x1
} CUmemAllocationHandleType;

typedef enum CUmemLocationType_enum
{
    CU_MEM_LOCATION_TYPE_INVALID = 0x0,
    CU_MEM_LOCATION_TYPE_DEVICE = 0x1
} CUmemLocationType;

typedef enum CUmemAccessFlags_enum
{
    CU_MEM_ACCESS_FLAGS_PROT_NONE = 0x0,
    CU_MEM_ACCESS_FLAGS_PROT_READ = 0x1,
    CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 0x3
} CUmemAccessFlags;

typedef enum CUmemAllocationGranularity_flags
{
    CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    CU_MEM_ALLOC_GRANULARITY_RECOMMENDED = 0x1
} CUmemAllocationGranularity_flags;

typedef struct CUmemLocation_st
{
    CUmemLocationType type;
    int id;
} CUmemLocation;

typedef struct CUmemAllocationProp_st
{
    CUmemAllocationType type;
    CUmemAllocationHandleType requestedHandleTypes;
    CUmemLocation location;
    struct
    {
        unsigned char compressType;
        unsigned char gpuDirectRDMACapable;
        unsigned short usage;
        unsigned int allocFlags;
    } allocFlags;
} CUmemAllocationProp;

typedef struct CUmemAccessDesc_st
{
    CUmemLocation location;
    CUmemAccessFlags flags;
} CUmemAccessDesc;

CUresult cuMemGetAllocationGranularity(size_t *granularity,
    const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags option);
CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
    size_t bytes,
    const CUmemAllocationProp *prop,
    unsigned long long flags);
CUresult cuMemAddressReserve(CUdeviceptr *pointer,
    size_t bytes,
    size_t alignment,
    CUdeviceptr address,
    unsigned long long flags);
CUresult cuMemMap(CUdeviceptr pointer,
    size_t bytes,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags);
CUresult cuMemSetAccess(CUdeviceptr pointer,
    size_t bytes,
    const CUmemAccessDesc *descriptors,
    size_t descriptor_count);
CUresult cuMemUnmap(CUdeviceptr pointer, size_t bytes);
CUresult cuMemRelease(CUmemGenericAllocationHandle handle);
CUresult cuMemAddressFree(CUdeviceptr pointer, size_t bytes);

#ifdef __cplusplus
}
#endif
