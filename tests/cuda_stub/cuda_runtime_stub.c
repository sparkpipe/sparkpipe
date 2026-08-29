#include "cuda_runtime_api.h"
#include "cuda.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t cuda_capture_depth;

/* Fault-injection and allocation-ledger hooks. The stub prefixes every live
 * allocation with a header so tests can (a) assert zero outstanding
 * allocations after a destroy path and (b) fail the Nth allocation-family
 * call to exercise partial-initialization cleanup. */
#define CUDA_STUB_ALLOC_MAGIC UINT32_C(0x53545542)
#define CUDA_STUB_MAX_TRACKED 4096

typedef struct cuda_stub_alloc_header
{
    uint32_t magic;
    uint32_t tracked;
} cuda_stub_alloc_header;

static void *cuda_stub_tracked[CUDA_STUB_MAX_TRACKED];
static uint32_t cuda_stub_tracked_count;
static uint32_t cuda_stub_alloc_calls;
static int32_t cuda_stub_fail_alloc_at = -1; /* 1-based call index, -1 = off */
static uint32_t cuda_stub_host_map_calls;
static int32_t cuda_stub_fail_host_map_at = -1;

static cudaError_t cuda_stub_alloc(void **pointer, size_t bytes)
{
    cuda_stub_alloc_header *header;
    cuda_stub_alloc_calls++;
    if (pointer == 0 || bytes == 0u || cuda_stub_alloc_calls ==
        (uint32_t)cuda_stub_fail_alloc_at)
    {
        return cudaErrorMemoryAllocation;
    }
    header = (cuda_stub_alloc_header *)malloc(bytes + sizeof(*header));
    if (header == 0)
    {
        return cudaErrorMemoryAllocation;
    }
    header->magic = CUDA_STUB_ALLOC_MAGIC;
    if (cuda_stub_tracked_count < CUDA_STUB_MAX_TRACKED)
    {
        header->tracked = 1u;
        cuda_stub_tracked[cuda_stub_tracked_count++] = header + 1;
    }
    else
    {
        header->tracked = 0u;
    }
    *pointer = header + 1;
    return cudaSuccess;
}

static cudaError_t cuda_stub_free(void *pointer)
{
    cuda_stub_alloc_header *header;
    uint32_t index;
    if (pointer == 0)
    {
        return cudaSuccess;
    }
    header = ((cuda_stub_alloc_header *)pointer) - 1;
    if (header->magic != CUDA_STUB_ALLOC_MAGIC)
    {
        return cudaErrorInvalidValue;
    }
    header->magic = 0u;
    if (header->tracked != 0u)
    {
        for (index = 0u; index < cuda_stub_tracked_count; index++)
        {
            if (cuda_stub_tracked[index] == pointer)
            {
                cuda_stub_tracked[index] =
                    cuda_stub_tracked[--cuda_stub_tracked_count];
                break;
            }
        }
    }
    free(header);
    return cudaSuccess;
}

void spark_stub_cuda_reset_faults(void)
{
    while (cuda_stub_tracked_count != 0u)
    {
        (void)cuda_stub_free(cuda_stub_tracked[0]);
    }
    cuda_stub_alloc_calls = 0u;
    cuda_stub_fail_alloc_at = -1;
    cuda_stub_host_map_calls = 0u;
    cuda_stub_fail_host_map_at = -1;
}

void spark_stub_cuda_fail_alloc_call(uint32_t one_based_call_index)
{
    cuda_stub_fail_alloc_at = (int32_t)one_based_call_index;
}

void spark_stub_cuda_fail_host_map_call(uint32_t one_based_call_index)
{
    cuda_stub_fail_host_map_at = (int32_t)one_based_call_index;
}

uint32_t spark_stub_cuda_outstanding_allocs(void)
{
    return cuda_stub_tracked_count;
}

cudaError_t cudaMalloc(void **pointer, size_t bytes)
{
    return cuda_stub_alloc(pointer, bytes);
}

cudaError_t cudaFree(void *pointer)
{
    return cuda_stub_free(pointer);
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
    return cuda_stub_alloc(pointer, bytes);
}

cudaError_t cudaFreeHost(void *pointer)
{
    return cuda_stub_free(pointer);
}

cudaError_t cudaHostGetDevicePointer(
    void **device_pointer,
    void *host_pointer,
    unsigned int flags)
{
    (void)flags;
    cuda_stub_host_map_calls++;
    if (device_pointer == 0 || host_pointer == 0 ||
        cuda_stub_host_map_calls == (uint32_t)cuda_stub_fail_host_map_at)
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

cudaError_t cudaGetDevice(int *device)
{
    if (device == 0)
    {
        return cudaErrorInvalidValue;
    }
    *device = 0;
    return cudaSuccess;
}

/* ------------------------------ cuMem* VMM ------------------------------ */

/* The VMM stand-in models the real physical/virtual split closely enough to
 * catch daemon lifecycle bugs: a physical chunk is a small tracked record
 * (cuMemCreate), a virtual reservation is a tracked real allocation sized to
 * the full span so mapped reads and writes touch real bytes (cuMemAddress-
 * Reserve), and cuMemMap/cuMemUnmap enforce exact (pointer, bytes) pairs per
 * chunk with cuMemRelease/cuMemAddressFree refusing while anything is still
 * mapped - the same orderings the real driver enforces. Both records go
 * through the shared tracked allocator, so the tests' zero-outstanding-
 * allocations check covers the VMM path too. */

#define CUDA_STUB_VMM_MAGIC UINT32_C(0x564D4D31) /* "VMM1" */
#define CUDA_STUB_VMM_MAPPED_MAX 64

typedef struct cuda_stub_vmm_phys
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    CUmemAllocationProp prop;
} cuda_stub_vmm_phys;

typedef struct cuda_stub_vmm_reservation
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    uint64_t mapped_bytes;
    struct
    {
        CUdeviceptr pointer;
        size_t bytes;
        cuda_stub_vmm_phys *phys;
    } mappings[CUDA_STUB_VMM_MAPPED_MAX];
} cuda_stub_vmm_reservation;

static cuda_stub_vmm_reservation *cuda_stub_vmm_reservation_at(
    void *user_pointer)
{
    cuda_stub_alloc_header *header =
        ((cuda_stub_alloc_header *)user_pointer) - 1;
    cuda_stub_vmm_reservation *reservation;
    if (header->magic != CUDA_STUB_ALLOC_MAGIC)
    {
        return 0;
    }
    reservation = (cuda_stub_vmm_reservation *)user_pointer;
    return reservation->magic == CUDA_STUB_VMM_MAGIC ? reservation : 0;
}

/* The reservation owning a mapped virtual address: the VA handed out is the
 * tail of the tracked allocation (struct end), so walk the tracked headers
 * and match struct+1. */
static cuda_stub_vmm_reservation *cuda_stub_vmm_reservation_for_va(
    CUdeviceptr pointer)
{
    uint32_t index;
    for (index = 0u; index < cuda_stub_tracked_count; index++)
    {
        cuda_stub_vmm_reservation *reservation = cuda_stub_vmm_reservation_at(
            cuda_stub_tracked[index]);
        if (reservation != 0 && (CUdeviceptr)(reservation + 1) == pointer)
        {
            return reservation;
        }
    }
    return 0;
}

CUresult cuMemGetAllocationGranularity(size_t *granularity,
    const CUmemAllocationProp *prop,
    CUmemAllocationGranularity_flags option)
{
    if (granularity == 0 || prop == 0 ||
        prop->type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *granularity = option == CU_MEM_ALLOC_GRANULARITY_MINIMUM
        ? (size_t)65536
        : (size_t)(2ull * 1024ull * 1024ull); /* the 2 MiB page law */
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
    size_t bytes,
    const CUmemAllocationProp *prop,
    unsigned long long flags)
{
    cuda_stub_vmm_phys *phys;
    if (handle == 0 || prop == 0 ||
        prop->type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        flags != 0ull || bytes == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    /* one tracked allocation per physical chunk; the alloc-call family fault
     * index (cudaMalloc + cudaHostAlloc + cuMemCreate) injects here too */
    if (cuda_stub_alloc((void **)&phys, sizeof(*phys)) != cudaSuccess)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    phys->magic = CUDA_STUB_VMM_MAGIC;
    phys->mapped_count = 0u;
    phys->bytes = (uint64_t)bytes;
    phys->prop = *prop;
    *handle = (CUmemGenericAllocationHandle)phys;
    return CUDA_SUCCESS;
}

CUresult cuMemAddressReserve(CUdeviceptr *pointer,
    size_t bytes,
    size_t alignment,
    CUdeviceptr address,
    unsigned long long flags)
{
    cuda_stub_vmm_reservation *reservation;
    if (pointer == 0 || bytes == 0u || alignment > (size_t)4096u ||
        address != 0ull || flags != 0ull)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (cuda_stub_alloc((void **)&reservation,
            sizeof(*reservation) + bytes) != cudaSuccess)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    memset(reservation, 0, sizeof(*reservation) + bytes);
    reservation->magic = CUDA_STUB_VMM_MAGIC;
    reservation->bytes = (uint64_t)bytes;
    *pointer = (CUdeviceptr)(reservation + 1);
    return CUDA_SUCCESS;
}

CUresult cuMemMap(CUdeviceptr pointer,
    size_t bytes,
    size_t offset,
    CUmemGenericAllocationHandle handle,
    unsigned long long flags)
{
    cuda_stub_vmm_reservation *reservation;
    cuda_stub_vmm_phys *phys = (cuda_stub_vmm_phys *)handle;
    if (phys == 0 || ((cuda_stub_alloc_header *)phys - 1)->magic !=
                         CUDA_STUB_ALLOC_MAGIC ||
        phys->magic != CUDA_STUB_VMM_MAGIC || offset != (size_t)0u ||
        flags != 0ull || bytes == 0u || (uint64_t)bytes != phys->bytes)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation = cuda_stub_vmm_reservation_for_va(pointer);
    if (reservation == 0 ||
        (uint64_t)bytes > reservation->bytes - reservation->mapped_bytes)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (reservation->mapped_count >= CUDA_STUB_VMM_MAPPED_MAX)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    reservation->mappings[reservation->mapped_count].pointer = pointer;
    reservation->mappings[reservation->mapped_count].bytes = bytes;
    reservation->mappings[reservation->mapped_count].phys = phys;
    reservation->mapped_count++;
    reservation->mapped_bytes += (uint64_t)bytes;
    phys->mapped_count++;
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr pointer,
    size_t bytes,
    const CUmemAccessDesc *descriptors,
    size_t descriptor_count)
{
    cuda_stub_vmm_reservation *reservation;
    if (descriptors == 0 || descriptor_count != (size_t)1u ||
        descriptors[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        (descriptors[0].flags & ~CU_MEM_ACCESS_FLAGS_PROT_READWRITE) != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation = cuda_stub_vmm_reservation_for_va(pointer);
    if (reservation != 0 && reservation->bytes == (uint64_t)bytes)
    {
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemUnmap(CUdeviceptr pointer, size_t bytes)
{
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    uint32_t mapping_index;
    if (reservation != 0)
    {
        for (mapping_index = 0u; mapping_index < reservation->mapped_count;
            mapping_index++)
        {
            if (reservation->mappings[mapping_index].pointer == pointer &&
                reservation->mappings[mapping_index].bytes == bytes)
            {
                reservation->mappings[mapping_index].phys->mapped_count--;
                reservation->mapped_bytes -= (uint64_t)bytes;
                reservation->mappings[mapping_index] =
                    reservation->mappings[reservation->mapped_count - 1u];
                reservation->mapped_count--;
                return CUDA_SUCCESS;
            }
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}

CUresult cuMemRelease(CUmemGenericAllocationHandle handle)
{
    cuda_stub_vmm_phys *phys = (cuda_stub_vmm_phys *)handle;
    if (phys == 0 || ((cuda_stub_alloc_header *)phys - 1)->magic !=
                         CUDA_STUB_ALLOC_MAGIC ||
        phys->magic != CUDA_STUB_VMM_MAGIC)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (phys->mapped_count != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE; /* still mapped somewhere */
    }
    return cuda_stub_free(phys);
}

CUresult cuMemAddressFree(CUdeviceptr pointer, size_t bytes)
{
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    (void)bytes;
    if (reservation == 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (reservation->mapped_count != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE; /* mappings still live */
    }
    return cuda_stub_free(reservation);
}

