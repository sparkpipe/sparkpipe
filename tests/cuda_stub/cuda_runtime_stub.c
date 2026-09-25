#include "cuda_runtime_api.h"
#include "cuda.h"
#include "sparkpipe/spark_tp_mesh_round_control.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static uint32_t cuda_capture_depth;

#define CUDA_STUB_ALLOC_MAGIC UINT32_C(0x53545542)
#define CUDA_STUB_MAX_TRACKED 4096

typedef struct cuda_stub_alloc_header
{
    uint32_t magic;
    uint32_t tracked;
    uint64_t bytes;
} cuda_stub_alloc_header;

static void *cuda_stub_tracked[CUDA_STUB_MAX_TRACKED];
static uint32_t cuda_stub_tracked_count;
static uint32_t cuda_stub_alloc_calls;
static int32_t cuda_stub_fail_alloc_at = -1;
static uint32_t cuda_stub_host_map_calls;
static int32_t cuda_stub_fail_host_map_at = -1;
static uint32_t cuda_stub_export_calls;
static uint32_t cuda_stub_fail_export_at;
static pthread_mutex_t cuda_stub_ledger_mutex = PTHREAD_MUTEX_INITIALIZER;

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    return ctx != 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    static uint8_t context;
    if (pctx == 0)
        return CUDA_ERROR_INVALID_VALUE;
    *pctx = (CUcontext)&context;
    return CUDA_SUCCESS;
}

static void cuda_stub_ledger_lock(void)
{
    (void)pthread_mutex_lock(&cuda_stub_ledger_mutex);
}

static void cuda_stub_ledger_unlock(void)
{
    (void)pthread_mutex_unlock(&cuda_stub_ledger_mutex);
}

static cudaError_t cuda_stub_alloc(void **pointer, size_t bytes)
{
    cuda_stub_alloc_header *header;
    cudaError_t result;
    cuda_stub_ledger_lock();
    cuda_stub_alloc_calls++;
    if (pointer == 0 || bytes == 0u || cuda_stub_alloc_calls ==
        (uint32_t)cuda_stub_fail_alloc_at)
    {
        cuda_stub_ledger_unlock();
        return cudaErrorMemoryAllocation;
    }
    header = (cuda_stub_alloc_header *)malloc(bytes + sizeof(*header));
    if (header == 0)
    {
        cuda_stub_ledger_unlock();
        return cudaErrorMemoryAllocation;
    }
    header->magic = CUDA_STUB_ALLOC_MAGIC;
    header->bytes = bytes;
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
    result = cudaSuccess;
    cuda_stub_ledger_unlock();
    return result;
}

static cudaError_t cuda_stub_free(void *pointer)
{
    cuda_stub_alloc_header *header;
    uint32_t index;
    cudaError_t result;
    if (pointer == 0)
    {
        return cudaSuccess;
    }
    cuda_stub_ledger_lock();
    header = ((cuda_stub_alloc_header *)pointer) - 1;
    if (header->magic != CUDA_STUB_ALLOC_MAGIC)
    {
        cuda_stub_ledger_unlock();
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
    result = cudaSuccess;
    cuda_stub_ledger_unlock();
    free(header);
    return result;
}

void spark_stub_cuda_reset_faults(void)
{
    cuda_stub_ledger_lock();
    while (cuda_stub_tracked_count != 0u)
    {
        void *pointer = cuda_stub_tracked[0];
        cuda_stub_alloc_header *header =
            ((cuda_stub_alloc_header *)pointer) - 1;
        header->magic = 0u;
        cuda_stub_tracked[0] = cuda_stub_tracked[--cuda_stub_tracked_count];
        free(header);
    }
    cuda_stub_alloc_calls = 0u;
    cuda_stub_fail_alloc_at = -1;
    cuda_stub_host_map_calls = 0u;
    cuda_stub_fail_host_map_at = -1;
    cuda_stub_export_calls = 0u;
    cuda_stub_fail_export_at = 0u;
    cuda_stub_ledger_unlock();
}

void spark_stub_cuda_fail_alloc_call(uint32_t one_based_call_index)
{
    cuda_stub_fail_alloc_at = (int32_t)one_based_call_index;
}

void spark_stub_cuda_fail_next_alloc(void)
{
    cuda_stub_ledger_lock();
    cuda_stub_fail_alloc_at = (int32_t)(cuda_stub_alloc_calls + 1u);
    cuda_stub_ledger_unlock();
}

void spark_stub_cuda_fail_alloc_after(uint32_t calls)
{
    cuda_stub_ledger_lock();
    cuda_stub_fail_alloc_at = (int32_t)(cuda_stub_alloc_calls + calls);
    cuda_stub_ledger_unlock();
}

void spark_stub_cuda_fail_export_after(uint32_t calls)
{
    cuda_stub_fail_export_at = cuda_stub_export_calls + calls;
}

void spark_stub_cuda_fail_host_map_call(uint32_t one_based_call_index)
{
    cuda_stub_fail_host_map_at = (int32_t)one_based_call_index;
}

uint32_t spark_stub_cuda_outstanding_allocs(void)
{
    uint32_t outstanding;
    cuda_stub_ledger_lock();
    outstanding = cuda_stub_tracked_count;
    cuda_stub_ledger_unlock();
    return outstanding;
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

cudaError_t cuda_stub_stream_query_result;

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    (void)stream;
    return cuda_stub_stream_query_result;
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

static uint32_t cuda_stub_destroy_count,cuda_stub_destroy_fail_at;

void spark_stub_cuda_fail_event_destroy_after(uint32_t calls)
{
    cuda_stub_destroy_fail_at = cuda_stub_destroy_count + calls;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    cuda_stub_destroy_count++;
    if (cuda_stub_destroy_count == cuda_stub_destroy_fail_at)
        return cudaErrorInvalidValue;
    free(event);
    return cudaSuccess;
}

static uint32_t cuda_stub_event_pending;
static uint32_t cuda_stub_event_record_failure;

void spark_stub_cuda_event_pending(uint32_t pending)
{
    cuda_stub_event_pending = pending;
}

void spark_stub_cuda_event_record_failure(uint32_t failure)
{
    cuda_stub_event_record_failure = failure;
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    (void)stream;
    if (cuda_stub_event_record_failure != 0u)
        return cudaErrorInvalidValue;
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    if (event != 0 && cuda_stub_event_pending != 0u)
        return cudaErrorNotReady;
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

cudaError_t cudaEventElapsedTime(float *milliseconds, cudaEvent_t start,
    cudaEvent_t stop)
{
    if (milliseconds == 0 || start == 0 || stop == 0)
        return cudaErrorInvalidValue;
    *milliseconds = 0.0f;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    return event != 0 ? cudaSuccess : cudaErrorInvalidValue;
}

static void *cuda_stub_host_registered[CUDA_STUB_MAX_TRACKED];
static size_t cuda_stub_host_registered_bytes[CUDA_STUB_MAX_TRACKED];
static uint32_t cuda_stub_host_registered_count;
uint32_t cuda_stub_host_register_calls;
uint32_t cuda_stub_host_register_flags;
uint32_t cuda_stub_host_unregister_calls;
int cuda_stub_host_register_result;
int cuda_stub_host_unregister_result;

uint32_t spark_stub_cuda_host_registered(void *address)
{
    uint32_t index,found = 0u;
    cuda_stub_ledger_lock();
    for ( index = 0u; index < cuda_stub_host_registered_count; index++ )
        if ( cuda_stub_host_registered[index] == address )
            found = 1u;
    cuda_stub_ledger_unlock();
    return found;
}

cudaError_t cudaHostRegister(void *address,size_t bytes,unsigned int flags)
{
    uint32_t index;
    uintptr_t first = (uintptr_t)address;
    cudaError_t result;
    cuda_stub_ledger_lock();
    cuda_stub_host_register_calls++;
    cuda_stub_host_register_flags = flags;
    result = cuda_stub_host_register_result;
    if ( result == cudaSuccess && (address == 0 || bytes == 0u ||
            bytes > UINTPTR_MAX - first || cuda_stub_host_registered_count == CUDA_STUB_MAX_TRACKED) )
        result = cudaErrorInvalidValue;
    for ( index = 0u; result == cudaSuccess && index < cuda_stub_host_registered_count; index++ )
    {
        uintptr_t registered = (uintptr_t)cuda_stub_host_registered[index];
        if ( first < registered + cuda_stub_host_registered_bytes[index] && registered < first + bytes )
            result = cudaErrorHostMemoryAlreadyRegistered;
    }
    if ( result == cudaSuccess )
    {
        cuda_stub_host_registered[cuda_stub_host_registered_count] = address;
        cuda_stub_host_registered_bytes[cuda_stub_host_registered_count++] = bytes;
    }
    cuda_stub_ledger_unlock();
    return result;
}

cudaError_t cudaHostUnregister(void *address)
{
    uint32_t index;
    cudaError_t result;
    cuda_stub_ledger_lock();
    cuda_stub_host_unregister_calls++;
    result = cuda_stub_host_unregister_result;
    for ( index = 0u; index < cuda_stub_host_registered_count; index++ )
        if ( cuda_stub_host_registered[index] == address )
            break;
    if ( result == cudaSuccess && index == cuda_stub_host_registered_count )
        result = cudaErrorHostMemoryNotRegistered;
    if ( result == cudaSuccess )
    {
        cuda_stub_host_registered[index] = cuda_stub_host_registered[--cuda_stub_host_registered_count];
        cuda_stub_host_registered_bytes[index] = cuda_stub_host_registered_bytes[cuda_stub_host_registered_count];
    }
    cuda_stub_ledger_unlock();
    return result;
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

cudaError_t cudaGraphUpload(
    cudaGraphExec_t graph_exec,
    cudaStream_t stream)
{
    (void)graph_exec;
    (void)stream;
    return cudaSuccess;
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

cudaError_t cudaGetLastError(void)
{
    return cudaSuccess;
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



#define CUDA_STUB_VMM_MAGIC UINT32_C(0x564D4D31)
#define CUDA_STUB_RESERVATION_MAGIC UINT32_C(0x564D4D32)
#define CUDA_STUB_VMM_MAPPED_MAX 128

typedef struct cuda_stub_vmm_phys
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    CUmemAllocationProp prop;
    uint8_t *image;
} cuda_stub_vmm_phys;

typedef struct cuda_stub_vmm_reservation
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    uint64_t mapped_bytes;
    unsigned int granted_access;
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
    if (header->magic != CUDA_STUB_ALLOC_MAGIC ||
        header->bytes < sizeof(cuda_stub_vmm_reservation))
    {
        return 0;
    }
    reservation = (cuda_stub_vmm_reservation *)user_pointer;
    return reservation->magic == CUDA_STUB_RESERVATION_MAGIC ? reservation : 0;
}

static cuda_stub_vmm_reservation *cuda_stub_vmm_reservation_for_va(
    CUdeviceptr pointer)
{
    uint32_t index;
    cuda_stub_vmm_reservation *found = 0;
    cuda_stub_ledger_lock();
    for (index = 0u; index < cuda_stub_tracked_count; index++)
    {
        cuda_stub_vmm_reservation *reservation = cuda_stub_vmm_reservation_at(
            cuda_stub_tracked[index]);
        if (reservation != 0 &&
            (CUdeviceptr)(reservation + 1) <= pointer &&
            pointer < (CUdeviceptr)(reservation + 1) + reservation->bytes)
        {
            found = reservation;
            break;
        }
    }
    cuda_stub_ledger_unlock();
    return found;
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
        : (size_t)(2ull * 1024ull * 1024ull);
    return CUDA_SUCCESS;
}


#define CUDA_STUB_SHARE_MAGIC UINT32_C(0x53505846)
#define CUDA_STUB_SHARE_VERSION 1u

typedef struct cuda_stub_share_header
{
    uint32_t magic;
    uint32_t version;
    uint64_t bytes;
} cuda_stub_share_header;

static int cuda_stub_vmm_single_mapping(const cuda_stub_vmm_phys *phys,
    CUdeviceptr *pointer_out)
{
    uint32_t index;
    uint32_t found = 0u;
    cuda_stub_ledger_lock();
    for (index = 0u; index < cuda_stub_tracked_count; index++)
    {
        cuda_stub_vmm_reservation *reservation = cuda_stub_vmm_reservation_at(
            cuda_stub_tracked[index]);
        uint32_t mapping_index;
        if (reservation == 0)
        {
            continue;
        }
        for (mapping_index = 0u; mapping_index < reservation->mapped_count;
            mapping_index++)
        {
            if (reservation->mappings[mapping_index].phys == phys)
            {
                *pointer_out = reservation->mappings[mapping_index].pointer;
                found++;
            }
        }
    }
    cuda_stub_ledger_unlock();
    return found == 1u;
}

CUresult cuMemExportToShareableHandle(void *shareable_handle,
    CUmemGenericAllocationHandle handle,
    CUmemAllocationHandleType handle_type,
    unsigned long long flags)
{
    cuda_stub_vmm_phys *phys = (cuda_stub_vmm_phys *)handle;
    cuda_stub_share_header header;
    CUdeviceptr mapped_at = 0;
    const char *tmp_dir;
    char *template_path;
    size_t tmp_bytes;
    uint8_t *cursor;
    size_t remaining;
    int fd;
    int written;
    (void)flags;
    cuda_stub_export_calls++;
    if (cuda_stub_export_calls == cuda_stub_fail_export_at)
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (shareable_handle == 0 || phys == 0 ||
        ((cuda_stub_alloc_header *)phys - 1)->magic != CUDA_STUB_ALLOC_MAGIC ||
        phys->magic != CUDA_STUB_VMM_MAGIC ||
        handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
        flags != 0ull || phys->bytes == 0ull)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((phys->prop.requestedHandleTypes &
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (phys->image == 0 && !cuda_stub_vmm_single_mapping(phys, &mapped_at))
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    tmp_dir = getenv("TMPDIR");
    if (tmp_dir == 0 || tmp_dir[0] == '\0')
    {
        tmp_dir = "/tmp";
    }
    tmp_bytes = strlen(tmp_dir) + sizeof("/sp_cuda_stub.XXXXXX");
    template_path = (char *)malloc(tmp_bytes);
    if (template_path == 0)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    snprintf(template_path, tmp_bytes, "%s/sp_cuda_stub.XXXXXX", tmp_dir);
    fd = mkstemp(template_path);
    if (fd < 0)
    {
        free(template_path);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    (void)unlink(template_path);
    free(template_path);
    (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
    if (ftruncate(fd, (off_t)(sizeof(header) + phys->bytes)) != 0)
    {
        (void)close(fd);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    header.magic = CUDA_STUB_SHARE_MAGIC;
    header.version = CUDA_STUB_SHARE_VERSION;
    header.bytes = phys->bytes;
    cursor = (uint8_t *)&header;
    remaining = sizeof(header);
    while (remaining != 0u)
    {
        written = (int)write(fd, cursor, remaining);
        if (written <= 0)
        {
            if (written < 0 && errno == EINTR)
            {
                continue;
            }
            (void)close(fd);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        cursor += written;
        remaining -= (size_t)written;
    }
    cursor = phys->image != 0 ? phys->image : (uint8_t *)(uintptr_t)mapped_at;
    remaining = (size_t)phys->bytes;
    while (remaining != 0u)
    {
        written = (int)write(fd, cursor, remaining);
        if (written <= 0)
        {
            if (written < 0 && errno == EINTR)
            {
                continue;
            }
            (void)close(fd);
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
        cursor += written;
        remaining -= (size_t)written;
    }
    *(int *)shareable_handle = fd;
    return CUDA_SUCCESS;
}

static uint32_t cuda_stub_import_count,cuda_stub_import_fail_at,cuda_stub_unmap_fail;
static uint32_t cuda_stub_import_delay_us;
static uint32_t cuda_stub_create_delay_us;

void spark_stub_cuda_set_import_delay(uint32_t delay)
{
    cuda_stub_import_delay_us = delay;
}

void spark_stub_cuda_set_create_delay(uint32_t delay)
{
    cuda_stub_create_delay_us = delay;
}

void spark_stub_cuda_fail_import_after(uint32_t calls)
{
    cuda_stub_import_fail_at = cuda_stub_import_count + calls;
}

void spark_stub_cuda_fail_next_unmap(void)
{
    cuda_stub_unmap_fail = 1u;
}

CUresult cuMemImportFromShareableHandle(CUmemGenericAllocationHandle *handle,
    void *shareable_handle,
    CUmemAllocationHandleType handle_type)
{
    cuda_stub_vmm_phys *phys;
    cuda_stub_share_header header;
    uint8_t *image = 0;
    struct stat status;
    uint8_t *cursor;
    uint8_t *header_cursor;
    size_t remaining;
    int fd;
    int received;
    if (cuda_stub_import_delay_us != 0u)
        usleep(cuda_stub_import_delay_us);
    cuda_stub_import_count++;
    if (cuda_stub_import_count == cuda_stub_import_fail_at)
        return CUDA_ERROR_OUT_OF_MEMORY;
    if (handle == 0 || shareable_handle == 0 ||
        handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    fd = (int)(uintptr_t)shareable_handle;
    if (fd < 0 || fcntl(fd, F_GETFD) < 0 || fstat(fd, &status) != 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if ((status.st_mode & 077) != 0 ||
        (uint64_t)status.st_size < (uint64_t)sizeof(header))
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    header_cursor = (uint8_t *)&header;
    remaining = sizeof(header);
    while (remaining != 0u)
    {
        received = (int)pread(fd, header_cursor, remaining,
            (off_t)(header_cursor - (uint8_t *)&header));
        if (received <= 0)
        {
            if (received < 0 && errno == EINTR)
            {
                continue;
            }
            return CUDA_ERROR_INVALID_VALUE;
        }
        header_cursor += received;
        remaining -= (size_t)received;
    }
    if (header.magic != CUDA_STUB_SHARE_MAGIC ||
        header.version != CUDA_STUB_SHARE_VERSION || header.bytes == 0ull ||
        (uint64_t)status.st_size < (uint64_t)sizeof(header) + header.bytes)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (cuda_stub_alloc((void **)&phys, sizeof(*phys)) != cudaSuccess)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    if (cuda_stub_alloc((void **)&image, (size_t)header.bytes) != cudaSuccess)
    {
        (void)cuda_stub_free(phys);
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    cursor = image;
    remaining = (size_t)header.bytes;
    while (remaining != 0u)
    {
        received = (int)pread(fd, cursor, remaining,
            (off_t)(sizeof(header) + (cursor - image)));
        if (received <= 0)
        {
            if (received < 0 && errno == EINTR)
            {
                continue;
            }
            (void)cuda_stub_free(image);
            (void)cuda_stub_free(phys);
            return CUDA_ERROR_INVALID_VALUE;
        }
        cursor += received;
        remaining -= (size_t)received;
    }
    phys->magic = CUDA_STUB_VMM_MAGIC;
    phys->mapped_count = 0u;
    phys->bytes = header.bytes;
    phys->prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    phys->prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    phys->prop.location.id = 0;
    phys->prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    phys->image = image;
    *handle = (CUmemGenericAllocationHandle)phys;
    return CUDA_SUCCESS;
}

CUresult cuMemCreate(CUmemGenericAllocationHandle *handle,
    size_t bytes,
    const CUmemAllocationProp *prop,
    unsigned long long flags)
{
    cuda_stub_vmm_phys *phys;
    if (cuda_stub_create_delay_us != 0u)
        usleep(cuda_stub_create_delay_us);
    if (handle == 0 || prop == 0 ||
        prop->type != CU_MEM_ALLOCATION_TYPE_PINNED ||
        prop->location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        flags != 0ull || bytes == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (cuda_stub_alloc((void **)&phys, sizeof(*phys)) != cudaSuccess)
    {
        return CUDA_ERROR_OUT_OF_MEMORY;
    }
    phys->magic = CUDA_STUB_VMM_MAGIC;
    phys->mapped_count = 0u;
    phys->bytes = (uint64_t)bytes;
    phys->prop = *prop;
    phys->image = 0;
    *handle = (CUmemGenericAllocationHandle)phys;
    return CUDA_SUCCESS;
}

CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr pointer,
    size_t bytes, CUmemRangeHandleType type, unsigned long long flags)
{
    (void)handle; (void)pointer; (void)bytes; (void)type; (void)flags;
    return CUDA_ERROR_INVALID_VALUE;
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
    reservation->magic = CUDA_STUB_RESERVATION_MAGIC;
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
    if (phys->image != 0)
    {
        memcpy((void *)(uintptr_t)pointer, phys->image, bytes);
    }
    return CUDA_SUCCESS;
}

CUresult cuMemSetAccess(CUdeviceptr pointer,
    size_t bytes,
    const CUmemAccessDesc *descriptors,
    size_t descriptor_count)
{
    cuda_stub_vmm_reservation *reservation;
    CUdeviceptr span_start;
    uint64_t offset;
    if (descriptors == 0 || descriptor_count != (size_t)1u ||
        descriptors[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        (descriptors[0].flags & ~CU_MEM_ACCESS_FLAGS_PROT_READWRITE) != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation = cuda_stub_vmm_reservation_for_va(pointer);
    if (reservation == 0 || bytes == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    span_start = (CUdeviceptr)(reservation + 1);
    offset = pointer - span_start;
    if (offset > reservation->bytes ||
        (uint64_t)bytes > reservation->bytes - offset)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation->granted_access = (unsigned int)descriptors[0].flags;
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoD(CUdeviceptr destination, CUdeviceptr source,
    size_t bytes)
{
    /* Host-model D2D: VMM reservations and plain allocations are host
     * memory in this stub; validate VMM bounds/access when resolvable and
     * copy. Unresolvable (plain cudaMalloc) pointers copy directly, same
     * as the stub cudaMemcpy device paths. */
    cuda_stub_vmm_reservation *target =
        cuda_stub_vmm_reservation_for_va(destination);
    cuda_stub_vmm_reservation *origin =
        cuda_stub_vmm_reservation_for_va(source);
    if ( bytes == 0u )
        return CUDA_SUCCESS;
    if ( target != 0 )
    {
        CUdeviceptr span_start = (CUdeviceptr)(target + 1);
        uint64_t offset = destination - span_start;
        if ( offset > target->bytes ||
            (uint64_t)bytes > target->bytes - offset ||
            target->granted_access != CU_MEM_ACCESS_FLAGS_PROT_READWRITE )
            return CUDA_ERROR_INVALID_VALUE;
    }
    if ( origin != 0 )
    {
        CUdeviceptr span_start = (CUdeviceptr)(origin + 1);
        uint64_t offset = source - span_start;
        if ( offset > origin->bytes ||
            (uint64_t)bytes > origin->bytes - offset )
            return CUDA_ERROR_INVALID_VALUE;
    }
    memmove((void *)(uintptr_t)destination,
        (const void *)(uintptr_t)source,bytes);
    return CUDA_SUCCESS;
}

CUresult cuda_stub_vmm_probe_write(CUdeviceptr pointer,
    const void *bytes,
    size_t count)
{
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    CUdeviceptr span_start;
    uint64_t offset;
    if (reservation == 0 || bytes == 0 || count == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    span_start = (CUdeviceptr)(reservation + 1);
    offset = pointer - span_start;
    if (offset > reservation->bytes ||
        (uint64_t)count > reservation->bytes - offset)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (reservation->granted_access != CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    memcpy((void *)(uintptr_t)pointer, bytes, count);
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr pointer, size_t bytes)
{
    if (cuda_stub_unmap_fail != 0u)
    {
        cuda_stub_unmap_fail = 0u;
        return CUDA_ERROR_INVALID_VALUE;
    }
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    uint32_t mapping_index;
    uint32_t removed = 0u;
    if (reservation != 0)
    {
        for (mapping_index = 0u; mapping_index < reservation->mapped_count;
            mapping_index++)
        {
            CUdeviceptr mapped_at = reservation->mappings[mapping_index].pointer;
            uint64_t mapped_bytes = reservation->mappings[mapping_index].bytes;
            if (mapped_at >= pointer &&
                mapped_at + mapped_bytes <= pointer + (uint64_t)bytes)
            {
                reservation->mappings[mapping_index].phys->mapped_count--;
                reservation->mapped_bytes -= mapped_bytes;
                reservation->mappings[mapping_index] =
                    reservation->mappings[reservation->mapped_count - 1u];
                reservation->mapped_count--;
                removed++;
                mapping_index--;
            }
            else if (mapped_at < pointer + (uint64_t)bytes &&
                pointer < mapped_at + mapped_bytes)
            {
                return CUDA_ERROR_INVALID_VALUE;
            }
        }
    }
    return removed != 0u ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
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
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (phys->image != 0)
    {
        (void)cuda_stub_free(phys->image);
        phys->image = 0;
    }
    return cuda_stub_free(phys);
}

CUresult cuMemAddressFree(CUdeviceptr pointer, size_t bytes)
{
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    (void)bytes;
    if (reservation == 0 ||
        (CUdeviceptr)(reservation + 1) != pointer)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (reservation->mapped_count != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    return cuda_stub_free(reservation);
}

uint32_t cuda_stub_mesh_publish_calls = 0u;
uint32_t cuda_stub_mesh_publish_null_seq_cell = 0u;
uint32_t cuda_stub_mesh_publish_null_epoch_cell = 0u;

static uint64_t cuda_stub_roundloop_now_ns(void);

cudaError_t SparkGlm5NextLaunchMeshGuard(cudaStream_t stream,
    volatile void *error_word,void *output)
{
    (void)stream;
    if ( error_word == NULL || output == NULL )
        return cudaErrorInvalidValue;
    if ( *(volatile uint64_t *)error_word != 0u )
        *(uint64_t *)output = UINT64_MAX;
    return cudaSuccess;
}

cudaError_t SparkGlm5NextLaunchMeshCopyDown(cudaStream_t stream,
    volatile void *destination,const void *source,uint64_t bytes,
    const volatile void *shipped_cell,void *round_control,
    const volatile void *cancel_cell,uint64_t timeout_ns)
{
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    const volatile uint64_t *shipped = (const volatile uint64_t *)shipped_cell;
    const volatile uint64_t *cancel = (const volatile uint64_t *)cancel_cell;
    uint64_t deadline = cuda_stub_roundloop_now_ns() + timeout_ns;
    uint64_t previous,expected_cancel;
    (void)stream;
    if ( destination == NULL || source == NULL || bytes == 0u ||
         shipped == NULL || control == NULL || cancel == NULL || timeout_ns == 0u )
        return cudaErrorInvalidValue;
    previous = control->round_seq;
    expected_cancel = control->cancel_expected;
    while ( previous != 0u && *shipped != previous )
    {
        if ( control->error_word != 0u || *cancel != expected_cancel ||
             cuda_stub_roundloop_now_ns() >= deadline )
        {
            control->error_word = previous;
            return cudaSuccess;
        }
        sched_yield();
    }
    if ( *cancel != expected_cancel )
        control->error_word = UINT64_C(0xFFFFFFFFFE000000) | previous;
    if ( control->error_word == 0u )
    {
        memcpy((void *)destination,source,(size_t)bytes);
        __sync_synchronize();
    }
    return cudaSuccess;
}

cudaError_t SparkGlm5NextLaunchMeshPublish(cudaStream_t stream,
    volatile void *entry,void *seq_cell,const void *epoch_cell,
    void *round_seq,uint64_t bytes,
    uint64_t slot_index,uint64_t slots_per_rank,volatile void *slot_tail,
    void *error_word,uint32_t peer_mask)
{
    (void)stream;(void)round_seq;(void)bytes;
    (void)slots_per_rank;(void)error_word;
    (void)__sync_add_and_fetch(&cuda_stub_mesh_publish_calls,1u);
    if ( seq_cell == NULL )
        cuda_stub_mesh_publish_null_seq_cell++;
    if ( epoch_cell == NULL )
        cuda_stub_mesh_publish_null_epoch_cell++;
    if ( error_word != NULL && *(uint64_t *)error_word != 0u )
        return cudaSuccess;
    if ( entry != NULL && seq_cell != NULL && epoch_cell != NULL && slot_tail != NULL )
    {
        uint64_t tag = (*(uint64_t *)epoch_cell << 32) | ((*(uint64_t *)seq_cell + 1u) & 0xffffffffu);
        volatile uint64_t *ent = (volatile uint64_t *)entry;
        ent[2] = slot_index;
        ent[1] = bytes;
        ent[3] = peer_mask;
        __sync_synchronize();
        *(volatile uint64_t *)slot_tail = tag;
        *(uint64_t *)round_seq = tag;
        *(uint64_t *)seq_cell = *(uint64_t *)seq_cell + 1u;
        __sync_synchronize();
        ent[0] = tag;
    }
    return cudaSuccess;
}

uint32_t cuda_stub_mesh_seq_pad_calls = 0u;

cudaError_t SparkGlm5NextLaunchMeshSeqPad(cudaStream_t stream,
    void *seq_cell)
{
    (void)stream;
    cuda_stub_mesh_seq_pad_calls++;
    if ( seq_cell != NULL )
        *(uint64_t *)seq_cell = *(uint64_t *)seq_cell + 1u;
    return cudaSuccess;
}

cudaError_t SparkGlm5NextLaunchMeshWait(cudaStream_t stream,
    volatile void *band_base,uint64_t slot_bytes,const void *round_seq,
    uint64_t slots_per_rank,uint32_t rank,uint32_t degree,void *error_word,
    unsigned long long deadline_ns,void *diag_word,volatile void *cancel_cell,
    const void *cancel_expected,void *arrival_ring)
{
    uint64_t sequence = *(const uint64_t *)round_seq;
    uint64_t expected_cancel = *(const uint64_t *)cancel_expected;
    uint64_t ring = (sequence - 1u) & (slots_per_rank - 1u);
    uint64_t deadline = cuda_stub_roundloop_now_ns() + deadline_ns;
    uint32_t peer;
    (void)stream;
    if ( *(volatile uint64_t *)cancel_cell != expected_cancel )
    {
        *(volatile uint64_t *)error_word = UINT64_C(0xFFFFFFFFFE000000) | sequence;
        return cudaSuccess;
    }
    for ( peer = 0u; peer < degree; peer++ )
    {
        const volatile uint64_t *tail;
        if ( peer == rank ) continue;
        tail = (const volatile uint64_t *)((const uint8_t *)band_base +
            ((uint64_t)peer * slots_per_rank + ring) * slot_bytes + slot_bytes - 8u);
        while ( (*tail >> 32u) != (sequence >> 32u) || *tail < sequence )
        {
            if ( *(volatile uint64_t *)error_word != 0u ) return cudaSuccess;
            if ( *(volatile uint64_t *)cancel_cell != expected_cancel )
            {
                *(volatile uint64_t *)error_word = UINT64_C(0xFFFFFFFFFE000000) | sequence;
                return cudaSuccess;
            }
            if ( cuda_stub_roundloop_now_ns() >= deadline )
            {
                *(volatile uint64_t *)diag_word = ((uint64_t)peer << 56u) |
                    ((sequence & 0xffffu) << 16u) | (*tail & 0xffffu);
                *(volatile uint64_t *)error_word = sequence;
                return cudaSuccess;
            }
            sched_yield();
        }
    }
    __sync_synchronize();
    if ( arrival_ring != NULL )
        ((uint64_t *)arrival_ring)[sequence & 255u] = cuda_stub_roundloop_now_ns();
    return cudaSuccess;
}

static int cuda_stub_tree_wait(const volatile uint64_t *cell,uint64_t tag,
    const volatile uint64_t *cancel,SparkTpMeshRoundControl *control,uint64_t deadline,uint64_t expected_cancel)
{
    for (;;)
    {
        if ( control->error_word != 0u ) return 0;
        if ( *cancel != expected_cancel )
        {
            control->error_word = UINT64_C(0xFFFFFFFFFE000000) | tag;
            return 0;
        }
        if ( tag == 0u || *cell == tag ) return 1;
        if ( cuda_stub_roundloop_now_ns() >= deadline )
        {
            control->error_word = tag;
            control->diag_word = *cell;
            return 0;
        }
        sched_yield();
    }
}

cudaError_t SparkGlm5NextLaunchMeshTree(cudaStream_t stream,void *band_base,
    uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry_address,
    const volatile void *shipped_address,const volatile void *cancel_address,
    void *round_control,uint32_t rank,uint32_t degree,const void *local,
    void *output,void *scratch,uint64_t elements,
    uint32_t operation,uint32_t rounds,uint64_t timeout_ns)
{
    uint8_t *band = (uint8_t *)band_base;
    volatile uint64_t *entry = (volatile uint64_t *)entry_address;
    const volatile uint64_t *shipped = (const volatile uint64_t *)shipped_address;
    const volatile uint64_t *cancel = (const volatile uint64_t *)cancel_address;
    SparkTpMeshRoundControl *control = (SparkTpMeshRoundControl *)round_control;
    uint32_t levels = SparkTpMeshTreeLevels(degree);
    uint32_t width = operation == 2u ? 8u : operation == 1u ? 4u : 2u;
    uint64_t capacity = (slot_bytes - 16u) / width;
    uint64_t deadline = cuda_stub_roundloop_now_ns() + timeout_ns;
    uint64_t expected_cancel = control->cancel_expected;
    uint32_t round;
    (void)stream;
    for ( round = 0u; round < rounds; round++ )
    {
        uint64_t begin;
        for ( begin = 0u; begin < elements; begin += capacity )
        {
            uint64_t count = elements - begin < capacity ? elements - begin : capacity;
            uint64_t i;
            uint32_t phase;
            for ( i = 0u; i < count; i++ )
            {
                uint64_t index = begin + i;
                if ( operation == 1u )
                {
                    uint32_t bits = (uint32_t)((const uint16_t *)local)[index] << 16u;
                    memcpy((float *)scratch + i,&bits,sizeof(bits));
                }
                else if ( operation == 2u ) ((uint64_t *)scratch)[i] = ((const uint64_t *)local)[index];
                else
                {
                    uint32_t owner = (uint32_t)(index / (elements / degree));
                    uint64_t source = index % (elements / degree);
                    ((uint16_t *)scratch)[i] = owner == rank ? ((const uint16_t *)local)[source] : 0u;
                }
            }
            for ( phase = 0u; phase < 2u * levels; phase++ )
            {
                uint32_t route = SparkTpMeshTreeRoute(rank,degree,phase);
                uint32_t send = route >> 16u, receive = route & 0xffffu;
                uint64_t tag,ring;
                if ( control->seq >= UINT32_MAX || control->error_word != 0u )
                { control->error_word = UINT64_MAX; return cudaSuccess; }
                tag = (control->epoch << 32u) | (control->seq + 1u);
                ring = (tag - 1u) & (slots_per_rank - 1u);
                if ( send != 0u )
                {
                    uint64_t slot = ((uint64_t)rank * slots_per_rank + ring) * slot_bytes;
                    if ( !cuda_stub_tree_wait(shipped,control->round_seq,cancel,control,deadline,expected_cancel) ) return cudaSuccess;
                    memcpy(band + slot,scratch,(size_t)(count * width));
                    entry[1] = count * width;
                    entry[2] = slot / slot_bytes;
                    entry[3] = 1u << (send - 1u);
                    control->round_seq = tag;
                    __sync_synchronize();
                    *(volatile uint64_t *)(band + slot + slot_bytes - 8u) = tag;
                    __sync_synchronize();
                    entry[0] = tag;
                }
                if ( receive != 0u )
                {
                    uint8_t *source = band + ((uint64_t)(receive - 1u) * slots_per_rank + ring) * slot_bytes;
                    if ( !cuda_stub_tree_wait((const volatile uint64_t *)(source + slot_bytes - 8u),tag,cancel,control,deadline,expected_cancel) ) return cudaSuccess;
                    __sync_synchronize();
                    for ( i = 0u; i < count; i++ )
                    {
                        if ( operation == 1u ) ((float *)scratch)[i] = phase < levels ?
                            ((float *)scratch)[i] + ((const float *)source)[i] : ((const float *)source)[i];
                        else if ( operation == 2u )
                        {
                            uint64_t value = ((const uint64_t *)source)[i];
                            if ( phase >= levels || value > ((uint64_t *)scratch)[i] ) ((uint64_t *)scratch)[i] = value;
                        }
                        else ((uint16_t *)scratch)[i] = phase < levels ?
                            ((uint16_t *)scratch)[i] | ((const uint16_t *)source)[i] : ((const uint16_t *)source)[i];
                    }
                }
                control->seq++;
            }
            for ( i = 0u; i < count; i++ )
            {
                if ( operation == 1u )
                {
                    uint32_t bits;
                    memcpy(&bits,(const float *)scratch + i,sizeof(bits));
                    ((uint16_t *)output)[begin + i] = (uint16_t)(bits >> 16u);
                }
                else if ( operation == 2u ) ((uint64_t *)output)[begin + i] = ((uint64_t *)scratch)[i];
                else ((uint16_t *)output)[begin + i] = ((uint16_t *)scratch)[i];
            }
        }
        control->slot_cursor = control->seq;
        control->rounds_done++;
    }
    return cudaSuccess;
}

uint32_t cuda_stub_roundloop_launches = 0u;
uint64_t cuda_stub_roundloop_rounds = 0u;

static uint64_t cuda_stub_roundloop_now_ns(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
        return 0ull;
    return (uint64_t)now.tv_sec * UINT64_C(1000000000) +
        (uint64_t)now.tv_nsec;
}

static uint16_t cuda_stub_roundloop_bf16_rne(float value)
{
    uint32_t bits;
    uint32_t lsb;
    memcpy(&bits, &value, 4u);
    lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    return (uint16_t)(bits >> 16);
}

static float cuda_stub_roundloop_bf16_load(uint32_t packed, uint32_t high)
{
    uint32_t bits = (high != 0u ? packed & UINT32_C(0xffff0000)
        : (packed & UINT32_C(0x0000ffff)) << 16);
    float out;
    memcpy(&out, &bits, 4u);
    return out;
}

cudaError_t SparkGlm5NextLaunchMeshRoundLoop(cudaStream_t stream,
    volatile void *band_base,uint64_t slot_bytes,uint64_t slots_per_rank,
    volatile void *entry,void *shipped_cell,volatile void *cancel_cell,
    void *round_control,uint32_t rank,uint32_t degree,
    const void *local_device,void *full_device,uint64_t bytes)
{
    SparkTpMeshRoundControl *control =
        (SparkTpMeshRoundControl *)round_control;
    uint8_t *band = (uint8_t *)band_base;
    volatile uint64_t *entry_words = (volatile uint64_t *)entry;
    volatile uint64_t *shipped = (volatile uint64_t *)shipped_cell;
    volatile uint64_t *cancel = (volatile uint64_t *)cancel_cell;
    uint64_t expected_cancel = ((SparkTpMeshRoundControl *)round_control)->cancel_expected;
    uint64_t cursor, prev_tag, stop_at, parity;
    uint32_t failed = 0u;
    (void)stream;
    (void)__sync_add_and_fetch(&cuda_stub_roundloop_launches, 1u);
    cursor = control->slot_cursor;
    prev_tag = control->round_seq;
    stop_at = cuda_stub_roundloop_now_ns() + control->deadline_ns;
    while (control->rounds_done < control->rounds_total)
    {
        uint64_t slot;
        uint64_t tag;
        uint32_t peer;
        if (prev_tag != 0ull)
        {
            while (*shipped != prev_tag)
            {
                if (*cancel != expected_cancel)
                {
                    failed = 1u;
                    break;
                }
                if (cuda_stub_roundloop_now_ns() >= stop_at)
                {
                    control->diag_word = (UINT64_C(0xa5) << 56) |
                        ((prev_tag & 0xffffull) << 16) |
                        (*shipped & 0xffffull);
                    control->error_word = prev_tag;
                    fprintf(stderr,
                        "MESH-ROUNDLOOP-TIMEOUT rank=%u phase=ship-ack want=%llu got=%llu\n",
                        rank, (unsigned long long)prev_tag, (unsigned long long)*shipped);
                    failed = 1u;
                    break;
                }
                sched_yield();
            }
            if (failed != 0u)
                break;
        }
        slot = (uint64_t)rank * slots_per_rank +
            (cursor & (slots_per_rank - 1ull));
        parity = cursor & (slots_per_rank - 1ull);
        tag = (control->epoch << 32) |
            ((control->seq + 1ull) & UINT64_C(0xffffffff));
        memcpy(band + slot * slot_bytes, local_device, (size_t)bytes);
        entry_words[2] = slot;
        entry_words[1] = bytes;
        entry_words[3] = ((1u << degree) - 1u) & ~(1u << rank);
        control->seq = control->seq + 1ull;
        control->round_seq = tag;
        __sync_synchronize();
        *(volatile uint64_t *)(band + slot * slot_bytes +
            slot_bytes - 8ull) = tag;
        __sync_synchronize();
        entry_words[0] = tag;
        for (peer = 0u; peer < degree - 1u; peer++)
        {
            uint32_t peer_rank = peer < rank ? peer : peer + 1u;
            volatile uint64_t *end_word = (volatile uint64_t *)
                (band + ((uint64_t)peer_rank * slots_per_rank + parity) *
                    slot_bytes + slot_bytes - 8ull);
            while (*end_word < tag)
            {
                if (*cancel != expected_cancel)
                {
                    failed = 1u;
                    break;
                }
                if (cuda_stub_roundloop_now_ns() >= stop_at)
                {
                    control->diag_word = ((uint64_t)peer_rank << 56) |
                        ((cursor & 0xffull) << 48) |
                        (parity << 32) |
                        ((tag & 0xffffull) << 16) |
                        (*end_word & 0xffffull);
                    control->error_word = tag;
                    fprintf(stderr,
                        "MESH-ROUNDLOOP-TIMEOUT rank=%u phase=peer-wait peer=%u want=%u got=%u\n",
                        rank, peer_rank, (uint32_t)tag, (uint32_t)*end_word);
                    failed = 1u;
                    break;
                }
                sched_yield();
            }
            if (failed != 0u)
                break;
        }
        if (failed != 0u)
            break;
        __sync_synchronize();
        {
            const uint32_t *sources = (const uint32_t *)
                (band + parity * slot_bytes);
            uint32_t *destination = (uint32_t *)full_device;
            uint64_t pairs = bytes >> 2;
            uint64_t pair;
            uint32_t source;
            for (pair = 0ull; pair < pairs; pair++)
            {
                float acc_x = 0.0f;
                float acc_y = 0.0f;
                uint32_t packed;
                for (source = 0u; source < degree; source++)
                {
                    packed = sources[source * slots_per_rank *
                        (slot_bytes / 4ull) + pair];
                    acc_x += cuda_stub_roundloop_bf16_load(packed, 0u);
                    acc_y += cuda_stub_roundloop_bf16_load(packed, 1u);
                }
                packed = (uint32_t)cuda_stub_roundloop_bf16_rne(acc_x) |
                    ((uint32_t)cuda_stub_roundloop_bf16_rne(acc_y) << 16);
                destination[pair] = packed;
            }
        }
        __sync_synchronize();
        cursor = cursor + 1ull;
        prev_tag = tag;
        control->slot_cursor = cursor;
        control->rounds_done = control->rounds_done + 1ull;
        (void)__sync_add_and_fetch(&cuda_stub_roundloop_rounds, 1ull);
    }
    return cudaSuccess;
}

uint32_t cuda_stub_mesh_hardware_calls;
int cuda_stub_mesh_hardware_prepare_result;
int cuda_stub_mesh_hardware_launch_result = cudaErrorUnknown;
void *cuda_stub_mesh_hardware_alias;
void *cuda_stub_mesh_hardware_band;
void *cuda_stub_mesh_hardware_gate;
void *cuda_stub_mesh_hardware_control;
uint64_t cuda_stub_mesh_hardware_elements;
uint32_t cuda_stub_mesh_hardware_operation;
uint32_t cuda_stub_mesh_hardware_logical_rows;
uint32_t cuda_stub_mesh_hardware_slice_routes;

cudaError_t SparkGlm5NextMeshHardwarePrepare(void *host,void **device)
{
    if ( cuda_stub_mesh_hardware_prepare_result != 0 )
        return cuda_stub_mesh_hardware_prepare_result;
    *device = cuda_stub_mesh_hardware_alias != 0 ? cuda_stub_mesh_hardware_alias : host;
    return cudaSuccess;
}

cudaError_t SparkGlm5NextLaunchMeshHardware(cudaStream_t stream,void *band,
    uint64_t slot_bytes,uint64_t slots_per_rank,volatile void *entry,void *gate,
    void *round_control,uint32_t rank,uint32_t degree,const void *local,
    void *output,void *scratch,uint64_t elements,uint32_t operation,
    uint32_t rounds,uint32_t logical_rows,uint32_t slice_routes,uint64_t timeout_ns)
{
    (void)stream;(void)slot_bytes;(void)slots_per_rank;(void)entry;
    cuda_stub_mesh_hardware_slice_routes = slice_routes;
    (void)round_control;(void)rank;(void)degree;(void)local;(void)output;
    (void)scratch;(void)rounds;(void)timeout_ns;
    cuda_stub_mesh_hardware_calls++;
    cuda_stub_mesh_hardware_band = band;
    cuda_stub_mesh_hardware_gate = gate;
    cuda_stub_mesh_hardware_control = round_control;
    cuda_stub_mesh_hardware_elements = elements;
    cuda_stub_mesh_hardware_operation = operation;
    cuda_stub_mesh_hardware_logical_rows = logical_rows;
    return cuda_stub_mesh_hardware_launch_result;
}
