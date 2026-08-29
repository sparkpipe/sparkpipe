#include "cuda_runtime_api.h"
#include "cuda.h"

#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

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
/* W3: the daemon thread's cuMem* calls (export walks this ledger) and the
 * consumer's (import inserts) overlap in-process, exactly as two driver
 * calls from two threads overlap on the real GPU - the driver serializes
 * its internals, so the stub serializes its ledger. */
static pthread_mutex_t cuda_stub_ledger_mutex = PTHREAD_MUTEX_INITIALIZER;

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
    cuda_stub_ledger_unlock();
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
#define CUDA_STUB_VMM_MAPPED_MAX 128

typedef struct cuda_stub_vmm_phys
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    CUmemAllocationProp prop;
    /* W3 fd tier: an IMPORTED physical chunk owns a host image of the bytes
     * its shareable fd carried (heap pages cannot alias across that fd the
     * way device pages do); cuMemMap of such a chunk reveals the image at
     * the mapping. Chunks created here keep image == 0: their content lives
     * at the (single) mapped span exactly as in W2b. */
    uint8_t *image;
} cuda_stub_vmm_phys;

typedef struct cuda_stub_vmm_reservation
{
    uint32_t magic;
    uint32_t mapped_count;
    uint64_t bytes;
    uint64_t mapped_bytes;
    /* the access grant cuMemSetAccess last applied over the whole span
       (CU_MEM_ACCESS_FLAGS_PROT_NONE while ungranted). The real driver
       enforces this in its page tables; the stub enforces it in the
       cuda_stub_vmm_probe_write receipt probe below. */
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
    if (header->magic != CUDA_STUB_ALLOC_MAGIC)
    {
        return 0;
    }
    reservation = (cuda_stub_vmm_reservation *)user_pointer;
    return reservation->magic == CUDA_STUB_VMM_MAGIC ? reservation : 0;
}

/* The reservation containing a mapped virtual address: the VA handed out is
 * the tail of the tracked allocation (struct end), so walk the tracked
 * headers and match struct+1. W3 fix: the match is CONTAINMENT, not equality
 * - an arena maps chunk 2 at base+chunk_bytes, and a multi-chunk span is
 * one reservation (W2b's tests never exceeded one 2 MiB chunk, so the old
 * exact-start match stood untested; the real driver maps anywhere inside
 * the reserved span). */
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
        : (size_t)(2ull * 1024ull * 1024ull); /* the 2 MiB page law */
    return CUDA_SUCCESS;
}

/* ------------------- W3 fd tier: shareable-handle export/import -------------------
 *
 * The stub models a shareable handle as what it is on the real OS: a POSIX
 * file descriptor whose referent outlives the exporter and crosses processes
 * by SCM_RIGHTS alone. Export writes the chunk's committed bytes into a
 * freshly created ANONYMOUS regular file - mkstemp (mode 0600), immediate
 * unlink, FD_CLOEXEC - so the bytes exist only behind the fd: no path to
 * follow (the fd-world O_NOFOLLOW analog), no group/other access (the KV
 * backing's owner-only discipline), no leak across exec. Import validates
 * the fd (live, owner-only mode, magic header, exact byte count), material-
 * izes a fresh local physical record, and copies the bytes into its image;
 * the caller's fd is its own from there and close() is the caller's duty.
 * The one fidelity trade vs the GPU: pages are copied at export, not
 * aliased - sound here because arena bytes are frozen at cold load, and it
 * is exactly what the spark-gated GPU receipt re-proves aliased. */

#define CUDA_STUB_SHARE_MAGIC UINT32_C(0x53505846) /* "SPXF" */
#define CUDA_STUB_SHARE_VERSION 1u

typedef struct cuda_stub_share_header
{
    uint32_t magic;
    uint32_t version;
    uint64_t bytes;
} cuda_stub_share_header;

/* Locate the one live mapping of a created chunk: its committed bytes live
 * at that mapped VA (the W2b model). */
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
    if (shareable_handle == 0 || phys == 0 ||
        ((cuda_stub_alloc_header *)phys - 1)->magic != CUDA_STUB_ALLOC_MAGIC ||
        phys->magic != CUDA_STUB_VMM_MAGIC ||
        handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR ||
        flags != 0ull || phys->bytes == 0ull)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    /* the exportable-property discipline: the chunk must have been created
     * with the POSIX-fd handle type requested (the daemon's prop carries it) */
    if ((phys->prop.requestedHandleTypes &
            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR) == 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (phys->image == 0 && !cuda_stub_vmm_single_mapping(phys, &mapped_at))
    {
        return CUDA_ERROR_INVALID_VALUE; /* nothing committed to export */
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
    (void)unlink(template_path); /* anonymous from here: fd-only referent */
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
    if (handle == 0 || shareable_handle == 0 ||
        handle_type != CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    fd = *(int *)shareable_handle;
    if (fd < 0 || fcntl(fd, F_GETFD) < 0 || fstat(fd, &status) != 0)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    /* OS-level access scoping on the received fd: owner-only, exactly the
     * discipline the exporter created it with (no group/other bits) */
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
    phys->image = 0;
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
    /* an imported chunk's bytes arrive with the map (the stub's stand-in for
     * physical aliasing); created chunks keep the W2b behavior where the
     * mapping's span itself is the storage */
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
    if (descriptors == 0 || descriptor_count != (size_t)1u ||
        descriptors[0].location.type != CU_MEM_LOCATION_TYPE_DEVICE ||
        (descriptors[0].flags & ~CU_MEM_ACCESS_FLAGS_PROT_READWRITE) != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    reservation = cuda_stub_vmm_reservation_for_va(pointer);
    if (reservation != 0 && reservation->bytes == (uint64_t)bytes)
    {
        /* record the grant: the probe (below) enforces it the way the real
           driver's page tables do */
        reservation->granted_access = (unsigned int)descriptors[0].flags;
        return CUDA_SUCCESS;
    }
    return CUDA_ERROR_INVALID_VALUE;
}

/* Test-only receipt probe (NOT CUDA API): model a DEVICE WRITE through a
 * mapped VA the way the real driver enforces it in hardware - a store
 * against a span whose granted access lacks WRITE is refused and touches
 * nothing. This is the enforcement half of the W3 scribble-probe receipt:
 * the production consumer leg grants CU_MEM_ACCESS_FLAGS_PROT_READWRITE
 * today, so a probe through a real ImportMap mapping LANDS; the staged
 * one-constant flip to CU_MEM_ACCESS_FLAGS_PROT_READ (runtime/spark_
 * weightd_attach.c) is what turns the same probe into a refusal, and this
 * stub path is the host proof that the flip will actually bite. */
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
        return CUDA_ERROR_INVALID_VALUE; /* outside the reserved span */
    }
    if (reservation->granted_access != CU_MEM_ACCESS_FLAGS_PROT_READWRITE)
    {
        return CUDA_ERROR_INVALID_VALUE; /* write refused: RO or ungranted */
    }
    memcpy((void *)(uintptr_t)pointer, bytes, count);
    return CUDA_SUCCESS;
}

CUresult cuMemUnmap(CUdeviceptr pointer, size_t bytes)
{
    cuda_stub_vmm_reservation *reservation =
        cuda_stub_vmm_reservation_for_va(pointer);
    uint32_t mapping_index;
    uint32_t removed = 0u;
    if (reservation != 0)
    {
        /* driver range semantics: one unmap may cover several per-chunk
         * mappings (the daemon maps chunk by chunk and tears the span down
         * whole); every mapping fully inside the range goes, a partial
         * overlap is refused, and an unmapped range is an error */
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
                mapping_index--; /* the swapped-in entry must be checked too */
            }
            else if (mapped_at < pointer + (uint64_t)bytes &&
                pointer < mapped_at + mapped_bytes)
            {
                return CUDA_ERROR_INVALID_VALUE; /* partial overlap */
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
        return CUDA_ERROR_INVALID_VALUE; /* still mapped somewhere */
    }
    if (phys->image != 0)
    {
        (void)cuda_stub_free(phys->image); /* the imported byte image */
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
        (CUdeviceptr)(reservation + 1) != pointer) /* the exact span base */
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (reservation->mapped_count != 0u)
    {
        return CUDA_ERROR_INVALID_VALUE; /* mappings still live */
    }
    return cuda_stub_free(reservation);
}

