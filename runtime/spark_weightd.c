#if defined(__APPLE__)
/* st_mtimespec lives behind the Darwin extensions */
#define _DARWIN_C_SOURCE 1
#endif

#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_error_site.h"
#include "sparkpipe/spark_weightd_worker.h"
#include "sparkpipe/spark_weightd_spine.h"
#include <stdatomic.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

#ifndef POLLRDHUP
#define POLLRDHUP 0
#endif

extern uint32_t SparkWeightdMeshReady(void);
extern uint64_t SparkWeightdMeshBufferAddress(void);
extern uint32_t SparkWeightdMeshBufferLkey(void);
extern int SparkWeightdMeshBufferFd(void);
extern void SparkWeightdMeshPoll(void);
extern SparkStatus SparkWeightdMeshPostWrite(uint32_t peer,
    uint64_t local_addr, uint32_t lkey, uint32_t length,
    uint64_t remote_offset);
extern uint32_t SparkWeightdMeshBroadcast(uint32_t peer_mask,
    uint64_t source_offset, uint32_t length, uint64_t remote_offset,
    uint64_t seq_value, uint64_t seq_remote_offset);

__attribute__((weak)) uint32_t SparkWeightdMeshReady(void) { return 0u; }
__attribute__((weak)) uint64_t SparkWeightdMeshBufferAddress(void) { return 0ull; }
__attribute__((weak)) uint32_t SparkWeightdMeshBufferLkey(void) { return 0u; }
__attribute__((weak)) int SparkWeightdMeshBufferFd(void) { return -1; }

extern void SparkWeightdMeshDeviceProbe(const char *tag,void *device_pointer,
    uint64_t bytes);
__attribute__((weak)) void SparkWeightdMeshDeviceProbe(const char *tag,
    void *device_pointer,uint64_t bytes)
{
    (void)tag; (void)device_pointer; (void)bytes;
}
__attribute__((weak)) void SparkWeightdMeshPoll(void) { }
__attribute__((weak)) SparkStatus SparkWeightdMeshLaneConfigure(uint32_t lane,
    const SparkWeightdMeshTopology *topology)
{
    (void)lane; (void)topology;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) SparkStatus SparkWeightdMeshSetActivity(uint32_t lane,uint32_t active)
{
    (void)lane; (void)active;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) SparkStatus SparkWeightdMeshPostWrite(uint32_t peer,
    uint64_t local_addr, uint32_t lkey, uint32_t length,
    uint64_t remote_offset)
{
    (void)peer;(void)local_addr;(void)lkey;(void)length;(void)remote_offset;
    return SPARK_STATUS_UNSUPPORTED;
}
__attribute__((weak)) uint32_t SparkWeightdMeshBroadcast(uint32_t peer_mask,
    uint64_t source_offset, uint32_t length, uint64_t remote_offset,
    uint64_t seq_value, uint64_t seq_remote_offset)
{
    (void)peer_mask;(void)source_offset;(void)length;(void)remote_offset;
    (void)seq_value;(void)seq_remote_offset;
    return 0u;
}

#include <cuda_runtime.h>
#include <cuda.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"

#define SPARK_WEIGHTD_LOAD_CHUNK_BYTES (64ull * 1024ull * 1024ull)

/* The VMM page law (docs/WEIGHTD_DESIGN.md): a 25-100 GiB arena must not
 * drown the TLB in 4 KiB pages. Physical chunks are created at the driver's
 * recommended allocation granularity but never below 2 MiB, and every chunk
 * size stays a multiple of that granularity (cuMemCreate requires it). */
#define SPARK_WEIGHTD_VMM_CHUNK_BYTES (64ull * 1024ull * 1024ull)

typedef struct SparkWeightdExpertEntry
{
    uint64_t last_use_ns;
    uint32_t layer;
    uint32_t expert;
    uint32_t present;
} SparkWeightdExpertEntry;

typedef struct SparkWeightdArena
{
    SparkWeightdIdentity identity; /* canonical (prepared) */
    void *device_base;             /* mapped VMM virtual base (stable VA) */
    uint64_t virtual_bytes;        /* reserved span: chunk_count * chunk_bytes */
    uint64_t chunk_bytes;          /* physical chunk granularity */
    void **chunk_handles;          /* cuMem physical handles, one per chunk */
    uint32_t *chunk_refs;          /* committed-expert sharers, one per chunk */
    uint32_t chunk_count;
    uint64_t generation;
    uint32_t refcount;
    uint32_t lazy;
    uint64_t expert_pool_bytes;
    uint64_t preload_chunk_bytes;
    uint64_t pool_committed_bytes;
    uint64_t expert_present_bytes;
    uint32_t expert_count;
    struct stat pack_stat;
    char pack_path[SPARK_WEIGHTD_PATH_BYTES];
    SparkWeightdExpertEntry *experts;
    SparkWeightdManifest manifest;
    SparkStatus failure_status;
    SparkWeightdLeaseTable *leases;
    uint8_t *needed_chunks;
    uint8_t *created_chunks;
    void *epoch_device;
    void *epoch_handle;
    uint64_t epoch;
    void *pool_export_handle;
    uint8_t *staging;
    uint64_t *recorded_keys;
    uint32_t recorded_count;
    uint32_t recorded_capacity;
} SparkWeightdArena;

#define SPARK_WEIGHTD_ARENA_STAGING_BYTES (4ull * 1024ull * 1024ull)

static SparkStatus SparkWeightdLoadRecordedWorkingSet(SparkWeightdArena *arena);

typedef struct SparkWeightdAttachRef
{
    uint64_t arena_generation;
    uint32_t arena_slot;
    uint32_t reserved0;
} SparkWeightdAttachRef;

typedef enum SparkWeightdConnectionState
{
    SPARK_WEIGHTD_CONNECTION_OPEN = 0,
    SPARK_WEIGHTD_CONNECTION_CLOSED = 1
} SparkWeightdConnectionState;

typedef struct SparkWeightdConnection
{
    int fd;
    uint32_t state;
    uint32_t hello_done;
    uint64_t owner;
    uint64_t mesh_generation;
    uint32_t mesh_active;
    uint32_t mesh_lane;
    uint32_t request_bytes;
    uint32_t request_ready;
    uint32_t attach_count;
    _Alignas(SparkWeightdIpcHeader) uint8_t request[SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX];
    _Alignas(SparkWeightdIpcHeader) uint8_t response[SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX];
    uint32_t response_bytes;
    uint32_t response_written;
    /* W3 fd tier: the EXPORT_RESULT reply leaves with the chunk fds in its
     * SCM_RIGHTS ancillary. The fds are staged here until the first
     * completed send (the kernel takes its own references at send time, so
     * they are handed across EXACTLY once) and are ours to close on every
     * other path — flush error, connection death, server teardown. */
    uint32_t response_fd_count;
    int response_fds[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
    uint32_t lane_mask;
    SparkWeightdAttachRef attaches[SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX];
} SparkWeightdConnection;

struct SparkWeightdServer
{
    SparkWeightdServerConfig config;
    char socket_path[SPARK_WEIGHTD_SOCKET_PATH_BYTES];
    int listen_fd;
    int dispatch_notify[2];
    SparkWeightdWorker *worker;
    atomic_uint dispatch_state;
    uint32_t dispatch_connection;
    uint32_t dispatch_next;
    atomic_uint published_arena_count;
    atomic_ullong published_resident_bytes;
    uint64_t daemon_generation;
    uint64_t next_arena_generation;
    uint64_t next_owner;
    uint32_t orphan_mesh_owners;
    uint32_t arena_count;
    uint64_t resident_bytes;
    uint16_t lane_owner[SPARK_WEIGHTD_MESH_MAX_LANES];
    SparkWeightdArena arenas[SPARK_WEIGHTD_ARENA_COUNT_MAX];
    SparkWeightdConnection connections[SPARK_WEIGHTD_CONNECTION_COUNT_MAX];
};

struct SparkWeightdClient
{
    atomic_int fd;
    uint64_t next_request_id;
    uint64_t daemon_generation;
    uint32_t lane;
    atomic_uint lane_bands;
    SparkWeightdMeshTopology topology;
};

uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client)
{
    struct pollfd pfd;
    char probe;
    ssize_t got;
    if ( client == 0 || client->fd < 0 )
        return(0u);
    memset(&pfd,0,sizeof(pfd));
    pfd.fd = client->fd;
    pfd.events = POLLIN | POLLHUP | POLLERR | POLLRDHUP;
    if ( poll(&pfd,1,0) < 0 )
        return(0u);
    if ( (pfd.revents & (POLLHUP | POLLERR | POLLRDHUP)) != 0 )
        return(0u);
    if ( (pfd.revents & POLLIN) != 0 )
    {
        got = recv(client->fd,&probe,1u,MSG_PEEK | MSG_DONTWAIT);
        if ( got == 0 )
            return(0u);
    }
    return(1u);
}

static uint64_t SparkWeightdMonotonicTimeNs(void)
{
    struct timespec now;
    if (clock_gettime(CLOCK_MONOTONIC, &now) != 0)
    {
        return 0ull;
    }
    return (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
}

/* The stat change-detector for the verify/load window: size + mtime with
 * whatever nanosecond field the platform names it. */
static uint64_t SparkWeightdStatMtimeNs(const struct stat *status)
{
#if defined(__APPLE__)
    return (uint64_t)status->st_mtimespec.tv_sec * 1000000000ull +
        (uint64_t)status->st_mtimespec.tv_nsec;
#else
    return (uint64_t)status->st_mtim.tv_sec * 1000000000ull +
        (uint64_t)status->st_mtim.tv_nsec;
#endif
}

static SparkStatus SparkWeightdStringBounded(const char *text, uint32_t capacity)
{
    uint32_t index;
    for (index = 0u; index < capacity; index++)
    {
        if (text[index] == '\0')
        {
            return SPARK_STATUS_OK;
        }
    }
    SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
}

/* ------------------------------ identity ------------------------------ */

SparkStatus SparkWeightdManifestIdentity(const SparkWeightdManifest *manifest,uint8_t digest[32])
{
	SparkSha256Context hash;
	const SparkWeightdRange *range;
	uint8_t record[48] = {0};
	uint32_t i,j,values[4];
	if ( manifest == 0 || digest == 0 || manifest->ranges == 0 || manifest->range_count == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	SparkSha256Initialize(&hash);
	for (i=0u; i<manifest->range_count; i++)
	{
		range = &manifest->ranges[i];
		values[0] = range->layer;
		values[1] = range->expert;
		values[2] = range->kind;
		values[3] = SPARK_WEIGHTD_RANGE_MANIFEST_VERSION;
		for (j=0u; j<16u; j++)
			record[j] = (uint8_t)(values[j / 4u] >> ((j % 4u) * 8u));
		for (j=0u; j<8u; j++)
		{
			record[16u + j] = (uint8_t)(range->offset >> (j * 8u));
			record[24u + j] = (uint8_t)(range->bytes >> (j * 8u));
		}
		memcpy(record + 32u,range->digest,16u);
		SparkSha256Update(&hash,record,sizeof(record));
	}
	SparkSha256Finalize(&hash,digest);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdIdentityPrepare(SparkWeightdIdentity *identity)
{
    uint32_t index;
    size_t model_bytes;
    size_t revision_bytes;

    if (identity == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (identity->abi_version == 0u || identity->arena_bytes == 0ull)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (SparkWeightdStringBounded(identity->model, SPARK_WEIGHTD_ID_BYTES) !=
        SPARK_STATUS_OK)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (SparkWeightdStringBounded(identity->revision,
            SPARK_WEIGHTD_REVISION_BYTES) != SPARK_STATUS_OK)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (!SparkSha256HexIsValid(identity->pack_sha256))
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    model_bytes = strlen(identity->model) + 1u;
    revision_bytes = strlen(identity->revision) + 1u;
    identity->reserved0 = 0u;
    identity->reserved1 = 0u;
    memset(identity->reserved_tail, 0, sizeof(identity->reserved_tail));
    for (index = (uint32_t)model_bytes; index < SPARK_WEIGHTD_ID_BYTES; index++)
    {
        identity->model[index] = '\0';
    }
    for (index = (uint32_t)revision_bytes;
        index < SPARK_WEIGHTD_REVISION_BYTES; index++)
    {
        identity->revision[index] = '\0';
    }
    return SPARK_STATUS_OK;
}

bool SparkWeightdIdentityEqual(const SparkWeightdIdentity *left,
    const SparkWeightdIdentity *right)
{
    if (left == 0 || right == 0)
    {
        return false;
    }
    return memcmp(left, right, sizeof(*left)) == 0;
}

/* ------------------------------ wire helpers ------------------------------ */

static uint32_t SparkWeightdKindBodyBytes(uint32_t kind)
{
    switch (kind)
    {
        case SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY:
            return sizeof(SparkWeightdIpcMeshActivity) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY_RESULT:
            return sizeof(SparkWeightdIpcMeshActivityResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_MESH_WRITE:
            return sizeof(SparkWeightdIpcMeshWrite) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_MESH_WRITE_RESULT:
            return sizeof(SparkWeightdIpcMeshWriteResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST:
            return sizeof(SparkWeightdIpcMeshBroadcast) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST_RESULT:
            return sizeof(SparkWeightdIpcMeshBroadcastResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE:
            return sizeof(SparkWeightdIpcExportLease) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT:
            return sizeof(SparkWeightdIpcExportLeaseResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT:
            return sizeof(SparkWeightdIpcEpochExport) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE:
            return sizeof(SparkWeightdIpcLaneAcquire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE_RESULT:
            return sizeof(SparkWeightdIpcLaneAcquireResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EVICT:
            return SPARK_WEIGHTD_IPC_EVICT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EVICT_RESULT:
            return SPARK_WEIGHTD_IPC_EVICT_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT_RESULT:
            return sizeof(SparkWeightdIpcEpochExportResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ACQUIRE:
            return sizeof(SparkWeightdIpcAcquire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ACQUIRE_RESULT:
        case SPARK_WEIGHTD_IPC_KIND_RELEASE_RESULT:
            return sizeof(SparkWeightdIpcAcquireResult) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_RELEASE:
            return sizeof(SparkWeightdIpcRelease) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_HELLO:
            return SPARK_WEIGHTD_IPC_HELLO_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_HELLO_ACK:
            return SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH:
            return SPARK_WEIGHTD_IPC_ATTACH_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH_RESULT:
            return SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_DETACH:
            return SPARK_WEIGHTD_IPC_DETACH_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_DETACH_RESULT:
            return SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_RECLAIM:
            return SPARK_WEIGHTD_IPC_RECLAIM_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_RECLAIM_RESULT:
            return SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EXPORT:
            return SPARK_WEIGHTD_IPC_EXPORT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_EXPORT_RESULT:
            return SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY:
            return SPARK_WEIGHTD_IPC_ATTACH_LAZY_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_RESULT:
            return SPARK_WEIGHTD_IPC_ATTACH_LAZY_RESULT_BYTES -
                SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ENSURE:
            return SPARK_WEIGHTD_IPC_ENSURE_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        case SPARK_WEIGHTD_IPC_KIND_ENSURE_RESULT:
            return SPARK_WEIGHTD_IPC_ENSURE_RESULT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        default:
            return 0u;
    }
}

static uint32_t SparkWeightdKindResultKind(uint32_t kind)
{
    switch (kind)
    {
        case SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE:
            return SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT:
            return SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE:
            return SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_EVICT:
            return SPARK_WEIGHTD_IPC_KIND_EVICT_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_ACQUIRE:
            return SPARK_WEIGHTD_IPC_KIND_ACQUIRE_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_RELEASE:
            return SPARK_WEIGHTD_IPC_KIND_RELEASE_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_HELLO:
            return SPARK_WEIGHTD_IPC_KIND_HELLO_ACK;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH:
            return SPARK_WEIGHTD_IPC_KIND_ATTACH_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_DETACH:
            return SPARK_WEIGHTD_IPC_KIND_DETACH_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_RECLAIM:
            return SPARK_WEIGHTD_IPC_KIND_RECLAIM_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_EXPORT:
            return SPARK_WEIGHTD_IPC_KIND_EXPORT_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY:
            return SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_MESH_WRITE:
            return SPARK_WEIGHTD_IPC_KIND_MESH_WRITE_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST:
            return SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY:
            return SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_RESULT;
        case SPARK_WEIGHTD_IPC_KIND_ENSURE:
            return SPARK_WEIGHTD_IPC_KIND_ENSURE_RESULT;
        default:
            return 0u;
    }
}

static void SparkWeightdBuildHeader(uint8_t *response,
    uint32_t kind,
    uint64_t request_id)
{
    SparkWeightdIpcHeader *header = (SparkWeightdIpcHeader *)response;
    header->magic = SPARK_WEIGHTD_IPC_MAGIC;
    header->abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    header->kind = kind;
    header->body_bytes = SparkWeightdKindBodyBytes(kind);
    header->request_id = request_id;
}

SparkStatus SparkWeightdIpcValidateHeader(const SparkWeightdIpcHeader *header,
    uint32_t message_bytes,
    uint32_t expected_kind)
{
    if (header == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (message_bytes < SPARK_WEIGHTD_IPC_HEADER_BYTES)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (header->magic != SPARK_WEIGHTD_IPC_MAGIC)
    {
        SPARK_FAIL(SPARK_STATUS_PARSE_ERROR);
    }
    if (header->abi_version != SPARK_WEIGHTD_IPC_ABI_VERSION)
    {
        SPARK_FAIL(SPARK_STATUS_ABI_MISMATCH);
    }
    if (header->kind != expected_kind)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (header->body_bytes != SparkWeightdKindBodyBytes(expected_kind))
    {
        SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
    }
    if (message_bytes != SPARK_WEIGHTD_IPC_HEADER_BYTES + header->body_bytes)
    {
        SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdStatusFromWire(uint32_t wire_status)
{
    if (wire_status > (uint32_t)SPARK_STATUS_EVICT_DENIED)
    {
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    return (SparkStatus)wire_status;
}

static uint32_t SparkWeightdConnectionLane(
    const SparkWeightdConnection *connection)
{
    uint32_t lane;
    if (connection->lane_mask == 0u ||
        (connection->lane_mask & (connection->lane_mask - 1u)) != 0u)
    {
        return SPARK_WEIGHTD_LANE_NONE;
    }
    for (lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
    {
        if ((connection->lane_mask & (uint16_t)(1u << lane)) != 0u)
        {
            return lane;
        }
    }
    return SPARK_WEIGHTD_LANE_NONE;
}

/* ------------------------------ server: VMM arenas ------------------------------ */

/* W2b (docs/WEIGHTD_DESIGN.md): arenas are cuMem* VMM virtual arenas, not
 * cudaMalloc blocks. The virtual span is reserved ONCE and stays put for the
 * arena's whole life; the physical backing is a vector of independent
 * 2 MiB-granular cuMemCreate handles mapped into that span one by one. A
 * later tier can therefore attach or detach individual physical chunks (or
 * export them POSIX-fd for a consumer's read-only import + map) without
 * moving the base or copying a byte - the pointer IS the weight. */

static uint64_t SparkWeightdVmmRoundUp(uint64_t value, uint64_t multiple)
{
    return (value + multiple - 1ull) / multiple * multiple;
}

static void SparkWeightdVmmRelease(SparkWeightdArena *arena)
{
    CUdeviceptr base = (CUdeviceptr)(uintptr_t)arena->device_base;
    uint32_t index;
    /* full teardown of a mapped arena; nothing here can recover, so every
     * result is ignored - the stub's leak ledger catches ordering bugs in
     * the host tests. Lazy arenas unmap per COMMITTED chunk: handles left
     * null were never created. */
    if (base != 0)
    {
        if (arena->pool_export_handle != 0)
        {
            (void)cuMemUnmap(base,
                (size_t)(arena->chunk_bytes * arena->chunk_count));
        }
        else
        {
            for (index = 0u; index < arena->chunk_count; index++)
            {
                if (arena->chunk_handles != 0 && arena->chunk_handles[index] != 0)
                {
                    (void)cuMemUnmap(base + (CUdeviceptr)index * arena->chunk_bytes,
                        (size_t)arena->chunk_bytes);
                }
            }
        }
    }
    if (base != 0 && arena->epoch_device != 0)
    {
        (void)cuMemUnmap(
            (CUdeviceptr)(uintptr_t)arena->epoch_device,
            (size_t)arena->chunk_bytes);
    }
    if (arena->epoch_handle != 0)
    {
        (void)cuMemRelease(
            (CUmemGenericAllocationHandle)arena->epoch_handle);
        arena->epoch_handle = 0;
    }
    {
        void *pool = arena->pool_export_handle;
        if (pool != 0)
            (void)cuMemRelease((CUmemGenericAllocationHandle)pool);
        arena->pool_export_handle = 0;
        if (arena->chunk_handles != 0)
        {
            for (index = 0u; index < arena->chunk_count; index++)
            {
                if (arena->chunk_handles[index] != 0 &&
                    arena->chunk_handles[index] != pool)
                {
                    (void)cuMemRelease(
                        (CUmemGenericAllocationHandle)arena->chunk_handles[index]);
                }
            }
            free(arena->chunk_handles);
        }
    }
    free(arena->chunk_refs);
    free(arena->staging);
    free(arena->recorded_keys);
    arena->recorded_keys = 0;
    arena->recorded_count = 0u;
    arena->recorded_capacity = 0u;
    arena->staging = 0;
    if (base != 0)
    {
        (void)cuMemAddressFree(base, (size_t)arena->virtual_bytes);
    }
    arena->chunk_handles = 0;
    arena->chunk_refs = 0;
    arena->chunk_count = 0u;
    arena->device_base = 0;
    arena->virtual_bytes = 0ull;
    arena->chunk_bytes = 0ull;
    arena->epoch_device = 0;
}

static SparkStatus SparkWeightdVmmAllocate(uint64_t arena_bytes,
    SparkWeightdArena *arena)
{
    CUmemAllocationProp prop;
    CUmemAccessDesc access;
    CUdeviceptr base = 0;
    size_t granularity = 0;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint32_t created = 0u;
    uint32_t mapped = 0u;
    uint32_t index;
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    }
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    /* the shareable-handle shape the consumer import+map tier attaches
     * through; requesting it costs nothing today */
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
            CU_MEM_ALLOC_GRANULARITY_RECOMMENDED) != CUDA_SUCCESS ||
        granularity == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    }
    chunk_bytes = SparkWeightdVmmRoundUp(
        (uint64_t)granularity < SPARK_WEIGHTD_VMM_CHUNK_BYTES
            ? SPARK_WEIGHTD_VMM_CHUNK_BYTES
            : (uint64_t)granularity,
        (uint64_t)granularity);
    chunk_count = (uint32_t)((arena_bytes + chunk_bytes - 1ull) / chunk_bytes);

    arena->chunk_handles = (void **)calloc(chunk_count, sizeof(void *));
    arena->chunk_refs = (uint32_t *)calloc(chunk_count, sizeof(uint32_t));
    if (arena->chunk_handles == 0 || arena->chunk_refs == 0)
    {
        free(arena->chunk_handles);
        free(arena->chunk_refs);
        arena->chunk_handles = 0;
        arena->chunk_refs = 0;
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    if (cuMemAddressReserve(&base,
            (size_t)(chunk_count * chunk_bytes), 0u, 0ull, 0ull) != CUDA_SUCCESS)
    {
        free(arena->chunk_handles);
        free(arena->chunk_refs);
        arena->chunk_handles = 0;
        arena->chunk_refs = 0;
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    for (index = 0u; index < chunk_count; index++)
    {
        CUmemGenericAllocationHandle handle = 0;
        if (cuMemCreate(&handle, (size_t)chunk_bytes, &prop, 0ull) !=
            CUDA_SUCCESS)
        {
            break;
        }
        arena->chunk_handles[index] = (void *)handle;
        created++;
        if (cuMemMap(base + (CUdeviceptr)index * chunk_bytes,
                (size_t)chunk_bytes, 0u, handle, 0ull) != CUDA_SUCCESS)
        {
            break;
        }
        mapped++;
    }
    access.location.type = prop.location.type;
    access.location.id = prop.location.id;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    if (mapped == chunk_count &&
        cuMemSetAccess(base, (size_t)(chunk_count * chunk_bytes), &access,
            1u) == CUDA_SUCCESS)
    {
        arena->device_base = (void *)(uintptr_t)base;
        arena->virtual_bytes = chunk_count * chunk_bytes;
        arena->chunk_bytes = chunk_bytes;
        arena->chunk_count = chunk_count;
        arena->staging = (uint8_t *)malloc(SPARK_WEIGHTD_ARENA_STAGING_BYTES);
        if (arena->staging != 0)
        {
            return SPARK_STATUS_OK;
        }
    }
    /* unwind: unmap what was mapped, release what was created, free the VA */
    if (mapped != 0u)
    {
        (void)cuMemUnmap(base, (size_t)(mapped * chunk_bytes));
    }
    for (index = 0u; index < created; index++)
    {
        (void)cuMemRelease((CUmemGenericAllocationHandle)arena->chunk_handles[index]);
    }
    (void)cuMemAddressFree(base, (size_t)(chunk_count * chunk_bytes));
    free(arena->chunk_handles);
    free(arena->chunk_refs);
    arena->chunk_handles = 0;
    arena->chunk_refs = 0;
    SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
}

static SparkStatus SparkWeightdVmmReserve(uint64_t arena_bytes,
    SparkWeightdArena *arena)
{
    CUmemAllocationProp prop;
    size_t granularity = 0;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    CUdeviceptr base = 0;
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    }
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    if (cuMemGetAllocationGranularity(&granularity, &prop,
            CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
        granularity == 0u)
    {
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    }
    chunk_bytes = (uint64_t)granularity < (2ull * 1024ull * 1024ull)
        ? (2ull * 1024ull * 1024ull)
        : (uint64_t)granularity;
    chunk_count = (uint32_t)((arena_bytes + chunk_bytes - 1ull) / chunk_bytes);
    arena->chunk_handles = (void **)calloc(chunk_count, sizeof(void *));
    arena->chunk_refs = (uint32_t *)calloc(chunk_count, sizeof(uint32_t));
    if (arena->chunk_handles == 0 || arena->chunk_refs == 0)
    {
        free(arena->chunk_handles);
        free(arena->chunk_refs);
        arena->chunk_handles = 0;
        arena->chunk_refs = 0;
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    {
        CUmemAccessDesc access_stub;
        memset(&access_stub,0,sizeof(access_stub));
        access_stub.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access_stub.location.id = device;
        access_stub.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
        if (cuMemAddressReserve(&base,
                (size_t)((chunk_count + 1ull) * chunk_bytes), 0u, 0ull,
                0ull) != CUDA_SUCCESS ||
            cuMemCreate((CUmemGenericAllocationHandle *)
                &arena->epoch_handle,(size_t)chunk_bytes,&prop,0ull) !=
                CUDA_SUCCESS ||
            cuMemMap(base + (CUdeviceptr)chunk_count * chunk_bytes,
                (size_t)chunk_bytes,0u,
                (CUmemGenericAllocationHandle)arena->epoch_handle,0ull) !=
                CUDA_SUCCESS ||
            cuMemSetAccess(base + (CUdeviceptr)chunk_count * chunk_bytes,
                (size_t)chunk_bytes,&access_stub,1u) != CUDA_SUCCESS)
        {
            if ( base != 0 )
                (void)cuMemAddressFree(base,
                    (size_t)((chunk_count + 1ull) * chunk_bytes));
            if ( arena->epoch_handle != 0 )
            {
                (void)cuMemRelease((CUmemGenericAllocationHandle)
                    arena->epoch_handle);
                arena->epoch_handle = 0;
            }
            free(arena->chunk_handles);
            free(arena->chunk_refs);
            arena->chunk_handles = 0;
            arena->chunk_refs = 0;
            SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
        }
        arena->epoch_device = (void *)(uintptr_t)
            (base + (CUdeviceptr)chunk_count * chunk_bytes);
        arena->epoch = 0ull;
    }
    arena->device_base = (void *)(uintptr_t)base;
    arena->virtual_bytes = chunk_count * chunk_bytes;
    arena->chunk_bytes = chunk_bytes;
    arena->chunk_count = chunk_count;
    arena->staging = (uint8_t *)malloc(SPARK_WEIGHTD_ARENA_STAGING_BYTES);
    if (arena->staging == 0)
    {
        SparkWeightdVmmRelease(arena);
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    return SPARK_STATUS_OK;
}

/* ------------------------------ server: arena map ------------------------------ */

static SparkWeightdArena *SparkWeightdServerFindArena(SparkWeightdServer *server,
    const SparkWeightdIdentity *identity)
{
    uint32_t index;
    for (index = 0u; index < server->arena_count; index++)
    {
        if (SparkWeightdIdentityEqual(&server->arenas[index].identity, identity))
        {
            return &server->arenas[index];
        }
    }
    return 0;
}

/* Free the arena at `slot` (the VMM span's physical chunks released, the
 * virtual reservation torn down) and close the slot by moving the tail arena
 * into it. Content-addressed means no external order to preserve — but every
 * connection attach reference that named the MOVED arena must follow it, or
 * its detach would miss and pin the refcount forever. */
static uint32_t SparkWeightdArenaHasOwnerLeases(const SparkWeightdArena *arena,uint64_t owner)
{
	uint32_t i;
	if ( arena->leases == 0 )
		return(0u);
	for (i=0u; i<SPARK_WEIGHTD_LEASE_COUNT_MAX; i++)
		if ( arena->leases->leases[i].count != 0u && (owner == 0u || arena->leases->leases[i].owner == owner) )
			return(1u);
	return(0u);
}

static uint32_t SparkWeightdArenaHasLeases(const SparkWeightdArena *arena)
{
	return(SparkWeightdArenaHasOwnerLeases(arena,0u));
}

static void SparkWeightdServerFreeArenaSlot(SparkWeightdServer *server,
    uint32_t slot)
{
    uint64_t moved_generation;
    uint32_t moved_slot;
    uint32_t index;
    if (server->arenas[slot].device_base != 0)
    {
        SparkWeightdVmmRelease(&server->arenas[slot]);
    }
    if (server->arenas[slot].lazy != 0u)
    {
        server->resident_bytes -= server->arenas[slot].pool_committed_bytes;
    }
    else
    {
        server->resident_bytes -= server->arenas[slot].identity.arena_bytes;
    }
    free(server->arenas[slot].experts);
    if ( server->arenas[slot].leases != 0 )
    {
        /* Only terminal server teardown may reach this with active leases.
         * Imported CUDA handles retain physical backing after daemon exit. */
        free(server->arenas[slot].leases->pins);
        free(server->arenas[slot].leases);
    }
    SparkWeightdManifestDestroy(&server->arenas[slot].manifest);
    free(server->arenas[slot].needed_chunks);
    free(server->arenas[slot].created_chunks);
    server->arena_count--;
    if (slot == server->arena_count)
    {
        memset(&server->arenas[slot], 0, sizeof(server->arenas[slot]));
        return;
    }
    moved_generation = server->arenas[server->arena_count].generation;
    server->arenas[slot] = server->arenas[server->arena_count];
    if ( server->arenas[slot].leases != 0 )
        server->arenas[slot].leases->manifest = &server->arenas[slot].manifest;
    memset(&server->arenas[server->arena_count], 0,
        sizeof(server->arenas[server->arena_count]));
    moved_slot = slot;
    for (index = 0u; index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX; index++)
    {
        SparkWeightdConnection *connection = &server->connections[index];
        uint32_t attach_index;
        for (attach_index = 0u; attach_index < connection->attach_count;
            attach_index++)
        {
            if (connection->attaches[attach_index].arena_generation ==
                    moved_generation &&
                connection->attaches[attach_index].arena_slot !=
                    moved_slot)
            {
                connection->attaches[attach_index].arena_slot = moved_slot;
            }
        }
    }
}

/* Reclaim cold (refcount == 0) arenas, oldest generation first, until
 * `needed_bytes` fits under the ceiling. Live arenas are NEVER evicted —
 * that is the NO-2x law: a serving process's weights cannot be pulled out
 * from under it, and an update that would need that goes through
 * stop-attach-start or fails closed. */
static void SparkWeightdServerReclaimCold(SparkWeightdServer *server,
    uint64_t needed_bytes)
{
    while (server->resident_bytes > server->config.device_bytes_max ||
        needed_bytes > server->config.device_bytes_max - server->resident_bytes)
    {
        uint32_t oldest_slot = server->arena_count;
        uint32_t index;
        for (index = 0u; index < server->arena_count; index++)
        {
            if (server->arenas[index].refcount == 0u && SparkWeightdArenaHasLeases(&server->arenas[index]) == 0u &&
                (oldest_slot == server->arena_count ||
                    server->arenas[index].generation <
                        server->arenas[oldest_slot].generation))
            {
                oldest_slot = index;
            }
        }
        if (oldest_slot == server->arena_count)
        {
            return; /* nothing cold left; the caller fails closed */
        }
        SparkWeightdServerFreeArenaSlot(server, oldest_slot);
    }
}

static SparkStatus SparkWeightdServerAttachRegister(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    uint32_t arena_slot)
{
    SparkWeightdArena *arena = &server->arenas[arena_slot];
    uint32_t index;
    for (index = 0u; index < connection->attach_count; index++)
    {
        if (connection->attaches[index].arena_generation == arena->generation)
        {
            /* one attach per identity per connection: a serving process
             * maps the arena once; a second claim is a protocol error */
            SPARK_FAIL(SPARK_STATUS_DUPLICATE);
        }
    }
    if (connection->attach_count >= SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX)
    {
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    arena->refcount++;
    connection->attaches[connection->attach_count].arena_generation =
        arena->generation;
    connection->attaches[connection->attach_count].arena_slot = arena_slot;
    connection->attach_count++;
    return SPARK_STATUS_OK;
}

static void SparkWeightdServerDetachRelease(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    uint32_t attach_index)
{
    uint32_t arena_slot = connection->attaches[attach_index].arena_slot;
    uint64_t generation = connection->attaches[attach_index].arena_generation;
    uint32_t index;
    if (arena_slot < server->arena_count &&
        server->arenas[arena_slot].generation == generation &&
        server->arenas[arena_slot].refcount != 0u)
    {
        server->arenas[arena_slot].refcount--;
    }
    for (index = attach_index; index + 1u < connection->attach_count; index++)
    {
        connection->attaches[index] = connection->attaches[index + 1u];
    }
    connection->attach_count--;
}

/* ------------------------------ server: load path ------------------------------ */

static SparkStatus SparkWeightdSidecarCk128(const char *pack_path,
    char hex[SPARK_CK128_HEX_BYTES])
{
    char sidecar_path[SPARK_WEIGHTD_PATH_BYTES + 8];
    FILE *sidecar;
    int written;
    uint32_t index;

    written = snprintf(sidecar_path, sizeof(sidecar_path), "%s.ck128", pack_path);
    if (written <= 0 || (size_t)written >= sizeof(sidecar_path))
    {
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    sidecar = fopen(sidecar_path, "rb");
    if (sidecar == 0)
    {
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    if (fread(hex, 1u, 32u, sidecar) != 32u || hex[0] == '\0')
    {
        (void)fclose(sidecar);
        SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
    }
    hex[32] = '\0';
    (void)fclose(sidecar);
    for (index = 0u; index < 32u; index++)
    {
        char c = hex[index];
        if ((c < '0' || c > '9') && (c < 'a' || c > 'f'))
        {
            SPARK_FAIL(SPARK_STATUS_NOT_FOUND);
        }
    }
    return SPARK_STATUS_OK;
}

/* The cold path: verify-then-load, all through ONE open file so the bytes
 * loaded are the bytes digested, with a size+mtime re-check after the load
 * closing the window against an in-flight pack rewrite. */
static void SparkWeightdServerAttachCold(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    const SparkWeightdIpcAttach *request,
    SparkWeightdIpcAttachResult *result)
{
    SparkWeightdIdentity identity = request->identity;
    SparkWeightdArena *arena;
    SparkWeightdArena pending;
    char expected_hex[SPARK_SHA256_HEX_BYTES];
    char computed_hex[SPARK_SHA256_HEX_BYTES];
    struct stat pack_stat_before;
    struct stat pack_stat_after;
    FILE *file = 0;
    uint8_t *staging = 0;
    uint8_t materialized_sha[SPARK_SHA256_DIGEST_BYTES];
    SparkCk128Context ck_context;
    SparkSha256Context sha_context;
    SparkStatus verify_status;
    int use_ck128;
    SparkStatus status;
    uint64_t loaded = 0ull;
    uint64_t load_start_ns;
    uint64_t load_end_ns;
    uint32_t slot;

    memset(&pending, 0, sizeof(pending));

    result->resident_bytes = server->resident_bytes;
    result->arena_count = server->arena_count;

    status = SparkWeightdIdentityPrepare(&identity);
    if (status != SPARK_STATUS_OK)
    {
        result->status = (uint32_t)status;
        return;
    }
    if (SparkWeightdStringBounded(request->pack_path,
            SPARK_WEIGHTD_PATH_BYTES) != SPARK_STATUS_OK)
    {
        result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        return;
    }

    /* warm hit: the sub-second path (a code-only redeploy lands here) */
    arena = SparkWeightdServerFindArena(server, &identity);
    if (arena != 0)
    {
        if (arena->lazy != 0u)
        {
            result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
            return;
        }
        slot = (uint32_t)(arena - server->arenas);
        status = SparkWeightdServerAttachRegister(server, connection, slot);
        result->status = (uint32_t)status;
        if (status == SPARK_STATUS_OK)
        {
            result->loaded_from_pack = 0u;
            result->refcount = server->arenas[slot].refcount;
            result->arena_generation = server->arenas[slot].generation;
            result->device_handle =
                (uint64_t)(uintptr_t)server->arenas[slot].device_base;
            result->arena_bytes = server->arenas[slot].identity.arena_bytes;
        }
        return;
    }

    /* cold: claim vs disk, on one open handle */
    file = fopen(request->pack_path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "WDATTACH fopen-fail path=%s errno=%d\n",
            request->pack_path, errno);
        result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
        return;
    }
    if (fstat(fileno(file), &pack_stat_before) != 0 ||
        pack_stat_before.st_size < 0 ||
        (uint64_t)pack_stat_before.st_size != identity.arena_bytes)
    {
        fprintf(stderr, "WDATTACH size-mismatch path=%s size=%lld arena=%llu\n",
            request->pack_path, (long long)pack_stat_before.st_size,
            (unsigned long long)identity.arena_bytes);
        (void)fclose(file);
        result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        return;
    }
    use_ck128 = SparkWeightdSidecarCk128(request->pack_path, expected_hex) ==
        SPARK_STATUS_OK;
    if (!use_ck128)
    {
        memcpy(expected_hex, identity.pack_sha256, SPARK_SHA256_HEX_BYTES);
    }

    /* the NO-2x gate: make room by reclaiming COLD arenas only; if a live
     * arena still blocks the fit, fail closed with nothing allocated — the
     * update then goes through stop (detach) before attach, per the design */
    SparkWeightdServerReclaimCold(server, identity.arena_bytes);
    if (server->resident_bytes + identity.arena_bytes >
            server->config.device_bytes_max ||
        server->arena_count >= SPARK_WEIGHTD_ARENA_COUNT_MAX)
    {
        (void)fclose(file);
        result->status = (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED;
        result->resident_bytes = server->resident_bytes;
        result->arena_count = server->arena_count;
        return;
    }

    /* the VMM arena: a reserved virtual span carrying physical chunks
     * mapped read-write for the load below. The identity — not the
     * allocation — is the contract; the span's chunks can be attached or
     * detached later without moving the base (docs/WEIGHTD_DESIGN.md). */
    if (SparkWeightdVmmAllocate(identity.arena_bytes, &pending) !=
        SPARK_STATUS_OK)
    {
        (void)fclose(file);
        result->status = (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED;
        return;
    }
    staging = (uint8_t *)malloc(SPARK_WEIGHTD_LOAD_CHUNK_BYTES);
    if (staging == 0)
    {
        (void)fclose(file);
        SparkWeightdVmmRelease(&pending);
        result->status = (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED;
        return;
    }
    load_start_ns = SparkWeightdMonotonicTimeNs();
    if (use_ck128)
    {
        SparkCk128Initialize(&ck_context);
    }
    else
    {
        SparkSha256Initialize(&sha_context);
    }
    while (loaded < identity.arena_bytes)
    {
        uint64_t remaining = identity.arena_bytes - loaded;
        size_t chunk = remaining < SPARK_WEIGHTD_LOAD_CHUNK_BYTES
            ? (size_t)remaining
            : (size_t)SPARK_WEIGHTD_LOAD_CHUNK_BYTES;
        if (fread(staging, 1u, chunk, file) != chunk)
        {
            free(staging);
            (void)fclose(file);
            SparkWeightdVmmRelease(&pending);
            result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
            return;
        }
        if (use_ck128)
        {
            SparkCk128Update(&ck_context, staging, chunk);
        }
        else
        {
            SparkSha256Update(&sha_context, staging, chunk);
        }
        if (cudaMemcpy((void *)((uint8_t *)pending.device_base + loaded),
                staging, chunk, cudaMemcpyHostToDevice) != cudaSuccess)
        {
            free(staging);
            (void)fclose(file);
            SparkWeightdVmmRelease(&pending);
            result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
            return;
        }
        loaded += (uint64_t)chunk;
    }
    free(staging);
    if (use_ck128)
    {
        uint8_t digest[16];
        SparkCk128Finalize(&ck_context, digest);
        SparkCk128DigestToHex(digest, computed_hex);
    }
    else
    {
        uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
        SparkSha256Finalize(&sha_context, digest);
        SparkSha256DigestToHex(digest, computed_hex);
        memcpy(materialized_sha, digest, SPARK_SHA256_DIGEST_BYTES);
    }
    verify_status = memcmp(computed_hex, expected_hex, use_ck128 ? 32u : 64u) == 0
        ? SPARK_STATUS_OK
        : SPARK_STATUS_HASH_MISMATCH;
    if (fstat(fileno(file), &pack_stat_after) != 0 ||
        pack_stat_after.st_size != pack_stat_before.st_size ||
        SparkWeightdStatMtimeNs(&pack_stat_after) !=
            SparkWeightdStatMtimeNs(&pack_stat_before))
    {
        verify_status = SPARK_STATUS_HASH_MISMATCH;
    }
    if (verify_status == SPARK_STATUS_OK && !use_ck128)
    {
        /* Spine fast-path receipt (hill-climb, dedicated PR): the daemon
         * just proved SHA256 over the whole pack image against
         * identity.pack_sha256 with the file stat stable across the read.
         * Record the DAEMON_SHA receipt so the client's first spine load
         * copies spans without repeating that hash (strictly redundant;
         * see the trust-chain document). ck128-sidecar mode writes none -
         * there the client hash is the only SHA256 proof. Best-effort:
         * failure only costs the one-time client hash. */
        (void)SparkWeightdSpineReceiptRecordDaemon(fileno(file),
            expected_hex, materialized_sha);
    }
    (void)fclose(file);
    if (verify_status != SPARK_STATUS_OK)
    {
        SparkWeightdVmmRelease(&pending);
        result->status = (uint32_t)verify_status;
        return;
    }
    load_end_ns = SparkWeightdMonotonicTimeNs();
    if (load_end_ns > load_start_ns)
    {
        printf("weightd cold-load bytes=%llu seconds=%.3f gbps=%.2f checksum=%s\n",
            (unsigned long long)identity.arena_bytes,
            (double)(load_end_ns - load_start_ns) / 1e9,
            (double)identity.arena_bytes /
                ((double)(load_end_ns - load_start_ns) / 1e9) / 1e9,
            use_ck128 ? "ck128" : "sha256");
        fflush(stdout);
    }

    slot = server->arena_count;
    memset(&server->arenas[slot], 0, sizeof(server->arenas[slot]));
    server->arenas[slot].identity = identity;
    server->arenas[slot].device_base = pending.device_base;
    server->arenas[slot].virtual_bytes = pending.virtual_bytes;
    server->arenas[slot].chunk_bytes = pending.chunk_bytes;
    server->arenas[slot].chunk_handles = pending.chunk_handles;
    server->arenas[slot].chunk_count = pending.chunk_count;
    server->next_arena_generation++;
    server->arenas[slot].generation = server->next_arena_generation;
    server->arena_count++;
    server->resident_bytes += identity.arena_bytes;

    status = SparkWeightdServerAttachRegister(server, connection, slot);
    if (status != SPARK_STATUS_OK)
    {
        /* the arena loaded but this connection cannot hold it */
        SparkWeightdServerFreeArenaSlot(server, slot);
        result->status = (uint32_t)status;
        result->resident_bytes = server->resident_bytes;
        result->arena_count = server->arena_count;
        return;
    }
    result->status = (uint32_t)SPARK_STATUS_OK;
    result->loaded_from_pack = 1u;
    result->refcount = server->arenas[slot].refcount;
    result->arena_generation = server->arenas[slot].generation;
    result->device_handle = (uint64_t)(uintptr_t)server->arenas[slot].device_base;
    result->arena_bytes = identity.arena_bytes;
    result->resident_bytes = server->resident_bytes;
    result->arena_count = server->arena_count;
}

/* ------------------------------ server: lazy expert tier ------------------------------ */

static SparkStatus SparkWeightdExpertManifestLoad(const char *pack_path,uint64_t arena_bytes,SparkWeightdManifest *manifest,SparkWeightdExpertEntry **entries_out,uint32_t *count_out)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 10];
	SparkWeightdExpertEntry *entries;
	SparkStatus status;
	uint32_t i;
	int32_t written;
	written = snprintf(path,sizeof(path),"%s.experts",pack_path);
	if ( written <= 0 || (uint32_t)written >= sizeof(path) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	status = SparkWeightdManifestLoad(path,arena_bytes,manifest);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	entries = calloc(manifest->group_count,sizeof(*entries));
	if ( entries == 0 )
	{
		SparkWeightdManifestDestroy(manifest);
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	for (i=0u; i<manifest->group_count; i++)
	{
		entries[i].layer = manifest->groups[i].layer;
		entries[i].expert = manifest->groups[i].expert;
	}
	*entries_out = entries;
	*count_out = manifest->group_count;
	return(SPARK_STATUS_OK);
}

static void SparkWeightdServerStageMeshFd(SparkWeightdConnection *connection)
{
    SparkWeightdIpcAttachLazyResult *result = (SparkWeightdIpcAttachLazyResult *)connection->response;
    int fd;
    if (result->status != (uint32_t)SPARK_STATUS_OK)
        return;
    result->mesh_ready = SparkWeightdMeshReady();
    result->mesh_send_buffer_bytes = SPARK_WEIGHTD_MESH_REGION_BYTES;
    /* The daemon's own buffer pointer is meaningless (and lethal) in the
     * client until the mesh fd is staged for a client-side mapping; only
     * a ready mesh publishes an address, and the client then OVERWRITES
     * it with its local mmap of that fd anyway. */
    if (result->mesh_ready != 0u)
        result->mesh_send_buffer_addr = SparkWeightdMeshBufferAddress();
    else
        result->mesh_send_buffer_addr = 0u;
    if (result->mesh_ready == 0u)
        return;
    fd = SparkWeightdMeshBufferFd();
    if (fd < 0 || fcntl(fd,F_SETFD,FD_CLOEXEC) != 0 ||
        connection->response_fd_count >= SPARK_WEIGHTD_EXPORT_BATCH_MAX)
    {
        if (fd >= 0)
            (void)close(fd);
        while (connection->response_fd_count != 0u)
            (void)close(connection->response_fds[--connection->response_fd_count]);
        result->mesh_ready = 0u;
        result->pool_fd_staged = 0u;
        result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
        return;
    }
    memmove(connection->response_fds + 1u,connection->response_fds,
        connection->response_fd_count * sizeof(connection->response_fds[0]));
    connection->response_fds[0] = fd;
    connection->response_fd_count++;
}

static SparkStatus SparkWeightdArenaChunkEnsure(
    SparkWeightdServer *server,SparkWeightdArena *arena,
    uint32_t first_chunk,uint32_t last_chunk);
static SparkStatus SparkWeightdPremapPool(SparkWeightdServer *server,
    SparkWeightdArena *arena)
{
	CUmemAllocationProp prop;
	CUmemAccessDesc access;
	CUmemGenericAllocationHandle handle = 0;
	CUdeviceptr base = (CUdeviceptr)(uintptr_t)arena->device_base;
	size_t granularity = 0;
	uint64_t span_bytes,create_bytes;
	uint32_t index;
	int device = 0;

	if ( arena->expert_pool_bytes <= arena->identity.arena_bytes )
	{
		fprintf(stderr,"pool per-chunk lazy (declared pool=%llu arena=%llu)\n",
		    (unsigned long long)arena->expert_pool_bytes,
		    (unsigned long long)arena->identity.arena_bytes);
		SPARK_RETURN(SPARK_STATUS_OK);
	}
	if ( cudaGetDevice(&device) != cudaSuccess )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	memset(&prop,0,sizeof(prop));
	prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
	prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	prop.location.id = device;
	prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
	if ( cuMemGetAllocationGranularity(&granularity,&prop,
	        CU_MEM_ALLOC_GRANULARITY_MINIMUM) != CUDA_SUCCESS ||
	     granularity == 0u )
		SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
	span_bytes = arena->chunk_bytes * (uint64_t)arena->chunk_count;
	create_bytes = SparkWeightdVmmRoundUp(span_bytes,(uint64_t)granularity);
	if (server->resident_bytes > server->config.device_bytes_max ||
	    create_bytes > server->config.device_bytes_max - server->resident_bytes)
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cuMemCreate(&handle,(size_t)create_bytes,&prop,0ull) != CUDA_SUCCESS )
	{
		fprintf(stderr,"WD-POOL-PREMAP-FAIL single-alloc create span=%llu failed\n",
		    (unsigned long long)create_bytes);
		SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
	}
	if ( cuMemMap(base,(size_t)span_bytes,0u,handle,0ull) != CUDA_SUCCESS )
	{
		fprintf(stderr,"WD-POOL-PREMAP-FAIL map base=%llu span=%llu\n",
		    (unsigned long long)base,(unsigned long long)span_bytes);
		(void)cuMemRelease(handle);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	memset(&access,0,sizeof(access));
	access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
	access.location.id = device;
	access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
	if ( cuMemSetAccess(base,(size_t)span_bytes,&access,1u) != CUDA_SUCCESS )
	{
		fprintf(stderr,"WD-POOL-PREMAP-FAIL access span=%llu\n",
		    (unsigned long long)span_bytes);
		(void)cuMemUnmap(base,(size_t)span_bytes);
		(void)cuMemRelease(handle);
		SPARK_FAIL(SPARK_STATUS_IO_ERROR);
	}
	arena->pool_export_handle = (void *)handle;
	for ( index = 0u; index < arena->chunk_count; index++ )
		arena->chunk_handles[index] = (void *)handle;
	arena->pool_committed_bytes += span_bytes;
	server->resident_bytes += span_bytes;
	SparkWeightdMeshDeviceProbe("pool",arena->device_base,span_bytes);
	printf("pool single-alloc chunks=%u span=%llu (one allocation for the whole arena; pooled chunks never evicted)\n",
	    arena->chunk_count,(unsigned long long)span_bytes);
	fflush(stdout);
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdPreloadSpine(SparkWeightdServer *server,
    SparkWeightdArena *arena,int32_t fd);
static SparkStatus SparkWeightdArenaChunkEnsure(SparkWeightdServer *server,
    SparkWeightdArena *arena,uint32_t first_chunk,uint32_t last_chunk);
static void SparkWeightdServerAttachLazy(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    const SparkWeightdIpcAttachLazy *request,
    SparkWeightdIpcAttachLazyResult *result)
{
    SparkWeightdIdentity identity = request->identity;
    SparkWeightdArena *arena;
    SparkWeightdExpertEntry *entries = 0;
    SparkWeightdManifest manifest;
    struct stat pack_stat;
    uint32_t expert_count = 0u;
    uint32_t slot;
    SparkStatus status;

    result->resident_bytes = server->resident_bytes;
    result->arena_count = server->arena_count;

    status = SparkWeightdIdentityPrepare(&identity);
    if (status != SPARK_STATUS_OK)
    {
        result->status = (uint32_t)status;
        return;
    }
    if (SparkWeightdStringBounded(request->pack_path,
            SPARK_WEIGHTD_PATH_BYTES) != SPARK_STATUS_OK ||
        request->expert_pool_bytes == 0ull ||
        request->expert_pool_bytes == UINT64_MAX ||
        request->expert_pool_bytes > server->config.device_bytes_max)
    {
        result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        return;
    }

    if (stat(request->pack_path, &pack_stat) != 0 ||
        pack_stat.st_size < 0 ||
        (uint64_t)pack_stat.st_size != identity.arena_bytes)
    {
        result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        return;
    }
    status = SparkWeightdExpertManifestLoad(request->pack_path,
        identity.arena_bytes, &manifest, &entries, &expert_count);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr,"weightd lazy attach rejected: %s.experts status=%s; generate the model-specific expert manifest before loading\n",
            request->pack_path,SparkStatusToString(status));
        result->status = (uint32_t)status;
        return;
    }

    arena = SparkWeightdServerFindArena(server, &identity);
    if (arena != 0)
    {
        free(entries);
        status = arena->lazy != 0u && arena->manifest.range_count == manifest.range_count && memcmp(arena->manifest.ranges,manifest.ranges,(manifest.range_count * sizeof(*manifest.ranges))) == 0 ? SPARK_STATUS_OK : SPARK_STATUS_HASH_MISMATCH;
        SparkWeightdManifestDestroy(&manifest);
        if ( status != SPARK_STATUS_OK )
        {
            result->status = status;
            return;
        }
        if (arena->lazy == 0u)
        {
            result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
            return;
        }
        if (request->expert_pool_bytes != arena->expert_pool_bytes)
        {
            result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
            return;
        }
        slot = (uint32_t)(arena - server->arenas);
        status = SparkWeightdServerAttachRegister(server, connection, slot);
        result->status = (uint32_t)status;
        if (status == SPARK_STATUS_OK)
        {
            result->arena_generation = server->arenas[slot].generation;
            result->device_handle =
                (uint64_t)(uintptr_t)server->arenas[slot].device_base;
            result->arena_bytes = server->arenas[slot].identity.arena_bytes;
            result->resident_bytes = server->resident_bytes;
            result->expert_pool_bytes = server->arenas[slot].lazy != 0u
                ? server->arenas[slot].expert_pool_bytes
                : server->arenas[slot].identity.arena_bytes;
            result->refcount = server->arenas[slot].refcount;
            result->arena_count = server->arena_count;
            result->expert_count = server->arenas[slot].expert_count;
            result->chunk_bytes = server->arenas[slot].chunk_bytes;
            result->chunk_count = server->arenas[slot].chunk_count;
            result->loaded_from_pack = 0u;
            if ( arena->pool_export_handle != 0 &&
                 connection->response_fd_count < SPARK_WEIGHTD_EXPORT_BATCH_MAX )
            {
                int pool_fd = -1;
                if ( cuMemExportToShareableHandle(&pool_fd,
                         (CUmemGenericAllocationHandle)(uintptr_t)arena->pool_export_handle,
                         CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0ull) == CUDA_SUCCESS &&
                     pool_fd >= 0 )
                {
                    (void)fcntl(pool_fd,F_SETFD,FD_CLOEXEC);
                    connection->response_fds[connection->response_fd_count++] = pool_fd;
                    result->pool_fd_staged = 1u;
                }
            }
            (void)SparkWeightdManifestIdentity(&server->arenas[slot].manifest,result->manifest_sha256);
        }
        return;
    }

    SparkWeightdServerReclaimCold(server, request->expert_pool_bytes);
    if (server->arena_count >= SPARK_WEIGHTD_ARENA_COUNT_MAX)
    {
        free(entries);
        SparkWeightdManifestDestroy(&manifest);
        result->status = (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED;
        result->resident_bytes = server->resident_bytes;
        result->arena_count = server->arena_count;
        return;
    }

    slot = server->arena_count;
    memset(&server->arenas[slot], 0, sizeof(server->arenas[slot]));
    server->arenas[slot].identity = identity;
    if (SparkWeightdVmmReserve(identity.arena_bytes,
            &server->arenas[slot]) != SPARK_STATUS_OK)
    {
        memset(&server->arenas[slot], 0, sizeof(server->arenas[slot]));
        free(entries);
        SparkWeightdManifestDestroy(&manifest);
        result->status = (uint32_t)SPARK_STATUS_CAPACITY_EXCEEDED;
        return;
    }
    server->arenas[slot].lazy = 1u;
    server->arenas[slot].expert_pool_bytes = request->expert_pool_bytes;
    server->arenas[slot].experts = entries;
    server->arenas[slot].manifest = manifest;
    server->arenas[slot].expert_count = expert_count;
    server->arenas[slot].pack_stat = pack_stat;
    memcpy(server->arenas[slot].pack_path, request->pack_path,
        strlen(request->pack_path) + 1u);
    server->next_arena_generation++;
    server->arenas[slot].generation = server->next_arena_generation;
    server->arena_count++;
    arena = &server->arenas[slot];
    status = SparkWeightdLoadRecordedWorkingSet(arena);
    if ( status != SPARK_STATUS_OK )
    {
        SparkWeightdServerFreeArenaSlot(server,slot);
        result->status = (uint32_t)status;
        result->arena_count = server->arena_count;
        return;
    }
    arena->needed_chunks = calloc(arena->chunk_count,1u);
    arena->created_chunks = calloc(arena->chunk_count,1u);
    status = SparkWeightdLeaseTableCreate(&arena->manifest,&arena->leases);
    if ( status != SPARK_STATUS_OK || arena->needed_chunks == 0 || arena->created_chunks == 0 )
    {
        SparkWeightdServerFreeArenaSlot(server,slot);
        result->status = SPARK_STATUS_CAPACITY_EXCEEDED;
        return;
    }

    {
        int32_t pack_fd = open(request->pack_path,O_RDONLY);
        status = pack_fd < 0 ? SPARK_STATUS_IO_ERROR : SparkWeightdPremapPool(server,arena);
        if (status == SPARK_STATUS_OK && arena->pool_export_handle != 0)
            status = SparkWeightdPreloadSpine(server,arena,pack_fd);
        if (pack_fd >= 0)
            (void)close(pack_fd);
        if (status != SPARK_STATUS_OK)
        {
            SparkWeightdServerFreeArenaSlot(server,slot);
            result->status = (uint32_t)status;
            result->resident_bytes = server->resident_bytes;
            result->arena_count = server->arena_count;
            return;
        }
    }
    arena->preload_chunk_bytes = arena->pool_committed_bytes;

    status = SparkWeightdServerAttachRegister(server, connection, slot);
    if (status != SPARK_STATUS_OK)
    {
        SparkWeightdServerFreeArenaSlot(server, slot);
        result->status = (uint32_t)status;
        result->resident_bytes = server->resident_bytes;
        result->arena_count = server->arena_count;
        return;
    }
    result->status = (uint32_t)SPARK_STATUS_OK;
    result->arena_generation = server->arenas[slot].generation;
    result->device_handle =
        (uint64_t)(uintptr_t)server->arenas[slot].device_base;
    result->arena_bytes = identity.arena_bytes;
    result->resident_bytes = server->resident_bytes;
    result->expert_pool_bytes = request->expert_pool_bytes;
    result->refcount = server->arenas[slot].refcount;
    result->arena_count = server->arena_count;
    result->expert_count = expert_count;
    result->chunk_bytes = server->arenas[slot].chunk_bytes;
    result->chunk_count = server->arenas[slot].chunk_count;
    result->loaded_from_pack = 1u;
    if ( arena->pool_export_handle != 0 &&
         connection->response_fd_count < SPARK_WEIGHTD_EXPORT_BATCH_MAX )
    {
        int pool_fd = -1;
        if ( cuMemExportToShareableHandle(&pool_fd,
                 (CUmemGenericAllocationHandle)(uintptr_t)arena->pool_export_handle,
                 CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0ull) == CUDA_SUCCESS &&
             pool_fd >= 0 )
        {
            (void)fcntl(pool_fd,F_SETFD,FD_CLOEXEC);
            connection->response_fds[connection->response_fd_count++] = pool_fd;
            result->pool_fd_staged = 1u;
        }
    }
    (void)SparkWeightdManifestIdentity(&server->arenas[slot].manifest,result->manifest_sha256);
    printf("weightd lazy-attach model=%s experts=%u arena=%llu pool=%llu\n",
        identity.model, expert_count,
        (unsigned long long)identity.arena_bytes,
        (unsigned long long)request->expert_pool_bytes);
    fflush(stdout);
}

static SparkStatus SparkWeightdPreloadSpine(SparkWeightdServer *server,
    SparkWeightdArena *arena,int32_t fd)
{
    const SparkWeightdSpan *span;
    uint64_t span_offset,span_end;
    uint32_t first_chunk,last_chunk,chunk,index;
    ssize_t moved;
    if (arena->manifest.spine_count == 0u)
        return(SPARK_STATUS_OK);
    if (arena->staging == 0)
        return(SPARK_STATUS_INVALID_ARGUMENT);
    for ( index = 0u; index < arena->manifest.spine_count; index++ )
    {
        span = &arena->manifest.spine[index];
        first_chunk = (uint32_t)(span->offset / arena->chunk_bytes);
        last_chunk = (uint32_t)((span->offset + span->bytes - 1u) /
            arena->chunk_bytes);
        for ( chunk = first_chunk; chunk <= last_chunk; chunk++ )
        {
            if ( arena->chunk_handles[chunk] == 0 )
            {
                SparkStatus chunk_status;
                chunk_status = SparkWeightdArenaChunkEnsure(server,arena,
                    chunk,chunk);
                if ( chunk_status != SPARK_STATUS_OK )
                    SPARK_RETURN(chunk_status);
            }
        }
    }
    for ( index = 0u; index < arena->manifest.spine_count; index++ )
    {
        span = &arena->manifest.spine[index];
        span_offset = span->offset;
        span_end = span->offset + span->bytes;
        while ( span_offset < span_end )
        {
            uint64_t remain = span_end - span_offset;
            size_t bytes = remain < SPARK_WEIGHTD_ARENA_STAGING_BYTES ?
                (size_t)remain : (size_t)SPARK_WEIGHTD_ARENA_STAGING_BYTES;
            moved = pread(fd,arena->staging,bytes,(off_t)span_offset);
            if (moved < 0 && errno == EINTR)
                continue;
            if ( moved <= 0 )
                return(SPARK_STATUS_IO_ERROR);
            if ( cudaMemcpy((uint8_t *)arena->device_base + span_offset,
                    arena->staging,moved,cudaMemcpyHostToDevice) != cudaSuccess )
                return(SPARK_STATUS_IO_ERROR);
            span_offset += (uint64_t)moved;
        }
    }
    return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdArenaChunkEnsure(SparkWeightdServer *server,SparkWeightdArena *arena,
    uint32_t first_chunk,
    uint32_t last_chunk)
{
    CUmemAllocationProp prop;
    CUmemAccessDesc access;
    CUdeviceptr base = (CUdeviceptr)(uintptr_t)arena->device_base;
    uint32_t index;
    int device = 0;

    if (cudaGetDevice(&device) != cudaSuccess)
    {
        SPARK_FAIL(SPARK_STATUS_UNSUPPORTED);
    }
    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    for (index = first_chunk; index <= last_chunk; index++)
    {
        if (arena->chunk_handles[index] == 0)
        {
            CUmemGenericAllocationHandle handle = 0;
            if (server->resident_bytes > server->config.device_bytes_max ||
                arena->chunk_bytes > server->config.device_bytes_max - server->resident_bytes)
                SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
            if (cuMemCreate(&handle, (size_t)arena->chunk_bytes, &prop,
                    0ull) != CUDA_SUCCESS)
            {
                fprintf(stderr,"WD-CHUNK-FAIL create idx=%u bytes=%llu\n",
                    (unsigned)index,(unsigned long long)arena->chunk_bytes);
                SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
            }
            if (cuMemMap(base + (CUdeviceptr)index * arena->chunk_bytes,
                    (size_t)arena->chunk_bytes, 0u, handle, 0ull) != CUDA_SUCCESS)
            {
                (void)cuMemRelease(handle);
                fprintf(stderr,"WD-CHUNK-FAIL map idx=%u bytes=%llu\n",
                    (unsigned)index,(unsigned long long)arena->chunk_bytes);
                SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
            }
            arena->chunk_handles[index] = (void *)handle;
            arena->pool_committed_bytes += arena->chunk_bytes;
            server->resident_bytes += arena->chunk_bytes;
            arena->created_chunks[index] = 1u;
        }
        access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        access.location.id = device;
        access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        if (cuMemSetAccess(base + (CUdeviceptr)index * arena->chunk_bytes,
                (size_t)arena->chunk_bytes, &access, 1u) != CUDA_SUCCESS)
        {
            SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
        }
    }
    return SPARK_STATUS_OK;
}

static SparkWeightdArena *SparkWeightdAttachedArena(SparkWeightdServer *server,SparkWeightdConnection *connection,uint64_t generation)
{
	uint32_t i,slot;
	for (i=0u; i<connection->attach_count; i++)
	{
		slot = connection->attaches[i].arena_slot;
		if ( connection->attaches[i].arena_generation == generation && slot < server->arena_count && server->arenas[slot].generation == generation )
			return(&server->arenas[slot]);
	}
	return(0);
}

static void SparkWeightdMarkGroupChunks(SparkWeightdArena *arena,uint32_t group_index)
{
	const SparkWeightdRangeGroup *group = &arena->manifest.groups[group_index];
	const SparkWeightdRange *range;
	uint32_t i,first,last;
	for (i=0u; i<group->range_count; i++)
	{
		range = &arena->manifest.ranges[group->first_range + i];
		first = (uint32_t)(range->offset / arena->chunk_bytes);
		last = (uint32_t)((range->offset + range->bytes - 1u) / arena->chunk_bytes);
		memset(arena->needed_chunks + first,1u,(last - first + 1u));
	}
}

static SparkStatus SparkWeightdFreeChunk(SparkWeightdServer *server,SparkWeightdArena *arena,uint32_t index)
{
	if ( arena->pool_export_handle != 0 &&
	     arena->chunk_handles[index] == arena->pool_export_handle )
		return(SPARK_STATUS_OK);
	CUdeviceptr base = (CUdeviceptr)(uintptr_t)arena->device_base;
	if ( arena->chunk_handles[index] == 0 )
		return(SPARK_STATUS_OK);
	if ( cuMemUnmap(base + ((CUdeviceptr)index * arena->chunk_bytes),(size_t)arena->chunk_bytes) != CUDA_SUCCESS || cuMemRelease((CUmemGenericAllocationHandle)arena->chunk_handles[index]) != CUDA_SUCCESS )
	{
		arena->failure_status = SPARK_STATUS_IO_ERROR;
		return(SPARK_STATUS_IO_ERROR);
	}
	arena->chunk_handles[index] = 0;
	arena->pool_committed_bytes -= arena->chunk_bytes;
	server->resident_bytes -= arena->chunk_bytes;
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdEvictGroup(SparkWeightdServer *server,SparkWeightdArena *arena,uint32_t group_index)
{
	const SparkWeightdRangeGroup *group = &arena->manifest.groups[group_index];
	const SparkWeightdRange *range;
	uint32_t i,j,first,last;
	SparkStatus status;
	if ( arena->pool_export_handle != 0 )
		return(SPARK_STATUS_BUSY);
	if ( arena->leases->pins[group_index] != 0u )
		return(SPARK_STATUS_BUSY);
	for (i=0u; i<group->range_count; i++)
	{
		range = &arena->manifest.ranges[group->first_range + i];
		first = (uint32_t)(range->offset / arena->chunk_bytes);
		last = (uint32_t)((range->offset + range->bytes - 1u) / arena->chunk_bytes);
		for (j=first; j<=last; j++)
		{
			arena->chunk_refs[j]--;
			if ( arena->chunk_refs[j] == 0u )
			{
				status = SparkWeightdFreeChunk(server,arena,j);
				if ( status != SPARK_STATUS_OK )
					SPARK_RETURN(status);
			}
		}
		arena->expert_present_bytes -= range->bytes;
	}
	arena->experts[group_index].present = 0u;
	arena->epoch++;
	if ( arena->epoch_device != 0 )
		(void)cudaMemcpy(arena->epoch_device,&arena->epoch,
		    sizeof(uint64_t),cudaMemcpyHostToDevice);
	return(SPARK_STATUS_OK);
}

static uint64_t SparkWeightdAcquisitionBytes(const SparkWeightdArena *arena)
{
	uint64_t bytes = arena->pool_committed_bytes;
	uint32_t i;
	for (i=0u; i<arena->chunk_count; i++)
		if ( arena->needed_chunks[i] != 0u && arena->chunk_handles[i] == 0 )
			bytes += arena->chunk_bytes;
	return(bytes);
}

static SparkStatus SparkWeightdAcquireBudget(SparkWeightdServer *server,SparkWeightdArena *arena)
{
	uint64_t retained = 0u,other = (server->resident_bytes - arena->pool_committed_bytes),bytes,oldest,budget;
	uint32_t i,victim;
	SparkStatus status;
	budget = arena->pool_export_handle != 0
	    ? (arena->chunk_bytes * (uint64_t)arena->chunk_count)
	    : arena->expert_pool_bytes;
	memset(arena->needed_chunks,0,arena->chunk_count);
	for (i=0u; i<arena->manifest.group_count; i++)
		if ( arena->leases->pins[i] != 0u )
			SparkWeightdMarkGroupChunks(arena,i);
	for (i=0u; i<arena->chunk_count; i++)
		if ( arena->needed_chunks[i] != 0u )
			retained += arena->chunk_bytes;
	if ( retained > budget || other > server->config.device_bytes_max || retained > (server->config.device_bytes_max - other) )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	for (;;)
	{
		bytes = SparkWeightdAcquisitionBytes(arena);
		if ( (bytes <= arena->preload_chunk_bytes ||
		    bytes - arena->preload_chunk_bytes <= arena->expert_pool_bytes) && bytes <= (server->config.device_bytes_max - other) )
			return(SPARK_STATUS_OK);
		victim = arena->manifest.group_count;
		oldest = UINT64_MAX;
		for (i=0u; i<arena->manifest.group_count; i++)
			if ( arena->experts[i].present != 0u && arena->leases->pins[i] == 0u && (victim == arena->manifest.group_count || arena->experts[i].last_use_ns < oldest) )
			{
				victim = i;
				oldest = arena->experts[i].last_use_ns;
			}
		if ( victim == arena->manifest.group_count )
			return(SPARK_STATUS_INTERNAL_ERROR);
		status = SparkWeightdEvictGroup(server,arena,victim);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
}


static SparkStatus SparkWeightdLoadRangeFromStaging(SparkWeightdArena *arena,
	const SparkWeightdRange *range,const uint8_t *staging,uint64_t staging_base)
{
	SparkCk128Context context;
	uint8_t digest[16];
	const uint8_t *source;
	source = staging + (range->offset - staging_base);
	SparkCk128Initialize(&context);
	SparkCk128Update(&context,source,(size_t)range->bytes);
	SparkCk128Finalize(&context,digest);
	if ( memcmp(digest,range->digest,sizeof(digest)) != 0 )
		return(SPARK_STATUS_HASH_MISMATCH);
	if ( cudaMemcpy((uint8_t *)arena->device_base + range->offset,source,
		(size_t)range->bytes,cudaMemcpyHostToDevice) != cudaSuccess )
	{
		fprintf(stderr,"WD-LOADRANGE-FAIL memcpy dst_off=%llu bytes=%llu cuda=%s\n",
			(unsigned long long)range->offset,
			(unsigned long long)range->bytes,cudaGetErrorString(cudaGetLastError()));
		return(SPARK_STATUS_IO_ERROR);
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdLoadRangeGroup(SparkWeightdArena *arena,int32_t fd,
	const SparkWeightdRange *ranges,uint32_t range_count)
{
	uint64_t staging_bytes = (uint64_t)SPARK_WEIGHTD_ARENA_STAGING_BYTES;
	uint32_t first = 0u;
	ssize_t moved;
	while ( first < range_count )
	{
		uint64_t span_offset = ranges[first].offset;
		uint64_t span_bytes = ranges[first].bytes;
		uint32_t last = first;
		while ( last + 1u < range_count &&
			ranges[last + 1u].offset == ranges[last].offset + ranges[last].bytes &&
			span_bytes + ranges[last + 1u].bytes <= staging_bytes )
		{
			last++;
			span_bytes += ranges[last].bytes;
		}
		if ( ranges[first].bytes > staging_bytes )
		{
			uint64_t offset = 0u,bytes;
			SparkCk128Context context;
			uint8_t digest[16];
			SparkCk128Initialize(&context);
			while ( offset < ranges[first].bytes )
			{
				bytes = (ranges[first].bytes - offset);
				if ( bytes > staging_bytes )
					bytes = staging_bytes;
				moved = pread(fd,arena->staging,(size_t)bytes,(off_t)(ranges[first].offset + offset));
				if ( moved < 0 && errno == EINTR )
					continue;
				if ( moved <= 0 )
				{
					fprintf(stderr,"WD-LOADRANGE-FAIL pread fd=%d off=%llu req=%llu moved=%ld errno=%d\n",
						fd,(unsigned long long)(ranges[first].offset + offset),
						(unsigned long long)bytes,(long)moved,errno);
					return(SPARK_STATUS_IO_ERROR);
				}
				SparkCk128Update(&context,arena->staging,(size_t)moved);
				if ( cudaMemcpy((uint8_t *)arena->device_base + ranges[first].offset + offset,
					arena->staging,(size_t)moved,cudaMemcpyHostToDevice) != cudaSuccess )
				{
					fprintf(stderr,"WD-LOADRANGE-FAIL memcpy dst_off=%llu bytes=%llu cuda=%s\n",
						(unsigned long long)(ranges[first].offset + offset),
						(unsigned long long)moved,cudaGetErrorString(cudaGetLastError()));
					return(SPARK_STATUS_IO_ERROR);
				}
				offset += (uint64_t)moved;
			}
			SparkCk128Finalize(&context,digest);
			if ( memcmp(digest,ranges[first].digest,sizeof(digest)) != 0 )
				return(SPARK_STATUS_HASH_MISMATCH);
		}
		else
		{
			uint64_t offset = 0u;
			while ( offset < span_bytes )
			{
				uint64_t bytes = span_bytes - offset;
				if ( bytes > staging_bytes )
					bytes = staging_bytes;
				moved = pread(fd,arena->staging + offset,(size_t)bytes,(off_t)(span_offset + offset));
				if ( moved < 0 && errno == EINTR )
					continue;
				if ( moved <= 0 )
				{
					fprintf(stderr,"WD-LOADRANGE-FAIL pread fd=%d off=%llu req=%llu moved=%ld errno=%d\n",
						fd,(unsigned long long)(span_offset + offset),
						(unsigned long long)bytes,(long)moved,errno);
					return(SPARK_STATUS_IO_ERROR);
				}
				offset += (uint64_t)moved;
			}
			for (uint32_t index = first; index <= last; index++)
			{
				SparkStatus status = SparkWeightdLoadRangeFromStaging(arena,
					&ranges[index],arena->staging,span_offset);
				if ( status != SPARK_STATUS_OK )
					return(status);
			}
		}
		first = last + 1u;
	}
	return(SPARK_STATUS_OK);
}

static SparkStatus SparkWeightdLoadLease(SparkWeightdArena *arena,int32_t fd,const SparkWeightdLease *lease)
{
	const SparkWeightdRangeGroup *group;
	struct stat info;
	uint32_t i;
	SparkStatus status;
	if ( fstat(fd,&info) != 0 || info.st_dev != arena->pack_stat.st_dev || info.st_ino != arena->pack_stat.st_ino || info.st_size != arena->pack_stat.st_size || SparkWeightdStatMtimeNs(&info) != SparkWeightdStatMtimeNs(&arena->pack_stat) )
		return(SPARK_STATUS_HASH_MISMATCH);
	for (i=0u; i<lease->count; i++)
	{
		if ( arena->experts[lease->groups[i]].present != 0u )
			continue;
		group = &arena->manifest.groups[lease->groups[i]];
		status = SparkWeightdLoadRangeGroup(arena,fd,
			&arena->manifest.ranges[group->first_range],group->range_count);
		if ( status != SPARK_STATUS_OK )
			SPARK_RETURN(status);
	}
	if ( fstat(fd,&info) != 0 || info.st_size != arena->pack_stat.st_size || SparkWeightdStatMtimeNs(&info) != SparkWeightdStatMtimeNs(&arena->pack_stat) )
		return(SPARK_STATUS_HASH_MISMATCH);
	return(SPARK_STATUS_OK);
}

static void SparkWeightdCommitLease(SparkWeightdArena *arena,const SparkWeightdLease *lease)
{
	const SparkWeightdRangeGroup *group;
	const SparkWeightdRange *range;
	uint32_t i,j,k,first,last,index;
	uint64_t now = SparkWeightdMonotonicTimeNs();
	for (i=0u; i<lease->count; i++)
	{
		index = lease->groups[i];
		arena->experts[index].last_use_ns = now;
		if ( arena->experts[index].present != 0u )
			continue;
		group = &arena->manifest.groups[index];
		for (j=0u; j<group->range_count; j++)
		{
			range = &arena->manifest.ranges[group->first_range + j];
			first = (uint32_t)(range->offset / arena->chunk_bytes);
			last = (uint32_t)((range->offset + range->bytes - 1u) / arena->chunk_bytes);
			for (k=first; k<=last; k++)
				arena->chunk_refs[k]++;
			arena->expert_present_bytes += range->bytes;
		}
		arena->experts[index].present = 1u;
	}
}

static SparkStatus SparkWeightdAcquireLoad(SparkWeightdServer *server,SparkWeightdArena *arena,const SparkWeightdLease *lease)
{
	uint32_t i;
	int32_t fd;
	SparkStatus status;
	memset(arena->created_chunks,0,arena->chunk_count);
	status = SparkWeightdAcquireBudget(server,arena);
	if ( status != SPARK_STATUS_OK )
	{
		fprintf(stderr,"ACQUIRE-LOAD-STAGE stage=budget status=%u\n",(unsigned)status);
		SPARK_RETURN(status);
	}
	for (i=0u; i<arena->chunk_count; i++)
		if ( arena->needed_chunks[i] != 0u && arena->chunk_handles[i] == 0 )
		{
			status = SparkWeightdArenaChunkEnsure(server,arena,i,i);
			if ( status != SPARK_STATUS_OK )
			{
				fprintf(stderr,"ACQUIRE-LOAD-STAGE stage=chunk_ensure index=%u status=%u\n",i,(unsigned)status);
				SPARK_RETURN(status);
			}
		}
	fd = open(arena->pack_path,O_RDONLY | O_NONBLOCK);
	if ( fd < 0 )
	{
		fprintf(stderr,"ACQUIRE-LOAD-STAGE stage=open_pack path=%s errno=%d\n",arena->pack_path,errno);
		return(SPARK_STATUS_IO_ERROR);
	}
	status = SparkWeightdLoadLease(arena,fd,lease);
	if ( status != SPARK_STATUS_OK )
		fprintf(stderr,"ACQUIRE-LOAD-STAGE stage=load_lease status=%u\n",(unsigned)status);
	(void)close(fd);
	if ( status == SPARK_STATUS_OK )
		SparkWeightdCommitLease(arena,lease);
	SPARK_RETURN(status);
}

static SparkStatus SparkWeightdRecordWorkingSet(SparkWeightdArena *arena,
	const SparkWeightdExpertKey *keys,uint32_t count)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u],temporary[SPARK_WEIGHTD_PATH_BYTES + 16u] = {0};
	FILE *out;
	uint32_t i,index,original_count = arena->recorded_count;
	SparkStatus status = SPARK_STATUS_IO_ERROR;
	int fd;
	for (i=0u; i<count; i++)
	{
		uint64_t key = ((uint64_t)keys[i].layer << 32u) | keys[i].expert;
		for (index=0u; index<arena->recorded_count; index++)
			if ( arena->recorded_keys[index] == key )
				break;
		if ( index < arena->recorded_count )
			continue;
		if ( arena->recorded_count == arena->recorded_capacity )
		{
			uint32_t capacity = arena->recorded_capacity != 0u ?
				arena->recorded_capacity * 2u : 1024u;
			uint64_t *grown = (uint64_t *)realloc(arena->recorded_keys,
				(size_t)capacity * sizeof(*grown));
			if ( grown == 0 )
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				goto failed;
			}
			arena->recorded_keys = grown;
			arena->recorded_capacity = capacity;
		}
		arena->recorded_keys[arena->recorded_count++] = key;
	}
	if ( arena->recorded_count == original_count )
		return SPARK_STATUS_OK;
	if ( snprintf(path,sizeof(path),"%s.wset",arena->pack_path) >= (int)sizeof(path) ||
		snprintf(temporary,sizeof(temporary),"%s.XXXXXX",path) >= (int)sizeof(temporary) )
		goto failed;
	fd = mkstemp(temporary);
	if ( fd < 0 )
		goto failed;
	out = fdopen(fd,"wb");
	if ( out == 0 )
	{
		(void)close(fd);
		(void)unlink(temporary);
		goto failed;
	}
	for (index=0u; index<arena->recorded_count; index++)
	{
		uint32_t pair[2] = {(uint32_t)(arena->recorded_keys[index] >> 32u),
			(uint32_t)arena->recorded_keys[index]};
		if ( fwrite(pair,sizeof(pair),1u,out) != 1u )
			break;
	}
	if ( fclose(out) != 0 || index != arena->recorded_count || rename(temporary,path) != 0 )
	{
		(void)unlink(temporary);
		goto failed;
	}
	return SPARK_STATUS_OK;
failed:
	arena->recorded_count = original_count;
	fprintf(stderr,"weightd: recording failed pack=%s status=%d\n",arena->pack_path,(int)status);
	return status;
}

static SparkStatus SparkWeightdLoadRecordedWorkingSet(SparkWeightdArena *arena)
{
	char path[SPARK_WEIGHTD_PATH_BYTES + 8u];
	FILE *in;
	uint32_t pair[2];
	size_t bytes;
	SparkStatus status = SPARK_STATUS_OK;
	if ( snprintf(path,sizeof(path),"%s.wset",arena->pack_path) >= (int)sizeof(path) )
		return SPARK_STATUS_INVALID_ARGUMENT;
	in = fopen(path,"rb");
	if ( in == 0 )
		return errno == ENOENT ? SPARK_STATUS_OK : SPARK_STATUS_IO_ERROR;
	while ( (bytes = fread(pair,1u,sizeof(pair),in)) != 0u )
	{
		uint64_t key;
		uint32_t index;
		if ( bytes != sizeof(pair) || SparkWeightdManifestFind(&arena->manifest,pair[0],pair[1]) == 0 )
		{
			status = SPARK_STATUS_SCHEMA_ERROR;
			break;
		}
		key = ((uint64_t)pair[0] << 32u) | pair[1];
		for (index=0u; index<arena->recorded_count; index++)
			if ( arena->recorded_keys[index] == key )
				break;
		if ( index < arena->recorded_count )
			continue;
		if ( arena->recorded_count == arena->recorded_capacity )
		{
			uint32_t capacity = arena->recorded_capacity != 0u ?
				arena->recorded_capacity * 2u : 1024u;
			uint64_t *grown = (uint64_t *)realloc(arena->recorded_keys,
				(size_t)capacity * sizeof(*grown));
			if ( grown == 0 )
			{
				status = SPARK_STATUS_CAPACITY_EXCEEDED;
				break;
			}
			arena->recorded_keys = grown;
			arena->recorded_capacity = capacity;
		}
		arena->recorded_keys[arena->recorded_count++] = key;
	}
	if ( arena->recorded_count == 0u && status == SPARK_STATUS_OK )
		status = SPARK_STATUS_SCHEMA_ERROR;
	if ( ferror(in) )
		status = SPARK_STATUS_IO_ERROR;
	if ( fclose(in) != 0 )
		status = SPARK_STATUS_IO_ERROR;
	fprintf(stderr,"WD-WSET loaded=%u status=%d path=%s\n",arena->recorded_count,(int)status,path);
	return status;
}

static SparkStatus SparkWeightdAcquireWorkingSet(SparkWeightdServer *server,SparkWeightdConnection *connection,SparkWeightdArena *arena,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier)
{
	const SparkWeightdLease *lease;
	SparkStatus status;
	uint32_t i;
	if ( arena->lazy == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( arena->failure_status != SPARK_STATUS_OK )
	{
		static uint64_t arena_fail_count_,arena_fail_shown_;
		arena_fail_count_++;
		if ( arena_fail_shown_ < 8u || (arena_fail_count_ % 100000u) == 0u )
			fprintf(stderr,
				"ACQUIRE-ARENA-FAILED count=%llu sticky_status=%u layer0=%u key_count=%u\n",
				(unsigned long long)arena_fail_count_,
				(unsigned)arena->failure_status,
				count != 0u ? (unsigned)keys[0].layer : 0u,count);
		arena_fail_shown_++;
		return(arena->failure_status);
	}
	{
		uint32_t lease_lane = SparkWeightdConnectionLane(connection);
		status = SparkWeightdLeaseAcquire(arena->leases,connection->owner,lease_lane,keys,count,identifier);
	}
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	lease = SparkWeightdLeaseFind(arena->leases,connection->owner,*identifier);
	status = SparkWeightdAcquireLoad(server,arena,lease);
	if ( status == SPARK_STATUS_OK )
		SPARK_RETURN(status);
	{
		static uint64_t acquire_fail_count_,acquire_fail_shown_;
		acquire_fail_count_++;
		if ( acquire_fail_shown_ < 8u || (acquire_fail_count_ % 100000u) == 0u )
			fprintf(stderr,
				"ACQUIRE-FAIL count=%llu layer0=%u expert0=%u key_count=%u status=%u budget: pool=%llu retained=%llu other_resident=%llu device_max=%llu chunk_count=%u\n",
				(unsigned long long)acquire_fail_count_,
				count != 0u ? (unsigned)keys[0].layer : 0u,
				count != 0u ? (unsigned)keys[0].expert : 0u,
				count,(unsigned)status,
				(unsigned long long)arena->expert_pool_bytes,
				(unsigned long long)(arena->chunk_bytes * (uint64_t)arena->chunk_count),
				(unsigned long long)(server->resident_bytes - arena->pool_committed_bytes),
				(unsigned long long)server->config.device_bytes_max,
				arena->chunk_count);
		acquire_fail_shown_++;
	}
	for (i=0u; i<arena->chunk_count; i++)
		if ( arena->created_chunks[i] != 0u && SparkWeightdFreeChunk(server,arena,i) != SPARK_STATUS_OK )
			status = SPARK_STATUS_IO_ERROR;
	(void)SparkWeightdLeaseRelease(arena->leases,connection->owner,*identifier);
	*identifier = 0u;
	SPARK_RETURN(status);
}

static void SparkWeightdServerCloseStagedFds(SparkWeightdConnection *connection)
{
    uint32_t index;
    for (index = 0u; index < connection->response_fd_count; index++)
    {
        (void)close(connection->response_fds[index]);
    }
    connection->response_fd_count = 0u;
}

static SparkStatus SparkWeightdServerExportOne(SparkWeightdConnection *connection,void *handle)
{
	int32_t fd = -1;
	if ( connection->response_fd_count >= SPARK_WEIGHTD_EXPORT_BATCH_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	if ( cuMemExportToShareableHandle(&fd,(CUmemGenericAllocationHandle)handle,CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,0u) != CUDA_SUCCESS || fd < 0 )
		return(SPARK_STATUS_IO_ERROR);
	if ( fcntl(fd,F_SETFD,FD_CLOEXEC) != 0 )
	{
		(void)close(fd);
		return(SPARK_STATUS_IO_ERROR);
	}
	connection->response_fds[connection->response_fd_count++] = fd;
	return(SPARK_STATUS_OK);
}

/* The daemon's half of the fd tier: hand the attached arena's physical
 * chunks to THIS connection, one position-addressed batch at a time, each
 * chunk as a CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR shareable fd. Access
 * scoping is two-layered — the socket is 0600 (owner-only, the KV backing's
 * file discipline carried to the socket) and the request's arena generation
 * must be one THIS connection holds an attach reference for: the attach ref
 * is the export capability, nothing else on the node can fish chunks out.
 * The daemon holds its fd copies only until the reply flushes; the consumer
 * closes each fd after its cuMemImportFromShareableHandle takes the driver's
 * own reference. */
static void SparkWeightdServerExportBatch(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    const SparkWeightdIpcExport *request,
    SparkWeightdIpcExportResult *result)
{
    const SparkWeightdArena *arena = 0;
    uint32_t index;
    uint32_t count;

    result->arena_generation = request->arena_generation;
    result->batch_offset = request->batch_offset;
    for (index = 0u; index < connection->attach_count; index++)
    {
        if (connection->attaches[index].arena_generation ==
                request->arena_generation &&
            connection->attaches[index].arena_slot < server->arena_count)
        {
            arena = &server->arenas[connection->attaches[index].arena_slot];
            break;
        }
    }
    if (arena == 0 || arena->generation != request->arena_generation)
    {
        result->status = (uint32_t)SPARK_STATUS_NOT_FOUND;
        return;
    }
    if (arena->lazy != 0u || request->batch_offset >= arena->chunk_count)
    {
        result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        return;
    }
    count = arena->chunk_count - request->batch_offset;
    if (count > SPARK_WEIGHTD_EXPORT_BATCH_MAX)
    {
        count = SPARK_WEIGHTD_EXPORT_BATCH_MAX;
    }
    for (index = 0u; index < count; index++)
    {
        if (arena->chunk_handles[request->batch_offset + index] == 0)
        {
            result->status = (uint32_t)SPARK_STATUS_NOT_FOUND;
            return;
        }
    }
    for (index = 0u; index < count; index++)
    {
        if (SparkWeightdServerExportOne(connection,arena->chunk_handles[request->batch_offset + index]) != SPARK_STATUS_OK)
        {
            SparkWeightdServerCloseStagedFds(connection);
            result->status = SPARK_STATUS_IO_ERROR;
            return;
        }
    }
    result->chunk_bytes = arena->chunk_bytes;
    result->chunk_count = arena->chunk_count;
    result->batch_count = count;
    result->status = (uint32_t)SPARK_STATUS_OK;
}

static void SparkWeightdServerExportLease(SparkWeightdServer *server,SparkWeightdConnection *connection,const SparkWeightdIpcExportLease *request,SparkWeightdIpcExportLeaseResult *result)
{
	SparkWeightdArena *arena = SparkWeightdAttachedArena(server,connection,request->arena_generation);
	const SparkWeightdLease *lease;
	uint32_t i,ordinal = 0u,count = 0u;
	result->base.arena_generation = request->arena_generation;
	result->base.batch_offset = request->batch_offset;
	result->lease_identifier = request->lease_identifier;
	result->base.status = SPARK_STATUS_NOT_FOUND;
	if ( arena == 0 || arena->lazy == 0u )
		return;
	lease = SparkWeightdLeaseFind(arena->leases,connection->owner,request->lease_identifier);
	if ( lease == 0 )
		return;
	result->base.status = SPARK_STATUS_INVALID_ARGUMENT;
	if ( request->reserved0 != 0u || arena->failure_status != SPARK_STATUS_OK )
		return;
	if ( arena->pool_export_handle != 0 )
	{
		result->lease_chunk_count = 0u;
		result->base.chunk_bytes = arena->chunk_bytes;
		result->base.chunk_count = arena->chunk_count;
		result->base.batch_count = 0u;
		result->base.status = SPARK_STATUS_OK;
		return;
	}
	memset(arena->needed_chunks,0,arena->chunk_count);
	for (i=0u; i<lease->count; i++)
		SparkWeightdMarkGroupChunks(arena,lease->groups[i]);
	for (i=0u; i<arena->chunk_count; i++)
		count += arena->needed_chunks[i] != 0u;
	if ( request->batch_offset >= count )
		return;
	result->lease_chunk_count = count;
	for (i=0u; i<arena->chunk_count && connection->response_fd_count<SPARK_WEIGHTD_EXPORT_BATCH_MAX; i++)
	{
		if ( arena->needed_chunks[i] == 0u || ordinal++ < request->batch_offset )
			continue;
		result->chunk_indices[connection->response_fd_count] = i;
		if ( arena->chunk_handles[i] == 0 || SparkWeightdServerExportOne(connection,arena->chunk_handles[i]) != SPARK_STATUS_OK )
		{
			SparkWeightdServerCloseStagedFds(connection);
			result->base.status = SPARK_STATUS_IO_ERROR;
			return;
		}
	}
	result->base.chunk_bytes = arena->chunk_bytes;
	result->base.chunk_count = arena->chunk_count;
	result->base.batch_count = connection->response_fd_count;
	result->base.status = SPARK_STATUS_OK;
}

/* ------------------------------ server: dispatch ------------------------------ */

static uint32_t SparkWeightdServerDispatch(SparkWeightdServer *server,
    SparkWeightdConnection *connection,
    const uint8_t *request,
    uint8_t *response)
{
    const SparkWeightdIpcHeader *request_header =
        (const SparkWeightdIpcHeader *)request;
    uint32_t result_kind = SparkWeightdKindResultKind(request_header->kind);
    uint64_t request_id = request_header->request_id;

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_HELLO)
    {
        SparkWeightdIpcHelloAck *ack = (SparkWeightdIpcHelloAck *)response;
        if (connection->hello_done != 0u)
        {
            return 0u; /* lockstep violation: fail the connection closed */
        }
        if ( server->next_owner == UINT64_MAX )
            return(0u);
        connection->owner = ++server->next_owner;
        connection->hello_done = 1u;
        SparkWeightdBuildHeader(response, result_kind, request_id);
        ack->daemon_generation = server->daemon_generation;
        ack->resident_bytes = atomic_load_explicit(&server->published_resident_bytes,memory_order_acquire);
        ack->device_bytes_max = server->config.device_bytes_max;
        ack->status = (uint32_t)SPARK_STATUS_OK;
        ack->arena_count = atomic_load_explicit(&server->published_arena_count,memory_order_acquire);
        return SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES;
    }

    if (connection->hello_done == 0u)
    {
        return 0u; /* every exchange opens with HELLO */
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_ATTACH)
    {
        SparkWeightdIpcAttachResult *result =
            (SparkWeightdIpcAttachResult *)response;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        SparkWeightdServerAttachCold(server, connection,
            (const SparkWeightdIpcAttach *)request, result);
        return SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES;
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_EXPORT)
    {
        SparkWeightdIpcExportResult *result =
            (SparkWeightdIpcExportResult *)response;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        SparkWeightdServerExportBatch(server, connection,
            (const SparkWeightdIpcExport *)request, result);
        return SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES;
    }

    if ( request_header->kind == SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE )
    {
        SparkWeightdIpcExportLeaseResult *result = (SparkWeightdIpcExportLeaseResult *)response;
        memset(result,0,sizeof(*result));
        SparkWeightdBuildHeader(response,result_kind,request_id);
        SparkWeightdServerExportLease(server,connection,(const SparkWeightdIpcExportLease *)request,result);
        return(sizeof(*result));
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY)
    {
        SparkWeightdIpcAttachLazyResult *result =
            (SparkWeightdIpcAttachLazyResult *)response;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        SparkWeightdServerAttachLazy(server, connection,
            (const SparkWeightdIpcAttachLazy *)request, result);
        return SPARK_WEIGHTD_IPC_ATTACH_LAZY_RESULT_BYTES;
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_ENSURE)
    {
        SparkWeightdIpcEnsureResult *result =
            (SparkWeightdIpcEnsureResult *)response;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        result->status = SPARK_STATUS_UNSUPPORTED;
        return SPARK_WEIGHTD_IPC_ENSURE_RESULT_BYTES;
    }

    if ( request_header->kind == SPARK_WEIGHTD_IPC_KIND_ACQUIRE || request_header->kind == SPARK_WEIGHTD_IPC_KIND_RELEASE )
    {
        const SparkWeightdIpcAcquire *acquire = (const SparkWeightdIpcAcquire *)request;
        const SparkWeightdIpcRelease *release = (const SparkWeightdIpcRelease *)request;
        SparkWeightdIpcAcquireResult *result = (SparkWeightdIpcAcquireResult *)response;
        uint64_t generation = request_header->kind == SPARK_WEIGHTD_IPC_KIND_ACQUIRE ? acquire->arena_generation : release->arena_generation;
        SparkWeightdArena *arena = SparkWeightdAttachedArena(server,connection,generation);
        memset(result,0,sizeof(*result));
        SparkWeightdBuildHeader(response,result_kind,request_id);
        result->arena_generation = generation;
        result->status = SPARK_STATUS_NOT_FOUND;
        if ( arena != 0 && arena->lazy != 0u )
        {
            if ( request_header->kind == SPARK_WEIGHTD_IPC_KIND_RELEASE )
            {
                uint32_t occ_i,occ_n = 0u;
                for (occ_i = 0u; occ_i < SPARK_WEIGHTD_LEASE_COUNT_MAX; occ_i++)
                    if ( arena->leases->leases[occ_i].count != 0u )
                        occ_n++;
                result->lease_identifier = release->lease_identifier;
                result->status = SparkWeightdLeaseRelease(arena->leases,connection->owner,release->lease_identifier);
                fprintf(stderr,"WD-LEASE-TRACE kind=release owner=%llu id=%llu status=%d occupied=%u\n",
                    (unsigned long long)connection->owner,
                    (unsigned long long)release->lease_identifier,
                    (int)result->status,occ_n);
            }
            else if ( acquire->reserved0 != 0u )
                result->status = SPARK_STATUS_INVALID_ARGUMENT;
            else
            {
                uint32_t occ_i,occ_n = 0u;
                for (occ_i = 0u; occ_i < SPARK_WEIGHTD_LEASE_COUNT_MAX; occ_i++)
                    if ( arena->leases->leases[occ_i].count != 0u )
                        occ_n++;
                result->status = SparkWeightdAcquireWorkingSet(server,connection,arena,acquire->keys,acquire->count,&result->lease_identifier);
                if ( result->status == SPARK_STATUS_OK )
                {
                    result->status = SparkWeightdRecordWorkingSet(arena,acquire->keys,acquire->count);
                    if ( result->status != SPARK_STATUS_OK )
                    {
                        SparkStatus released = SparkWeightdLeaseRelease(arena->leases,connection->owner,result->lease_identifier);
                        if ( released != SPARK_STATUS_OK )
                            arena->failure_status = released;
                        result->lease_identifier = 0u;
                    }
                }
                fprintf(stderr,"WD-LEASE-TRACE kind=%s owner=%llu keys=%u status=%d occupied=%u id=%llu\n",
                    request_header->kind == SPARK_WEIGHTD_IPC_KIND_ACQUIRE ? "acquire" : "release",
                    (unsigned long long)connection->owner,
                    request_header->kind == SPARK_WEIGHTD_IPC_KIND_ACQUIRE ? (unsigned)acquire->count : 0u,
                    (int)result->status,occ_n,
                    (unsigned long long)result->lease_identifier);
            }
        }
        result->resident_bytes = server->resident_bytes;
        return(sizeof(*result));
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY)
    {
        const SparkWeightdIpcMeshActivity *activity =
            (const SparkWeightdIpcMeshActivity *)request;
        SparkWeightdIpcMeshActivityResult *result =
            (SparkWeightdIpcMeshActivityResult *)response;
        SparkStatus status;
        memset(result,0,sizeof(*result));
        SparkWeightdBuildHeader(response,result_kind,request_id);
        if ( activity->generation == 0u || activity->active > 1u ||
             activity->lane >= SPARK_WEIGHTD_MESH_MAX_LANES ||
             server->lane_owner[activity->lane] == 0u )
            status = SPARK_STATUS_INVALID_ARGUMENT;
        else if ( activity->active != 0u && server->orphan_mesh_owners != 0u )
            status = SPARK_STATUS_IO_ERROR;
        else if ( activity->active != 0u && connection->mesh_active != 0u )
            status = activity->generation == connection->mesh_generation ?
                SPARK_STATUS_DUPLICATE : SPARK_STATUS_BUSY;
        else if ( activity->active != 0u &&
                  activity->generation <= connection->mesh_generation )
            status = SPARK_STATUS_INVALID_ARGUMENT;
        else if ( activity->active == 0u &&
                 (connection->mesh_active == 0u ||
                  activity->generation != connection->mesh_generation ||
                  activity->lane != connection->mesh_lane) )
            status = SPARK_STATUS_INVALID_ARGUMENT;
        else
        {
            status = SparkWeightdMeshSetActivity(activity->lane,activity->active);
            if ( status == SPARK_STATUS_OK )
            {
                connection->mesh_generation = activity->generation;
                connection->mesh_active = activity->active;
                connection->mesh_lane = activity->lane;
            }
        }
        result->status = (uint32_t)status;
        return sizeof(*result);
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_MESH_WRITE)
    {
        SparkWeightdIpcMeshWriteResult *result =
            (SparkWeightdIpcMeshWriteResult *)response;
        const SparkWeightdIpcMeshWrite *write =
            (const SparkWeightdIpcMeshWrite *)request;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        result->status = (uint32_t)SparkWeightdMeshPostWrite(
            write->peer_rank,
            SparkWeightdMeshBufferAddress() + write->source_offset,
            SparkWeightdMeshBufferLkey(),
            write->length,
            write->remote_offset);
        return(sizeof(*result));
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST)
    {
        SparkWeightdIpcMeshBroadcastResult *result =
            (SparkWeightdIpcMeshBroadcastResult *)response;
        const SparkWeightdIpcMeshBroadcast *broadcast =
            (const SparkWeightdIpcMeshBroadcast *)request;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        result->posted_count = SparkWeightdMeshBroadcast(
            broadcast->peer_mask,
            broadcast->source_offset,
            broadcast->length,
            broadcast->remote_offset,
            broadcast->seq_value,
            broadcast->seq_remote_offset);
        result->status = result->posted_count != 0u ?
            (uint32_t)SPARK_STATUS_OK :
            (uint32_t)SPARK_STATUS_BUSY;
        return(sizeof(*result));
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_DETACH)
    {
        SparkWeightdIpcDetachResult *result =
            (SparkWeightdIpcDetachResult *)response;
        const SparkWeightdIpcDetach *detach =
            (const SparkWeightdIpcDetach *)request;
        uint32_t index;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        result->status = (uint32_t)SPARK_STATUS_NOT_FOUND;
        for (index = 0u; index < connection->attach_count; index++)
        {
            if (connection->attaches[index].arena_generation ==
                detach->arena_generation)
            {
                uint32_t arena_slot = connection->attaches[index].arena_slot;
                if ( arena_slot < server->arena_count && SparkWeightdArenaHasOwnerLeases(&server->arenas[arena_slot],connection->owner) != 0u )
                {
                    result->status = SPARK_STATUS_BUSY;
                    result->refcount = server->arenas[arena_slot].refcount;
                    break;
                }
                SparkWeightdServerDetachRelease(server, connection, index);
                result->status = (uint32_t)SPARK_STATUS_OK;
                if (arena_slot < server->arena_count &&
                    server->arenas[arena_slot].generation ==
                        detach->arena_generation)
                {
                    result->refcount = server->arenas[arena_slot].refcount;
                }
                break;
            }
        }
        result->resident_bytes = server->resident_bytes;
        result->arena_count = server->arena_count;
        return SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES;
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE)
    {
        SparkWeightdIpcLaneAcquireResult *result =
            (SparkWeightdIpcLaneAcquireResult *)response;
        const SparkWeightdIpcLaneAcquire *acquire =
            (const SparkWeightdIpcLaneAcquire *)request_header;
        uint32_t lane;
        const SparkWeightdMeshTopology empty_topology = {0};
        memset(result,0,sizeof(*result));
        SparkWeightdBuildHeader(response,result_kind,request_id);
        result->lane = SPARK_WEIGHTD_LANE_NONE;
        if (acquire->reserved != 0u ||
            (acquire->topology.rank_count == 0u &&
             memcmp(&acquire->topology,&empty_topology,sizeof(empty_topology)) != 0) ||
            (acquire->requested_lane != SPARK_WEIGHTD_LANE_NONE &&
             acquire->requested_lane >= SPARK_WEIGHTD_MESH_MAX_LANES))
            result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
        else if (server->orphan_mesh_owners != 0u)
            result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
        else if (connection->lane_mask != 0u)
            result->status = (uint32_t)SPARK_STATUS_DUPLICATE;
        else
            result->status = (uint32_t)SPARK_STATUS_NO_LANE;
        if (result->status != (uint32_t)SPARK_STATUS_NO_LANE)
            return sizeof(*result);
        for (lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
            if (server->lane_owner[lane] == 0u &&
                (acquire->requested_lane == SPARK_WEIGHTD_LANE_NONE ||
                 acquire->requested_lane == lane))
            {
                if (acquire->topology.rank_count != 0u)
                {
                    result->status = (uint32_t)SparkWeightdMeshLaneConfigure(
                        lane,&acquire->topology);
                    if (result->status != (uint32_t)SPARK_STATUS_OK)
                        break;
                }
                server->lane_owner[lane] =
                    (uint16_t)(connection - server->connections) + 1u;
                connection->lane_mask |= (uint16_t)(1u << lane);
                result->status = (uint32_t)SPARK_STATUS_OK;
                result->lane = lane;
                break;
            }
        return(sizeof(*result));
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_EVICT)
    {
        SparkWeightdIpcEvictResult *result =
            (SparkWeightdIpcEvictResult *)response;
        const SparkWeightdIpcEvict *request =
            (const SparkWeightdIpcEvict *)request_header;
        uint32_t requester_lane;
        uint32_t released;
        uint32_t arena_slot;
        memset(result,0,sizeof(*result));
        SparkWeightdBuildHeader(response,result_kind,request_id);
        requester_lane = SparkWeightdConnectionLane(connection);
        if (request->target_lane >= SPARK_WEIGHTD_MESH_MAX_LANES ||
            requester_lane == SPARK_WEIGHTD_LANE_NONE)
        {
            result->status = (uint32_t)SPARK_STATUS_INVALID_ARGUMENT;
            return sizeof(*result);
        }
        if (requester_lane >= request->target_lane)
        {
            result->status = (uint32_t)SPARK_STATUS_EVICT_DENIED;
            fprintf(stderr,
                "weightd evict_denied requester_lane=%u target_lane=%u\n",
                requester_lane,request->target_lane);
            return sizeof(*result);
        }
        released = 0u;
        for (arena_slot = 0u; arena_slot < server->arena_count; arena_slot++)
        {
            SparkWeightdLeaseTable *leases =
                server->arenas[arena_slot].leases;
            uint32_t lane_released = 0u;
            if (leases == 0)
            {
                continue;
            }
            if (SparkWeightdLeaseReleaseForLane(leases,request->target_lane,
                    &lane_released) == SPARK_STATUS_OK)
            {
                released += lane_released;
            }
        }
        if (released == 0u && server->lane_owner[request->target_lane] == 0u)
        {
            result->status = (uint32_t)SPARK_STATUS_NOT_FOUND;
            return sizeof(*result);
        }
        fprintf(stderr,"weightd evict requester_lane=%u target_lane=%u released=%u\n",
            requester_lane,request->target_lane,released);
        result->status = (uint32_t)SPARK_STATUS_OK;
        result->released_leases = released;
        return sizeof(*result);
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT)
    {
        SparkWeightdIpcEpochExportResult *result =
            (SparkWeightdIpcEpochExportResult *)response;
        const SparkWeightdIpcEpochExport *request =
            (const SparkWeightdIpcEpochExport *)request_header;
        SparkWeightdArena *arena;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        arena = SparkWeightdAttachedArena(server, connection,
            request->arena_generation);
        if (arena == 0 || arena->epoch_handle == 0 ||
            SparkWeightdServerExportOne(connection, arena->epoch_handle) !=
                SPARK_STATUS_OK)
        {
            if (arena != 0 && arena->epoch_handle != 0)
                SparkWeightdServerCloseStagedFds(connection);
            result->status = (uint32_t)SPARK_STATUS_NOT_FOUND;
            return SPARK_WEIGHTD_IPC_EPOCH_EXPORT_RESULT_BYTES;
        }
        result->status = (uint32_t)SPARK_STATUS_OK;
        return SPARK_WEIGHTD_IPC_EPOCH_EXPORT_RESULT_BYTES;
    }

    if (request_header->kind == SPARK_WEIGHTD_IPC_KIND_RECLAIM)
    {
        SparkWeightdIpcReclaimResult *result =
            (SparkWeightdIpcReclaimResult *)response;
        uint64_t resident_before = server->resident_bytes;
        uint32_t arenas_before = server->arena_count;
        uint32_t index;
        memset(result, 0, sizeof(*result));
        SparkWeightdBuildHeader(response, result_kind, request_id);
        for (index = server->arena_count; index > 0u; index--)
        {
            if (server->arenas[index - 1u].refcount == 0u && SparkWeightdArenaHasLeases(&server->arenas[index - 1u]) == 0u)
            {
                SparkWeightdServerFreeArenaSlot(server, index - 1u);
            }
        }
        result->status = (uint32_t)SPARK_STATUS_OK;
        result->reclaimed_bytes = resident_before - server->resident_bytes;
        result->resident_bytes = server->resident_bytes;
        result->reclaimed_arena_count = arenas_before - server->arena_count;
        result->arena_count = server->arena_count;
        return SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES;
    }

    return 0u;
}

static void SparkWeightdServerDispatchWork(void *context)
{
    SparkWeightdServer *server = context;
    SparkWeightdConnection *connection = &server->connections[server->dispatch_connection];
    uint8_t byte = 1u;
    connection->response_bytes = SparkWeightdServerDispatch(server,connection,
        connection->request,connection->response);
    atomic_store_explicit(&server->dispatch_state,2u,memory_order_release);
    while (write(server->dispatch_notify[1],&byte,1u) < 0 && errno == EINTR)
        ;
}

static void SparkWeightdServerPublish(SparkWeightdServer *server)
{
    atomic_store_explicit(&server->published_resident_bytes,server->resident_bytes,memory_order_release);
    atomic_store_explicit(&server->published_arena_count,server->arena_count,memory_order_release);
}

static void SparkWeightdServerCompleteWork(SparkWeightdServer *server)
{
    if (atomic_load_explicit(&server->dispatch_state,memory_order_acquire) == 2u)
    {
        SparkWeightdConnection *connection = &server->connections[server->dispatch_connection];
        const SparkWeightdIpcHeader *request = (const SparkWeightdIpcHeader *)connection->request;
        if (connection->state == SPARK_WEIGHTD_CONNECTION_OPEN &&
            request->kind == SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY &&
            connection->response_bytes == SPARK_WEIGHTD_IPC_ATTACH_LAZY_RESULT_BYTES)
            SparkWeightdServerStageMeshFd(connection);
        connection->request_ready = 0u;
        connection->request_bytes = 0u;
        connection->response_written = 0u;
        if (connection->response_bytes == 0u)
            connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
        SparkWeightdServerPublish(server);
        atomic_store_explicit(&server->dispatch_state,0u,memory_order_release);
    }
}

/* ------------------------------ server: connections ------------------------------ */

static void SparkWeightdServerCloseConnection(SparkWeightdServer *server,
    uint32_t connection_index)
{
    SparkWeightdConnection *connection = &server->connections[connection_index];
    if ( connection->mesh_active != 0u )
    {
        server->orphan_mesh_owners++;
        fprintf(stderr,"weightd mesh owner disconnected before GPU drain: owner=%llu generation=%llu; new mesh activity blocked\n",
            (unsigned long long)connection->owner,
            (unsigned long long)connection->mesh_generation);
        connection->mesh_active = 0u;
    }
    connection->mesh_generation = 0u;
    if (connection->owner != 0u && connection->attach_count != 0u)
    {
        uint32_t arena_index;
        for (arena_index = 0u; arena_index < server->arena_count; arena_index++)
            if (server->arenas[arena_index].leases != 0)
                (void)SparkWeightdLeaseReleaseOwner(server->arenas[arena_index].leases,connection->owner);
    }
    while (connection->attach_count != 0u)
    {
        /* consumer death drops a refcount — every one of them */
        SparkWeightdServerDetachRelease(server, connection, 0u);
    }
    if (connection->fd >= 0)
    {
        (void)close(connection->fd);
    }
    if (connection->lane_mask != 0u)
    {
        uint32_t lane;
        for (lane = 0u; lane < SPARK_WEIGHTD_MESH_MAX_LANES; lane++)
            if ((connection->lane_mask & (uint16_t)(1u << lane)) != 0u &&
                server->lane_owner[lane] ==
                    (uint16_t)(connection - server->connections) + 1u)
                server->lane_owner[lane] = 0u;
        connection->lane_mask = 0u;
    }
    SparkWeightdServerCloseStagedFds(connection); /* unsent chunk fds die here */
    connection->fd = -1;
    connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
    connection->hello_done = 0u;
    connection->request_bytes = 0u;
    connection->request_ready = 0u;
    connection->response_bytes = 0u;
    connection->response_written = 0u;
}

/* Drain the pending response as far as the socket takes it. PENDING means
 * "not yet fully written — retry on the next step"; the connection is
 * failed closed only on a real write error. A response with staged fds
 * leaves via sendmsg with the SCM_RIGHTS ancillary instead of write(): the
 * fds cross EXACTLY once (the kernel dups them into the receiver's queue at
 * send time), so the staged set clears on the first completed send whatever
 * the byte count; on EAGAIN/EINTR nothing left the socket and the same fds
 * retry on the next step. */
static SparkStatus SparkWeightdServerFlushResponse(
    SparkWeightdConnection *connection)
{
    while (connection->response_written < connection->response_bytes)
    {
        ssize_t chunk;
        if (connection->response_fd_count != 0u)
        {
            struct msghdr message;
            struct iovec vector;
            struct cmsghdr *control_header;
            char control[CMSG_SPACE(SPARK_WEIGHTD_EXPORT_BATCH_MAX *
                sizeof(int))];
            memset(&message, 0, sizeof(message));
            vector.iov_base =
                connection->response + connection->response_written;
            vector.iov_len =
                connection->response_bytes - connection->response_written;
            message.msg_iov = &vector;
            message.msg_iovlen = 1u;
            message.msg_control = control;
            message.msg_controllen = CMSG_SPACE(
                connection->response_fd_count * sizeof(int));
            control_header = CMSG_FIRSTHDR(&message);
            control_header->cmsg_level = SOL_SOCKET;
            control_header->cmsg_type = SCM_RIGHTS;
            control_header->cmsg_len = CMSG_LEN(
                connection->response_fd_count * sizeof(int));
            memcpy(CMSG_DATA(control_header), connection->response_fds,
                connection->response_fd_count * sizeof(int));
            chunk = sendmsg(connection->fd, &message, 0);
            if (chunk >= 0)
            {
                /* delivered: the receiver's dup holds the referent now */
                SparkWeightdServerCloseStagedFds(connection);
            }
            else if (errno == EINTR)
            {
                continue;
            }
            else if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                SPARK_FAIL(SPARK_STATUS_PENDING);
            }
            else
            {
                return SPARK_STATUS_IO_ERROR; /* fds close with the socket */
            }
        }
        else
        {
            chunk = write(connection->fd,
                connection->response + connection->response_written,
                connection->response_bytes - connection->response_written);
            if (chunk < 0)
            {
                if (errno == EAGAIN || errno == EWOULDBLOCK)
                {
                    SPARK_FAIL(SPARK_STATUS_PENDING);
                }
                if (errno == EINTR)
                {
                    continue;
                }
                SPARK_FAIL(SPARK_STATUS_IO_ERROR);
            }
        }
        connection->response_written += (uint32_t)chunk;
    }
    connection->response_bytes = 0u;
    connection->response_written = 0u;
    return SPARK_STATUS_OK;
}

static void SparkWeightdServerHandleReadable(SparkWeightdServer *server,
    SparkWeightdConnection *connection)
{
    for (;;)
    {
        ssize_t chunk;
        if (connection->request_ready != 0u || connection->response_bytes != 0u)
        {
            /* the previous response has not fully left yet: stop reading so
             * one slow consumer cannot pin the daemon (responses are tiny
             * fixed frames, so this drains within a step or two) */
            return;
        }
        chunk = recv(connection->fd,
            connection->request + connection->request_bytes,
            sizeof(connection->request) - connection->request_bytes, 0);
        if (chunk < 0)
        {
            if (errno == EAGAIN || errno == EWOULDBLOCK)
            {
                return;
            }
            if (errno == EINTR)
            {
                continue;
            }
            connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
            return;
        }
        if (chunk == 0)
        {
            /* consumer death: EOF drops every refcount this connection held */
            connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
            return;
        }
        connection->request_bytes += (uint32_t)chunk;
        if (connection->request_bytes < SPARK_WEIGHTD_IPC_HEADER_BYTES)
        {
            continue;
        }
        {
            const SparkWeightdIpcHeader *header =
                (const SparkWeightdIpcHeader *)connection->request;
            uint32_t expected_bytes;
            if (header->magic != SPARK_WEIGHTD_IPC_MAGIC ||
                header->abi_version != SPARK_WEIGHTD_IPC_ABI_VERSION ||
                SparkWeightdKindResultKind(header->kind) == 0u ||
                header->body_bytes >
                    SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX -
                        SPARK_WEIGHTD_IPC_HEADER_BYTES)
            {
                connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
                return;
            }
            expected_bytes =
                SPARK_WEIGHTD_IPC_HEADER_BYTES + header->body_bytes;
            if (connection->request_bytes < expected_bytes)
            {
                continue;
            }
            /* wire-shape check against the exact frame size for the kind */
            if (SparkWeightdIpcValidateHeader(header, expected_bytes,
                    header->kind) != SPARK_STATUS_OK)
            {
                connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
                return;
            }
            if (header->kind != SPARK_WEIGHTD_IPC_KIND_HELLO &&
                header->kind != SPARK_WEIGHTD_IPC_KIND_MESH_WRITE &&
                header->kind != SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST &&
                header->kind != SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY)
            {
                connection->request_ready = 1u;
                return;
            }
            connection->response_bytes = SparkWeightdServerDispatch(server,
                connection, connection->request, connection->response);
            connection->response_written = 0u;
            connection->request_bytes = 0u;
            if (connection->response_bytes == 0u)
            {
                /* dispatch refused the exchange: fail the connection closed */
                connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
                return;
            }
        }
    }
}

SparkStatus SparkWeightdServerStep(SparkWeightdServer *server)
{
    struct pollfd poll_fds[SPARK_WEIGHTD_CONNECTION_COUNT_MAX + 2u];
    uint32_t poll_count = 0u;
    uint32_t connection_index;
    uint32_t poll_index;
    int poll_result;

    if (server == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }

    SparkWeightdServerCompleteWork(server);
    SparkWeightdMeshPoll();

    if (server->listen_fd >= 0)
    {
        poll_fds[poll_count].fd = server->listen_fd;
        poll_fds[poll_count].events = POLLIN;
        poll_fds[poll_count].revents = 0;
        poll_count++;
    }
    poll_fds[poll_count].fd = server->dispatch_notify[0];
    poll_fds[poll_count].events = POLLIN;
    poll_fds[poll_count].revents = 0;
    poll_count++;
    for (connection_index = 0u;
        connection_index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
        connection_index++)
    {
        if (server->connections[connection_index].state ==
            SPARK_WEIGHTD_CONNECTION_OPEN)
        {
            poll_fds[poll_count].fd = server->connections[connection_index].fd;
            poll_fds[poll_count].events = server->connections[connection_index].request_ready != 0u ? 0 : POLLIN;
            poll_fds[poll_count].revents = 0;
            poll_count++;
        }
    }
    poll_result = poll(poll_fds, (nfds_t)poll_count,
        (int)SPARK_WEIGHTD_SERVER_POLL_TIMEOUT_MS);
    if (poll_result < 0)
    {
        if (errno == EINTR)
        {
            return SPARK_STATUS_OK;
        }
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }

    poll_index = 0u;
    if (server->listen_fd >= 0)
    {
        if ((poll_fds[0].revents & POLLIN) != 0)
        {
            for (connection_index = 0u;
                connection_index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
                connection_index++)
            {
                SparkWeightdConnection *connection =
                    &server->connections[connection_index];
                if (connection->fd < 0)
                {
                    int fd = accept(server->listen_fd, 0, 0);
                    if (fd < 0)
                    {
                        break;
                    }
                    (void)fcntl(fd, F_SETFL, O_NONBLOCK);
                    connection->fd = fd;
                    connection->state = SPARK_WEIGHTD_CONNECTION_OPEN;
                    connection->hello_done = 0u;
                    connection->request_bytes = 0u;
                    connection->request_ready = 0u;
                    connection->response_bytes = 0u;
                    connection->response_written = 0u;
                    connection->response_fd_count = 0u;
                }
            }
            /* a full table just declines further connects; a pending peer
             * sees its own connect deadline expire (fail-closed, both ways) */
        }
        poll_index = 1u;
    }

    if ((poll_fds[poll_index].revents & POLLIN) != 0)
    {
        uint8_t bytes[64];
        while (read(server->dispatch_notify[0],bytes,sizeof(bytes)) > 0)
            ;
        SparkWeightdServerCompleteWork(server);
    }
    poll_index++;

    for (connection_index = 0u;
        connection_index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
        connection_index++)
    {
        SparkWeightdConnection *connection =
            &server->connections[connection_index];
        if (connection->state != SPARK_WEIGHTD_CONNECTION_OPEN)
        {
            continue;
        }
        if (poll_index < poll_count &&
            poll_fds[poll_index].fd == connection->fd)
        {
            short revents = poll_fds[poll_index].revents;
            poll_index++;
            if ((revents & (POLLHUP | POLLERR | POLLNVAL)) != 0)
            {
                connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
            }
            else if ((revents & POLLIN) != 0)
            {
                SparkWeightdServerHandleReadable(server, connection);
            }
        }
        if (connection->state == SPARK_WEIGHTD_CONNECTION_OPEN &&
            !(atomic_load_explicit(&server->dispatch_state,memory_order_acquire) != 0u &&
                server->dispatch_connection == connection_index) &&
            connection->response_bytes != 0u &&
            SparkWeightdServerFlushResponse(connection) == SPARK_STATUS_IO_ERROR)
        {
            connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
        }
    }

    for (connection_index = 0u;
        connection_index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
        connection_index++)
    {
        if (server->connections[connection_index].state ==
                SPARK_WEIGHTD_CONNECTION_CLOSED &&
            server->connections[connection_index].fd >= 0 &&
            (atomic_load_explicit(&server->dispatch_state,memory_order_acquire) == 0u ||
                (server->dispatch_connection != connection_index &&
                    server->connections[connection_index].attach_count == 0u &&
                    server->connections[connection_index].lane_mask == 0u)))
        {
            SparkWeightdServerCloseConnection(server, connection_index);
        }
    }
    if (atomic_load_explicit(&server->dispatch_state,memory_order_acquire) == 0u)
    {
        SparkWeightdServerPublish(server);
        for (uint32_t offset = 0u; offset < SPARK_WEIGHTD_CONNECTION_COUNT_MAX; offset++)
        {
            uint32_t index = (server->dispatch_next + offset) % SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
            SparkWeightdConnection *connection = &server->connections[index];
            if (connection->state != SPARK_WEIGHTD_CONNECTION_OPEN || connection->request_ready == 0u)
                continue;
            server->dispatch_connection = index;
            server->dispatch_next = (index + 1u) % SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
            atomic_store_explicit(&server->dispatch_state,1u,memory_order_release);
            SparkStatus status = SparkWeightdWorkerSubmit(server->worker,SparkWeightdServerDispatchWork,server);
            if (status != SPARK_STATUS_OK)
            {
                atomic_store_explicit(&server->dispatch_state,0u,memory_order_release);
                SPARK_RETURN(status);
            }
            break;
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdServerRun(SparkWeightdServer *server,
    const volatile sig_atomic_t *stop)
{
    if (server == 0 || stop == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    /* the stop flag is written by a signal handler in the daemon and by
     * the driving thread in tests: the load is atomic so both callers are
     * race-free by construction */
    while (__atomic_load_n(stop, __ATOMIC_SEQ_CST) == 0)
    {
        SparkStatus status = SparkWeightdServerStep(server);
        if (status != SPARK_STATUS_OK)
        {
            SPARK_RETURN(status);
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdServerCreate(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server)
{
    struct sockaddr_un address;
    SparkWeightdServer *instance;
    uint32_t index;

    if (config == 0 || config->socket_path == 0 ||
        config->socket_path[0] == '\0' ||
        strlen(config->socket_path) >= sizeof(address.sun_path) ||
        config->device_bytes_max == 0ull || server == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    instance = (SparkWeightdServer *)calloc(1u, sizeof(*instance));
    if (instance == 0)
    {
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    instance->config = *config;
    instance->config.socket_path = instance->socket_path;
    memset(instance->socket_path, 0, sizeof(instance->socket_path));
    memcpy(instance->socket_path, config->socket_path,
        strlen(config->socket_path) + 1u);
    instance->listen_fd = -1;
    instance->dispatch_notify[0] = -1;
    instance->dispatch_notify[1] = -1;
    atomic_init(&instance->dispatch_state,0u);
    atomic_init(&instance->published_arena_count,0u);
    atomic_init(&instance->published_resident_bytes,0u);
    for (index = 0u; index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX; index++)
    {
        /* calloc zero-fills state to OPEN: fresh slots must start CLOSED
         * (fd = -1, not accepting) or the accept loop never finds a slot */
        instance->connections[index].fd = -1;
        instance->connections[index].state = SPARK_WEIGHTD_CONNECTION_CLOSED;
    }
    (void)signal(SIGPIPE, SIG_IGN);

    /* a stale socket from a crashed predecessor is reclaimed, never feared */
    (void)unlink(instance->socket_path);
    instance->listen_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (instance->listen_fd < 0)
    {
        free(instance);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, instance->socket_path,
        strlen(instance->socket_path) + 1u);
    if (bind(instance->listen_fd, (const struct sockaddr *)&address,
            sizeof(address)) != 0 ||
        listen(instance->listen_fd,
            (int)SPARK_WEIGHTD_CONNECTION_COUNT_MAX) != 0)
    {
        (void)close(instance->listen_fd);
        (void)unlink(instance->socket_path);
        free(instance);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    if (fcntl(instance->listen_fd,F_SETFL,O_NONBLOCK) != 0 ||
        pipe(instance->dispatch_notify) != 0 ||
        fcntl(instance->dispatch_notify[0],F_SETFL,O_NONBLOCK) != 0 ||
        fcntl(instance->dispatch_notify[1],F_SETFL,O_NONBLOCK) != 0 ||
        cudaFree(0) != cudaSuccess ||
        SparkWeightdWorkerCreate(&instance->worker) != SPARK_STATUS_OK)
    {
        SparkWeightdServerDestroy(instance);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
    (void)chmod(instance->socket_path, 0600);
    instance->daemon_generation = SparkWeightdMonotonicTimeNs();
    *server = instance;
    return SPARK_STATUS_OK;
}

void SparkWeightdServerDestroy(SparkWeightdServer *server)
{
    uint32_t index;
    if (server == 0)
    {
        return;
    }
    if (server->worker != 0)
    {
        while (SparkWeightdWorkerWaitIdle(server->worker,UINT64_C(1000000000)) == SPARK_STATUS_BUSY)
            ;
        if (SparkWeightdWorkerDestroy(server->worker) != SPARK_STATUS_OK)
            return;
        server->worker = 0;
    }
    for (index = 0u; index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX; index++)
    {
        if (server->connections[index].state == SPARK_WEIGHTD_CONNECTION_OPEN)
        {
            server->connections[index].state = SPARK_WEIGHTD_CONNECTION_CLOSED;
        }
        if (server->connections[index].fd >= 0)
        {
            SparkWeightdServerCloseConnection(server, index);
        }
    }
    while (server->arena_count != 0u)
    {
        SparkWeightdServerFreeArenaSlot(server, server->arena_count - 1u);
    }
    if (server->listen_fd >= 0)
    {
        (void)close(server->listen_fd);
    }
    if (server->dispatch_notify[0] >= 0)
        (void)close(server->dispatch_notify[0]);
    if (server->dispatch_notify[1] >= 0)
        (void)close(server->dispatch_notify[1]);
    (void)unlink(server->config.socket_path);
    free(server);
}

uint32_t SparkWeightdServerArenaCount(const SparkWeightdServer *server)
{
    return server != 0 ? atomic_load_explicit(&server->published_arena_count,memory_order_acquire) : 0u;
}

uint64_t SparkWeightdServerResidentBytes(const SparkWeightdServer *server)
{
    return server != 0 ? atomic_load_explicit(&server->published_resident_bytes,memory_order_acquire) : 0ull;
}

/* ------------------------------ client ------------------------------ */

static SparkStatus SparkWeightdDeadlineRemaining(uint64_t deadline_ns,
    int *timeout_ms)
{
    uint64_t now = SparkWeightdMonotonicTimeNs();
    if (now == 0ull)
    {
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    }
    if (now >= deadline_ns)
    {
        SPARK_FAIL(SPARK_STATUS_BUSY);
    }
    *timeout_ms = (int)((deadline_ns - now) / 1000000ull) + 1;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdClientWriteAll(SparkWeightdClient *client,
    const uint8_t *buffer,
    uint32_t bytes,
    uint64_t deadline_ns)
{
    uint32_t written = 0u;
    while (written < bytes)
    {
        ssize_t chunk = send(client->fd, buffer + written, bytes - written,
            MSG_NOSIGNAL);
        int timeout_ms;
        if (chunk < 0)
        {
            struct pollfd poll_fd;
            if (errno == EINTR)
            {
                continue;
            }
            if (errno != EAGAIN && errno != EWOULDBLOCK)
            {
                SPARK_FAIL(SPARK_STATUS_IO_ERROR);
            }
            poll_fd.fd = client->fd;
            poll_fd.events = POLLOUT;
            poll_fd.revents = 0;
            if (SparkWeightdDeadlineRemaining(deadline_ns, &timeout_ms) !=
                SPARK_STATUS_OK)
            {
                SPARK_FAIL(SPARK_STATUS_BUSY);
            }
            if (poll(&poll_fd, 1u, timeout_ms) <= 0)
            {
                SPARK_FAIL(SPARK_STATUS_BUSY);
            }
            continue;
        }
        written += (uint32_t)chunk;
    }
    return SPARK_STATUS_OK;
}

/* Read exactly `bytes` with a hard deadline; short = dead daemon or
 * protocol fault, both fail closed. */
static SparkStatus SparkWeightdClientReadAll(SparkWeightdClient *client,
    uint8_t *buffer,
    uint32_t bytes,
    uint64_t deadline_ns)
{
    uint32_t received = 0u;
    while (received < bytes)
    {
        ssize_t chunk;
        struct pollfd poll_fd;
        int timeout_ms;
        SparkStatus deadline_status = SparkWeightdDeadlineRemaining(
            deadline_ns, &timeout_ms);
        if (deadline_status != SPARK_STATUS_OK)
        {
            SPARK_RETURN(deadline_status);
        }
        poll_fd.fd = client->fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        if (poll(&poll_fd, 1u, timeout_ms) <= 0)
        {
            SPARK_FAIL(SPARK_STATUS_BUSY);
        }
        chunk = recv(client->fd, buffer + received, bytes - received, 0);
        if (chunk <= 0)
        {
            if (chunk < 0 && errno == EINTR)
            {
                continue;
            }
            /* EOF mid-exchange: the daemon is gone (crash semantics:
             * fail closed, never chase) */
            SPARK_FAIL(SPARK_STATUS_IO_ERROR);
        }
        received += (uint32_t)chunk;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdClientExchange(SparkWeightdClient *client,
    const void *request,
    uint32_t request_bytes,
    void *response,
    uint32_t response_bytes,
    uint64_t timeout_nanoseconds)
{
    const SparkWeightdIpcHeader *request_header = request;
    SparkWeightdIpcHeader *response_header = response;
    uint64_t now = SparkWeightdMonotonicTimeNs();
    uint64_t timeout = timeout_nanoseconds != 0ull
        ? timeout_nanoseconds : SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS;
    uint64_t deadline;
    SparkStatus status;
    if (client == 0 || client->fd < 0)
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    if (now == 0ull || timeout > UINT64_MAX - now)
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    deadline = now + timeout;
    status = SparkWeightdClientWriteAll(client,request,request_bytes,deadline);
    if (status == SPARK_STATUS_OK)
        status = SparkWeightdClientReadAll(client,response,response_bytes,deadline);
    if (status == SPARK_STATUS_OK)
        status = SparkWeightdIpcValidateHeader(response_header,response_bytes,
            SparkWeightdKindResultKind(request_header->kind));
    if (status == SPARK_STATUS_OK && response_header->request_id != request_header->request_id)
        status = SPARK_STATUS_SCHEMA_ERROR;
    if (status != SPARK_STATUS_OK)
    {
        (void)close(client->fd);
        client->fd = -1;
    }
    SPARK_RETURN(status);
}

SparkStatus SparkWeightdClientConnect(const char *socket_path,
    SparkWeightdClient **client,
    SparkWeightdHelloResult *hello_out)
{
    struct sockaddr_un address;
    SparkWeightdClient *instance;
    SparkWeightdIpcHello wire_hello;
    SparkWeightdIpcHelloAck wire_ack;
    SparkStatus status;

    if (socket_path == 0 || socket_path[0] == '\0' ||
        strlen(socket_path) >= sizeof(address.sun_path) ||
        client == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (hello_out != 0)
    {
        memset(hello_out, 0, sizeof(*hello_out));
    }
    instance = (SparkWeightdClient *)calloc(1u, sizeof(*instance));
    if (instance == 0)
    {
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    }
    instance->lane = SPARK_WEIGHTD_LANE_NONE;
    instance->fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (instance->fd < 0)
    {
        free(instance);
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }
#if defined(SO_NOSIGPIPE)
    {
        int enable = 1;
        if (setsockopt(instance->fd, SOL_SOCKET, SO_NOSIGPIPE, &enable,
                sizeof(enable)) != 0)
        {
            (void)close(instance->fd);
            free(instance);
            SPARK_FAIL(SPARK_STATUS_IO_ERROR);
        }
    }
#endif
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    memcpy(address.sun_path, socket_path, strlen(socket_path) + 1u);
    if (connect(instance->fd, (const struct sockaddr *)&address,
            sizeof(address)) != 0)
    {
        (void)close(instance->fd);
        free(instance);
        /* refused / missing daemon: the consumer fails closed here, it does
         * not fall back to a stale pointer (the crash contract) */
        SPARK_FAIL(SPARK_STATUS_IO_ERROR);
    }

    memset(&wire_hello, 0, sizeof(wire_hello));
    wire_hello.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire_hello.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire_hello.header.kind = SPARK_WEIGHTD_IPC_KIND_HELLO;
    wire_hello.header.body_bytes =
        SPARK_WEIGHTD_IPC_HELLO_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire_hello.header.request_id = ++instance->next_request_id;
    memset(&wire_ack, 0, sizeof(wire_ack));
    status = SparkWeightdClientExchange(instance, &wire_hello,
        SPARK_WEIGHTD_IPC_HELLO_BYTES, &wire_ack,
        SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES, 0ull);
    if (status != SPARK_STATUS_OK)
    {
        (void)close(instance->fd);
        free(instance);
        SPARK_RETURN(status);
    }
    if (wire_ack.status != (uint32_t)SPARK_STATUS_OK)
    {
        (void)close(instance->fd);
        free(instance);
        return SparkWeightdStatusFromWire(wire_ack.status);
    }
    if (hello_out != 0)
    {
        hello_out->status = SPARK_STATUS_OK;
        hello_out->daemon_generation = wire_ack.daemon_generation;
        hello_out->resident_bytes = wire_ack.resident_bytes;
        hello_out->device_bytes_max = wire_ack.device_bytes_max;
        hello_out->arena_count = wire_ack.arena_count;
    }
    instance->daemon_generation = wire_ack.daemon_generation;
    *client = instance;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientAttach(SparkWeightdClient *client,
    const SparkWeightdAttachRequest *request,
    SparkWeightdAttachResult *result,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcAttach wire;
    SparkWeightdIpcAttachResult wire_result;
    SparkWeightdIdentity identity;
    SparkStatus status;

    if (client == 0 || request == 0 || result == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(result, 0, sizeof(*result));
    identity = request->identity;
    status = SparkWeightdIdentityPrepare(&identity);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    if (SparkWeightdStringBounded(request->pack_path,
            SPARK_WEIGHTD_PATH_BYTES) != SPARK_STATUS_OK)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_ATTACH;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_ATTACH_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.identity = identity;
    memcpy(wire.pack_path, request->pack_path,
        strlen(request->pack_path) + 1u);
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        SPARK_WEIGHTD_IPC_ATTACH_BYTES, &wire_result,
        SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES, timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    result->status = SparkWeightdStatusFromWire(wire_result.status);
    result->arena_generation = wire_result.arena_generation;
    result->device_handle = wire_result.device_handle;
    result->arena_bytes = wire_result.arena_bytes;
    result->resident_bytes = wire_result.resident_bytes;
    result->refcount = wire_result.refcount;
    result->arena_count = wire_result.arena_count;
    result->loaded_from_pack = wire_result.loaded_from_pack;
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdClientReadFrameWithFds(SparkWeightdClient *client,
    uint8_t *buffer,
    uint32_t bytes,
    uint64_t deadline_ns,
    int *fds_out,
    uint32_t fds_capacity,
    uint32_t *fds_received);

SparkStatus SparkWeightdClientAttachLazy(SparkWeightdClient *client,
    const SparkWeightdLazyAttachRequest *request,
    SparkWeightdLazyAttachResult *result,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcAttachLazy wire;
    SparkWeightdIpcAttachLazyResult wire_result;
    SparkWeightdIdentity identity;
    SparkStatus status;

    if (client == 0 || request == 0 || result == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(result, 0, sizeof(*result));
    identity = request->identity;
    status = SparkWeightdIdentityPrepare(&identity);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    if (SparkWeightdStringBounded(request->pack_path,
            SPARK_WEIGHTD_PATH_BYTES) != SPARK_STATUS_OK)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_ATTACH_LAZY_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.identity = identity;
    memcpy(wire.pack_path, request->pack_path,
        strlen(request->pack_path) + 1u);
    wire.expert_pool_bytes = request->expert_pool_bytes;
    memset(&wire_result, 0, sizeof(wire_result));
    {
        uint64_t now = SparkWeightdMonotonicTimeNs();
        uint64_t deadline;
        int fds[2];
        uint32_t fds_received = 0u;
        if (now == 0ull)
        {
            SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
        }
        deadline = now + (timeout_nanoseconds != 0ull
            ? timeout_nanoseconds
            : SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS);
        status = SparkWeightdClientWriteAll(client,
            (const uint8_t *)&wire, SPARK_WEIGHTD_IPC_ATTACH_LAZY_BYTES,
            deadline);
        if (status == SPARK_STATUS_OK)
        {
            status = SparkWeightdClientReadFrameWithFds(client,
                (uint8_t *)&wire_result,
                SPARK_WEIGHTD_IPC_ATTACH_LAZY_RESULT_BYTES, deadline,
                fds, 2u, &fds_received);
        }
        if (status != SPARK_STATUS_OK)
        {
            (void)close(client->fd);
            client->fd = -1;
            SPARK_RETURN(status);
        }
        {
            const SparkWeightdIpcHeader *response_header =
                (const SparkWeightdIpcHeader *)&wire_result;
            if (SparkWeightdIpcValidateHeader(response_header,
                    sizeof(wire_result),
                    SparkWeightdKindResultKind(wire.header.kind)) !=
                    SPARK_STATUS_OK ||
                response_header->request_id != wire.header.request_id)
            {
                uint32_t close_index;
                for ( close_index = 0u; close_index < fds_received;
                    close_index++ )
                    (void)close(fds[close_index]);
                (void)close(client->fd);
                client->fd = -1;
                SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
            }
        }
        {
            uint32_t expected = (wire_result.mesh_ready != 0u ? 1u : 0u) +
                (wire_result.pool_fd_staged != 0u ? 1u : 0u);
            int32_t mesh_fd = -1,pool_fd = -1;
            if ( fds_received != expected )
            {
                uint32_t close_index;
                for ( close_index = 0u; close_index < fds_received;
                    close_index++ )
                    (void)close(fds[close_index]);
                (void)close(client->fd);
                client->fd = -1;
                SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
            }
            if ( wire_result.mesh_ready != 0u )
                mesh_fd = fds[0];
            if ( wire_result.pool_fd_staged != 0u )
                pool_fd = fds[wire_result.mesh_ready != 0u ? 1u : 0u];
            /* A mesh address from the daemon is its OWN virtual mapping.
             * Without the mesh fd there is no client-side mapping, and a
             * nonzero foreign pointer here is a guaranteed SIGSEGV in the
             * first doorbell scan (k3 first-launch find #11: residents
             * that attached while the fresh daemon was still wiring mesh
             * peers crashed in SparkTpDeviceCollectivePrepareReceiveBf16).
             * Zero it — the runners' mesh_send_buffer_addr != 0 guards
             * then correctly skip the device-collective receive prep. */
            if ( wire_result.mesh_ready == 0u )
                wire_result.mesh_send_buffer_addr = 0u;
            if ( mesh_fd >= 0 )
            {
                uint64_t bytes = wire_result.mesh_send_buffer_bytes;
                const size_t alignment = SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES;
                struct stat mesh_stat;
                void *mapped = MAP_FAILED;
                uint8_t *raw;
                size_t reserved;
                if ( bytes != SPARK_WEIGHTD_MESH_REGION_BYTES ||
                     bytes > SIZE_MAX - alignment || fstat(mesh_fd,&mesh_stat) != 0 ||
                     mesh_stat.st_size < 0 || (uint64_t)mesh_stat.st_size < bytes )
                {
                    (void)close(mesh_fd);
                    if ( pool_fd >= 0 ) (void)close(pool_fd);
                    SPARK_FAIL(SPARK_STATUS_SCHEMA_ERROR);
                }
                reserved = (size_t)bytes + alignment;
                raw = mmap(0,reserved,PROT_NONE,MAP_PRIVATE | MAP_ANONYMOUS,-1,0);
                if ( raw != MAP_FAILED )
                {
                    uintptr_t aligned = ((uintptr_t)raw + alignment - 1u) &
                        ~(uintptr_t)(alignment - 1u);
                    size_t prefix = aligned - (uintptr_t)raw;
                    size_t suffix = reserved - prefix - (size_t)bytes;
                    mapped = mmap((void *)aligned,(size_t)bytes,PROT_READ | PROT_WRITE,
                        MAP_SHARED | MAP_FIXED,mesh_fd,0);
                    if ( mapped != MAP_FAILED && prefix != 0u )
                    {
                        if ( munmap(raw,prefix) != 0 ) mapped = MAP_FAILED;
                        else { raw += prefix; reserved -= prefix; }
                    }
                    if ( mapped != MAP_FAILED && suffix != 0u )
                    {
                        if ( munmap((uint8_t *)mapped + bytes,suffix) != 0 ) mapped = MAP_FAILED;
                        else reserved -= suffix;
                    }
                    if ( mapped == MAP_FAILED ) (void)munmap(raw,reserved);
                }
                (void)close(mesh_fd);
                if ( mapped == MAP_FAILED )
                {
                    if ( pool_fd >= 0 ) (void)close(pool_fd);
                    SPARK_FAIL(SPARK_STATUS_IO_ERROR);
                }
                wire_result.mesh_send_buffer_addr = (uint64_t)(uintptr_t)mapped;
                result->mesh_mapping = mapped;
            }
            wire_result.pool_fd = pool_fd;
        }
    }
    if (wire_result.status != (uint32_t)SPARK_STATUS_OK)
    {
        result->status = SparkWeightdStatusFromWire(wire_result.status);
        result->resident_bytes = wire_result.resident_bytes;
        result->arena_count = wire_result.arena_count;
        return result->status;
    }
    result->status = SPARK_STATUS_OK;
    result->arena_generation = wire_result.arena_generation;
    result->device_handle = wire_result.device_handle;
    result->arena_bytes = wire_result.arena_bytes;
    result->resident_bytes = wire_result.resident_bytes;
    result->expert_pool_bytes = wire_result.expert_pool_bytes;
    result->refcount = wire_result.refcount;
    result->arena_count = wire_result.arena_count;
    result->expert_count = wire_result.expert_count;
    memcpy(result->manifest_sha256,wire_result.manifest_sha256,sizeof(result->manifest_sha256));
    result->chunk_bytes = wire_result.chunk_bytes;
    result->chunk_count = wire_result.chunk_count;
    result->loaded_from_pack = wire_result.loaded_from_pack;
    result->mesh_ready = wire_result.mesh_ready;
    result->mesh_send_buffer_addr = wire_result.mesh_send_buffer_addr;
    result->mesh_send_buffer_bytes = wire_result.mesh_send_buffer_bytes;
    result->pool_fd = wire_result.pool_fd;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientMeshActivity(SparkWeightdClient *client,
    uint64_t generation, uint32_t active, uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcMeshActivity wire;
    SparkWeightdIpcMeshActivityResult wire_result;
    SparkStatus status;
    if ( client == 0 || generation == 0u || active > 1u ||
         client->lane >= SPARK_WEIGHTD_MESH_MAX_LANES ||
         client->topology.rank_count == 0u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    memset(&wire,0,sizeof(wire));
    SparkWeightdBuildHeader((uint8_t *)&wire,SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY,
        ++client->next_request_id);
    wire.generation = generation;
    wire.active = active;
    wire.lane = client->lane;
    memset(&wire_result,0,sizeof(wire_result));
    status = SparkWeightdClientExchange(client,&wire,sizeof(wire),&wire_result,
        sizeof(wire_result),timeout_nanoseconds);
    return status == SPARK_STATUS_OK ?
        SparkWeightdStatusFromWire(wire_result.status) : status;
}

SparkStatus SparkWeightdClientMeshWrite(
    SparkWeightdClient *client,
    uint32_t peer_rank,
    uint64_t source_offset,
    uint64_t remote_offset,
    uint32_t length,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcMeshWrite wire;
    SparkWeightdIpcMeshWriteResult wire_result;
    SparkStatus status;

    if ( client == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_MESH_WRITE;
    wire.header.body_bytes =
        sizeof(wire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.peer_rank = peer_rank;
    wire.source_offset = source_offset;
    wire.remote_offset = remote_offset;
    wire.length = length;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        (uint32_t)sizeof(wire), &wire_result,
        (uint32_t)sizeof(wire_result), timeout_nanoseconds);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    if ( wire_result.status != (uint32_t)SPARK_STATUS_OK )
        return SparkWeightdStatusFromWire(wire_result.status);
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientMeshBroadcast(
    SparkWeightdClient *client,
    uint32_t peer_mask,
    uint64_t source_offset,
    uint64_t remote_offset,
    uint32_t length,
    uint64_t seq_value,
    uint64_t seq_remote_offset,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcMeshBroadcast wire;
    SparkWeightdIpcMeshBroadcastResult wire_result;
    SparkStatus status;

    if ( client == 0 )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST;
    wire.header.body_bytes =
        sizeof(wire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.peer_mask = peer_mask;
    wire.source_offset = source_offset;
    wire.remote_offset = remote_offset;
    wire.length = length;
    wire.seq_value = seq_value;
    wire.seq_remote_offset = seq_remote_offset;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        (uint32_t)sizeof(wire), &wire_result,
        (uint32_t)sizeof(wire_result), timeout_nanoseconds);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    if ( wire_result.status != (uint32_t)SPARK_STATUS_OK )
        return SparkWeightdStatusFromWire(wire_result.status);
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,
    uint32_t requested_lane,const SparkWeightdMeshTopology *topology,
    uint32_t *lane_out,uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcLaneAcquire wire;
    SparkWeightdIpcLaneAcquireResult wire_result;
    SparkStatus status;

    if ( client == 0 || lane_out == 0 ||
         (requested_lane != SPARK_WEIGHTD_LANE_NONE &&
          requested_lane >= SPARK_WEIGHTD_MESH_MAX_LANES) )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE;
    wire.header.body_bytes =
        sizeof(wire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.requested_lane = requested_lane;
    if (topology != 0) wire.topology = *topology;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        (uint32_t)sizeof(wire), &wire_result,
        (uint32_t)sizeof(wire_result), timeout_nanoseconds);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    if ( wire_result.status != (uint32_t)SPARK_STATUS_OK )
        return SparkWeightdStatusFromWire(wire_result.status);
    if ( wire_result.lane >= SPARK_WEIGHTD_MESH_MAX_LANES ||
         (requested_lane != SPARK_WEIGHTD_LANE_NONE &&
          wire_result.lane != requested_lane) )
        SPARK_FAIL(SPARK_STATUS_INTERNAL_ERROR);
    client->lane = wire_result.lane;
    client->topology = wire.topology;
    *lane_out = wire_result.lane;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneBind(SparkWeightdClient *owner,
    SparkWeightdClient *peer,uint32_t band,
    const SparkWeightdMeshTopology *topology,uint32_t *lane_out)
{
    uint32_t prior;
    if ( owner == 0 || peer == 0 || topology == 0 || lane_out == 0 || band >= 2u ||
         topology->rank_count == 0u ||
         memcmp(&owner->topology,topology,sizeof(*topology)) != 0 ||
         owner->lane >= SPARK_WEIGHTD_MESH_MAX_LANES ||
         owner->daemon_generation != peer->daemon_generation )
        return SPARK_STATUS_INVALID_ARGUMENT;
    if ( SparkWeightdClientAlive(owner) == 0u || SparkWeightdClientAlive(peer) == 0u )
        return SPARK_STATUS_IO_ERROR;
    prior = atomic_fetch_or_explicit(&owner->lane_bands,1u << band,memory_order_acq_rel);
    if ( (prior & (1u << band)) != 0u )
        return SPARK_STATUS_DUPLICATE;
    peer->lane = owner->lane;
    peer->topology = owner->topology;
    *lane_out = owner->lane;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientLaneUnbind(SparkWeightdClient *owner,uint32_t band)
{
    if ( owner == 0 || band >= 2u )
        return SPARK_STATUS_INVALID_ARGUMENT;
    return (atomic_fetch_and_explicit(&owner->lane_bands,~(1u << band),
        memory_order_acq_rel) & (1u << band)) != 0u ?
        SPARK_STATUS_OK : SPARK_STATUS_INVALID_ARGUMENT;
}

SparkStatus SparkWeightdClientEvict(SparkWeightdClient *client,
    uint32_t target_lane,
    uint32_t *released_leases_out,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcEvict wire;
    SparkWeightdIpcEvictResult wire_result;
    SparkStatus status;

    if ( client == 0 || released_leases_out == 0 ||
        target_lane >= SPARK_WEIGHTD_MESH_MAX_LANES )
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_EVICT;
    wire.header.body_bytes =
        sizeof(wire) - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.target_lane = target_lane;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        (uint32_t)sizeof(wire), &wire_result,
        (uint32_t)sizeof(wire_result), timeout_nanoseconds);
    if ( status != SPARK_STATUS_OK )
        SPARK_RETURN(status);
    *released_leases_out = wire_result.released_leases;
    if ( wire_result.status != (uint32_t)SPARK_STATUS_OK )
        return SparkWeightdStatusFromWire(wire_result.status);
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientEnsure(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t layer,
    uint32_t expert,
    SparkWeightdEnsureResult *result,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcEnsure wire;
    SparkWeightdIpcEnsureResult wire_result;
    SparkStatus status;

    if (client == 0 || result == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(result, 0, sizeof(*result));
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_ENSURE;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_ENSURE_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.arena_generation = arena_generation;
    wire.layer = layer;
    wire.expert = expert;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        SPARK_WEIGHTD_IPC_ENSURE_BYTES, &wire_result,
        SPARK_WEIGHTD_IPC_ENSURE_RESULT_BYTES, timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    result->status = SparkWeightdStatusFromWire(wire_result.status);
    result->arena_generation = wire_result.arena_generation;
    result->device_offset = wire_result.device_offset;
    result->expert_bytes = wire_result.expert_bytes;
    result->resident_bytes = wire_result.resident_bytes;
    result->load_ns = wire_result.load_ns;
    result->loaded = wire_result.loaded;
    return result->status;
}

/* ------------------------------ client: export (W3 fd tier) ------------------------------ */

static SparkStatus SparkWeightdReceiveFds(struct msghdr *message,int *fds,uint32_t capacity,uint32_t *count)
{
	struct cmsghdr *header;
	int32_t fd;
	uint32_t i,bytes,n,bad = ((message->msg_flags & MSG_CTRUNC) != 0);
	for (header=CMSG_FIRSTHDR(message); header!=0; header=CMSG_NXTHDR(message,header))
	{
		if ( header->cmsg_level != SOL_SOCKET || header->cmsg_type != SCM_RIGHTS )
			continue;
		if ( header->cmsg_len < CMSG_LEN(0) )
			return(SPARK_STATUS_IO_ERROR);
		bytes = (uint32_t)(header->cmsg_len - CMSG_LEN(0));
		n = (bytes / sizeof(int));
		if ( (bytes % sizeof(int)) != 0u )
			bad = 1u;
		for (i=0u; i<n; i++)
		{
			memcpy(&fd,(uint8_t *)CMSG_DATA(header) + (i * sizeof(int)),sizeof(fd));
			if ( *count >= capacity || fcntl(fd,F_SETFD,FD_CLOEXEC) != 0 )
			{
				(void)close(fd);
				bad = 1u;
			}
			else
				fds[(*count)++] = fd;
		}
	}
	return(bad != 0u ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK);
}

/* Receive exactly `bytes` of a reply frame, collecting the SCM_RIGHTS
 * ancillary that rides the FIRST data (fds always arrive with the frame's
 * leading bytes, never detached from it). Every fd that lands is made
 * close-on-exec and accounted into *fds_received; anything over capacity,
 * or a short frame, or a dead daemon fails closed with ALL received fds
 * closed — the caller never sees a partial hand it could miscount. */
static SparkStatus SparkWeightdClientReadFrameWithFds(
    SparkWeightdClient *client,
    uint8_t *buffer,
    uint32_t bytes,
    uint64_t deadline_ns,
    int *fds_out,
    uint32_t fds_capacity,
    uint32_t *fds_received)
{
    /* Receive the kernel's full SCM_RIGHTS allowance before enforcing our
     * smaller protocol cap. Truncating control data can hide installed FDs. */
    char control[253u * CMSG_SPACE(sizeof(int))];
    uint32_t received = 0u;
    uint32_t fd_count = 0u;

    *fds_received = 0u;
    while (received < bytes)
    {
        ssize_t chunk;
        struct pollfd poll_fd;
        int timeout_ms;
        SparkStatus deadline_status = SparkWeightdDeadlineRemaining(
            deadline_ns, &timeout_ms);
        if (deadline_status != SPARK_STATUS_OK)
        {
            goto fail;
        }
        poll_fd.fd = client->fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        if (poll(&poll_fd, 1u, timeout_ms) <= 0)
        {
            goto fail; /* deadline expired or poll error */
        }
        if (received == 0u)
        {
            struct msghdr message;
            struct iovec vector;
            memset(&message, 0, sizeof(message));
            vector.iov_base = buffer;
            vector.iov_len = bytes;
            message.msg_iov = &vector;
            message.msg_iovlen = 1u;
            message.msg_control = control;
            message.msg_controllen = sizeof(control);
#ifdef MSG_CMSG_CLOEXEC
            chunk = recvmsg(client->fd, &message, MSG_CMSG_CLOEXEC);
#else
            chunk = recvmsg(client->fd, &message, 0);
#endif
            if (chunk < 0 && errno == EINTR)
            {
                continue;
            }
            if (chunk <= 0)
            {
                goto fail; /* EOF mid-exchange: the daemon is gone */
            }
            if (SparkWeightdReceiveFds(&message,fds_out,fds_capacity,&fd_count) != SPARK_STATUS_OK)
                goto fail;

        }
        else
        {
            chunk = recv(client->fd, buffer + received, bytes - received, 0);
            if (chunk < 0 && errno == EINTR)
            {
                continue;
            }
            if (chunk <= 0)
            {
                goto fail;
            }
        }
        received += (uint32_t)chunk;
    }

    *fds_received = fd_count;
    return SPARK_STATUS_OK;
fail:
    while (fd_count != 0u)
    {
        (void)close(fds_out[--fd_count]);
    }
    *fds_received = 0u;
    SPARK_FAIL(SPARK_STATUS_IO_ERROR);
}

static SparkStatus SparkWeightdClientExportExchange(SparkWeightdClient *client,const void *request,uint32_t request_bytes,void *response,uint32_t response_bytes,int *fds,uint32_t *received,uint64_t timeout)
{
	uint64_t now = SparkWeightdMonotonicTimeNs(),deadline;
	SparkStatus status;
	*received = 0u;
	if ( timeout == 0u )
		timeout = SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS;
	if ( now == 0u || timeout > (UINT64_MAX - now) )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	deadline = (now + timeout);
	status = SparkWeightdClientWriteAll(client,request,request_bytes,deadline);
	if ( status == SPARK_STATUS_OK )
		status = SparkWeightdClientReadFrameWithFds(client,response,response_bytes,deadline,fds,SPARK_WEIGHTD_EXPORT_BATCH_MAX,received);
	if ( status != SPARK_STATUS_OK )
	{
		(void)close(client->fd);
		client->fd = -1;
	}
	SPARK_RETURN(status);
}

SparkStatus SparkWeightdClientExportBatch(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t batch_offset,
    SparkWeightdExportBatch *batch,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcExport wire;
    SparkWeightdIpcExportResult wire_result;
    uint32_t fds_received = 0u;
    SparkStatus status;

    if (client == 0 || batch == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    if (client->next_request_id == UINT64_MAX)
        SPARK_FAIL(SPARK_STATUS_CAPACITY_EXCEEDED);
    memset(batch, 0, sizeof(*batch));
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_EXPORT;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_EXPORT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.arena_generation = arena_generation;
    wire.batch_offset = batch_offset;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExportExchange(client,&wire,sizeof(wire),&wire_result,sizeof(wire_result),batch->fds,&fds_received,timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    status = SparkWeightdIpcValidateHeader(&wire_result.header,
        SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES,
        SPARK_WEIGHTD_IPC_KIND_EXPORT_RESULT);
    if (status != SPARK_STATUS_OK ||
        wire_result.header.request_id != wire.header.request_id ||
        wire_result.batch_offset != batch_offset ||
        wire_result.batch_count > SPARK_WEIGHTD_EXPORT_BATCH_MAX ||
        wire_result.batch_count != fds_received)
    {
        /* frame/refusal-shape faults close every fd already received: the
         * batch is either exactly what the frame declares or it is nothing */
        while (fds_received != 0u)
        {
            (void)close(batch->fds[--fds_received]);
        }
        (void)close(client->fd);
        client->fd = -1;
        return status != SPARK_STATUS_OK ? status : SPARK_STATUS_SCHEMA_ERROR;
    }
    batch->status = SparkWeightdStatusFromWire(wire_result.status);
    batch->arena_generation = wire_result.arena_generation;
    batch->chunk_bytes = wire_result.chunk_bytes;
    batch->chunk_count = wire_result.chunk_count;
    batch->batch_offset = wire_result.batch_offset;
    batch->batch_count = wire_result.batch_count;
    return SPARK_STATUS_OK;
}


static SparkStatus SparkWeightdValidateLeaseExport(const SparkWeightdIpcExportLease *request,const SparkWeightdIpcExportLeaseResult *response,uint32_t fds_received)
{
	const SparkWeightdIpcExportResult *base = &response->base;
	uint32_t i,count;
	SparkStatus status;
	status = SparkWeightdIpcValidateHeader(&base->header,sizeof(*response),SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( base->header.request_id != request->header.request_id || base->arena_generation != request->arena_generation || response->lease_identifier != request->lease_identifier || base->batch_offset != request->batch_offset || base->batch_count != fds_received || base->batch_count > SPARK_WEIGHTD_EXPORT_BATCH_MAX || base->reserved0 != 0u || response->reserved1 != 0u || base->status > SPARK_STATUS_UNSUPPORTED )
		return(SPARK_STATUS_SCHEMA_ERROR);
	if ( base->status != SPARK_STATUS_OK )
		return(base->batch_count == 0u ? SPARK_STATUS_OK : SPARK_STATUS_SCHEMA_ERROR);
	if ( base->chunk_bytes == 0u || base->chunk_count == 0u || base->chunk_count > SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX || base->chunk_bytes > (UINT64_MAX / base->chunk_count) || response->lease_chunk_count > base->chunk_count || (response->lease_chunk_count != 0u && base->batch_offset >= response->lease_chunk_count) )
		return(SPARK_STATUS_SCHEMA_ERROR);
	count = (response->lease_chunk_count - base->batch_offset);
	if ( count > SPARK_WEIGHTD_EXPORT_BATCH_MAX )
		count = SPARK_WEIGHTD_EXPORT_BATCH_MAX;
	if ( base->batch_count != count )
		return(SPARK_STATUS_SCHEMA_ERROR);
	for (i=0u; i<count; i++)
		if ( response->chunk_indices[i] >= base->chunk_count || (i != 0u && response->chunk_indices[i] <= response->chunk_indices[i - 1u]) )
			return(SPARK_STATUS_SCHEMA_ERROR);
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdClientExportLeaseBatch(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,uint32_t batch_offset,SparkWeightdExportBatch *batch,uint64_t timeout_nanoseconds)
{
	SparkWeightdIpcExportLease request;
	SparkWeightdIpcExportLeaseResult response;
	uint32_t received = 0u;
	SparkStatus status;
	if ( client == 0 || batch == 0 || lease_identifier == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	memset(batch,0,sizeof(*batch));
	if ( client->next_request_id == UINT64_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&request,0,sizeof(request));
	memset(&response,0,sizeof(response));
	SparkWeightdBuildHeader((uint8_t *)&request,SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE,++client->next_request_id);
	request.arena_generation = arena_generation;
	request.lease_identifier = lease_identifier;
	request.batch_offset = batch_offset;
	status = SparkWeightdClientExportExchange(client,&request,sizeof(request),&response,sizeof(response),batch->fds,&received,timeout_nanoseconds);
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	status = SparkWeightdValidateLeaseExport(&request,&response,received);
	if ( status != SPARK_STATUS_OK )
	{
		while ( received != 0u )
			(void)close(batch->fds[--received]);
		(void)close(client->fd);
		client->fd = -1;
		SPARK_RETURN(status);
	}
	batch->status = SparkWeightdStatusFromWire(response.base.status);
	batch->arena_generation = response.base.arena_generation;
	batch->lease_identifier = response.lease_identifier;
	batch->chunk_bytes = response.base.chunk_bytes;
	batch->chunk_count = response.base.chunk_count;
	batch->lease_chunk_count = response.lease_chunk_count;
	batch->batch_offset = response.base.batch_offset;
	batch->batch_count = response.base.batch_count;
	memcpy(batch->chunk_indices,response.chunk_indices,sizeof(batch->chunk_indices));
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdClientEpochExport(SparkWeightdClient *client,
    uint64_t arena_generation,
    int *fd_out,
    uint64_t timeout_nanoseconds)
{
	SparkWeightdIpcEpochExport request;
	SparkWeightdIpcEpochExportResult response;
	int fds[1];
	uint32_t received = 0u;
	SparkStatus status;
	if (client == 0 || fd_out == 0)
		return(SPARK_STATUS_INVALID_ARGUMENT);
	*fd_out = -1;
	if (client->next_request_id == UINT64_MAX)
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&request,0,sizeof(request));
	memset(&response,0,sizeof(response));
	fds[0] = -1;
	SparkWeightdBuildHeader((uint8_t *)&request,
	    SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT,++client->next_request_id);
	request.arena_generation = arena_generation;
	status = SparkWeightdClientExportExchange(client,&request,
	    sizeof(request),&response,sizeof(response),fds,&received,
	    timeout_nanoseconds);
	if (status != SPARK_STATUS_OK)
		SPARK_RETURN(status);
	if (response.status != (uint32_t)SPARK_STATUS_OK || received != 1u)
	{
		while (received != 0u)
			(void)close(fds[--received]);
		return(SPARK_STATUS_IO_ERROR);
	}
	*fd_out = fds[0];
	return(SPARK_STATUS_OK);
}

SparkStatus SparkWeightdClientDetach(SparkWeightdClient *client,
    uint64_t arena_generation,
    SparkWeightdDetachResult *result,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcDetach wire;
    SparkWeightdIpcDetachResult wire_result;
    SparkStatus status;

    if (client == 0 || result == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(result, 0, sizeof(*result));
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_DETACH;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_DETACH_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    wire.arena_generation = arena_generation;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        SPARK_WEIGHTD_IPC_DETACH_BYTES, &wire_result,
        SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES, timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    result->status = SparkWeightdStatusFromWire(wire_result.status);
    result->resident_bytes = wire_result.resident_bytes;
    result->refcount = wire_result.refcount;
    result->arena_count = wire_result.arena_count;
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *client,
    SparkWeightdReclaimResult *result,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcReclaim wire;
    SparkWeightdIpcReclaimResult wire_result;
    SparkStatus status;

    if (client == 0 || result == 0)
    {
        SPARK_FAIL(SPARK_STATUS_INVALID_ARGUMENT);
    }
    memset(result, 0, sizeof(*result));
    memset(&wire, 0, sizeof(wire));
    wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    wire.header.kind = SPARK_WEIGHTD_IPC_KIND_RECLAIM;
    wire.header.body_bytes =
        SPARK_WEIGHTD_IPC_RECLAIM_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
    wire.header.request_id = ++client->next_request_id;
    memset(&wire_result, 0, sizeof(wire_result));
    status = SparkWeightdClientExchange(client, &wire,
        SPARK_WEIGHTD_IPC_RECLAIM_BYTES, &wire_result,
        SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES, timeout_nanoseconds);
    if (status != SPARK_STATUS_OK)
    {
        SPARK_RETURN(status);
    }
    result->status = SparkWeightdStatusFromWire(wire_result.status);
    result->reclaimed_bytes = wire_result.reclaimed_bytes;
    result->resident_bytes = wire_result.resident_bytes;
    result->reclaimed_arena_count = wire_result.reclaimed_arena_count;
    result->arena_count = wire_result.arena_count;
    return SPARK_STATUS_OK;
}

void SparkWeightdClientClose(SparkWeightdClient *client)
{
    if (client == 0)
    {
        return;
    }
    if (client->fd >= 0)
    {
        (void)close(client->fd);
    }
    free(client);
}

static SparkStatus SparkWeightdWorkingSetExchange(SparkWeightdClient *client,void *wire,uint32_t bytes,SparkWeightdWorkingSetResult *result,uint64_t generation,uint64_t timeout)
{
	SparkWeightdIpcAcquireResult response;
	SparkWeightdIpcHeader *header = wire;
	SparkStatus status;
	memset(result,0,sizeof(*result));
	memset(&response,0,sizeof(response));
	status = SparkWeightdClientExchange(client,wire,bytes,&response,sizeof(response),timeout);
	result->status = status;
	if ( status != SPARK_STATUS_OK )
		SPARK_RETURN(status);
	if ( response.arena_generation != generation || (response.status == SPARK_STATUS_OK && header->kind == SPARK_WEIGHTD_IPC_KIND_ACQUIRE && response.lease_identifier == 0u) )
	{
		result->status = SPARK_STATUS_SCHEMA_ERROR;
		return(result->status);
	}
	result->status = SparkWeightdStatusFromWire(response.status);
	result->arena_generation = response.arena_generation;
	result->lease_identifier = response.lease_identifier;
	result->resident_bytes = response.resident_bytes;
	return(result->status);
}

SparkStatus SparkWeightdClientAcquire(SparkWeightdClient *client,uint64_t arena_generation,const SparkWeightdExpertKey *keys,uint32_t count,SparkWeightdWorkingSetResult *result,uint64_t timeout_nanoseconds)
{
	SparkWeightdIpcAcquire wire;
	if ( client == 0 || result == 0 || keys == 0 || count == 0u || count > SPARK_WEIGHTD_LEASE_GROUPS_MAX )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->next_request_id == UINT64_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&wire,0,sizeof(wire));
	SparkWeightdBuildHeader((uint8_t *)&wire,SPARK_WEIGHTD_IPC_KIND_ACQUIRE,++client->next_request_id);
	wire.arena_generation = arena_generation;
	wire.count = count;
	memcpy(wire.keys,keys,(count * sizeof(*keys)));
	return(SparkWeightdWorkingSetExchange(client,&wire,sizeof(wire),result,arena_generation,timeout_nanoseconds));
}

SparkStatus SparkWeightdClientRelease(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,SparkWeightdWorkingSetResult *result,uint64_t timeout_nanoseconds)
{
	SparkWeightdIpcRelease wire;
	if ( client == 0 || result == 0 || lease_identifier == 0u )
		return(SPARK_STATUS_INVALID_ARGUMENT);
	if ( client->next_request_id == UINT64_MAX )
		return(SPARK_STATUS_CAPACITY_EXCEEDED);
	memset(&wire,0,sizeof(wire));
	SparkWeightdBuildHeader((uint8_t *)&wire,SPARK_WEIGHTD_IPC_KIND_RELEASE,++client->next_request_id);
	wire.arena_generation = arena_generation;
	wire.lease_identifier = lease_identifier;
	return(SparkWeightdWorkingSetExchange(client,&wire,sizeof(wire),result,arena_generation,timeout_nanoseconds));
}
