/* spark_weightd core (docs/WEIGHTD_DESIGN.md W2a): the identity-keyed arena
 * map, the attach/detach IPC surface, and the poll-driven server loop.
 *
 * Shape of the skeleton:
 * - The server is event-driven (one non-blocking Step; no worker threads),
 *   so the fleet-facing Run loop always comes back around to its stop flag
 *   within one poll quantum — the TERM path cannot be starved by a peer.
 * - Loads are synchronous inside Step: the NO-2x budget rules out the
 *   background-load variant (design doc), so an update is stop-attach-
 *   start and the transient copy count is exactly zero.
 * - Every allocation names its space kind (the memory-space rule):
 *   arenas are cuMem* VMM virtual arenas (cuMemCreate physical chunks at
 *   2 MiB granularity mapped into a reserved span — W2b; consumer import+
 *   map is the staged tier), pack staging is host anonymous memory, one
 *   bounded chunk.
 * - The pack streams once through explicit 64 MiB pread chunks into a
 *   staging buffer: every chunk feeds the digest (the fast ck128 sidecar
 *   when the placement wrote one, sha256 otherwise) and the arena copy in
 *   the same pass, and the file's identity (size + mtime) is re-checked
 *   after the load. pread (not mmap fault-in) keeps the read at full
 *   sequential speed independent of page-cache state. The arena is
 *   published only after both pass, so bytes that do not hash to the
 *   identity are never served (fail-closed HASH_MISMATCH). That is the
 *   exact guarantee the tenant-scribble protection and the determinism
 *   receipts ride on. */

#define _POSIX_C_SOURCE 200809L
#if defined(__APPLE__)
/* st_mtimespec lives behind the Darwin extensions, which _POSIX_C_SOURCE
 * alone turns off */
#define _DARWIN_C_SOURCE 1
#endif

#include "sparkpipe/spark_weightd.h"

#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <time.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <cuda.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_ck128.h"

#define SPARK_WEIGHTD_LOAD_CHUNK_BYTES (64ull * 1024ull * 1024ull)
#define SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS 10000000000ull

/* The VMM page law (docs/WEIGHTD_DESIGN.md): a 25-100 GiB arena must not
 * drown the TLB in 4 KiB pages. Physical chunks are created at the driver's
 * recommended allocation granularity but never below 2 MiB, and every chunk
 * size stays a multiple of that granularity (cuMemCreate requires it). */
#define SPARK_WEIGHTD_VMM_CHUNK_BYTES (64ull * 1024ull * 1024ull)

typedef struct SparkWeightdArena
{
    SparkWeightdIdentity identity; /* canonical (prepared) */
    void *device_base;             /* mapped VMM virtual base (stable VA) */
    uint64_t virtual_bytes;        /* reserved span: chunk_count * chunk_bytes */
    uint64_t chunk_bytes;          /* physical chunk granularity */
    void **chunk_handles;          /* cuMem physical handles, one per chunk */
    uint32_t chunk_count;
    uint64_t generation;
    uint32_t refcount;
} SparkWeightdArena;

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
    uint32_t request_bytes;
    uint32_t attach_count;
    uint8_t request[SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX];
    uint8_t response[SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX];
    uint32_t response_bytes;
    uint32_t response_written;
    /* W3 fd tier: the EXPORT_RESULT reply leaves with the chunk fds in its
     * SCM_RIGHTS ancillary. The fds are staged here until the first
     * completed send (the kernel takes its own references at send time, so
     * they are handed across EXACTLY once) and are ours to close on every
     * other path — flush error, connection death, server teardown. */
    uint32_t response_fd_count;
    int response_fds[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
    SparkWeightdAttachRef attaches[SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX];
} SparkWeightdConnection;

struct SparkWeightdServer
{
    SparkWeightdServerConfig config;
    char socket_path[SPARK_WEIGHTD_SOCKET_PATH_BYTES];
    int listen_fd;
    uint64_t daemon_generation;
    uint64_t next_arena_generation;
    uint32_t arena_count;
    uint64_t resident_bytes;
    SparkWeightdArena arenas[SPARK_WEIGHTD_ARENA_COUNT_MAX];
    SparkWeightdConnection connections[SPARK_WEIGHTD_CONNECTION_COUNT_MAX];
};

struct SparkWeightdClient
{
    int fd;
    uint64_t next_request_id;
};

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
    return SPARK_STATUS_INVALID_ARGUMENT;
}

/* ------------------------------ identity ------------------------------ */

SparkStatus SparkWeightdIdentityPrepare(SparkWeightdIdentity *identity)
{
    uint32_t index;
    size_t model_bytes;
    size_t revision_bytes;

    if (identity == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (identity->abi_version == 0u || identity->arena_bytes == 0ull)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkWeightdStringBounded(identity->model, SPARK_WEIGHTD_ID_BYTES) !=
        SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkWeightdStringBounded(identity->revision,
            SPARK_WEIGHTD_REVISION_BYTES) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!SparkSha256HexIsValid(identity->pack_sha256))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
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
        default:
            return 0u;
    }
}

static uint32_t SparkWeightdKindResultKind(uint32_t kind)
{
    switch (kind)
    {
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
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (message_bytes < SPARK_WEIGHTD_IPC_HEADER_BYTES)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (header->magic != SPARK_WEIGHTD_IPC_MAGIC)
    {
        return SPARK_STATUS_PARSE_ERROR;
    }
    if (header->abi_version != SPARK_WEIGHTD_IPC_ABI_VERSION)
    {
        return SPARK_STATUS_ABI_MISMATCH;
    }
    if (header->kind != expected_kind)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (header->body_bytes != SparkWeightdKindBodyBytes(expected_kind))
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    if (message_bytes != SPARK_WEIGHTD_IPC_HEADER_BYTES + header->body_bytes)
    {
        return SPARK_STATUS_SCHEMA_ERROR;
    }
    return SPARK_STATUS_OK;
}

static SparkStatus SparkWeightdStatusFromWire(uint32_t wire_status)
{
    if (wire_status > (uint32_t)SPARK_STATUS_UNSUPPORTED)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return (SparkStatus)wire_status;
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
     * the host tests */
    (void)cuMemUnmap(base, (size_t)arena->virtual_bytes);
    for (index = 0u; index < arena->chunk_count; index++)
    {
        (void)cuMemRelease((CUmemGenericAllocationHandle)arena->chunk_handles[index]);
    }
    (void)cuMemAddressFree(base, (size_t)arena->virtual_bytes);
    free(arena->chunk_handles);
    arena->chunk_handles = 0;
    arena->chunk_count = 0u;
    arena->device_base = 0;
    arena->virtual_bytes = 0ull;
    arena->chunk_bytes = 0ull;
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
        return SPARK_STATUS_UNSUPPORTED;
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
        return SPARK_STATUS_UNSUPPORTED;
    }
    chunk_bytes = SparkWeightdVmmRoundUp(
        (uint64_t)granularity < SPARK_WEIGHTD_VMM_CHUNK_BYTES
            ? SPARK_WEIGHTD_VMM_CHUNK_BYTES
            : (uint64_t)granularity,
        (uint64_t)granularity);
    chunk_count = (uint32_t)((arena_bytes + chunk_bytes - 1ull) / chunk_bytes);

    arena->chunk_handles = (void **)calloc(chunk_count, sizeof(void *));
    if (arena->chunk_handles == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (cuMemAddressReserve(&base,
            (size_t)(chunk_count * chunk_bytes), 0u, 0ull, 0ull) != CUDA_SUCCESS)
    {
        free(arena->chunk_handles);
        arena->chunk_handles = 0;
        return SPARK_STATUS_CAPACITY_EXCEEDED;
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
        return SPARK_STATUS_OK;
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
    arena->chunk_handles = 0;
    return SPARK_STATUS_CAPACITY_EXCEEDED;
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
    server->resident_bytes -= server->arenas[slot].identity.arena_bytes;
    server->arena_count--;
    if (slot == server->arena_count)
    {
        memset(&server->arenas[slot], 0, sizeof(server->arenas[slot]));
        return;
    }
    moved_generation = server->arenas[server->arena_count].generation;
    server->arenas[slot] = server->arenas[server->arena_count];
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
    while (server->resident_bytes + needed_bytes > server->config.device_bytes_max)
    {
        uint32_t oldest_slot = server->arena_count;
        uint32_t index;
        for (index = 0u; index < server->arena_count; index++)
        {
            if (server->arenas[index].refcount == 0u &&
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
            return SPARK_STATUS_DUPLICATE;
        }
    }
    if (connection->attach_count >= SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
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
        return SPARK_STATUS_NOT_FOUND;
    }
    sidecar = fopen(sidecar_path, "rb");
    if (sidecar == 0)
    {
        return SPARK_STATUS_NOT_FOUND;
    }
    if (fread(hex, 1u, 32u, sidecar) != 32u || hex[0] == '\0')
    {
        (void)fclose(sidecar);
        return SPARK_STATUS_NOT_FOUND;
    }
    hex[32] = '\0';
    (void)fclose(sidecar);
    for (index = 0u; index < 32u; index++)
    {
        char c = hex[index];
        if ((c < '0' || c > '9') && (c < 'a' || c > 'f'))
        {
            return SPARK_STATUS_NOT_FOUND;
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
        result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
        return;
    }
    if (fstat(fileno(file), &pack_stat_before) != 0 ||
        pack_stat_before.st_size < 0 ||
        (uint64_t)pack_stat_before.st_size != identity.arena_bytes)
    {
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
        size_t got = 0u;
        while (got < chunk)
        {
            ssize_t n = pread(fileno(file), staging + got, chunk - got,
                (off_t)(loaded + got));
            if (n <= 0)
            {
                break;
            }
            got += (size_t)n;
        }
        if (got != chunk)
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

/* ------------------------------ server: export (W3 fd tier) ------------------------------ */

static void SparkWeightdServerCloseStagedFds(SparkWeightdConnection *connection)
{
    uint32_t index;
    for (index = 0u; index < connection->response_fd_count; index++)
    {
        (void)close(connection->response_fds[index]);
    }
    connection->response_fd_count = 0u;
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
    if (request->batch_offset >= arena->chunk_count)
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
        int fd = -1;
        if (cuMemExportToShareableHandle(&fd,
                (CUmemGenericAllocationHandle)
                    arena->chunk_handles[request->batch_offset + index],
                CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
                0ull) != CUDA_SUCCESS ||
            fd < 0)
        {
            SparkWeightdServerCloseStagedFds(connection);
            result->status = (uint32_t)SPARK_STATUS_IO_ERROR;
            return;
        }
        /* the fd discipline: chunk fds never survive an exec in the daemon */
        (void)fcntl(fd, F_SETFD, FD_CLOEXEC);
        connection->response_fds[index] = fd;
    }
    connection->response_fd_count = count;
    result->chunk_bytes = arena->chunk_bytes;
    result->chunk_count = arena->chunk_count;
    result->batch_count = count;
    result->status = (uint32_t)SPARK_STATUS_OK;
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
        connection->hello_done = 1u;
        SparkWeightdBuildHeader(response, result_kind, request_id);
        ack->daemon_generation = server->daemon_generation;
        ack->resident_bytes = server->resident_bytes;
        ack->device_bytes_max = server->config.device_bytes_max;
        ack->status = (uint32_t)SPARK_STATUS_OK;
        ack->arena_count = server->arena_count;
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
            if (server->arenas[index - 1u].refcount == 0u)
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

/* ------------------------------ server: connections ------------------------------ */

static void SparkWeightdServerCloseConnection(SparkWeightdServer *server,
    uint32_t connection_index)
{
    SparkWeightdConnection *connection = &server->connections[connection_index];
    while (connection->attach_count != 0u)
    {
        /* consumer death drops a refcount — every one of them */
        SparkWeightdServerDetachRelease(server, connection, 0u);
    }
    if (connection->fd >= 0)
    {
        (void)close(connection->fd);
    }
    SparkWeightdServerCloseStagedFds(connection); /* unsent chunk fds die here */
    connection->fd = -1;
    connection->state = SPARK_WEIGHTD_CONNECTION_CLOSED;
    connection->hello_done = 0u;
    connection->request_bytes = 0u;
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
            char control[CMSG_SPACE(SPARK_WEIGHTD_EXPORT_BATCH_MAX *
                sizeof(int))];
            struct cmsghdr *control_header;
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
                return SPARK_STATUS_PENDING;
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
                    return SPARK_STATUS_PENDING;
                }
                if (errno == EINTR)
                {
                    continue;
                }
                return SPARK_STATUS_IO_ERROR;
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
        if (connection->response_bytes != 0u)
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
    struct pollfd poll_fds[SPARK_WEIGHTD_CONNECTION_COUNT_MAX + 1u];
    uint32_t poll_count = 0u;
    uint32_t connection_index;
    uint32_t poll_index;
    int poll_result;

    if (server == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    if (server->listen_fd >= 0)
    {
        poll_fds[poll_count].fd = server->listen_fd;
        poll_fds[poll_count].events = POLLIN;
        poll_fds[poll_count].revents = 0;
        poll_count++;
    }
    for (connection_index = 0u;
        connection_index < SPARK_WEIGHTD_CONNECTION_COUNT_MAX;
        connection_index++)
    {
        if (server->connections[connection_index].state ==
            SPARK_WEIGHTD_CONNECTION_OPEN)
        {
            poll_fds[poll_count].fd = server->connections[connection_index].fd;
            poll_fds[poll_count].events = POLLIN;
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
        return SPARK_STATUS_IO_ERROR;
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
                if (connection->state != SPARK_WEIGHTD_CONNECTION_OPEN)
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
                    connection->response_bytes = 0u;
                    connection->response_written = 0u;
                    connection->response_fd_count = 0u;
                    connection->attach_count = 0u;
                    memset(connection->attaches, 0,
                        sizeof(connection->attaches));
                    break;
                }
            }
            /* a full table just declines further connects; a pending peer
             * sees its own connect deadline expire (fail-closed, both ways) */
        }
        poll_index = 1u;
    }

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
            server->connections[connection_index].fd >= 0)
        {
            SparkWeightdServerCloseConnection(server, connection_index);
        }
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkWeightdServerRun(SparkWeightdServer *server,
    const volatile sig_atomic_t *stop)
{
    if (server == 0 || stop == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    /* the stop flag is written by a signal handler in the daemon and by
     * the driving thread in tests: the load is atomic so both callers are
     * race-free by construction */
    while (__atomic_load_n(stop, __ATOMIC_SEQ_CST) == 0)
    {
        SparkStatus status = SparkWeightdServerStep(server);
        if (status != SPARK_STATUS_OK)
        {
            return status;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    instance = (SparkWeightdServer *)calloc(1u, sizeof(*instance));
    if (instance == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    instance->config = *config;
    instance->config.socket_path = instance->socket_path;
    memset(instance->socket_path, 0, sizeof(instance->socket_path));
    memcpy(instance->socket_path, config->socket_path,
        strlen(config->socket_path) + 1u);
    instance->listen_fd = -1;
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
        return SPARK_STATUS_IO_ERROR;
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
        return SPARK_STATUS_IO_ERROR;
    }
    (void)chmod(instance->socket_path, 0600);
    instance->daemon_generation = (uint64_t)time(0);
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
    (void)unlink(server->config.socket_path);
    free(server);
}

uint32_t SparkWeightdServerArenaCount(const SparkWeightdServer *server)
{
    return server != 0 ? server->arena_count : 0u;
}

uint64_t SparkWeightdServerResidentBytes(const SparkWeightdServer *server)
{
    return server != 0 ? server->resident_bytes : 0ull;
}

/* ------------------------------ client ------------------------------ */

static SparkStatus SparkWeightdDeadlineRemaining(uint64_t deadline_ns,
    int *timeout_ms)
{
    uint64_t now = SparkWeightdMonotonicTimeNs();
    if (now == 0ull)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (now >= deadline_ns)
    {
        return SPARK_STATUS_BUSY;
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
        ssize_t chunk = write(client->fd, buffer + written, bytes - written);
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
                return SPARK_STATUS_IO_ERROR;
            }
            poll_fd.fd = client->fd;
            poll_fd.events = POLLOUT;
            poll_fd.revents = 0;
            if (SparkWeightdDeadlineRemaining(deadline_ns, &timeout_ms) !=
                SPARK_STATUS_OK)
            {
                return SPARK_STATUS_BUSY;
            }
            if (poll(&poll_fd, 1u, timeout_ms) <= 0)
            {
                return SPARK_STATUS_BUSY;
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
            return deadline_status;
        }
        poll_fd.fd = client->fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        if (poll(&poll_fd, 1u, timeout_ms) <= 0)
        {
            return SPARK_STATUS_BUSY;
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
            return SPARK_STATUS_IO_ERROR;
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
    const SparkWeightdIpcHeader *request_header =
        (const SparkWeightdIpcHeader *)request;
    SparkWeightdIpcHeader *response_header = (SparkWeightdIpcHeader *)response;
    uint64_t now = SparkWeightdMonotonicTimeNs();
    uint64_t deadline_ns = now + (timeout_nanoseconds != 0ull
        ? timeout_nanoseconds
        : SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS);
    SparkStatus status;

    if (now == 0ull)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    status = SparkWeightdClientWriteAll(client, (const uint8_t *)request,
        request_bytes, deadline_ns);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWeightdClientReadAll(client, (uint8_t *)response,
        response_bytes, deadline_ns);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkWeightdIpcValidateHeader(response_header, response_bytes,
        SparkWeightdKindResultKind(request_header->kind));
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (response_header->request_id != request_header->request_id)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    return SPARK_STATUS_OK;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (hello_out != 0)
    {
        memset(hello_out, 0, sizeof(*hello_out));
    }
    instance = (SparkWeightdClient *)calloc(1u, sizeof(*instance));
    if (instance == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    instance->fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (instance->fd < 0)
    {
        free(instance);
        return SPARK_STATUS_IO_ERROR;
    }
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
        return SPARK_STATUS_IO_ERROR;
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
        return status;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(result, 0, sizeof(*result));
    identity = request->identity;
    status = SparkWeightdIdentityPrepare(&identity);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (SparkWeightdStringBounded(request->pack_path,
            SPARK_WEIGHTD_PATH_BYTES) != SPARK_STATUS_OK)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
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
        return status;
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

/* ------------------------------ client: export (W3 fd tier) ------------------------------ */

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
    char control[CMSG_SPACE(SPARK_WEIGHTD_EXPORT_BATCH_MAX * sizeof(int))];
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
            struct cmsghdr *control_header;
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
            if ((message.msg_flags & MSG_CTRUNC) != 0)
            {
                goto fail; /* truncated ancillary: a frame that cannot count */
            }
            for (control_header = CMSG_FIRSTHDR(&message); control_header != 0;
                control_header = CMSG_NXTHDR(&message, control_header))
            {
                if (control_header->cmsg_level == SOL_SOCKET &&
                    control_header->cmsg_type == SCM_RIGHTS)
                {
                    size_t payload = control_header->cmsg_len - CMSG_LEN(0);
                    uint32_t count = (uint32_t)(payload / sizeof(int));
                    if (payload % sizeof(int) != 0u ||
                        fd_count + count > fds_capacity)
                    {
                        /* over the batch bound: a lying frame, never mapped.
                         * Close what was copied; any overflow fds the kernel
                         * already queued die with this socket, which the
                         * caller closes on every failure path. */
                        goto fail;
                    }
                    memcpy(fds_out + fd_count, CMSG_DATA(control_header),
                        count * sizeof(int));
                    fd_count += count;
                }
            }
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
#ifndef MSG_CMSG_CLOEXEC
    {
        uint32_t index;
        for (index = 0u; index < fd_count; index++)
        {
            (void)fcntl(fds_out[index], F_SETFD, FD_CLOEXEC);
        }
    }
#endif
    *fds_received = fd_count;
    return SPARK_STATUS_OK;
fail:
    while (fd_count != 0u)
    {
        (void)close(fds_out[--fd_count]);
    }
    *fds_received = 0u;
    return SPARK_STATUS_IO_ERROR;
}

SparkStatus SparkWeightdClientExportBatch(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t batch_offset,
    SparkWeightdExportBatch *batch,
    uint64_t timeout_nanoseconds)
{
    SparkWeightdIpcExport wire;
    SparkWeightdIpcExportResult wire_result;
    uint64_t now;
    uint64_t deadline_ns;
    uint32_t fds_received = 0u;
    SparkStatus status;

    if (client == 0 || batch == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(batch, 0, sizeof(*batch));
    now = SparkWeightdMonotonicTimeNs();
    if (now == 0ull)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    deadline_ns = now + (timeout_nanoseconds != 0ull ? timeout_nanoseconds
                                                     : SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS);
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
    status = SparkWeightdClientWriteAll(client, (const uint8_t *)&wire,
        SPARK_WEIGHTD_IPC_EXPORT_BYTES, deadline_ns);
    if (status == SPARK_STATUS_OK)
    {
        status = SparkWeightdClientReadFrameWithFds(client,
            (uint8_t *)&wire_result, SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES,
            deadline_ns, batch->fds, SPARK_WEIGHTD_EXPORT_BATCH_MAX,
            &fds_received);
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
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
        return status;
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
        return SPARK_STATUS_INVALID_ARGUMENT;
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
        return status;
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
