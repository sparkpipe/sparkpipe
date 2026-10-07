#pragma once


#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_weightd_lease.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_CLIENT_TIMEOUT_DEFAULT_NS UINT64_C(10000000000)

#define SPARK_WEIGHTD_IPC_ABI_VERSION 13u
#define SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN 8u
#define SPARK_WEIGHTD_IPC_MAGIC UINT32_C(0x57444953)

#define SPARK_WEIGHTD_ID_BYTES 64u
#define SPARK_WEIGHTD_REVISION_BYTES 128u
#define SPARK_WEIGHTD_SHA256_HEX_BYTES 65u
#define SPARK_WEIGHTD_PATH_BYTES 1024u
#define SPARK_WEIGHTD_SOCKET_PATH_BYTES 108u

#define SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX 8192u

#define SPARK_WEIGHTD_ARENA_COUNT_MAX 16u
#define SPARK_WEIGHTD_MANIFEST_DAEMON_BYTES_MAX (512ull * 1024ull * 1024ull)
#define SPARK_WEIGHTD_CONNECTION_COUNT_MAX 128u
#define SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX 8u

#define SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT (110ull * 1024ull * 1024ull * 1024ull)

#define SPARK_WEIGHTD_SERVER_POLL_TIMEOUT_MS 20u

#define SPARK_WEIGHTD_IPC_KIND_HELLO 1u
#define SPARK_WEIGHTD_IPC_KIND_HELLO_ACK 2u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH 3u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_RESULT 4u
#define SPARK_WEIGHTD_IPC_KIND_DETACH 5u
#define SPARK_WEIGHTD_IPC_KIND_DETACH_RESULT 6u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM 7u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM_RESULT 8u
#define SPARK_WEIGHTD_IPC_KIND_EXPORT 9u
#define SPARK_WEIGHTD_IPC_KIND_EXPORT_RESULT 10u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY 11u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_RESULT 12u
#define SPARK_WEIGHTD_IPC_KIND_ENSURE 13u
#define SPARK_WEIGHTD_IPC_KIND_ENSURE_RESULT 14u
#define SPARK_WEIGHTD_IPC_KIND_ACQUIRE 15u
#define SPARK_WEIGHTD_IPC_KIND_ACQUIRE_RESULT 16u
#define SPARK_WEIGHTD_IPC_KIND_RELEASE 17u
#define SPARK_WEIGHTD_IPC_KIND_RELEASE_RESULT 18u
#define SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE 19u
#define SPARK_WEIGHTD_IPC_KIND_EXPORT_LEASE_RESULT 20u
#define SPARK_WEIGHTD_IPC_KIND_MESH_WRITE 21u
#define SPARK_WEIGHTD_IPC_KIND_MESH_WRITE_RESULT 22u
#define SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST 23u
#define SPARK_WEIGHTD_IPC_KIND_MESH_BROADCAST_RESULT 24u
#define SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT 25u
#define SPARK_WEIGHTD_IPC_KIND_EPOCH_EXPORT_RESULT 26u
#define SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE 27u
#define SPARK_WEIGHTD_IPC_KIND_LANE_ACQUIRE_RESULT 28u
#define SPARK_WEIGHTD_IPC_KIND_EVICT 29u
#define SPARK_WEIGHTD_IPC_KIND_EVICT_RESULT 30u
#define SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY 31u
#define SPARK_WEIGHTD_IPC_KIND_MESH_ACTIVITY_RESULT 32u
#define SPARK_WEIGHTD_IPC_KIND_MESH_MAP 33u
#define SPARK_WEIGHTD_IPC_KIND_MESH_MAP_RESULT 34u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM_PACK 35u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM_PACK_RESULT 36u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_SHARED 37u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_LAZY_SHARED_RESULT 38u
#define SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP 39u
#define SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP_RESULT 40u
#define SPARK_WEIGHTD_IPC_KIND_MESH_STATUS 41u
#define SPARK_WEIGHTD_IPC_KIND_MESH_STATUS_RESULT 42u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_ATTACH 43u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_ATTACH_RESULT 44u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_RESIZE 45u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_RESIZE_RESULT 46u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_EXPORT 47u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_EXPORT_RESULT 48u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_STATUS 49u
#define SPARK_WEIGHTD_IPC_KIND_KV_POOL_STATUS_RESULT 50u
#define SPARK_WEIGHTD_IPC_KIND_KV_SHARED_ATTACH 51u
#define SPARK_WEIGHTD_IPC_KIND_KV_SHARED_ATTACH_RESULT 52u
#define SPARK_WEIGHTD_IPC_KIND_SLOT_EXPORT 53u
#define SPARK_WEIGHTD_IPC_KIND_SLOT_EXPORT_RESULT 54u
#define SPARK_WEIGHTD_IPC_KIND_LEASE_SLOTS 55u
#define SPARK_WEIGHTD_IPC_KIND_LEASE_SLOTS_RESULT 56u
#define SPARK_WEIGHTD_IPC_KIND_RESIDENCY 57u
#define SPARK_WEIGHTD_IPC_KIND_RESIDENCY_RESULT 58u
#define SPARK_WEIGHTD_IPC_KIND_ABI_MIN(kind) ((kind) >= SPARK_WEIGHTD_IPC_KIND_SLOT_EXPORT ? 13u : (kind) >= SPARK_WEIGHTD_IPC_KIND_KV_SHARED_ATTACH ? 12u : (kind) >= SPARK_WEIGHTD_IPC_KIND_KV_POOL_ATTACH ? 11u : (kind) >= SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP ? 9u : SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN)
#define SPARK_WEIGHTD_SHARE_ENV "SPARK_WEIGHTD_SHARE"
#define SPARK_WEIGHTD_SHARE_READONLY "readonly"

#define SPARK_WEIGHTD_MESH_MAX_LANES 16u
#define SPARK_WEIGHTD_MESH_HOST_PAGE_BYTES (64u * 1024u)
#define SPARK_WEIGHTD_MESH_MAX_BATCH_ROWS 1024u
#define SPARK_WEIGHTD_MESH_ROW_BYTES_MAX (16u * 1024u * 2u)
#define SPARK_WEIGHTD_MESH_SLOT_ROWS 8u
#define SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES 64u
#define SPARK_WEIGHTD_MESH_SLOT_BYTES \
    (SPARK_WEIGHTD_MESH_SLOT_ROWS * SPARK_WEIGHTD_MESH_ROW_BYTES_MAX + SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES)
#define SPARK_WEIGHTD_MESH_RANKS_PER_BAND 16u
#define SPARK_WEIGHTD_MESH_RANKS SPARK_WEIGHTD_MESH_RANKS_PER_BAND
#define SPARK_WEIGHTD_MESH_SLOTS_PER_RANK 2u
#define SPARK_WEIGHTD_MESH_SLOTS_PER_BAND \
    (SPARK_WEIGHTD_MESH_RANKS_PER_BAND * SPARK_WEIGHTD_MESH_SLOTS_PER_RANK)
#define SPARK_WEIGHTD_MESH_BANDS (2u * SPARK_WEIGHTD_MESH_MAX_LANES)
#define SPARK_WEIGHTD_MESH_BUFFER_BYTES \
    ((uint64_t)SPARK_WEIGHTD_MESH_SLOT_BYTES * \
     SPARK_WEIGHTD_MESH_SLOTS_PER_BAND * SPARK_WEIGHTD_MESH_BANDS)
#define SPARK_WEIGHTD_MESH_DOORBELL_BYTES 131072u
#define SPARK_WEIGHTD_MESH_REGION_BYTES \
    (SPARK_WEIGHTD_MESH_BUFFER_BYTES + (uint64_t)SPARK_WEIGHTD_MESH_DOORBELL_BYTES)
#define SPARK_WEIGHTD_MESH_DOORBELL_OFFSET SPARK_WEIGHTD_MESH_BUFFER_BYTES
#define SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES 32u
#define SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS \
    (SPARK_WEIGHTD_MESH_BANDS * SPARK_WEIGHTD_MESH_RANKS_PER_BAND)
#define SPARK_WEIGHTD_MESH_DOORBELL_CELL_BASE \
    (SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS)
#define SPARK_WEIGHTD_MESH_DOORBELL_CELL_CANCEL \
    (SPARK_WEIGHTD_MESH_DOORBELL_CELL_BASE + 1u)
#define SPARK_WEIGHTD_MESH_DOORBELL_ENTRY(band,rank) \
    (SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + \
     (((band) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + (rank)) * \
         SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES))
#define SPARK_WEIGHTD_MESH_SHIPPED_CELL_BYTES 8u
#define SPARK_WEIGHTD_MESH_SHIPPED_OFFSET \
    (SPARK_WEIGHTD_MESH_DOORBELL_OFFSET + \
     (uint64_t)(SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS + \
        2u * SPARK_WEIGHTD_MESH_BANDS) * \
        SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES)
#define SPARK_WEIGHTD_MESH_SHIPPED_ENTRY(band,rank) \
    (SPARK_WEIGHTD_MESH_SHIPPED_OFFSET + \
     (((band) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + (rank)) * \
         SPARK_WEIGHTD_MESH_SHIPPED_CELL_BYTES))
#define SPARK_WEIGHTD_MESH_WAIT_VERSION 1u
#define SPARK_WEIGHTD_MESH_WAIT_SHIPPED 1u
#define SPARK_WEIGHTD_MESH_WAIT_PEERS 2u
#define SPARK_WEIGHTD_MESH_WAIT_ERROR_CANCELLED UINT64_C(0xFFFFFFFFFE000000)
#define SPARK_WEIGHTD_MESH_WAIT_ERROR_PEER_RESET UINT64_C(0xFFFFFFFFFD000000)
#define SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES 128u
#define SPARK_WEIGHTD_MESH_WAIT_OFFSET \
    (SPARK_WEIGHTD_MESH_SHIPPED_OFFSET + \
     (uint64_t)SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * \
        SPARK_WEIGHTD_MESH_SHIPPED_CELL_BYTES)
#define SPARK_WEIGHTD_MESH_WAIT_ENTRY(band,rank) \
    (SPARK_WEIGHTD_MESH_WAIT_OFFSET + \
     ((uint64_t)(band) * SPARK_WEIGHTD_MESH_RANKS_PER_BAND + (rank)) * \
        SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES)

typedef struct SparkWeightdMeshWaitRequest
{
    uint64_t request_id;
    uint64_t kind;
    uint64_t tag;
    uint64_t peer_mask;
    uint64_t cancel_expected;
    uint64_t timeout_ns;
    uint64_t version;
    uint64_t upstream_error;
    uint64_t ready;
    uint64_t error;
    uint64_t diag;
    uint64_t capabilities;
    uint64_t reserved[4];
} SparkWeightdMeshWaitRequest;

#define SPARK_WEIGHTD_MESH_CAPABILITY_SLICE_ROUTES 1u
#define SPARK_WEIGHTD_MESH_CAPABILITY_PEER_ROUTES 2u
#define SPARK_WEIGHTD_MESH_CAPABILITIES \
    (SPARK_WEIGHTD_MESH_CAPABILITY_SLICE_ROUTES | SPARK_WEIGHTD_MESH_CAPABILITY_PEER_ROUTES)
#define SPARK_WEIGHTD_MESH_ROUTE_FULL 0u
#define SPARK_WEIGHTD_MESH_ROUTE_SCATTER 1u
#define SPARK_WEIGHTD_MESH_ROUTE_GATHER 2u
#define SPARK_WEIGHTD_MESH_ROUTE_PEER 3u

#define SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES \
    ((uint64_t)SPARK_WEIGHTD_MESH_SLOT_ROWS * SPARK_WEIGHTD_MESH_ROW_BYTES_MAX)
#define SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES \
    (SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES * SPARK_WEIGHTD_MESH_RANKS_PER_BAND)
#define SPARK_WEIGHTD_MESH_STAGING_BYTES \
    (SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES * SPARK_WEIGHTD_MESH_BANDS)
#define SPARK_WEIGHTD_MESH_STAGING_OFFSET(band,peer) \
    ((uint64_t)(band) * SPARK_WEIGHTD_MESH_STAGING_BAND_BYTES + \
     (uint64_t)(peer) * SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES)

typedef struct SparkWeightdMeshRouteFields
{
    uint64_t peer_mask:32;
    uint64_t slice_bytes:24;
    uint64_t mode:2;
    uint64_t reserved:6;
} SparkWeightdMeshRouteFields;

typedef union SparkWeightdMeshRoute
{
    uint64_t word;
    SparkWeightdMeshRouteFields fields;
} SparkWeightdMeshRoute;

static inline uint32_t SparkWeightdMeshRouteValid(SparkWeightdMeshRoute route)
{
    return route.fields.reserved == 0u && (route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL ? route.fields.slice_bytes == 0u : route.fields.slice_bytes != 0u && (route.fields.slice_bytes & 7u) == 0u && (route.fields.mode != SPARK_WEIGHTD_MESH_ROUTE_PEER || route.fields.slice_bytes <= SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES));
}

static inline uint32_t SparkWeightdMeshRouteFits(SparkWeightdMeshRoute route,uint64_t bytes)
{
    return SparkWeightdMeshRouteValid(route) != 0u && bytes != 0u && (route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER ? bytes == route.fields.slice_bytes : bytes <= SPARK_WEIGHTD_MESH_SLOT_BYTES - 16u);
}

static inline uint64_t SparkWeightdMeshRouteSpan(SparkWeightdMeshRoute route,uint32_t peer,uint32_t local,uint64_t bytes,uint64_t *length)
{
    uint64_t begin;
    if ( route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_PEER )
    {
        *length = bytes < route.fields.slice_bytes ? bytes : route.fields.slice_bytes;
        return(0u);
    }
    begin = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL ? 0u : (uint64_t)(route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_SCATTER ? peer : local) * route.fields.slice_bytes;
    begin = begin < bytes ? begin : bytes;
    *length = route.fields.mode == SPARK_WEIGHTD_MESH_ROUTE_FULL || bytes - begin < route.fields.slice_bytes ? bytes - begin : route.fields.slice_bytes;
    return(begin);
}
#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdMeshWaitRequest) == SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES &&
    offsetof(SparkWeightdMeshWaitRequest,ready) == 64u,
    "mesh wait request and ready occupy distinct cache lines");
_Static_assert(sizeof(SparkWeightdMeshRoute) == sizeof(uint64_t),"mesh route is one doorbell word");
_Static_assert(SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES <= SPARK_WEIGHTD_MESH_SLOT_BYTES - SPARK_WEIGHTD_MESH_SLOT_TRAILER_BYTES &&
    SPARK_WEIGHTD_MESH_STAGING_SLOT_BYTES < (UINT64_C(1) << 24u),"a peer route fills at most one receive slot payload");
_Static_assert(SPARK_WEIGHTD_MESH_STAGING_BYTES <= UINT64_C(128) * 1024u * 1024u,"per-peer staging stays inside its node memory budget");
_Static_assert(SPARK_WEIGHTD_MESH_WAIT_OFFSET % SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES == 0u &&
    SPARK_WEIGHTD_MESH_WAIT_OFFSET - SPARK_WEIGHTD_MESH_DOORBELL_OFFSET +
    (uint64_t)SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * SPARK_WEIGHTD_MESH_WAIT_ENTRY_BYTES <=
    SPARK_WEIGHTD_MESH_DOORBELL_BYTES,"mesh wait requests fit the doorbell page");
_Static_assert(SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * \
    SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES <= SPARK_WEIGHTD_MESH_DOORBELL_BYTES,
    "doorbell entries must fit the doorbell page");
_Static_assert((SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS + 2u * \
        SPARK_WEIGHTD_MESH_BANDS) * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES <= \
    SPARK_WEIGHTD_MESH_DOORBELL_BYTES,
    "control cells must fit the doorbell page");
_Static_assert((SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS + 2u * \
        SPARK_WEIGHTD_MESH_BANDS) * SPARK_WEIGHTD_MESH_DOORBELL_ENTRY_BYTES + \
    SPARK_WEIGHTD_MESH_DOORBELL_RANK_CELLS * \
        SPARK_WEIGHTD_MESH_SHIPPED_CELL_BYTES <= \
    SPARK_WEIGHTD_MESH_DOORBELL_BYTES,
    "shipped cells must fit the doorbell page");
#endif

static inline uint32_t SparkWeightdMeshRewireNeeded(uint64_t record_boot_ns,
    uint64_t wired_boot_ns,uint32_t send_qp_in_rts,uint32_t recv_qp_in_rts)
{
	if ( record_boot_ns != wired_boot_ns )
		return(1u);
	if ( send_qp_in_rts == 0u || recv_qp_in_rts == 0u )
		return(1u);
	return(0u);
}

#define SPARK_WEIGHTD_EXPERT_COUNT_MAX 40960u

#define SPARK_WEIGHTD_EXPERT_BYTES_MAX (64ull * 1024ull * 1024ull)
#define SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC UINT32_C(0x58504557)
#define SPARK_WEIGHTD_EXPERT_MANIFEST_VERSION 1u

#define SPARK_WEIGHTD_EXPORT_BATCH_MAX 64u
#if !defined(__cplusplus)
_Static_assert(SPARK_WEIGHTD_EXPORT_BATCH_MAX <= 253u,
    "SPARK_WEIGHTD_EXPORT_BATCH_MAX must stay inside the kernel's "
    "SCM_MAX_FD (253): one EXPORT batch is one message's fd payload");
#endif

#define SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX 65536u
#define SPARK_WEIGHTD_LEASE_SLOTS_BATCH_MAX 512u
#define SPARK_WEIGHTD_SLOT_NONE UINT32_MAX

#define SPARK_WEIGHTD_KV_POOL_COUNT_MAX 32u
#define SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX 4096u
#define SPARK_WEIGHTD_KV_POOL_EXPORT_MAX SPARK_WEIGHTD_EXPORT_BATCH_MAX
#define SPARK_WEIGHTD_KV_POOL_KEY_BYTES 32u
#define SPARK_WEIGHTD_KV_POOL_LABEL_BYTES 64u
#define SPARK_WEIGHTD_KV_POOL_METADATA_BYTES_MAX (256ull * 1024ull * 1024ull)

typedef struct SparkWeightdIdentity
{
    uint64_t geometry_fingerprint;
    uint64_t arena_bytes;
    uint32_t abi_version;
    uint32_t topology;
    uint32_t reserved0;
    uint32_t reserved1;
    char model[SPARK_WEIGHTD_ID_BYTES];
    char revision[SPARK_WEIGHTD_REVISION_BYTES];
    char pack_sha256[SPARK_WEIGHTD_SHA256_HEX_BYTES];
    char reserved_tail[7];
} SparkWeightdIdentity;

typedef struct SparkWeightdIpcHeader
{
    uint32_t magic;
    uint32_t abi_version;
    uint32_t kind;
    uint32_t body_bytes;
    uint64_t request_id;
} SparkWeightdIpcHeader;

typedef struct SparkWeightdIpcMeshMapResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t reserved;
    uint64_t bytes;
} SparkWeightdIpcMeshMapResult;

typedef struct SparkWeightdIpcMeshStagingMapResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t capabilities;
    uint64_t bytes;
    uint64_t slot_bytes;
    uint64_t band_bytes;
} SparkWeightdIpcMeshStagingMapResult;

typedef struct SparkWeightdIpcHello
{
    SparkWeightdIpcHeader header;
} SparkWeightdIpcHello;

typedef struct SparkWeightdIpcHelloAck
{
    SparkWeightdIpcHeader header;
    uint64_t daemon_generation;
    uint64_t resident_bytes;
    uint64_t device_bytes_max;
    uint32_t status;
    uint32_t arena_count;
} SparkWeightdIpcHelloAck;

typedef struct SparkWeightdIpcAttach
{
    SparkWeightdIpcHeader header;
    SparkWeightdIdentity identity;
    char pack_path[SPARK_WEIGHTD_PATH_BYTES];
} SparkWeightdIpcAttach;

typedef struct SparkWeightdIpcAttachResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t device_handle;
    uint64_t arena_bytes;
    uint64_t resident_bytes;
    uint32_t status;
    uint32_t refcount;
    uint32_t arena_count;
    uint32_t loaded_from_pack;
} SparkWeightdIpcAttachResult;

typedef struct SparkWeightdIpcDetach
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
} SparkWeightdIpcDetach;

typedef struct SparkWeightdIpcDetachResult
{
    SparkWeightdIpcHeader header;
    uint64_t resident_bytes;
    uint32_t status;
    uint32_t refcount;
    uint32_t arena_count;
    uint32_t reserved0;
} SparkWeightdIpcDetachResult;

typedef struct SparkWeightdIpcReclaim
{
    SparkWeightdIpcHeader header;
} SparkWeightdIpcReclaim;

typedef struct SparkWeightdIpcReclaimResult
{
    SparkWeightdIpcHeader header;
    uint64_t reclaimed_bytes;
    uint64_t resident_bytes;
    uint32_t status;
    uint32_t reclaimed_arena_count;
    uint32_t arena_count;
    uint32_t busy_arena_count;
} SparkWeightdIpcReclaimResult;

typedef struct SparkWeightdIpcReclaimPack
{
    SparkWeightdIpcHeader header;
    char pack_sha256[SPARK_WEIGHTD_SHA256_HEX_BYTES];
    char reserved[7];
} SparkWeightdIpcReclaimPack;

typedef struct SparkWeightdIpcExport
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint32_t batch_offset;
    uint32_t reserved0;
} SparkWeightdIpcExport;

typedef struct SparkWeightdIpcExportResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint32_t batch_offset;
    uint32_t batch_count;
    uint32_t status;
    uint32_t reserved0;
} SparkWeightdIpcExportResult;

typedef struct SparkWeightdIpcAttachLazy
{
    SparkWeightdIpcHeader header;
    SparkWeightdIdentity identity;
    char pack_path[SPARK_WEIGHTD_PATH_BYTES];
    uint64_t expert_pool_bytes;
} SparkWeightdIpcAttachLazy;

typedef struct SparkWeightdIpcAttachLazyResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t device_handle;
    uint64_t arena_bytes;
    uint64_t resident_bytes;
    uint64_t expert_pool_bytes;
    uint32_t status;
    uint32_t refcount;
    uint32_t arena_count;
    uint32_t expert_count;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint32_t loaded_from_pack;
    uint32_t mesh_ready;
    uint32_t pool_fd_staged;
    uint64_t mesh_send_buffer_addr;
    uint64_t mesh_send_buffer_bytes;
    uint8_t manifest_sha256[32];
    int pool_fd;
} SparkWeightdIpcAttachLazyResult;

typedef struct SparkWeightdIpcSlotExport
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint32_t slot_offset;
    uint32_t reserved0;
} SparkWeightdIpcSlotExport;

typedef struct SparkWeightdIpcSlotExportResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t chunk_bytes;
    uint32_t status;
    uint32_t slot_count;
    uint32_t slot_offset;
    uint32_t batch_count;
} SparkWeightdIpcSlotExportResult;

typedef struct SparkWeightdIpcLeaseSlots
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t lease_identifier;
    uint32_t batch_offset;
    uint32_t reserved0;
} SparkWeightdIpcLeaseSlots;

typedef struct SparkWeightdIpcLeaseSlotsResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t lease_identifier;
    uint32_t status;
    uint32_t lease_chunk_count;
    uint32_t batch_offset;
    uint32_t batch_count;
    uint32_t chunk_indices[SPARK_WEIGHTD_LEASE_SLOTS_BATCH_MAX];
    uint32_t slot_indices[SPARK_WEIGHTD_LEASE_SLOTS_BATCH_MAX];
} SparkWeightdIpcLeaseSlotsResult;

#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdIpcLeaseSlotsResult) <= SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX,"lease slot batch exceeds IPC frame");
#endif

typedef struct SparkWeightdIpcEpochExport
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint32_t reserved;
} SparkWeightdIpcEpochExport;

typedef struct SparkWeightdIpcEpochExportResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t reserved;
} SparkWeightdIpcEpochExportResult;

typedef struct SparkWeightdIpcResidency
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
} SparkWeightdIpcResidency;

typedef struct SparkWeightdIpcResidencyResult
{
    SparkWeightdIpcHeader header;
    uint64_t present_bytes;
    uint64_t epoch;
    uint32_t status;
    uint32_t group_count;
    uint32_t present_count;
    uint32_t fixed_pool;
} SparkWeightdIpcResidencyResult;

typedef struct SparkWeightdMeshTopology
{
    uint32_t rank_count;
    uint32_t local_rank;
    uint32_t physical_ranks[16];
} SparkWeightdMeshTopology;

typedef struct SparkWeightdIpcLaneAcquire
{
    SparkWeightdIpcHeader header;
    uint32_t requested_lane;
    uint32_t reserved;
    SparkWeightdMeshTopology topology;
} SparkWeightdIpcLaneAcquire;

typedef struct SparkWeightdIpcLaneAcquireResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t lane;
} SparkWeightdIpcLaneAcquireResult;

typedef struct SparkWeightdIpcEvict
{
    SparkWeightdIpcHeader header;
    uint32_t target_lane;
    uint32_t reserved;
} SparkWeightdIpcEvict;

typedef struct SparkWeightdIpcEvictResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t released_leases;
} SparkWeightdIpcEvictResult;

#define SPARK_WEIGHTD_IPC_EVICT_BYTES ((uint32_t)sizeof(SparkWeightdIpcEvict))
#define SPARK_WEIGHTD_IPC_EVICT_RESULT_BYTES \
    ((uint32_t)sizeof(SparkWeightdIpcEvictResult))

#define SPARK_WEIGHTD_IPC_EPOCH_EXPORT_BYTES ((uint32_t)sizeof(SparkWeightdIpcEpochExport))
#define SPARK_WEIGHTD_IPC_EPOCH_EXPORT_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcEpochExportResult))

typedef struct SparkWeightdIpcMeshActivity
{
    SparkWeightdIpcHeader header;
    uint64_t generation;
    uint32_t active;
    uint32_t lane;
} SparkWeightdIpcMeshActivity;

typedef struct SparkWeightdIpcMeshActivityResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t reserved0;
} SparkWeightdIpcMeshActivityResult;

typedef struct SparkWeightdIpcMeshWrite
{
    SparkWeightdIpcHeader header;
    uint32_t peer_rank;
    uint32_t reserved;
    uint64_t source_offset;
    uint64_t remote_offset;
    uint32_t length;
    uint32_t reserved2;
} SparkWeightdIpcMeshWrite;

typedef struct SparkWeightdIpcMeshWriteResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t reserved;
} SparkWeightdIpcMeshWriteResult;

typedef struct SparkWeightdIpcMeshBroadcast
{
    SparkWeightdIpcHeader header;
    uint32_t peer_mask;
    uint32_t reserved;
    uint64_t source_offset;
    uint64_t remote_offset;
    uint32_t length;
    uint32_t reserved2;
    uint64_t seq_value;
    uint64_t seq_remote_offset;
} SparkWeightdIpcMeshBroadcast;

typedef struct SparkWeightdIpcMeshBroadcastResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t posted_count;
} SparkWeightdIpcMeshBroadcastResult;

#define SPARK_WEIGHTD_MESH_STATUS_LAYOUT 1u
#define SPARK_WEIGHTD_MESH_STATUS_LAYOUT_COMPAT 1u
#define SPARK_WEIGHTD_MESH_STATUS_COUNTERS 12u
#define SPARK_WEIGHTD_MESH_STATE_DISABLED 0u
#define SPARK_WEIGHTD_MESH_STATE_WIRING 1u
#define SPARK_WEIGHTD_MESH_STATE_READY 2u
#define SPARK_WEIGHTD_MESH_PEER_ABSENT 0u
#define SPARK_WEIGHTD_MESH_PEER_SELF 1u
#define SPARK_WEIGHTD_MESH_PEER_NO_RECORD 2u
#define SPARK_WEIGHTD_MESH_PEER_RECORD_REJECTED 3u
#define SPARK_WEIGHTD_MESH_PEER_RECORD_INVALID 4u
#define SPARK_WEIGHTD_MESH_PEER_WIRE_FAILED 5u
#define SPARK_WEIGHTD_MESH_PEER_WIRED 6u
#define SPARK_WEIGHTD_MESH_PEER_QP_SEND_RTS 1u
#define SPARK_WEIGHTD_MESH_PEER_QP_RECV_RTS 2u
#define SPARK_WEIGHTD_MESH_PEER_QP_PAIR_LINK 4u
#define SPARK_WEIGHTD_MESH_LANE_CONFIGURED 1u
#define SPARK_WEIGHTD_MESH_LANE_OWNED 2u
#define SPARK_WEIGHTD_MESH_LANE_QUARANTINED 4u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_ACTIVITY 1u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_RPC 2u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_BUFFER 3u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_PENDING 4u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_WAIT 5u
#define SPARK_WEIGHTD_MESH_LANE_BUSY_DOORBELL 6u
#define SPARK_WEIGHTD_MESH_QUERY_ANSWERED 0u
#define SPARK_WEIGHTD_MESH_QUERY_ABSENT 1u
#define SPARK_WEIGHTD_MESH_QUERY_UNRESPONSIVE 2u
#define SPARK_WEIGHTD_MESH_QUERY_UNSERVED 3u
#define SPARK_WEIGHTD_MESH_QUERY_INCOMPATIBLE 4u
#define SPARK_WEIGHTD_MESH_QUERY_FAULT 5u

typedef struct SparkWeightdIpcMeshStatus
{
    SparkWeightdIpcHeader header;
    uint32_t layout;
    uint32_t reserved;
} SparkWeightdIpcMeshStatus;

typedef struct SparkWeightdMeshPeerStatus
{
    uint64_t wired_boot_ns;
    uint64_t record_boot_ns;
    uint64_t since_mono_ns;
    uint64_t last_ok_mono_ns;
    uint64_t last_err_mono_ns;
    uint32_t state;
    uint32_t record_status;
    uint32_t qp_flags;
    uint32_t send_pending;
    uint32_t rpc_pending;
    uint32_t cq_err_since_ok;
} SparkWeightdMeshPeerStatus;

typedef struct SparkWeightdMeshLaneStatus
{
    uint64_t packed_ranks;
    uint32_t flags;
    uint32_t rank_count;
    uint32_t local_rank;
    uint32_t physical_mask;
    uint32_t activity;
    uint32_t failed_cells;
    uint32_t pending_cells;
    uint32_t configure_status;
    uint32_t busy_reason;
    uint32_t busy_index;
} SparkWeightdMeshLaneStatus;

typedef struct SparkWeightdIpcMeshStatusResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t layout;
    uint32_t layout_compat;
    uint32_t mesh_state;
    uint32_t pid;
    uint32_t local_rank;
    uint32_t rank_mask;
    uint32_t wired_mask;
    uint32_t pair_rank;
    uint32_t reserved0;
    uint64_t daemon_generation;
    uint64_t boot_ns;
    uint64_t mesh_generation;
    uint64_t now_mono_ns;
    uint64_t ready_since_mono_ns;
    uint64_t counters[SPARK_WEIGHTD_MESH_STATUS_COUNTERS];
    SparkWeightdMeshPeerStatus peers[SPARK_WEIGHTD_MESH_RANKS];
    SparkWeightdMeshLaneStatus lanes[SPARK_WEIGHTD_MESH_MAX_LANES];
    uint8_t reserved_tail[2104];
} SparkWeightdIpcMeshStatusResult;

#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdIpcMeshStatus) == 32u &&
    sizeof(SparkWeightdMeshPeerStatus) == 64u &&
    sizeof(SparkWeightdMeshLaneStatus) == 48u &&
    sizeof(SparkWeightdIpcMeshStatusResult) == 4096u &&
    sizeof(SparkWeightdIpcMeshStatusResult) <= SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX,"mesh status frame sizes are fixed for layout 1");
_Static_assert(offsetof(SparkWeightdIpcMeshStatusResult,status) == 24u &&
    offsetof(SparkWeightdIpcMeshStatusResult,layout) == 28u &&
    offsetof(SparkWeightdIpcMeshStatusResult,layout_compat) == 32u &&
    offsetof(SparkWeightdIpcMeshStatusResult,mesh_state) == 36u &&
    offsetof(SparkWeightdIpcMeshStatusResult,pid) == 40u &&
    offsetof(SparkWeightdIpcMeshStatusResult,daemon_generation) == 64u &&
    offsetof(SparkWeightdIpcMeshStatusResult,boot_ns) == 72u &&
    offsetof(SparkWeightdIpcMeshStatusResult,mesh_generation) == 80u &&
    offsetof(SparkWeightdIpcMeshStatusResult,counters) == 104u &&
    offsetof(SparkWeightdIpcMeshStatusResult,peers) == 200u &&
    offsetof(SparkWeightdIpcMeshStatusResult,lanes) == 1224u &&
    offsetof(SparkWeightdIpcMeshStatusResult,reserved_tail) == 1992u,"mesh status layout 1 offsets are fixed");
#endif

typedef struct SparkWeightdIpcEnsure
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint32_t layer;
    uint32_t expert;
} SparkWeightdIpcEnsure;

typedef struct SparkWeightdIpcEnsureResult
{
    SparkWeightdIpcHeader header;
    uint64_t arena_generation;
    uint64_t device_offset;
    uint64_t expert_bytes;
    uint64_t resident_bytes;
    uint64_t load_ns;
    uint32_t status;
    uint32_t loaded;
} SparkWeightdIpcEnsureResult;

typedef struct SparkWeightdIpcAcquire
{
	SparkWeightdIpcHeader header;
	uint64_t arena_generation;
	uint32_t count;
	uint32_t reserved0;
	SparkWeightdExpertKey keys[SPARK_WEIGHTD_LEASE_GROUPS_MAX];
} SparkWeightdIpcAcquire;

typedef struct SparkWeightdIpcAcquireResult
{
	SparkWeightdIpcHeader header;
	uint64_t arena_generation;
	uint64_t lease_identifier;
	uint64_t resident_bytes;
	uint32_t status;
	uint32_t reserved0;
} SparkWeightdIpcAcquireResult;

typedef struct SparkWeightdIpcRelease
{
	SparkWeightdIpcHeader header;
	uint64_t arena_generation;
	uint64_t lease_identifier;
} SparkWeightdIpcRelease;

typedef SparkWeightdIpcAcquireResult SparkWeightdIpcReleaseResult;

typedef struct SparkWeightdIpcExportLease
{
	SparkWeightdIpcHeader header;
	uint64_t arena_generation;
	uint64_t lease_identifier;
	uint32_t batch_offset;
	uint32_t reserved0;
} SparkWeightdIpcExportLease;

typedef struct SparkWeightdIpcExportLeaseResult
{
	SparkWeightdIpcExportResult base;
	uint64_t lease_identifier;
	uint32_t lease_chunk_count;
	uint32_t reserved1;
	uint32_t chunk_indices[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
} SparkWeightdIpcExportLeaseResult;

#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdIpcAcquire) <= SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX,"working set request exceeds IPC frame");
#endif

typedef struct SparkWeightdIpcKvPoolAttach
{
    SparkWeightdIpcHeader header;
    uint8_t key[SPARK_WEIGHTD_KV_POOL_KEY_BYTES];
    uint64_t device_bytes;
    uint64_t minimum_bytes;
    uint64_t chunk_bytes;
    uint64_t metadata_bytes;
    char label[SPARK_WEIGHTD_KV_POOL_LABEL_BYTES];
} SparkWeightdIpcKvPoolAttach;

typedef struct SparkWeightdIpcKvPoolAttachResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t reattached;
    uint64_t pool_generation;
    uint64_t chunk_bytes;
    uint32_t chunk_capacity;
    uint32_t chunk_count;
    uint32_t metadata_fd_count;
    uint32_t reserved0;
    uint64_t device_bytes;
    uint64_t metadata_bytes;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
    uint64_t legacy_write_budget_bytes_per_day;
} SparkWeightdIpcKvPoolAttachResult;

typedef struct SparkWeightdIpcKvPoolResize
{
    SparkWeightdIpcHeader header;
    uint64_t pool_generation;
    uint32_t target_chunks;
    uint32_t reserved0;
} SparkWeightdIpcKvPoolResize;

typedef struct SparkWeightdIpcKvPoolResizeResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t chunk_count;
    uint32_t wanted_chunks;
    uint32_t reserved0;
    uint64_t kv_committed_bytes;
} SparkWeightdIpcKvPoolResizeResult;

typedef struct SparkWeightdIpcKvPoolExport
{
    SparkWeightdIpcHeader header;
    uint64_t pool_generation;
    uint32_t first_chunk;
    uint32_t chunk_count;
} SparkWeightdIpcKvPoolExport;

typedef struct SparkWeightdIpcKvPoolExportResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t first_chunk;
    uint32_t chunk_count;
    uint32_t reserved0;
} SparkWeightdIpcKvPoolExportResult;

typedef struct SparkWeightdIpcKvPoolStatus
{
    SparkWeightdIpcHeader header;
    uint64_t pool_generation;
} SparkWeightdIpcKvPoolStatus;

typedef struct SparkWeightdIpcKvPoolStatusResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t chunk_count;
    uint32_t chunk_capacity;
    uint32_t wanted_chunks;
    uint64_t reclaim_wanted_bytes;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
} SparkWeightdIpcKvPoolStatusResult;

typedef struct SparkWeightdIpcKvSharedAttach
{
    SparkWeightdIpcHeader header;
    uint8_t key[SPARK_WEIGHTD_KV_POOL_KEY_BYTES];
    uint8_t layout_sha256[32];
    uint64_t chunk_bytes;
    uint64_t page_bytes;
    uint32_t alignment_pages;
    uint32_t reserved0;
    char label[SPARK_WEIGHTD_KV_POOL_LABEL_BYTES];
} SparkWeightdIpcKvSharedAttach;

typedef struct SparkWeightdIpcKvSharedAttachResult
{
    SparkWeightdIpcHeader header;
    uint32_t status;
    uint32_t holder;
    uint64_t pool_generation;
    uint64_t chunk_bytes;
    uint64_t device_bytes;
    uint64_t metadata_bytes;
    uint32_t chunk_count;
    uint32_t metadata_fd_count;
    uint32_t created;
    uint32_t slot_count;
} SparkWeightdIpcKvSharedAttachResult;

#if !defined(__cplusplus)
_Static_assert(sizeof(SparkWeightdIpcKvPoolAttach) == 152u && sizeof(SparkWeightdIpcKvPoolAttachResult) == 104u &&
    sizeof(SparkWeightdIpcKvPoolResize) == 40u && sizeof(SparkWeightdIpcKvPoolResizeResult) == 48u &&
    sizeof(SparkWeightdIpcKvPoolExport) == 40u && sizeof(SparkWeightdIpcKvPoolExportResult) == 40u &&
    sizeof(SparkWeightdIpcKvPoolStatus) == 32u && sizeof(SparkWeightdIpcKvPoolStatusResult) == 64u,"kv pool frames are fixed for ABI 11");
#endif

#define SPARK_WEIGHTD_IPC_HEADER_BYTES ((uint32_t)sizeof(SparkWeightdIpcHeader))
#define SPARK_WEIGHTD_IPC_HELLO_BYTES ((uint32_t)sizeof(SparkWeightdIpcHello))
#define SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES ((uint32_t)sizeof(SparkWeightdIpcHelloAck))
#define SPARK_WEIGHTD_IPC_ATTACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttach))
#define SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttachResult))
#define SPARK_WEIGHTD_IPC_DETACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetach))
#define SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetachResult))
#define SPARK_WEIGHTD_IPC_RECLAIM_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaim))
#define SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaimResult))
#define SPARK_WEIGHTD_IPC_RECLAIM_PACK_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaimPack))
#define SPARK_WEIGHTD_IPC_EXPORT_BYTES ((uint32_t)sizeof(SparkWeightdIpcExport))
#define SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcExportResult))
#define SPARK_WEIGHTD_IPC_ATTACH_LAZY_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttachLazy))
#define SPARK_WEIGHTD_IPC_ATTACH_LAZY_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttachLazyResult))
#define SPARK_WEIGHTD_IPC_ENSURE_BYTES ((uint32_t)sizeof(SparkWeightdIpcEnsure))
#define SPARK_WEIGHTD_IPC_ENSURE_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcEnsureResult))

SparkStatus SparkWeightdIdentityPrepare(SparkWeightdIdentity *identity);

bool SparkWeightdIdentityEqual(const SparkWeightdIdentity *left,
    const SparkWeightdIdentity *right);

SparkStatus SparkWeightdIpcValidateHeader(const SparkWeightdIpcHeader *header,
    uint32_t message_bytes,
    uint32_t expected_kind);

SparkStatus SparkWeightdIpcValidateHeaderVersion(const SparkWeightdIpcHeader *header,
    uint32_t message_bytes,
    uint32_t expected_kind,
    uint32_t abi_version);

uint32_t SparkWeightdIpcAbiServed(uint32_t abi_version,uint32_t kind);


typedef struct SparkWeightdServerConfig
{
    const char *socket_path;
    uint64_t device_bytes_max;
    uint64_t kv_reserve_bytes;
    uint64_t load_pace_bytes_per_second;
    uint64_t kv_shared_window_bytes;
} SparkWeightdServerConfig;

typedef struct SparkWeightdServer SparkWeightdServer;

SparkStatus SparkWeightdServerCreate(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server);
SparkStatus SparkWeightdServerCreateUnbound(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server);
SparkStatus SparkWeightdServerListen(SparkWeightdServer *server);
SparkStatus SparkWeightdMeshStatusFill(SparkWeightdIpcMeshStatusResult *result);
void SparkWeightdMeshStop(void);

SparkStatus SparkWeightdServerStep(SparkWeightdServer *server);

SparkStatus SparkWeightdServerRun(SparkWeightdServer *server,
    const volatile sig_atomic_t *stop);

void SparkWeightdServerDestroy(SparkWeightdServer *server);

uint32_t SparkWeightdServerArenaCount(const SparkWeightdServer *server);
uint64_t SparkWeightdServerResidentBytes(const SparkWeightdServer *server);


typedef struct SparkWeightdClient SparkWeightdClient;

typedef struct SparkWeightdHelloResult
{
    SparkStatus status;
    uint64_t daemon_generation;
    uint64_t resident_bytes;
    uint64_t device_bytes_max;
    uint32_t arena_count;
} SparkWeightdHelloResult;

typedef struct SparkWeightdAttachRequest
{
    SparkWeightdIdentity identity;
    char pack_path[SPARK_WEIGHTD_PATH_BYTES];
} SparkWeightdAttachRequest;

typedef struct SparkWeightdAttachResult
{
    SparkStatus status;
    uint64_t arena_generation;
    uint64_t device_handle;
    uint64_t arena_bytes;
    uint64_t resident_bytes;
    uint32_t refcount;
    uint32_t arena_count;
    uint32_t loaded_from_pack;
} SparkWeightdAttachResult;

typedef struct SparkWeightdDetachResult
{
    SparkStatus status;
    uint64_t resident_bytes;
    uint32_t refcount;
    uint32_t arena_count;
} SparkWeightdDetachResult;

typedef struct SparkWeightdReclaimResult
{
    SparkStatus status;
    uint64_t reclaimed_bytes;
    uint64_t resident_bytes;
    uint32_t reclaimed_arena_count;
    uint32_t arena_count;
    uint32_t busy_arena_count;
} SparkWeightdReclaimResult;

SparkStatus SparkWeightdManifestIdentity(const SparkWeightdManifest *manifest,uint8_t digest[32]);

typedef struct SparkWeightdLazyAttachRequest
{
    SparkWeightdIdentity identity;
    char pack_path[SPARK_WEIGHTD_PATH_BYTES];
    uint64_t expert_pool_bytes;
} SparkWeightdLazyAttachRequest;

typedef struct SparkWeightdLazyAttachResult
{
    SparkStatus status;
    uint64_t arena_generation;
    uint64_t device_handle;
    uint64_t arena_bytes;
    uint64_t resident_bytes;
    uint64_t expert_pool_bytes;
    uint32_t refcount;
    uint32_t arena_count;
    uint32_t expert_count;
    uint32_t loaded_from_pack;
    uint32_t mesh_ready;
    uint64_t mesh_send_buffer_addr;
    uint64_t mesh_send_buffer_bytes;
    int pool_fd;
    void *mesh_mapping;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint8_t manifest_sha256[32];
} SparkWeightdLazyAttachResult;

typedef struct SparkWeightdEnsureResult
{
    SparkStatus status;
    uint64_t arena_generation;
    uint64_t device_offset;
    uint64_t expert_bytes;
    uint64_t resident_bytes;
    uint64_t load_ns;
    uint32_t loaded;
} SparkWeightdEnsureResult;

SparkStatus SparkWeightdClientConnect(const char *socket_path,
    SparkWeightdClient **client,
    SparkWeightdHelloResult *hello_out);
SparkStatus SparkWeightdClientConnectWithin(const char *socket_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdClient **client,
    SparkWeightdHelloResult *hello_out);
SparkStatus SparkWeightdClientMeshStatus(SparkWeightdClient *client,
    SparkWeightdIpcMeshStatusResult *result,
    uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdMeshStatusQuery(const char *socket_path,
    uint64_t timeout_nanoseconds,
    SparkWeightdIpcMeshStatusResult *result,
    uint32_t *outcome);

void SparkWeightdClientClose(SparkWeightdClient *client);
uint32_t SparkWeightdClientAlive(const SparkWeightdClient *client);

SparkStatus SparkWeightdClientMeshActivity(SparkWeightdClient *client,
    uint64_t generation, uint32_t active, uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientMeshWrite(SparkWeightdClient *client,
    uint32_t peer_rank,
    uint64_t source_offset,
    uint64_t remote_offset,
    uint32_t length,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientMeshBroadcast(SparkWeightdClient *client,
    uint32_t peer_mask,
    uint64_t source_offset,
    uint64_t remote_offset,
    uint32_t length,
    uint64_t seq_value,
    uint64_t seq_remote_offset,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientAttach(SparkWeightdClient *client,
    const SparkWeightdAttachRequest *request,
    SparkWeightdAttachResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientAttachLazy(SparkWeightdClient *client,
    const SparkWeightdLazyAttachRequest *request,
    SparkWeightdLazyAttachResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientEnsure(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t layer,
    uint32_t expert,
    SparkWeightdEnsureResult *result,
    uint64_t timeout_nanoseconds);

typedef struct SparkWeightdResidency
{
    uint64_t present_bytes;
    uint64_t epoch;
    uint32_t group_count;
    uint32_t present_count;
    uint32_t fixed_pool;
} SparkWeightdResidency;

SparkStatus SparkWeightdClientResidency(SparkWeightdClient *client,
    uint64_t arena_generation,
    SparkWeightdResidency *residency,
    uint64_t timeout_nanoseconds);

typedef struct SparkWeightdWorkingSetResult
{
	SparkStatus status;
	uint64_t arena_generation;
	uint64_t lease_identifier;
	uint64_t resident_bytes;
} SparkWeightdWorkingSetResult;

SparkStatus SparkWeightdClientAcquire(SparkWeightdClient *client,uint64_t arena_generation,const SparkWeightdExpertKey *keys,uint32_t count,SparkWeightdWorkingSetResult *result,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdClientRelease(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,SparkWeightdWorkingSetResult *result,uint64_t timeout_nanoseconds);

typedef struct SparkWeightdExportBatch
{
    SparkStatus status;
    uint64_t arena_generation;
    uint64_t chunk_bytes;
    uint32_t chunk_count;
    uint32_t batch_offset;
    uint32_t batch_count;
    uint32_t reserved0;
    int fds[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
    uint64_t lease_identifier;
    uint32_t lease_chunk_count;
    uint32_t chunk_indices[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
} SparkWeightdExportBatch;

SparkStatus SparkWeightdClientExportLeaseBatch(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,uint32_t batch_offset,SparkWeightdExportBatch *batch,uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientExportBatch(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t batch_offset,
    SparkWeightdExportBatch *batch,
    uint64_t timeout_nanoseconds);

typedef struct SparkWeightdSlotBatch
{
    SparkStatus status;
    uint64_t chunk_bytes;
    uint32_t slot_count;
    uint32_t slot_offset;
    uint32_t batch_count;
    int fds[SPARK_WEIGHTD_EXPORT_BATCH_MAX];
} SparkWeightdSlotBatch;

SparkStatus SparkWeightdClientSlotExportBatch(SparkWeightdClient *client,uint64_t arena_generation,uint32_t slot_offset,SparkWeightdSlotBatch *batch,uint64_t timeout_nanoseconds);

typedef struct SparkWeightdLeaseSlotBatch
{
    SparkStatus status;
    uint32_t lease_chunk_count;
    uint32_t batch_offset;
    uint32_t batch_count;
    uint32_t chunk_indices[SPARK_WEIGHTD_LEASE_SLOTS_BATCH_MAX];
    uint32_t slot_indices[SPARK_WEIGHTD_LEASE_SLOTS_BATCH_MAX];
} SparkWeightdLeaseSlotBatch;

SparkStatus SparkWeightdClientLeaseSlotsBatch(SparkWeightdClient *client,uint64_t arena_generation,uint64_t lease_identifier,uint32_t batch_offset,SparkWeightdLeaseSlotBatch *batch,uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientDetach(SparkWeightdClient *client,
    uint64_t arena_generation,
    SparkWeightdDetachResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientEpochExport(SparkWeightdClient *client,
    uint64_t arena_generation,
    int *fd_out,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdMeshLaneConfigure(uint32_t lane,
    const SparkWeightdMeshTopology *topology);
SparkStatus SparkWeightdMeshSetActivity(uint32_t lane,uint32_t active);

SparkStatus SparkWeightdClientMeshMap(SparkWeightdClient *client,
    void **mapping,uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientMeshStagingMap(SparkWeightdClient *client,
    void **mapping,uint64_t timeout_nanoseconds);

typedef struct SparkWeightdKvPoolRequest
{
    uint8_t key[SPARK_WEIGHTD_KV_POOL_KEY_BYTES];
    uint64_t device_bytes;
    uint64_t minimum_bytes;
    uint64_t chunk_bytes;
    uint64_t metadata_bytes;
    const char *label;
    void *device_base;
    uint64_t reservation_bytes;
} SparkWeightdKvPoolRequest;

typedef struct SparkWeightdKvPoolGrant
{
    uint64_t pool_generation;
    uint64_t chunk_bytes;
    uint64_t device_bytes;
    uint64_t metadata_bytes;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
    uint32_t chunk_capacity;
    uint32_t chunk_count;
    uint32_t reattached;
    int metadata_fd;
} SparkWeightdKvPoolGrant;

typedef struct SparkWeightdKvSharedRequest
{
    uint8_t key[SPARK_WEIGHTD_KV_POOL_KEY_BYTES];
    uint8_t layout_sha256[32];
    uint64_t chunk_bytes;
    uint64_t page_bytes;
    uint32_t alignment_pages;
    const char *label;
} SparkWeightdKvSharedRequest;

typedef struct SparkWeightdKvSharedGrant
{
    uint64_t pool_generation;
    uint64_t chunk_bytes;
    uint64_t device_bytes;
    uint64_t metadata_bytes;
    uint32_t chunk_count;
    uint32_t holder;
    uint32_t created;
    uint32_t slot_count;
    int metadata_fd;
} SparkWeightdKvSharedGrant;

typedef struct SparkWeightdKvPoolState
{
    uint32_t chunk_count;
    uint32_t chunk_capacity;
    uint32_t wanted_chunks;
    uint32_t reserved0;
    uint64_t reclaim_wanted_bytes;
    uint64_t kv_reserve_bytes;
    uint64_t kv_committed_bytes;
} SparkWeightdKvPoolState;

SparkStatus SparkWeightdClientKvPoolAttach(SparkWeightdClient *client,
    const SparkWeightdKvPoolRequest *request,SparkWeightdKvPoolGrant *grant,
    uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdClientKvPoolExport(SparkWeightdClient *client,uint64_t pool_generation,
    uint32_t first_chunk,uint32_t chunk_count,int *fds,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdClientKvPoolResize(SparkWeightdClient *client,uint64_t pool_generation,
    uint32_t target_chunks,SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdClientKvPoolStatus(SparkWeightdClient *client,uint64_t pool_generation,
    SparkWeightdKvPoolState *state,uint64_t timeout_nanoseconds);
SparkStatus SparkWeightdClientKvSharedAttach(SparkWeightdClient *client,
    const SparkWeightdKvSharedRequest *request,SparkWeightdKvSharedGrant *grant,
    uint64_t timeout_nanoseconds);
uint64_t SparkWeightdServerKvCommittedBytes(const SparkWeightdServer *server);
uint32_t SparkWeightdServerKvPoolCount(const SparkWeightdServer *server);

SparkStatus SparkWeightdClientLaneAcquire(SparkWeightdClient *client,
    uint32_t requested_lane,
    const SparkWeightdMeshTopology *topology,
    uint32_t *lane_out,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientLaneBind(SparkWeightdClient *owner,
    SparkWeightdClient *peer,uint32_t band,
    const SparkWeightdMeshTopology *topology,uint32_t *lane_out);
SparkStatus SparkWeightdClientLaneUnbind(SparkWeightdClient *owner,uint32_t band);

SparkStatus SparkWeightdClientEvict(SparkWeightdClient *client,
    uint32_t target_lane,
    uint32_t *released_leases_out,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *client,
    SparkWeightdReclaimResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientAttachLazyShared(SparkWeightdClient *client,
    const SparkWeightdLazyAttachRequest *request,
    SparkWeightdLazyAttachResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdShareModeFromEnvironment(uint32_t *read_only);

SparkStatus SparkWeightdClientReclaimPack(SparkWeightdClient *client,
    const char *pack_sha256,
    SparkWeightdReclaimResult *result,
    uint64_t timeout_nanoseconds);

void SparkWeightdClientClose(SparkWeightdClient *client);


#ifdef __cplusplus
}
#endif
