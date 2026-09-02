#pragma once


#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_IPC_ABI_VERSION 1u
#define SPARK_WEIGHTD_IPC_MAGIC UINT32_C(0x57444953)

#define SPARK_WEIGHTD_ID_BYTES 64u
#define SPARK_WEIGHTD_REVISION_BYTES 128u
#define SPARK_WEIGHTD_SHA256_HEX_BYTES 65u
#define SPARK_WEIGHTD_PATH_BYTES 1024u
#define SPARK_WEIGHTD_SOCKET_PATH_BYTES 108u

#define SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX 2048u

#define SPARK_WEIGHTD_ARENA_COUNT_MAX 16u
#define SPARK_WEIGHTD_CONNECTION_COUNT_MAX 16u
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

#define SPARK_WEIGHTD_EXPORT_BATCH_MAX 64u
_Static_assert(SPARK_WEIGHTD_EXPORT_BATCH_MAX <= 253u,
    "SPARK_WEIGHTD_EXPORT_BATCH_MAX must stay inside the kernel's "
    "SCM_MAX_FD (253): one EXPORT batch is one message's fd payload");

#define SPARK_WEIGHTD_MAP_CHUNK_COUNT_MAX 65536u

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
    uint32_t reserved0;
} SparkWeightdIpcReclaimResult;

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

#define SPARK_WEIGHTD_IPC_HEADER_BYTES ((uint32_t)sizeof(SparkWeightdIpcHeader))
#define SPARK_WEIGHTD_IPC_HELLO_BYTES ((uint32_t)sizeof(SparkWeightdIpcHello))
#define SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES ((uint32_t)sizeof(SparkWeightdIpcHelloAck))
#define SPARK_WEIGHTD_IPC_ATTACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttach))
#define SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttachResult))
#define SPARK_WEIGHTD_IPC_DETACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetach))
#define SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetachResult))
#define SPARK_WEIGHTD_IPC_RECLAIM_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaim))
#define SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaimResult))
#define SPARK_WEIGHTD_IPC_EXPORT_BYTES ((uint32_t)sizeof(SparkWeightdIpcExport))
#define SPARK_WEIGHTD_IPC_EXPORT_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcExportResult))

SparkStatus SparkWeightdIdentityPrepare(SparkWeightdIdentity *identity);

bool SparkWeightdIdentityEqual(const SparkWeightdIdentity *left,
    const SparkWeightdIdentity *right);

SparkStatus SparkWeightdIpcValidateHeader(const SparkWeightdIpcHeader *header,
    uint32_t message_bytes,
    uint32_t expected_kind);


typedef struct SparkWeightdServerConfig
{
    const char *socket_path;
    uint64_t device_bytes_max;
} SparkWeightdServerConfig;

typedef struct SparkWeightdServer SparkWeightdServer;

SparkStatus SparkWeightdServerCreate(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server);

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
} SparkWeightdReclaimResult;

SparkStatus SparkWeightdClientConnect(const char *socket_path,
    SparkWeightdClient **client,
    SparkWeightdHelloResult *hello_out);

SparkStatus SparkWeightdClientAttach(SparkWeightdClient *client,
    const SparkWeightdAttachRequest *request,
    SparkWeightdAttachResult *result,
    uint64_t timeout_nanoseconds);

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
} SparkWeightdExportBatch;

SparkStatus SparkWeightdClientExportBatch(SparkWeightdClient *client,
    uint64_t arena_generation,
    uint32_t batch_offset,
    SparkWeightdExportBatch *batch,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientDetach(SparkWeightdClient *client,
    uint64_t arena_generation,
    SparkWeightdDetachResult *result,
    uint64_t timeout_nanoseconds);

SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *client,
    SparkWeightdReclaimResult *result,
    uint64_t timeout_nanoseconds);

void SparkWeightdClientClose(SparkWeightdClient *client);


#ifdef __cplusplus
}
#endif
