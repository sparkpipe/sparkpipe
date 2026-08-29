#pragma once

/* spark_weightd — the weight-residency daemon (docs/WEIGHTD_DESIGN.md W2a).
 *
 * One spark_weightd per node owns the weight arenas. Consumers attach by
 * CONTENT IDENTITY — (model, revision, topology, pack SHA-256, geometry
 * fingerprint, ABI version) — and never load pack bytes themselves; the
 * daemon loads each identity once, verifies the digest once, and shares one
 * arena across every attached serving process. Family-neutral: the daemon
 * serves every model family through this one identity tuple.
 *
 * W2a is the skeleton: the identity-keyed arena map, the attach/detach
 * IPC surface, refcount sharing, the NO-2x update law (stop-attach-start:
 * cold arenas are reclaimed to make room for a new identity, live ones are
 * never evicted — an attach that cannot fit WITHOUT evicting a live arena
 * fails closed with CAPACITY_EXCEEDED and nothing is allocated), and the
 * clean TERM path. The device ceiling below is the operator's 110 GiB law.
 *
 * Device-memory note (extraction inventory): W2a stands in for the VMM
 * surface with cudaMalloc (device-private). W2b replaces it with
 * cuMemCreate + CU_MEM_HANDLE_TYPE_POSIX_FD export (2 MiB pages) and the
 * consumer-side cuMemImportFromShareableHandle + cuMemMap read-only map;
 * `device_handle` in the attach result is the pid-local stand-in for the
 * exported shareable handle and means nothing across processes until then.
 */

#include <signal.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_WEIGHTD_IPC_ABI_VERSION 1u
#define SPARK_WEIGHTD_IPC_MAGIC UINT32_C(0x57444953) /* "WDIS" */

#define SPARK_WEIGHTD_ID_BYTES 64u
#define SPARK_WEIGHTD_REVISION_BYTES 128u
#define SPARK_WEIGHTD_SHA256_HEX_BYTES 65u
#define SPARK_WEIGHTD_PATH_BYTES 1024u
#define SPARK_WEIGHTD_SOCKET_PATH_BYTES 108u /* sizeof(sockaddr_un::sun_path) */

/* Wire-message bound: every frame below is fixed-size and smaller than
 * this, so the server can buffer a whole request/response per connection. */
#define SPARK_WEIGHTD_IPC_MESSAGE_BYTES_MAX 2048u

/* Skeleton bounds. The per-node multi-topology win (sixteen topologies'
 * packs daemon-managed) lives comfortably inside these; raise only with a
 * budget note — every bound here is per-daemon host memory, not device. */
#define SPARK_WEIGHTD_ARENA_COUNT_MAX 16u
#define SPARK_WEIGHTD_CONNECTION_COUNT_MAX 16u
#define SPARK_WEIGHTD_ATTACHES_PER_CONNECTION_MAX 8u

/* THE DEVICE LAW (docs/AGENT_LANE_BRIEFS/README.md, cluster rules): device
 * allocation on a node stays under 110 GiB of the 119 GiB unified memory —
 * NVRM kills daemons silently past it. The daemon's cold arenas count
 * against the same envelope as everything else on the node, so this is the
 * default ceiling; a deployment that shares the node must LOWER it, never
 * raise it. */
#define SPARK_WEIGHTD_DEVICE_BYTES_MAX_DEFAULT (110ull * 1024ull * 1024ull * 1024ull)

/* Server poll quantum: every wait carries a deadline (the no-TERM-immunity
 * rule — a daemon must always come back around to check its stop flag). */
#define SPARK_WEIGHTD_SERVER_POLL_TIMEOUT_MS 20u

#define SPARK_WEIGHTD_IPC_KIND_HELLO 1u
#define SPARK_WEIGHTD_IPC_KIND_HELLO_ACK 2u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH 3u
#define SPARK_WEIGHTD_IPC_KIND_ATTACH_RESULT 4u
#define SPARK_WEIGHTD_IPC_KIND_DETACH 5u
#define SPARK_WEIGHTD_IPC_KIND_DETACH_RESULT 6u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM 7u
#define SPARK_WEIGHTD_IPC_KIND_RECLAIM_RESULT 8u

/* The content identity. `Prepare` canonicalizes (pads zeroed, strings
 * bounds-checked, digest shape-checked) so equality is plain memcmp over
 * the whole struct — the arena map is content-addressed by this canonical
 * form, never by path, pointer, or arrival order. */
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
    uint32_t loaded_from_pack; /* 1 = cold load, 0 = warm identity hit */
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

#define SPARK_WEIGHTD_IPC_HEADER_BYTES ((uint32_t)sizeof(SparkWeightdIpcHeader))
#define SPARK_WEIGHTD_IPC_HELLO_BYTES ((uint32_t)sizeof(SparkWeightdIpcHello))
#define SPARK_WEIGHTD_IPC_HELLO_ACK_BYTES ((uint32_t)sizeof(SparkWeightdIpcHelloAck))
#define SPARK_WEIGHTD_IPC_ATTACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttach))
#define SPARK_WEIGHTD_IPC_ATTACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcAttachResult))
#define SPARK_WEIGHTD_IPC_DETACH_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetach))
#define SPARK_WEIGHTD_IPC_DETACH_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcDetachResult))
#define SPARK_WEIGHTD_IPC_RECLAIM_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaim))
#define SPARK_WEIGHTD_IPC_RECLAIM_RESULT_BYTES ((uint32_t)sizeof(SparkWeightdIpcReclaimResult))

/* Canonicalize + validate an identity for use as a map key or over the
 * wire: strings must be NUL-terminated inside their arrays (the terminator
 * position is preserved), the digest must be 64 lowercase hex characters,
 * abi_version and arena_bytes must be nonzero; reserved fields and all
 * bytes past the string terminators are forced to zero. */
SparkStatus SparkWeightdIdentityPrepare(SparkWeightdIdentity *identity);

/* Canonical equality: both identities must be prepared. */
bool SparkWeightdIdentityEqual(const SparkWeightdIdentity *left,
    const SparkWeightdIdentity *right);

/* Frame validation shared by the server and the client: magic, ABI, kind,
 * and the exact body size for that kind. */
SparkStatus SparkWeightdIpcValidateHeader(const SparkWeightdIpcHeader *header,
    uint32_t message_bytes,
    uint32_t expected_kind);

/* ------------------------------ server ------------------------------ */

typedef struct SparkWeightdServerConfig
{
    const char *socket_path;
    uint64_t device_bytes_max;
} SparkWeightdServerConfig;

typedef struct SparkWeightdServer SparkWeightdServer;

/* Binds + listens on the unix socket (0600), unlinks any stale socket at
 * that path first. Ignores SIGPIPE (write errors surface as statuses). */
SparkStatus SparkWeightdServerCreate(const SparkWeightdServerConfig *config,
    SparkWeightdServer **server);

/* One non-blocking pass: accept, drain readable connections, answer every
 * complete request. ATTACH misses load synchronously inside this call
 * (W2a has no background loads — the NO-2x budget rules them out; the
 * fleet is dark-briefly per the design). Returns OK even when idle. */
SparkStatus SparkWeightdServerStep(SparkWeightdServer *server);

/* Step loop until *stop becomes nonzero; the poll inside every step is
 * bounded by SPARK_WEIGHTD_SERVER_POLL_TIMEOUT_MS, so TERM lands fast. */
SparkStatus SparkWeightdServerRun(SparkWeightdServer *server,
    const volatile sig_atomic_t *stop);

/* Close every connection (dropping their refcounts), free every arena,
 * unlink the socket. */
void SparkWeightdServerDestroy(SparkWeightdServer *server);

uint32_t SparkWeightdServerArenaCount(const SparkWeightdServer *server);
uint64_t SparkWeightdServerResidentBytes(const SparkWeightdServer *server);

/* ------------------------------ client ------------------------------ */
/* The consumer surface a serving process embeds. Every call carries a
 * deadline; a dead daemon (or any protocol fault) fails the call closed —
 * the client never returns a handle it cannot vouch for, and per the crash
 * contract a consumer must treat all previously imported addresses as
 * invalid the moment its connection dies. */

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

/* Connect + handshake. Fails closed (IO_ERROR) on refused/missing daemon.
 * `hello_out` (nullable) receives the daemon's identity, live residency,
 * and its ceiling. */
SparkStatus SparkWeightdClientConnect(const char *socket_path,
    SparkWeightdClient **client,
    SparkWeightdHelloResult *hello_out);

SparkStatus SparkWeightdClientAttach(SparkWeightdClient *client,
    const SparkWeightdAttachRequest *request,
    SparkWeightdAttachResult *result,
    uint64_t timeout_nanoseconds);

/* Drops THIS connection's reference. The arena itself stays resident
 * (cold) so the next attach — including after a code-only redeploy — is
 * the sub-second warm path; residency is reclaimed only by budget
 * pressure or an explicit SparkWeightdClientReclaim. */
SparkStatus SparkWeightdClientDetach(SparkWeightdClient *client,
    uint64_t arena_generation,
    SparkWeightdDetachResult *result,
    uint64_t timeout_nanoseconds);

/* Drop every cold (refcount == 0) arena now. Maintenance handle. */
SparkStatus SparkWeightdClientReclaim(SparkWeightdClient *client,
    SparkWeightdReclaimResult *result,
    uint64_t timeout_nanoseconds);

/* Closes the connection WITHOUT detaching: the server discovers the dead
 * socket and drops this connection's references — this is the consumer-
 * death path, and the test proves it. */
void SparkWeightdClientClose(SparkWeightdClient *client);

/* Result-status convention for Attach/Detach/Reclaim: the SparkStatus
 * RETURN of the call is the transport verdict (did the exchange complete,
 * did the deadline hold); `result->status` is the daemon's verdict. A
 * daemon-refused attach (HASH_MISMATCH, CAPACITY_EXCEEDED, ...) is a
 * completed exchange: the call returns OK and result->status carries the
 * refusal. */

#ifdef __cplusplus
}
#endif
