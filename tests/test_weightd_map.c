/* W3 weightd lane (docs/WEIGHTD_DESIGN.md) fd-tier host tests — cuda-stub
 * only, no GPU. Proves the POSIX-fd export + consumer import/map tier:
 *
 * 1. EXPORT GATES: the EXPORT exchange is answered only for an arena the
 *    connection holds an attach reference for (NOT_FOUND otherwise), a bad
 *    batch offset is INVALID_ARGUMENT, export before HELLO fails the
 *    connection, and a detached arena is no longer exportable.
 * 2. EXPORT BATCH: the reply reveals the arena's chunk geometry (2 MiB
 *    chunk law, chunk_count covering the arena) and delivers REAL received
 *    fds — live, close-on-exec, owner-only mode (the 0600 discipline).
 * 3. IMPORT MAP: the attach helper maps the imported chunks at the
 *    CONSUMER's own span (RW via cuMemSetAccess); the bytes read back
 *    through the consumer-local mapping equal the pack bytes — the
 *    device_handle stopgap replaced — and Release unmaps + releases + frees
 *    exactly (the stub's leak ledger and driver orderings enforce it)
 *    WITHOUT detaching: the arena stays warm and a fresh attach+map after
 *    every consumer left is STILL warm (same generation).
 * 4. IDENTITY CHECK: a chunk set that does not cover the caller's expected
 *    byte range is refused ("import_short") with the attach released and
 *    nothing mapped.
 * 5. MULTI-BATCH: a 65-chunk arena arrives as two SCM_RIGHTS batches
 *    (64 + 1) and the mapped span still reads back byte-exact (SHA-256 over
 *    the map equals the pack digest).
 * 6. CROSS-PROCESS: a real forked consumer process attaches warm, imports,
 *    maps at ITS OWN VA, and reads the pack bytes through the imported
 *    mapping — fds genuinely cross the process boundary. */

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd.h"

#define SPARK_TEST_CHUNK_BYTES (2ull * 1024ull * 1024ull) /* the stub's law */
#define SPARK_TEST_SMALL_BYTES (256ull * 1024ull)
#define SPARK_TEST_TWO_CHUNK_BYTES (2ull * SPARK_TEST_CHUNK_BYTES)
#define SPARK_TEST_BATCH_ARENA_BYTES (65ull * SPARK_TEST_CHUNK_BYTES)
#define SPARK_TEST_TIMEOUT_NS 10000000000ull

static const char *SPARK_TEST_SOCKET = "/tmp/spark_weightd_map_test.sock";
static const char *SPARK_TEST_PACK = "/tmp/spark_weightd_map_test.spack";
static const char *SPARK_TEST_MODEL = "weightd-map-test";

void spark_stub_cuda_reset_faults(void);
uint32_t spark_stub_cuda_outstanding_allocs(void);

typedef struct SparkTestServerThread
{
    SparkWeightdServer *server;
    volatile sig_atomic_t stop;
    SparkStatus run_status;
} SparkTestServerThread;

static void *SparkTestServerThreadMain(void *raw_context)
{
    SparkTestServerThread *context = (SparkTestServerThread *)raw_context;
    context->run_status = SparkWeightdServerRun(context->server, &context->stop);
    return 0;
}

static void SparkTestStartServer(SparkTestServerThread *thread_context,
    pthread_t *thread_handle,
    const char *socket_path,
    uint64_t ceiling_bytes)
{
    SparkWeightdServerConfig config;
    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = ceiling_bytes;
    assert(SparkWeightdServerCreate(&config, &thread_context->server) ==
        SPARK_STATUS_OK);
    thread_context->stop = 0;
    thread_context->run_status = SPARK_STATUS_OK;
    assert(pthread_create(thread_handle, 0, SparkTestServerThreadMain,
        thread_context) == 0);
}

static void SparkTestStopServer(SparkTestServerThread *thread_context,
    pthread_t thread_handle)
{
    __atomic_store_n(&thread_context->stop, 1, __ATOMIC_SEQ_CST);
    assert(pthread_join(thread_handle, 0) == 0);
    assert(thread_context->run_status == SPARK_STATUS_OK);
    SparkWeightdServerDestroy(thread_context->server);
    thread_context->server = 0;
    if (spark_stub_cuda_outstanding_allocs() != 0u)
    {
        fprintf(stderr, "outstanding stub allocs=%u\n",
            spark_stub_cuda_outstanding_allocs());
    }
    assert(spark_stub_cuda_outstanding_allocs() == 0u); /* nothing leaked */
}

static uint32_t SparkTestNextRandom(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void SparkTestWritePack(const char *path, uint32_t seed,
    uint64_t bytes, char hex[SPARK_SHA256_HEX_BYTES])
{
    static uint8_t chunk[65536];
    FILE *file = fopen(path, "wb");
    uint64_t written = 0ull;
    uint32_t state = seed;
    assert(file != 0);
    while (written < bytes)
    {
        uint64_t remaining = bytes - written;
        size_t chunk_bytes = remaining < sizeof(chunk) ? (size_t)remaining
            : sizeof(chunk);
        size_t index;
        for (index = 0; index < chunk_bytes; index++)
        {
            if (index % 4u == 0u)
            {
                state = SparkTestNextRandom(&state);
            }
            chunk[index] = (uint8_t)(state >> ((index % 4u) * 8u));
        }
        assert(fwrite(chunk, 1u, chunk_bytes, file) == chunk_bytes);
        written += (uint64_t)chunk_bytes;
    }
    assert(fclose(file) == 0);
    assert(SparkSha256File(path, hex) == SPARK_STATUS_OK);
}

static void SparkTestSetEnv(const char *name, const char *value)
{
    if (value == 0)
    {
        assert(unsetenv(name) == 0);
    }
    else
    {
        assert(setenv(name, value, 1) == 0);
    }
}

static void SparkTestClearAttachEnv(void)
{
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_MODEL, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_REVISION, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
}

static void SparkTestMakeSlice(SparkWeightdPackSlice *slice,
    uint64_t pack_bytes)
{
    memset(slice, 0, sizeof(*slice));
    slice->model = SPARK_TEST_MODEL;
    slice->revision = "w3-map-rev";
    slice->topology = 4u;
    slice->geometry_fingerprint = 0x3FULL;
    slice->pack_bytes = pack_bytes;
}

/* Attach + import/map through the production helper path (env-published
 * identity, like every real consumer). */
static void SparkTestAttachAndMap(SparkWeightdAttachOutcome *outcome,
    const char *socket_path,
    const char *pack_path,
    const char *pack_sha256,
    uint64_t expected_bytes)
{
    SparkWeightdPackSlice slice;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    SparkTestMakeSlice(&slice, expected_bytes);
    SparkTestClearAttachEnv();
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, socket_path);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, pack_sha256);
    memset(outcome, 0, sizeof(*outcome));
    assert(SparkWeightdAttachPack(&slice, pack_path, SPARK_TEST_TIMEOUT_NS,
        outcome, reason) == SPARK_STATUS_OK);
    if (outcome->client == 0)
    {
        fprintf(stderr, "attach fell back reason=%s\n", reason);
    }
    assert(outcome->client != 0);
    assert(reason[0] == '\0');
    assert(SparkWeightdAttachImportMap(outcome, expected_bytes,
        SPARK_TEST_TIMEOUT_NS, reason) == SPARK_STATUS_OK);
    assert(outcome->map_base != 0);
    assert(outcome->device_handle == (uint64_t)(uintptr_t)outcome->map_base);
    assert(reason[0] == '\0');
}

/* ------------------------------ 1 + 2. export gates and batch shape ------------------------------ */

static void SparkTestExportGatesAndBatch(void)
{
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdAttachResult attach;
    SparkWeightdDetachResult detach;
    SparkWeightdExportBatch batch;
    SparkWeightdClient *client = 0;
    SparkWeightdClient *stranger = 0;
    char digest[SPARK_SHA256_HEX_BYTES];
    uint64_t generation;
    struct stat fd_status;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 701u, SPARK_TEST_SMALL_BYTES, digest);
    SparkTestStartServer(&thread_context, &thread_handle, SPARK_TEST_SOCKET,
        4ull * SPARK_TEST_CHUNK_BYTES);

    assert(SparkWeightdClientConnect(SPARK_TEST_SOCKET, &client, 0) ==
        SPARK_STATUS_OK);
    {
        SparkWeightdAttachRequest request;
        SparkWeightdIdentity identity;
        memset(&identity, 0, sizeof(identity));
        memcpy(identity.model, SPARK_TEST_MODEL, strlen(SPARK_TEST_MODEL) + 1u);
        memcpy(identity.revision, "w3-map-rev", 11u);
        identity.topology = 4u;
        identity.geometry_fingerprint = 0x3FULL;
        identity.arena_bytes = SPARK_TEST_SMALL_BYTES;
        identity.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
        memcpy(identity.pack_sha256, digest, SPARK_SHA256_HEX_BYTES);
        assert(SparkWeightdIdentityPrepare(&identity) == SPARK_STATUS_OK);
        memset(&request, 0, sizeof(request));
        request.identity = identity;
        memcpy(request.pack_path, SPARK_TEST_PACK,
            strlen(SPARK_TEST_PACK) + 1u);
        memset(&attach, 0, sizeof(attach));
        assert(SparkWeightdClientAttach(client, &request, &attach,
            SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        assert(attach.status == SPARK_STATUS_OK);
    }
    generation = attach.arena_generation;

    /* batch 0: geometry + REAL fds with the OS scoping discipline */
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_OK);
    assert(batch.chunk_bytes == SPARK_TEST_CHUNK_BYTES); /* the 2 MiB law */
    assert(batch.chunk_count == 1u); /* a 256 KiB arena, one chunk */
    assert(batch.batch_offset == 0u);
    assert(batch.batch_count == 1u);
    assert(fcntl(batch.fds[0], F_GETFD) >= 0);
    assert(fcntl(batch.fds[0], F_GETFD) & FD_CLOEXEC);
    assert(fstat(batch.fds[0], &fd_status) == 0);
    assert((fd_status.st_mode & 077) == 0); /* owner-only, the 0600 law */
    assert(close(batch.fds[0]) == 0);

    /* a second export of the same chunk is legal (fresh shareable fd) */
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 1u);
    assert(close(batch.fds[0]) == 0);

    /* offset beyond the chunk count: refused, no fds */
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 1u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_INVALID_ARGUMENT);
    assert(batch.batch_count == 0u);

    /* an unknown generation: NOT_FOUND, no fds */
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation + 0x5A5Aull, 0u,
        &batch, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_NOT_FOUND);
    assert(batch.batch_count == 0u);

    /* the export capability is the attach ref: a DIFFERENT connection that
     * never attached cannot fish chunks out of the daemon */
    assert(SparkWeightdClientConnect(SPARK_TEST_SOCKET, &stranger, 0) ==
        SPARK_STATUS_OK);
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(stranger, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_NOT_FOUND);
    assert(batch.batch_count == 0u);
    SparkWeightdClientClose(stranger);
    stranger = 0;

    /* detach: the arena is no longer exportable through this connection */
    assert(SparkWeightdClientDetach(client, generation, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_NOT_FOUND);
    assert(batch.batch_count == 0u);
    SparkWeightdClientClose(client);
    client = 0;

    /* EXPORT before HELLO: the connection dies lockstep-clean (raw socket) */
    {
        struct sockaddr_un address;
        uint8_t frame[sizeof(SparkWeightdIpcExport)];
        SparkWeightdIpcExport *export = (SparkWeightdIpcExport *)frame;
        char buffer[16];
        struct pollfd poll_fd;
        int fd = socket(AF_UNIX, SOCK_STREAM, 0);
        assert(fd >= 0);
        memset(&address, 0, sizeof(address));
        address.sun_family = AF_UNIX;
        memcpy(address.sun_path, SPARK_TEST_SOCKET,
            strlen(SPARK_TEST_SOCKET) + 1u);
        assert(connect(fd, (const struct sockaddr *)&address,
            sizeof(address)) == 0);
        memset(frame, 0, sizeof(frame));
        export->header.magic = SPARK_WEIGHTD_IPC_MAGIC;
        export->header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
        export->header.kind = SPARK_WEIGHTD_IPC_KIND_EXPORT;
        export->header.body_bytes =
            SPARK_WEIGHTD_IPC_EXPORT_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        export->header.request_id = 1ull;
        assert(write(fd, frame, sizeof(frame)) == (ssize_t)sizeof(frame));
        poll_fd.fd = fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        assert(poll(&poll_fd, 1u, 5000) > 0);
        assert(read(fd, buffer, sizeof(buffer)) == 0); /* EOF, no answer */
        (void)close(fd);
    }

    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: export gates + batch shape green\n");
}

/* ------------------------------ 3 + 4. import map, identity gate, warm ------------------------------ */

static void SparkTestImportMapWarmAndCoverageGate(void)
{
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdAttachOutcome first;
    SparkWeightdAttachOutcome second;
    SparkWeightdAttachOutcome again;
    SparkWeightdAttachOutcome gated;
    static uint8_t pack_image[SPARK_TEST_TWO_CHUNK_BYTES];
    static uint8_t map_probe[SPARK_TEST_TWO_CHUNK_BYTES];
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkWeightdPackSlice slice;
    FILE *pack_file;
    uint64_t generation;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 702u, sizeof(pack_image), digest);
    pack_file = fopen(SPARK_TEST_PACK, "rb");
    assert(pack_file != 0);
    assert(fread(pack_image, 1u, sizeof(pack_image), pack_file) ==
        sizeof(pack_image));
    assert(fclose(pack_file) == 0);
    SparkTestStartServer(&thread_context, &thread_handle, SPARK_TEST_SOCKET,
        8ull * SPARK_TEST_CHUNK_BYTES);

    /* first consumer: cold load + the import map at ITS OWN span */
    SparkTestAttachAndMap(&first, SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
        sizeof(pack_image));
    assert(first.loaded_from_pack == 1u);
    assert(first.map_chunk_bytes == SPARK_TEST_CHUNK_BYTES);
    assert(first.map_chunk_count == 2u); /* 4 MiB arena, 2 MiB chunks */
    assert(first.map_handle_count == 2u && first.map_mapped_count == 2u);
    assert(first.map_span_bytes == sizeof(pack_image));
    memcpy(map_probe, first.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0);
    generation = first.arena_generation;

    /* second consumer, SAME identity: warm hit + its own mapping of the
     * same arena — both consumers read identical bytes */
    SparkTestAttachAndMap(&second, SPARK_TEST_SOCKET, SPARK_TEST_PACK,
        digest, sizeof(pack_image));
    assert(second.loaded_from_pack == 0u);
    assert(second.arena_generation == generation);
    assert(second.map_base != first.map_base); /* distinct consumer VAs ... */
    memcpy(map_probe, second.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0); /* ...
        over the same bytes */

    /* THE IDENTITY CHECK: a caller whose expected byte range exceeds the
     * chunk set is refused before anything is mapped; the helper released
     * the attach (close-only), so the arena stays resident warm */
    SparkTestMakeSlice(&slice, sizeof(pack_image));
    SparkTestClearAttachEnv();
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    memset(&gated, 0, sizeof(gated));
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &gated, reason) == SPARK_STATUS_OK);
    assert(gated.client != 0);
    assert(SparkWeightdAttachImportMap(&gated,
        sizeof(pack_image) + SPARK_TEST_CHUNK_BYTES, SPARK_TEST_TIMEOUT_NS,
        reason) == SPARK_STATUS_OK);
    assert(gated.client == 0 && gated.map_base == 0);
    assert(strcmp(reason, "import_short") == 0);

    /* argument faults stay transport faults (nothing to map / double map) */
    {
        SparkWeightdAttachOutcome unmapped;
        memset(&unmapped, 0, sizeof(unmapped));
        assert(SparkWeightdAttachImportMap(&unmapped, 4096ull,
            SPARK_TEST_TIMEOUT_NS, reason) == SPARK_STATUS_INVALID_ARGUMENT);
        assert(strcmp(reason, "not_attached") == 0);
        assert(SparkWeightdAttachImportMap(&first, sizeof(pack_image),
            SPARK_TEST_TIMEOUT_NS, reason) == SPARK_STATUS_INVALID_ARGUMENT);
        assert(strcmp(reason, "already_mapped") == 0);
    }

    /* both consumers release: unmap + release + free, WITHOUT detaching —
     * the reaped refcounts leave the arena resident warm */
    SparkWeightdAttachRelease(&first);
    assert(first.map_base == 0 && first.map_handles == 0);
    assert(first.map_chunk_count == 0u && first.map_mapped_count == 0u);
    assert(first.client == 0 && first.device_handle == 0ull);
    SparkWeightdAttachRelease(&second);

    /* the code-redeploy leg: a fresh consumer after every consumer left is
     * STILL warm (same arena generation) and maps the same bytes again */
    SparkTestAttachAndMap(&again, SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
        sizeof(pack_image));
    assert(again.loaded_from_pack == 0u);
    assert(again.arena_generation == generation);
    memcpy(map_probe, again.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0);
    SparkWeightdAttachRelease(&again);

    SparkTestStopServer(&thread_context, thread_handle); /* ledger green */
    SparkTestClearAttachEnv();
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: consumer import map + identity gate + warm green\n");
}

/* ------------------------------ 5. multi-batch ------------------------------ */

static void SparkTestMultiBatch(void)
{
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdAttachOutcome outcome;
    static uint8_t probe[65536];
    char digest[SPARK_SHA256_HEX_BYTES];
    char map_hex[SPARK_SHA256_HEX_BYTES];
    SparkSha256Context sha;
    uint64_t offset;

    /* 65 chunks: one batch of 64 + one batch of 1 through the SCM_RIGHTS
     * ancillary, then mapped into ONE consumer span and verified whole */
    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 703u, SPARK_TEST_BATCH_ARENA_BYTES,
        digest);
    SparkTestStartServer(&thread_context, &thread_handle, SPARK_TEST_SOCKET,
        128ull * SPARK_TEST_CHUNK_BYTES);
    SparkTestAttachAndMap(&outcome, SPARK_TEST_SOCKET, SPARK_TEST_PACK,
        digest, SPARK_TEST_BATCH_ARENA_BYTES);
    assert(outcome.loaded_from_pack == 1u);
    assert(outcome.map_chunk_count == 65u);
    assert(outcome.map_handle_count == 65u && outcome.map_mapped_count == 65u);
    assert(outcome.map_span_bytes == SPARK_TEST_BATCH_ARENA_BYTES);

    /* SHA-256 over the consumer's mapping == the pack digest */
    SparkSha256Initialize(&sha);
    for (offset = 0; offset < SPARK_TEST_BATCH_ARENA_BYTES;
        offset += sizeof(probe))
    {
        memcpy(probe, (const uint8_t *)outcome.map_base + offset,
            sizeof(probe));
        SparkSha256Update(&sha, probe, sizeof(probe));
    }
    {
        uint8_t raw[SPARK_SHA256_DIGEST_BYTES];
        SparkSha256Finalize(&sha, raw);
        SparkSha256DigestToHex(raw, map_hex);
    }
    assert(memcmp(map_hex, digest, SPARK_SHA256_HEX_BYTES) == 0);

    SparkWeightdAttachRelease(&outcome);
    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: 65-chunk two-batch import map green\n");
}

/* ------------------------------ 6. cross-process ------------------------------ */

/* The child consumer body: a REAL second process. It attaches warm, runs
 * the import map, and verifies the bytes through ITS OWN mapping against
 * the pack file it reads itself — the fd crossing is genuine SCM_RIGHTS;
 * nothing about the daemon's address space is reachable from here. Never
 * returns: exits 0 on the verified mapping, nonzero otherwise. */
static void SparkTestConsumerChild(const char *socket_path,
    const char *pack_path,
    const char *digest,
    uint64_t pack_bytes,
    int report_fd)
{
    static uint8_t probe[65536];
    SparkWeightdAttachOutcome outcome;
    SparkSha256Context sha;
    char map_hex[SPARK_SHA256_HEX_BYTES];
    uint64_t offset;
    ssize_t written;

    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestAttachAndMap(&outcome, socket_path, pack_path, digest,
        pack_bytes);
    assert(outcome.loaded_from_pack == 0u); /* warm: the arena was resident */
    SparkSha256Initialize(&sha);
    for (offset = 0; offset < pack_bytes; offset += sizeof(probe))
    {
        memcpy(probe, (const uint8_t *)outcome.map_base + offset,
            sizeof(probe));
        SparkSha256Update(&sha, probe, sizeof(probe));
    }
    {
        uint8_t raw[SPARK_SHA256_DIGEST_BYTES];
        SparkSha256Finalize(&sha, raw);
        SparkSha256DigestToHex(raw, map_hex);
    }
    assert(memcmp(map_hex, digest, SPARK_SHA256_HEX_BYTES) == 0);
    SparkWeightdAttachRelease(&outcome);
    do
    {
        written = write(report_fd, "P", 1);
    } while (written < 0 && errno == EINTR);
    assert(written == 1);
    _exit(0);
}

static void SparkTestCrossProcess(void)
{
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdAttachOutcome parent;
    SparkWeightdAttachOutcome after;
    static uint8_t probe[65536];
    char digest[SPARK_SHA256_HEX_BYTES];
    char received;
    int pipe_fds[2];
    pid_t consumer_pid;
    int wait_status = 0;
    pid_t reaped;
    uint64_t generation;

    assert(pipe(pipe_fds) == 0);
    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 704u, SPARK_TEST_SMALL_BYTES, digest);
    SparkTestStartServer(&thread_context, &thread_handle, SPARK_TEST_SOCKET,
        4ull * SPARK_TEST_CHUNK_BYTES);

    /* the parent consumer first: the cold load + its own import map (the
     * arena must be resident for the child's attach to be the WARM fd
     * hand-off the tier exists for) */
    {
        static uint8_t pack_image[SPARK_TEST_SMALL_BYTES];
        FILE *pack_file = fopen(SPARK_TEST_PACK, "rb");
        assert(pack_file != 0);
        assert(fread(pack_image, 1u, sizeof(pack_image), pack_file) ==
            sizeof(pack_image));
        assert(fclose(pack_file) == 0);
        SparkTestAttachAndMap(&parent, SPARK_TEST_SOCKET, SPARK_TEST_PACK,
            digest, SPARK_TEST_SMALL_BYTES);
        assert(parent.loaded_from_pack == 1u);
        memcpy(probe, parent.map_base, sizeof(probe));
        assert(memcmp(probe, pack_image, sizeof(probe)) == 0);
    }
    generation = parent.arena_generation;
    SparkWeightdAttachRelease(&parent);

    consumer_pid = fork();
    assert(consumer_pid >= 0);
    if (consumer_pid == 0)
    {
        (void)close(pipe_fds[0]);
        SparkTestConsumerChild(SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
            SPARK_TEST_SMALL_BYTES, pipe_fds[1]);
        _exit(127); /* not reached */
    }
    (void)close(pipe_fds[1]);
    reaped = waitpid(consumer_pid, &wait_status, 0);
    assert(reaped == consumer_pid);
    assert(WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0);
    assert(read(pipe_fds[0], &received, 1) == 1 && received == 'P');
    (void)close(pipe_fds[0]);

    /* WARM RE-ATTACH AFTER CONSUMER EXIT: the child's release closed its
     * socket; the daemon reaped it, kept the arena resident, and a fresh
     * consumer maps the same generation warm without any reload */
    SparkTestAttachAndMap(&after, SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
        SPARK_TEST_SMALL_BYTES);
    assert(after.loaded_from_pack == 0u);
    assert(after.arena_generation == generation);
    assert(after.map_chunk_count == 1u);
    SparkWeightdAttachRelease(&after);

    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: cross-process import map + warm re-attach green\n");
}

int main(void)
{
    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestExportGatesAndBatch();
    SparkTestImportMapWarmAndCoverageGate();
    SparkTestMultiBatch();
    SparkTestCrossProcess();
    printf("w3 weightd lane: fd export + consumer import/map green\n");
    return 0;
}
