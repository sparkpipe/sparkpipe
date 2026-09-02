
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
#include "cuda.h"

#define SPARK_TEST_CHUNK_BYTES (2ull * 1024ull * 1024ull)
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
    assert(spark_stub_cuda_outstanding_allocs() == 0u);
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

    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_OK);
    assert(batch.chunk_bytes == SPARK_TEST_CHUNK_BYTES);
    assert(batch.chunk_count == 1u);
    assert(batch.batch_offset == 0u);
    assert(batch.batch_count == 1u);
    assert(fcntl(batch.fds[0], F_GETFD) >= 0);
    assert(fcntl(batch.fds[0], F_GETFD) & FD_CLOEXEC);
    assert(fstat(batch.fds[0], &fd_status) == 0);
    assert((fd_status.st_mode & 077) == 0);
    assert(close(batch.fds[0]) == 0);

    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_OK && batch.batch_count == 1u);
    assert(close(batch.fds[0]) == 0);

    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation, 1u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_INVALID_ARGUMENT);
    assert(batch.batch_count == 0u);

    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(client, generation + 0x5A5Aull, 0u,
        &batch, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_NOT_FOUND);
    assert(batch.batch_count == 0u);

    assert(SparkWeightdClientConnect(SPARK_TEST_SOCKET, &stranger, 0) ==
        SPARK_STATUS_OK);
    memset(&batch, 0, sizeof(batch));
    assert(SparkWeightdClientExportBatch(stranger, generation, 0u, &batch,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(batch.status == SPARK_STATUS_NOT_FOUND);
    assert(batch.batch_count == 0u);
    SparkWeightdClientClose(stranger);
    stranger = 0;

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
        assert(read(fd, buffer, sizeof(buffer)) == 0);
        (void)close(fd);
    }

    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: export gates + batch shape green\n");
}


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

    SparkTestAttachAndMap(&first, SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
        sizeof(pack_image));
    assert(first.loaded_from_pack == 1u);
    assert(first.map_chunk_bytes == SPARK_TEST_CHUNK_BYTES);
    assert(first.map_chunk_count == 2u);
    assert(first.map_handle_count == 2u && first.map_mapped_count == 2u);
    assert(first.map_span_bytes == sizeof(pack_image));
    memcpy(map_probe, first.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0);
    generation = first.arena_generation;

    SparkTestAttachAndMap(&second, SPARK_TEST_SOCKET, SPARK_TEST_PACK,
        digest, sizeof(pack_image));
    assert(second.loaded_from_pack == 0u);
    assert(second.arena_generation == generation);
    assert(second.map_base != first.map_base);
    memcpy(map_probe, second.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0);

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

    SparkWeightdAttachRelease(&first);
    assert(first.map_base == 0 && first.map_handles == 0);
    assert(first.map_chunk_count == 0u && first.map_mapped_count == 0u);
    assert(first.client == 0 && first.device_handle == 0ull);
    SparkWeightdAttachRelease(&second);

    SparkTestAttachAndMap(&again, SPARK_TEST_SOCKET, SPARK_TEST_PACK, digest,
        sizeof(pack_image));
    assert(again.loaded_from_pack == 0u);
    assert(again.arena_generation == generation);
    memcpy(map_probe, again.map_base, sizeof(map_probe));
    assert(memcmp(map_probe, pack_image, sizeof(pack_image)) == 0);
    SparkWeightdAttachRelease(&again);

    SparkTestStopServer(&thread_context, thread_handle);
    SparkTestClearAttachEnv();
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: consumer import map + identity gate + warm green\n");
}


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
    assert(outcome.loaded_from_pack == 0u);
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
        _exit(127);
    }
    (void)close(pipe_fds[1]);
    reaped = waitpid(consumer_pid, &wait_status, 0);
    assert(reaped == consumer_pid);
    assert(WIFEXITED(wait_status) && WEXITSTATUS(wait_status) == 0);
    assert(read(pipe_fds[0], &received, 1) == 1 && received == 'P');
    (void)close(pipe_fds[0]);

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


static void SparkTestScribbleProbeReceipt(void)
{
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdAttachOutcome outcome;
    static uint8_t pack_image[SPARK_TEST_TWO_CHUNK_BYTES];
    static uint8_t scribble[64];
    char digest[SPARK_SHA256_HEX_BYTES];
    CUmemAllocationProp prop;
    CUmemGenericAllocationHandle chunk;
    CUmemAccessDesc access;
    CUdeviceptr span;
    uint32_t index;

    memset(&prop, 0, sizeof(prop));
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = 0;
    for (index = 0u; index < sizeof(scribble); index++)
    {
        scribble[index] = (uint8_t)(index * 7u + 1u);
    }
    assert(cuMemCreate(&chunk, 4096u, &prop, 0ull) == CUDA_SUCCESS);
    assert(cuMemAddressReserve(&span, 4096u, 0u, 0ull, 0ull) == CUDA_SUCCESS);
    assert(cuMemMap(span, 4096u, 0u, chunk, 0ull) == CUDA_SUCCESS);

    assert(cuda_stub_vmm_probe_write(span, scribble, 8u) ==
        CUDA_ERROR_INVALID_VALUE);
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = 0;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READ;
    assert(cuMemSetAccess(span, 4096u, &access, 1u) == CUDA_SUCCESS);
    assert(cuda_stub_vmm_probe_write(span, scribble, sizeof(scribble)) ==
        CUDA_ERROR_INVALID_VALUE);
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    assert(cuMemSetAccess(span, 4096u, &access, 1u) == CUDA_SUCCESS);
    assert(cuda_stub_vmm_probe_write(span, scribble, sizeof(scribble)) ==
        CUDA_SUCCESS);
    assert(memcmp((const void *)(uintptr_t)span, scribble,
        sizeof(scribble)) == 0);
    assert(cuMemUnmap(span, 4096u) == CUDA_SUCCESS);
    assert(cuMemRelease(chunk) == CUDA_SUCCESS);
    assert(cuMemAddressFree(span, 4096u) == CUDA_SUCCESS);

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 703u, sizeof(pack_image), digest);
    {
        FILE *pack_file = fopen(SPARK_TEST_PACK, "rb");
        assert(pack_file != 0);
        assert(fread(pack_image, 1u, sizeof(pack_image), pack_file) ==
            sizeof(pack_image));
        assert(fclose(pack_file) == 0);
    }
    SparkTestStartServer(&thread_context, &thread_handle, SPARK_TEST_SOCKET,
        4ull * SPARK_TEST_CHUNK_BYTES);
    SparkTestAttachAndMap(&outcome, SPARK_TEST_SOCKET, SPARK_TEST_PACK,
        digest, sizeof(pack_image));
    assert(outcome.loaded_from_pack == 1u);
    assert(outcome.map_span_bytes == sizeof(pack_image));
    assert(memcmp(outcome.map_base, pack_image, sizeof(pack_image)) == 0);
    assert(cuda_stub_vmm_probe_write(
        (CUdeviceptr)(uintptr_t)outcome.map_base, scribble,
        sizeof(scribble)) == CUDA_SUCCESS);
    assert(memcmp(outcome.map_base, pack_image, sizeof(pack_image)) != 0);
    SparkWeightdAttachRelease(&outcome);
    assert(cuda_stub_vmm_probe_write(
        (CUdeviceptr)(uintptr_t)outcome.map_base, scribble, 8u) ==
        CUDA_ERROR_INVALID_VALUE);
    SparkTestStopServer(&thread_context, thread_handle);
    SparkTestClearAttachEnv();
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("w3 weightd: scribble-probe receipt green (the consumer map is"
        " RW today; the staged PROT_READ flip turns this probe into a"
        " refusal)\n");
}

int main(void)
{
    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestExportGatesAndBatch();
    SparkTestImportMapWarmAndCoverageGate();
    SparkTestMultiBatch();
    SparkTestCrossProcess();
    SparkTestScribbleProbeReceipt();
    printf("w3 weightd lane: fd export + consumer import/map green\n");
    return 0;
}
