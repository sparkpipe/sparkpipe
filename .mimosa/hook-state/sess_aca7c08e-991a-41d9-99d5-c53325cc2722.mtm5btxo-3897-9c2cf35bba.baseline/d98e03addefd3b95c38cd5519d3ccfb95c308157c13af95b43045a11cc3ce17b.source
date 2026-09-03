
#include <assert.h>
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
#include "sparkpipe/spark_weightd.h"

#define SPARK_TEST_ARENA_BYTES (1024ull * 1024ull)
#define SPARK_TEST_CEILING_BYTES (1536ull * 1024ull)
#define SPARK_TEST_TIMEOUT_NS 10000000000ull
#define SPARK_TEST_NO_SUCH_GENERATION UINT64_C(0xDEADBEEF0000)

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
    assert(spark_stub_cuda_outstanding_allocs() == 0u);
}

static uint32_t SparkTestNextRandom(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void SparkTestCopyBounded(char *destination, size_t capacity,
    const char *source)
{
    size_t length = strlen(source);
    assert(length + 1u <= capacity);
    memcpy(destination, source, length + 1u);
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

static void SparkTestMakeIdentity(SparkWeightdIdentity *identity,
    const char *model,
    const char *revision,
    uint32_t topology,
    uint64_t geometry_fingerprint,
    const char *pack_sha256,
    uint64_t arena_bytes)
{
    memset(identity, 0, sizeof(*identity));
    SparkTestCopyBounded(identity->model, SPARK_WEIGHTD_ID_BYTES, model);
    SparkTestCopyBounded(identity->revision, SPARK_WEIGHTD_REVISION_BYTES,
        revision);
    identity->topology = topology;
    identity->abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    identity->geometry_fingerprint = geometry_fingerprint;
    SparkTestCopyBounded(identity->pack_sha256, SPARK_WEIGHTD_SHA256_HEX_BYTES,
        pack_sha256);
    identity->arena_bytes = arena_bytes;
    assert(SparkWeightdIdentityPrepare(identity) == SPARK_STATUS_OK);
}

static void SparkTestMakeRequest(SparkWeightdAttachRequest *request,
    const SparkWeightdIdentity *identity,
    const char *pack_path)
{
    memset(request, 0, sizeof(*request));
    request->identity = *identity;
    SparkTestCopyBounded(request->pack_path, SPARK_WEIGHTD_PATH_BYTES,
        pack_path);
}

static void SparkTestConnect(SparkWeightdClient **client,
    const char *socket_path,
    uint64_t expect_ceiling)
{
    SparkWeightdHelloResult hello;
    memset(&hello, 0, sizeof(hello));
    assert(SparkWeightdClientConnect(socket_path, client, &hello) ==
        SPARK_STATUS_OK);
    assert(hello.status == SPARK_STATUS_OK);
    if (expect_ceiling != 0ull)
    {
        assert(hello.device_bytes_max == expect_ceiling);
    }
}

static void SparkTestAttach(SparkWeightdClient *client,
    const SparkWeightdAttachRequest *request,
    SparkWeightdAttachResult *result)
{
    memset(result, 0, sizeof(*result));
    assert(SparkWeightdClientAttach(client, request, result,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
}


static void SparkTestIdentityCanonicalization(void)
{
    SparkWeightdIdentity left;
    SparkWeightdIdentity right;
    SparkWeightdIdentity broken;
    char digest[SPARK_SHA256_HEX_BYTES];
    uint32_t index;

    SparkTestWritePack("/tmp/spark_weightd_test_canon.bin", 7u,
        4096ull, digest);
    SparkTestMakeIdentity(&left, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    SparkTestMakeIdentity(&right, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    for (index = 10u; index < SPARK_WEIGHTD_ID_BYTES; index++)
    {
        right.model[index] = (char)(0x80u + index);
    }
    right.reserved0 = 0xDEADBEEFu;
    right.reserved_tail[3] = (char)0x7F;
    assert(SparkWeightdIdentityPrepare(&right) == SPARK_STATUS_OK);
    assert(SparkWeightdIdentityEqual(&left, &right));
    assert(memcmp(&left, &right, sizeof(left)) == 0);

    SparkTestMakeIdentity(&broken, "model", "rev1", 16u, 0xAull, digest,
        4096ull);
    assert(!SparkWeightdIdentityEqual(&left, &broken));
    SparkTestMakeIdentity(&broken, "model", "rev2", 4u, 0xAull, digest,
        4096ull);
    assert(!SparkWeightdIdentityEqual(&left, &broken));

    SparkTestMakeIdentity(&broken, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    broken.abi_version = 0u;
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_INVALID_ARGUMENT);
    broken = left;
    broken.arena_bytes = 0ull;
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestMakeIdentity(&broken, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    broken.pack_sha256[10] = 'g';
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestMakeIdentity(&broken, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    memset(broken.model, 'x', SPARK_WEIGHTD_ID_BYTES);
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_INVALID_ARGUMENT);
    (void)remove("/tmp/spark_weightd_test_canon.bin");
    printf("identity canonicalization green\n");
}


static void SparkTestSharedRefcountAndConsumerDeath(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_share.sock";
    const char *pack_path = "/tmp/spark_weightd_test_share.spack";
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdDetachResult detach;
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdClient *client_a = 0;
    SparkWeightdClient *client_b = 0;
    SparkWeightdClient *client_c = 0;
    uint64_t generation;
    uint64_t handle;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(pack_path, 42u, SPARK_TEST_ARENA_BYTES, digest);
    SparkTestMakeIdentity(&identity, "weightd-test", "share-rev", 4u,
        0x1234ull, digest, SPARK_TEST_ARENA_BYTES);
    SparkTestMakeRequest(&request, &identity, pack_path);
    SparkTestStartServer(&thread_context, &thread_handle, socket_path,
        SPARK_TEST_CEILING_BYTES);

    SparkTestConnect(&client_a, socket_path, SPARK_TEST_CEILING_BYTES);
    SparkTestAttach(client_a, &request, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 1u);
    assert(result.refcount == 1u);
    assert(result.arena_count == 1u);
    assert(result.resident_bytes == SPARK_TEST_ARENA_BYTES);
    assert(result.device_handle != 0ull);
    generation = result.arena_generation;
    handle = result.device_handle;

    SparkTestConnect(&client_b, socket_path, 0ull);
    SparkTestAttach(client_b, &request, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 0u);
    assert(result.refcount == 2u);
    assert(result.arena_count == 1u);
    assert(result.resident_bytes == SPARK_TEST_ARENA_BYTES);
    assert(result.arena_generation == generation);
    assert(result.device_handle == handle);

    SparkTestAttach(client_b, &request, &result);
    assert(result.status == SPARK_STATUS_DUPLICATE);
    assert(result.arena_count == 1u);

    SparkWeightdClientClose(client_b);
    client_b = 0;
    for (;;)
    {
        struct timespec pause = {0, 20 * 1000 * 1000};
        SparkTestConnect(&client_c, socket_path, 0ull);
        SparkTestAttach(client_c, &request, &result);
        if (result.status == SPARK_STATUS_OK && result.refcount == 2u)
        {
            break;
        }
        assert(result.status == SPARK_STATUS_OK);
        assert(result.refcount == 3u);
        assert(SparkWeightdClientDetach(client_c, result.arena_generation,
            &detach, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        assert(detach.status == SPARK_STATUS_OK);
        SparkWeightdClientClose(client_c);
        client_c = 0;
        assert(nanosleep(&pause, 0) == 0);
    }
    assert(result.loaded_from_pack == 0u);
    assert(result.arena_count == 1u);
    assert(result.arena_generation == generation);

    assert(SparkWeightdClientDetach(client_a, SPARK_TEST_NO_SUCH_GENERATION,
        &detach, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_NOT_FOUND);

    assert(SparkWeightdClientDetach(client_a, generation, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    assert(detach.refcount == 1u);
    assert(SparkWeightdClientDetach(client_c, generation, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    assert(detach.refcount == 0u);
    assert(detach.arena_count == 1u);

    {
        SparkWeightdReclaimResult reclaim;
        memset(&reclaim, 0, sizeof(reclaim));
        assert(SparkWeightdClientReclaim(client_c, &reclaim,
            SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        assert(reclaim.status == SPARK_STATUS_OK);
        assert(reclaim.reclaimed_arena_count == 1u);
        assert(reclaim.reclaimed_bytes == SPARK_TEST_ARENA_BYTES);
        assert(reclaim.arena_count == 0u);
    }

    SparkWeightdClientClose(client_a);
    SparkWeightdClientClose(client_c);
    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(pack_path);
    (void)remove(socket_path);
    printf("identity sharing + consumer-death refcount green\n");
}


static void SparkTestStopAttachStartNeverHoldsTwoArenas(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_no2x.sock";
    const char *pack_a = "/tmp/spark_weightd_test_no2x_a.spack";
    const char *pack_b = "/tmp/spark_weightd_test_no2x_b.spack";
    char digest_a[SPARK_SHA256_HEX_BYTES];
    char digest_b[SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity_a;
    SparkWeightdIdentity identity_b;
    SparkWeightdAttachRequest request_a;
    SparkWeightdAttachRequest request_b;
    SparkWeightdAttachResult result;
    SparkWeightdDetachResult detach;
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdClient *serving = 0;
    SparkWeightdClient *updater = 0;
    SparkWeightdClient *probe = 0;
    uint64_t generation_a;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(pack_a, 100u, SPARK_TEST_ARENA_BYTES, digest_a);
    SparkTestWritePack(pack_b, 200u, SPARK_TEST_ARENA_BYTES, digest_b);
    SparkTestMakeIdentity(&identity_a, "weightd-test", "old-rev", 4u,
        0xABCDull, digest_a, SPARK_TEST_ARENA_BYTES);
    SparkTestMakeIdentity(&identity_b, "weightd-test", "new-rev", 4u,
        0xABCEull, digest_b, SPARK_TEST_ARENA_BYTES);
    SparkTestMakeRequest(&request_a, &identity_a, pack_a);
    SparkTestMakeRequest(&request_b, &identity_b, pack_b);

    SparkTestStartServer(&thread_context, &thread_handle, socket_path,
        SPARK_TEST_CEILING_BYTES);
    SparkTestConnect(&serving, socket_path, 0ull);
    SparkTestAttach(serving, &request_a, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 1u);
    generation_a = result.arena_generation;

    SparkTestConnect(&updater, socket_path, 0ull);
    SparkTestAttach(updater, &request_b, &result);
    assert(result.status == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(result.arena_count == 1u);
    assert(result.resident_bytes == SPARK_TEST_ARENA_BYTES);

    SparkTestAttach(updater, &request_a, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 0u);
    assert(result.refcount == 2u);
    assert(result.arena_generation == generation_a);

    assert(SparkWeightdClientDetach(serving, generation_a, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    assert(detach.refcount == 1u);
    assert(SparkWeightdClientDetach(updater, generation_a, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    assert(detach.refcount == 0u);
    assert(detach.arena_count == 1u);

    SparkTestAttach(updater, &request_b, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 1u);
    assert(result.refcount == 1u);
    assert(result.arena_count == 1u);
    assert(result.resident_bytes == SPARK_TEST_ARENA_BYTES);
    assert(result.arena_generation != generation_a);

    SparkTestAttach(serving, &request_b, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 0u);
    assert(result.refcount == 2u);

    SparkTestConnect(&probe, socket_path, 0ull);
    SparkTestAttach(probe, &request_a, &result);
    assert(result.status == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(result.arena_count == 1u);
    assert(result.resident_bytes == SPARK_TEST_ARENA_BYTES);

    SparkTestAttach(probe, &request_b, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 0u);
    assert(result.refcount == 3u);

    SparkWeightdClientClose(serving);
    SparkWeightdClientClose(updater);
    SparkWeightdClientClose(probe);
    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(pack_a);
    (void)remove(pack_b);
    (void)remove(socket_path);
    printf("stop-attach-start update holds ONE arena peak green\n");
}


static void SparkTestExpectConnectionClosed(const char *socket_path,
    const void *frame,
    size_t frame_bytes)
{
    struct sockaddr_un address;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    char buffer[16];
    struct pollfd poll_fd;
    assert(fd >= 0);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    SparkTestCopyBounded(address.sun_path, sizeof(address.sun_path),
        socket_path);
    assert(connect(fd, (const struct sockaddr *)&address, sizeof(address)) == 0);
    if (frame_bytes != 0u)
    {
        assert(write(fd, frame, frame_bytes) == (ssize_t)frame_bytes);
    }
    poll_fd.fd = fd;
    poll_fd.events = POLLIN;
    poll_fd.revents = 0;
    assert(poll(&poll_fd, 1u, 5000) > 0);
    assert(read(fd, buffer, sizeof(buffer)) == 0);
    (void)close(fd);
}

static void SparkTestFailClosedPaths(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_closed.sock";
    const char *pack_path = "/tmp/spark_weightd_test_closed.spack";
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity;
    SparkWeightdIdentity wrong_digest;
    SparkWeightdIdentity wrong_size;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdDetachResult detach;
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdClient *client = 0;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(pack_path, 300u, 65536ull, digest);
    SparkTestMakeIdentity(&identity, "weightd-test", "closed-rev", 4u,
        0x77ull, digest, 65536ull);
    SparkTestMakeRequest(&request, &identity, pack_path);

    SparkTestMakeIdentity(&wrong_digest, "weightd-test", "closed-rev", 4u,
        0x77ull,
        "0000000000000000000000000000000000000000000000000000000000000000",
        65536ull);
    SparkTestMakeIdentity(&wrong_size, "weightd-test", "closed-rev", 4u,
        0x77ull, digest, 65537ull);

    SparkTestStartServer(&thread_context, &thread_handle, socket_path,
        SPARK_TEST_CEILING_BYTES);
    SparkTestConnect(&client, socket_path, 0ull);

    SparkTestAttach(client, &request, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.refcount == 1u);
    assert(SparkWeightdClientDetach(client, result.arena_generation, &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(detach.status == SPARK_STATUS_OK);
    assert(detach.refcount == 0u);
    {
        SparkWeightdReclaimResult reclaim;
        memset(&reclaim, 0, sizeof(reclaim));
        assert(SparkWeightdClientReclaim(client, &reclaim,
            SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        assert(reclaim.status == SPARK_STATUS_OK);
        assert(reclaim.arena_count == 0u);
    }

    {
        SparkWeightdAttachRequest bad_request;
        SparkTestMakeRequest(&bad_request, &wrong_digest, pack_path);
        SparkTestAttach(client, &bad_request, &result);
        assert(result.status == SPARK_STATUS_HASH_MISMATCH);
        assert(result.arena_count == 0u);
    }
    {
        SparkWeightdAttachRequest bad_request;
        SparkTestMakeRequest(&bad_request, &wrong_size, pack_path);
        SparkTestAttach(client, &bad_request, &result);
        assert(result.status == SPARK_STATUS_INVALID_ARGUMENT);
        assert(result.arena_count == 0u);
    }
    {
        SparkWeightdAttachRequest missing_request;
        SparkTestMakeRequest(&missing_request, &identity,
            "/tmp/spark_weightd_test_missing.spack");
        SparkTestAttach(client, &missing_request, &result);
        assert(result.status == SPARK_STATUS_IO_ERROR);
        assert(result.arena_count == 0u);
    }
    {
        SparkWeightdAttachRequest bad_request;
        SparkTestMakeRequest(&bad_request, &identity, pack_path);
        memset(bad_request.identity.model, 'y', SPARK_WEIGHTD_ID_BYTES);
        assert(SparkWeightdClientAttach(client, &bad_request, &result,
            SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_INVALID_ARGUMENT);
    }

    SparkWeightdClientClose(client);

    {
        uint8_t junk[32];
        memset(junk, 0xEE, sizeof(junk));
        SparkTestExpectConnectionClosed(socket_path, junk, sizeof(junk));
    }
    {
        SparkWeightdIpcAttach wire;
        memset(&wire, 0, sizeof(wire));
        wire.header.magic = SPARK_WEIGHTD_IPC_MAGIC;
        wire.header.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
        wire.header.kind = SPARK_WEIGHTD_IPC_KIND_ATTACH;
        wire.header.body_bytes =
            SPARK_WEIGHTD_IPC_ATTACH_BYTES - SPARK_WEIGHTD_IPC_HEADER_BYTES;
        wire.header.request_id = 1ull;
        SparkTestExpectConnectionClosed(socket_path, &wire, sizeof(wire));
    }

    {
        SparkWeightdClient *doomed = 0;
        SparkTestConnect(&doomed, socket_path, 0ull);
        SparkTestStopServer(&thread_context, thread_handle);
        assert(SparkWeightdClientAttach(doomed, &request, &result,
            2000000000ull) != SPARK_STATUS_OK);
        SparkWeightdClientClose(doomed);
    }
    {
        SparkWeightdClient *ghost = 0;
        assert(SparkWeightdClientConnect(socket_path, &ghost, 0) ==
            SPARK_STATUS_IO_ERROR);
    }

    (void)remove(pack_path);
    (void)remove(socket_path);
    printf("fail-closed paths green\n");
}


static int SparkTestWaitReady(int pipe_fd, char *line, size_t line_capacity)
{
    size_t filled = 0u;
    time_t deadline = time(0) + 10;
    while (filled + 1u < line_capacity)
    {
        ssize_t bytes_read;
        struct pollfd poll_fd;
        poll_fd.fd = pipe_fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        if (poll(&poll_fd, 1u, 100) <= 0)
        {
            if (time(0) > deadline)
            {
                return 0;
            }
            continue;
        }
        bytes_read = read(pipe_fd, line + filled, line_capacity - 1u - filled);
        if (bytes_read <= 0)
        {
            return 0;
        }
        filled += (size_t)bytes_read;
        line[filled] = '\0';
        if (strstr(line, "\n") != 0)
        {
            return strstr(line, "spark_weightd ready") != 0 ? 1 : 0;
        }
    }
    return 0;
}

static void SparkTestDaemonProcessTermPath(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_daemon.sock";
    const char *stderr_path = "/tmp/spark_weightd_test_daemon.err";
    const char *pack_path = "/tmp/spark_weightd_test_daemon.spack";
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdClient *client = 0;
    char ready_line[256];
    int stdout_pipe[2];
    pid_t daemon_pid;
    int daemon_exit = -1;
    uint64_t waited_ms;

    assert(pipe(stdout_pipe) == 0);
    (void)unlink(stderr_path);
    SparkTestWritePack(pack_path, 400u, SPARK_TEST_ARENA_BYTES, digest);
    SparkTestMakeIdentity(&identity, "weightd-test", "daemon-rev", 4u,
        0x991ull, digest, SPARK_TEST_ARENA_BYTES);
    SparkTestMakeRequest(&request, &identity, pack_path);

    daemon_pid = fork();
    assert(daemon_pid >= 0);
    if (daemon_pid == 0)
    {
        char ceiling_text[32];
        snprintf(ceiling_text, sizeof(ceiling_text), "%llu",
            (unsigned long long)(2ull * SPARK_TEST_ARENA_BYTES));
        (void)dup2(stdout_pipe[1], 1);
        (void)close(stdout_pipe[0]);
        (void)close(stdout_pipe[1]);
        if (freopen(stderr_path, "w", stderr) == 0)
        {
            _exit(126);
        }
        execl(SPARK_TEST_WEIGHTD_BINARY, "spark_weightd",
            "--socket", socket_path, "--device-bytes-max", ceiling_text,
            (char *)0);
        _exit(127);
    }
    (void)close(stdout_pipe[1]);
    assert(SparkTestWaitReady(stdout_pipe[0], ready_line,
        sizeof(ready_line)) == 1);
    (void)close(stdout_pipe[0]);
    assert(strstr(ready_line, "unix=") != 0);
    assert(strstr(ready_line, "ceiling=2097152") != 0);

    SparkTestConnect(&client, socket_path, 2ull * SPARK_TEST_ARENA_BYTES);
    SparkTestAttach(client, &request, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 1u);
    assert(result.refcount == 1u);

    assert(kill(daemon_pid, SIGTERM) == 0);
    waited_ms = 0ull;
    while (waited_ms < 10000ull)
    {
        int wait_status = 0;
        pid_t reaped = waitpid(daemon_pid, &wait_status, WNOHANG);
        if (reaped == daemon_pid)
        {
            assert(WIFEXITED(wait_status));
            daemon_exit = WEXITSTATUS(wait_status);
            break;
        }
        assert(reaped == 0);
        {
            struct timespec pause = {0, 20 * 1000 * 1000};
            assert(nanosleep(&pause, 0) == 0);
        }
        waited_ms += 20ull;
    }
    assert(daemon_exit == 0);
    {
        struct stat socket_stat;
        assert(stat(socket_path, &socket_stat) != 0);
    }
    {
        FILE *err_file = fopen(stderr_path, "r");
        char err_text[512];
        size_t err_bytes;
        assert(err_file != 0);
        err_bytes = fread(err_text, 1u, sizeof(err_text) - 1u, err_file);
        err_text[err_bytes] = '\0';
        (void)fclose(err_file);
        assert(strstr(err_text, "spark_weightd stopped") != 0);
        assert(strstr(err_text, "arenas=1") != 0);
        assert(strstr(err_text, "bytes=1048576") != 0);
    }
    assert(SparkWeightdClientAttach(client, &request, &result,
        2000000000ull) != SPARK_STATUS_OK);
    SparkWeightdClientClose(client);
    (void)remove(pack_path);
    (void)remove(stderr_path);
    (void)remove(socket_path);
    printf("daemon process TERM path green (exit 0, socket unlinked)\n");
}

int main(void)
{
    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestIdentityCanonicalization();
    SparkTestSharedRefcountAndConsumerDeath();
    SparkTestStopAttachStartNeverHoldsTwoArenas();
    SparkTestFailClosedPaths();
    SparkTestDaemonProcessTermPath();
    printf("w2 weightd lane: identity arenas + NO-2x + TERM green\n");
    return 0;
}
