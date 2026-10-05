
#include <assert.h>
#include <dirent.h>
#include <poll.h>
#include <netinet/in.h>
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
#include "sparkpipe/spark_weightd_receipt.h"
#include "sparkpipe/spark_kv_shared_index.h"
#include <sys/mman.h>

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

static uint64_t spark_test_kv_shared_window_bytes;

static void SparkTestStartServerReserve(SparkTestServerThread *thread_context,
    pthread_t *thread_handle,
    const char *socket_path,
    uint64_t ceiling_bytes,
    uint64_t kv_reserve_bytes,
    uint64_t kv_write_budget_bytes_per_day)
{
    SparkWeightdServerConfig config;
    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = ceiling_bytes;
    config.kv_reserve_bytes = kv_reserve_bytes;
    config.kv_write_budget_bytes_per_day = kv_write_budget_bytes_per_day;
    config.kv_shared_window_bytes = spark_test_kv_shared_window_bytes;
    assert(SparkWeightdServerCreate(&config, &thread_context->server) ==
        SPARK_STATUS_OK);
    thread_context->stop = 0;
    thread_context->run_status = SPARK_STATUS_OK;
    assert(pthread_create(thread_handle, 0, SparkTestServerThreadMain,
        thread_context) == 0);
}

static void SparkTestStartServer(SparkTestServerThread *thread_context,
    pthread_t *thread_handle,
    const char *socket_path,
    uint64_t ceiling_bytes)
{
    SparkTestStartServerReserve(thread_context,thread_handle,socket_path,ceiling_bytes,0ull,0ull);
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

static int SparkTestFindPackReceipt(const char *pack_path, const char *digest)
{
    char resolved[4096];
    char receipt_path[4200];
    SparkWeightdReceipt receipt;
    if (realpath(pack_path, resolved) == 0 ||
        snprintf(receipt_path, sizeof(receipt_path), "%s%s", resolved,
            SPARK_WEIGHTD_RECEIPT_SUFFIX) <= 0 ||
        SparkWeightdReceiptLoad(receipt_path, &receipt) != SPARK_STATUS_OK)
        return 0;
    return strcmp(receipt.sha256, digest) == 0 &&
        strncmp(receipt.verifier, "weightd ", 8u) == 0;
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

    SparkTestMakeIdentity(&broken, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    broken.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN;
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_OK);
    assert(broken.abi_version == SPARK_WEIGHTD_IPC_ABI_VERSION);
    assert(SparkWeightdIdentityEqual(&left, &broken));
    SparkTestMakeIdentity(&broken, "model", "rev1", 4u, 0xAull, digest, 4096ull);
    broken.abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN - 1u;
    assert(SparkWeightdIdentityPrepare(&broken) == SPARK_STATUS_OK);
    assert(!SparkWeightdIdentityEqual(&left, &broken));
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

static void SparkTestReclaimPackFrees(SparkWeightdClient *client,
    const char *pack_sha256, uint32_t freed, uint32_t busy, uint32_t remaining)
{
    SparkWeightdReclaimResult reclaim;
    assert(SparkWeightdClientReclaimPack(client, pack_sha256, &reclaim,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(reclaim.status == SPARK_STATUS_OK);
    assert(reclaim.reclaimed_arena_count == freed);
    assert(reclaim.reclaimed_bytes == (uint64_t)freed * 65536ull);
    assert(reclaim.busy_arena_count == busy);
    assert(reclaim.arena_count == remaining);
    assert(reclaim.resident_bytes == (uint64_t)remaining * 65536ull);
}

static void SparkTestReclaimPackIsScopedToOnePack(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_reclaim_pack.sock";
    const char *pack_paths[3] = {"/tmp/spark_weightd_test_reclaim_a.spack",
        "/tmp/spark_weightd_test_reclaim_b.spack",
        "/tmp/spark_weightd_test_reclaim_c.spack"};
    char digests[3][SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult result;
    SparkWeightdDetachResult detach;
    SparkWeightdReclaimResult reclaim;
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdClient *lane_x = 0;
    SparkWeightdClient *lane_y = 0;
    uint64_t generations[4];
    uint32_t index;

    spark_stub_cuda_reset_faults();
    for (index = 0u; index < 3u; index++)
        SparkTestWritePack(pack_paths[index], 500u + index, 65536ull, digests[index]);
    SparkTestStartServer(&thread_context, &thread_handle, socket_path,
        SPARK_TEST_CEILING_BYTES);
    SparkTestConnect(&lane_x, socket_path, 0ull);
    SparkTestConnect(&lane_y, socket_path, 0ull);
    for (index = 0u; index < 4u; index++)
    {
        uint32_t pack = index < 3u ? index : 0u;
        SparkTestMakeIdentity(&identity, index < 3u ? "lane-x" : "lane-y",
            "rev", 4u, 0x51ull + index, digests[pack], 65536ull);
        SparkTestMakeRequest(&request, &identity, pack_paths[pack]);
        SparkTestAttach(index == 3u ? lane_y : lane_x, &request, &result);
        assert(result.status == SPARK_STATUS_OK);
        assert(result.loaded_from_pack == 1u);
        assert(result.arena_count == index + 1u);
        generations[index] = result.arena_generation;
    }
    for (index = 0u; index < 4u; index++)
    {
        if (index == 1u)
            continue;
        assert(SparkWeightdClientDetach(index == 3u ? lane_y : lane_x,
            generations[index], &detach, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        assert(detach.status == SPARK_STATUS_OK);
        assert(detach.refcount == 0u);
    }
    assert(detach.arena_count == 4u);

    SparkTestReclaimPackFrees(lane_y, digests[2], 1u, 0u, 3u);
    SparkTestReclaimPackFrees(lane_y, digests[1], 0u, 1u, 3u);
    SparkTestReclaimPackFrees(lane_y, digests[2], 0u, 0u, 3u);
    SparkTestReclaimPackFrees(lane_y,
        "0000000000000000000000000000000000000000000000000000000000000000", 0u, 0u, 3u);
    SparkTestReclaimPackFrees(lane_y, digests[0], 2u, 0u, 1u);

    memset(&reclaim, 0xA5, sizeof(reclaim));
    assert(SparkWeightdClientReclaimPack(lane_y, "ABCDEF", &reclaim,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(reclaim.reclaimed_arena_count == 0u && reclaim.busy_arena_count == 0u);
    {
        char upper[SPARK_SHA256_HEX_BYTES];
        memcpy(upper, digests[1], sizeof(upper));
        for (index = 0u; index < 64u; index++)
            if (upper[index] >= 'a' && upper[index] <= 'f')
                upper[index] = (char)(upper[index] - 'a' + 'A');
        assert(SparkWeightdClientReclaimPack(lane_y, upper, &reclaim,
            SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_INVALID_ARGUMENT);
    }

    assert(SparkWeightdClientReclaim(lane_y, &reclaim,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(reclaim.status == SPARK_STATUS_OK);
    assert(reclaim.reclaimed_arena_count == 0u);
    assert(reclaim.busy_arena_count == 1u);
    assert(reclaim.arena_count == 1u);

    assert(SparkWeightdClientDetach(lane_x, generations[1], &detach,
        SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    SparkTestReclaimPackFrees(lane_x, digests[1], 1u, 0u, 0u);

    SparkWeightdClientClose(lane_x);
    SparkWeightdClientClose(lane_y);
    SparkTestStopServer(&thread_context, thread_handle);
    for (index = 0u; index < 3u; index++)
        (void)remove(pack_paths[index]);
    (void)remove(socket_path);
    printf("pack-scoped reclaim frees only the named pack's cold arenas green\n");
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

static int SparkTestRawConnect(const char *socket_path)
{
    struct sockaddr_un address;
    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    assert(fd >= 0);
    memset(&address, 0, sizeof(address));
    address.sun_family = AF_UNIX;
    SparkTestCopyBounded(address.sun_path, sizeof(address.sun_path), socket_path);
    assert(connect(fd, (const struct sockaddr *)&address, sizeof(address)) == 0);
    return fd;
}

static void SparkTestRawFrame(SparkWeightdIpcHeader *header, uint32_t abi_version,
    uint32_t kind, uint64_t request_id)
{
    memset(header, 0, sizeof(*header));
    header->magic = SPARK_WEIGHTD_IPC_MAGIC;
    header->abi_version = abi_version;
    header->kind = kind;
    header->request_id = request_id;
}

static ssize_t SparkTestRawExchange(int fd, const SparkWeightdIpcHeader *request,
    void *response, size_t response_bytes)
{
    struct pollfd poll_fd;
    size_t filled = 0u;
    assert(write(fd, request, sizeof(*request)) == (ssize_t)sizeof(*request));
    while (filled < response_bytes)
    {
        ssize_t got;
        poll_fd.fd = fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        assert(poll(&poll_fd, 1u, 5000) > 0);
        got = read(fd, (uint8_t *)response + filled, response_bytes - filled);
        if (got <= 0)
            return got;
        filled += (size_t)got;
    }
    return (ssize_t)filled;
}

static void SparkTestServedAbiVersions(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_abi.sock";
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdIpcHeader request;
    SparkWeightdIpcHelloAck ack;
    SparkWeightdIpcMeshStagingMapResult staging;
    SparkWeightdIpcMeshMapResult map;
    SparkWeightdClient *client = 0;
    void *mapping = (void *)&thread_context;
    uint32_t version;
    int fd;
    (void)remove(socket_path);
    SparkTestStartServer(&thread_context, &thread_handle, socket_path,
        SPARK_TEST_CEILING_BYTES);
    assert(SparkWeightdIpcAbiServed(8u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(9u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(7u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 0u);
    assert(SparkWeightdIpcAbiServed(10u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(11u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(12u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(13u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 1u);
    assert(SparkWeightdIpcAbiServed(14u, SPARK_WEIGHTD_IPC_KIND_HELLO) == 0u);
    assert(SparkWeightdIpcAbiServed(12u, SPARK_WEIGHTD_IPC_KIND_SLOT_EXPORT) == 0u);
    assert(SparkWeightdIpcAbiServed(13u, SPARK_WEIGHTD_IPC_KIND_SLOT_EXPORT) == 1u);
    assert(SparkWeightdIpcAbiServed(12u, SPARK_WEIGHTD_IPC_KIND_LEASE_SLOTS) == 0u);
    assert(SparkWeightdIpcAbiServed(13u, SPARK_WEIGHTD_IPC_KIND_LEASE_SLOTS) == 1u);
    assert(SparkWeightdIpcAbiServed(11u, SPARK_WEIGHTD_IPC_KIND_KV_SHARED_ATTACH) == 0u);
    assert(SparkWeightdIpcAbiServed(12u, SPARK_WEIGHTD_IPC_KIND_KV_SHARED_ATTACH) == 1u);
    assert(SparkWeightdIpcAbiServed(8u, SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP) == 0u);
    assert(SparkWeightdIpcAbiServed(9u, SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP) == 1u);
    assert(SparkWeightdIpcAbiServed(10u, SPARK_WEIGHTD_IPC_KIND_KV_POOL_ATTACH) == 0u);
    assert(SparkWeightdIpcAbiServed(11u, SPARK_WEIGHTD_IPC_KIND_KV_POOL_ATTACH) == 1u);
    assert(SparkWeightdIpcAbiServed(10u, SPARK_WEIGHTD_IPC_KIND_KV_POOL_STATUS) == 0u);
    assert(SparkWeightdIpcAbiServed(11u, SPARK_WEIGHTD_IPC_KIND_KV_POOL_RESIZE) == 1u);
    for (version = SPARK_WEIGHTD_IPC_ABI_VERSION_SERVED_MIN; version <= SPARK_WEIGHTD_IPC_ABI_VERSION; version++)
    {
        fd = SparkTestRawConnect(socket_path);
        SparkTestRawFrame(&request, version, SPARK_WEIGHTD_IPC_KIND_HELLO, 7u);
        memset(&ack, 0, sizeof(ack));
        assert(SparkTestRawExchange(fd, &request, &ack, sizeof(ack)) == (ssize_t)sizeof(ack));
        assert(ack.header.abi_version == version && ack.header.kind == SPARK_WEIGHTD_IPC_KIND_HELLO_ACK &&
            ack.header.request_id == 7u && ack.status == SPARK_STATUS_OK);
        assert(SparkWeightdIpcValidateHeaderVersion(&ack.header, sizeof(ack),
            SPARK_WEIGHTD_IPC_KIND_HELLO_ACK, version) == SPARK_STATUS_OK);
        SparkTestRawFrame(&request, version, SPARK_WEIGHTD_IPC_KIND_MESH_MAP, 8u);
        memset(&map, 0, sizeof(map));
        assert(SparkTestRawExchange(fd, &request, &map, sizeof(map)) == (ssize_t)sizeof(map));
        assert(map.header.abi_version == version && map.header.kind == SPARK_WEIGHTD_IPC_KIND_MESH_MAP_RESULT &&
            map.status == SPARK_STATUS_INVALID_ARGUMENT);
        SparkTestRawFrame(&request, version == 8u ? 9u : 8u, SPARK_WEIGHTD_IPC_KIND_MESH_MAP, 9u);
        assert(SparkTestRawExchange(fd, &request, &map, sizeof(map)) == 0);
        (void)close(fd);
    }
    fd = SparkTestRawConnect(socket_path);
    SparkTestRawFrame(&request, 8u, SPARK_WEIGHTD_IPC_KIND_HELLO, 1u);
    assert(SparkTestRawExchange(fd, &request, &ack, sizeof(ack)) == (ssize_t)sizeof(ack));
    SparkTestRawFrame(&request, 8u, SPARK_WEIGHTD_IPC_KIND_MESH_STAGING_MAP, 2u);
    assert(SparkTestRawExchange(fd, &request, &staging, sizeof(staging)) == 0);
    (void)close(fd);
    SparkTestRawFrame(&request, 7u, SPARK_WEIGHTD_IPC_KIND_HELLO, 1u);
    SparkTestExpectConnectionClosed(socket_path, &request, sizeof(request));
    SparkTestRawFrame(&request, 14u, SPARK_WEIGHTD_IPC_KIND_HELLO, 1u);
    SparkTestExpectConnectionClosed(socket_path, &request, sizeof(request));
    SparkTestConnect(&client, socket_path, 0ull);
    assert(SparkWeightdClientMeshStagingMap(client, &mapping, SPARK_TEST_TIMEOUT_NS) ==
        SPARK_STATUS_INVALID_ARGUMENT);
    assert(mapping == 0);
    SparkWeightdClientClose(client);
    SparkTestStopServer(&thread_context, thread_handle);
    (void)remove(socket_path);
    printf("served ABI 8 to 12 green (replies echo the client ABI; staging map needs ABI 9, resizable kv pools ABI 11, shared kv pools ABI 12)\n");
}

static ssize_t SparkTestRawSend(int fd, const void *request, size_t request_bytes,
    void *response, size_t response_bytes)
{
    struct pollfd poll_fd;
    size_t filled = 0u;
    assert(write(fd, request, request_bytes) == (ssize_t)request_bytes);
    while (filled < response_bytes)
    {
        ssize_t got;
        poll_fd.fd = fd;
        poll_fd.events = POLLIN;
        poll_fd.revents = 0;
        assert(poll(&poll_fd, 1u, 5000) > 0);
        got = read(fd, (uint8_t *)response + filled, response_bytes - filled);
        if (got <= 0)
            return got;
        filled += (size_t)got;
    }
    return (ssize_t)filled;
}

static void SparkTestMeshStatusFrame(SparkWeightdIpcMeshStatus *request, uint32_t abi_version,
    uint64_t request_id, uint32_t layout, uint32_t reserved)
{
    memset(request, 0, sizeof(*request));
    request->header.magic = SPARK_WEIGHTD_IPC_MAGIC;
    request->header.abi_version = abi_version;
    request->header.kind = SPARK_WEIGHTD_IPC_KIND_MESH_STATUS;
    request->header.body_bytes = (uint32_t)(sizeof(*request) - sizeof(request->header));
    request->header.request_id = request_id;
    request->layout = layout;
    request->reserved = reserved;
}

static void SparkTestMeshStatus(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_mesh_status.sock";
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    SparkWeightdIpcHeader hello;
    SparkWeightdIpcHelloAck ack;
    SparkWeightdIpcMeshStatus request;
    static SparkWeightdIpcMeshStatusResult result;
    static SparkWeightdIpcMeshStatusResult clean;
    SparkWeightdClient *client = 0;
    SparkWeightdHelloResult hello_result;
    SparkWeightdMeshTopology empty;
    uint32_t lane = SPARK_WEIGHTD_LANE_NONE;
    uint32_t outcome = 0u;
    uint32_t index;
    int fd;
    (void)remove(socket_path);
    SparkTestStartServer(&thread_context, &thread_handle, socket_path, SPARK_TEST_CEILING_BYTES);
    fd = SparkTestRawConnect(socket_path);
    SparkTestRawFrame(&hello, 9u, SPARK_WEIGHTD_IPC_KIND_HELLO, 1u);
    assert(SparkTestRawExchange(fd, &hello, &ack, sizeof(ack)) == (ssize_t)sizeof(ack));
    SparkTestMeshStatusFrame(&request, 9u, 2u, SPARK_WEIGHTD_MESH_STATUS_LAYOUT, 0u);
    assert(SparkTestRawSend(fd, &request, sizeof(request), &result, sizeof(result)) == (ssize_t)sizeof(result));
    assert(result.header.kind == SPARK_WEIGHTD_IPC_KIND_MESH_STATUS_RESULT && result.header.abi_version == 9u &&
        result.header.body_bytes == 4072u && result.header.request_id == 2u);
    assert(result.layout == 1u && result.layout_compat == 1u && result.pid == (uint32_t)getpid() &&
        result.daemon_generation == ack.daemon_generation && result.daemon_generation != 0u);
    assert(result.status == (uint32_t)SPARK_STATUS_UNSUPPORTED && result.mesh_state == SPARK_WEIGHTD_MESH_STATE_DISABLED &&
        result.pair_rank == UINT32_MAX);
    assert(result.reserved0 == 0u);
    for (index = 0u; index < sizeof(result.reserved_tail); index++)
        assert(result.reserved_tail[index] == 0u);
    SparkTestMeshStatusFrame(&request, 9u, 3u, 0u, 0u);
    assert(SparkTestRawSend(fd, &request, sizeof(request), &result, sizeof(result)) == (ssize_t)sizeof(result));
    assert(result.status == (uint32_t)SPARK_STATUS_INVALID_ARGUMENT && result.header.request_id == 3u);
    SparkTestMeshStatusFrame(&request, 9u, 4u, 1u, 7u);
    assert(SparkTestRawSend(fd, &request, sizeof(request), &result, sizeof(result)) == (ssize_t)sizeof(result));
    assert(result.status == (uint32_t)SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestMeshStatusFrame(&request, 9u, 5u, 9u, 0u);
    assert(SparkTestRawSend(fd, &request, sizeof(request), &result, sizeof(result)) == (ssize_t)sizeof(result));
    assert(result.layout == 1u && result.status == (uint32_t)SPARK_STATUS_UNSUPPORTED);
    (void)close(fd);
    fd = SparkTestRawConnect(socket_path);
    SparkTestRawFrame(&hello, 8u, SPARK_WEIGHTD_IPC_KIND_HELLO, 1u);
    assert(SparkTestRawExchange(fd, &hello, &ack, sizeof(ack)) == (ssize_t)sizeof(ack));
    SparkTestMeshStatusFrame(&request, 8u, 2u, 1u, 0u);
    assert(SparkTestRawSend(fd, &request, sizeof(request), &result, sizeof(result)) == 0);
    (void)close(fd);
    SparkTestMeshStatusFrame(&request, 9u, 1u, 1u, 0u);
    SparkTestExpectConnectionClosed(socket_path, &request, sizeof(request));
    assert(SparkWeightdClientConnectWithin(socket_path, SPARK_TEST_TIMEOUT_NS, &client, &hello_result) == SPARK_STATUS_OK);
    memset(&empty, 0, sizeof(empty));
    assert(SparkWeightdClientLaneAcquire(client, 6u, &empty, &lane, SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK && lane == 6u);
    for (index = 0u; index < 200u; index++)
    {
        assert(SparkWeightdMeshStatusQuery(socket_path, SPARK_TEST_TIMEOUT_NS, &result, &outcome) == SPARK_STATUS_OK &&
            outcome == SPARK_WEIGHTD_MESH_QUERY_ANSWERED);
        if ((result.lanes[6].flags & SPARK_WEIGHTD_MESH_LANE_OWNED) != 0u)
            break;
        {
            struct timespec pause = {0, 10000000L};
            (void)nanosleep(&pause, 0);
        }
    }
    assert((result.lanes[6].flags & SPARK_WEIGHTD_MESH_LANE_OWNED) != 0u && (result.lanes[5].flags & SPARK_WEIGHTD_MESH_LANE_OWNED) == 0u);
    SparkWeightdClientClose(client);
    for (index = 0u; index < 200u; index++)
    {
        assert(SparkWeightdMeshStatusQuery(socket_path, SPARK_TEST_TIMEOUT_NS, &result, &outcome) == SPARK_STATUS_OK);
        if ((result.lanes[6].flags & SPARK_WEIGHTD_MESH_LANE_OWNED) == 0u)
            break;
        {
            struct timespec pause = {0, 10000000L};
            (void)nanosleep(&pause, 0);
        }
    }
    assert((result.lanes[6].flags & SPARK_WEIGHTD_MESH_LANE_OWNED) == 0u);
    SparkTestStopServer(&thread_context, thread_handle);
    assert(access(socket_path, F_OK) != 0);
    memset(&clean, 0, sizeof(clean));
    assert(SparkWeightdMeshStatusQuery(socket_path, UINT64_C(500000000), &clean, &outcome) != SPARK_STATUS_OK &&
        outcome == SPARK_WEIGHTD_MESH_QUERY_ABSENT);
    printf("mesh status green (kind 41 answers inline in layout 1, refuses bad requests, closes on ABI 8 and before HELLO, reports lane owners)\n");
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
    int latch_socket;
    struct sockaddr_in latch_address;
    socklen_t latch_address_bytes = sizeof(latch_address);
    pid_t daemon_pid;
    int daemon_exit = -1;
    uint64_t waited_ms;

    latch_socket = socket(AF_INET,SOCK_STREAM,0);
    assert(latch_socket >= 0);
    memset(&latch_address,0,sizeof(latch_address));
    latch_address.sin_family = AF_INET;
    latch_address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    assert(bind(latch_socket,(struct sockaddr *)&latch_address,sizeof(latch_address)) == 0);
    assert(getsockname(latch_socket,(struct sockaddr *)&latch_address,&latch_address_bytes) == 0);
    assert(ntohs(latch_address.sin_port) != 0u);
    assert(pipe(stdout_pipe) == 0);
    (void)unlink(stderr_path);
    SparkTestWritePack(pack_path, 400u, SPARK_TEST_ARENA_BYTES, digest);
    SparkTestMakeIdentity(&identity, "weightd-test", "daemon-rev", 4u,
        0x991ull, digest, SPARK_TEST_ARENA_BYTES);
    SparkTestMakeRequest(&request, &identity, pack_path);

    assert(close(latch_socket) == 0);
    daemon_pid = fork();
    assert(daemon_pid >= 0);
    if (daemon_pid == 0)
    {
        char ceiling_text[32];
        char latch_text[16];
        snprintf(latch_text,sizeof(latch_text),"%u",(unsigned)ntohs(latch_address.sin_port));
        setenv("SPARK_WEIGHTD_LATCH_PORT",latch_text,1);
        setenv("SPARK_WEIGHTD_KV_RESERVE_BYTES","0",1);
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
    usleep(1100000);
    SparkTestAttach(client, &request, &result);
    assert(result.status == SPARK_STATUS_OK);
    assert(result.loaded_from_pack == 1u);
    assert(result.refcount == 1u);
    assert(SparkTestFindPackReceipt(pack_path, request.identity.pack_sha256) == 1);

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
    {
        char receipt_path[512];
        (void)snprintf(receipt_path, sizeof(receipt_path), "%s%s", pack_path,
            SPARK_WEIGHTD_RECEIPT_SUFFIX);
        (void)remove(receipt_path);
    }
    (void)remove(pack_path);
    (void)remove(stderr_path);
    (void)remove(socket_path);
    printf("daemon process TERM path green (exit 0, socket unlinked)\n");
}

#define SPARK_TEST_KV_CHUNK_BYTES (2ull * 1024ull * 1024ull)

static SparkStatus SparkTestKvAttachSized(SparkWeightdClient *client,uint8_t key_byte,uint64_t bytes,uint64_t minimum_bytes,uint64_t chunk_bytes,uint64_t metadata_bytes,SparkWeightdKvPoolGrant *grant)
{
    SparkWeightdKvPoolRequest request;
    memset(&request,0,sizeof(request));
    memset(request.key,key_byte,sizeof(request.key));
    request.device_bytes = bytes;
    request.minimum_bytes = minimum_bytes;
    request.chunk_bytes = chunk_bytes;
    request.metadata_bytes = metadata_bytes;
    request.label = "kvpool-test";
    return(SparkWeightdClientKvPoolAttach(client,&request,grant,SPARK_TEST_TIMEOUT_NS));
}

static SparkStatus SparkTestKvAttach(SparkWeightdClient *client,uint8_t key_byte,uint64_t bytes,uint64_t minimum_bytes,uint64_t metadata_bytes,SparkWeightdKvPoolGrant *grant)
{
    return(SparkTestKvAttachSized(client,key_byte,bytes,minimum_bytes,SPARK_TEST_KV_CHUNK_BYTES,metadata_bytes,grant));
}

static void SparkTestKvGrantClose(SparkWeightdKvPoolGrant *grant)
{
    if (grant->metadata_fd >= 0)
        (void)close(grant->metadata_fd);
    grant->metadata_fd = -1;
}

static void SparkTestKvExportCloses(SparkWeightdClient *client,uint64_t generation,uint32_t first_chunk,uint32_t chunk_count,SparkStatus expected)
{
    int fds[SPARK_WEIGHTD_KV_POOL_EXPORT_MAX];
    uint32_t index;
    assert(SparkWeightdClientKvPoolExport(client,generation,first_chunk,chunk_count,fds,SPARK_TEST_TIMEOUT_NS) == expected);
    for (index = 0u; expected == SPARK_STATUS_OK && index < chunk_count; index++)
    {
        assert(fds[index] >= 0);
        (void)close(fds[index]);
    }
}

static void SparkTestKvPools(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_kv.sock";
    const char *pack_path = "/tmp/spark_weightd_test_kv.spack";
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkWeightdIdentity identity;
    SparkWeightdAttachRequest request;
    SparkWeightdAttachResult attach;
    SparkTestServerThread context;
    pthread_t thread;
    SparkWeightdClient *first = 0,*second = 0,*third = 0;
    SparkWeightdKvPoolGrant grant,other;
    SparkWeightdKvPoolState state;
    SparkWeightdServerConfig config;
    SparkWeightdServer *refused = 0;
    uint64_t generation,other_generation;
    uint32_t index;
    char probe[8];

    memset(&config,0,sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = 4u * SPARK_TEST_KV_CHUNK_BYTES;
    config.kv_reserve_bytes = config.device_bytes_max;
    assert(SparkWeightdServerCreate(&config,&refused) == SPARK_STATUS_INVALID_ARGUMENT && refused == 0);

    (void)unlink(socket_path);
    SparkTestStartServerReserve(&context,&thread,socket_path,8u * SPARK_TEST_KV_CHUNK_BYTES,3u * SPARK_TEST_KV_CHUNK_BYTES,0ull);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvAttach(first,0x31,SPARK_TEST_KV_CHUNK_BYTES,SPARK_TEST_KV_CHUNK_BYTES,0u,&grant) == SPARK_STATUS_CAPACITY_EXCEEDED && SparkWeightdServerKvPoolCount(context.server) == 0u);
    SparkWeightdClientClose(first);
    first = 0;
    SparkTestStopServer(&context,thread);
    (void)unlink(socket_path);
    SparkTestStartServerReserve(&context,&thread,socket_path,8u * SPARK_TEST_KV_CHUNK_BYTES,3u * SPARK_TEST_KV_CHUNK_BYTES,3000u);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvAttach(first,0x31,SPARK_TEST_KV_CHUNK_BYTES + 1u,SPARK_TEST_KV_CHUNK_BYTES,4096u,&grant) == SPARK_STATUS_OK);
    assert(grant.reattached == 0u && grant.pool_generation != 0u && grant.chunk_capacity == 2u && grant.chunk_count == 2u && grant.chunk_bytes == SPARK_TEST_KV_CHUNK_BYTES);
    assert(grant.device_bytes == 2u * SPARK_TEST_KV_CHUNK_BYTES && grant.metadata_bytes == 4096u && grant.metadata_fd >= 0);
    assert(grant.kv_reserve_bytes == 3u * SPARK_TEST_KV_CHUNK_BYTES && grant.kv_committed_bytes == 2u * SPARK_TEST_KV_CHUNK_BYTES && grant.write_budget_bytes_per_day == 1000u);
    generation = grant.pool_generation;
    SparkTestKvExportCloses(first,generation,1u,1u,SPARK_STATUS_OK);
    SparkTestKvExportCloses(first,generation,1u,2u,SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestKvExportCloses(first,generation + 99u,0u,1u,SPARK_STATUS_NOT_FOUND);
    assert(SparkWeightdClientKvPoolResize(first,generation,3u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkWeightdClientKvPoolResize(first,generation,1u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 1u && state.wanted_chunks == 0u && state.kv_committed_bytes == SPARK_TEST_KV_CHUNK_BYTES);
    SparkTestKvExportCloses(first,generation,1u,1u,SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkWeightdClientKvPoolResize(first,generation,2u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 2u && state.wanted_chunks == 0u && state.kv_committed_bytes == 2u * SPARK_TEST_KV_CHUNK_BYTES);
    SparkTestKvExportCloses(first,generation,0u,2u,SPARK_STATUS_OK);
    assert(pwrite(grant.metadata_fd,"kvmeta",6u,0) == 6);
    SparkTestKvGrantClose(&grant);
    assert(SparkWeightdServerKvPoolCount(context.server) == 1u && SparkWeightdServerKvCommittedBytes(context.server) == 2u * SPARK_TEST_KV_CHUNK_BYTES);

    SparkTestConnect(&second,socket_path,0ull);
    assert(SparkTestKvAttach(second,0x31,SPARK_TEST_KV_CHUNK_BYTES + 1u,SPARK_TEST_KV_CHUNK_BYTES,4096u,&other) == SPARK_STATUS_BUSY);
    assert(SparkTestKvAttach(second,0x00,SPARK_TEST_KV_CHUNK_BYTES,0u,0u,&other) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkTestKvAttach(second,0x33,SPARK_TEST_KV_CHUNK_BYTES,0u,SPARK_WEIGHTD_KV_POOL_METADATA_BYTES_MAX + 1u,&other) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkTestKvAttach(second,0x33,SPARK_TEST_KV_CHUNK_BYTES,2u * SPARK_TEST_KV_CHUNK_BYTES,0u,&other) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkTestKvAttachSized(second,0x33,SPARK_TEST_KV_CHUNK_BYTES,0u,SPARK_TEST_KV_CHUNK_BYTES + 4096u,0u,&other) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkTestKvAttachSized(second,0x33,(uint64_t)(SPARK_WEIGHTD_KV_POOL_CHUNKS_MAX + 1u) * 65536u,0u,65536u,0u,&other) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(SparkTestKvAttach(second,0x33,4u * SPARK_TEST_KV_CHUNK_BYTES,4u * SPARK_TEST_KV_CHUNK_BYTES,0u,&other) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkTestKvAttach(second,0x32,2u * SPARK_TEST_KV_CHUNK_BYTES,2u * SPARK_TEST_KV_CHUNK_BYTES,0u,&other) == SPARK_STATUS_OK);
    assert(other.reattached == 0u && other.chunk_capacity == 2u && other.chunk_count == 1u && other.kv_committed_bytes == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    other_generation = other.pool_generation;
    assert(SparkWeightdClientKvPoolResize(second,other_generation,2u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 1u && state.wanted_chunks == 1u);
    assert(SparkWeightdClientKvPoolStatus(first,generation,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 2u && state.chunk_capacity == 2u && state.reclaim_wanted_bytes == SPARK_TEST_KV_CHUNK_BYTES && state.kv_committed_bytes == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    assert(SparkWeightdClientKvPoolStatus(second,generation,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_NOT_FOUND);
    assert(SparkWeightdClientKvPoolResize(first,generation,1u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 1u && state.kv_committed_bytes == 2u * SPARK_TEST_KV_CHUNK_BYTES);
    assert(SparkWeightdClientKvPoolResize(second,other_generation,2u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
    assert(state.chunk_count == 2u && state.wanted_chunks == 0u && state.kv_committed_bytes == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    assert(SparkWeightdClientKvPoolStatus(first,generation,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK && state.reclaim_wanted_bytes == 0u);
    SparkTestKvGrantClose(&other);
    SparkWeightdClientClose(second);
    second = 0;

    SparkWeightdClientClose(first);
    first = 0;
    SparkTestConnect(&second,socket_path,0ull);
    for (index = 0u; index < 500u; index++)
    {
        SparkStatus status = SparkTestKvAttach(second,0x31,SPARK_TEST_KV_CHUNK_BYTES + 1u,SPARK_TEST_KV_CHUNK_BYTES,4096u,&grant);
        if (status == SPARK_STATUS_OK)
            break;
        assert(status == SPARK_STATUS_BUSY);
        usleep(2000u);
    }
    assert(grant.reattached == 1u && grant.pool_generation == generation && grant.chunk_count == 1u && grant.metadata_fd >= 0);
    assert(pread(grant.metadata_fd,probe,6u,0) == 6 && memcmp(probe,"kvmeta",6u) == 0);
    SparkTestKvGrantClose(&grant);
    SparkWeightdClientClose(second);
    SparkTestConnect(&second,socket_path,0ull);
    for (index = 0u; index < 500u; index++)
    {
        SparkStatus status = SparkTestKvAttach(second,0x31,SPARK_TEST_KV_CHUNK_BYTES,SPARK_TEST_KV_CHUNK_BYTES,4096u,&grant);
        if (status == SPARK_STATUS_OK)
            break;
        assert(status == SPARK_STATUS_BUSY);
        usleep(2000u);
    }
    assert(grant.reattached == 0u && grant.pool_generation > generation && grant.chunk_count == 1u);
    assert(pread(grant.metadata_fd,probe,6u,0) == 6 && memcmp(probe,"kvmeta",6u) != 0);
    assert(grant.kv_committed_bytes == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    SparkTestKvGrantClose(&grant);
    SparkWeightdClientClose(second);
    second = 0;

    SparkTestConnect(&third,socket_path,0ull);
    for (index = 0u; index < 500u; index++)
    {
        SparkStatus status = SparkTestKvAttach(third,0x34,3u * SPARK_TEST_KV_CHUNK_BYTES,3u * SPARK_TEST_KV_CHUNK_BYTES,0u,&grant);
        if (status == SPARK_STATUS_OK)
            break;
        assert(status == SPARK_STATUS_BUSY);
        usleep(2000u);
    }
    for (index = 0u; index < 500u && grant.chunk_count < 3u; index++)
    {
        assert(SparkWeightdClientKvPoolResize(third,grant.pool_generation,3u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
        grant.chunk_count = state.chunk_count;
        usleep(2000u);
    }
    assert(grant.reattached == 0u && grant.chunk_count == 3u && SparkWeightdServerKvCommittedBytes(context.server) == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    assert(SparkWeightdServerKvPoolCount(context.server) == 1u);
    SparkTestKvGrantClose(&grant);
    SparkWeightdClientClose(third);
    third = 0;
    SparkTestStopServer(&context,thread);

    (void)unlink(socket_path);
    SparkTestStartServer(&context,&thread,socket_path,8u * SPARK_TEST_KV_CHUNK_BYTES);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvAttach(first,0x35,SPARK_TEST_KV_CHUNK_BYTES,0u,0u,&grant) == SPARK_STATUS_CAPACITY_EXCEEDED && grant.kv_reserve_bytes == 0u);
    SparkWeightdClientClose(first);
    first = 0;
    SparkTestStopServer(&context,thread);

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(pack_path,77u,SPARK_TEST_ARENA_BYTES,digest);
    SparkTestMakeIdentity(&identity,"weightd-test","kv-reserve",4u,0x77ull,digest,SPARK_TEST_ARENA_BYTES);
    SparkTestMakeRequest(&request,&identity,pack_path);
    (void)unlink(socket_path);
    SparkTestStartServerReserve(&context,&thread,socket_path,SPARK_TEST_CEILING_BYTES,SPARK_TEST_CEILING_BYTES - SPARK_TEST_ARENA_BYTES + 1u,0ull);
    SparkTestConnect(&first,socket_path,SPARK_TEST_CEILING_BYTES);
    SparkTestAttach(first,&request,&attach);
    assert(attach.status == SPARK_STATUS_CAPACITY_EXCEEDED && attach.arena_count == 0u);
    SparkWeightdClientClose(first);
    SparkTestStopServer(&context,thread);
    (void)unlink(socket_path);
    SparkTestStartServerReserve(&context,&thread,socket_path,SPARK_TEST_CEILING_BYTES,SPARK_TEST_CEILING_BYTES - SPARK_TEST_ARENA_BYTES,0ull);
    SparkTestConnect(&first,socket_path,SPARK_TEST_CEILING_BYTES);
    SparkTestAttach(first,&request,&attach);
    assert(attach.status == SPARK_STATUS_OK && attach.arena_count == 1u);
    SparkWeightdClientClose(first);
    SparkTestStopServer(&context,thread);
    printf("weightd kv pools: take the free reserve up to the pool size at attach and resize by chunk within it, a short grow records the wanted chunks and other pools see them as reclaim, a shrink releases chunks to the waiting pool, export only granted chunks, one owner, reattach keeps generation and metadata, a resized pool is new, detached pools evicted under pressure, no reserve or no write budget refuses, the write budget is shared by pool size, weight arenas admitted under the ceiling minus the reserve\n");
}

static SparkStatus SparkTestKvSharedAttach(SparkWeightdClient *client,uint8_t key_byte,uint8_t layout_byte,uint64_t page_bytes,SparkWeightdKvSharedGrant *grant)
{
    SparkWeightdKvSharedRequest request;
    memset(&request,0,sizeof(request));
    memset(request.key,key_byte,sizeof(request.key));
    memset(request.layout_sha256,layout_byte,sizeof(request.layout_sha256));
    request.chunk_bytes = SPARK_TEST_KV_CHUNK_BYTES;
    request.page_bytes = page_bytes;
    request.alignment_pages = page_bytes >= SPARK_TEST_KV_CHUNK_BYTES ? 1u : (uint32_t)(SPARK_TEST_KV_CHUNK_BYTES / page_bytes);
    request.label = "kvshared-test";
    return(SparkWeightdClientKvSharedAttach(client,&request,grant,SPARK_TEST_TIMEOUT_NS));
}

#define SparkTestAwait(condition) do { uint32_t spark_test_waited; for (spark_test_waited = 0u; !(condition) && spark_test_waited < 5000u; spark_test_waited++) { struct timespec spark_test_pause = {0,1000000}; (void)nanosleep(&spark_test_pause,0); } assert(condition); } while (0)

static void SparkTestKvSharedPools(void)
{
    const char *socket_path = "/tmp/spark_weightd_test_kv_shared.sock";
    const uint64_t page_bytes = SPARK_TEST_KV_CHUNK_BYTES / 4u;
    uint8_t layout[SPARK_KV_SHARED_INDEX_LAYOUT_BYTES],identity[32];
    SparkTestServerThread context;
    pthread_t thread;
    SparkWeightdClient *first = 0,*second = 0,*third = 0;
    SparkWeightdKvSharedGrant grant,joined,refused,again;
    SparkWeightdKvPoolGrant private_grant;
    SparkWeightdKvPoolState state;
    SparkKvSharedIndex owner,member;
    void *owner_map,*member_map;
    uint64_t generation;
    uint32_t ready,writing;

    memset(layout,0x6c,sizeof(layout));
    memset(identity,0x21,sizeof(identity));
    (void)unlink(socket_path);
    spark_test_kv_shared_window_bytes = 0u;
    SparkTestStartServerReserve(&context,&thread,socket_path,8u * SPARK_TEST_KV_CHUNK_BYTES,4u * SPARK_TEST_KV_CHUNK_BYTES,3000u);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvSharedAttach(first,0x51,0x6c,page_bytes,&refused) == SPARK_STATUS_UNSUPPORTED && SparkWeightdServerKvPoolCount(context.server) == 0u);
    SparkWeightdClientClose(first);
    first = 0;
    SparkTestStopServer(&context,thread);
    (void)unlink(socket_path);
    spark_test_kv_shared_window_bytes = 2u * SPARK_TEST_KV_CHUNK_BYTES + page_bytes;
    SparkTestStartServerReserve(&context,&thread,socket_path,8u * SPARK_TEST_KV_CHUNK_BYTES,4u * SPARK_TEST_KV_CHUNK_BYTES,3000u);
    SparkTestConnect(&first,socket_path,0ull);
    SparkTestConnect(&second,socket_path,0ull);
    SparkTestConnect(&third,socket_path,0ull);
    assert(SparkTestKvSharedAttach(first,0x51,0x6c,4u * SPARK_TEST_KV_CHUNK_BYTES,&refused) == SPARK_STATUS_UNSUPPORTED);
    assert(SparkTestKvSharedAttach(first,0x51,0x6c,page_bytes,&grant) == SPARK_STATUS_OK);
    assert(grant.created == 1u && grant.holder == 0u && grant.slot_count == 8u && grant.chunk_count == 2u && grant.device_bytes == 2u * SPARK_TEST_KV_CHUNK_BYTES);
    assert(grant.metadata_bytes == SparkKvSharedIndexBytes(8u) && grant.metadata_fd >= 0);
    assert(SparkTestKvSharedAttach(second,0x51,0x6c,page_bytes,&joined) == SPARK_STATUS_OK);
    assert(joined.created == 0u && joined.holder == 1u && joined.slot_count == 8u && joined.pool_generation == grant.pool_generation);
    assert(SparkTestKvSharedAttach(second,0x51,0x6c,page_bytes,&refused) == SPARK_STATUS_CAPACITY_EXCEEDED);
    assert(SparkTestKvSharedAttach(third,0x51,0x6d,page_bytes,&refused) == SPARK_STATUS_SCHEMA_ERROR);
    assert(SparkTestKvSharedAttach(third,0x51,0x6c,page_bytes * 2u,&refused) == SPARK_STATUS_SCHEMA_ERROR);
    assert(SparkTestKvAttach(third,0x51,SPARK_TEST_KV_CHUNK_BYTES,SPARK_TEST_KV_CHUNK_BYTES,0u,&private_grant) == SPARK_STATUS_BUSY);
    SparkTestKvExportCloses(second,grant.pool_generation,0u,2u,SPARK_STATUS_OK);
    SparkTestKvExportCloses(third,grant.pool_generation,0u,1u,SPARK_STATUS_NOT_FOUND);
    assert(SparkWeightdClientKvPoolResize(first,grant.pool_generation,1u,&state,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_INVALID_ARGUMENT);
    owner_map = mmap(0,(size_t)grant.metadata_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,grant.metadata_fd,0);
    member_map = mmap(0,(size_t)joined.metadata_bytes,PROT_READ | PROT_WRITE,MAP_SHARED,joined.metadata_fd,0);
    assert(owner_map != MAP_FAILED && member_map != MAP_FAILED);
    assert(SparkKvSharedIndexAttach(&owner,owner_map,grant.metadata_bytes,grant.holder,page_bytes,layout) == SPARK_STATUS_OK && owner.slot_count == 8u);
    assert(SparkKvSharedIndexAttach(&member,member_map,joined.metadata_bytes,joined.holder,page_bytes,layout) == SPARK_STATUS_OK);
    assert(SparkKvSharedIndexReserve(&member,&ready,&generation) == SPARK_STATUS_OK);
    assert(SparkKvSharedIndexPublish(&member,ready,identity,64u,SPARK_KV_SHARED_NO_SLOT,0u) == SPARK_STATUS_OK);
    assert(SparkKvSharedIndexReserve(&member,&writing,&generation) == SPARK_STATUS_OK);
    assert(atomic_load(&owner.slots[ready].state) == SPARK_KV_SHARED_SLOT_READY && atomic_load(&owner.slots[writing].state) == SPARK_KV_SHARED_SLOT_WRITING);
    SparkWeightdClientClose(second);
    second = 0;
    SparkTestAwait(atomic_load(&owner.slots[writing].state) == SPARK_KV_SHARED_SLOT_FREE);
    SparkTestConnect(&second,socket_path,0ull);
    assert(SparkTestKvSharedAttach(second,0x51,0x6c,page_bytes,&refused) == SPARK_STATUS_OK && refused.holder == 1u);
    assert(atomic_load(&owner.slots[writing].state) == SPARK_KV_SHARED_SLOT_FREE);
    assert(atomic_load(&owner.slots[ready].state) == SPARK_KV_SHARED_SLOT_READY && atomic_load(&owner.slots[ready].holders) == 0u);
    (void)close(refused.metadata_fd);
    SparkWeightdClientClose(second);
    second = 0;
    SparkTestAwait(__builtin_popcountll(atomic_load(&owner.slots[ready].holders)) == 0 && SparkWeightdServerKvPoolCount(context.server) == 1u);
    assert(SparkWeightdServerKvCommittedBytes(context.server) == 2u * SPARK_TEST_KV_CHUNK_BYTES);
    SparkWeightdClientClose(first);
    first = 0;
    SparkTestAwait(SparkWeightdServerKvPoolCount(context.server) == 1u && SparkWeightdServerKvCommittedBytes(context.server) == 2u * SPARK_TEST_KV_CHUNK_BYTES);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvSharedAttach(first,0x51,0x6c,page_bytes,&again) == SPARK_STATUS_OK && again.created == 0u && again.pool_generation == grant.pool_generation);
    assert(atomic_load(&owner.slots[ready].state) == SPARK_KV_SHARED_SLOT_READY);
    (void)close(again.metadata_fd);
    SparkWeightdClientClose(first);
    first = 0;
    SparkTestAwait(SparkWeightdServerKvPoolCount(context.server) == 1u);
    SparkTestConnect(&first,socket_path,0ull);
    assert(SparkTestKvAttach(first,0x52,3u * SPARK_TEST_KV_CHUNK_BYTES,3u * SPARK_TEST_KV_CHUNK_BYTES,0u,&private_grant) == SPARK_STATUS_OK);
    assert(SparkWeightdServerKvPoolCount(context.server) == 1u && SparkWeightdServerKvCommittedBytes(context.server) == 3u * SPARK_TEST_KV_CHUNK_BYTES);
    (void)munmap(owner_map,(size_t)grant.metadata_bytes);
    (void)munmap(member_map,(size_t)joined.metadata_bytes);
    (void)close(grant.metadata_fd);
    (void)close(joined.metadata_fd);
    SparkTestKvGrantClose(&private_grant);
    SparkWeightdClientClose(first);
    SparkWeightdClientClose(third);
    SparkTestStopServer(&context,thread);
    spark_test_kv_shared_window_bytes = 0u;
    (void)unlink(socket_path);
    printf("weightd kv shared pools: no window refuses, the window is sized in whole alignment groups, the first holder creates the whole pool and formats the index, later holders join with the same page and layout only, holders export but never resize, a private key cannot collide, a leaving holder's bits and writing slots are cleared, the pool outlives its last holder with its published slots for the next attach, and a private pool that needs the room evicts it\n");
}

int main(void)
{
    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestIdentityCanonicalization();
    SparkTestSharedRefcountAndConsumerDeath();
    SparkTestStopAttachStartNeverHoldsTwoArenas();
    SparkTestReclaimPackIsScopedToOnePack();
    SparkTestServedAbiVersions();
    SparkTestMeshStatus();
    SparkTestFailClosedPaths();
    SparkTestKvPools();
    SparkTestKvSharedPools();
    SparkTestDaemonProcessTermPath();
    printf("w2 weightd lane: identity arenas + NO-2x + TERM green\n");
    return 0;
}
