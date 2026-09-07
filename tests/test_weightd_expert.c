
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "cuda_runtime.h"
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd.h"

#define SPARK_TEST_EXPERT_BYTES (512ull * 1024ull)
#define SPARK_TEST_EXPERTS_PER_MODEL 8u
#define SPARK_TEST_ARENA_BYTES \
    (SPARK_TEST_EXPERT_BYTES * SPARK_TEST_EXPERTS_PER_MODEL)
#define SPARK_TEST_CEILING_BYTES (8ull * 1024ull * 1024ull)
#define SPARK_TEST_POOL_BYTES (5ull * 512ull * 1024ull)
#define SPARK_TEST_CHUNK_BYTES (2ull * 1024ull * 1024ull)
#define SPARK_TEST_TIMEOUT_NS 10000000000ull

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
    const char *socket_path)
{
    SparkWeightdServerConfig config;
    memset(&config, 0, sizeof(config));
    config.socket_path = socket_path;
    config.device_bytes_max = SPARK_TEST_CEILING_BYTES;
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

static uint8_t SparkTestExpertByte(uint32_t model_seed,
    uint32_t layer,
    uint32_t expert,
    uint64_t index)
{
    uint64_t mixed = model_seed * 2654435761ull + layer * 40503ull +
        expert * 2246822519ull + index * 3266489917ull;
    return (uint8_t)((mixed >> 24) ^ (mixed >> 7));
}

static void SparkTestFillExpert(uint32_t model_seed,
    uint32_t layer,
    uint32_t expert,
    uint8_t *buffer)
{
    uint64_t index;
    for (index = 0; index < SPARK_TEST_EXPERT_BYTES; index++)
    {
        buffer[index] = SparkTestExpertByte(model_seed, layer, expert, index);
    }
}

static void SparkTestWritePack(const char *path,
    uint32_t model_seed,
    uint32_t layer,
    uint32_t stale_digest_expert,
    char sha_hex[SPARK_SHA256_HEX_BYTES])
{
    static uint8_t staging[SPARK_TEST_EXPERT_BYTES];
    uint8_t header[16];
    uint8_t record[40];
    uint8_t digest[16];
    SparkCk128Context context;
    FILE *file;
    FILE *manifest;
    char manifest_path[1024];
    uint32_t expert;
    uint32_t digest_seed;

    file = fopen(path, "wb");
    assert(file != 0);
    for (expert = 0u; expert < SPARK_TEST_EXPERTS_PER_MODEL; expert++)
    {
        SparkTestFillExpert(model_seed, layer, expert, staging);
        assert(fwrite(staging, 1u, (size_t)SPARK_TEST_EXPERT_BYTES, file) ==
            (size_t)SPARK_TEST_EXPERT_BYTES);
    }
    assert(fclose(file) == 0);
    assert(SparkSha256File(path, sha_hex) == SPARK_STATUS_OK);

    assert(snprintf(manifest_path, sizeof(manifest_path), "%s.experts",
        path) > 0);
    manifest = fopen(manifest_path, "wb");
    assert(manifest != 0);
    memset(header, 0, sizeof(header));
    memcpy(header + 0u, &(uint32_t){SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC}, 4u);
    memcpy(header + 4u, &(uint32_t){SPARK_WEIGHTD_EXPERT_MANIFEST_VERSION}, 4u);
    memcpy(header + 8u, &(uint32_t){SPARK_TEST_EXPERTS_PER_MODEL}, 4u);
    assert(fwrite(header, 1u, sizeof(header), manifest) == sizeof(header));
    for (expert = 0u; expert < SPARK_TEST_EXPERTS_PER_MODEL; expert++)
    {
        uint64_t offset = (uint64_t)expert * SPARK_TEST_EXPERT_BYTES;
        uint64_t expert_bytes = SPARK_TEST_EXPERT_BYTES;
        digest_seed = expert == stale_digest_expert ? model_seed + 1u
            : model_seed;
        SparkTestFillExpert(digest_seed, layer, expert, staging);
        SparkCk128Initialize(&context);
        SparkCk128Update(&context, staging, (size_t)SPARK_TEST_EXPERT_BYTES);
        SparkCk128Finalize(&context, digest);
        memset(record, 0, sizeof(record));
        memcpy(record + 0u, &layer, 4u);
        memcpy(record + 4u, &expert, 4u);
        memcpy(record + 8u, &offset, 8u);
        memcpy(record + 16u, &expert_bytes, 8u);
        memcpy(record + 24u, digest, 16u);
        assert(fwrite(record, 1u, sizeof(record), manifest) == sizeof(record));
    }
    assert(fclose(manifest) == 0);
}

static void SparkTestRewritePackGrown(const char *path,
    uint32_t model_seed,
    uint32_t layer)
{
    static uint8_t staging[SPARK_TEST_EXPERT_BYTES];
    FILE *file;
    uint32_t expert;
    uint8_t extra = 0x5Au;

    file = fopen(path, "wb");
    assert(file != 0);
    for (expert = 0u; expert < SPARK_TEST_EXPERTS_PER_MODEL; expert++)
    {
        SparkTestFillExpert(model_seed, layer, expert, staging);
        assert(fwrite(staging, 1u, (size_t)SPARK_TEST_EXPERT_BYTES, file) ==
            (size_t)SPARK_TEST_EXPERT_BYTES);
    }
    assert(fwrite(&extra, 1u, 1u, file) == 1u);
    assert(fclose(file) == 0);
}

static void SparkTestMakeIdentity(SparkWeightdIdentity *identity,
    const char *model,
    const char *pack_sha256,
    uint64_t arena_bytes)
{
    memset(identity, 0, sizeof(*identity));
    snprintf(identity->model, SPARK_WEIGHTD_ID_BYTES, "%s", model);
    snprintf(identity->revision, SPARK_WEIGHTD_REVISION_BYTES, "r1");
    identity->topology = 16u;
    identity->abi_version = SPARK_WEIGHTD_IPC_ABI_VERSION;
    identity->geometry_fingerprint = 0x504F4355ull;
    memcpy(identity->pack_sha256, pack_sha256,
        strlen(pack_sha256) + 1u);
    identity->arena_bytes = arena_bytes;
    assert(SparkWeightdIdentityPrepare(identity) == SPARK_STATUS_OK);
}

static void SparkTestLazyAttach(SparkWeightdClient *client,
    const SparkWeightdIdentity *identity,
    const char *pack_path,
    uint64_t pool_bytes,
    SparkWeightdLazyAttachResult *result)
{
    SparkWeightdLazyAttachRequest request;
    memset(&request, 0, sizeof(request));
    request.identity = *identity;
    snprintf(request.pack_path, SPARK_WEIGHTD_PATH_BYTES, "%s", pack_path);
    request.expert_pool_bytes = pool_bytes;
    memset(result, 0, sizeof(*result));
    assert(SparkWeightdClientAttachLazy(client, &request, result,
        SPARK_TEST_TIMEOUT_NS) == result->status);
}

static void SparkTestEnsure(SparkWeightdClient *client,
    uint64_t generation,
    uint32_t layer,
    uint32_t expert,
    SparkWeightdEnsureResult *result)
{
    memset(result, 0, sizeof(*result));
    assert(SparkWeightdClientEnsure(client, generation, layer, expert,
        result, SPARK_TEST_TIMEOUT_NS) == result->status);
}

static void SparkTestReadback(uint64_t device_handle,
    const SparkWeightdEnsureResult *ensure,
    uint32_t model_seed,
    uint32_t layer,
    uint32_t expert)
{
    static uint8_t expected[SPARK_TEST_EXPERT_BYTES];
    static uint8_t observed[SPARK_TEST_EXPERT_BYTES];
    assert(ensure->status == SPARK_STATUS_OK);
    assert(cudaMemcpy(observed,
            (const void *)(uintptr_t)(device_handle + ensure->device_offset),
            (size_t)ensure->expert_bytes, cudaMemcpyDeviceToHost) ==
        cudaSuccess);
    SparkTestFillExpert(model_seed, layer, expert, expected);
    assert(memcmp(observed, expected, (size_t)ensure->expert_bytes) == 0);
}

int main(void)
{
    static const char *const socket_path = "/tmp/spark_weightd_expert_test.sock";
    static char hex_a[SPARK_SHA256_HEX_BYTES];
    static char hex_b[SPARK_SHA256_HEX_BYTES];
    static char hex_c[SPARK_SHA256_HEX_BYTES];
    SparkTestServerThread server;
    pthread_t server_thread;
    SparkWeightdHelloResult hello;
    SparkWeightdLazyAttachResult attach_a;
    SparkWeightdLazyAttachResult attach_b;
    SparkWeightdLazyAttachResult attach_c;
    SparkWeightdEnsureResult ensure;
    SparkWeightdIdentity identity_a;
    SparkWeightdIdentity identity_b;
    SparkWeightdIdentity identity_c;
    SparkWeightdClient *client;

    (void)unlink(socket_path);
    spark_stub_cuda_reset_faults();

    SparkTestWritePack("/tmp/spark_weightd_expert_a.bin", 11u, 2u,
        SPARK_TEST_EXPERTS_PER_MODEL, hex_a);
    SparkTestWritePack("/tmp/spark_weightd_expert_b.bin", 29u, 7u,
        SPARK_TEST_EXPERTS_PER_MODEL, hex_b);
    SparkTestWritePack("/tmp/spark_weightd_expert_c.bin", 47u, 5u, 5u, hex_c);
    SparkTestMakeIdentity(&identity_a, "poc-model-a", hex_a,
        SPARK_TEST_ARENA_BYTES);
    SparkTestMakeIdentity(&identity_b, "poc-model-b", hex_b,
        SPARK_TEST_ARENA_BYTES);
    SparkTestMakeIdentity(&identity_c, "poc-model-c", hex_c,
        SPARK_TEST_ARENA_BYTES);

    SparkTestStartServer(&server, &server_thread, socket_path);
    memset(&hello, 0, sizeof(hello));
    assert(SparkWeightdClientConnect(socket_path, &client, &hello) ==
        SPARK_STATUS_OK);
    assert(hello.status == SPARK_STATUS_OK);

    SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
        SPARK_TEST_POOL_BYTES, &attach_a);
    assert(attach_a.status == SPARK_STATUS_OK);
    assert(attach_a.expert_count == SPARK_TEST_EXPERTS_PER_MODEL);
    assert(attach_a.resident_bytes == 0ull);
    SparkTestLazyAttach(client, &identity_b, "/tmp/spark_weightd_expert_b.bin",
        SPARK_TEST_POOL_BYTES, &attach_b);
    assert(attach_b.status == SPARK_STATUS_OK);
    assert(attach_b.arena_count == 2u);
    assert(attach_b.resident_bytes == 0ull);
    printf("lazy attach two models zero resident green\n");

    SparkTestEnsure(client, attach_a.arena_generation, 2u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK);
    assert(ensure.loaded == 1u);
    assert(ensure.device_offset == 0ull);
    assert(ensure.resident_bytes == SPARK_TEST_CHUNK_BYTES);
    assert(ensure.load_ns < SPARK_TEST_TIMEOUT_NS);
    SparkTestReadback(attach_a.device_handle, &ensure, 11u, 2u, 0u);
    printf("cold ensure fault-in byte-exact green\n");

    SparkTestEnsure(client, attach_a.arena_generation, 2u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK);
    assert(ensure.loaded == 0u);
    assert(ensure.resident_bytes == SPARK_TEST_CHUNK_BYTES);
    printf("warm ensure no-reload green\n");

    SparkTestEnsure(client, attach_a.arena_generation, 2u, 1u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    assert(ensure.resident_bytes == SPARK_TEST_CHUNK_BYTES);
    SparkTestEnsure(client, attach_b.arena_generation, 7u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    SparkTestReadback(attach_b.device_handle, &ensure, 29u, 7u, 0u);
    assert(ensure.resident_bytes == 2ull * SPARK_TEST_CHUNK_BYTES);
    printf("cross-model co-residency shared-chunk green\n");

    SparkTestEnsure(client, attach_a.arena_generation, 2u, 0u, &ensure);
    assert(ensure.loaded == 0u);
    SparkTestEnsure(client, attach_a.arena_generation, 2u, 2u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    assert(ensure.resident_bytes == 2ull * SPARK_TEST_CHUNK_BYTES);
    SparkTestEnsure(client, attach_a.arena_generation, 2u, 3u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    assert(ensure.resident_bytes == 2ull * SPARK_TEST_CHUNK_BYTES);
    SparkTestEnsure(client, attach_a.arena_generation, 2u, 4u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    assert(ensure.resident_bytes == 2ull * SPARK_TEST_CHUNK_BYTES);
    assert(ensure.device_offset == 4ull * SPARK_TEST_EXPERT_BYTES);
    SparkTestReadback(attach_a.device_handle, &ensure, 11u, 2u, 4u);
    SparkTestEnsure(client, attach_a.arena_generation, 2u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    printf("pool pressure lru evict reload green\n");

    SparkTestLazyAttach(client, &identity_c, "/tmp/spark_weightd_expert_c.bin",
        SPARK_TEST_POOL_BYTES, &attach_c);
    assert(attach_c.status == SPARK_STATUS_OK);
    SparkTestEnsure(client, attach_c.arena_generation, 5u, 5u, &ensure);
    assert(ensure.status == SPARK_STATUS_HASH_MISMATCH);
    assert(ensure.loaded == 0u);
    SparkTestEnsure(client, attach_c.arena_generation, 5u, 6u, &ensure);
    assert(ensure.status == SPARK_STATUS_OK && ensure.loaded == 1u);
    SparkTestReadback(attach_c.device_handle, &ensure, 47u, 5u, 6u);
    printf("stale digest fail-closed green\n");

    SparkTestRewritePackGrown("/tmp/spark_weightd_expert_a.bin", 11u, 2u);
    SparkTestEnsure(client, attach_a.arena_generation, 2u, 1u, &ensure);
    assert(ensure.status == SPARK_STATUS_HASH_MISMATCH);
    printf("pack drift fail-closed green\n");

    SparkTestEnsure(client, attach_a.arena_generation, 99u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestEnsure(client, 0xDEADBEEFull, 2u, 0u, &ensure);
    assert(ensure.status == SPARK_STATUS_NOT_FOUND);
    printf("unknown expert and generation fail-closed green\n");

    SparkWeightdClientClose(client);
    SparkTestStopServer(&server, server_thread);

    printf("weightd lazy-expert poc green\n");
    return 0;
}
