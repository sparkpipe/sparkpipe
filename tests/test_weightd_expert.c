
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
    uint8_t record[48];
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
    memcpy(header + 4u, &(uint32_t){SPARK_WEIGHTD_RANGE_MANIFEST_VERSION}, 4u);
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
        memcpy(record + 16u, &offset, 8u);
        memcpy(record + 24u, &expert_bytes, 8u);
        memcpy(record + 32u, digest, 16u);
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

static void SparkTestAcquireReadRelease(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,uint32_t layer,uint32_t expert,uint32_t seed,SparkStatus expected_status)
{
	SparkWeightdExpertKey key = {layer,expert};
	SparkWeightdWorkingSetResult result,released;
	static uint8_t expected[SPARK_TEST_EXPERT_BYTES],observed[SPARK_TEST_EXPERT_BYTES];
	assert(SparkWeightdClientAcquire(client,attached->arena_generation,&key,1u,&result,SPARK_TEST_TIMEOUT_NS) == expected_status);
	assert(result.status == expected_status);
	if ( expected_status != SPARK_STATUS_OK )
	{
		assert(result.lease_identifier == 0u);
		return;
	}
	assert(result.lease_identifier != 0u && result.resident_bytes <= SPARK_TEST_CEILING_BYTES);
	// Same-process CUDA-stub oracle; real consumer imports are tested separately.
	assert(cudaMemcpy(observed,(const void *)(uintptr_t)(attached->device_handle + (expert * SPARK_TEST_EXPERT_BYTES)),sizeof(observed),cudaMemcpyDeviceToHost) == cudaSuccess);
	SparkTestFillExpert(seed,layer,expert,expected);
	assert(memcmp(observed,expected,sizeof(observed)) == 0);
	assert(SparkWeightdClientRelease(client,attached->arena_generation,result.lease_identifier,&released,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_OK);
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

    assert(rename("/tmp/spark_weightd_expert_a.bin.experts",
        "/tmp/spark_weightd_expert_a.bin.experts.saved") == 0);
    SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
        SPARK_TEST_POOL_BYTES, &attach_a);
    assert(attach_a.status == SPARK_STATUS_NOT_FOUND);
    assert(attach_a.resident_bytes == 0ull && attach_a.arena_count == 0u);
    {
        FILE *invalid = fopen("/tmp/spark_weightd_expert_a.bin.experts","wb");
        assert(invalid != 0);
        assert(fwrite("bad",1u,3u,invalid) == 3u);
        assert(fclose(invalid) == 0);
    }
    SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
        SPARK_TEST_POOL_BYTES, &attach_a);
    assert(attach_a.status == SPARK_STATUS_PARSE_ERROR);
    assert(attach_a.resident_bytes == 0ull && attach_a.arena_count == 0u);
    assert(rename("/tmp/spark_weightd_expert_a.bin.experts.saved",
        "/tmp/spark_weightd_expert_a.bin.experts") == 0);
    printf("missing and corrupt expert manifests fail before residency green\n");

    SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
        SPARK_TEST_POOL_BYTES, &attach_a);
    assert(attach_a.status == SPARK_STATUS_OK);
    assert(attach_a.expert_count == SPARK_TEST_EXPERTS_PER_MODEL);
    {
        SparkWeightdLazyAttachResult refused;
        assert(rename("/tmp/spark_weightd_expert_a.bin.experts",
            "/tmp/spark_weightd_expert_a.bin.experts.saved") == 0);
        SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
            SPARK_TEST_POOL_BYTES, &refused);
        assert(refused.status == SPARK_STATUS_NOT_FOUND);
        assert(rename("/tmp/spark_weightd_expert_a.bin.experts.saved",
            "/tmp/spark_weightd_expert_a.bin.experts") == 0);
        SparkTestLazyAttach(client, &identity_a, "/tmp/spark_weightd_expert_a.bin",
            SPARK_TEST_POOL_BYTES + 1u, &refused);
        assert(refused.status == SPARK_STATUS_INVALID_ARGUMENT);
    }
    assert(attach_a.resident_bytes == 0ull);
    SparkTestLazyAttach(client, &identity_b, "/tmp/spark_weightd_expert_b.bin",
        SPARK_TEST_POOL_BYTES, &attach_b);
    assert(attach_b.status == SPARK_STATUS_OK);
    assert(attach_b.arena_count == 2u);
    assert(attach_b.resident_bytes == 0ull);
    printf("lazy attach two models zero resident green\n");

    assert(SparkWeightdClientEnsure(client,attach_a.arena_generation,2u,0u,&ensure,SPARK_TEST_TIMEOUT_NS) == SPARK_STATUS_UNSUPPORTED);
    SparkTestAcquireReadRelease(client,&attach_a,2u,0u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,0u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,1u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_b,7u,0u,29u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,2u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,3u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,4u,11u,SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_a,2u,0u,11u,SPARK_STATUS_OK);
    printf("leased byte-exact reads across models and pool eviction pass\n");
    SparkTestLazyAttach(client,&identity_c,"/tmp/spark_weightd_expert_c.bin",SPARK_TEST_POOL_BYTES,&attach_c);
    assert(attach_c.status == SPARK_STATUS_OK);
    SparkTestAcquireReadRelease(client,&attach_c,5u,5u,47u,SPARK_STATUS_HASH_MISMATCH);
    SparkTestAcquireReadRelease(client,&attach_c,5u,6u,47u,SPARK_STATUS_OK);
    SparkTestRewritePackGrown("/tmp/spark_weightd_expert_a.bin",11u,2u);
    SparkTestAcquireReadRelease(client,&attach_a,2u,1u,11u,SPARK_STATUS_HASH_MISMATCH);
    SparkTestAcquireReadRelease(client,&attach_a,99u,0u,11u,SPARK_STATUS_NOT_FOUND);
    attach_a.arena_generation = 0xDEADBEEFull;
    SparkTestAcquireReadRelease(client,&attach_a,2u,0u,11u,SPARK_STATUS_NOT_FOUND);
    printf("stale checksum, pack drift, unknown expert and generation rejected\n");

    SparkWeightdClientClose(client);
    SparkTestStopServer(&server, server_thread);

    printf("weightd v2 lazy-expert lease regression PASS\n");
    return 0;
}
