
#include <assert.h>
#include <pthread.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd_attach.h"
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_receipt.h"

#define SPARK_TEST_ARENA_BYTES (1024ull * 1024ull)
#define SPARK_TEST_CEILING_BYTES (1536ull * 1024ull)
#define SPARK_TEST_TIMEOUT_NS 10000000000ull
#define SPARK_TEST_SOCKET "/tmp/spark_weightd_attach_test.sock"
#define SPARK_TEST_PACK "/tmp/spark_weightd_attach_test.spack"

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
    pthread_t *thread_handle)
{
    SparkWeightdServerConfig config;
    memset(&config, 0, sizeof(config));
    config.socket_path = SPARK_TEST_SOCKET;
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
    slice->model = "weightd-attach-test";
    slice->revision = "test-rev";
    slice->topology = 4u;
    slice->geometry_fingerprint = 0x1234ull;
    slice->pack_bytes = pack_bytes;
}


static void SparkTestConfigurationGates(void)
{
    SparkWeightdPackSlice slice;
    SparkWeightdAttachOutcome outcome;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    char digest[SPARK_SHA256_HEX_BYTES];

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 11u, 65536ull, digest);
    SparkTestMakeSlice(&slice, 65536ull);
    SparkTestClearAttachEnv();

    assert(SparkWeightdAttachRequested() == SPARK_STATUS_BUSY);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    assert(SparkWeightdAttachRequested() == SPARK_STATUS_OK);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, "0");
    assert(SparkWeightdAttachRequested() == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, "typo");
    assert(SparkWeightdAttachRequested() == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, "");
    assert(SparkWeightdAttachRequested() == SPARK_STATUS_INVALID_ARGUMENT);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, 0);

    SparkTestClearAttachEnv();
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "no_socket") == 0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    SparkTestSetEnv(SPARK_WEIGHTD_SHARE_ENV, SPARK_WEIGHTD_SHARE_READONLY);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) == SPARK_STATUS_UNSUPPORTED);
    assert(outcome.client == 0);
    assert(strcmp(reason, "share_unsupported") == 0);
    SparkTestSetEnv(SPARK_WEIGHTD_SHARE_ENV, "typo");
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) == SPARK_STATUS_UNSUPPORTED);
    assert(outcome.client == 0);
    assert(strcmp(reason, "share_unsupported") == 0);
    SparkTestSetEnv(SPARK_WEIGHTD_SHARE_ENV, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, "0");
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "attach_config") == 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SWITCH, 0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    (void)unlink(SPARK_TEST_SOCKET);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "no_daemon") == 0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "no_identity") == 0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, "zz48");
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "identity") == 0);

    {
        SparkWeightdAttachOutcome ignored;
        char unused_reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
        assert(SparkWeightdAttachPack(0, SPARK_TEST_PACK,
            SPARK_TEST_TIMEOUT_NS, &ignored, unused_reason) ==
            SPARK_STATUS_INVALID_ARGUMENT);
        assert(SparkWeightdAttachPack(&slice, "", SPARK_TEST_TIMEOUT_NS,
            &ignored, unused_reason) == SPARK_STATUS_INVALID_ARGUMENT);
    }
    SparkWeightdAttachRelease(0);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, 0);
    (void)remove(SPARK_TEST_PACK);
    printf("attach explicit error gates green\n");
}


static void SparkTestRefusedAttachFailsAndAllocatesNothing(void)
{
    SparkWeightdPackSlice slice;
    SparkWeightdAttachOutcome outcome;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    uint32_t arena_count = 1u;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 22u, 65536ull, digest);
    SparkTestMakeSlice(&slice, 65536ull);

    SparkTestClearAttachEnv();
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,
        "1111111111111111111111111111111111111111111111111111111111111111");
    SparkTestStartServer(&thread_context, &thread_handle);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "hash_mismatch") == 0);

    {
        SparkWeightdPackSlice wrong_size;
        SparkTestMakeSlice(&wrong_size, 65537ull);
        SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
        assert(SparkWeightdAttachPack(&wrong_size, SPARK_TEST_PACK,
            SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
        assert(outcome.client == 0);
        assert(strcmp(reason, "invalid_argument") == 0);
    }
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    assert(SparkWeightdAttachPack(&slice, "/tmp/spark_weightd_attach_missing.spack",
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(outcome.client == 0);
    assert(strcmp(reason, "io_error") == 0);

    {
        SparkWeightdClient *probe = 0;
        SparkWeightdHelloResult hello;
        memset(&hello, 0, sizeof(hello));
        assert(SparkWeightdClientConnect(SPARK_TEST_SOCKET, &probe,
            &hello) == SPARK_STATUS_OK);
        arena_count = hello.arena_count;
        assert(hello.resident_bytes == 0ull);
        SparkWeightdClientClose(probe);
    }
    assert(arena_count == 0u);

    SparkTestStopServer(&thread_context, thread_handle);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, 0);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("refused attach returns error with nothing resident green\n");
}


static void SparkTestColdWarmAndRetention(void)
{
    SparkWeightdPackSlice slice;
    SparkWeightdAttachOutcome cold;
    SparkWeightdAttachOutcome warm;
    SparkWeightdAttachOutcome again;
    SparkWeightdAttachOutcome revisioned;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    char digest[SPARK_SHA256_HEX_BYTES];
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    FILE *pack_file = 0;
    static uint8_t pack_bytes[65536];
    uint8_t arena_probe[65536];

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 33u, sizeof(pack_bytes), digest);
    SparkTestMakeSlice(&slice, sizeof(pack_bytes));

    SparkTestClearAttachEnv();
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    SparkTestStartServer(&thread_context, &thread_handle);

    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &cold, reason) == SPARK_STATUS_OK);
    assert(cold.client != 0);
    assert(reason[0] == '\0');
    assert(cold.loaded_from_pack == 1u);
    assert(cold.refcount == 1u);
    assert(cold.arena_bytes == sizeof(pack_bytes));
    assert(cold.device_handle != 0ull);

    pack_file = fopen(SPARK_TEST_PACK, "rb");
    assert(pack_file != 0);
    assert(fread(pack_bytes, 1u, sizeof(pack_bytes), pack_file) ==
        sizeof(pack_bytes));
    assert(fclose(pack_file) == 0);
    memcpy(arena_probe, (const void *)(uintptr_t)cold.device_handle,
        sizeof(arena_probe));
    assert(memcmp(arena_probe, pack_bytes, sizeof(pack_bytes)) == 0);

    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &warm, reason) == SPARK_STATUS_OK);
    assert(warm.client != 0);
    assert(warm.loaded_from_pack == 0u);
    assert(warm.refcount == 2u);
    assert(warm.arena_generation == cold.arena_generation);
    assert(warm.device_handle == cold.device_handle);

    {
        uint64_t generation = cold.arena_generation;
        uint64_t handle = cold.device_handle;
        SparkWeightdAttachRelease(&cold);
        assert(cold.client == 0 && cold.device_handle == 0ull);
        assert(cold.arena_generation == 0ull);
        SparkWeightdAttachRelease(&warm);

        assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
            SPARK_TEST_TIMEOUT_NS, &again, reason) == SPARK_STATUS_OK);
        assert(again.client != 0);
        assert(again.loaded_from_pack == 0u);
        assert(again.refcount == 1u);
        assert(again.arena_generation == generation);
        assert(again.device_handle == handle);

        SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_REVISION, "next-rev");
        assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
            SPARK_TEST_TIMEOUT_NS, &revisioned, reason) == SPARK_STATUS_OK);
        assert(revisioned.client != 0);
        assert(revisioned.loaded_from_pack == 1u);
        assert(revisioned.arena_generation != generation);
        SparkWeightdAttachRelease(&revisioned);
        SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_REVISION, 0);

        SparkWeightdAttachRelease(&again);
    }
    SparkTestStopServer(&thread_context, thread_handle);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, 0);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("cold load, warm hit, and cold retention green\n");
}

typedef struct SparkTestConcurrentAttach
{
    const SparkWeightdPackSlice *slice;
    SparkWeightdAttachOutcome outcome;
    SparkStatus status;
} SparkTestConcurrentAttach;

static void *SparkTestConcurrentAttachMain(void *raw)
{
    SparkTestConcurrentAttach *attach = (SparkTestConcurrentAttach *)raw;
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    attach->status = SparkWeightdAttachPack(attach->slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &attach->outcome, reason);
    return 0;
}

static void SparkTestFlipPackByte(uint64_t offset)
{
    FILE *file = fopen(SPARK_TEST_PACK, "r+b");
    int value;
    assert(file != 0);
    assert(fseek(file, (long)offset, SEEK_SET) == 0);
    value = fgetc(file);
    assert(value != EOF);
    assert(fseek(file, (long)offset, SEEK_SET) == 0);
    assert(fputc(value ^ 0x5a, file) != EOF);
    assert(fclose(file) == 0);
}

static void SparkTestVerifyOnce(void)
{
    SparkWeightdPackSlice slice;
    SparkWeightdAttachOutcome outcome;
    SparkWeightdReceipt first;
    SparkWeightdReceipt second;
    SparkTestConcurrentAttach racers[2];
    pthread_t racer_threads[2];
    char reason[SPARK_WEIGHTD_ATTACH_REASON_BYTES];
    char digest[SPARK_SHA256_HEX_BYTES];
    char resolved[4096];
    char receipt_path[4200];
    SparkTestServerThread thread_context;
    pthread_t thread_handle;
    static uint8_t pack_bytes[SPARK_TEST_ARENA_BYTES];
    FILE *pack_file;
    uint32_t index;

    spark_stub_cuda_reset_faults();
    SparkTestWritePack(SPARK_TEST_PACK, 51u, SPARK_TEST_ARENA_BYTES, digest);
    assert(realpath(SPARK_TEST_PACK, resolved) != 0);
    assert(snprintf(receipt_path, sizeof(receipt_path), "%s%s", resolved,
        SPARK_WEIGHTD_RECEIPT_SUFFIX) > 0);
    (void)remove(receipt_path);
    SparkTestMakeSlice(&slice, SPARK_TEST_ARENA_BYTES);
    SparkTestClearAttachEnv();
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, SPARK_TEST_SOCKET);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);
    usleep(1100000);

    SparkTestStartServer(&thread_context, &thread_handle);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) == SPARK_STATUS_OK);
    assert(outcome.loaded_from_pack == 1u);
    SparkWeightdAttachRelease(&outcome);
    SparkTestStopServer(&thread_context, thread_handle);
    assert(SparkWeightdReceiptLoad(receipt_path, &first) == SPARK_STATUS_OK);
    assert(strcmp(first.sha256, digest) == 0);
    assert(strncmp(first.verifier, "weightd ", 8u) == 0);

    SparkTestStartServer(&thread_context, &thread_handle);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) == SPARK_STATUS_OK);
    assert(outcome.loaded_from_pack == 1u);
    pack_file = fopen(SPARK_TEST_PACK, "rb");
    assert(pack_file != 0);
    assert(fread(pack_bytes, 1u, sizeof(pack_bytes), pack_file) ==
        sizeof(pack_bytes));
    assert(fclose(pack_file) == 0);
    assert(memcmp((const void *)(uintptr_t)outcome.device_handle, pack_bytes,
        sizeof(pack_bytes)) == 0);
    SparkWeightdAttachRelease(&outcome);
    SparkTestStopServer(&thread_context, thread_handle);
    assert(SparkWeightdReceiptLoad(receipt_path, &second) == SPARK_STATUS_OK);
    assert(second.verified_unix_ns == first.verified_unix_ns);

    SparkTestStartServer(&thread_context, &thread_handle);
    for (index = 0u; index < 2u; index++)
    {
        memset(&racers[index], 0, sizeof(racers[index]));
        racers[index].slice = &slice;
        assert(pthread_create(&racer_threads[index], 0,
            SparkTestConcurrentAttachMain, &racers[index]) == 0);
    }
    for (index = 0u; index < 2u; index++)
    {
        assert(pthread_join(racer_threads[index], 0) == 0);
        assert(racers[index].status == SPARK_STATUS_OK);
    }
    assert(racers[0].outcome.arena_generation ==
        racers[1].outcome.arena_generation);
    assert(racers[0].outcome.loaded_from_pack +
        racers[1].outcome.loaded_from_pack == 1u);
    SparkWeightdAttachRelease(&racers[0].outcome);
    SparkWeightdAttachRelease(&racers[1].outcome);
    SparkTestStopServer(&thread_context, thread_handle);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256,
        "2222222222222222222222222222222222222222222222222222222222222222");
    SparkTestStartServer(&thread_context, &thread_handle);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(strcmp(reason, "hash_mismatch") == 0);
    SparkTestStopServer(&thread_context, thread_handle);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, digest);

    SparkTestFlipPackByte(4097u);
    SparkTestStartServer(&thread_context, &thread_handle);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) != SPARK_STATUS_OK);
    assert(strcmp(reason, "hash_mismatch") == 0);
    SparkTestFlipPackByte(4097u);
    usleep(1100000);
    assert(SparkWeightdAttachPack(&slice, SPARK_TEST_PACK,
        SPARK_TEST_TIMEOUT_NS, &outcome, reason) == SPARK_STATUS_OK);
    SparkWeightdAttachRelease(&outcome);
    SparkTestStopServer(&thread_context, thread_handle);
    assert(SparkWeightdReceiptLoad(receipt_path, &second) == SPARK_STATUS_OK);
    assert(second.verified_unix_ns > first.verified_unix_ns);
    assert(second.inode == first.inode && second.ctime_ns != first.ctime_ns);

    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SHA256, 0);
    SparkTestSetEnv(SPARK_WEIGHTD_ATTACH_ENV_SOCKET, 0);
    assert(remove(receipt_path) == 0);
    (void)remove(SPARK_TEST_PACK);
    (void)remove(SPARK_TEST_SOCKET);
    printf("verify once: receipt skips the rehash, stale or mismatched receipts force a full verify, concurrent cold loads share one arena\n");
}

int main(void)
{
    (void)signal(SIGPIPE, SIG_IGN);
    SparkTestConfigurationGates();
    SparkTestRefusedAttachFailsAndAllocatesNothing();
    SparkTestColdWarmAndRetention();
    SparkTestVerifyOnce();
    printf("w2 weightd lane: serving-side attach errors + warm hit green\n");
    return 0;
}
