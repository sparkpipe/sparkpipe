#include <assert.h>
#include <stdint.h>
#include <pthread.h>
#include <signal.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_weightd.h"

static volatile sig_atomic_t SparkTestWeightdStop;
static char SparkTestWeightdSocket[128];

static void *SparkTestWeightdRun(void *context)
{
    assert(SparkWeightdServerRun(context,&SparkTestWeightdStop) == SPARK_STATUS_OK);
    return 0;
}

static void SparkTestUseWeightdPack(const char *path)
{
    char digest[SPARK_SHA256_HEX_BYTES];
    assert(SparkSha256File(path,digest) == SPARK_STATUS_OK);
    assert(setenv("SPARK_WEIGHTD_ATTACH","1",1) == 0);
    assert(setenv("SPARK_WEIGHTD_SOCKET",SparkTestWeightdSocket,1) == 0);
    assert(setenv("SPARK_WEIGHTD_PACK_SHA256",digest,1) == 0);
}

static uint32_t SparkTestNextRandom(uint32_t *state)
{
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

static void SparkTestAssertHexEquals(
    const char *actual_hex,
    const char *expected_hex)
{
    assert(actual_hex != 0);
    assert(strlen(actual_hex) == 64u);
    assert(strcmp(actual_hex, expected_hex) == 0);
}

static void SparkTestSha256NistVectors(void)
{
    char hex[SPARK_SHA256_HEX_BYTES];

    assert(SparkSha256Bytes("abc", 3u, hex) == SPARK_STATUS_OK);
    SparkTestAssertHexEquals(
        hex, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    assert(SparkSha256Bytes("", 0u, hex) == SPARK_STATUS_OK);
    SparkTestAssertHexEquals(
        hex, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    assert(SparkSha256Bytes(
        "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
        56u, hex) == SPARK_STATUS_OK);
    SparkTestAssertHexEquals(
        hex, "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

static void SparkTestSha256LongMessageVector(void)
{
    static const uint32_t stream_bytes = 1000003u;
    uint8_t *stream =
        (uint8_t *)malloc((size_t)stream_bytes);
    SparkSha256Context context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    char hex[SPARK_SHA256_HEX_BYTES];
    uint32_t state = 42u;
    uint32_t index;
    size_t fed;

    assert(stream != 0);
    for (index = 0u; index < stream_bytes; index++)
    {
        state = state * 1664525u + 1013904223u;
        stream[index] = (uint8_t)(state >> 24u);
    }

    SparkSha256Initialize(&context);
    SparkSha256Update(&context, stream, (size_t)stream_bytes);
    SparkSha256Finalize(&context, digest);
    SparkSha256DigestToHex(digest, hex);
    SparkTestAssertHexEquals(
        hex, "e1c2b53ce00fca4ba820d0207d7bca19adbd4fdcd492188bd9015f2284bbe6fc");

    SparkSha256Initialize(&context);
    for (fed = 0u; fed < (size_t)stream_bytes; fed += 7u)
    {
        size_t remaining = (size_t)stream_bytes - fed;
        SparkSha256Update(&context, stream + fed,
            remaining < 7u ? remaining : 7u);
    }
    SparkSha256Finalize(&context, digest);
    SparkSha256DigestToHex(digest, hex);
    SparkTestAssertHexEquals(
        hex, "e1c2b53ce00fca4ba820d0207d7bca19adbd4fdcd492188bd9015f2284bbe6fc");
    free(stream);
}

static void SparkTestWritePatternFile(
    const char *path,
    const uint32_t *pattern_words,
    uint64_t file_words)
{
    FILE *file = fopen(path, "wb");
    uint64_t index;
    assert(file != 0);
    for (index = 0; index < file_words; index++)
    {
        uint32_t word = pattern_words[index];
        assert(fwrite(&word, sizeof(word), 1u, file) == 1u);
    }
    assert(fclose(file) == 0);
}

static void SparkTestReferenceFileDigest(
    const char *path,
    char hex[SPARK_SHA256_HEX_BYTES])
{
    FILE *file = fopen(path, "rb");
    SparkSha256Context context;
    uint8_t buffer[8192];
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    size_t bytes_read;
    assert(file != 0);
    SparkSha256Initialize(&context);
    while ((bytes_read = fread(buffer, 1u, sizeof(buffer), file)) != 0u)
    {
        SparkSha256Update(&context, buffer, bytes_read);
    }
    assert(ferror(file) == 0);
    assert(fclose(file) == 0);
    SparkSha256Finalize(&context, digest);
    SparkSha256DigestToHex(digest, hex);
}

static void SparkTestSha256FileIdentityAcrossReadModes(const char *path)
{
    char pipeline_hex[SPARK_SHA256_HEX_BYTES];
    char sequential_hex[SPARK_SHA256_HEX_BYTES];
    char reference_hex[SPARK_SHA256_HEX_BYTES];

    SparkTestReferenceFileDigest(path, reference_hex);

    setenv("SPARK_SHA256_FILE_PIPELINE", "1", 1);
    assert(SparkSha256File(path, pipeline_hex) == SPARK_STATUS_OK);

    setenv("SPARK_SHA256_FILE_PIPELINE", "0", 1);
    assert(SparkSha256File(path, sequential_hex) == SPARK_STATUS_OK);

    SparkTestAssertHexEquals(pipeline_hex, reference_hex);
    SparkTestAssertHexEquals(sequential_hex, reference_hex);

    unsetenv("SPARK_SHA256_FILE_PIPELINE");
}

static void SparkTestSha256FileSizes(void)
{
    static const uint64_t file_word_counts[] = {
        0u, 1u, 14u, 15u, 16u, 63u, 1024u, 65536u, 65537u
    };
    const char *path = "/tmp/spark_w1_sha_identity.bin";
    uint32_t state = 20260829u;
    uint32_t word_index;
    uint64_t words_max = 65537u;
    uint32_t *pattern_words =
        (uint32_t *)malloc((size_t)words_max * sizeof(uint32_t));
    size_t case_index;
    assert(pattern_words != 0);
    for (word_index = 0; (uint64_t)word_index < words_max; word_index++)
    {
        pattern_words[word_index] = SparkTestNextRandom(&state);
    }
    for (case_index = 0;
        case_index < sizeof(file_word_counts) / sizeof(file_word_counts[0]);
        case_index++)
    {
        uint64_t words = file_word_counts[case_index];
        SparkTestWritePatternFile(path, pattern_words, words);
        SparkTestSha256FileIdentityAcrossReadModes(path);
    }
    free(pattern_words);
    (void)remove(path);
}

static void SparkTestSha256FileMultiBufferBoundaries(void)
{
    static const uint64_t file_bytes[] = {
        4ull * 1024ull * 1024ull,
        4ull * 1024ull * 1024ull + 1ull,
        12ull * 1024ull * 1024ull + 5ull
    };
    const char *path = "/tmp/spark_w1_sha_multibuffer.bin";
    uint32_t state = 777u;
    FILE *file = fopen(path, "wb");
    uint64_t total = file_bytes[2];
    uint64_t written = 0u;
    size_t case_index;
    assert(file != 0);
    while (written < total)
    {
        uint32_t word = SparkTestNextRandom(&state);
        uint64_t remaining = total - written;
        uint64_t chunk = remaining < sizeof(word) ? remaining : sizeof(word);
        assert(fwrite(&word, 1u, (size_t)chunk, file) == (size_t)chunk);
        written += chunk;
    }
    assert(fclose(file) == 0);
    for (case_index = 0; case_index < sizeof(file_bytes) / sizeof(file_bytes[0]);
        case_index++)
    {
        char command[512];
        snprintf(command, sizeof(command),
            "truncate -s %llu %s", (unsigned long long)file_bytes[case_index],
            path);
        assert(system(command) == 0);
        SparkTestSha256FileIdentityAcrossReadModes(path);
    }
    (void)remove(path);
}

typedef struct SparkTestLoaderRegion
{
    uint64_t offset;
    uint64_t bytes;
    uint32_t seed;
} SparkTestLoaderRegion;

static void SparkTestFillPattern(
    uint8_t *buffer,
    uint64_t bytes,
    uint32_t seed)
{
    uint32_t state = seed;
    uint64_t index;
    for (index = 0; index < bytes; index++)
    {
        if (index % 4u == 0u)
        {
            state = state * 1664525u + 1013904223u;
        }
        buffer[index] = (uint8_t)(state >> ((index % 4u) * 8u));
    }
}

static void SparkTestWriteLoaderPack(
    const char *path,
    const SparkTestLoaderRegion *regions,
    uint32_t region_count)
{
    FILE *file = fopen(path, "wb");
    uint8_t *buffer;
    uint64_t file_bytes = 0u;
    uint32_t region;
    for (region = 0; region < region_count; region++)
    {
        uint64_t end = regions[region].offset + regions[region].bytes;
        if (end > file_bytes)
        {
            file_bytes = end;
        }
    }
    assert(file != 0);
    buffer = (uint8_t *)malloc((size_t)file_bytes);
    assert(buffer != 0);
    memset(buffer, 0xA5, (size_t)file_bytes);
    for (region = 0; region < region_count; region++)
    {
        SparkTestFillPattern(buffer + regions[region].offset,
            regions[region].bytes, regions[region].seed);
    }
    assert(fwrite(buffer, 1u, (size_t)file_bytes, file) ==
        (size_t)file_bytes);
    assert(fclose(file) == 0);
    free(buffer);
}

static void SparkTestMappedLoaderLandsExactBytes(void)
{
    const char *path = "/tmp/spark_w1_loader_pack.bin";
    static const SparkTestLoaderRegion regions[] = {
        {0ull, 1024ull * 1024ull, 11u},
        {2ull * 1024ull * 1024ull, 64ull * 1024ull, 22u},
        {5ull * 1024ull * 1024ull, 17ull, 33u},
        {1ull * 1024ull * 1024ull, 512ull * 1024ull, 44u},
        {9ull * 1024ull * 1024ull, 3ull, 55u},
        {6ull * 1024ull * 1024ull, 2048ull, 66u}
    };
    const uint32_t region_count =
        (uint32_t)(sizeof(regions) / sizeof(regions[0]));
    SparkStageModuleLedger ledger;
    FILE *file;
    void *device_pointers[6];
    uint8_t *expected;
    uint32_t region;

    assert(region_count <= 6u);
    SparkTestWriteLoaderPack(path, regions, region_count);
    SparkTestUseWeightdPack(path);

    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "w1_loader_test";
    file = fopen(path, "rb");
    assert(file != 0);
    for (region = 0; region < region_count; region++)
    {
        assert(SparkStageModuleLoadDeviceRegion(&ledger, file,
            regions[region].offset, regions[region].bytes,
            &device_pointers[region]) == SPARK_STATUS_OK);
        assert(device_pointers[region] != 0);
    }

    expected = (uint8_t *)malloc(10ull * 1024ull * 1024ull);
    assert(expected != 0);
    memset(expected, 0xA5, 10ull * 1024ull * 1024ull);
    for (region = 0; region < region_count; region++)
    {
        SparkTestFillPattern(expected + regions[region].offset,
            regions[region].bytes, regions[region].seed);
    }
    for (region = 0; region < region_count; region++)
    {
        assert((uint8_t *)device_pointers[region] - (uint8_t *)device_pointers[0] ==
            (int64_t)regions[region].offset);
        assert(memcmp(device_pointers[region],
            expected + regions[region].offset,
            (size_t)regions[region].bytes) == 0);
    }
    free(expected);
    assert(ledger.pack_arena != 0 && ledger.device_allocation_count == 0u &&
        ledger.device_bytes_resident == 0u);

    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    (void)remove(path);
}

static void SparkTestMappedLoaderRejectsShortPack(void)
{
    const char *path = "/tmp/spark_w1_loader_short.bin";
    SparkStageModuleLedger ledger = {0};
    FILE *file = fopen(path,"wb");
    void *pointer = 0;
    assert(file != 0 && fwrite("0123456789",1u,10u,file) == 10u);
    assert(fclose(file) == 0);
    SparkTestUseWeightdPack(path);
    file = fopen(path,"rb");
    assert(file != 0);
    ledger.module_tag = "w1_loader_test";
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,0u,8u,&pointer) == SPARK_STATUS_OK);
    assert(pointer != 0 && memcmp(pointer,"01234567",8u) == 0);
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,4096u,16u,&pointer) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(pointer == 0 && ledger.device_allocation_count == 0u && ledger.device_bytes_resident == 0u);
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,0u,11u,&pointer) == SPARK_STATUS_INVALID_ARGUMENT);
    assert(pointer == 0);
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,0u,10u,&pointer) == SPARK_STATUS_OK);
    assert(pointer != 0 && memcmp(pointer,"0123456789",10u) == 0);
    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    assert(remove(path) == 0);
}

static void SparkTestLoaderRequiresWeightd(void)
{
    const char *path = "/tmp/spark_w1_loader_dispatch.bin";
    SparkStageModuleLedger ledger;
    SparkTestLoaderRegion one = {0ull, 4096ull, 99u};
    uint8_t expected[4096];
    FILE *file;
    void *pointer = 0;
    SparkStageModuleLoadPipeline *pipeline = 0;

    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "w1_loader_test";

    SparkTestWriteLoaderPack(path, &one, 1u);
    SparkTestFillPattern(expected, one.bytes, one.seed);
    file = fopen(path, "rb");
    assert(file != 0);
    unsetenv("SPARK_WEIGHTD_SOCKET");
    unsetenv("SPARK_WEIGHTD_ATTACH");
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,one.offset,one.bytes,&pointer) == SPARK_STATUS_UNSUPPORTED);
    assert(pointer == 0 && ledger.pack_arena == 0 && ledger.device_allocation_count == 0u);
    assert(SparkStageModuleLoadPipelineCreate("w1_loader_test",file,&pipeline) == SPARK_STATUS_OK);
    assert(SparkStageModuleLoadPipelineRegion(pipeline,&ledger,one.offset,one.bytes,&pointer) == SPARK_STATUS_UNSUPPORTED);
    assert(pointer == 0 && ledger.pack_arena == 0 && ledger.device_allocation_count == 0u);
    SparkTestUseWeightdPack(path);
    assert(SparkStageModuleLoadPipelineRegion(pipeline,&ledger,one.offset,one.bytes,&pointer) == SPARK_STATUS_UNSUPPORTED);
    assert(pointer == 0 && ledger.pack_arena == 0 && ledger.device_allocation_count == 0u);
    SparkStageModuleLoadPipelineDestroy(pipeline);
    assert(SparkStageModuleLoadDeviceRegion(&ledger,file,one.offset,one.bytes,&pointer) == SPARK_STATUS_OK);
    assert(pointer != 0 && ledger.pack_arena != 0 && memcmp(pointer,expected,sizeof(expected)) == 0);
    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    (void)remove(path);
}

int main(void)
{
    SparkWeightdServerConfig config = {0};
    SparkWeightdServer *server = 0;
    pthread_t worker;
    signal(SIGPIPE,SIG_IGN);
    assert(snprintf(SparkTestWeightdSocket,sizeof(SparkTestWeightdSocket),
        "/tmp/spark_w1_weightd_%ld.sock",(long)getpid()) > 0);
    config.socket_path=SparkTestWeightdSocket;
    config.device_bytes_max=256ull << 20;
    config.kv_reserve_bytes=0u;
    unsetenv("SPARK_WEIGHTD_ATTACH_LAZY");
    assert(SparkWeightdServerCreate(&config,&server) == SPARK_STATUS_OK);
    assert(pthread_create(&worker,0,SparkTestWeightdRun,server) == 0);
    SparkTestSha256NistVectors();
    SparkTestSha256LongMessageVector();
    SparkTestSha256FileSizes();
    SparkTestSha256FileMultiBufferBoundaries();
    SparkTestMappedLoaderLandsExactBytes();
    SparkTestMappedLoaderRejectsShortPack();
    SparkTestLoaderRequiresWeightd();
    __atomic_store_n(&SparkTestWeightdStop,1,__ATOMIC_SEQ_CST);
    assert(pthread_join(worker,0) == 0);
    assert(SparkWeightdServerArenaCount(server) == 3u);
    SparkWeightdServerDestroy(server);
    unsetenv("SPARK_WEIGHTD_ATTACH");
    unsetenv("SPARK_WEIGHTD_SOCKET");
    unsetenv("SPARK_WEIGHTD_PACK_SHA256");
    printf("w1 loader: SHA identity, required weightd mapping bytes, stride and bounds PASS\n");
    return 0;
}
