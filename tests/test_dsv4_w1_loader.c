/* W1 loader lane (docs/WEIGHTD_DESIGN.md L1+L2) host tests:
 * - the SHA-256 file digest is bit-identical whether the read pass runs
 *   sequential or read-pipelined (identity keys depend on that), and
 *   matches the streaming Initialize/Update/Finalize primitives and the
 *   published NIST vectors;
 * - the pipelined pack loader lands the exact bytes of every requested
 *   region at the exact device addresses, fails closed on a short pack,
 *   and its env kill switch restores the synchronous path. */
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_module_common.h"

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
    /* the primitive path is the sequential ground truth */
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

    /* THE gate: identical digest value regardless of the read schedule */
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
    /* 4 MiB pipeline buffers: cover exactly-one-buffer, one-plus-one-byte,
     * and a three-buffer file with an odd tail (cross-buffer handoffs) */
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

static void SparkTestLoaderPipelineLandsExactBytes(void)
{
    const char *path = "/tmp/spark_w1_loader_pack.bin";
    /* non-contiguous, unordered offsets: each chunk carries its own */
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
    SparkStageModuleLoadPipeline *pipeline = 0;
    FILE *file;
    void *device_pointers[6];
    uint8_t *expected;
    uint32_t region;

    assert(region_count <= 6u);
    setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "1", 1);
    SparkTestWriteLoaderPack(path, regions, region_count);

    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "w1_loader_test";
    file = fopen(path, "rb");
    assert(file != 0);
    assert(SparkStageModuleLoadPipelineCreate("w1_loader_test", file,
        &pipeline) == SPARK_STATUS_OK);
    for (region = 0; region < region_count; region++)
    {
        assert(SparkStageModuleLoadPipelineRegion(pipeline, &ledger,
            regions[region].offset, regions[region].bytes,
            &device_pointers[region]) == SPARK_STATUS_OK);
        assert(device_pointers[region] != 0);
    }
    assert(SparkStageModuleLoadPipelineFinish(pipeline) == SPARK_STATUS_OK);
    SparkStageModuleLoadPipelineDestroy(pipeline);

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
        /* every byte landed at its exact device address */
        assert(memcmp(device_pointers[region],
            expected + regions[region].offset,
            (size_t)regions[region].bytes) == 0);
    }
    free(expected);
    assert(ledger.device_allocation_count == region_count);
    assert(ledger.device_bytes_resident ==
        1024ull * 1024ull + 64ull * 1024ull + 512ull * 1024ull + 3ull + 2048ull + 17ull);

    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    (void)remove(path);
    unsetenv("SPARK_STAGE_MODULE_LOAD_PIPELINE");
}

static void SparkTestLoaderPipelineFailsClosedOnShortPack(void)
{
    const char *path = "/tmp/spark_w1_loader_short.bin";
    SparkStageModuleLedger ledger;
    SparkStageModuleLoadPipeline *pipeline = 0;
    FILE *file;
    void *pointer = 0;

    setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "1", 1);
    file = fopen(path, "wb");
    assert(file != 0);
    assert(fwrite("0123456789", 1u, 10u, file) == 10u);
    assert(fclose(file) == 0);
    file = fopen(path, "rb");
    assert(file != 0);

    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "w1_loader_test";
    assert(SparkStageModuleLoadPipelineCreate("w1_loader_test", file,
        &pipeline) == SPARK_STATUS_OK);
    assert(SparkStageModuleLoadPipelineRegion(pipeline, &ledger,
        0ull, 8ull, &pointer) == SPARK_STATUS_OK);
    assert(pointer != 0);
    /* past end of file: the worker read fails, Finish reports it */
    assert(SparkStageModuleLoadPipelineRegion(pipeline, &ledger,
        4096ull, 16ull, &pointer) == SPARK_STATUS_OK ||
        SparkStageModuleLoadPipelineRegion(pipeline, &ledger,
            4096ull, 16ull, &pointer) == SPARK_STATUS_IO_ERROR);
    assert(SparkStageModuleLoadPipelineFinish(pipeline) ==
        SPARK_STATUS_IO_ERROR);
    /* the pipeline is poisoned after a failure */
    assert(SparkStageModuleLoadPipelineRegion(pipeline, &ledger,
        0ull, 4ull, &pointer) == SPARK_STATUS_IO_ERROR);
    SparkStageModuleLoadPipelineDestroy(pipeline);
    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    (void)remove(path);
    unsetenv("SPARK_STAGE_MODULE_LOAD_PIPELINE");
}

static void SparkTestLoaderRegionDispatcherHonorsKillSwitch(void)
{
    const char *path = "/tmp/spark_w1_loader_dispatch.bin";
    SparkStageModuleLedger ledger;
    SparkTestLoaderRegion one = {0ull, 4096ull, 99u};
    uint8_t expected[4096];
    FILE *file;
    void *pointer = 0;

    memset(&ledger, 0, sizeof(ledger));
    ledger.module_tag = "w1_loader_test";

    /* the kill switch flips the request predicate both ways */
    setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "0", 1);
    assert(SparkStageModuleLoadPipelineRequested() == SPARK_STATUS_BUSY);
    setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "1", 1);
    assert(SparkStageModuleLoadPipelineRequested() == SPARK_STATUS_OK);
    unsetenv("SPARK_STAGE_MODULE_LOAD_PIPELINE");
    assert(SparkStageModuleLoadPipelineRequested() == SPARK_STATUS_OK);

    /* with the kill switch set the dispatcher takes the synchronous
     * path and still lands exact bytes */
    SparkTestWriteLoaderPack(path, &one, 1u);
    SparkTestFillPattern(expected, one.bytes, one.seed);
    file = fopen(path, "rb");
    assert(file != 0);
    setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "0", 1);
    assert(SparkStageModuleLoadDeviceRegion(&ledger, file,
        one.offset, one.bytes, &pointer) == SPARK_STATUS_OK);
    assert(memcmp(pointer, expected, sizeof(expected)) == 0);
    unsetenv("SPARK_STAGE_MODULE_LOAD_PIPELINE");
    assert(fclose(file) == 0);
    SparkStageModuleLedgerRelease(&ledger);
    (void)remove(path);
}

int main(void)
{
    SparkTestSha256NistVectors();
    SparkTestSha256FileSizes();
    SparkTestSha256FileMultiBufferBoundaries();
    SparkTestLoaderPipelineLandsExactBytes();
    SparkTestLoaderPipelineFailsClosedOnShortPack();
    SparkTestLoaderRegionDispatcherHonorsKillSwitch();
    printf("w1 loader lane: sha identity + pipeline byte-exactness green\n");
    return 0;
}
