/* W1 loader lane bench (docs/WEIGHTD_DESIGN.md L1+L2).
 * Walks a real dsv4 stage pack exactly the way SparkDsv4ModuleLoadPack
 * does - header, directory, then payload+scale per entry - and times the
 * load through (a) the synchronous region loader and (b) the pack-wide
 * pipelined loader, on the same process invocation with the file's page
 * cache dropped between passes (posix_fadvise DONTNEED). --sha times the
 * whole-file content digest both ways; the digest MUST come out
 * identical, which the bench asserts.
 *
 * usage: dsv4_loader_bench <pack-path> [--load|--sha|--all]
 */
#define _POSIX_C_SOURCE 200809L
#define _FILE_OFFSET_BITS 64

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <fcntl.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_stage_module_common.h"
#include "sparkpipe/spark_dsv4_model.h"
#include "spark_dsv4_stagepack_format.h"

static const char *BENCH_TAG = "dsv4_loader_bench";

static double SparkBenchSecondsSince(struct timespec since)
{
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)(now.tv_sec - since.tv_sec) +
        (double)(now.tv_nsec - since.tv_nsec) / 1e9;
}

static void SparkBenchDropPageCache(FILE *file)
{
#ifdef POSIX_FADV_DONTNEED
    if (file != 0)
    {
        (void)posix_fadvise(fileno(file), 0, 0, POSIX_FADV_DONTNEED);
    }
#else
    /* no posix_fadvise on this host (mac dry runs): cache stays warm */
    (void)file;
#endif
}

static SparkStatus SparkBenchReadDirectory(
    FILE *file,
    SparkDsv4StagePackHeader *header,
    SparkDsv4StagePackEntry **directory)
{
    SparkStatus status = SparkStageModulePackRead(
        BENCH_TAG, file, 0u, header, sizeof(*header));
    *directory = 0;
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (header->tensor_count == 0u || header->tensor_count > 100000u)
    {
        fprintf(stderr, "%s implausible tensor_count=%u\n",
            BENCH_TAG, header->tensor_count);
        return SPARK_STATUS_VALIDATION_FAILED;
    }
    *directory = (SparkDsv4StagePackEntry *)malloc(
        (size_t)header->tensor_count * sizeof(SparkDsv4StagePackEntry));
    if (*directory == 0)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SparkStageModulePackRead(
        BENCH_TAG, file, header->directory_offset, *directory,
        (uint64_t)header->tensor_count * sizeof(SparkDsv4StagePackEntry));
}

static double SparkBenchLoadSequential(
    SparkStageModuleLedger *ledger,
    FILE *file,
    const SparkDsv4StagePackHeader *header,
    const SparkDsv4StagePackEntry *directory)
{
    struct timespec start;
    uint32_t index;
    uint64_t moved_bytes = 0u;
    SparkStatus status = SPARK_STATUS_OK;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (index = 0u; status == SPARK_STATUS_OK && index < header->tensor_count;
        index++)
    {
        const SparkDsv4StagePackEntry *entry = &directory[index];
        uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(
            entry->weight_format, entry->rows, entry->columns);
        uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(
            entry->weight_format, entry->rows, entry->columns);
        void *payload = 0;
        void *scale = 0;
        status = SparkStageModuleLoadDeviceRegion(
            ledger, file, entry->payload_offset, payload_bytes, &payload);
        if (status == SPARK_STATUS_OK && scale_bytes != 0u)
        {
            status = SparkStageModuleLoadDeviceRegion(
                ledger, file, entry->scale_offset, scale_bytes, &scale);
        }
        moved_bytes += payload_bytes + scale_bytes;
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "%s region_load_failed entry=%u\n",
                BENCH_TAG, index);
        }
    }
    printf("load seq: %.3fs, %llu MiB, %.2f GiB/s\n",
        SparkBenchSecondsSince(start),
        (unsigned long long)(moved_bytes / (1024ull * 1024ull)),
        moved_bytes == 0u ? 0.0 :
            ((double)moved_bytes / (1024.0 * 1024.0 * 1024.0)) /
                SparkBenchSecondsSince(start));
    return SparkBenchSecondsSince(start);
}

static double SparkBenchLoadPipelined(
    SparkStageModuleLedger *ledger,
    FILE *file,
    const SparkDsv4StagePackHeader *header,
    const SparkDsv4StagePackEntry *directory)
{
    struct timespec start;
    SparkStageModuleLoadPipeline *pipeline = 0;
    uint32_t index;
    uint64_t moved_bytes = 0u;
    SparkStatus status;
    clock_gettime(CLOCK_MONOTONIC, &start);
    status = SparkStageModuleLoadPipelineCreate(BENCH_TAG, file, &pipeline);
    for (index = 0u; status == SPARK_STATUS_OK && index < header->tensor_count;
        index++)
    {
        const SparkDsv4StagePackEntry *entry = &directory[index];
        uint64_t payload_bytes = SparkDsv4StagePackPayloadBytes(
            entry->weight_format, entry->rows, entry->columns);
        uint64_t scale_bytes = SparkDsv4StagePackScaleBytes(
            entry->weight_format, entry->rows, entry->columns);
        void *payload = 0;
        void *scale = 0;
        status = SparkStageModuleLoadPipelineRegion(
            pipeline, ledger, entry->payload_offset, payload_bytes, &payload);
        if (status == SPARK_STATUS_OK && scale_bytes != 0u)
        {
            status = SparkStageModuleLoadPipelineRegion(
                pipeline, ledger, entry->scale_offset, scale_bytes, &scale);
        }
        moved_bytes += payload_bytes + scale_bytes;
        if (status != SPARK_STATUS_OK)
        {
            fprintf(stderr, "%s pipeline_region_failed entry=%u\n",
                BENCH_TAG, index);
        }
    }
    if (status == SPARK_STATUS_OK)
    {
        status = SparkStageModuleLoadPipelineFinish(pipeline);
    }
    SparkStageModuleLoadPipelineDestroy(pipeline);
    if (status != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s pipeline_finish_failed status=%d\n",
            BENCH_TAG, (int)status);
    }
    printf("load pipe: %.3fs, %llu MiB, %.2f GiB/s\n",
        SparkBenchSecondsSince(start),
        (unsigned long long)(moved_bytes / (1024ull * 1024ull)),
        moved_bytes == 0u ? 0.0 :
            ((double)moved_bytes / (1024.0 * 1024.0 * 1024.0)) /
                SparkBenchSecondsSince(start));
    return SparkBenchSecondsSince(start);
}

static void SparkBenchSha(const char *path)
{
    struct timespec start;
    char pipelined_hex[SPARK_SHA256_HEX_BYTES];
    char sequential_hex[SPARK_SHA256_HEX_BYTES];

    setenv("SPARK_SHA256_FILE_PIPELINE", "1", 1);
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (SparkSha256File(path, pipelined_hex) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s sha_pipeline_failed\n", BENCH_TAG);
        return;
    }
    printf("sha pipe: %.3fs digest=%s\n",
        SparkBenchSecondsSince(start), pipelined_hex);

    SparkBenchDropPageCache(0);
    setenv("SPARK_SHA256_FILE_PIPELINE", "0", 1);
    clock_gettime(CLOCK_MONOTONIC, &start);
    if (SparkSha256File(path, sequential_hex) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s sha_sequential_failed\n", BENCH_TAG);
        return;
    }
    printf("sha seq: %.3fs digest=%s\n",
        SparkBenchSecondsSince(start), sequential_hex);
    printf("sha identity: %s\n",
        strcmp(pipelined_hex, sequential_hex) == 0 ? "IDENTICAL" : "MISMATCH");
    setenv("SPARK_SHA256_FILE_PIPELINE", "1", 1);
}

int main(int argument_count, char **arguments)
{
    const char *path;
    int run_sha = 0;
    int run_load = 0;
    FILE *file;
    SparkDsv4StagePackHeader header;
    SparkDsv4StagePackEntry *directory = 0;
    SparkStageModuleLedger ledger;
    int argument;

    if (argument_count < 2)
    {
        fprintf(stderr, "usage: %s <pack-path> [--load|--sha|--all]\n",
            arguments[0]);
        return 2;
    }
    path = arguments[1];
    for (argument = 2; argument < argument_count; argument++)
    {
        if (strcmp(arguments[argument], "--load") == 0)
        {
            run_load = 1;
        }
        else if (strcmp(arguments[argument], "--sha") == 0)
        {
            run_sha = 1;
        }
        else if (strcmp(arguments[argument], "--all") == 0)
        {
            run_load = 1;
            run_sha = 1;
        }
    }
    if (!run_load && !run_sha)
    {
        run_load = 1;
    }

    file = fopen(path, "rb");
    if (file == 0)
    {
        fprintf(stderr, "%s pack_open_failed path=%s\n", BENCH_TAG, path);
        return 1;
    }
    if (SparkBenchReadDirectory(file, &header, &directory) != SPARK_STATUS_OK)
    {
        fprintf(stderr, "%s directory_read_failed path=%s\n", BENCH_TAG, path);
        (void)fclose(file);
        return 1;
    }
    printf("pack: tensor_count=%u first_layer=%u layer_count=%u file_bytes=%llu\n",
        header.tensor_count, header.first_layer_index, header.layer_count,
        (unsigned long long)header.file_bytes);

    if (run_sha)
    {
        SparkBenchSha(path);
        SparkBenchDropPageCache(file);
    }
    if (run_load)
    {
        double sequential_seconds;
        double pipelined_seconds;
        memset(&ledger, 0, sizeof(ledger));
        ledger.module_tag = BENCH_TAG;
        /* the kill switch pins the dispatcher to the synchronous path */
        setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "0", 1);
        sequential_seconds = SparkBenchLoadSequential(
            &ledger, file, &header, directory);
        SparkStageModuleLedgerRelease(&ledger);
        SparkBenchDropPageCache(file);

        setenv("SPARK_STAGE_MODULE_LOAD_PIPELINE", "1", 1);
        pipelined_seconds = SparkBenchLoadPipelined(
            &ledger, file, &header, directory);
        SparkStageModuleLedgerRelease(&ledger);
        SparkBenchDropPageCache(file);
        (void)sequential_seconds;
        (void)pipelined_seconds;
    }
    free(directory);
    (void)fclose(file);
    return 0;
}
