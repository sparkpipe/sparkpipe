/* weightd_expert_segments — generate a lazy-expert manifest for any pack
 * without format knowledge: fixed 64 MiB segments, one ck128 each,
 * streamed through an 8 MiB pread buffer (the node memory law: bounded
 * RSS, no whole-file buffers). Layer is 0 and the expert id is the
 * segment index; a real packer emits true per-expert records later and
 * the server protocol is identical.
 *
 *   weightd_expert_segments <pack> [segment_bytes]
 * writes <pack>.experts
 */
#define _POSIX_C_SOURCE 200809L
#include "sparkpipe/spark_ck128.h"
#include "sparkpipe/spark_weightd.h"
#include <errno.h>
#include <fcntl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define STREAM_BYTES (8u * 1024u * 1024u)
#define SEGMENT_DEFAULT (64ull * 1024ull * 1024ull)

int main(int argc, char **argv)
{
    const char *pack;
    char manifest_path[4096];
    uint8_t *buffer;
    uint8_t digest[16];
    uint8_t header[16];
    uint8_t record[40];
    uint64_t segment_bytes;
    uint64_t file_bytes;
    uint64_t count;
    uint64_t index;
    uint32_t magic = SPARK_WEIGHTD_EXPERT_MANIFEST_MAGIC;
    uint32_t version = SPARK_WEIGHTD_EXPERT_MANIFEST_VERSION;
    uint32_t layer = 0u;
    uint32_t count32;
    int in;
    int out;
    struct stat status;
    uint64_t total_begin_ns;
    uint64_t total_end_ns;
    struct timespec now;

    if (argc < 2 || argc > 3)
    {
        fprintf(stderr, "usage: %s <pack> [segment_bytes]\n", argv[0]);
        return 2;
    }
    pack = argv[1];
    segment_bytes = argc == 3 ? strtoull(argv[2], 0, 10) : SEGMENT_DEFAULT;
    if (segment_bytes < (2ull * 1024ull * 1024ull) || segment_bytes > (1ull << 30))
    {
        fprintf(stderr, "weightd_expert_segments: segment out of range\n");
        return 2;
    }
    in = open(pack, O_RDONLY);
    if (in < 0 || fstat(in, &status) != 0 || status.st_size <= 0)
    {
        fprintf(stderr, "weightd_expert_segments: cannot size %s\n", pack);
        return 2;
    }
    file_bytes = (uint64_t)status.st_size;
    count = (file_bytes + segment_bytes - 1ull) / segment_bytes;
    if (count == 0ull || count > SPARK_WEIGHTD_EXPERT_COUNT_MAX)
    {
        fprintf(stderr, "weightd_expert_segments: %llu segments out of range\n",
            (unsigned long long)count);
        return 2;
    }
    if (snprintf(manifest_path, sizeof(manifest_path), "%s.experts", pack) >=
        (int)sizeof(manifest_path))
    {
        return 2;
    }
    out = open(manifest_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (out < 0)
    {
        fprintf(stderr, "weightd_expert_segments: cannot write %s\n",
            manifest_path);
        return 2;
    }
    buffer = (uint8_t *)malloc(STREAM_BYTES);
    if (buffer == 0)
    {
        return 2;
    }
    count32 = (uint32_t)count;
    memset(header, 0, sizeof(header));
    memcpy(header + 0u, &magic, 4u);
    memcpy(header + 4u, &version, 4u);
    memcpy(header + 8u, &count32, 4u);
    if (write(out, header, sizeof(header)) != sizeof(header))
    {
        return 2;
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    total_begin_ns = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
    for (index = 0ull; index < count; index++)
    {
        uint64_t offset = index * segment_bytes;
        uint64_t remaining = file_bytes - offset;
        uint64_t bytes = remaining < segment_bytes ? remaining : segment_bytes;
        uint64_t done = 0ull;
        SparkCk128Context context;
        SparkCk128Initialize(&context);
        while (done < bytes)
        {
            uint64_t piece = bytes - done < STREAM_BYTES ? bytes - done
                : STREAM_BYTES;
            ssize_t got = pread(in, buffer, (size_t)piece, (off_t)(offset + done));
            if (got <= 0)
            {
                fprintf(stderr, "weightd_expert_segments: read fault at %llu\n",
                    (unsigned long long)(offset + done));
                return 2;
            }
            SparkCk128Update(&context, buffer, (size_t)got);
            done += (uint64_t)got;
        }
        SparkCk128Finalize(&context, digest);
        memcpy(record + 0u, &layer, 4u);
        memcpy(record + 4u, &index, 4u);
        memcpy(record + 8u, &offset, 8u);
        memcpy(record + 16u, &bytes, 8u);
        memcpy(record + 24u, digest, 16u);
        if (write(out, record, sizeof(record)) != sizeof(record))
        {
            return 2;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &now);
    total_end_ns = (uint64_t)now.tv_sec * 1000000000ull + (uint64_t)now.tv_nsec;
#if !defined(__APPLE__)
    (void)posix_fadvise(in, 0, 0, POSIX_FADV_DONTNEED);
#endif
    (void)close(in);
    if (close(out) != 0)
    {
        return 2;
    }
    printf("SEGMENTS file_bytes=%llu segment=%llu count=%llu ns=%llu\n",
        (unsigned long long)file_bytes, (unsigned long long)segment_bytes,
        (unsigned long long)count,
        (unsigned long long)(total_end_ns - total_begin_ns));
    return 0;
}