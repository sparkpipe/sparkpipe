#include "sparkpipe/spark_sha256.h"

#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <unistd.h>
#include <errno.h>

#define SPARK_SHA256_BLOCK_BYTES 64u
#define SPARK_SHA256_ROUND_COUNT 64u
#define SPARK_SHA256_READ_BUFFER_BYTES 65536u
#define SPARK_SHA256_PIPELINE_BUFFER_BYTES (4ull * 1024ull * 1024ull)
#define SPARK_SHA256_PIPELINE_BUFFERS 2u

static const uint32_t SparkSha256RoundConstants[SPARK_SHA256_ROUND_COUNT] =
{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
};

static uint32_t SparkSha256RotateRight(uint32_t value, uint32_t bits)
{
    return (value >> bits) | (value << (32u - bits));
}

static uint32_t SparkSha256LoadBigEndianU32(const uint8_t *bytes)
{
    return ((uint32_t)bytes[0] << 24u) |
           ((uint32_t)bytes[1] << 16u) |
           ((uint32_t)bytes[2] << 8u) |
           (uint32_t)bytes[3];
}

static void SparkSha256StoreBigEndianU64(uint8_t *bytes, uint64_t value)
{
    uint32_t byte_index;

    for (byte_index = 0u; byte_index < 8u; ++byte_index)
    {
        bytes[7u - byte_index] = (uint8_t)(value & 0xffu);
        value >>= 8u;
    }
}

static void SparkSha256Transform(uint32_t state[8], const uint8_t block[SPARK_SHA256_BLOCK_BYTES])
{
    uint32_t words[SPARK_SHA256_ROUND_COUNT];
    uint32_t working[8];
    uint32_t round_index;

    for (round_index = 0u; round_index < 16u; ++round_index)
    {
        words[round_index] = SparkSha256LoadBigEndianU32(block + (round_index * 4u));
    }
    for (round_index = 16u; round_index < SPARK_SHA256_ROUND_COUNT; ++round_index)
    {
        uint32_t sigma_zero;
        uint32_t sigma_one;

        sigma_zero = SparkSha256RotateRight(words[round_index - 15u], 7u) ^
                     SparkSha256RotateRight(words[round_index - 15u], 18u) ^
                     (words[round_index - 15u] >> 3u);
        sigma_one = SparkSha256RotateRight(words[round_index - 2u], 17u) ^
                    SparkSha256RotateRight(words[round_index - 2u], 19u) ^
                    (words[round_index - 2u] >> 10u);
        words[round_index] = words[round_index - 16u] + sigma_zero + words[round_index - 7u] + sigma_one;
    }

    memcpy(working, state, sizeof(working));
    for (round_index = 0u; round_index < SPARK_SHA256_ROUND_COUNT; ++round_index)
    {
        uint32_t choice;
        uint32_t majority;
        uint32_t sum_zero;
        uint32_t sum_one;
        uint32_t temporary_one;
        uint32_t temporary_two;

        sum_one = SparkSha256RotateRight(working[4], 6u) ^
                  SparkSha256RotateRight(working[4], 11u) ^
                  SparkSha256RotateRight(working[4], 25u);
        choice = (working[4] & working[5]) ^ ((~working[4]) & working[6]);
        temporary_one = working[7] + sum_one + choice + SparkSha256RoundConstants[round_index] + words[round_index];
        sum_zero = SparkSha256RotateRight(working[0], 2u) ^
                   SparkSha256RotateRight(working[0], 13u) ^
                   SparkSha256RotateRight(working[0], 22u);
        majority = (working[0] & working[1]) ^ (working[0] & working[2]) ^ (working[1] & working[2]);
        temporary_two = sum_zero + majority;

        working[7] = working[6];
        working[6] = working[5];
        working[5] = working[4];
        working[4] = working[3] + temporary_one;
        working[3] = working[2];
        working[2] = working[1];
        working[1] = working[0];
        working[0] = temporary_one + temporary_two;
    }

    for (round_index = 0u; round_index < 8u; ++round_index)
    {
        state[round_index] += working[round_index];
    }
}

/* Bulk transform: `block_count` consecutive 64-byte blocks folded into
 * `state` in order. The portable loop and the ARMv8 SHA-2 crypto-extension
 * loop below are two implementations of the same FIPS 180-4 function -
 * the digest is identical either way (proven per call site by the lane
 * tests, which cross-check both against the NIST vectors). */
static void SparkSha256TransformBlocksPortability(
    uint32_t state[8],
    const uint8_t *data,
    size_t block_count)
{
    while (block_count != 0u)
    {
        SparkSha256Transform(state, data);
        data += SPARK_SHA256_BLOCK_BYTES;
        block_count--;
    }
}

typedef void (*SparkSha256TransformBlocksFunction)(
    uint32_t state[8],
    const uint8_t *data,
    size_t block_count);

/* selected once per process by the first hash call; points at the
 * accelerated transform when this CPU has aarch64 FEAT_SHA2 */
static SparkSha256TransformBlocksFunction g_transform_blocks =
    SparkSha256TransformBlocksPortability;

#if defined(__aarch64__) && defined(__ARM_NEON) && \
    (defined(__GNUC__) || defined(__clang__)) && !defined(__CUDACC__)
#include <arm_neon.h>
#if defined(__GNUC__) && !defined(__clang__)
#define SPARK_SHA256_SHA2_TARGET __attribute__((target("arch=armv8-a+sha2")))
#else
#define SPARK_SHA256_SHA2_TARGET __attribute__((target("sha2")))
#endif

/* aarch64 FEAT_SHA2 path: the h/h2 pairs run four rounds each (the
 * round constants are added to the message words here), and the su0/su1
 * pairs expand the schedule. Same compression function, same digest. */
SPARK_SHA256_SHA2_TARGET
static void SparkSha256TransformBlocksSha2(
    uint32_t state[8],
    const uint8_t *data,
    size_t block_count)
{
    uint32x4_t state_save_abcd;
    uint32x4_t state_save_efgh;
    uint32x4_t abcd;
    uint32x4_t efgh;
    uint32x4_t m[4];
    size_t quad;

    while (block_count != 0u)
    {
        const uint32_t *constants = SparkSha256RoundConstants;
        state_save_abcd = vld1q_u32(state);
        state_save_efgh = vld1q_u32(state + 4u);
        abcd = state_save_abcd;
        efgh = state_save_efgh;
        m[0] = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(
            vld1q_u8(data))));
        m[1] = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(
            vld1q_u8(data + 16u))));
        m[2] = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(
            vld1q_u8(data + 32u))));
        m[3] = vreinterpretq_u32_u8(vrev32q_u8(vreinterpretq_u8_u32(
            vld1q_u8(data + 48u))));
#define SPARK_SHA256_SHA2_ROUND(message_quad)                                  \
        do                                                                     \
        {                                                                      \
            uint32x4_t message =                                               \
                vaddq_u32((message_quad), vld1q_u32(constants));               \
            uint32x4_t abcd_before = abcd;                                     \
            constants += 4u;                                                   \
            abcd = vsha256hq_u32(abcd, efgh, message);                         \
            efgh = vsha256h2q_u32(efgh, abcd_before, message);                 \
        } while (0)
        SPARK_SHA256_SHA2_ROUND(m[0]);
        SPARK_SHA256_SHA2_ROUND(m[1]);
        SPARK_SHA256_SHA2_ROUND(m[2]);
        SPARK_SHA256_SHA2_ROUND(m[3]);
        for (quad = 4u; quad < 16u; ++quad)
        {
            /* m[quad&3] holds W[quad-4], m[(quad+1)&3] W[quad-3],
             * m[(quad+2)&3] W[quad-2], m[(quad+3)&3] W[quad-1] */
            uint32x4_t schedule_partial =
                vsha256su0q_u32(m[quad & 3u], m[(quad + 1u) & 3u]);
            m[quad & 3u] = vsha256su1q_u32(
                schedule_partial, m[(quad + 2u) & 3u], m[(quad + 3u) & 3u]);
            SPARK_SHA256_SHA2_ROUND(m[quad & 3u]);
        }
#undef SPARK_SHA256_SHA2_ROUND
        vst1q_u32(state, vaddq_u32(abcd, state_save_abcd));
        vst1q_u32(state + 4u, vaddq_u32(efgh, state_save_efgh));
        data += 64u;
        block_count--;
    }
}

#if defined(__APPLE__)
#include <sys/sysctl.h>
static int SparkSha256CpuHasSha2(void)
{
    int value = 0;
    size_t value_size = sizeof(value);
    return sysctlbyname(
        "hw.optional.arm.FEAT_SHA256", &value, &value_size, 0, 0) == 0 &&
        value == 1;
}
#elif defined(__linux__)
#include <sys/auxv.h>
#ifndef HWCAP_SHA2
#define HWCAP_SHA2 (1u << 6)
#endif
static int SparkSha256CpuHasSha2(void)
{
    return (getauxval(AT_HWCAP) & (unsigned long)HWCAP_SHA2) != 0u;
}
#else
static int SparkSha256CpuHasSha2(void)
{
    return 0;
}
#endif

static SparkSha256TransformBlocksFunction SparkSha256TransformForThisCpu(void)
{
    /* the choice is a pure function of the CPU; racing first calls all
     * write the same pointer */
    static atomic_uint selected;
    unsigned int expected = 0u;
    if (atomic_compare_exchange_strong(&selected, &expected, 1u))
    {
        if (SparkSha256CpuHasSha2())
        {
            g_transform_blocks = SparkSha256TransformBlocksSha2;
        }
    }
    return g_transform_blocks;
}
#else
static SparkSha256TransformBlocksFunction SparkSha256TransformForThisCpu(void)
{
    return SparkSha256TransformBlocksPortability;
}
#endif

void SparkSha256Initialize(SparkSha256Context *context)
{
    if (context == 0)
    {
        return;
    }

    context->state[0] = 0x6a09e667u;
    context->state[1] = 0xbb67ae85u;
    context->state[2] = 0x3c6ef372u;
    context->state[3] = 0xa54ff53au;
    context->state[4] = 0x510e527fu;
    context->state[5] = 0x9b05688cu;
    context->state[6] = 0x1f83d9abu;
    context->state[7] = 0x5be0cd19u;
    context->total_bytes = 0u;
    context->block_bytes = 0u;
    memset(context->block, 0, sizeof(context->block));
}

void SparkSha256Update(SparkSha256Context *context, const void *data, size_t data_bytes)
{
    const uint8_t *input;
    size_t remaining_bytes;

    if (context == 0 || (data == 0 && data_bytes != 0u))
    {
        return;
    }

    input = (const uint8_t *)data;
    remaining_bytes = data_bytes;
    context->total_bytes += (uint64_t)data_bytes;

    if (context->block_bytes != 0u)
    {
        size_t available_bytes;
        size_t copy_bytes;

        available_bytes = SPARK_SHA256_BLOCK_BYTES - context->block_bytes;
        copy_bytes = remaining_bytes < available_bytes ? remaining_bytes : available_bytes;
        memcpy(context->block + context->block_bytes, input, copy_bytes);
        context->block_bytes += (uint32_t)copy_bytes;
        input += copy_bytes;
        remaining_bytes -= copy_bytes;

        if (context->block_bytes == SPARK_SHA256_BLOCK_BYTES)
        {
            SparkSha256Transform(context->state, context->block);
            context->block_bytes = 0u;
        }
    }
    if (remaining_bytes >= SPARK_SHA256_BLOCK_BYTES)
    {
        /* bulk: complete blocks go straight from the caller's buffer
         * through the per-CPU transform (portable or FEAT_SHA2) */
        size_t block_count = remaining_bytes / SPARK_SHA256_BLOCK_BYTES;
        SparkSha256TransformForThisCpu()(context->state, input, block_count);
        input += block_count * SPARK_SHA256_BLOCK_BYTES;
        remaining_bytes -= block_count * SPARK_SHA256_BLOCK_BYTES;
    }
    if (remaining_bytes != 0u)
    {
        memcpy(context->block, input, remaining_bytes);
        context->block_bytes = (uint32_t)remaining_bytes;
    }
}

void SparkSha256Finalize(SparkSha256Context *context, uint8_t digest[SPARK_SHA256_DIGEST_BYTES])
{
    uint64_t total_bits;
    uint32_t state_index;

    if (context == 0 || digest == 0)
    {
        return;
    }

    total_bits = context->total_bytes * 8u;
    context->block[context->block_bytes++] = 0x80u;
    if (context->block_bytes > 56u)
    {
        memset(context->block + context->block_bytes, 0, SPARK_SHA256_BLOCK_BYTES - context->block_bytes);
        SparkSha256Transform(context->state, context->block);
        context->block_bytes = 0u;
    }
    memset(context->block + context->block_bytes, 0, 56u - context->block_bytes);
    SparkSha256StoreBigEndianU64(context->block + 56u, total_bits);
    SparkSha256Transform(context->state, context->block);

    for (state_index = 0u; state_index < 8u; ++state_index)
    {
        digest[(state_index * 4u) + 0u] = (uint8_t)(context->state[state_index] >> 24u);
        digest[(state_index * 4u) + 1u] = (uint8_t)(context->state[state_index] >> 16u);
        digest[(state_index * 4u) + 2u] = (uint8_t)(context->state[state_index] >> 8u);
        digest[(state_index * 4u) + 3u] = (uint8_t)context->state[state_index];
    }

    memset(context, 0, sizeof(*context));
}

void SparkSha256DigestToHex(const uint8_t digest[SPARK_SHA256_DIGEST_BYTES], char hex[SPARK_SHA256_HEX_BYTES])
{
    static const char HexDigits[] = "0123456789abcdef";
    uint32_t byte_index;

    if (digest == 0 || hex == 0)
    {
        return;
    }

    for (byte_index = 0u; byte_index < SPARK_SHA256_DIGEST_BYTES; ++byte_index)
    {
        hex[byte_index * 2u] = HexDigits[digest[byte_index] >> 4u];
        hex[(byte_index * 2u) + 1u] = HexDigits[digest[byte_index] & 0x0fu];
    }
    hex[SPARK_SHA256_HEX_BYTES - 1u] = '\0';
}

SparkStatus SparkSha256Bytes(const void *data, size_t data_bytes, char hex[SPARK_SHA256_HEX_BYTES])
{
    SparkSha256Context context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];

    if ((data == 0 && data_bytes != 0u) || hex == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    SparkSha256Initialize(&context);
    SparkSha256Update(&context, data, data_bytes);
    SparkSha256Finalize(&context, digest);
    SparkSha256DigestToHex(digest, hex);
    return SPARK_STATUS_OK;
}

/*
 * Whole-file digest (docs/WEIGHTD_DESIGN.md L2). FIPS 180-4 SHA-256 is a
 * serial chaining: block N's compression consumes block N-1's state, so
 * NO multi-thread split of one byte stream can reproduce the sequential
 * digest - and identity keys (pack SHA-256 in attach identity) require
 * exactly that value. "Parallel pre-hash then serial-combine" produces a
 * different (tree) digest and is refused here. What CAN run in parallel
 * is the read pass: a reader thread fills a double buffer ahead of the
 * hashing thread, chunks are still hashed strictly in file order, and
 * the digest is bit-identical to the sequential pass by construction.
 * SPARK_SHA256_FILE_PIPELINE=0 forces the sequential reader (A/B).
 */
typedef struct SparkSha256FilePipeline
{
    int file_descriptor;
    pthread_t reader_thread;
    int reader_started;
    pthread_mutex_t mutex;
    pthread_cond_t progress;
    /* host heap staging: fed to the CPU hashing thread only */
    uint8_t *buffers[SPARK_SHA256_PIPELINE_BUFFERS];
    size_t buffer_bytes[SPARK_SHA256_PIPELINE_BUFFERS];
    uint64_t filled_count;
    uint64_t hashed_count;
    int end_of_file;
    int failure;
} SparkSha256FilePipeline;

static void *SparkSha256FilePipelineReader(void *argument)
{
    SparkSha256FilePipeline *pipeline = (SparkSha256FilePipeline *)argument;
    uint64_t fill_index = 0u;
    for (;;)
    {
        size_t read_bytes = 0u;
        pthread_mutex_lock(&pipeline->mutex);
        while (pipeline->failure == 0 &&
            fill_index >= pipeline->hashed_count + SPARK_SHA256_PIPELINE_BUFFERS)
        {
            /* both buffers hold unhashed data: wait for the hash thread */
            pthread_cond_wait(&pipeline->progress, &pipeline->mutex);
        }
        pthread_mutex_unlock(&pipeline->mutex);
        if (pipeline->failure != 0)
        {
            break;
        }
        for (;;)
        {
            ssize_t chunk = pread(
                pipeline->file_descriptor,
                pipeline->buffers[fill_index % SPARK_SHA256_PIPELINE_BUFFERS] +
                    read_bytes,
                (size_t)SPARK_SHA256_PIPELINE_BUFFER_BYTES - read_bytes,
                (off_t)(fill_index * SPARK_SHA256_PIPELINE_BUFFER_BYTES +
                    read_bytes));
            if (chunk <= 0)
            {
                if (chunk < 0 && errno == EINTR)
                {
                    continue;
                }
                if (chunk < 0)
                {
                    pipeline->failure = 1;
                }
                else
                {
                    pipeline->end_of_file = 1;
                }
                break;
            }
            read_bytes += (size_t)chunk;
            if (read_bytes == (size_t)SPARK_SHA256_PIPELINE_BUFFER_BYTES)
            {
                break;
            }
        }
        pthread_mutex_lock(&pipeline->mutex);
        pipeline->buffer_bytes[
            fill_index % SPARK_SHA256_PIPELINE_BUFFERS] = read_bytes;
        pipeline->filled_count = fill_index + 1u;
        pthread_cond_broadcast(&pipeline->progress);
        pthread_mutex_unlock(&pipeline->mutex);
        if (pipeline->end_of_file != 0 || pipeline->failure != 0)
        {
            break;
        }
        fill_index++;
    }
    return 0;
}

static SparkStatus SparkSha256FilePipelined(
    int file_descriptor,
    SparkSha256Context *context)
{
    SparkSha256FilePipeline pipeline;
    uint64_t hash_index = 0u;
    int index;
    if (pthread_mutex_init(&pipeline.mutex, 0) != 0)
    {
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    if (pthread_cond_init(&pipeline.progress, 0) != 0)
    {
        pthread_mutex_destroy(&pipeline.mutex);
        return SPARK_STATUS_INTERNAL_ERROR;
    }
    pipeline.file_descriptor = file_descriptor;
    pipeline.reader_started = 0;
    pipeline.end_of_file = 0;
    pipeline.failure = 0;
    pipeline.filled_count = 0u;
    pipeline.hashed_count = 0u;
    for (index = 0; index < (int)SPARK_SHA256_PIPELINE_BUFFERS; index++)
    {
        /* host heap staging (one buffer per pipeline slot) */
        pipeline.buffers[index] =
            (uint8_t *)malloc((size_t)SPARK_SHA256_PIPELINE_BUFFER_BYTES);
        pipeline.buffer_bytes[index] = 0u;
        if (pipeline.buffers[index] == 0)
        {
            pipeline.failure = 1;
        }
    }
    if (pipeline.failure == 0 &&
        pthread_create(
            &pipeline.reader_thread, 0, SparkSha256FilePipelineReader,
            &pipeline) == 0)
    {
        pipeline.reader_started = 1;
    }
    while (pipeline.reader_started != 0)
    {
        const uint8_t *data;
        size_t bytes;
        pthread_mutex_lock(&pipeline.mutex);
        while (pipeline.hashed_count == pipeline.filled_count &&
            pipeline.end_of_file == 0 && pipeline.failure == 0)
        {
            pthread_cond_wait(&pipeline.progress, &pipeline.mutex);
        }
        if (pipeline.hashed_count == pipeline.filled_count)
        {
            pthread_mutex_unlock(&pipeline.mutex);
            break;
        }
        data = pipeline.buffers[hash_index % SPARK_SHA256_PIPELINE_BUFFERS];
        bytes = pipeline.buffer_bytes[
            hash_index % SPARK_SHA256_PIPELINE_BUFFERS];
        pthread_mutex_unlock(&pipeline.mutex);
        /* chunks hash strictly in file order: the digest equals the
         * sequential one regardless of how the reads overlapped */
        SparkSha256Update(context, data, bytes);
        hash_index++;
        pthread_mutex_lock(&pipeline.mutex);
        pipeline.hashed_count = hash_index;
        pthread_cond_broadcast(&pipeline.progress);
        pthread_mutex_unlock(&pipeline.mutex);
    }
    if (pipeline.reader_started != 0)
    {
        void *reader_result = 0;
        /* the reader stopped on its own (eof or failure); reap it */
        (void)pthread_join(pipeline.reader_thread, &reader_result);
    }
    else if (pipeline.failure == 0)
    {
        /* no reader thread: hash the file on this thread instead */
        uint8_t *buffer = pipeline.buffers[0];
        if (buffer == 0)
        {
            pipeline.failure = 1;
        }
        for (;;)
        {
            ssize_t chunk = buffer == 0 ? 0
                : pread(file_descriptor, buffer,
                    (size_t)SPARK_SHA256_PIPELINE_BUFFER_BYTES,
                    (off_t)(hash_index * SPARK_SHA256_PIPELINE_BUFFER_BYTES));
            if (chunk <= 0)
            {
                if (chunk < 0 && errno == EINTR)
                {
                    continue;
                }
                if (chunk < 0)
                {
                    pipeline.failure = 1;
                }
                break;
            }
            SparkSha256Update(context, buffer, (size_t)chunk);
            hash_index++;
        }
    }
    for (index = 0; index < (int)SPARK_SHA256_PIPELINE_BUFFERS; index++)
    {
        free(pipeline.buffers[index]);
    }
    pthread_cond_destroy(&pipeline.progress);
    pthread_mutex_destroy(&pipeline.mutex);
    return pipeline.failure != 0 ? SPARK_STATUS_IO_ERROR : SPARK_STATUS_OK;
}

SparkStatus SparkSha256File(const char *path, char hex[SPARK_SHA256_HEX_BYTES])
{
    FILE *file;
    SparkSha256Context context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    const char *pipeline_setting;
    SparkStatus status;

    if (path == 0 || hex == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }

    file = fopen(path, "rb");
    if (file == 0)
    {
        return SPARK_STATUS_IO_ERROR;
    }

    SparkSha256Initialize(&context);
    pipeline_setting = getenv("SPARK_SHA256_FILE_PIPELINE");
    if (pipeline_setting != 0 && pipeline_setting[0] == '0' &&
        pipeline_setting[1] == '\0')
    {
        /* sequential reference reader (the digest is identical either way) */
        uint8_t buffer[SPARK_SHA256_READ_BUFFER_BYTES];
        size_t bytes_read;
        while ((bytes_read = fread(buffer, 1u, sizeof(buffer), file)) != 0u)
        {
            SparkSha256Update(&context, buffer, bytes_read);
        }
        if (ferror(file) != 0)
        {
            fclose(file);
            return SPARK_STATUS_IO_ERROR;
        }
        status = SPARK_STATUS_OK;
    }
    else
    {
        status = SparkSha256FilePipelined(fileno(file), &context);
    }
    if (fclose(file) != 0 && status == SPARK_STATUS_OK)
    {
        return SPARK_STATUS_IO_ERROR;
    }
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }

    SparkSha256Finalize(&context, digest);
    SparkSha256DigestToHex(digest, hex);
    return SPARK_STATUS_OK;
}

bool SparkSha256HexIsValid(const char *hex)
{
    uint32_t character_index;

    if (hex == 0 || strlen(hex) != SPARK_SHA256_HEX_BYTES - 1u)
    {
        return false;
    }

    for (character_index = 0u; character_index < SPARK_SHA256_HEX_BYTES - 1u; ++character_index)
    {
        char character;

        character = hex[character_index];
        if (!((character >= '0' && character <= '9') || (character >= 'a' && character <= 'f')))
        {
            return false;
        }
    }
    return true;
}
