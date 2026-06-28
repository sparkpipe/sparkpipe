#include "sparkpipe/spark_sha256.h"

#include <stdio.h>
#include <string.h>

#define SPARK_SHA256_BLOCK_BYTES 64u
#define SPARK_SHA256_ROUND_COUNT 64u
#define SPARK_SHA256_READ_BUFFER_BYTES 65536u

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

static void SparkSha256Transform(SparkSha256Context *context, const uint8_t block[SPARK_SHA256_BLOCK_BYTES])
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

    memcpy(working, context->state, sizeof(working));
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
        context->state[round_index] += working[round_index];
    }
}

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

    while (remaining_bytes != 0u)
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
            SparkSha256Transform(context, context->block);
            context->block_bytes = 0u;
        }
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
        SparkSha256Transform(context, context->block);
        context->block_bytes = 0u;
    }
    memset(context->block + context->block_bytes, 0, 56u - context->block_bytes);
    SparkSha256StoreBigEndianU64(context->block + 56u, total_bits);
    SparkSha256Transform(context, context->block);

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

SparkStatus SparkSha256File(const char *path, char hex[SPARK_SHA256_HEX_BYTES])
{
    FILE *file;
    SparkSha256Context context;
    uint8_t digest[SPARK_SHA256_DIGEST_BYTES];
    uint8_t buffer[SPARK_SHA256_READ_BUFFER_BYTES];
    size_t bytes_read;

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
    while ((bytes_read = fread(buffer, 1u, sizeof(buffer), file)) != 0u)
    {
        SparkSha256Update(&context, buffer, bytes_read);
    }
    if (ferror(file) != 0)
    {
        fclose(file);
        return SPARK_STATUS_IO_ERROR;
    }
    if (fclose(file) != 0)
    {
        return SPARK_STATUS_IO_ERROR;
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
