#include "sparkpipe/spark_ck128.h"

#include <string.h>

static uint64_t SparkCk128Rotl64(uint64_t x, int8_t r)
{
    return (x << r) | (x >> (64 - r));
}

static uint64_t SparkCk128Fmix(uint64_t k)
{
    k ^= k >> 33;
    k *= 0xff51afd7ed558ccdull;
    k ^= k >> 33;
    k *= 0xc4ceb9fe1a85ec53ull;
    k ^= k >> 33;
    return k;
}

void SparkCk128Initialize(SparkCk128Context *context)
{
    context->h1 = 0ull;
    context->h2 = 0ull;
    context->total_bytes = 0ull;
    context->tail_bytes = 0u;
    memset(context->tail, 0, sizeof(context->tail));
}

static inline void SparkCk128Step(uint64_t *h1_io, uint64_t *h2_io, const uint8_t *block)
{
    const uint64_t c1 = 0x87c37b91114253d5ull;
    const uint64_t c2 = 0x4cf5ad432745937full;
    uint64_t h1 = *h1_io;
    uint64_t h2 = *h2_io;
    uint64_t k1;
    uint64_t k2;

    memcpy(&k1, block, 8);
    memcpy(&k2, block + 8, 8);
    k1 *= c1;
    k1 = SparkCk128Rotl64(k1, 31);
    k1 *= c2;
    h1 ^= k1;
    h1 = SparkCk128Rotl64(h1, 27);
    h1 += h2;
    h1 = h1 * 5 + 0x52dce729ull;
    k2 *= c2;
    k2 = SparkCk128Rotl64(k2, 33);
    k2 *= c1;
    h2 ^= k2;
    h2 = SparkCk128Rotl64(h2, 31);
    h2 += h1;
    h2 = h2 * 5 + 0x38495ab5ull;
    *h1_io = h1;
    *h2_io = h2;
}

void SparkCk128Update(SparkCk128Context *context, const void *data, size_t bytes)
{
    const uint8_t *cursor = (const uint8_t *)data;
    size_t whole = bytes;
    uint64_t h1 = context->h1;
    uint64_t h2 = context->h2;

    if (context->tail_bytes != 0u)
    {
        size_t want = 16u - context->tail_bytes;
        if (want > bytes)
        {
            want = bytes;
        }
        memcpy(context->tail + context->tail_bytes, cursor, want);
        context->tail_bytes += (uint32_t)want;
        cursor += want;
        whole = bytes - want;
        if (context->tail_bytes == 16u)
        {
            SparkCk128Step(&h1, &h2, context->tail);
            context->tail_bytes = 0u;
        }
    }
    while (whole >= 64u)
    {
        SparkCk128Step(&h1, &h2, cursor);
        SparkCk128Step(&h1, &h2, cursor + 16u);
        SparkCk128Step(&h1, &h2, cursor + 32u);
        SparkCk128Step(&h1, &h2, cursor + 48u);
        cursor += 64u;
        whole -= 64u;
    }
    while (whole >= 16u)
    {
        SparkCk128Step(&h1, &h2, cursor);
        cursor += 16u;
        whole -= 16u;
    }
    context->h1 = h1;
    context->h2 = h2;
    if (whole != 0u)
    {
        memcpy(context->tail, cursor, whole);
        context->tail_bytes = (uint32_t)whole;
    }
    context->total_bytes += bytes;
}

void SparkCk128Finalize(SparkCk128Context *context, uint8_t digest[16])
{
    const uint64_t c1 = 0x87c37b91114253d5ull;
    const uint64_t c2 = 0x4cf5ad432745937full;
    uint64_t k1 = 0ull;
    uint64_t k2 = 0ull;

    if (context->tail_bytes > 8u)
    {
        k2 = 0ull;
        memcpy(&k2, context->tail + 8, context->tail_bytes - 8u);
        k2 *= c2;
        k2 = SparkCk128Rotl64(k2, 33);
        k2 *= c1;
        context->h2 ^= k2;
    }
    k1 = 0ull;
    memcpy(&k1, context->tail, context->tail_bytes > 8u ? 8u : context->tail_bytes);
    k1 *= c1;
    k1 = SparkCk128Rotl64(k1, 31);
    k1 *= c2;
    context->h1 ^= k1;

    context->h1 ^= context->total_bytes;
    context->h2 ^= context->total_bytes;
    context->h1 += context->h2;
    context->h2 += context->h1;
    context->h1 = SparkCk128Fmix(context->h1);
    context->h2 = SparkCk128Fmix(context->h2);
    context->h1 += context->h2;
    context->h2 += context->h1;
    memcpy(digest, &context->h1, 8);
    memcpy(digest + 8, &context->h2, 8);
}

void SparkCk128DigestToHex(const uint8_t digest[16], char hex[SPARK_CK128_HEX_BYTES])
{
    static const char digits[] = "0123456789abcdef";
    uint32_t index;

    for (index = 0u; index < 16u; index++)
    {
        hex[index * 2u] = digits[digest[index] >> 4];
        hex[index * 2u + 1u] = digits[digest[index] & 0x0f];
    }
    hex[32] = '\0';
}
