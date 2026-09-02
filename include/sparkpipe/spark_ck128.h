#pragma once

#include <stddef.h>
#include <stdint.h>

#define SPARK_CK128_HEX_BYTES 33u

typedef struct SparkCk128Context
{
    uint64_t h1;
    uint64_t h2;
    uint64_t total_bytes;
    uint8_t tail[16];
    uint32_t tail_bytes;
} SparkCk128Context;

void SparkCk128Initialize(SparkCk128Context *context);
void SparkCk128Update(SparkCk128Context *context, const void *data, size_t bytes);
void SparkCk128Finalize(SparkCk128Context *context, uint8_t digest[16]);
void SparkCk128DigestToHex(const uint8_t digest[16], char hex[SPARK_CK128_HEX_BYTES]);
