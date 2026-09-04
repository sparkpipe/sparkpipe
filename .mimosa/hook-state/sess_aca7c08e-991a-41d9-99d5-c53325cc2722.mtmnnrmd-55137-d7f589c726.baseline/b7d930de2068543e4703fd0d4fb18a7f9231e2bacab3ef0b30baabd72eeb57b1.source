#ifndef SPARKPIPE_SPARK_SHA256_H
#define SPARKPIPE_SPARK_SHA256_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_SHA256_DIGEST_BYTES 32u
#define SPARK_SHA256_HEX_BYTES 65u

typedef struct SparkSha256Context
{
    uint32_t state[8];
    uint64_t total_bytes;
    uint8_t block[64];
    uint32_t block_bytes;
} SparkSha256Context;

void SparkSha256Initialize(SparkSha256Context *context);
void SparkSha256Update(SparkSha256Context *context, const void *data, size_t data_bytes);
void SparkSha256Finalize(SparkSha256Context *context, uint8_t digest[SPARK_SHA256_DIGEST_BYTES]);
void SparkSha256DigestToHex(const uint8_t digest[SPARK_SHA256_DIGEST_BYTES], char hex[SPARK_SHA256_HEX_BYTES]);
SparkStatus SparkSha256Bytes(const void *data, size_t data_bytes, char hex[SPARK_SHA256_HEX_BYTES]);
SparkStatus SparkSha256File(const char *path, char hex[SPARK_SHA256_HEX_BYTES]);
bool SparkSha256HexIsValid(const char *hex);

#ifdef __cplusplus
}
#endif

#endif
