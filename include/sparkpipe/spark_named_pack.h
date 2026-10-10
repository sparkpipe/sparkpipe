#ifndef SPARKPIPE_SPARK_NAMED_PACK_H
#define SPARKPIPE_SPARK_NAMED_PACK_H

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_NAMED_PACK_MAGIC 0x504E5053u
#define SPARK_NAMED_PACK_VERSION 1u
#define SPARK_NAMED_PACK_ALIGNMENT 128u
#define SPARK_NAMED_PACK_MAX_NAME_BYTES 96u
#define SPARK_NAMED_PACK_MAX_KIND_BYTES 32u
#define SPARK_NAMED_PACK_MAX_SHAPE 4u

typedef struct SparkNamedPackEntry
{
	char name[SPARK_NAMED_PACK_MAX_NAME_BYTES];
	char kind[SPARK_NAMED_PACK_MAX_KIND_BYTES];
	uint64_t payload_offset;
	uint64_t bytes;
	uint32_t shape_count;
	uint32_t shape[SPARK_NAMED_PACK_MAX_SHAPE];
} SparkNamedPackEntry;

typedef struct SparkNamedPack
{
	uint64_t file_bytes;
	uint64_t payload_base;
	uint32_t magic;
	uint32_t version;
	struct SparkNamedPackPrivate *private_state;
} SparkNamedPack;

SparkStatus SparkNamedPackOpen(const char *path, uint32_t magic, uint32_t version, uint32_t alignment, SparkNamedPack *pack);
void SparkNamedPackClose(SparkNamedPack *pack);
SparkStatus SparkNamedPackEntryLoad(const SparkNamedPack *pack, const char *name, SparkNamedPackEntry *entry);
SparkStatus SparkNamedPackConfigU32(const SparkNamedPack *pack, const char *key, uint32_t *value);
SparkStatus SparkNamedPackTensorFieldU32(const SparkNamedPack *pack, const char *name, const char *object,
	const char *field, uint32_t *value);

#ifdef __cplusplus
}
#endif

#endif
