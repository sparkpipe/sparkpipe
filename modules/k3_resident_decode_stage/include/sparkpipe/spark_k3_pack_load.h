#ifndef SPARKPIPE_SPARK_K3_PACK_LOAD_H
#define SPARKPIPE_SPARK_K3_PACK_LOAD_H

#include <stdint.h>
#include <sys/types.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif


#define SPARK_K3_PACK_MAGIC 0x4B33504Bu
#define SPARK_K3_PACK_FORMAT_VERSION 2u
#define SPARK_K3_PACK_ALIGNMENT 128u
#define SPARK_K3_PACK_MAX_NAME_BYTES 96u

#define SPARK_K3_PACK_KIND_BF16 0u
#define SPARK_K3_PACK_KIND_F32 1u
#define SPARK_K3_PACK_KIND_MXFP4_WS_INTERLEAVED_V1 2u

typedef struct SparkK3PackEntry
{
	char name[SPARK_K3_PACK_MAX_NAME_BYTES];
	uint64_t payload_offset;
	uint64_t bytes;
	uint32_t kind;
	uint32_t shape_count;
	uint32_t shape[4];
} SparkK3PackEntry;

typedef struct SparkK3PackConfig
{
	uint32_t hidden;
	uint32_t layers;
	uint32_t first_layer;
	uint32_t total_layers;
	uint32_t experts;
	uint32_t top_k;
	uint32_t latent;
	uint32_t intermediate;
	uint32_t group;
	uint32_t vocab;
	uint32_t kda_heads;
	uint32_t kda_head;
	uint32_t heads;
	uint32_t kv_lora;
	uint32_t rope;
	uint32_t v_head;
	uint32_t nope;
	uint32_t shared;
	uint32_t q_lora;
} SparkK3PackConfig;

typedef struct SparkK3Pack
{
	int fd;
	uint8_t *mapping;
	uint64_t file_bytes;
	uint64_t payload_base;
	uint32_t version;
	SparkK3PackConfig config;
	struct SparkK3PackPrivate *private_state;
} SparkK3Pack;

SparkStatus SparkK3PackOpen(const char *path, SparkK3Pack *pack);
void SparkK3PackClose(SparkK3Pack *pack);
SparkStatus SparkK3PackLoadEntry(SparkK3Pack *pack, const char *name,
	SparkK3PackEntry *entry);
SparkStatus SparkK3PackLoadInterleaveTileK(SparkK3Pack *pack, const char *name,
	uint32_t *tile_k);
const void *SparkK3PackPayload(const SparkK3Pack *pack,
	const SparkK3PackEntry *entry);

#ifdef __cplusplus
}
#endif

#endif
