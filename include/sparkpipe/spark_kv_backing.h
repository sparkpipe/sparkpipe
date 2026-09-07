#ifndef SPARK_KV_BACKING_H
#define SPARK_KV_BACKING_H

#include <stdint.h>
#include <stddef.h>
#include "spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_BACKING_MAGIC "SPKVBS01"
#define SPARK_KV_BACKING_HEADER_BYTES 4096u
#define SPARK_KV_BACKING_SLOT_BYTES (4u * 1024u * 1024u)
#define SPARK_KV_BACKING_MAX_SLOT_COUNT 1000000u

typedef struct SparkKvBacking
{
	int file_descriptor;
	uint64_t slot_bytes;
	uint32_t slot_count;
	uint32_t free_hint;
	uint8_t *slot_live;
	uint32_t live_count;
} SparkKvBacking;

typedef struct SparkKvBackingConfiguration
{
	const char *path;
	uint64_t maximum_bytes;
	uint32_t reserved;
} SparkKvBackingConfiguration;


SparkStatus SparkKvBackingResolvePath(char *path_out, size_t path_out_bytes,
	const char *root_directory, const char *deployment_id,
	const char *tenant_id, const char *model_id);

SparkStatus SparkKvBackingCreateNamespaces(const char *root_directory,
	const char *deployment_id, const char *tenant_id);

SparkStatus SparkKvBackingOpen(const SparkKvBackingConfiguration *configuration,
	SparkKvBacking *backing);
void SparkKvBackingClose(SparkKvBacking *backing);

int64_t SparkKvBackingAllocate(SparkKvBacking *backing);
void SparkKvBackingRelease(SparkKvBacking *backing, uint32_t slot);

SparkStatus SparkKvBackingWriteBlock(SparkKvBacking *backing, uint32_t slot,
	const void *host_buffer);
SparkStatus SparkKvBackingReadBlock(SparkKvBacking *backing, uint32_t slot,
	void *host_buffer);

uint64_t SparkKvBackingLiveBytes(const SparkKvBacking *backing);

#ifdef __cplusplus
}
#endif

#endif
