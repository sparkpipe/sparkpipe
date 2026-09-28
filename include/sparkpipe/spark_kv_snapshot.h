#pragma once

#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_SNAPSHOT_ABI_VERSION 1u
#define SPARK_KV_SNAPSHOT_FORMAT_VERSION 1u
#define SPARK_KV_SNAPSHOT_MAGIC UINT64_C(0x31504e534b565053)
#define SPARK_KV_SNAPSHOT_MAX_SEGMENTS 4u
#define SPARK_KV_SNAPSHOT_PATH_BYTES 1024u
#define SPARK_KV_SNAPSHOT_ALIGNMENT UINT64_C(4096)
#define SPARK_KV_SNAPSHOT_SUFFIX ".kvs"
#define SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX ".kvs-writing-"

typedef struct SparkKvSnapshotKey
{
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t token_count;
	uint32_t reserved0;
} SparkKvSnapshotKey;

typedef struct SparkKvSnapshotSegment
{
	uint32_t kind;
	uint32_t reserved0;
	uint64_t bytes;
	void *data;
} SparkKvSnapshotSegment;

typedef struct SparkKvSnapshotFileSegment
{
	uint32_t kind;
	uint32_t reserved0;
	uint64_t offset;
	uint64_t bytes;
	uint8_t sha256[SPARK_SHA256_DIGEST_BYTES];
} SparkKvSnapshotFileSegment;

typedef struct SparkKvSnapshotFileHeader
{
	uint64_t magic;
	uint32_t format_version;
	uint32_t header_bytes;
	SparkKvSnapshotKey key;
	uint32_t segment_count;
	uint32_t reserved0;
	uint64_t file_bytes;
	SparkKvSnapshotFileSegment segments[SPARK_KV_SNAPSHOT_MAX_SEGMENTS];
	uint8_t header_sha256[SPARK_SHA256_DIGEST_BYTES];
} SparkKvSnapshotFileHeader;

typedef struct SparkKvSnapshotStore
{
	uint32_t abi_version;
	uint32_t descriptor_bytes;
	char directory[SPARK_KV_SNAPSHOT_PATH_BYTES];
	uint64_t maximum_bytes;
	uint64_t used_bytes;
	uint64_t file_count;
	uint64_t write_count;
	uint64_t write_bytes;
	uint64_t write_ns;
	uint64_t duplicate_count;
	uint64_t read_count;
	uint64_t read_bytes;
	uint64_t read_ns;
	uint64_t miss_count;
	uint64_t checksum_failure_count;
	uint64_t removed_temporary_count;
} SparkKvSnapshotStore;

#define SPARK_KV_SNAPSHOT_STORE_BYTES ((uint32_t)sizeof(SparkKvSnapshotStore))

SparkStatus SparkKvSnapshotStoreOpen(SparkKvSnapshotStore *store,const char *directory,uint64_t maximum_bytes);
void SparkKvSnapshotStoreClose(SparkKvSnapshotStore *store);
SparkStatus SparkKvSnapshotPath(const SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,char *path,uint32_t path_capacity);
SparkStatus SparkKvSnapshotWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotSegment *segments,uint32_t segment_count);
SparkStatus SparkKvSnapshotStat(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header);
SparkStatus SparkKvSnapshotRead(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotSegment *segments,uint32_t segment_count);
SparkStatus SparkKvSnapshotReadSegment(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,uint32_t segment_index,uint32_t kind,void *data,uint64_t capacity,uint64_t *bytes_out);
SparkStatus SparkKvSnapshotRemove(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key);

#ifdef __cplusplus
}
#endif
