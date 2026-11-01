#pragma once

#include <stdint.h>

#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_KV_SNAPSHOT_ABI_VERSION 6u
#define SPARK_KV_SNAPSHOT_FORMAT_VERSION 2u
#define SPARK_KV_SNAPSHOT_MAGIC UINT64_C(0x31504e534b565053)
#define SPARK_KV_SNAPSHOT_MAX_SEGMENTS 4u
#define SPARK_KV_SNAPSHOT_PATH_BYTES 1024u
#define SPARK_KV_SNAPSHOT_ALIGNMENT UINT64_C(4096)
#define SPARK_KV_SNAPSHOT_SUFFIX ".kvs"
#define SPARK_KV_SNAPSHOT_STATE_SUFFIX ".kvst"
#define SPARK_KV_SNAPSHOT_FILE_PAGES 0u
#define SPARK_KV_SNAPSHOT_FILE_STATE 1u
#define SPARK_KV_SNAPSHOT_TEMPORARY_PREFIX ".kvs-writing-"
#define SPARK_KV_SNAPSHOT_LOCK_NAME ".kvs-lock"
#define SPARK_KV_SNAPSHOT_SHARD_COUNT 256u
#define SPARK_KV_SNAPSHOT_SCAN_THREADS 8u
#define SPARK_KV_RETENTION_SECONDS UINT64_C(86400)
#define SPARK_KV_EXPIRY_SWEEP_SECONDS 60u
#define SPARK_KV_SNAPSHOT_USE_STAMP_SECONDS UINT64_C(60)
#define SPARK_KV_DETACHED_POOL_SECONDS UINT64_C(3600)
#define SPARK_KV_PURGE_SECONDS 60u
#define SPARK_KV_PURGE_DEPTH_MAX 16u
#define SPARK_KV_SNAPSHOT_OPEN_WAIT_NS UINT64_C(30000000000)

typedef struct SparkKvSnapshotPurgeResult
{
	uint64_t deleted_files;
	uint64_t deleted_bytes;
	uint32_t stores;
	uint32_t busy_stores;
	uint32_t failed_unlinks;
	uint32_t reserved0;
} SparkKvSnapshotPurgeResult;

typedef struct SparkKvSnapshotKey
{
	uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t identity_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t token_count;
	uint32_t file_kind;
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
	SparkKvSnapshotKey parent;
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
	uint64_t queue_maximum_bytes;
	uint64_t reserve_bytes;
	uint64_t used_bytes;
	uint64_t file_count;
	uint64_t write_count;
	uint64_t write_bytes;
	uint64_t write_ns;
	uint64_t write_failure_count;
	uint64_t queued_count;
	uint64_t queued_bytes;
	uint64_t queue_full_count;
	uint64_t spare_bytes;
	uint64_t spare_maximum_bytes;
	uint64_t duplicate_count;
	uint64_t eviction_count;
	uint64_t evicted_bytes;
	uint64_t reserve_eviction_count;
	uint64_t reserve_refusal_count;
	uint64_t pruned_count;
	uint64_t read_count;
	uint64_t read_bytes;
	uint64_t read_ns;
	uint64_t miss_count;
	uint64_t checksum_failure_count;
	uint64_t removed_temporary_count;
	uint64_t removed_legacy_count;
	uint64_t orphan_count;
	uint64_t expired_count;
	uint64_t expired_bytes;
	uint64_t expiry_sweep_seconds;
	SparkStatus failed_status;
	uint32_t reserved0;
	uint64_t failed_at_ns;
	uint64_t failure_retry_ns;
	uint64_t recovered_count;
	void *runtime;
} SparkKvSnapshotStore;

typedef struct SparkKvSnapshotWriteTicket
{
	SparkKvSnapshotKey key;
	SparkKvSnapshotKey parent;
	uint32_t segment_count;
	uint32_t reserved0;
	uint64_t file_bytes;
	SparkKvSnapshotSegment segments[SPARK_KV_SNAPSHOT_MAX_SEGMENTS];
	void *job;
} SparkKvSnapshotWriteTicket;

#define SPARK_KV_SNAPSHOT_STORE_BYTES ((uint32_t)sizeof(SparkKvSnapshotStore))
#define SPARK_KV_SNAPSHOT_FAILURE_RETRY_NS UINT64_C(10000000000)

SparkStatus SparkKvSnapshotStoreOpen(SparkKvSnapshotStore *store,const char *directory,uint64_t maximum_bytes,uint64_t queue_maximum_bytes,uint64_t reserve_bytes);
void SparkKvSnapshotStoreClose(SparkKvSnapshotStore *store);
SparkStatus SparkKvSnapshotStoreSample(SparkKvSnapshotStore *store,SparkKvSnapshotStore *sample);
SparkStatus SparkKvSnapshotExpire(SparkKvSnapshotStore *store,uint64_t now_seconds);
SparkStatus SparkKvSnapshotPrune(SparkKvSnapshotStore *store,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES]);
SparkStatus SparkKvSnapshotCountLayout(SparkKvSnapshotStore *store,const uint8_t layout_sha256[SPARK_SHA256_DIGEST_BYTES],uint64_t *matching_files,uint64_t *foreign_files);
SparkStatus SparkKvSnapshotBinaryDigest(const void *symbol,uint8_t digest[SPARK_SHA256_DIGEST_BYTES],char *path,uint32_t path_capacity);
SparkStatus SparkKvSnapshotPath(const SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,char *path,uint32_t path_capacity);
SparkStatus SparkKvSnapshotWrite(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const SparkKvSnapshotSegment *segments,uint32_t segment_count);
SparkStatus SparkKvSnapshotWriteBegin(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,const SparkKvSnapshotKey *parent,const uint32_t *kinds,const uint64_t *bytes,uint32_t segment_count,uint64_t used_seconds,SparkKvSnapshotWriteTicket *ticket);
SparkStatus SparkKvSnapshotParent(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotKey *parent_out);
uint32_t SparkKvSnapshotContains(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key);
SparkStatus SparkKvSnapshotWriteCommit(SparkKvSnapshotStore *store,SparkKvSnapshotWriteTicket *ticket);
void SparkKvSnapshotWriteCancel(SparkKvSnapshotStore *store,SparkKvSnapshotWriteTicket *ticket);
SparkStatus SparkKvSnapshotFlush(SparkKvSnapshotStore *store);
SparkStatus SparkKvSnapshotStat(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotFileHeader *header);
SparkStatus SparkKvSnapshotRead(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,SparkKvSnapshotSegment *segments,uint32_t segment_count);
SparkStatus SparkKvSnapshotReadSegment(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key,uint32_t segment_index,uint32_t kind,void *data,uint64_t capacity,uint64_t *bytes_out);
SparkStatus SparkKvSnapshotRemove(SparkKvSnapshotStore *store,const SparkKvSnapshotKey *key);
uint32_t SparkKvSnapshotIsCacheFileName(const char *name);
uint32_t SparkKvSnapshotIsShardName(const char *name);
SparkStatus SparkKvSnapshotPurgeDirectory(const char *directory,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result);
SparkStatus SparkKvSnapshotPurgeTree(const char *root,uint64_t now_seconds,SparkKvSnapshotPurgeResult *result);

#ifdef __cplusplus
}
#endif
