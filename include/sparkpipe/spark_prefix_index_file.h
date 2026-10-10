#pragma once

#include <pthread.h>
#include <stdint.h>

#include "sparkpipe/spark_prefix_cache.h"
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARK_PREFIX_INDEX_FILE_MAGIC 0x3158444950534b53ull
#define SPARK_PREFIX_INDEX_FILE_VERSION 2u
#define SPARK_PREFIX_INDEX_FILE_PATH_BYTES 1024u

typedef struct SparkPrefixIndexFileHeader
{
	uint64_t magic;
	uint32_t version;
	uint32_t header_bytes;
	uint32_t block_token_count;
	uint32_t record_count;
	uint32_t record_bytes;
	uint32_t reserved0;
	uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t records_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint8_t header_sha256[SPARK_SHA256_DIGEST_BYTES];
} SparkPrefixIndexFileHeader;

typedef struct SparkPrefixIndexWriter
{
	char path[SPARK_PREFIX_INDEX_FILE_PATH_BYTES];
	uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES];
	uint32_t block_token_count;
	uint32_t record_capacity;
	SparkPrefixCacheCommittedRecord *records;
	uint32_t record_count;
	uint32_t pending;
	uint32_t busy;
	uint32_t stop;
	uint32_t started;
	uint32_t reserved0;
	pthread_t thread;
	pthread_mutex_t mutex;
	pthread_cond_t condition;
	uint64_t save_count;
	uint64_t save_failure_count;
	uint64_t write_ns_total;
	uint64_t write_ns_maximum;
} SparkPrefixIndexWriter;

void SparkPrefixIndexModelDigest(const char *adapter_id,const char *model_id,const char *model_revision,const char *artifact_sha256,uint8_t digest[SPARK_SHA256_DIGEST_BYTES]);
SparkStatus SparkPrefixIndexFileRead(const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,SparkPrefixCacheCommittedRecord *records,uint32_t record_capacity,uint32_t *record_count_out,const char **reason_out);
SparkStatus SparkPrefixIndexFileRemoveStale(const char *path,uint32_t *removed_out);
SparkStatus SparkPrefixIndexFileWrite(const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,const SparkPrefixCacheCommittedRecord *records,uint32_t record_count);
SparkStatus SparkPrefixIndexWriterStart(SparkPrefixIndexWriter *writer,const char *path,const uint8_t model_sha256[SPARK_SHA256_DIGEST_BYTES],uint32_t block_token_count,uint32_t record_capacity);
uint32_t SparkPrefixIndexWriterIdle(SparkPrefixIndexWriter *writer);
void SparkPrefixIndexWriterWaitIdle(SparkPrefixIndexWriter *writer);
SparkPrefixCacheCommittedRecord *SparkPrefixIndexWriterBuffer(SparkPrefixIndexWriter *writer);
void SparkPrefixIndexWriterSubmit(SparkPrefixIndexWriter *writer,uint32_t record_count);
void SparkPrefixIndexWriterStop(SparkPrefixIndexWriter *writer);
void SparkPrefixIndexWriterSample(SparkPrefixIndexWriter *writer,uint64_t *save_count,uint64_t *save_failure_count,uint64_t *write_ns_total,uint64_t *write_ns_maximum);

#ifdef __cplusplus
}
#endif
