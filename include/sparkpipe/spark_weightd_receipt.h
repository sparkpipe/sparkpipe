#pragma once
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include "sparkpipe/spark_status.h"

#define SPARK_WEIGHTD_RECEIPT_SUFFIX ".verified"
#define SPARK_WEIGHTD_RECEIPT_BYTES_MAX 1024u
#define SPARK_WEIGHTD_RECEIPT_PATH_BYTES 4200u
#define SPARK_WEIGHTD_RECEIPT_VERIFIER_BYTES 96u
#define SPARK_WEIGHTD_RECEIPT_SETTLE_NS UINT64_C(1000000000)
#define SPARK_WEIGHTD_STREAM_CHUNK_BYTES (UINT64_C(32) << 20)
#define SPARK_WEIGHTD_STREAM_DEPTH 3u
#define SPARK_WEIGHTD_STREAM_ALIGN 4096u

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdReceipt
{
	uint64_t device;
	uint64_t inode;
	uint64_t size;
	uint64_t mtime_ns;
	uint64_t ctime_ns;
	uint64_t verified_unix_ns;
	char sha256[65];
	char ck128[33];
	char verifier[SPARK_WEIGHTD_RECEIPT_VERIFIER_BYTES];
} SparkWeightdReceipt;

typedef SparkStatus (*SparkWeightdStreamSink)(void *context,const uint8_t *data,uint64_t offset,uint64_t bytes);

typedef struct SparkWeightdStreamStats
{
	uint64_t bytes;
	uint64_t elapsed_ns;
	uint64_t wait_ns;
	uint64_t sink_ns;
	uint32_t direct;
} SparkWeightdStreamStats;

uint64_t SparkWeightdPackMtimeNs(const struct stat *status);
uint64_t SparkWeightdPackCtimeNs(const struct stat *status);
int32_t SparkWeightdPackStatSame(const struct stat *before,const struct stat *after);

SparkStatus SparkWeightdPackDigestRead(const char *pack_path,char hex[65]);
SparkStatus SparkWeightdReceiptLocate(const char *pack_path,const struct stat *pack,char primary[SPARK_WEIGHTD_RECEIPT_PATH_BYTES],char fallback[SPARK_WEIGHTD_RECEIPT_PATH_BYTES]);
SparkStatus SparkWeightdReceiptLoad(const char *receipt_path,SparkWeightdReceipt *out);
SparkStatus SparkWeightdReceiptCheck(const char *pack_path,int32_t fd,const char *sha256_hex,const char *ck128_hex,const char **reason);
SparkStatus SparkWeightdReceiptRecord(const char *pack_path,int32_t fd,const struct stat *verified,const char *sha256_hex,const char *ck128_hex,const char *role);

SparkStatus SparkWeightdPackStream(const char *pack_path,int32_t fd,uint64_t bytes,SparkWeightdStreamSink sink,void *context,SparkWeightdStreamStats *stats);

#ifdef __cplusplus
}
#endif
