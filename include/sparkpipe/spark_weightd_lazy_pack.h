#pragma once
#include "sparkpipe/spark_sha256.h"
#include "sparkpipe/spark_weightd_map.h"
#include "sparkpipe/spark_weightd_worker.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdLazyPack
{
	SparkWeightdManifest manifest;
	SparkWeightdPoolPlan pool;
	SparkWeightdLazyAttachResult attached;
	SparkWeightdClient *client;
	SparkWeightdMap *map;
	SparkWeightdWorker *worker;
	void *spine_allocation;
	void *spine;
	uint64_t spine_allocation_bytes;
	uint32_t ready;
	uint32_t read_only;
	uint8_t pack_sha256[SPARK_SHA256_DIGEST_BYTES];
} SparkWeightdLazyPack;

SparkStatus SparkWeightdLazyPackSlice(const SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,const void **pointer);
SparkStatus SparkWeightdLazyPackPrivateSlice(SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,void **pointer);
typedef SparkStatus (*SparkWeightdManifestCheck)(const SparkWeightdManifest *manifest,void *context);

typedef struct SparkWeightdLazyPackOptions
{
	uint64_t expert_pool_cap_bytes;
	uint64_t timeout_nanoseconds;
	uint32_t read_only;
	SparkWeightdManifestCheck manifest_check;
	void *manifest_check_context;
} SparkWeightdLazyPackOptions;

SparkStatus SparkWeightdLazyPackCreate(const char *socket,const SparkWeightdLazyAttachRequest *request,const SparkWeightdLazyPackOptions *options,SparkWeightdLazyPack **out);
SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack);

#ifdef __cplusplus
}
#endif
