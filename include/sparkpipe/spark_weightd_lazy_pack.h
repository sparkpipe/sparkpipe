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

SparkStatus SparkWeightdLazyPackCreate(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdLazyPack **out);
SparkStatus SparkWeightdLazyPackSlice(const SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,const void **pointer);
typedef SparkStatus (*SparkWeightdManifestCheck)(const SparkWeightdManifest *manifest,void *context);
SparkStatus SparkWeightdLazyPackCreateChecked(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context,SparkWeightdLazyPack **out);
SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack);

#ifdef __cplusplus
}
#endif
