#pragma once
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
} SparkWeightdLazyPack;

// Startup-only, explicit budgets/identity, strict .experts, no eager fallback.
// A nonnull result on failure owns cleanup only; call Destroy again as needed.
SparkStatus SparkWeightdLazyPackCreate(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdLazyPack **out);
SparkStatus SparkWeightdLazyPackSlice(const SparkWeightdLazyPack *pack,uint64_t offset,uint64_t bytes,const void **pointer);
// Startup model validation runs after parsing and before any spine GPU allocation.
typedef SparkStatus (*SparkWeightdManifestCheck)(const SparkWeightdManifest *manifest,void *context);
SparkStatus SparkWeightdLazyPackCreateChecked(const char *socket,const SparkWeightdLazyAttachRequest *request,uint64_t spine_budget,uint64_t timeout,SparkWeightdManifestCheck check,void *context,SparkWeightdLazyPack **out);
// Caller must stop submissions and drain all GPU spine readers first. Uses the
// creating CUDA context. BUSY preserves outstanding worker/map ownership. Once
// destruction starts, only cleanup is permitted; never publish new slices.
SparkStatus SparkWeightdLazyPackDestroy(SparkWeightdLazyPack *pack);

#ifdef __cplusplus
}
#endif
