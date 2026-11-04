#pragma once
#include "sparkpipe/spark_weightd.h"
#include "sparkpipe/spark_weightd_manifest.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdMap SparkWeightdMap;
typedef struct SparkWeightdLeaseSets SparkWeightdLeaseSets;
typedef SparkStatus (*SparkWeightdLeaseRecord)(void *context,SparkWeightdMap *map,uint64_t identifier);

SparkStatus SparkWeightdLeaseSetsCreate(uint32_t set_count,uint32_t keys_max,SparkWeightdLeaseRecord record,void *context,SparkWeightdLeaseSets **out);
SparkStatus SparkWeightdLeaseSetsDestroy(SparkWeightdLeaseSets *sets);
SparkStatus SparkWeightdLeaseSetsAcquire(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t timeout,void **base);
SparkStatus SparkWeightdLeaseSetsRecord(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set);
SparkStatus SparkWeightdLeaseSetsRelease(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint32_t set,uint64_t timeout);
SparkStatus SparkWeightdLeaseSetsReleaseAll(SparkWeightdLeaseSets *sets,SparkWeightdMap *map,uint64_t timeout);
uint32_t SparkWeightdLeaseSetsHeld(const SparkWeightdLeaseSets *sets,uint32_t set);
uint32_t SparkWeightdLeaseSetsCount(const SparkWeightdLeaseSets *sets);
SparkStatus SparkWeightdLeaseSetsFit(const SparkWeightdManifest *manifest,const SparkWeightdLazyAttachResult *attached,uint32_t layers,uint32_t set_count,uint64_t *widest,uint64_t *pool_chunks);

#ifdef __cplusplus
}
#endif
