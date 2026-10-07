#pragma once
#include <cuda_runtime_api.h>
#include "sparkpipe/spark_weightd.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SparkWeightdMap SparkWeightdMap;

SparkStatus SparkWeightdMapCreate(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,SparkWeightdMap **out);
SparkStatus SparkWeightdMapCreateAccess(SparkWeightdClient *client,const SparkWeightdLazyAttachResult *attached,int epoch_fd,int pool_fd,uint32_t read_only,SparkWeightdMap **out);
uint32_t SparkWeightdMapReadOnly(const SparkWeightdMap *map);

const void *SparkWeightdMapEpochDevice(const SparkWeightdMap *map);
SparkStatus SparkWeightdMapDestroy(SparkWeightdMap *map);

SparkStatus SparkWeightdMapAcquire(SparkWeightdMap *map,const SparkWeightdExpertKey *keys,uint32_t count,uint64_t *identifier,uint64_t timeout);
SparkStatus SparkWeightdMapBeginUse(SparkWeightdMap *map,uint64_t identifier,void **address);
SparkStatus SparkWeightdMapBase(const SparkWeightdMap *map,void **address);
SparkStatus SparkWeightdMapRecordCompletion(SparkWeightdMap *map,uint64_t identifier,cudaStream_t stream);
SparkStatus SparkWeightdMapRelease(SparkWeightdMap *map,uint64_t identifier,uint64_t timeout);

SparkStatus SparkWeightdMapPool(const SparkWeightdMap *map,const void **address);
SparkStatus SparkWeightdMapResident(SparkWeightdMap *map,uint64_t timeout,uint32_t *resident);

#ifdef __cplusplus
}
#endif
