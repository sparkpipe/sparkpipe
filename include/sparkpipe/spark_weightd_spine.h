#pragma once
#include <stdint.h>
#include "sparkpipe/spark_weightd_manifest.h"

typedef SparkStatus (*SparkWeightdSpinePace)(void *context,uint64_t bytes);

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkWeightdSpineLoad(const char *pack_path,int32_t fd,const SparkWeightdManifest *manifest,uint64_t pack_bytes,const char *sha256,void *destination,uint64_t capacity);
SparkStatus SparkWeightdSpinePreload(int32_t fd,const SparkWeightdManifest *manifest,void *device_base,SparkWeightdSpinePace pace,void *pace_context);

#ifdef __cplusplus
}
#endif
