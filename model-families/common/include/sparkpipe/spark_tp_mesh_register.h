#pragma once
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"
#include "sparkpipe/spark_gpu.h"

#ifdef __cplusplus
extern "C" {
#endif
SparkGpuError SparkTpLaunchSumRanksF32(SparkGpuStream stream,void *destination,const void *const *sources,uint32_t source_count,uint32_t element_count);
SparkGpuError SparkTpLaunchAccumAdd(SparkGpuStream stream,void *destination,const void *source,uint32_t row_count,uint32_t width);
SparkGpuError SparkTpLaunchAccumU64Max(SparkGpuStream stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);
SparkGpuError SparkTpLaunchGatherRanks(SparkGpuStream stream,void *destination,const void *const *sources,uint32_t source_count,uint32_t elements_per_rank);
#ifdef __cplusplus
}
#endif
