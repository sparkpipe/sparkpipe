#pragma once
#include <cuda_runtime.h>
#include "sparkpipe/spark_tp_device_collective.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif
cudaError_t SparkGlm5NextLaunchSumRanksF32(cudaStream_t stream,void *destination,const void *const *sources,uint32_t source_count,uint32_t element_count);
cudaError_t SparkGlm5NextLaunchSeedF32(cudaStream_t stream,float *destination,const void *a,const void *b,uint32_t element_count);
cudaError_t SparkGlm5NextLaunchAddF32(cudaStream_t stream,float *destination,const void *b,uint32_t element_count);
cudaError_t SparkGlm5NextLaunchRoundF32(cudaStream_t stream,void *destination,const float *source,uint32_t element_count);
cudaError_t SparkGlm5NextLaunchAccumAdd(cudaStream_t stream,void *destination,const void *source,uint32_t row_count,uint32_t width);
cudaError_t SparkGlm5NextLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);
#ifdef __cplusplus
}
#endif
