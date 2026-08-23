#pragma once

/* C-safe prototypes for the shared stage-module device helpers (see
 * stage_module_kernels.cuh for the bodies and the bit-faithfulness note).
 * Family nvcc TUs define these on top of the shared launchers; host module
 * translation units call them through this header without touching C++. */

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

cudaError_t SparkStageLaunchAccumAdd(cudaStream_t stream,void *destination_bf16,const void *source_bf16,uint32_t row_count,uint32_t width);
cudaError_t SparkStageLaunchAccumU64Max(cudaStream_t stream,uint64_t *destination,const uint64_t *source,uint32_t element_count);

#ifdef __cplusplus
}
#endif
