#pragma once

#include <stdint.h>

#include <cuda_runtime.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

SparkStatus SparkStageModuleCudaStatus(const char *module_tag, cudaError_t error, const char *site);
SparkStatus SparkDsv41FlashCudaContextEnsure(void);

#ifdef __cplusplus
}
#endif
