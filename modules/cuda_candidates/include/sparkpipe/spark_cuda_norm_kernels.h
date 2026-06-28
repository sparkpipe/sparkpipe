#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_NORM_SENTINEL 0x535043554E4F524Dull

typedef struct SparkCudaNormRequest
{
    uint32_t row_count;
    uint32_t hidden_size;
    float epsilon;
    uint64_t sentinel;
} SparkCudaNormRequest;

typedef struct SparkCudaNormReport
{
    uint64_t element_count;
    uint32_t row_count;
    uint32_t hidden_size;
    uint32_t rms_norm_kernel_count;
    uint32_t fused_add_rms_norm_kernel_count;
    uint32_t rms_norm_fp8_quant_kernel_count;
    uint32_t sentinel_violation_count;
} SparkCudaNormReport;

SparkStatus SparkValidateCudaNormRequest(const SparkCudaNormRequest *request);
SparkStatus SparkRunCudaRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_bf16, SparkCudaNormReport *report);
SparkStatus SparkRunCudaFusedAddRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_residual_bf16, const void *device_weight_bf16, void *device_output_bf16, void *device_residual_output_bf16, SparkCudaNormReport *report);
SparkStatus SparkRunCudaRmsNormQuantFp8E4m3(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_fp8, float *device_output_scales, SparkCudaNormReport *report);

#ifdef __cplusplus
}
#endif
