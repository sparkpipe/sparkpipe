#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_SAMPLER_SENTINEL 0x535053414D504C52ull
#define SPARKPIPE_CUDA_SAMPLER_MAX_TOP_K 64u
#define SPARKPIPE_CUDA_SAMPLER_ERROR_COUNTERS 4u

typedef enum SparkCudaSamplerErrorCounter
{
    SPARK_CUDA_SAMPLER_ERROR_NONFINITE_LOGIT = 0,
    SPARK_CUDA_SAMPLER_ERROR_EMPTY_FILTER = 1,
    SPARK_CUDA_SAMPLER_ERROR_BAD_UNIFORM = 2,
    SPARK_CUDA_SAMPLER_ERROR_SENTINEL = 3
} SparkCudaSamplerErrorCounter;

typedef struct SparkCudaSamplerRequest
{
    uint32_t row_count;
    uint32_t vocab_size;
    uint32_t top_k;
    uint32_t reserved;
    float temperature;
    float top_p;
    float min_p;
    float reserved_float;
    uint64_t sentinel;
} SparkCudaSamplerRequest;

typedef struct SparkCudaSamplerReport
{
    uint64_t logit_count;
    uint64_t candidate_count;
    uint32_t clear_kernel_count;
    uint32_t sampler_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t error_counter_count;
    uint32_t unsupported_shape_count;
    uint32_t sentinel_violation_count;
} SparkCudaSamplerReport;

SparkStatus SparkValidateCudaSamplerRequest(const SparkCudaSamplerRequest *request);
SparkStatus SparkRunCudaSamplerTopKTopPMinP(const SparkCudaSamplerRequest *request, const float *device_logits, const float *device_uniforms, uint32_t *device_output_token_ids, float *device_output_probabilities, uint32_t *device_error_counters, SparkCudaSamplerReport *report);

#ifdef __cplusplus
}
#endif
