#pragma once

#include <stdbool.h>
#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_GLM_BF16_LINEAR_CUDA_SENTINEL 0x5350474C4D42464Cull
#define SPARKPIPE_GLM_BF16_LINEAR_CUDA_WORKSPACE_BYTES (32u * 1024u * 1024u)
#define SPARKPIPE_GLM_BF16_LINEAR_CUDA_PATH_BYTES 4096u

typedef struct SparkGlmBf16LinearCudaBinding
{
    char path[SPARKPIPE_GLM_BF16_LINEAR_CUDA_PATH_BYTES];
    uint64_t offset_bytes;
    uint64_t size_bytes;
    bool ready;
} SparkGlmBf16LinearCudaBinding;

typedef struct SparkGlmBf16LinearCudaRequest
{
    uint32_t input_count;
    uint32_t output_count;
    uint64_t sentinel;
} SparkGlmBf16LinearCudaRequest;

typedef struct SparkGlmBf16LinearCudaReport
{
    uint64_t input_element_count;
    uint64_t output_element_count;
    uint64_t weight_element_count;
    uint64_t weight_checksum;
    uint64_t output_checksum;
    uint32_t input_count;
    uint32_t output_count;
    uint32_t device_allocation_count;
    uint32_t weight_cache_hit_count;
    uint32_t weight_cache_miss_count;
    uint32_t weight_cache_store_count;
    uint32_t gemm_run_count;
    uint32_t phase_code;
    uint32_t sentinel_violation_count;
} SparkGlmBf16LinearCudaReport;

void SparkGlmBf16LinearCudaRequestReset(SparkGlmBf16LinearCudaRequest *request);
SparkStatus SparkValidateGlmBf16LinearCudaRequest(const SparkGlmBf16LinearCudaRequest *request, const SparkGlmBf16LinearCudaBinding *weight);
SparkStatus SparkRunGlmBf16LinearCuda(const SparkGlmBf16LinearCudaRequest *request, const SparkGlmBf16LinearCudaBinding *weight, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmBf16LinearCudaReport *report);

#ifdef __cplusplus
}
#endif
