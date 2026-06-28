#pragma once

#include <stdint.h>

#include "sparkpipe/spark_glm_stage_executor.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_SENTINEL 0x5350474C4D4E5845ull
#define SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_WORKSPACE_BYTES (32u * 1024u * 1024u)

typedef struct SparkGlmNvfp4ExpertCudaRequest
{
    uint32_t row_count;
    uint32_t hidden_size;
    uint32_t intermediate_rows;
    uint64_t sentinel;
} SparkGlmNvfp4ExpertCudaRequest;

typedef struct SparkGlmNvfp4ExpertCudaReport
{
    uint64_t input_element_count;
    uint64_t intermediate_element_count;
    uint64_t output_element_count;
    uint64_t output_checksum;
    uint32_t row_count;
    uint32_t compute_row_count;
    uint32_t hidden_size;
    uint32_t intermediate_rows;
    uint32_t gate_quant_kernel_count;
    uint32_t gate_gemm_run_count;
    uint32_t up_quant_kernel_count;
    uint32_t up_gemm_run_count;
    uint32_t activation_kernel_count;
    uint32_t down_quant_kernel_count;
    uint32_t down_gemm_run_count;
    uint32_t device_allocation_count;
    uint32_t projection_cache_hit_count;
    uint32_t projection_cache_miss_count;
    uint32_t projection_cache_store_count;
    uint32_t projection_cache_eviction_count;
    uint32_t phase_code;
    uint32_t sentinel_violation_count;
} SparkGlmNvfp4ExpertCudaReport;

void SparkGlmNvfp4ExpertCudaRequestReset(SparkGlmNvfp4ExpertCudaRequest *request);
SparkStatus SparkValidateGlmNvfp4ExpertCudaRequest(const SparkGlmNvfp4ExpertCudaRequest *request, const SparkGlmStageNvfp4ExpertDescriptor *descriptor);
SparkStatus SparkRunGlmNvfp4ExpertCuda(const SparkGlmNvfp4ExpertCudaRequest *request, const SparkGlmStageNvfp4ExpertDescriptor *descriptor, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmNvfp4ExpertCudaReport *report);

#ifdef __cplusplus
}
#endif
