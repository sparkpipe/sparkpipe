#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_STRUCTURED_MASK_SENTINEL 0x53504B4D41534B38ull

typedef struct SparkCudaStructuredMaskRequest
{
    uint32_t row_count;
    uint32_t vocab_size;
    uint32_t allowed_token_count;
    uint32_t max_allowed_tokens_per_row;
    uint32_t compute_checksum;
    uint32_t reserved;
    float mask_value;
    uint64_t expected_output_checksum;
    uint64_t sentinel;
} SparkCudaStructuredMaskRequest;

typedef struct SparkCudaStructuredMaskReport
{
    uint64_t logit_count;
    uint64_t allowed_token_count;
    uint64_t unique_allowed_token_count;
    uint64_t masked_token_count;
    uint64_t output_checksum;
    uint64_t trace_checksum;
    uint32_t fill_kernel_count;
    uint32_t scatter_kernel_count;
    uint32_t checksum_kernel_count;
    uint32_t trace_kernel_count;
    uint32_t host_reference_count;
    uint32_t invalid_token_count;
    uint32_t duplicate_token_count;
    uint32_t empty_row_count;
    uint32_t checksum_mismatch_count;
    uint32_t sentinel_violation_count;
} SparkCudaStructuredMaskReport;

uint64_t SparkComputeCudaStructuredMaskHostChecksum(const float *values, uint64_t value_count);
SparkStatus SparkValidateCudaStructuredMaskRequest(const SparkCudaStructuredMaskRequest *request, const uint32_t *allowed_offsets, uint32_t allowed_offset_count);
SparkStatus SparkApplyStructuredMaskHostReference(const SparkCudaStructuredMaskRequest *request, const float *input_host, uint64_t input_host_values, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report);
SparkStatus SparkRunCudaStructuredMaskCandidateDeviceKernels(const SparkCudaStructuredMaskRequest *request, const float *device_input_logits, uint64_t input_value_count, const uint32_t *device_allowed_offsets, uint32_t allowed_offset_count, const uint32_t *device_allowed_tokens, uint32_t allowed_token_count, float *device_candidate_logits, uint32_t candidate_value_count, SparkCudaStructuredMaskReport *device_report);
SparkStatus SparkRunCudaStructuredMaskKernels(const SparkCudaStructuredMaskRequest *request, const float *input_host, uint64_t input_host_values, const uint32_t *allowed_offsets, uint32_t allowed_offset_count, const uint32_t *allowed_tokens, uint32_t allowed_token_count, float *masked_host, uint64_t masked_host_values, SparkCudaStructuredMaskReport *report);

#ifdef __cplusplus
}
#endif
