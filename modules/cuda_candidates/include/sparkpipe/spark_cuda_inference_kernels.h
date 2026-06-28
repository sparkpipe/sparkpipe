#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_INFERENCE_SENTINEL 0x5350494E46455231ull

typedef struct SparkCudaInferenceUtilityRequest
{
    uint32_t row_count;
    uint32_t vocab_size;
    uint32_t scale_count;
    uint32_t workspace_value_count;
    uint32_t compute_checksum;
    uint32_t reserved;
    float temperature;
    float logit_bias;
    float scale_min;
    float scale_max;
    float workspace_value;
    uint64_t expected_logits_checksum;
    uint64_t expected_workspace_checksum;
    uint64_t sentinel;
} SparkCudaInferenceUtilityRequest;

typedef struct SparkCudaInferenceUtilityReport
{
    uint64_t logit_count;
    uint64_t scale_count;
    uint64_t workspace_value_count;
    uint64_t logits_checksum;
    uint64_t workspace_checksum;
    uint64_t token_checksum;
    uint64_t trace_checksum;
    uint32_t scale_audit_kernel_count;
    uint32_t logits_transform_kernel_count;
    uint32_t greedy_argmax_kernel_count;
    uint32_t workspace_clear_kernel_count;
    uint32_t checksum_kernel_count;
    uint32_t trace_kernel_count;
    uint32_t host_reference_count;
    uint32_t nonfinite_logit_count;
    uint32_t invalid_scale_count;
    uint32_t nonfinite_scale_count;
    uint32_t zero_scale_count;
    uint32_t scale_underflow_count;
    uint32_t scale_overflow_count;
    uint32_t sentinel_violation_count;
    uint32_t checksum_mismatch_count;
} SparkCudaInferenceUtilityReport;

uint64_t SparkComputeCudaInferenceFloatHostChecksum(const float *values, uint64_t value_count);
uint64_t SparkComputeCudaInferenceU32HostChecksum(const uint32_t *values, uint64_t value_count);
SparkStatus SparkValidateCudaInferenceUtilityRequest(const SparkCudaInferenceUtilityRequest *request);
SparkStatus SparkRunCudaInferenceUtilityHostReference(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report);
SparkStatus SparkRunCudaInferenceUtilityDeviceKernels(const SparkCudaInferenceUtilityRequest *request, const float *device_input_logits, uint64_t input_logit_count, const float *device_scale_values, uint32_t scale_count, float *device_output_logits, uint64_t output_logit_count, uint32_t *device_output_token_ids, uint32_t output_token_count, float *device_workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *device_report);
SparkStatus SparkRunCudaInferenceUtilityKernels(const SparkCudaInferenceUtilityRequest *request, const float *input_logits, uint64_t input_logit_count, const float *scale_values, uint32_t scale_count, float *output_logits, uint64_t output_logit_count, uint32_t *output_token_ids, uint32_t output_token_count, float *workspace_values, uint64_t workspace_value_count, SparkCudaInferenceUtilityReport *report);

#ifdef __cplusplus
}
#endif
