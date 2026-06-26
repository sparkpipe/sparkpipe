#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_REMAINING_SENTINEL 0x535052454D41494Eull

typedef struct SparkCudaRemainingInferenceRequest
{
    uint32_t token_count;
    uint32_t hidden_size;
    uint32_t output_size;
    uint32_t expert_count;
    uint32_t query_count;
    uint32_t kv_count;
    uint32_t head_dim;
    uint32_t sparse_top_k;
    uint32_t draft_count;
    uint32_t vocab_size;
    uint32_t compute_checksum;
    uint32_t reserved;
    float fp4_scale;
    float expert_fp4_scale;
    uint64_t sentinel;
} SparkCudaRemainingInferenceRequest;

typedef struct SparkCudaRemainingInferenceInputs
{
    const float *token_input;
    const uint8_t *fp4_linear_weight;
    const float *expert_weight;
    const uint8_t *expert_fp4_weight;
    const float *shared_weight;
    const uint32_t *expert_ids;
    const float *query;
    const float *key;
    const float *value;
    const uint32_t *sparse_indices;
    const uint32_t *physical_block_map;
    const float *draft_hidden;
    const float *draft_weight;
    const uint32_t *expected_tokens;
} SparkCudaRemainingInferenceInputs;

typedef struct SparkCudaRemainingInferenceOutputs
{
    float *fp4_linear_output;
    float *expert_output;
    float *expert_fp4_output;
    float *shared_output;
    uint32_t *mapped_sparse_indices;
    float *dense_mla_output;
    float *sparse_mla_output;
    float *draft_logits;
    uint32_t *draft_token_ids;
    uint32_t *accepted_mask;
} SparkCudaRemainingInferenceOutputs;

typedef struct SparkCudaRemainingInferenceReport
{
    uint64_t fp4_linear_checksum;
    uint64_t expert_checksum;
    uint64_t expert_fp4_checksum;
    uint64_t shared_checksum;
    uint64_t map_checksum;
    uint64_t dense_mla_checksum;
    uint64_t sparse_mla_checksum;
    uint64_t draft_checksum;
    uint64_t token_checksum;
    uint64_t accept_checksum;
    uint64_t trace_checksum;
    uint32_t fp4_linear_kernel_count;
    uint32_t expert_kernel_count;
    uint32_t expert_fp4_kernel_count;
    uint32_t shared_kernel_count;
    uint32_t sparse_map_kernel_count;
    uint32_t dense_mla_kernel_count;
    uint32_t sparse_mla_kernel_count;
    uint32_t draft_projection_kernel_count;
    uint32_t spec_verify_kernel_count;
    uint32_t checksum_kernel_count;
    uint32_t trace_kernel_count;
    uint32_t host_reference_count;
    uint32_t invalid_expert_count;
    uint32_t invalid_sparse_index_count;
    uint32_t accepted_token_count;
    uint32_t rejected_token_count;
    uint32_t sentinel_violation_count;
    uint32_t checksum_mismatch_count;
} SparkCudaRemainingInferenceReport;

SparkStatus SparkValidateCudaRemainingInferenceRequest(const SparkCudaRemainingInferenceRequest *request);
SparkStatus SparkRunCudaRemainingInferenceHostReference(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *inputs, SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report);
SparkStatus SparkRunCudaRemainingInferenceDeviceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *device_inputs, SparkCudaRemainingInferenceOutputs *device_outputs, SparkCudaRemainingInferenceReport *device_report);
SparkStatus SparkRunCudaRemainingInferenceKernels(const SparkCudaRemainingInferenceRequest *request, const SparkCudaRemainingInferenceInputs *inputs, SparkCudaRemainingInferenceOutputs *outputs, SparkCudaRemainingInferenceReport *report);

#ifdef __cplusplus
}
#endif
