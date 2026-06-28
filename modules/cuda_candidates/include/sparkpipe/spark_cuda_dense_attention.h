#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_DENSE_ATTENTION_SENTINEL 0x5350434441545431ull
#define SPARKPIPE_CUDA_DENSE_ATTENTION_MAX_HEAD_SIZE 256u
#define SPARKPIPE_CUDA_DENSE_ATTENTION_DEVICE_COUNTERS 2u
#define SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_BLOCK_COUNTER 0u
#define SPARKPIPE_CUDA_DENSE_ATTENTION_INVALID_CONTEXT_COUNTER 1u

typedef struct SparkCudaDenseAttentionRequest
{
	uint32_t query_count;
	uint32_t head_count;
	uint32_t head_size;
	uint32_t block_size;
	uint32_t kv_block_count;
	uint32_t max_context_tokens;
	uint32_t max_blocks_per_query;
	uint32_t reserved;
	float qk_scale;
	uint64_t sentinel;
} SparkCudaDenseAttentionRequest;

typedef struct SparkCudaDenseAttentionReport
{
	uint64_t query_element_count;
	uint64_t kv_cache_element_count;
	uint64_t score_workspace_count;
	uint64_t output_element_count;
	uint32_t attention_kernel_count;
	uint32_t hot_path_allocation_count;
	uint32_t device_counter_count;
	uint32_t unsupported_shape_count;
} SparkCudaDenseAttentionReport;

SparkStatus SparkValidateCudaDenseAttentionRequest(const SparkCudaDenseAttentionRequest *request);
SparkStatus SparkRunCudaDensePagedAttentionBf16(const SparkCudaDenseAttentionRequest *request, const void *device_query_bf16, const void *device_key_cache_bf16, const void *device_value_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_bf16, SparkCudaDenseAttentionReport *report);

#ifdef __cplusplus
}
#endif
