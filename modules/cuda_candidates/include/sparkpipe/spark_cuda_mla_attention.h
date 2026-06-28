#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_MLA_ATTENTION_SENTINEL 0x535043554D4C4131ull
#define SPARKPIPE_CUDA_MLA_ATTENTION_MAX_LATENT_DIM 1024u
#define SPARKPIPE_CUDA_MLA_ATTENTION_MAX_ROPE_DIM 256u
#define SPARKPIPE_CUDA_MLA_ATTENTION_MAX_SPARSE_TOP_K 256u
#define SPARKPIPE_CUDA_MLA_ATTENTION_DEVICE_COUNTERS 3u
#define SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_BLOCK_COUNTER 0u
#define SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_CONTEXT_COUNTER 1u
#define SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SPARSE_COUNTER 2u
#define SPARKPIPE_CUDA_MLA_ATTENTION_INVALID_SLOT UINT32_MAX

typedef struct SparkCudaMlaAttentionRequest
{
	uint32_t query_count;
	uint32_t query_head_count;
	uint32_t latent_dim;
	uint32_t rope_dim;
	uint32_t block_size;
	uint32_t kv_block_count;
	uint32_t cache_token_capacity;
	uint32_t cache_token_stride_elements;
	uint32_t max_context_tokens;
	uint32_t max_blocks_per_query;
	uint32_t sparse_top_k;
	uint32_t first_block_token_offset;
	float qk_scale;
	uint32_t reserved;
	uint64_t sentinel;
} SparkCudaMlaAttentionRequest;

typedef struct SparkCudaMlaAttentionReport
{
	uint64_t query_latent_element_count;
	uint64_t query_rope_element_count;
	uint64_t cache_element_count;
	uint64_t cache_token_stride_elements;
	uint64_t score_workspace_count;
	uint64_t sparse_index_count;
	uint64_t output_element_count;
	uint32_t cache_token_capacity;
	uint32_t first_block_token_offset;
	uint32_t dense_kernel_count;
	uint32_t sparse_kernel_count;
	uint32_t hot_path_allocation_count;
	uint32_t device_counter_count;
	uint32_t unsupported_shape_count;
	uint32_t explicit_cache_stride_count;
	uint32_t partial_first_block_count;
	uint32_t explicit_stream_count;
	uint32_t default_stream_count;
} SparkCudaMlaAttentionReport;

void SparkCudaMlaAttentionRequestReset(SparkCudaMlaAttentionRequest *request);
uint64_t SparkCudaMlaAttentionRequiredCacheElements(const SparkCudaMlaAttentionRequest *request);
SparkStatus SparkValidateCudaMlaAttentionRequest(const SparkCudaMlaAttentionRequest *request);
SparkStatus SparkRunCudaMlaDenseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report);
SparkStatus SparkRunCudaMlaSparseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report);
SparkStatus SparkRunCudaMlaDenseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, uint64_t cuda_stream, SparkCudaMlaAttentionReport *report);
SparkStatus SparkRunCudaMlaSparseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, uint64_t cuda_stream, SparkCudaMlaAttentionReport *report);

#ifdef __cplusplus
}
#endif
