#include <math.h>
#include <string.h>

#include "sparkpipe/spark_cuda_mla_attention.h"

static uint64_t SparkCudaMlaAttentionCheckedProduct(uint64_t left, uint64_t right)
{
    if (left == 0u || right == 0u || left > UINT64_MAX / right)
    {
        return 0u;
    }
    return left * right;
}

static uint64_t SparkCudaMlaAttentionQueryLatentCount(const SparkCudaMlaAttentionRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return SparkCudaMlaAttentionCheckedProduct(SparkCudaMlaAttentionCheckedProduct(request->query_count, request->query_head_count), request->latent_dim);
}

static uint64_t SparkCudaMlaAttentionQueryRopeCount(const SparkCudaMlaAttentionRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return SparkCudaMlaAttentionCheckedProduct(SparkCudaMlaAttentionCheckedProduct(request->query_count, request->query_head_count), request->rope_dim);
}

static uint64_t SparkCudaMlaAttentionScoreCount(const SparkCudaMlaAttentionRequest *request, uint32_t sparse_mode)
{
    uint32_t score_columns;

    if (request == 0)
    {
        return 0u;
    }
    score_columns = sparse_mode != 0u ? request->sparse_top_k : request->max_context_tokens;
    return SparkCudaMlaAttentionCheckedProduct(SparkCudaMlaAttentionCheckedProduct(request->query_count, request->query_head_count), score_columns);
}

static uint32_t SparkCudaMlaAttentionRequiredBlockCount(const SparkCudaMlaAttentionRequest *request)
{
    uint64_t addressed_token_count;

    if (request == 0 || request->block_size == 0u)
    {
        return 0u;
    }
    addressed_token_count = (uint64_t)request->first_block_token_offset + (uint64_t)request->max_context_tokens;
    return (uint32_t)((addressed_token_count + (uint64_t)request->block_size - 1u) / (uint64_t)request->block_size);
}

void SparkCudaMlaAttentionRequestReset(SparkCudaMlaAttentionRequest *request)
{
    if (request == 0)
    {
        return;
    }
    memset(request, 0, sizeof(*request));
    request->sparse_top_k = 1u;
    request->qk_scale = 1.0f;
    request->sentinel = SPARKPIPE_CUDA_MLA_ATTENTION_SENTINEL;
}

uint64_t SparkCudaMlaAttentionRequiredCacheElements(const SparkCudaMlaAttentionRequest *request)
{
    uint64_t entry_element_count;
    uint64_t final_token_offset;

    if (request == 0 || request->cache_token_capacity == 0u || request->cache_token_stride_elements == 0u)
    {
        return 0u;
    }
    entry_element_count = (uint64_t)request->latent_dim + (uint64_t)request->rope_dim;
    final_token_offset = ((uint64_t)request->cache_token_capacity - 1u) * (uint64_t)request->cache_token_stride_elements;
    if (entry_element_count == 0u || final_token_offset > UINT64_MAX - entry_element_count)
    {
        return 0u;
    }
    return final_token_offset + entry_element_count;
}

SparkStatus SparkValidateCudaMlaAttentionRequest(const SparkCudaMlaAttentionRequest *request)
{
    uint64_t cache_slot_capacity;
    uint64_t cache_entry_elements;
    uint32_t required_block_count;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_MLA_ATTENTION_SENTINEL || request->reserved != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->query_count == 0u || request->query_head_count == 0u || request->latent_dim == 0u || request->rope_dim == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->block_size == 0u || request->kv_block_count == 0u || request->cache_token_capacity == 0u || request->cache_token_stride_elements == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->max_context_tokens == 0u || request->max_blocks_per_query == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->latent_dim > SPARKPIPE_CUDA_MLA_ATTENTION_MAX_LATENT_DIM || request->rope_dim > SPARKPIPE_CUDA_MLA_ATTENTION_MAX_ROPE_DIM)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if ((request->rope_dim & 1u) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cache_entry_elements = (uint64_t)request->latent_dim + (uint64_t)request->rope_dim;
    if ((uint64_t)request->cache_token_stride_elements < cache_entry_elements)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    cache_slot_capacity = (uint64_t)request->kv_block_count * (uint64_t)request->block_size;
    if (cache_slot_capacity > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if ((uint64_t)request->cache_token_capacity > cache_slot_capacity || request->max_context_tokens > request->cache_token_capacity)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->first_block_token_offset >= request->block_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((uint64_t)request->first_block_token_offset + (uint64_t)request->max_context_tokens > UINT32_MAX)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    if (request->sparse_top_k == 0u || request->sparse_top_k > SPARKPIPE_CUDA_MLA_ATTENTION_MAX_SPARSE_TOP_K || request->sparse_top_k > request->max_context_tokens)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    required_block_count = SparkCudaMlaAttentionRequiredBlockCount(request);
    if (required_block_count == 0u || request->max_blocks_per_query < required_block_count || request->kv_block_count < required_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->max_blocks_per_query > request->kv_block_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(request->qk_scale) || request->qk_scale <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaMlaAttentionQueryLatentCount(request) == 0u || SparkCudaMlaAttentionQueryRopeCount(request) == 0u || SparkCudaMlaAttentionRequiredCacheElements(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaMlaAttentionScoreCount(request, 0u) == 0u || SparkCudaMlaAttentionScoreCount(request, 1u) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static uint64_t SparkCudaMlaAttentionSparseIndexCount(const SparkCudaMlaAttentionRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return SparkCudaMlaAttentionCheckedProduct(SparkCudaMlaAttentionCheckedProduct(request->query_count, request->query_head_count), request->sparse_top_k);
}

static void SparkFillCudaMlaAttentionReport(const SparkCudaMlaAttentionRequest *request, SparkCudaMlaAttentionReport *report, uint32_t sparse_mode, uint64_t cuda_stream)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->query_latent_element_count = SparkCudaMlaAttentionQueryLatentCount(request);
    report->query_rope_element_count = SparkCudaMlaAttentionQueryRopeCount(request);
    report->cache_element_count = SparkCudaMlaAttentionRequiredCacheElements(request);
    report->cache_token_stride_elements = request->cache_token_stride_elements;
    report->score_workspace_count = SparkCudaMlaAttentionScoreCount(request, sparse_mode);
    report->sparse_index_count = sparse_mode != 0u ? SparkCudaMlaAttentionSparseIndexCount(request) : 0u;
    report->output_element_count = report->query_latent_element_count;
    report->cache_token_capacity = request->cache_token_capacity;
    report->first_block_token_offset = request->first_block_token_offset;
    report->device_counter_count = SPARKPIPE_CUDA_MLA_ATTENTION_DEVICE_COUNTERS;
    report->explicit_cache_stride_count = 1u;
    report->partial_first_block_count = request->first_block_token_offset != 0u ? 1u : 0u;
    report->explicit_stream_count = cuda_stream != 0u ? 1u : 0u;
    report->default_stream_count = cuda_stream == 0u ? 1u : 0u;
}

SparkStatus SparkRunCudaMlaDenseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, uint64_t cuda_stream, SparkCudaMlaAttentionReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMlaAttentionRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_query_latent_bf16 == 0 || device_query_rope_bf16 == 0 || device_mla_cache_bf16 == 0 || device_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (device_context_lengths == 0 || device_score_workspace == 0 || device_error_counters == 0 || device_output_latent_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaMlaAttentionReport(request, report, 0u, cuda_stream);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaMlaSparseAttentionBf16OnStream(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, uint64_t cuda_stream, SparkCudaMlaAttentionReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMlaAttentionRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_query_latent_bf16 == 0 || device_query_rope_bf16 == 0 || device_mla_cache_bf16 == 0 || device_block_table == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (device_context_lengths == 0 || device_sparse_token_indices == 0 || device_score_workspace == 0 || device_error_counters == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (device_mapped_sparse_slots == 0 || device_output_latent_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaMlaAttentionReport(request, report, 1u, cuda_stream);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
SparkStatus SparkRunCudaMlaDenseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report)
{
    return SparkRunCudaMlaDenseAttentionBf16OnStream(request, device_query_latent_bf16, device_query_rope_bf16, device_mla_cache_bf16, device_block_table, device_context_lengths, device_score_workspace, device_error_counters, device_output_latent_bf16, 0u, report);
}

SparkStatus SparkRunCudaMlaSparseAttentionBf16(const SparkCudaMlaAttentionRequest *request, const void *device_query_latent_bf16, const void *device_query_rope_bf16, const void *device_mla_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, const uint32_t *device_sparse_token_indices, float *device_score_workspace, uint32_t *device_error_counters, uint32_t *device_mapped_sparse_slots, void *device_output_latent_bf16, SparkCudaMlaAttentionReport *report)
{
    return SparkRunCudaMlaSparseAttentionBf16OnStream(request, device_query_latent_bf16, device_query_rope_bf16, device_mla_cache_bf16, device_block_table, device_context_lengths, device_sparse_token_indices, device_score_workspace, device_error_counters, device_mapped_sparse_slots, device_output_latent_bf16, 0u, report);
}

#endif
