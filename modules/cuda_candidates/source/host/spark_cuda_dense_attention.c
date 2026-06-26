#include <math.h>
#include <string.h>

#include "sparkpipe/spark_cuda_dense_attention.h"

static uint64_t SparkCudaDenseAttentionQueryElementCount(const SparkCudaDenseAttentionRequest *request)
{
	if (request == 0)
		return 0u;
	return (uint64_t)request->query_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaDenseAttentionKvElementCount(const SparkCudaDenseAttentionRequest *request)
{
	if (request == 0)
		return 0u;
	return (uint64_t)request->kv_block_count * (uint64_t)request->block_size * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaDenseAttentionScoreCount(const SparkCudaDenseAttentionRequest *request)
{
	if (request == 0)
		return 0u;
	return (uint64_t)request->query_count * (uint64_t)request->head_count * (uint64_t)request->max_context_tokens;
}

SparkStatus SparkValidateCudaDenseAttentionRequest(const SparkCudaDenseAttentionRequest *request)
{
	uint64_t query_element_count;
	uint64_t kv_element_count;
	uint64_t score_count;

	if (request == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->sentinel != SPARKPIPE_CUDA_DENSE_ATTENTION_SENTINEL)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->query_count == 0u || request->head_count == 0u || request->head_size == 0u || request->block_size == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->kv_block_count == 0u || request->max_context_tokens == 0u || request->max_blocks_per_query == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->head_size > SPARKPIPE_CUDA_DENSE_ATTENTION_MAX_HEAD_SIZE)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (request->max_blocks_per_query < ((request->max_context_tokens + request->block_size - 1u) / request->block_size))
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (!isfinite(request->qk_scale) || request->qk_scale <= 0.0f)
		return SPARK_STATUS_INVALID_ARGUMENT;
	query_element_count = SparkCudaDenseAttentionQueryElementCount(request);
	kv_element_count = SparkCudaDenseAttentionKvElementCount(request);
	score_count = SparkCudaDenseAttentionScoreCount(request);
	if (query_element_count == 0u || kv_element_count == 0u || score_count == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaDenseAttentionFillReportShape(const SparkCudaDenseAttentionRequest *request, SparkCudaDenseAttentionReport *report)
{
	if (request == 0 || report == 0)
		return;
	report->query_element_count = SparkCudaDenseAttentionQueryElementCount(request);
	report->kv_cache_element_count = SparkCudaDenseAttentionKvElementCount(request);
	report->score_workspace_count = SparkCudaDenseAttentionScoreCount(request);
	report->output_element_count = report->query_element_count;
	report->device_counter_count = SPARKPIPE_CUDA_DENSE_ATTENTION_DEVICE_COUNTERS;
}

SparkStatus SparkRunCudaDensePagedAttentionBf16(const SparkCudaDenseAttentionRequest *request, const void *device_query_bf16, const void *device_key_cache_bf16, const void *device_value_cache_bf16, const uint32_t *device_block_table, const uint32_t *device_context_lengths, float *device_score_workspace, uint32_t *device_error_counters, void *device_output_bf16, SparkCudaDenseAttentionReport *report)
{
	SparkStatus status;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaDenseAttentionRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_query_bf16 == 0 || device_key_cache_bf16 == 0 || device_value_cache_bf16 == 0 || device_block_table == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (device_context_lengths == 0 || device_score_workspace == 0 || device_error_counters == 0 || device_output_bf16 == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkCudaDenseAttentionFillReportShape(request, report);
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
