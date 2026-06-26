#include <string.h>

#include "sparkpipe/spark_cuda_sparse_index_kernels.h"

static uint64_t SparkCudaSparseIndexScoreCount(const SparkCudaSparseIndexRequest *request)
{
	if (request == 0)
		return 0u;
	return (uint64_t)request->row_count * (uint64_t)request->candidate_count;
}

static uint64_t SparkCudaSparseIndexTopKCount(const SparkCudaSparseIndexRequest *request)
{
	if (request == 0)
		return 0u;
	return (uint64_t)request->row_count * (uint64_t)request->top_k;
}

SparkStatus SparkValidateCudaSparseIndexRequest(const SparkCudaSparseIndexRequest *request)
{
	if (request == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->sentinel != SPARKPIPE_CUDA_SPARSE_INDEX_SENTINEL)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->row_count == 0u || request->candidate_count == 0u || request->top_k == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->top_k > SPARKPIPE_CUDA_SPARSE_INDEX_MAX_TOP_K)
		return SPARK_STATUS_CAPACITY_EXCEEDED;
	if (request->top_k > request->candidate_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (request->score_stride < request->candidate_count)
		return SPARK_STATUS_INVALID_ARGUMENT;
	if (SparkCudaSparseIndexScoreCount(request) == 0u || SparkCudaSparseIndexTopKCount(request) == 0u)
		return SPARK_STATUS_INVALID_ARGUMENT;
	return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaSparseIndexFillReportShape(const SparkCudaSparseIndexRequest *request, SparkCudaSparseIndexReport *report)
{
	if (request == 0 || report == 0)
		return;
	report->score_count = SparkCudaSparseIndexScoreCount(request);
	report->topk_value_count = SparkCudaSparseIndexTopKCount(request);
}

SparkStatus SparkRunCudaSparseIndexTopK(const SparkCudaSparseIndexRequest *request, const float *device_scores, float *device_topk_scores, uint32_t *device_topk_indices, SparkCudaSparseIndexReport *report)
{
	SparkStatus status;

	if (report != 0)
		memset(report, 0, sizeof(*report));
	status = SparkValidateCudaSparseIndexRequest(request);
	if (status != SPARK_STATUS_OK)
		return status;
	if (device_scores == 0 || device_topk_scores == 0 || device_topk_indices == 0 || report == 0)
		return SPARK_STATUS_INVALID_ARGUMENT;
	SparkCudaSparseIndexFillReportShape(request, report);
	return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
