#include <string.h>

#include "sparkpipe/spark_cuda_embedding_kernels.h"

static uint64_t SparkCudaEmbeddingOutputValueCount(const SparkCudaEmbeddingRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->token_count * (uint64_t)request->hidden_size;
}

static uint64_t SparkCudaEmbeddingTableValueCount(const SparkCudaEmbeddingRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->vocab_size * (uint64_t)request->hidden_size;
}

SparkStatus SparkValidateCudaEmbeddingRequest(const SparkCudaEmbeddingRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_EMBEDDING_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->hidden_size == 0u || request->vocab_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaEmbeddingOutputValueCount(request) == 0u || SparkCudaEmbeddingTableValueCount(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaEmbeddingFillReportShape(const SparkCudaEmbeddingRequest *request, SparkCudaEmbeddingReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->output_value_count = SparkCudaEmbeddingOutputValueCount(request);
    report->table_value_count = SparkCudaEmbeddingTableValueCount(request);
    report->error_counter_count = SPARKPIPE_CUDA_EMBEDDING_ERROR_COUNTERS;
}

SparkStatus SparkRunCudaEmbeddingGatherBf16(const SparkCudaEmbeddingRequest *request, const uint32_t *device_token_ids, const void *device_embedding_table_bf16, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaEmbeddingReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaEmbeddingRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_token_ids == 0 || device_embedding_table_bf16 == 0 || device_output_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaEmbeddingFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
