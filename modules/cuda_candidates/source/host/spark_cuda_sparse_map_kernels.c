#include <string.h>

#include "sparkpipe/spark_cuda_sparse_map_kernels.h"

static uint64_t SparkCudaSparseMapValueCount(const SparkCudaSparseMapRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->query_count * (uint64_t)request->sparse_top_k;
}

SparkStatus SparkValidateCudaSparseMapRequest(const SparkCudaSparseMapRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_SPARSE_MAP_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->query_count == 0u || request->sparse_top_k == 0u || request->logical_block_count == 0u || request->physical_block_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaSparseMapValueCount(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaSparseMapFillReportShape(const SparkCudaSparseMapRequest *request, SparkCudaSparseMapReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->map_value_count = SparkCudaSparseMapValueCount(request);
    report->error_counter_count = SPARKPIPE_CUDA_SPARSE_MAP_ERROR_COUNTERS;
}

SparkStatus SparkRunCudaSparsePhysicalBlockMap(const SparkCudaSparseMapRequest *request, const uint32_t *device_logical_block_ids, const uint32_t *device_physical_block_map, uint32_t *device_physical_block_ids, uint32_t *device_error_counters, SparkCudaSparseMapReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSparseMapRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logical_block_ids == 0 || device_physical_block_map == 0 || device_physical_block_ids == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSparseMapFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
