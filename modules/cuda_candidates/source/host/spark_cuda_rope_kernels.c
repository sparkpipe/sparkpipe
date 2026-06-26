#include <string.h>

#include "sparkpipe/spark_cuda_rope_kernels.h"

static uint64_t SparkCudaRopeElementCount(const SparkCudaRopeRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * (uint64_t)request->head_count * (uint64_t)request->head_size;
}

static uint64_t SparkCudaRopePairCount(const SparkCudaRopeRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * (uint64_t)request->head_count * ((uint64_t)request->rotary_dim / 2u);
}

SparkStatus SparkValidateCudaRopeRequest(const SparkCudaRopeRequest *request)
{
    uint64_t element_count;
    uint64_t pair_count;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->head_count == 0u || request->head_size == 0u || request->rotary_dim == 0u || request->position_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_ROPE_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->rotary_dim & 1u) != 0u || request->rotary_dim > request->head_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    element_count = SparkCudaRopeElementCount(request);
    pair_count = SparkCudaRopePairCount(request);
    if (element_count == 0u || pair_count == 0u || element_count < (pair_count * 2u))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkFillCudaRopeReportShape(const SparkCudaRopeRequest *request, SparkCudaRopeReport *report)
{
    uint64_t rotated_element_count;

    if (request == 0 || report == 0)
    {
        return;
    }
    report->element_count = SparkCudaRopeElementCount(request);
    report->rotary_pair_count = SparkCudaRopePairCount(request);
    rotated_element_count = report->rotary_pair_count * 2u;
    report->copy_element_count = report->element_count >= rotated_element_count ? report->element_count - rotated_element_count : 0u;
}

SparkStatus SparkRunCudaRopeBf16(const SparkCudaRopeRequest *request, const void *device_input_bf16, const uint32_t *device_positions, const float *device_cos_table, const float *device_sin_table, void *device_output_bf16, SparkCudaRopeReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRopeRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_positions == 0 || device_cos_table == 0 || device_sin_table == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaRopeReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
