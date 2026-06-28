#include <string.h>

#include "sparkpipe/spark_cuda_spec_verify_kernels.h"

static uint64_t SparkCudaSpecVerifyTokenCount(const SparkCudaSpecVerifyRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->sequence_count * (uint64_t)request->draft_count;
}

SparkStatus SparkValidateCudaSpecVerifyRequest(const SparkCudaSpecVerifyRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_SPEC_VERIFY_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sequence_count == 0u || request->draft_count == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaSpecVerifyTokenCount(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaSpecVerifyFillReportShape(const SparkCudaSpecVerifyRequest *request, SparkCudaSpecVerifyReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->token_count = SparkCudaSpecVerifyTokenCount(request);
    report->error_counter_count = SPARKPIPE_CUDA_SPEC_VERIFY_ERROR_COUNTERS;
}

SparkStatus SparkRunCudaSpecVerify(const SparkCudaSpecVerifyRequest *request, const uint32_t *device_draft_token_ids, const uint32_t *device_target_token_ids, uint32_t *device_accepted_mask, uint32_t *device_accepted_prefix_lengths, uint32_t *device_error_counters, SparkCudaSpecVerifyReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSpecVerifyRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_draft_token_ids == 0 || device_target_token_ids == 0 || device_accepted_mask == 0 || device_accepted_prefix_lengths == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSpecVerifyFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
