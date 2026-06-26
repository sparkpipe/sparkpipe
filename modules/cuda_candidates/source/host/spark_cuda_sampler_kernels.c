#include <math.h>
#include <string.h>

#include "sparkpipe/spark_cuda_sampler_kernels.h"

static uint64_t SparkCudaSamplerLogitCount(const SparkCudaSamplerRequest *request)
{
    if (request == 0)
    {
        return 0;
    }
    return (uint64_t)request->row_count * (uint64_t)request->vocab_size;
}

SparkStatus SparkValidateCudaSamplerRequest(const SparkCudaSamplerRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_SAMPLER_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0u || request->vocab_size == 0u || request->top_k == 0u || request->top_k > SPARKPIPE_CUDA_SAMPLER_MAX_TOP_K || request->top_k > request->vocab_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (!isfinite(request->temperature) || request->temperature <= 0.0f || !isfinite(request->top_p) || request->top_p <= 0.0f || request->top_p > 1.0f || !isfinite(request->min_p) || request->min_p < 0.0f || request->min_p > 1.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaSamplerLogitCount(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaSamplerFillReportShape(const SparkCudaSamplerRequest *request, SparkCudaSamplerReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->logit_count = SparkCudaSamplerLogitCount(request);
    report->candidate_count = (uint64_t)request->row_count * (uint64_t)request->top_k;
    report->error_counter_count = SPARKPIPE_CUDA_SAMPLER_ERROR_COUNTERS;
}

SparkStatus SparkRunCudaSamplerTopKTopPMinP(const SparkCudaSamplerRequest *request, const float *device_logits, const float *device_uniforms, uint32_t *device_output_token_ids, float *device_output_probabilities, uint32_t *device_error_counters, SparkCudaSamplerReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaSamplerRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logits == 0 || device_uniforms == 0 || device_output_token_ids == 0 || device_output_probabilities == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaSamplerFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
