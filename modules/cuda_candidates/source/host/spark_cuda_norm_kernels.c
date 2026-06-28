#include <string.h>

#include "sparkpipe/spark_cuda_norm_kernels.h"

SparkStatus SparkValidateCudaNormRequest(const SparkCudaNormRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0u || request->hidden_size == 0u || request->epsilon <= 0.0f || request->epsilon != request->epsilon)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_bf16, SparkCudaNormReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_weight_bf16 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaFusedAddRmsNormBf16(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_residual_bf16, const void *device_weight_bf16, void *device_output_bf16, void *device_residual_output_bf16, SparkCudaNormReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_residual_bf16 == 0 || device_weight_bf16 == 0 || device_output_bf16 == 0 || device_residual_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaRmsNormQuantFp8E4m3(const SparkCudaNormRequest *request, const void *device_input_bf16, const void *device_weight_bf16, void *device_output_fp8, float *device_output_scales, SparkCudaNormReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaNormRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_input_bf16 == 0 || device_weight_bf16 == 0 || device_output_fp8 == 0 || device_output_scales == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    if (request->sentinel != SPARKPIPE_CUDA_NORM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
