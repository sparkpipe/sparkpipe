#include <string.h>

#include "sparkpipe/spark_cuda_activation_kernels.h"

SparkStatus SparkValidateCudaActivationRequest(const SparkCudaActivationRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0u || request->hidden_size == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->input_stride < (request->hidden_size * 2u) || request->output_stride < request->hidden_size)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->activation_kind != SPARK_CUDA_GATED_ACTIVATION_SILU && request->activation_kind != SPARK_CUDA_GATED_ACTIVATION_GELU && request->activation_kind != SPARK_CUDA_GATED_ACTIVATION_GELU_TANH)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunCudaGatedActivationBf16(const SparkCudaActivationRequest *request, const void *device_gate_up_bf16, void *device_output_bf16, SparkCudaActivationReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaActivationRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_gate_up_bf16 == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    report->element_count = (uint64_t)request->row_count * (uint64_t)request->hidden_size;
    report->row_count = request->row_count;
    report->hidden_size = request->hidden_size;
    report->activation_kind = (uint32_t)request->activation_kind;
    if (request->sentinel != SPARKPIPE_CUDA_ACTIVATION_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
