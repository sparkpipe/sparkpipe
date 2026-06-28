#include <string.h>

#include "sparkpipe/spark_glm_bf16_linear_cuda.h"

void SparkGlmBf16LinearCudaRequestReset(SparkGlmBf16LinearCudaRequest *request)
{
    if (request == 0)
        return;
    memset(request, 0, sizeof(*request));
    request->sentinel = SPARKPIPE_GLM_BF16_LINEAR_CUDA_SENTINEL;
}

SparkStatus SparkValidateGlmBf16LinearCudaRequest(const SparkGlmBf16LinearCudaRequest *request, const SparkGlmBf16LinearCudaBinding *weight)
{
    uint64_t expected_weight_bytes;

    if (request == 0 || weight == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->sentinel != SPARKPIPE_GLM_BF16_LINEAR_CUDA_SENTINEL)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (!weight->ready || weight->path[0] == '\0' || request->input_count == 0u || request->output_count == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    expected_weight_bytes = (uint64_t)request->input_count * (uint64_t)request->output_count * sizeof(uint16_t);
    if (weight->size_bytes != expected_weight_bytes)
        return SPARK_STATUS_TENSOR_MANIFEST_ERROR;
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunGlmBf16LinearCuda(const SparkGlmBf16LinearCudaRequest *request, const SparkGlmBf16LinearCudaBinding *weight, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmBf16LinearCudaReport *report)
{
    SparkStatus status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateGlmBf16LinearCudaRequest(request, weight);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_bf16 == 0 || output_bf16 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->sentinel != SPARKPIPE_GLM_BF16_LINEAR_CUDA_SENTINEL)
        report->sentinel_violation_count = 1u;
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
