#include <string.h>

#include "sparkpipe/spark_cuda_fp4_gemm.h"
#include "sparkpipe/spark_glm_nvfp4_expert_cuda.h"

void SparkGlmNvfp4ExpertCudaRequestReset(SparkGlmNvfp4ExpertCudaRequest *request)
{
    if (request == 0)
        return;
    memset(request, 0, sizeof(*request));
    request->sentinel = SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_SENTINEL;
}

SparkStatus SparkValidateGlmNvfp4ExpertCudaRequest(const SparkGlmNvfp4ExpertCudaRequest *request, const SparkGlmStageNvfp4ExpertDescriptor *descriptor)
{
    if (request == 0 || descriptor == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->sentinel != SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_SENTINEL)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (!descriptor->ready || request->row_count == 0u || request->hidden_size == 0u || request->intermediate_rows == 0u)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (descriptor->fp4_format != SPARK_GLM_STAGE_FP4_EXPERT_FORMAT_NVFP4_E2M1_E4M3 || descriptor->fp4_group_size != SPARKPIPE_CUDA_FP4_GEMM_SCALE_BLOCK)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->hidden_size != descriptor->hidden_size || request->intermediate_rows != descriptor->intermediate_rows)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkValidateCudaFp4GemmShape((request->row_count + 15u) & ~15u, request->intermediate_rows, request->hidden_size) != SPARK_STATUS_OK)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (SparkValidateCudaFp4GemmShape((request->row_count + 15u) & ~15u, request->hidden_size, request->intermediate_rows) != SPARK_STATUS_OK)
        return SPARK_STATUS_INVALID_ARGUMENT;
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkRunGlmNvfp4ExpertCuda(const SparkGlmNvfp4ExpertCudaRequest *request, const SparkGlmStageNvfp4ExpertDescriptor *descriptor, const uint16_t *input_bf16, uint16_t *output_bf16, SparkGlmNvfp4ExpertCudaReport *report)
{
    SparkStatus status;

    if (report != 0)
        memset(report, 0, sizeof(*report));
    status = SparkValidateGlmNvfp4ExpertCudaRequest(request, descriptor);
    if (status != SPARK_STATUS_OK)
        return status;
    if (input_bf16 == 0 || output_bf16 == 0 || report == 0)
        return SPARK_STATUS_INVALID_ARGUMENT;
    if (request->sentinel != SPARKPIPE_GLM_NVFP4_EXPERT_CUDA_SENTINEL)
        report->sentinel_violation_count = 1u;
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
