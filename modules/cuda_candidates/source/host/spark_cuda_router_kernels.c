#include <math.h>
#include <string.h>

#include "sparkpipe/spark_cuda_router_kernels.h"

static uint64_t SparkCudaRouterScoreCount(const SparkCudaRouterRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->row_count * (uint64_t)request->expert_count;
}

static uint64_t SparkCudaRouterTopKValueCount(const SparkCudaRouterRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->row_count * (uint64_t)request->top_k;
}

static SparkStatus SparkValidateCudaRouterGroupedRequest(const SparkCudaRouterRequest *request)
{
    uint32_t group_size;

    if (request->score_kind != (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID)
    {
        return SPARK_STATUS_OK;
    }
    if (request->expert_group_count == 0u || request->expert_group_count > SPARKPIPE_CUDA_ROUTER_MAX_GROUPS || request->top_k_group == 0u || request->top_k_group > request->expert_group_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->expert_count % request->expert_group_count) != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    group_size = request->expert_count / request->expert_group_count;
    if (group_size < 2u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkValidateCudaRouterRequest(const SparkCudaRouterRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_ROUTER_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->row_count == 0u || request->expert_count == 0u || request->expert_count > SPARKPIPE_CUDA_ROUTER_MAX_EXPERTS || request->top_k == 0u || request->top_k > SPARKPIPE_CUDA_ROUTER_MAX_TOP_K || request->top_k > request->expert_count)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->score_kind != (uint32_t)SPARK_CUDA_ROUTER_SCORE_SOFTMAX && request->score_kind != (uint32_t)SPARK_CUDA_ROUTER_SCORE_SIGMOID && request->score_kind != (uint32_t)SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (isfinite(request->routed_scaling_factor) == 0 || request->routed_scaling_factor <= 0.0f)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (SparkCudaRouterScoreCount(request) == 0u || SparkCudaRouterTopKValueCount(request) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SparkValidateCudaRouterGroupedRequest(request);
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkFillCudaRouterReportShape(const SparkCudaRouterRequest *request, SparkCudaRouterReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->score_count = SparkCudaRouterScoreCount(request);
    report->topk_value_count = SparkCudaRouterTopKValueCount(request);
}

SparkStatus SparkRunCudaRouterTopK(const SparkCudaRouterRequest *request, const void *device_logits_bf16, const float *device_bias, float *device_topk_weights, uint32_t *device_topk_ids, uint32_t *device_token_expert_indices, SparkCudaRouterReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRouterRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_logits_bf16 == 0 || device_topk_weights == 0 || device_topk_ids == 0 || device_token_expert_indices == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->use_bias != 0u && device_bias == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkFillCudaRouterReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
