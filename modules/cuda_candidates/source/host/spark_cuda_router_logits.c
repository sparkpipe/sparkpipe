#include <string.h>

#include "sparkpipe/spark_cuda_router_logits.h"

SparkStatus SparkValidateCudaRouterLogitsRequest(const SparkCudaRouterLogitsRequest *request)
{
    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_ROUTER_LOGITS_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->hidden_size == 0u || request->expert_count == 0u || request->expert_count > SPARKPIPE_CUDA_ROUTER_MAX_EXPERTS)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static void SparkCudaRouterLogitsFillReportShape(const SparkCudaRouterLogitsRequest *request, SparkCudaRouterLogitsReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->logits_value_count = (uint64_t)request->token_count * (uint64_t)request->expert_count;
    report->flops_per_run = 2ull * (uint64_t)request->token_count * (uint64_t)request->hidden_size * (uint64_t)request->expert_count;
}

SparkStatus SparkRunCudaRouterLogitsBf16(const SparkCudaRouterLogitsRequest *request, SparkCudaCublasLtBf16GemmPlan *plan, const void *device_hidden_bf16, const void *device_router_weight_bf16, const float *device_expert_scale, const float *device_expert_bias, void *device_logits_bf16, SparkCudaRouterLogitsReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaRouterLogitsRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (plan == 0 || device_hidden_bf16 == 0 || device_router_weight_bf16 == 0 || device_logits_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->use_scale != 0u && device_expert_scale == 0) || (request->use_bias != 0u && device_expert_bias == 0))
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaRouterLogitsFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
