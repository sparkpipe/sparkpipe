#include <string.h>

#include "sparkpipe/spark_cuda_moe_kernels.h"

static uint64_t SparkCudaMoeRouteCount(const SparkCudaMoeDispatchRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * (uint64_t)request->top_k;
}

static uint64_t SparkCudaMoeExpertCapacity(const SparkCudaMoeDispatchRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->expert_count * (uint64_t)request->capacity_per_expert;
}

SparkStatus SparkValidateCudaMoeDispatchRequest(const SparkCudaMoeDispatchRequest *request)
{
    uint64_t route_count;
    uint64_t expert_capacity;

    if (request == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->sentinel != SPARKPIPE_CUDA_MOE_SENTINEL)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (request->token_count == 0u || request->hidden_size == 0u || request->top_k == 0u || request->top_k > SPARKPIPE_CUDA_MOE_MAX_TOP_K || request->expert_count == 0u || request->expert_count > SPARKPIPE_CUDA_MOE_MAX_EXPERTS || request->capacity_per_expert == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    route_count = SparkCudaMoeRouteCount(request);
    expert_capacity = SparkCudaMoeExpertCapacity(request);
    if (route_count == 0u || expert_capacity == 0u || expert_capacity < route_count)
    {
        return SPARK_STATUS_CAPACITY_EXCEEDED;
    }
    return SPARK_STATUS_OK;
}

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
static uint64_t SparkCudaMoeHiddenValueCount(const SparkCudaMoeDispatchRequest *request)
{
    if (request == 0)
    {
        return 0u;
    }
    return (uint64_t)request->token_count * (uint64_t)request->hidden_size;
}

static void SparkCudaMoeFillReportShape(const SparkCudaMoeDispatchRequest *request, SparkCudaMoeDispatchReport *report)
{
    if (request == 0 || report == 0)
    {
        return;
    }
    report->route_count = SparkCudaMoeRouteCount(request);
    report->hidden_value_count = SparkCudaMoeHiddenValueCount(request);
    report->expert_capacity = SparkCudaMoeExpertCapacity(request);
    report->error_counter_count = SPARKPIPE_CUDA_MOE_ERROR_COUNTERS;
}

static SparkStatus SparkCudaMoeValidateDispatchPointers(const void *device_hidden_bf16, const uint32_t *device_topk_ids, uint32_t *device_expert_counts, uint32_t *device_expert_offsets, uint32_t *device_expert_cursors, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report)
{
    if (device_hidden_bf16 == 0 || device_topk_ids == 0 || device_expert_counts == 0 || device_expert_offsets == 0 || device_expert_cursors == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaMoeDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_topk_ids, uint32_t *device_expert_counts, uint32_t *device_expert_offsets, uint32_t *device_expert_cursors, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    status = SparkCudaMoeValidateDispatchPointers(device_hidden_bf16, device_topk_ids, device_expert_counts, device_expert_offsets, device_expert_cursors, device_route_to_permuted_index, device_permuted_token_ids, device_permuted_route_ids, device_permuted_hidden_bf16, device_error_counters, report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    SparkCudaMoeFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaMoePreparedDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_token_ids, const uint32_t *device_assignment_route_ids, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_assignment_token_ids == 0 || device_assignment_route_ids == 0 || device_assignment_permuted_indices == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaMoePreparedRouteMajorDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_hidden_bf16 == 0 || device_assignment_permuted_indices == 0 || device_route_to_permuted_index == 0 || device_permuted_token_ids == 0 || device_permuted_route_ids == 0 || device_permuted_hidden_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaMoeCombine(const SparkCudaMoeDispatchRequest *request, const void *device_expert_output_bf16, const float *device_topk_weights, const uint32_t *device_route_to_permuted_index, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_expert_output_bf16 == 0 || device_topk_weights == 0 || device_route_to_permuted_index == 0 || device_output_bf16 == 0 || device_error_counters == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkRunCudaMoeCombineRouteMajor(const SparkCudaMoeDispatchRequest *request, const void *device_route_major_expert_output_bf16, const float *device_topk_weights, void *device_output_bf16, SparkCudaMoeDispatchReport *report)
{
    SparkStatus status;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    status = SparkValidateCudaMoeDispatchRequest(request);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    if (device_route_major_expert_output_bf16 == 0 || device_topk_weights == 0 || device_output_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if ((request->flags & SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES) == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaMoeFillReportShape(request, report);
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
