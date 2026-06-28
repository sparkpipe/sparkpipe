#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_MOE_SENTINEL 0x53504D4F454B3135ull
#define SPARKPIPE_CUDA_MOE_MAX_TOP_K 16u
#define SPARKPIPE_CUDA_MOE_MAX_EXPERTS 1024u
#define SPARKPIPE_CUDA_MOE_ERROR_COUNTERS 4u
#define SPARKPIPE_CUDA_MOE_ROUTE_INVALID UINT32_MAX
#define SPARKPIPE_CUDA_MOE_FLAG_TRUSTED_ROUTES 1u
#define SPARKPIPE_CUDA_MOE_FLAG_TOKEN_GROUPED_DISPATCH 2u
#define SPARKPIPE_CUDA_MOE_FLAG_TOP8_WARP_DISPATCH 4u

typedef enum SparkCudaMoeErrorCounter
{
    SPARK_CUDA_MOE_ERROR_INVALID_EXPERT = 0,
    SPARK_CUDA_MOE_ERROR_CAPACITY = 1,
    SPARK_CUDA_MOE_ERROR_INVALID_ROUTE = 2,
    SPARK_CUDA_MOE_ERROR_SENTINEL = 3
} SparkCudaMoeErrorCounter;

typedef struct SparkCudaMoeDispatchRequest
{
    uint32_t token_count;
    uint32_t hidden_size;
    uint32_t top_k;
    uint32_t expert_count;
    uint32_t capacity_per_expert;
    uint32_t flags;
    uint64_t sentinel;
} SparkCudaMoeDispatchRequest;

typedef struct SparkCudaMoeDispatchReport
{
    uint64_t route_count;
    uint64_t hidden_value_count;
    uint64_t expert_capacity;
    uint32_t clear_kernel_count;
    uint32_t count_kernel_count;
    uint32_t prefix_kernel_count;
    uint32_t permute_kernel_count;
    uint32_t permute_fast_kernel_count;
    uint32_t prepared_dispatch_kernel_count;
    uint32_t combine_kernel_count;
    uint32_t combine_fast_kernel_count;
    uint32_t combine_route_major_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t error_counter_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaMoeDispatchReport;

SparkStatus SparkValidateCudaMoeDispatchRequest(const SparkCudaMoeDispatchRequest *request);
SparkStatus SparkRunCudaMoeDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_topk_ids, uint32_t *device_expert_counts, uint32_t *device_expert_offsets, uint32_t *device_expert_cursors, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report);
SparkStatus SparkRunCudaMoePreparedDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_token_ids, const uint32_t *device_assignment_route_ids, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report);
SparkStatus SparkRunCudaMoePreparedRouteMajorDispatch(const SparkCudaMoeDispatchRequest *request, const void *device_hidden_bf16, const uint32_t *device_assignment_permuted_indices, uint32_t *device_route_to_permuted_index, uint32_t *device_permuted_token_ids, uint32_t *device_permuted_route_ids, void *device_permuted_hidden_bf16, SparkCudaMoeDispatchReport *report);
SparkStatus SparkRunCudaMoeCombine(const SparkCudaMoeDispatchRequest *request, const void *device_expert_output_bf16, const float *device_topk_weights, const uint32_t *device_route_to_permuted_index, void *device_output_bf16, uint32_t *device_error_counters, SparkCudaMoeDispatchReport *report);
SparkStatus SparkRunCudaMoeCombineRouteMajor(const SparkCudaMoeDispatchRequest *request, const void *device_route_major_expert_output_bf16, const float *device_topk_weights, void *device_output_bf16, SparkCudaMoeDispatchReport *report);

#ifdef __cplusplus
}
#endif
