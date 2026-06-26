#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_ROUTER_SENTINEL 0x5350524F55544B31ull
#define SPARKPIPE_CUDA_ROUTER_MAX_EXPERTS 512u
#define SPARKPIPE_CUDA_ROUTER_MAX_TOP_K 16u
#define SPARKPIPE_CUDA_ROUTER_MAX_GROUPS 64u

typedef enum SparkCudaRouterScoreKind
{
    SPARK_CUDA_ROUTER_SCORE_SOFTMAX = 1,
    SPARK_CUDA_ROUTER_SCORE_SIGMOID = 2,
    SPARK_CUDA_ROUTER_SCORE_GROUPED_SIGMOID = 3
} SparkCudaRouterScoreKind;

typedef struct SparkCudaRouterRequest
{
    uint32_t row_count;
    uint32_t expert_count;
    uint32_t top_k;
    uint32_t expert_group_count;
    uint32_t top_k_group;
    uint32_t renormalize;
    uint32_t use_bias;
    uint32_t score_kind;
    float routed_scaling_factor;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaRouterRequest;

typedef struct SparkCudaRouterReport
{
    uint64_t score_count;
    uint64_t topk_value_count;
    uint32_t router_kernel_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaRouterReport;

SparkStatus SparkValidateCudaRouterRequest(const SparkCudaRouterRequest *request);
SparkStatus SparkRunCudaRouterTopK(const SparkCudaRouterRequest *request, const void *device_logits_bf16, const float *device_bias, float *device_topk_weights, uint32_t *device_topk_ids, uint32_t *device_token_expert_indices, SparkCudaRouterReport *report);

#ifdef __cplusplus
}
#endif
