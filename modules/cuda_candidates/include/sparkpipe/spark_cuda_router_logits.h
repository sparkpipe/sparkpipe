#pragma once

#include <stdint.h>

#include "sparkpipe/spark_cuda_cublaslt_gemm.h"
#include "sparkpipe/spark_cuda_router_kernels.h"
#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_ROUTER_LOGITS_SENTINEL 0x5350524C4F474954ull

typedef struct SparkCudaRouterLogitsRequest
{
    uint32_t token_count;
    uint32_t hidden_size;
    uint32_t expert_count;
    uint32_t use_scale;
    uint32_t use_bias;
    uint32_t reserved;
    uint64_t sentinel;
} SparkCudaRouterLogitsRequest;

typedef struct SparkCudaRouterLogitsReport
{
    uint64_t logits_value_count;
    uint64_t flops_per_run;
    uint32_t gemm_run_count;
    uint32_t correction_kernel_count;
    uint32_t hot_path_allocation_count;
    uint32_t sentinel_violation_count;
    uint32_t unsupported_shape_count;
} SparkCudaRouterLogitsReport;

SparkStatus SparkValidateCudaRouterLogitsRequest(const SparkCudaRouterLogitsRequest *request);
SparkStatus SparkRunCudaRouterLogitsBf16(const SparkCudaRouterLogitsRequest *request, SparkCudaCublasLtBf16GemmPlan *plan, const void *device_hidden_bf16, const void *device_router_weight_bf16, const float *device_expert_scale, const float *device_expert_bias, void *device_logits_bf16, SparkCudaRouterLogitsReport *report);

#ifdef __cplusplus
}
#endif
