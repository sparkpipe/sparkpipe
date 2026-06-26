#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_router_logits.h"

#define SPARK_CUDA_ROUTER_LOGITS_THREADS 256u

static __device__ float SparkCudaRouterLogitsBf16ToFloat(uint16_t value)
{
    union
    {
        uint32_t u;
        float f;
    } bits;

    bits.u = ((uint32_t)value) << 16u;
    return bits.f;
}

static __device__ uint16_t SparkCudaRouterLogitsFloatToBf16(float value)
{
    union
    {
        float f;
        uint32_t u;
    } bits;
    uint32_t rounding_bias;

    bits.f = value;
    rounding_bias = 0x7fffu + ((bits.u >> 16u) & 1u);
    return (uint16_t)((bits.u + rounding_bias) >> 16u);
}

static uint32_t SparkCudaRouterLogitsBlockCount(uint64_t value_count)
{
    uint64_t block_count;

    block_count = (value_count + (uint64_t)SPARK_CUDA_ROUTER_LOGITS_THREADS - 1u) / (uint64_t)SPARK_CUDA_ROUTER_LOGITS_THREADS;
    if (block_count == 0u)
    {
        return 1u;
    }
    if (block_count > 65535u)
    {
        return 65535u;
    }
    return (uint32_t)block_count;
}

static void SparkCudaRouterLogitsFillReportShape(const SparkCudaRouterLogitsRequest *request, SparkCudaRouterLogitsReport *report)
{
    report->logits_value_count = (uint64_t)request->token_count * (uint64_t)request->expert_count;
    report->flops_per_run = 2ull * (uint64_t)request->token_count * (uint64_t)request->hidden_size * (uint64_t)request->expert_count;
}

static __global__ void SparkCudaRouterLogitsCorrectionKernel(SparkCudaRouterLogitsRequest request, const float *expert_scale, const float *expert_bias, uint16_t *logits)
{
    uint64_t value_index;
    uint32_t expert_index;
    float value;

    value_index = ((uint64_t)blockIdx.x * (uint64_t)blockDim.x) + (uint64_t)threadIdx.x;
    while (value_index < (uint64_t)request.token_count * (uint64_t)request.expert_count)
    {
        expert_index = (uint32_t)(value_index % request.expert_count);
        value = SparkCudaRouterLogitsBf16ToFloat(logits[value_index]);
        if (request.use_scale != 0u)
        {
            value *= expert_scale[expert_index];
        }
        if (request.use_bias != 0u)
        {
            value += expert_bias[expert_index];
        }
        logits[value_index] = SparkCudaRouterLogitsFloatToBf16(value);
        value_index += (uint64_t)blockDim.x * (uint64_t)gridDim.x;
    }
}

extern "C" SparkStatus SparkRunCudaRouterLogitsBf16(const SparkCudaRouterLogitsRequest *request, SparkCudaCublasLtBf16GemmPlan *plan, const void *device_hidden_bf16, const void *device_router_weight_bf16, const float *device_expert_scale, const float *device_expert_bias, void *device_logits_bf16, SparkCudaRouterLogitsReport *report)
{
    SparkCudaCublasLtBf16GemmReport gemm_report;
    cudaError_t cuda_status;
    SparkStatus status;
    uint64_t logits_value_count;

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
    if (plan->m != request->token_count || plan->n != request->expert_count || plan->k != request->hidden_size)
    {
        report->unsupported_shape_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkCudaRouterLogitsFillReportShape(request, report);
    status = SparkRunCudaCublasLtBf16GemmPlan(plan, device_hidden_bf16, device_router_weight_bf16, device_logits_bf16, &gemm_report);
    if (status != SPARK_STATUS_OK)
    {
        return status;
    }
    report->gemm_run_count = gemm_report.run_count;
    logits_value_count = report->logits_value_count;
    if (request->use_scale != 0u || request->use_bias != 0u)
    {
        SparkCudaRouterLogitsCorrectionKernel<<<SparkCudaRouterLogitsBlockCount(logits_value_count), SPARK_CUDA_ROUTER_LOGITS_THREADS>>>(*request, device_expert_scale, device_expert_bias, (uint16_t *)device_logits_bf16);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess)
        {
            return SPARK_STATUS_INTERNAL_ERROR;
        }
        report->correction_kernel_count = 1u;
    }
    report->hot_path_allocation_count = 0u;
    return SPARK_STATUS_OK;
}
