#include <cublasLt.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <string.h>

#include "sparkpipe/spark_cuda_cublaslt_gemm.h"

static SparkStatus SparkStatusFromCublasLt(cublasStatus_t status)
{
    if (status == CUBLAS_STATUS_SUCCESS)
    {
        return SPARK_STATUS_OK;
    }
    if (status == CUBLAS_STATUS_INVALID_VALUE || status == CUBLAS_STATUS_NOT_INITIALIZED)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkStatusFromCudaLt(cudaError_t status)
{
    if (status == cudaSuccess)
    {
        return SPARK_STATUS_OK;
    }
    if (status == cudaErrorInvalidValue)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_INTERNAL_ERROR;
}

static SparkStatus SparkSetRowMajorLayout(cublasLtMatrixLayout_t layout)
{
    cublasLtOrder_t order;
    cublasStatus_t status;

    order = CUBLASLT_ORDER_ROW;
    status = cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    return SparkStatusFromCublasLt(status);
}

static SparkStatus SparkDestroyLtPlanObjects(SparkCudaCublasLtBf16GemmPlan *plan)
{
    if (plan->preference != 0u)
    {
        cublasLtMatmulPreferenceDestroy((cublasLtMatmulPreference_t)(uintptr_t)plan->preference);
    }
    if (plan->c_desc != 0u)
    {
        cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->c_desc);
    }
    if (plan->b_desc != 0u)
    {
        cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->b_desc);
    }
    if (plan->a_desc != 0u)
    {
        cublasLtMatrixLayoutDestroy((cublasLtMatrixLayout_t)(uintptr_t)plan->a_desc);
    }
    if (plan->operation_desc != 0u)
    {
        cublasLtMatmulDescDestroy((cublasLtMatmulDesc_t)(uintptr_t)plan->operation_desc);
    }
    if (plan->lt_handle != 0u)
    {
        cublasLtDestroy((cublasLtHandle_t)(uintptr_t)plan->lt_handle);
    }
    plan->preference = 0u;
    plan->c_desc = 0u;
    plan->b_desc = 0u;
    plan->a_desc = 0u;
    plan->operation_desc = 0u;
    plan->lt_handle = 0u;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkInitCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, void *workspace, uint64_t workspace_bytes, uint64_t stream)
{
    cublasOperation_t transa;
    cublasOperation_t transb;
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasLtHandle_t lt_handle;
    cublasLtMatmulDesc_t operation_desc;
    cublasLtMatrixLayout_t a_desc;
    cublasLtMatrixLayout_t b_desc;
    cublasLtMatrixLayout_t c_desc;
    cublasLtMatmulPreference_t preference;
    cublasStatus_t cublas_status;
    int32_t returned_results;
    SparkStatus status;

    if (plan == 0 || workspace == 0 || m == 0u || n == 0u || k == 0u || workspace_bytes == 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    memset(&heuristic, 0, sizeof(heuristic));
    lt_handle = 0;
    operation_desc = 0;
    a_desc = 0;
    b_desc = 0;
    c_desc = 0;
    preference = 0;
    returned_results = 0;
    cublas_status = cublasLtCreate(&lt_handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkStatusFromCublasLt(cublas_status);
    }
    cublas_status = cublasLtMatmulDescCreate(&operation_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        transa = CUBLAS_OP_N;
        transb = CUBLAS_OP_T;
        cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
        if (cublas_status == CUBLAS_STATUS_SUCCESS)
        {
            cublas_status = cublasLtMatmulDescSetAttribute(operation_desc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));
        }
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, (uint64_t)m, (uint64_t)k, (int64_t)k);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        status = SparkSetRowMajorLayout(a_desc);
        if (status != SPARK_STATUS_OK)
        {
            cublas_status = CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, (uint64_t)n, (uint64_t)k, (int64_t)k);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        status = SparkSetRowMajorLayout(b_desc);
        if (status != SPARK_STATUS_OK)
        {
            cublas_status = CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_16BF, (uint64_t)m, (uint64_t)n, (int64_t)n);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        status = SparkSetRowMajorLayout(c_desc);
        if (status != SPARK_STATUS_OK)
        {
            cublas_status = CUBLAS_STATUS_INVALID_VALUE;
        }
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatmulPreferenceCreate(&preference);
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_bytes, sizeof(workspace_bytes));
    }
    if (cublas_status == CUBLAS_STATUS_SUCCESS)
    {
        cublas_status = cublasLtMatmulAlgoGetHeuristic(lt_handle, operation_desc, a_desc, b_desc, c_desc, c_desc, preference, 1, &heuristic, &returned_results);
    }
    if (cublas_status != CUBLAS_STATUS_SUCCESS || returned_results <= 0)
    {
        if (preference != 0)
        {
            cublasLtMatmulPreferenceDestroy(preference);
        }
        if (c_desc != 0)
        {
            cublasLtMatrixLayoutDestroy(c_desc);
        }
        if (b_desc != 0)
        {
            cublasLtMatrixLayoutDestroy(b_desc);
        }
        if (a_desc != 0)
        {
            cublasLtMatrixLayoutDestroy(a_desc);
        }
        if (operation_desc != 0)
        {
            cublasLtMatmulDescDestroy(operation_desc);
        }
        cublasLtDestroy(lt_handle);
        return cublas_status == CUBLAS_STATUS_SUCCESS ? SPARK_STATUS_INTERNAL_ERROR : SparkStatusFromCublasLt(cublas_status);
    }
    plan->m = m;
    plan->n = n;
    plan->k = k;
    plan->lda = k;
    plan->ldb = k;
    plan->ldc = n;
    plan->workspace = workspace;
    plan->workspace_bytes = workspace_bytes;
    plan->stream = stream;
    plan->lt_handle = (uint64_t)(uintptr_t)lt_handle;
    plan->operation_desc = (uint64_t)(uintptr_t)operation_desc;
    plan->a_desc = (uint64_t)(uintptr_t)a_desc;
    plan->b_desc = (uint64_t)(uintptr_t)b_desc;
    plan->c_desc = (uint64_t)(uintptr_t)c_desc;
    plan->preference = (uint64_t)(uintptr_t)preference;
    memcpy(plan->heuristic, &heuristic, sizeof(heuristic));
    plan->initialized = 1u;
    plan->sentinel = SPARKPIPE_CUDA_CUBLASLT_GEMM_SENTINEL;
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkDestroyCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan)
{
    if (plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    SparkDestroyLtPlanObjects(plan);
    memset(plan, 0, sizeof(*plan));
    return SPARK_STATUS_OK;
}

extern "C" SparkStatus SparkRunCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, const void *device_a_bf16, const void *device_b_bf16, void *device_c_bf16, SparkCudaCublasLtBf16GemmReport *report)
{
    cublasLtMatmulHeuristicResult_t heuristic;
    cublasStatus_t cublas_status;
    cudaError_t cuda_status;
    float alpha;
    float beta;

    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    if (plan == 0 || device_a_bf16 == 0 || device_b_bf16 == 0 || device_c_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    if (plan->initialized == 0u || plan->sentinel != SPARKPIPE_CUDA_CUBLASLT_GEMM_SENTINEL)
    {
        report->sentinel_violation_count = 1u;
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memcpy(&heuristic, plan->heuristic, sizeof(heuristic));
    alpha = 1.0f;
    beta = 0.0f;
    cublas_status = cublasLtMatmul((cublasLtHandle_t)(uintptr_t)plan->lt_handle, (cublasLtMatmulDesc_t)(uintptr_t)plan->operation_desc, &alpha, device_a_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->a_desc, device_b_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->b_desc, &beta, device_c_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->c_desc, device_c_bf16, (cublasLtMatrixLayout_t)(uintptr_t)plan->c_desc, &heuristic.algo, plan->workspace, (size_t)plan->workspace_bytes, (cudaStream_t)(uintptr_t)plan->stream);
    if (cublas_status != CUBLAS_STATUS_SUCCESS)
    {
        return SparkStatusFromCublasLt(cublas_status);
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        return SparkStatusFromCudaLt(cuda_status);
    }
    report->operation_count = 1u;
    report->element_count = (uint64_t)plan->m * plan->n;
    report->flops_per_run = 2ull * (uint64_t)plan->m * plan->n * plan->k;
    report->workspace_bytes = plan->workspace_bytes;
    report->plan_initialized = plan->initialized;
    report->heuristic_result_count = 1u;
    report->tensor_core_candidate = 1u;
    report->run_count = 1u;
    return SPARK_STATUS_OK;
}
