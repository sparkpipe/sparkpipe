#include <string.h>

#include "sparkpipe/spark_cuda_cublaslt_gemm.h"

#ifndef SPARKPIPE_ENABLE_CUDA_DUMMY
SparkStatus SparkInitCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, void *workspace, uint64_t workspace_bytes, uint64_t stream)
{
    if (plan == 0 || workspace == 0 || m == 0 || n == 0 || k == 0 || workspace_bytes == 0u || stream != 0u)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    plan->m = m;
    plan->n = n;
    plan->k = k;
    plan->lda = k;
    plan->ldb = k;
    plan->ldc = n;
    plan->workspace = workspace;
    plan->workspace_bytes = workspace_bytes;
    plan->sentinel = SPARKPIPE_CUDA_CUBLASLT_GEMM_SENTINEL;
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}

SparkStatus SparkDestroyCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan)
{
    if (plan == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    memset(plan, 0, sizeof(*plan));
    return SPARK_STATUS_OK;
}

SparkStatus SparkRunCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, const void *device_a_bf16, const void *device_b_bf16, void *device_c_bf16, SparkCudaCublasLtBf16GemmReport *report)
{
    if (report != 0)
    {
        memset(report, 0, sizeof(*report));
    }
    if (plan == 0 || device_a_bf16 == 0 || device_b_bf16 == 0 || device_c_bf16 == 0 || report == 0)
    {
        return SPARK_STATUS_INVALID_ARGUMENT;
    }
    return SPARK_STATUS_GRAPH_NOT_AVAILABLE;
}
#endif
