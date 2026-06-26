#pragma once

#include <stdint.h>

#include "sparkpipe/spark_status.h"

#ifdef __cplusplus
extern "C" {
#endif

#define SPARKPIPE_CUDA_CUBLASLT_GEMM_SENTINEL 0x53504355424C5447ull
#define SPARKPIPE_CUDA_CUBLASLT_HEURISTIC_BYTES 8192u

typedef struct SparkCudaCublasLtBf16GemmPlan
{
    uint32_t m;
    uint32_t n;
    uint32_t k;
    uint32_t lda;
    uint32_t ldb;
    uint32_t ldc;
    uint64_t workspace_bytes;
    void *workspace;
    uint64_t stream;
    uint64_t lt_handle;
    uint64_t operation_desc;
    uint64_t a_desc;
    uint64_t b_desc;
    uint64_t c_desc;
    uint64_t preference;
    uint8_t heuristic[SPARKPIPE_CUDA_CUBLASLT_HEURISTIC_BYTES];
    uint32_t initialized;
    uint64_t sentinel;
} SparkCudaCublasLtBf16GemmPlan;

typedef struct SparkCudaCublasLtBf16GemmReport
{
    uint64_t operation_count;
    uint64_t element_count;
    uint64_t flops_per_run;
    uint64_t workspace_bytes;
    uint32_t plan_initialized;
    uint32_t heuristic_result_count;
    uint32_t tensor_core_candidate;
    uint32_t run_count;
    uint32_t sentinel_violation_count;
} SparkCudaCublasLtBf16GemmReport;

SparkStatus SparkInitCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, uint32_t m, uint32_t n, uint32_t k, void *workspace, uint64_t workspace_bytes, uint64_t stream);
SparkStatus SparkDestroyCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan);
SparkStatus SparkRunCudaCublasLtBf16GemmPlan(SparkCudaCublasLtBf16GemmPlan *plan, const void *device_a_bf16, const void *device_b_bf16, void *device_c_bf16, SparkCudaCublasLtBf16GemmReport *report);

#ifdef __cplusplus
}
#endif
